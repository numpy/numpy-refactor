/*
 * This module corresponds to the `Special functions for PyArray_OBJECT`
 * section in the numpy reference for C-API.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "npy_api.h"
#include "npy_dict.h"
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "numpy_3kcompat.h"

static void
_fillobject(char *optr, PyObject *obj, NpyArray_Descr *dtype);


/*
 * Reference counting callbacks provided to the core. 
 */

/* Returns a new handle to the object.  For garbage collected systems this will be different than
   the incoming object.  For refcounted systems such as CPython we just return the original ptr. */
NPY_NO_EXPORT void *
NpyInterface_Incref(void *obj)
{
    Py_INCREF(obj);
    return obj;
}

NPY_NO_EXPORT void
NpyInterface_Decref(void *obj)
{
    /* Used XDECREF, so this function cleanly handles NULL. */
    Py_XDECREF(obj);
}



/* Incref all objects found at this record */
/*NUMPY_API
 */
NPY_NO_EXPORT void
PyArray_Item_INCREF(char *data, PyArray_Descr *descr)
{
    NpyArray_Item_INCREF(data, descr->descr);
    return;
}

/* XDECREF all objects found at this record */
/*NUMPY_API
 */
NPY_NO_EXPORT void
PyArray_Item_XDECREF(char *data, PyArray_Descr *descr)
{
    NpyArray_Item_XDECREF(data, descr->descr);
    return;
}

/* Used for arrays of python objects to increment the reference count of */
/* every python object in the array. */
/*NUMPY_API
  For object arrays, increment all internal references.
*/
NPY_NO_EXPORT int
PyArray_INCREF(PyArrayObject *mp)
{
    return NpyArray_INCREF(PyArray_ARRAY(mp));
}



/*NUMPY_API
  Decrement all internal references for object arrays.
  (or arrays with object fields)
*/
NPY_NO_EXPORT int
PyArray_XDECREF(PyArrayObject *mp)
{
    return NpyArray_XDECREF(PyArray_ARRAY(mp));
}

/*NUMPY_API
 * Assumes contiguous
 */
NPY_NO_EXPORT void
PyArray_FillObjectArray(PyArrayObject *arr, PyObject *obj)
{
    intp i,n;
    n = PyArray_SIZE(arr);
    if (PyArray_TYPE(arr) == PyArray_OBJECT) {
        PyObject **optr;
        optr = (PyObject **)(PyArray_BYTES(arr));
        n = PyArray_SIZE(arr);
        if (obj == NULL) {
            for (i = 0; i < n; i++) {
                *optr++ = NULL;
            }
        }
        else {
            for (i = 0; i < n; i++) {
                Py_INCREF(obj);
                *optr++ = obj;
            }
        }
    }
    else {
        char *optr;
        optr = PyArray_BYTES(arr);
        for (i = 0; i < n; i++) {
            _fillobject(optr, obj, PyArray_DESCR(arr));
            optr += PyArray_ITEMSIZE(arr);
        }
    }
}

static void
_fillobject(char *optr, PyObject *obj, NpyArray_Descr *dtype)
{
    if (!NpyDataType_FLAGCHK(dtype, NPY_ITEM_REFCOUNT)) {
        if ((obj == Py_None) || (PyInt_Check(obj) && PyInt_AsLong(obj)==0)) {
            return;
        }
        else {
            NpyArray *arr;
            _Npy_INCREF(dtype);
            arr = NpyArray_NewFromDescr(dtype,
                                       0, NULL, NULL, NULL,
                                       0, NPY_TRUE, NULL, NULL);
            if (arr!=NULL) {
                dtype->f->setitem(obj, optr, arr);
            }
            _Npy_DECREF(arr);
        }
    }
    else if (NpyDataType_HASFIELDS(dtype)) {
        const char *key;
        NpyArray_DescrField *value;
        NpyDict_Iter pos;
        
        NpyDict_IterInit(&pos);
        while (NpyDict_IterNext(dtype->fields, &pos, (void **)&key, (void **)&value)) {
            if (NULL != value->title && !strcmp(value->title, key)) {
                continue;
            }
            _fillobject(optr + value->offset, obj, value->descr);   /* TODO: Either need to wrap/unwrap descr or move this func to core */
        }
    }
    else {
        Py_XINCREF(obj);
        NPY_COPY_PYOBJECT_PTR(optr, &obj);
        return;
    }
}

