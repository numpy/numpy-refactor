/*
 * This module corresponds to the `Special functions for PyArray_OBJECT`
 * section in the numpy reference for C-API.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "numpy/numpy_api.h"

#include "npy_config.h"

#include "npy_3kcompat.h"

static void
_fillobject(char *optr, PyObject *obj, PyArray_Descr *dtype);


/*
 * Reference counting callbacks provided to the core. 
 */

void
NpyInterface_Incref(void *objtmp)
{
    PyObject *obj = (PyObject *)objtmp;
    Py_INCREF(obj);
}

void
NpyInterface_Decref(void *objtmp)
{
    PyObject *obj = (PyObject *)objtmp;
    Py_DECREF(obj);
}




/* Incref all objects found at this record */
/*NUMPY_API
 */
NPY_NO_EXPORT void
PyArray_Item_INCREF(char *data, PyArray_Descr *descr)
{
    PyObject *temp;

    if (!PyDataType_REFCHK(descr)) {
        return;
    }
    if (descr->type_num == PyArray_OBJECT) {
        NPY_COPY_PYOBJECT_PTR(&temp, data);
        Py_XINCREF(temp);
    }
    else if (PyDescr_HASFIELDS(descr)) {
        const char *key;
        NpyArray_DescrField *value;
        NpyDict_Iter pos;

        NpyDict_IterInit(&pos);
        while (NpyDict_IterNext(descr->fields, &pos, (void **)&key, (void **)&value)) {
            if (NULL != value->title && !strcmp(value->title, key)) {
                continue;
            }
            NpyArray_Item_INCREF(data + value->offset, value->descr);
        }
    }
    return;
}

/* XDECREF all objects found at this record */
/*NUMPY_API
 */
NPY_NO_EXPORT void
PyArray_Item_XDECREF(char *data, PyArray_Descr *descr)
{
    PyObject *temp;

    if (!PyDataType_REFCHK(descr)) {
        return;
    }

    if (descr->type_num == PyArray_OBJECT) {
        NPY_COPY_PYOBJECT_PTR(&temp, data);
        Py_XDECREF(temp);
    }
    else if PyDescr_HASFIELDS(descr) {
        const char *key;
        NpyArray_DescrField *value;
        NpyDict_Iter pos;
        
        NpyDict_IterInit(&pos);
        while (NpyDict_IterNext(descr->fields, &pos, (void **)&key, (void **)&value)) {
            if (NULL != value->title && !strcmp(value->title, key)) {
                continue;
            }
            NpyArray_Item_XDECREF(data + value->offset, value->descr);
        }
    }
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
    intp i, n;
    PyObject **data;
    PyObject *temp;
    NpyArrayIterObject *it;

    if (!PyDataType_REFCHK(PyArray_DESCR(mp))) {
        return 0;
    }
    if (PyArray_TYPE(mp) != PyArray_OBJECT) {
        it = NpyArray_IterNew(PyArray_ARRAY(mp));
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            PyArray_Item_INCREF(it->dataptr, PyArray_DESCR(mp));
            NpyArray_ITER_NEXT(it);
        }
        _Npy_DECREF(it);
        return 0;
    }

    if (PyArray_ISONESEGMENT(mp)) {
        data = (PyObject **)PyArray_BYTES(mp);
        n = PyArray_SIZE(mp);
        if (PyArray_ISALIGNED(mp)) {
            for (i = 0; i < n; i++, data++) {
                Py_XINCREF(*data);
            }
        }
        else {
            for( i = 0; i < n; i++, data++) {
                NPY_COPY_PYOBJECT_PTR(&temp, data);
                Py_XINCREF(temp);
            }
        }
    }
    else { /* handles misaligned data too */
        it = NpyArray_IterNew(PyArray_ARRAY(mp));
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            NPY_COPY_PYOBJECT_PTR(&temp, it->dataptr);
            Py_XINCREF(temp);
            NpyArray_ITER_NEXT(it);
        }
        _Npy_DECREF(it);
    }
    return 0;
}

/*NUMPY_API
  Decrement all internal references for object arrays.
  (or arrays with object fields)
*/
NPY_NO_EXPORT int
PyArray_XDECREF(PyArrayObject *mp)
{
    intp i, n;
    PyObject **data;
    PyObject *temp;
    NpyArrayIterObject *it;

    if (!PyDataType_REFCHK(PyArray_DESCR(mp))) {
        return 0;
    }
    if (PyArray_TYPE(mp) != PyArray_OBJECT) {
        it = NpyArray_IterNew(PyArray_ARRAY(mp));
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            PyArray_Item_XDECREF(it->dataptr, PyArray_DESCR(mp));
            NpyArray_ITER_NEXT(it);
        }
        _Npy_DECREF(it);
        return 0;
    }

    if (PyArray_ISONESEGMENT(mp)) {
        data = (PyObject **)PyArray_BYTES(mp);
        n = PyArray_SIZE(mp);
        if (PyArray_ISALIGNED(mp)) {
            for (i = 0; i < n; i++, data++) Py_XDECREF(*data);
        }
        else {
            for (i = 0; i < n; i++, data++) {
                NPY_COPY_PYOBJECT_PTR(&temp, data);
                Py_XDECREF(temp);
            }
        }
    }
    else { /* handles misaligned data too */
        it = NpyArray_IterNew(PyArray_ARRAY(mp));
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            NPY_COPY_PYOBJECT_PTR(&temp, it->dataptr);
            Py_XDECREF(temp);
            NpyArray_ITER_NEXT(it);
        }
        _Npy_DECREF(it);
    }
    return 0;
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
_fillobject(char *optr, PyObject *obj, PyArray_Descr *dtype)
{
    if (!PyDataType_FLAGCHK(dtype, NPY_ITEM_REFCOUNT)) {
        if ((obj == Py_None) || (PyInt_Check(obj) && PyInt_AsLong(obj)==0)) {
            return;
        }
        else {
            PyObject *arr;
            Py_INCREF(dtype);
            arr = PyArray_NewFromDescr(&PyArray_Type, dtype,
                                       0, NULL, NULL, NULL,
                                       0, NULL);
            if (arr!=NULL) {
                dtype->f->setitem(obj, optr, PyArray_ARRAY(arr));
            }
            Py_XDECREF(arr);
        }
    }
    else if (PyDescr_HASFIELDS(dtype)) {
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

