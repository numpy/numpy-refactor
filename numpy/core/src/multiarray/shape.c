#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "npy_math.h"
#include "npy_api.h"
#include "npy_dict.h"

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "numpy/npy_3kcompat.h"

#include "ctors.h"
#include "arrayobject.h"

#include "shape.h"

#define PyAO PyArrayObject

static void
_putzero(char *optr, PyObject *zero, NpyArray_Descr *dtype);

/*NUMPY_API
 * Resize (reallocate data).  Only works if nothing else is referencing this
 * array and it is contiguous.  If refcheck is 0, then the reference count is
 * not checked and assumed to be 1.  You still must own this data and have no
 * weak-references and no base object.
 */
NPY_NO_EXPORT PyObject *
PyArray_Resize(PyArrayObject *self, PyArray_Dims *newshape, int refcheck,
               NPY_ORDER fortran)
{
    intp newsize, oldsize;

    oldsize = PyArray_SIZE(self);
    newsize = NpyArray_MultiplyList(newshape->ptr, newshape->len);
    if (newsize != oldsize) {
        int refcnt;
        if (refcheck) {
            refcnt = Py_REFCNT(self);
        } else {
            refcnt = 1;
        }
            
        if (refcnt > 2 || self->weakreflist != NULL) {
            PyErr_SetString(PyExc_ValueError,
                    "cannot resize an array references or is referenced\n"\
                    "by another array in this way.  Use the resize function");
            return NULL;
        }
    }
    if (NpyArray_Resize(PyArray_ARRAY(self), newshape, refcheck, fortran) != 0) {
        return NULL;
    }
    if (newsize > oldsize && 
        NpyDataType_FLAGCHK(PyArray_DESCR(self), NPY_ITEM_REFCOUNT)) {
        /* Fill with zeros. */
        int n, k;
        char *optr;
        int elsize = PyArray_ITEMSIZE(self);
        PyObject *zero = PyInt_FromLong(0);

        optr = PyArray_BYTES(self) + oldsize*elsize;
        n = newsize - oldsize;
        for (k = 0; k < n; k++) {
            _putzero((char *)optr, zero, PyArray_DESCR(self));
            optr += elsize;
        }
        Py_DECREF(zero);
    }
    Py_INCREF(Py_None);
    return Py_None;
}

/*
 * Returns a new array
 * with the new shape from the data
 * in the old array --- order-perspective depends on fortran argument.
 * copy-only-if-necessary
 */

/*NUMPY_API
 * New shape for an array
 */
NPY_NO_EXPORT PyObject *
PyArray_Newshape(PyArrayObject *self, PyArray_Dims *newdims,
                 NPY_ORDER fortran)
{
    RETURN_PYARRAY(NpyArray_Newshape(PyArray_ARRAY(self), newdims, fortran));
}


/* For back-ward compatability -- Not recommended */

/*NUMPY_API
 * Reshape
 */
NPY_NO_EXPORT PyObject *
PyArray_Reshape(PyArrayObject *self, PyObject *shape)
{
    PyObject *ret;
    PyArray_Dims newdims;

    if (!PyArray_IntpConverter(shape, &newdims)) {
        return NULL;
    }
    ret = PyArray_Newshape(self, &newdims, PyArray_CORDER);
    PyDimMem_FREE(newdims.ptr);
    return ret;
}


static void
_putzero(char *optr, PyObject *zero, NpyArray_Descr *dtype)
{
    if (!NpyDataType_FLAGCHK(dtype, NPY_ITEM_REFCOUNT)) {
        memset(optr, 0, dtype->elsize);
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
            _putzero(optr + value->offset, zero, value->descr);
        }
    }
    else {
        Py_INCREF(zero);
        NPY_COPY_VOID_PTR(optr, &zero);
    }
    return;
}




/*NUMPY_API
 *
 * return a new view of the array object with all of its unit-length
 * dimensions squeezed out if needed, otherwise
 * return the same array.
 */
NPY_NO_EXPORT PyObject *
PyArray_Squeeze(PyArrayObject *self)
{
    RETURN_PYARRAY(NpyArray_Squeeze(PyArray_ARRAY(self)));
}

/*NUMPY_API
 * SwapAxes
 */
NPY_NO_EXPORT PyObject *
PyArray_SwapAxes(PyArrayObject *ap, int a1, int a2)
{
    RETURN_PYARRAY(NpyArray_SwapAxes(PyArray_ARRAY(ap), a1, a2));
}

/*NUMPY_API
 * Return Transpose.
 */
NPY_NO_EXPORT PyObject *
PyArray_Transpose(PyArrayObject *ap, PyArray_Dims *permute)
{
    RETURN_PYARRAY(NpyArray_Transpose(PyArray_ARRAY(ap), permute));
}

/*NUMPY_API
 * Ravel
 * Returns a contiguous array
 */
NPY_NO_EXPORT PyObject *
PyArray_Ravel(PyArrayObject *a, NPY_ORDER fortran)
{
    RETURN_PYARRAY(NpyArray_Ravel(PyArray_ARRAY(a), fortran));
}

/*NUMPY_API
 * Flatten
 */
NPY_NO_EXPORT PyObject *
PyArray_Flatten(PyArrayObject *a, NPY_ORDER order)
{
    RETURN_PYARRAY(NpyArray_Flatten(PyArray_ARRAY(a), order));
}
