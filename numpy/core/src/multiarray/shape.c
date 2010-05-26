#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "numpy/npy_math.h"
#include "numpy/numpy_api.h"

#include "npy_config.h"

#include "npy_3kcompat.h"

#include "ctors.h"

#include "shape.h"

#define PyAO PyArrayObject

static void
_putzero(char *optr, PyObject *zero, PyArray_Descr *dtype);

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
    intp oldsize, newsize;
    int new_nd=newshape->len, k, n, elsize;
    int refcnt;
    intp* new_dimensions=newshape->ptr;
    intp new_strides[MAX_DIMS];
    size_t sd;
    intp *dimptr;
    char *new_data;
    intp largest;

    if (!PyArray_ISONESEGMENT(self)) {
        PyErr_SetString(PyExc_ValueError,
                "resize only works on single-segment arrays");
        return NULL;
    }

    if (self->descr->elsize == 0) {
        PyErr_SetString(PyExc_ValueError,
                "Bad data-type size.");
        return NULL;
    }
    newsize = 1;
    largest = MAX_INTP / self->descr->elsize;
    for(k = 0; k < new_nd; k++) {
        if (new_dimensions[k] == 0) {
            break;
        }
        if (new_dimensions[k] < 0) {
            PyErr_SetString(PyExc_ValueError,
                    "negative dimensions not allowed");
            return NULL;
        }
        newsize *= new_dimensions[k];
        if (newsize <= 0 || newsize > largest) {
            return PyErr_NoMemory();
        }
    }
    oldsize = PyArray_SIZE(self);

    if (oldsize != newsize) {
        if (!(self->flags & OWNDATA)) {
            PyErr_SetString(PyExc_ValueError,
                    "cannot resize this array: it does not own its data");
            return NULL;
        }

        if (refcheck) {
            refcnt = REFCOUNT(self);
        }
        else {
            refcnt = 1;
        }
        if ((refcnt > 2)
                || (self->base != NULL)
                || (self->weakreflist != NULL)) {
            PyErr_SetString(PyExc_ValueError,
                    "cannot resize an array references or is referenced\n"\
                    "by another array in this way.  Use the resize function");
            return NULL;
        }

        if (newsize == 0) {
            sd = self->descr->elsize;
        }
        else {
            sd = newsize*self->descr->elsize;
        }
        /* Reallocate space if needed */
        new_data = PyDataMem_RENEW(self->data, sd);
        if (new_data == NULL) {
            PyErr_SetString(PyExc_MemoryError,
                    "cannot allocate memory for array");
            return NULL;
        }
        self->data = new_data;
    }

    if ((newsize > oldsize) && PyArray_ISWRITEABLE(self)) {
        /* Fill new memory with zeros */
        elsize = self->descr->elsize;
        if (PyDataType_FLAGCHK(self->descr, NPY_ITEM_REFCOUNT)) {
            PyObject *zero = PyInt_FromLong(0);
            char *optr;
            optr = self->data + oldsize*elsize;
            n = newsize - oldsize;
            for (k = 0; k < n; k++) {
                _putzero((char *)optr, zero, self->descr);
                optr += elsize;
            }
            Py_DECREF(zero);
        }
        else{
            memset(self->data+oldsize*elsize, 0, (newsize-oldsize)*elsize);
        }
    }

    if (self->nd != new_nd) {
        /* Different number of dimensions. */
        self->nd = new_nd;
        /* Need new dimensions and strides arrays */
        dimptr = PyDimMem_RENEW(self->dimensions, 2*new_nd);
        if (dimptr == NULL) {
            PyErr_SetString(PyExc_MemoryError,
                    "cannot allocate memory for array");
            return NULL;
        }
        self->dimensions = dimptr;
        self->strides = dimptr + new_nd;
    }

    /* make new_strides variable */
    sd = (size_t) self->descr->elsize;
    sd = (size_t) _array_fill_strides(new_strides, new_dimensions, new_nd, sd,
            self->flags, &(self->flags));
    memmove(self->dimensions, new_dimensions, new_nd*sizeof(intp));
    memmove(self->strides, new_strides, new_nd*sizeof(intp));
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
    return (PyObject*) NpyArray_Newshape(self, newdims, fortran);
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
_putzero(char *optr, PyObject *zero, PyArray_Descr *dtype)
{
    if (!PyDataType_FLAGCHK(dtype, NPY_ITEM_REFCOUNT)) {
        memset(optr, 0, dtype->elsize);
    }
    else if (PyDescr_HASFIELDS(dtype)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;
        while (PyDict_Next(dtype->fields, &pos, &key, &value)) {
            if NPY_TITLE_KEY(key, value) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                return;
            }
            _putzero(optr + offset, zero, new);
        }
    }
    else {
        Py_INCREF(zero);
        NPY_COPY_PYOBJECT_PTR(optr, &zero);
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
    return (PyObject*) NpyArray_Squeeze(self);
}

/*NUMPY_API
 * SwapAxes
 */
NPY_NO_EXPORT PyObject *
PyArray_SwapAxes(PyArrayObject *ap, int a1, int a2)
{
    return (PyObject*) NpyArray_SwapAxes(ap, a1, a2);
}

/*NUMPY_API
 * Return Transpose.
 */
NPY_NO_EXPORT PyObject *
PyArray_Transpose(PyArrayObject *ap, PyArray_Dims *permute)
{
    return (PyObject*) NpyArray_Transpose(ap, permute);
}

/*NUMPY_API
 * Ravel
 * Returns a contiguous array
 */
NPY_NO_EXPORT PyObject *
PyArray_Ravel(PyArrayObject *a, NPY_ORDER fortran)
{
    return (PyObject*) NpyArray_Ravel(a, fortran);
}

/*NUMPY_API
 * Flatten
 */
NPY_NO_EXPORT PyObject *
PyArray_Flatten(PyArrayObject *a, NPY_ORDER order)
{
    return (PyObject*) NpyArray_Flatten(a, order);
}
