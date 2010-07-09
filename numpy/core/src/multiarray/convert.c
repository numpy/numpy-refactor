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

#include "arrayobject.h"
#include "mapping.h"

#include "convert.h"

/*NUMPY_API
 * To List
 */
NPY_NO_EXPORT PyObject *
PyArray_ToList(PyArrayObject *self)
{
    PyObject *lp;
    PyArrayObject *v;
    intp sz, i;

    if (!PyArray_Check(self)) {
        return (PyObject *)self;
    }
    if (PyArray_NDIM(self) == 0) {
        return PyArray_DESCR(self)->f->getitem(PyArray_BYTES(self),self);
    }

    sz = PyArray_DIM(self, 0);
    lp = PyList_New(sz);
    for (i = 0; i < sz; i++) {
        v = (PyArrayObject *)array_big_item(self, i);
        if (PyArray_Check(v) && (PyArray_NDIM(v) >= PyArray_NDIM(self))) {
            PyErr_SetString(PyExc_RuntimeError,
                            "array_item not returning smaller-" \
                            "dimensional array");
            Py_DECREF(v);
            Py_DECREF(lp);
            return NULL;
        }
        PyList_SetItem(lp, i, PyArray_ToList(v));
        Py_DECREF(v);
    }
    return lp;
}


int PyArray_ToTextFile(PyArrayObject *self, FILE *fp, char *sep, char *format)
{
    intp n, n2;
    size_t n3, n4;
    NpyArrayIterObject *it;
    PyObject *obj, *strobj, *tupobj, *byteobj;
        
    it = NpyArray_IterNew(PyArray_ARRAY(self));
    n3 = (sep ? strlen((const char *)sep) : 0);
    n4 = (format ? strlen((const char *)format) : 0);
    while (it->index < it->size) {
        obj = PyArray_DESCR(self)->f->getitem(it->dataptr, self);
        if (obj == NULL) {
            _Npy_DECREF(it);
            return -1;
        }
        if (n4 == 0) {
            /*
             * standard writing
             */
            strobj = PyObject_Str(obj);
            Py_DECREF(obj);
            if (strobj == NULL) {
                _Npy_DECREF(it);
                return -1;
            }
        }
        else {
            /*
             * use format string
             */
            tupobj = PyTuple_New(1);
            if (tupobj == NULL) {
                _Npy_DECREF(it);
                return -1;
            }
            PyTuple_SET_ITEM(tupobj,0,obj);
            obj = PyUString_FromString((const char *)format);
            if (obj == NULL) {
                Py_DECREF(tupobj);
                _Npy_DECREF(it);
                return -1;
            }
            strobj = PyUString_Format(obj, tupobj);
            Py_DECREF(obj);
            Py_DECREF(tupobj);
            if (strobj == NULL) {
                _Npy_DECREF(it);
                return -1;
            }
        }
#if defined(NPY_PY3K)
        byteobj = PyUnicode_AsASCIIString(strobj);
#else
        byteobj = strobj;
#endif
        NPY_BEGIN_ALLOW_THREADS;
        n2 = PyBytes_GET_SIZE(byteobj);
        n = fwrite(PyBytes_AS_STRING(byteobj), 1, n2, fp);
        NPY_END_ALLOW_THREADS;
#if defined(NPY_PY3K)
        Py_DECREF(byteobj);
#endif
        if (n < n2) {
            PyErr_Format(PyExc_IOError,
                         "problem writing element %"INTP_FMT\
                         " to file", it->index);
            Py_DECREF(strobj);
            _Npy_DECREF(it);
            return -1;
        }
        /* write separator for all but last one */
        if (it->index != it->size-1) {
            if (fwrite(sep, 1, n3, fp) < n3) {
                PyErr_Format(PyExc_IOError,
                             "problem writing "\
                             "separator to file");
                Py_DECREF(strobj);
                _Npy_DECREF(it);
                return -1;
            }
        }
        Py_DECREF(strobj);
        NpyArray_ITER_NEXT(it);
    }
    _Npy_DECREF(it);
    return 0;
}


/* XXX: FIXME --- add ordering argument to
   Allow Fortran ordering on write
   This will need the addition of a Fortran-order iterator.
 */

/*NUMPY_API
  To File
*/
NPY_NO_EXPORT int
PyArray_ToFile(PyArrayObject *self, FILE *fp, char *sep, char *format)
{
    size_t n3;

    n3 = (sep ? strlen((const char *)sep) : 0);
    if (n3 == 0) {
        NpyArray_ToBinaryFile(PyArray_ARRAY(self), fp);
    }
    else {
        PyArray_ToTextFile(self, fp, sep, format);
    }
    return 0;
}

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_ToString(PyArrayObject *self, NPY_ORDER order)
{
    intp numbytes;
    intp index;
    char *dptr;
    int elsize;
    PyObject *ret;
    NpyArrayIterObject *it;

    if (order == NPY_ANYORDER)
        order = PyArray_ISFORTRAN(self);

    /*        if (PyArray_TYPE(self) == PyArray_OBJECT) {
              PyErr_SetString(PyExc_ValueError, "a string for the data" \
              "in an object array is not appropriate");
              return NULL;
              }
    */

    numbytes = PyArray_NBYTES(self);
    if ((PyArray_ISCONTIGUOUS(self) && (order == NPY_CORDER))
        || (PyArray_ISFORTRAN(self) && (order == NPY_FORTRANORDER))) {
        ret = PyBytes_FromStringAndSize(PyArray_BYTES(self), (Py_ssize_t) numbytes);
    }
    else {
        PyObject *new;
        if (order == NPY_FORTRANORDER) {
            /* iterators are always in C-order */
            new = PyArray_Transpose(self, NULL);
            if (new == NULL) {
                return NULL;
            }
        }
        else {
            Py_INCREF(self);
            new = (PyObject *)self;
        }
        it = NpyArray_IterNew(PyArray_ARRAY(new));
        Py_DECREF(new);
        if (it == NULL) {
            return NULL;
        }
        ret = PyBytes_FromStringAndSize(NULL, (Py_ssize_t) numbytes);
        if (ret == NULL) {
            _Npy_DECREF(it);
            return NULL;
        }
        dptr = PyBytes_AS_STRING(ret);
        index = it->size;
        elsize = PyArray_ITEMSIZE(self);
        while (index--) {
            memcpy(dptr, it->dataptr, elsize);
            dptr += elsize;
            NpyArray_ITER_NEXT(it);
        }
        _Npy_DECREF(it);
    }
    return ret;
}

/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_FillWithScalar(PyArrayObject *arr, PyObject *obj)
{
    PyObject *newarr;
    int itemsize, swap;
    void *fromptr;
    PyArray_Descr *descr;
    intp size;
    PyArray_CopySwapFunc *copyswap;

    itemsize = PyArray_ITEMSIZE(arr);
    if (PyArray_ISOBJECT(arr)) {
        fromptr = &obj;
        swap = 0;
        newarr = NULL;
    }
    else {
        descr = PyArray_DESCR(arr);
        Py_INCREF(descr);
        newarr = PyArray_FromAny(obj, descr, 0,0, ALIGNED, NULL);
        if (newarr == NULL) {
            return -1;
        }
        fromptr = PyArray_DATA(newarr);
        swap = (PyArray_ISNOTSWAPPED(arr) != PyArray_ISNOTSWAPPED(newarr));
    }
    size=PyArray_SIZE(arr);
    copyswap = PyArray_DESCR(arr)->f->copyswap;
    if (PyArray_ISONESEGMENT(arr)) {
        char *toptr=PyArray_DATA(arr);
        PyArray_FillWithScalarFunc* fillwithscalar =
            PyArray_DESCR(arr)->f->fillwithscalar;
        if (fillwithscalar && PyArray_ISALIGNED(arr)) {
            copyswap(fromptr, NULL, swap, newarr);
            fillwithscalar(toptr, size, fromptr, arr);
        }
        else {
            while (size--) {
                copyswap(toptr, fromptr, swap, arr);
                toptr += itemsize;
            }
        }
    }
    else {
        NpyArrayIterObject *iter;

        iter = NpyArray_IterNew(PyArray_ARRAY(arr));
        if (iter == NULL) {
            Py_XDECREF(newarr);
            return -1;
        }
        while (size--) {
            copyswap(iter->dataptr, fromptr, swap, arr);
            NpyArray_ITER_NEXT(iter);
        }
        _Npy_DECREF(iter);
    }
    Py_XDECREF(newarr);
    return 0;
}

/*NUMPY_API
  Copy an array.
*/
NPY_NO_EXPORT PyObject *
PyArray_NewCopy(PyArrayObject *m1, NPY_ORDER fortran)
{
    PyArrayObject *ret;
    if (fortran == PyArray_ANYORDER)
        fortran = PyArray_ISFORTRAN(m1);

    Py_INCREF(PyArray_DESCR(m1));
    ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(m1),
                                                PyArray_DESCR(m1),
                                                PyArray_NDIM(m1),
                                                PyArray_DIMS(m1),
                                                NULL, NULL,
                                                fortran,
                                                (PyObject *)m1);
    if (ret == NULL) {
        return NULL;
    }
    if (PyArray_CopyInto(ret, m1) == -1) {
        Py_DECREF(ret);
        return NULL;
    }

    return (PyObject *)ret;
}

/*NUMPY_API
 * View
 * steals a reference to type -- accepts NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_View(PyArrayObject *self, PyArray_Descr *type, PyTypeObject *pytype)
{
    PyArrayObject *new = NULL;
    PyTypeObject *subtype;

    if (pytype) {
        subtype = pytype;
    }
    else {
        subtype = Py_TYPE(self);
    }
    Py_INCREF(PyArray_DESCR(self));
    new = (PyArrayObject* )PyArray_NewFromDescr(subtype,
                                                PyArray_DESCR(self),
                                                PyArray_NDIM(self), PyArray_DIMS(self),
                                                PyArray_STRIDES(self),
                                                PyArray_BYTES(self),
                                                PyArray_FLAGS(self), (PyObject *)self);
    if (new == NULL) {
        return NULL;
    }
    
    /* TODO: Unwrap array structure, increment NpyArray, not PyArrayObject refcnt. */
    PyArray_BASE_ARRAY(new) = PyArray_ARRAY(self);
    Npy_INCREF(PyArray_BASE_ARRAY(new));
    assert(NULL == PyArray_BASE_ARRAY(new) || NULL == PyArray_BASE(new));

    if (type != NULL) {
        if (PyObject_SetAttrString((PyObject *)new, "dtype",
                                   (PyObject *)type) < 0) {
            Py_DECREF(new);
            Py_DECREF(type);
            return NULL;
        }
        Py_DECREF(type);
    }
    return (PyObject *)new;
}
