#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "npy_api.h"
#include "npy_math.h"

#include "npy_config.h"

#include "numpy/npy_3kcompat.h"

#include "common.h"
#include "ctors.h"
#include "arrayobject.h"

#define PyAO PyArrayObject
#define _check_axis PyArray_CheckAxis


NPY_NO_EXPORT NpyArray_Descr *
PyArray_DescrFromObjectUnwrap(PyObject *op, NpyArray_Descr *mintype);


/*NUMPY_API
 * Take
 */
NPY_NO_EXPORT PyObject *
PyArray_TakeFrom(PyArrayObject *self0, PyObject *indices0, int axis,
                 PyArrayObject *ret, NPY_CLIPMODE clipmode)
{
    PyArrayObject* indices;
    PyObject* result;

    /* Get indices array. */
    if (PyArray_Check(indices0)) {
        indices = (PyArrayObject*) indices0;
        Py_INCREF(indices);
    } else {
        indices = (PyArrayObject*) PyArray_ContiguousFromAny(indices0,
                                                             PyArray_INTP,
                                                             1, 0);
        if (indices == NULL) {
            Py_XINCREF(ret);
            return NULL;
        }
    }
    ASSIGN_TO_PYARRAY(result, 
                      NpyArray_TakeFrom(PyArray_ARRAY(self0), 
                                        PyArray_ARRAY(indices), 
                                        axis, PyArray_ARRAY(ret), 
                                        clipmode));
    Py_DECREF(indices);
    return result;
}

/*NUMPY_API
 * Put values into an array
 */
NPY_NO_EXPORT PyObject *
PyArray_PutTo(PyArrayObject *self, PyObject* values0, PyObject *indices0,
              NPY_CLIPMODE clipmode)
{
    PyArrayObject* indices = NULL;
    PyArrayObject* values = NULL;

    if (!PyArray_Check(self)) {
        PyErr_SetString(PyExc_TypeError,
                        "put: first argument must be an array");
        return NULL;
    }
    if (PyArray_Check(indices0)) {
        indices = (PyArrayObject *) indices0;
        Py_INCREF(indices);
    } else {
        indices = (PyArrayObject*) 
            PyArray_ContiguousFromAny(indices0, PyArray_INTP, 1, 0);
        if (indices == NULL) {
            return NULL;
        }
    }
    if (PyArray_Check(values0)) {
        values = (PyArrayObject *) values0;
        Py_INCREF(values);
    } else {
        Npy_INCREF(PyArray_DESCR(self));
        values = (PyArrayObject *)PyArray_FromAnyUnwrap(values0, PyArray_DESCR(self), 0, 0, NPY_CARRAY, NULL);
        if (values == NULL) {
            goto fail;
        }
    }
    NpyArray_PutTo(PyArray_ARRAY(self), PyArray_ARRAY(values), 
                   PyArray_ARRAY(indices), clipmode);
    Py_XDECREF(indices);
    Py_XDECREF(values);
    Py_INCREF(Py_None);
    return Py_None;
  fail:
    Py_XDECREF(indices);
    Py_XDECREF(values);
    Py_INCREF(Py_None);
    return Py_None;
}

/*NUMPY_API
 * Put values into an array according to a mask.
 */
NPY_NO_EXPORT PyObject *
PyArray_PutMask(PyArrayObject *self, PyObject* values0, PyObject* mask0)
{
    PyArrayObject* values = NULL;
    PyArrayObject* mask = NULL;

    if (!PyArray_Check(self)) {
        PyErr_SetString(PyExc_TypeError,
                        "putmask: first argument must "\
                        "be an array");
        return NULL;
    }

    if (PyArray_Check(values0)) {
        values = (PyArrayObject*) values0;
        Py_INCREF(values);
    } else {
        Npy_INCREF(PyArray_DESCR(self));
        values = (PyArrayObject *)PyArray_FromAnyUnwrap(values0, PyArray_DESCR(self), 0, 0, NPY_CARRAY, NULL);
        if (values == NULL) {
            return NULL;
        }
    }

    if (PyArray_Check(mask0)) {
        mask = (PyArrayObject*) mask0;
        Py_INCREF(mask);
    } else {
        mask = (PyArrayObject*) PyArray_FROM_OTF(mask0, PyArray_BOOL,
                                                 NPY_CARRAY | NPY_FORCECAST);
        if (mask == NULL) {
            goto fail;
        }
    }

    if (NpyArray_PutMask(PyArray_ARRAY(self), 
                         PyArray_ARRAY(values), PyArray_ARRAY(mask)) == -1) {
        goto fail;
    }

    Py_XDECREF(values);
    Py_XDECREF(mask);
    Py_INCREF(Py_None);
    return Py_None;

  fail:
    Py_XDECREF(values);
    Py_XDECREF(mask);
    return NULL;
}

/*NUMPY_API
 * Repeat the array.
 */
NPY_NO_EXPORT PyObject *
PyArray_Repeat(PyArrayObject *aop, PyObject *op, int axis)
{
    PyArrayObject* repeats = NULL;
    PyObject* result = NULL;

    if (PyArray_Check(op)) {
        repeats = (PyArrayObject *) op;
        Py_INCREF(repeats);
    } else {
        repeats = (PyArrayObject *)PyArray_ContiguousFromAny(op, PyArray_INTP, 0, 1);
        if (repeats == NULL) {
            goto finish;
        }
    }
    ASSIGN_TO_PYARRAY(result,
                      NpyArray_Repeat(PyArray_ARRAY(aop), 
                                      PyArray_ARRAY(repeats), axis));
  finish:
    Py_XDECREF(repeats);
    return result;
}

/*NUMPY_API
 */
NPY_NO_EXPORT PyObject *
PyArray_Choose(PyArrayObject *ip, PyObject *op, PyArrayObject *ret,
               NPY_CLIPMODE clipmode)
{
    PyArrayObject** mps;
    PyObject* result = NULL;
    int i, n;
    NpyArray** nmps = NULL;

    /*
     * Convert all inputs to arrays of a common type
     * Also makes them C-contiguous
     */
    mps = PyArray_ConvertToCommonType(op, &n);
    if (mps == NULL) {
        return NULL;
    }
    for (i = 0; i < n; i++) {
        if (mps[i] == NULL) {
            goto finish;
        }
    }
    /* TODO: Make a ConvertToCommonType that returns core objects. */
    nmps = (NpyArray **)PyDataMem_NEW(n*sizeof(NpyArray*));
    for (i = 0; i < n; i++) {
        nmps[i] = PyArray_ARRAY(mps[i]);
    }
    ASSIGN_TO_PYARRAY(result,
                      NpyArray_Choose(PyArray_ARRAY(ip), nmps, n, 
                                      PyArray_ARRAY(ret), 
                                      clipmode));
    PyDataMem_FREE(nmps);

  finish:
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    NpyDataMem_FREE(mps);
    return result;
}


/*NUMPY_API
 * Sort an array in-place
 */
NPY_NO_EXPORT int
PyArray_Sort(PyArrayObject *op, int axis, NPY_SORTKIND which)
{
    return NpyArray_Sort(PyArray_ARRAY(op), axis, which);
}


/*NUMPY_API
 * ArgSort an array
 */
NPY_NO_EXPORT PyObject *
PyArray_ArgSort(PyArrayObject *op, int axis, NPY_SORTKIND which)
{
    RETURN_PYARRAY(NpyArray_ArgSort(PyArray_ARRAY(op), axis, which));
}


/*NUMPY_API
 *LexSort an array providing indices that will sort a collection of arrays
 *lexicographically.  The first key is sorted on first, followed by the second key
 *-- requires that arg"merge"sort is available for each sort_key
 *
 *Returns an index array that shows the indexes for the lexicographic sort along
 *the given axis.
 */
NPY_NO_EXPORT PyObject *
PyArray_LexSort(PyObject *sort_keys, int axis)
{
    NpyArray **mps;
    NpyArray *sorted = NULL;
    PyArrayObject *ret;
    int n;
    int i;

    if (!PySequence_Check(sort_keys)
           || ((n = PySequence_Size(sort_keys)) <= 0)) {
        PyErr_SetString(PyExc_TypeError,
                "need sequence of keys with len > 0 in lexsort");
        return NULL;
    }
    mps = (NpyArray **) _pya_malloc(n*sizeof(PyArrayObject*));
    if (mps == NULL) {
        return PyErr_NoMemory();
    }
    for (i = 0; i < n; i++) {
        mps[i] = NULL;
    }
    for (i = 0; i < n; i++) {
        PyObject *obj;
        PyArrayObject *arr;
        obj = PySequence_GetItem(sort_keys, i);
        arr = (PyArrayObject *)PyArray_FROM_O(obj);
        Py_DECREF(obj);
        if (arr == NULL) {
            goto fail;
        }
        mps[i] = PyArray_ARRAY(arr);
        Npy_INCREF(mps[i]);
        Py_DECREF(arr);
    }

    sorted = NpyArray_LexSort(mps, n, axis);
    if (sorted == NULL) {
        goto fail;
    }

    for (i = 0; i < n; i++) {
        Npy_XDECREF(mps[i]);
    }
    _pya_free(mps);

    ret = Npy_INTERFACE(sorted);
    Py_INCREF(ret);
    Npy_DECREF(sorted);
    
    return (PyObject *)ret;

 fail:
    Npy_XDECREF(sorted);
    for (i = 0; i < n; i++) {
        Npy_XDECREF(mps[i]);
    }
    _pya_free(mps);
    return NULL;
}

/*NUMPY_API
 * Numeric.searchsorted(a,v)
 */
NPY_NO_EXPORT PyObject *
PyArray_SearchSorted(PyArrayObject *op1, PyObject *op2, NPY_SEARCHSIDE side)
{
    PyArrayObject* ap2 = NULL;
    PyArrayObject* ret = NULL;

    if (PyArray_Check(op2)) {
        ap2 = (PyArrayObject*)op2;
        Py_INCREF(ap2);
    } else {
        NpyArray_Descr* dtype;

        dtype = PyArray_DescrFromObjectUnwrap((PyObject *)op2, PyArray_DESCR(op1));
        /* need ap2 as contiguous array and of right type */
        ap2 = (PyArrayObject *)PyArray_FromAnyUnwrap(op2, dtype,
                                                     0, 0, NPY_DEFAULT, NULL);
        if (ap2 == NULL) {
            Npy_DECREF(dtype);
            goto finish;
        }
    }

    ASSIGN_TO_PYARRAY(ret, 
                      NpyArray_SearchSorted(PyArray_ARRAY(op1), 
                                            PyArray_ARRAY(ap2), side));
  finish:
    Py_XDECREF(ap2);
    return (PyObject *) ret;
}

/*NUMPY_API
 * Diagonal
 */
NPY_NO_EXPORT PyObject *
PyArray_Diagonal(PyArrayObject *self, int offset, int axis1, int axis2)
{
    int n = PyArray_NDIM(self);
    PyObject *new;
    PyArray_Dims newaxes;
    intp dims[MAX_DIMS];
    int i, pos;

    newaxes.ptr = dims;
    if (n < 2) {
        PyErr_SetString(PyExc_ValueError,
                        "array.ndim must be >= 2");
        return NULL;
    }
    if (axis1 < 0) {
        axis1 += n;
    }
    if (axis2 < 0) {
        axis2 += n;
    }
    if ((axis1 == axis2) || (axis1 < 0) || (axis1 >= n) ||
        (axis2 < 0) || (axis2 >= n)) {
        PyErr_Format(PyExc_ValueError, "axis1(=%d) and axis2(=%d) "\
                     "must be different and within range (nd=%d)",
                     axis1, axis2, n);
        return NULL;
    }

    newaxes.len = n;
    /* insert at the end */
    newaxes.ptr[n-2] = axis1;
    newaxes.ptr[n-1] = axis2;
    pos = 0;
    for (i = 0; i < n; i++) {
        if ((i==axis1) || (i==axis2)) {
            continue;
        }
        newaxes.ptr[pos++] = i;
    }
    new = PyArray_Transpose(self, &newaxes);
    if (new == NULL) {
        return NULL;
    }
    self = (PyAO *)new;

    if (n == 2) {
        PyObject *a = NULL, *indices= NULL, *ret = NULL;
        intp n1, n2, start, stop, step, count;
        intp *dptr;

        n1 = PyArray_DIM(self, 0);
        n2 = PyArray_DIM(self, 1);
        step = n2 + 1;
        if (offset < 0) {
            start = -n2 * offset;
            stop = MIN(n2, n1+offset)*(n2+1) - n2*offset;
        }
        else {
            start = offset;
            stop = MIN(n1, n2-offset)*(n2+1) + offset;
        }

        /* count = ceil((stop-start)/step) */
        count = ((stop-start) / step) + (((stop-start) % step) != 0);
        indices = PyArray_New(&PyArray_Type, 1, &count,
                              PyArray_INTP, NULL, NULL, 0, 0, NULL);
        if (indices == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        dptr = (intp *)PyArray_DATA(indices);
        for (n1 = start; n1 < stop; n1 += step) {
            *dptr++ = n1;
        }
        a = PyArray_IterNew((PyObject *)self);
        Py_DECREF(self);
        if (a == NULL) {
            Py_DECREF(indices);
            return NULL;
        }
        ret = PyObject_GetItem(a, indices);
        Py_DECREF(a);
        Py_DECREF(indices);
        return ret;
    }

    else {
        /*
         * my_diagonal = []
         * for i in range (s [0]) :
         * my_diagonal.append (diagonal (a [i], offset))
         * return array (my_diagonal)
         */
        PyObject *mydiagonal = NULL, *new = NULL, *ret = NULL, *sel = NULL;
        intp i, n1;
        int res;
        NpyArray_Descr *typecode;

        typecode = PyArray_DESCR(self);
        mydiagonal = PyList_New(0);
        if (mydiagonal == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        n1 = PyArray_DIM(self, 0);
        for (i = 0; i < n1; i++) {
            new = PyInt_FromLong((long) i);
            sel = PyArray_EnsureAnyArray(PyObject_GetItem((PyObject *)self, new));
            Py_DECREF(new);
            if (sel == NULL) {
                Py_DECREF(self);
                Py_DECREF(mydiagonal);
                return NULL;
            }
            new = PyArray_Diagonal((PyAO *)sel, offset, n-3, n-2);
            Py_DECREF(sel);
            if (new == NULL) {
                Py_DECREF(self);
                Py_DECREF(mydiagonal);
                return NULL;
            }
            res = PyList_Append(mydiagonal, new);
            Py_DECREF(new);
            if (res < 0) {
                Py_DECREF(self);
                Py_DECREF(mydiagonal);
                return NULL;
            }
        }
        Py_DECREF(self);
        Npy_INCREF(typecode);
        ret =  PyArray_FromAnyUnwrap(mydiagonal, typecode, 0, 0, 0, NULL);
        Py_DECREF(mydiagonal);
        return ret;
    }
}

/*NUMPY_API
 * Compress
 */
NPY_NO_EXPORT PyObject *
PyArray_Compress(PyArrayObject *self, PyObject *condition, int axis,
                 PyArrayObject *out)
{
    PyArrayObject *cond;
    PyObject *res, *ret;

    cond = (PyAO *)PyArray_FROM_O(condition);
    if (cond == NULL) {
        return NULL;
    }
    if (PyArray_NDIM(cond) != 1) {
        Py_DECREF(cond);
        PyErr_SetString(PyExc_ValueError,
                        "condition must be 1-d array");
        return NULL;
    }

    res = PyArray_Nonzero(cond);
    Py_DECREF(cond);
    if (res == NULL) {
        return res;
    }
    ret = PyArray_TakeFrom(self, PyTuple_GET_ITEM(res, 0), axis,
                           out, NPY_RAISE);
    Py_DECREF(res);
    return ret;
}

/*NUMPY_API
 * Nonzero
 */
NPY_NO_EXPORT PyObject *
PyArray_Nonzero(PyArrayObject *self)
{
    int n = PyArray_NDIM(self), j;
    NpyArray *arrays[MAX_DIMS];
    PyObject* ret;

    ret = PyTuple_New(n);
    if (ret == NULL) {
        return NULL;
    }

    if (NpyArray_NonZero(PyArray_ARRAY(self), arrays, self) < 0) {
        Py_DECREF(ret);
        return NULL;
    }

    for (j=0; j<n; j++) {
        PyTuple_SET_ITEM(ret, j, Npy_INTERFACE(arrays[j]));
    }

    return ret;
}

