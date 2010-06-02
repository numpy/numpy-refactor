#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "numpy/numpy_api.h"

#include "numpy/npy_math.h"

#include "npy_config.h"

#include "npy_3kcompat.h"

#include "common.h"
#include "ctors.h"

#define PyAO PyArrayObject
#define _check_axis PyArray_CheckAxis

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
        indices = (PyArrayObject*) PyArray_ContiguousFromAny(indices0, PyArray_INTP, 1, 0);
        if (indices == NULL) {
            return NULL;
        }
    }
    result = (PyObject*) NpyArray_TakeFrom(self0, indices, axis, ret, clipmode);
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
    NpyArray* indices = NULL;
    NpyArray* values = NULL;

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
        Py_INCREF(self->descr);
        values = (PyArrayObject*) 
            PyArray_FromAny(values0, self->descr, 0, 0, NPY_CARRAY, NULL);
        if (values == NULL) {
            goto fail;
        }
    }
    NpyArray_PutTo(self, values, indices, clipmode);
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
        Py_INCREF(self->descr);
        values = (PyArrayObject*) 
            PyArray_FromAny(values0, self->descr, 0, 0, NPY_CARRAY, NULL);
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

    if (NpyArray_PutMask(self, values, mask) == -1) {
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
    result = (PyObject*) NpyArray_Repeat(aop, repeats, axis);
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
    result =  (PyObject*) NpyArray_Choose(ip, mps, n, ret, clipmode);
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
    return NpyArray_Sort(op, axis, which);
}


/*NUMPY_API
 * ArgSort an array
 */
NPY_NO_EXPORT PyObject *
PyArray_ArgSort(PyArrayObject *op, int axis, NPY_SORTKIND which)
{
    return (PyObject*) NpyArray_ArgSort(op, axis, which);
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
    PyArrayObject **mps;
    PyArrayIterObject **its;
    PyArrayObject *ret = NULL;
    PyArrayIterObject *rit = NULL;
    int n;
    int nd;
    int needcopy = 0, i,j;
    intp N, size;
    int elsize;
    int maxelsize;
    intp astride, rstride, *iptr;
    int object = 0;
    PyArray_ArgSortFunc *argsort;
    NPY_BEGIN_THREADS_DEF;

    if (!PySequence_Check(sort_keys)
           || ((n = PySequence_Size(sort_keys)) <= 0)) {
        PyErr_SetString(PyExc_TypeError,
                "need sequence of keys with len > 0 in lexsort");
        return NULL;
    }
    mps = (PyArrayObject **) _pya_malloc(n*sizeof(PyArrayObject));
    if (mps == NULL) {
        return PyErr_NoMemory();
    }
    its = (PyArrayIterObject **) _pya_malloc(n*sizeof(PyArrayIterObject));
    if (its == NULL) {
        _pya_free(mps);
        return PyErr_NoMemory();
    }
    for (i = 0; i < n; i++) {
        mps[i] = NULL;
        its[i] = NULL;
    }
    for (i = 0; i < n; i++) {
        PyObject *obj;
        obj = PySequence_GetItem(sort_keys, i);
        mps[i] = (PyArrayObject *)PyArray_FROM_O(obj);
        Py_DECREF(obj);
        if (mps[i] == NULL) {
            goto fail;
        }
        if (i > 0) {
            if ((mps[i]->nd != mps[0]->nd)
                || (!PyArray_CompareLists(mps[i]->dimensions,
                                       mps[0]->dimensions,
                                       mps[0]->nd))) {
                PyErr_SetString(PyExc_ValueError,
                                "all keys need to be the same shape");
                goto fail;
            }
        }
        if (!mps[i]->descr->f->argsort[PyArray_MERGESORT]) {
            PyErr_Format(PyExc_TypeError,
                         "merge sort not available for item %d", i);
            goto fail;
        }
        if (!object
            && PyDataType_FLAGCHK(mps[i]->descr, NPY_NEEDS_PYAPI)) {
            object = 1;
        }
        its[i] = (PyArrayIterObject *)PyArray_IterAllButAxis(
                (PyObject *)mps[i], &axis);
        if (its[i] == NULL) {
            goto fail;
        }
    }

    /* Now we can check the axis */
    nd = mps[0]->nd;
    if ((nd == 0) || (PyArray_SIZE(mps[0]) == 1)) {
        /* single element case */
        ret = (PyArrayObject *)PyArray_New(&PyArray_Type, mps[0]->nd,
                                           mps[0]->dimensions,
                                           PyArray_INTP,
                                           NULL, NULL, 0, 0, NULL);

        if (ret == NULL) {
            goto fail;
        }
        *((intp *)(ret->data)) = 0;
        goto finish;
    }
    if (axis < 0) {
        axis += nd;
    }
    if ((axis < 0) || (axis >= nd)) {
        PyErr_Format(PyExc_ValueError,
                "axis(=%d) out of bounds", axis);
        goto fail;
    }

    /* Now do the sorting */
    ret = (PyArrayObject *)PyArray_New(&PyArray_Type, mps[0]->nd,
                                       mps[0]->dimensions, PyArray_INTP,
                                       NULL, NULL, 0, 0, NULL);
    if (ret == NULL) {
        goto fail;
    }
    rit = (PyArrayIterObject *)
            PyArray_IterAllButAxis((PyObject *)ret, &axis);
    if (rit == NULL) {
        goto fail;
    }
    if (!object) {
        NPY_BEGIN_THREADS;
    }
    size = rit->size;
    N = mps[0]->dimensions[axis];
    rstride = PyArray_STRIDE(ret, axis);
    maxelsize = mps[0]->descr->elsize;
    needcopy = (rstride != sizeof(intp));
    for (j = 0; j < n; j++) {
        needcopy = needcopy
            || PyArray_ISBYTESWAPPED(mps[j])
            || !(mps[j]->flags & ALIGNED)
            || (mps[j]->strides[axis] != (intp)mps[j]->descr->elsize);
        if (mps[j]->descr->elsize > maxelsize) {
            maxelsize = mps[j]->descr->elsize;
        }
    }

    if (needcopy) {
        char *valbuffer, *indbuffer;
        int *swaps;

        valbuffer = PyDataMem_NEW(N*maxelsize);
        indbuffer = PyDataMem_NEW(N*sizeof(intp));
        swaps = malloc(n*sizeof(int));
        for (j = 0; j < n; j++) {
            swaps[j] = PyArray_ISBYTESWAPPED(mps[j]);
        }
        while (size--) {
            iptr = (intp *)indbuffer;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            for (j = 0; j < n; j++) {
                elsize = mps[j]->descr->elsize;
                astride = mps[j]->strides[axis];
                argsort = mps[j]->descr->f->argsort[PyArray_MERGESORT];
                _unaligned_strided_byte_copy(valbuffer, (intp) elsize,
                                             its[j]->dataptr, astride, N, elsize);
                if (swaps[j]) {
                    _strided_byte_swap(valbuffer, (intp) elsize, N, elsize);
                }
                if (argsort(valbuffer, (intp *)indbuffer, N, mps[j]) < 0) {
                    PyDataMem_FREE(valbuffer);
                    PyDataMem_FREE(indbuffer);
                    free(swaps);
                    goto fail;
                }
                PyArray_ITER_NEXT(its[j]);
            }
            _unaligned_strided_byte_copy(rit->dataptr, rstride, indbuffer,
                                         sizeof(intp), N, sizeof(intp));
            PyArray_ITER_NEXT(rit);
        }
        PyDataMem_FREE(valbuffer);
        PyDataMem_FREE(indbuffer);
        free(swaps);
    }
    else {
        while (size--) {
            iptr = (intp *)rit->dataptr;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            for (j = 0; j < n; j++) {
                argsort = mps[j]->descr->f->argsort[PyArray_MERGESORT];
                if (argsort(its[j]->dataptr, (intp *)rit->dataptr,
                            N, mps[j]) < 0) {
                    goto fail;
                }
                PyArray_ITER_NEXT(its[j]);
            }
            PyArray_ITER_NEXT(rit);
        }
    }

    if (!object) {
        NPY_END_THREADS;
    }

 finish:
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
        Py_XDECREF(its[i]);
    }
    Py_XDECREF(rit);
    _pya_free(mps);
    _pya_free(its);
    return (PyObject *)ret;

 fail:
    NPY_END_THREADS;
    Py_XDECREF(rit);
    Py_XDECREF(ret);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
        Py_XDECREF(its[i]);
    }
    _pya_free(mps);
    _pya_free(its);
    return NULL;
}


/** @brief Use bisection of sorted array to find first entries >= keys.
 *
 * For each key use bisection to find the first index i s.t. key <= arr[i].
 * When there is no such index i, set i = len(arr). Return the results in ret.
 * All arrays are assumed contiguous on entry and both arr and key must be of
 * the same comparable type.
 *
 * @param arr contiguous sorted array to be searched.
 * @param key contiguous array of keys.
 * @param ret contiguous array of intp for returned indices.
 * @return void
 */
static void
local_search_left(PyArrayObject *arr, PyArrayObject *key, PyArrayObject *ret)
{
    PyArray_CompareFunc *compare = key->descr->f->compare;
    intp nelts = arr->dimensions[arr->nd - 1];
    intp nkeys = PyArray_SIZE(key);
    char *parr = arr->data;
    char *pkey = key->data;
    intp *pret = (intp *)ret->data;
    int elsize = arr->descr->elsize;
    intp i;

    for (i = 0; i < nkeys; ++i) {
        intp imin = 0;
        intp imax = nelts;
        while (imin < imax) {
            intp imid = imin + ((imax - imin) >> 1);
            if (compare(parr + elsize*imid, pkey, key) < 0) {
                imin = imid + 1;
            }
            else {
                imax = imid;
            }
        }
        *pret = imin;
        pret += 1;
        pkey += elsize;
    }
}


/** @brief Use bisection of sorted array to find first entries > keys.
 *
 * For each key use bisection to find the first index i s.t. key < arr[i].
 * When there is no such index i, set i = len(arr). Return the results in ret.
 * All arrays are assumed contiguous on entry and both arr and key must be of
 * the same comparable type.
 *
 * @param arr contiguous sorted array to be searched.
 * @param key contiguous array of keys.
 * @param ret contiguous array of intp for returned indices.
 * @return void
 */
static void
local_search_right(PyArrayObject *arr, PyArrayObject *key, PyArrayObject *ret)
{
    PyArray_CompareFunc *compare = key->descr->f->compare;
    intp nelts = arr->dimensions[arr->nd - 1];
    intp nkeys = PyArray_SIZE(key);
    char *parr = arr->data;
    char *pkey = key->data;
    intp *pret = (intp *)ret->data;
    int elsize = arr->descr->elsize;
    intp i;

    for(i = 0; i < nkeys; ++i) {
        intp imin = 0;
        intp imax = nelts;
        while (imin < imax) {
            intp imid = imin + ((imax - imin) >> 1);
            if (compare(parr + elsize*imid, pkey, key) <= 0) {
                imin = imid + 1;
            }
            else {
                imax = imid;
            }
        }
        *pret = imin;
        pret += 1;
        pkey += elsize;
    }
}

/*NUMPY_API
 * Numeric.searchsorted(a,v)
 */
NPY_NO_EXPORT PyObject *
PyArray_SearchSorted(PyArrayObject *op1, PyObject *op2, NPY_SEARCHSIDE side)
{
    PyArrayObject *ap1 = NULL;
    PyArrayObject *ap2 = NULL;
    PyArrayObject *ret = NULL;
    PyArray_Descr *dtype;
    NPY_BEGIN_THREADS_DEF;

    dtype = PyArray_DescrFromObject((PyObject *)op2, op1->descr);
    /* need ap1 as contiguous array and of right type */
    Py_INCREF(dtype);
    ap1 = (PyArrayObject *)PyArray_FromAny((PyObject *)op1, dtype,
                                           1, 1, NPY_DEFAULT, NULL);
    if (ap1 == NULL) {
        Py_DECREF(dtype);
        return NULL;
    }

    /* need ap2 as contiguous array and of right type */
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, dtype,
                                          0, 0, NPY_DEFAULT, NULL);
    if (ap2 == NULL) {
        goto fail;
    }
    /* ret is a contiguous array of intp type to hold returned indices */
    ret = (PyArrayObject *)PyArray_New(Py_TYPE(ap2), ap2->nd,
                                       ap2->dimensions, PyArray_INTP,
                                       NULL, NULL, 0, 0, (PyObject *)ap2);
    if (ret == NULL) {
        goto fail;
    }
    /* check that comparison function exists */
    if (ap2->descr->f->compare == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "compare not supported for type");
        goto fail;
    }

    if (side == NPY_SEARCHLEFT) {
        NPY_BEGIN_THREADS_DESCR(ap2->descr);
        local_search_left(ap1, ap2, ret);
        NPY_END_THREADS_DESCR(ap2->descr);
    }
    else if (side == NPY_SEARCHRIGHT) {
        NPY_BEGIN_THREADS_DESCR(ap2->descr);
        local_search_right(ap1, ap2, ret);
        NPY_END_THREADS_DESCR(ap2->descr);
    }
    Py_DECREF(ap1);
    Py_DECREF(ap2);
    return (PyObject *)ret;

 fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ret);
    return NULL;
}

/*NUMPY_API
 * Diagonal
 */
NPY_NO_EXPORT PyObject *
PyArray_Diagonal(PyArrayObject *self, int offset, int axis1, int axis2)
{
    int n = self->nd;
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

        n1 = self->dimensions[0];
        n2 = self->dimensions[1];
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
        PyArray_Descr *typecode;

        typecode = self->descr;
        mydiagonal = PyList_New(0);
        if (mydiagonal == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        n1 = self->dimensions[0];
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
        Py_INCREF(typecode);
        ret =  PyArray_FromAny(mydiagonal, typecode, 0, 0, 0, NULL);
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
    if (cond->nd != 1) {
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
    int n = self->nd, j;
    intp count = 0, i, size;
    PyArrayIterObject *it = NULL;
    PyObject *ret = NULL, *item;
    intp *dptr[MAX_DIMS];

    it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
    if (it == NULL) {
        return NULL;
    }
    size = it->size;
    for (i = 0; i < size; i++) {
        if (self->descr->f->nonzero(it->dataptr, self)) {
            count++;
        }
        PyArray_ITER_NEXT(it);
    }

    PyArray_ITER_RESET(it);
    ret = PyTuple_New(n);
    if (ret == NULL) {
        goto fail;
    }
    for (j = 0; j < n; j++) {
        item = PyArray_New(Py_TYPE(self), 1, &count,
                           PyArray_INTP, NULL, NULL, 0, 0,
                           (PyObject *)self);
        if (item == NULL) {
            goto fail;
        }
        PyTuple_SET_ITEM(ret, j, item);
        dptr[j] = (intp *)PyArray_DATA(item);
    }
    if (n == 1) {
        for (i = 0; i < size; i++) {
            if (self->descr->f->nonzero(it->dataptr, self)) {
                *(dptr[0])++ = i;
            }
            PyArray_ITER_NEXT(it);
        }
    }
    else {
        /* reset contiguous so that coordinates gets updated */
        it->contiguous = 0;
        for (i = 0; i < size; i++) {
            if (self->descr->f->nonzero(it->dataptr, self)) {
                for (j = 0; j < n; j++) {
                    *(dptr[j])++ = it->coordinates[j];
                }
            }
            PyArray_ITER_NEXT(it);
        }
    }

    Py_DECREF(it);
    return ret;

 fail:
    Py_XDECREF(ret);
    Py_XDECREF(it);
    return NULL;

}

