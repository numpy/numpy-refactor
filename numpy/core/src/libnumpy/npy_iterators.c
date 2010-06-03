#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"
#include "npy_3kcompat.h"

/* XXX: We should be getting this from an include. */
#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif

/* get the dataptr from its current coordinates for simple iterator */
static char*
get_ptr_simple(NpyArrayIter* iter, npy_intp *coordinates)
{
    npy_intp i;
    char *ret;

    ret = iter->ao->data;

    for(i = 0; i < iter->ao->nd; ++i) {
            ret += coordinates[i] * iter->strides[i];
    }

    return ret;
}


/*
 * This is common initialization code between NpyArrayIter and
 * PyArrayNeighborhoodIterObject
 *
 * Increase ao refcount
 */
static int
array_iter_base_init(NpyArrayIter *it, NpyArray *ao)
{
    int nd, i;

    nd = ao->nd;
    NpyArray_UpdateFlags(ao, NPY_CONTIGUOUS);
    if (NpyArray_ISCONTIGUOUS(ao)) {
        it->contiguous = 1;
    }
    else {
        it->contiguous = 0;
    }
    Npy_INCREF(ao);
    it->ao = ao;
    it->size = NpyArray_SIZE(ao);
    it->nd_m1 = nd - 1;
    it->factors[nd-1] = 1;
    for (i = 0; i < nd; i++) {
        it->dims_m1[i] = ao->dimensions[i] - 1;
        it->strides[i] = ao->strides[i];
        it->backstrides[i] = it->strides[i] * it->dims_m1[i];
        if (i > 0) {
            it->factors[nd-i-1] = it->factors[nd-i] * ao->dimensions[nd-i];
        }
        it->bounds[i][0] = 0;
        it->bounds[i][1] = ao->dimensions[i] - 1;
        it->limits[i][0] = 0;
        it->limits[i][1] = ao->dimensions[i] - 1;
        it->limits_sizes[i] = it->limits[i][1] - it->limits[i][0] + 1;
    }

    it->translate = &get_ptr_simple;
    NpyArray_ITER_RESET(it);

    return 0;
}


/*
 * Get Iterator.
 */
NpyArrayIter *
NpyArray_IterNew(NpyArray *ao)
{
    NpyArrayIter *it;

    it = (NpyArrayIter *)NpyArray_malloc(sizeof(NpyArrayIter));
    NpyObject_Init((NpyObject *)it, &NpyArrayIter_Type);
    if (it == NULL) {
        return NULL;
    }

    array_iter_base_init(it, ao);
    return it;
}

/*
 * Get Iterator broadcast to a particular shape
 */
NpyArrayIter *
NpyArray_BroadcastToShape(NpyArray *ao, npy_intp *dims, int nd)
{
    NpyArrayIter *it;
    int i, diff, j, compat, k;

    if (ao->nd > nd) {
        goto err;
    }
    compat = 1;
    diff = j = nd - ao->nd;
    for (i = 0; i < ao->nd; i++, j++) {
        if (ao->dimensions[i] == 1) {
            continue;
        }
        if (ao->dimensions[i] != dims[j]) {
            compat = 0;
            break;
        }
    }
    if (!compat) {
        goto err;
    }
    it = (NpyArrayIter *) PyArray_malloc(sizeof(NpyArrayIter));
    NpyObject_Init((NpyObject *)it, &NpyArrayIter_Type);

    if (it == NULL) {
        return NULL;
    }
    NpyArray_UpdateFlags(ao, NPY_CONTIGUOUS);
    if (NpyArray_ISCONTIGUOUS(ao)) {
        it->contiguous = 1;
    }
    else {
        it->contiguous = 0;
    }
    Npy_INCREF(ao);
    it->ao = ao;
    it->size = NpyArray_MultiplyList(dims, nd);
    it->nd_m1 = nd - 1;
    it->factors[nd-1] = 1;
    for (i = 0; i < nd; i++) {
        it->dims_m1[i] = dims[i] - 1;
        k = i - diff;
        if ((k < 0) || ao->dimensions[k] != dims[i]) {
            it->contiguous = 0;
            it->strides[i] = 0;
        }
        else {
            it->strides[i] = ao->strides[k];
        }
        it->backstrides[i] = it->strides[i] * it->dims_m1[i];
        if (i > 0) {
            it->factors[nd-i-1] = it->factors[nd-i] * dims[nd-i];
        }
    }
    NpyArray_ITER_RESET(it);
    return it;

 err:
    NpyErr_SetString(NpyExc_ValueError, "array is not broadcastable to "\
                    "correct shape");
    return NULL;
}


/*
 * Get Iterator that iterates over all but one axis (don't use this with
 * PyArray_ITER_GOTO1D).  The axis will be over-written if negative
 * with the axis having the smallest stride.
 */
NpyArrayIter *
NpyArray_IterAllButAxis(NpyArray* obj, int *inaxis)
{
    NpyArrayIter* it;
    int axis;
    it = NpyArray_IterNew(obj);
    if (it == NULL) {
        return NULL;
    }
    if (NpyArray_NDIM(obj)==0) {
        return it;
    }
    if (*inaxis < 0) {
        int i, minaxis = 0;
        npy_intp minstride = 0;
        i = 0;
        while (minstride == 0 && i < NpyArray_NDIM(obj)) {
            minstride = NpyArray_STRIDE(obj,i);
            i++;
        }
        for (i = 1; i < NpyArray_NDIM(obj); i++) {
            if (NpyArray_STRIDE(obj,i) > 0 &&
                NpyArray_STRIDE(obj, i) < minstride) {
                minaxis = i;
                minstride = NpyArray_STRIDE(obj,i);
            }
        }
        *inaxis = minaxis;
    }
    axis = *inaxis;
    /* adjust so that will not iterate over axis */
    it->contiguous = 0;
    if (it->size != 0) {
        it->size /= NpyArray_DIM(obj,axis);
    }
    it->dims_m1[axis] = 0;
    it->backstrides[axis] = 0;

    /*
     * (won't fix factors so don't use
     * PyArray_ITER_GOTO1D with this iterator)
     */
    return it;
}

/*
 * Adjusts previously broadcasted iterators so that the axis with
 * the smallest sum of iterator strides is not iterated over.
 * Returns dimension which is smallest in the range [0,multi->nd).
 * A -1 is returned if multi->nd == 0.
 *
 * don't use with PyArray_ITER_GOTO1D because factors are not adjusted
 */
int
NpyArray_RemoveSmallest(NpyArrayMultiIterObject *multi)
{
    NpyArrayIter *it;
    int i, j;
    int axis;
    npy_intp smallest;
    npy_intp sumstrides[NPY_MAXDIMS];

    if (multi->nd == 0) {
        return -1;
    }
    for (i = 0; i < multi->nd; i++) {
        sumstrides[i] = 0;
        for (j = 0; j < multi->numiter; j++) {
            sumstrides[i] += multi->iters[j]->strides[i];
        }
    }
    axis = 0;
    smallest = sumstrides[0];
    /* Find longest dimension */
    for (i = 1; i < multi->nd; i++) {
        if (sumstrides[i] < smallest) {
            axis = i;
            smallest = sumstrides[i];
        }
    }
    for(i = 0; i < multi->numiter; i++) {
        it = multi->iters[i];
        it->contiguous = 0;
        if (it->size != 0) {
            it->size /= (it->dims_m1[axis]+1);
        }
        it->dims_m1[axis] = 0;
        it->backstrides[axis] = 0;
    }
    multi->size = multi->iters[0]->size;
    return axis;
}

/* Adjust dimensionality and strides for index object iterators
   --- i.e. broadcast
*/
int
NpyArray_Broadcast(NpyArrayMultiIterObject *mit)
{
    int i, nd, k, j;
    npy_intp tmp;
    NpyArrayIter *it;

    /* Discover the broadcast number of dimensions */
    for (i = 0, nd = 0; i < mit->numiter; i++) {
        nd = MAX(nd, mit->iters[i]->ao->nd);
    }
    mit->nd = nd;

    /* Discover the broadcast shape in each dimension */
    for (i = 0; i < nd; i++) {
        mit->dimensions[i] = 1;
        for (j = 0; j < mit->numiter; j++) {
            it = mit->iters[j];
            /* This prepends 1 to shapes not already equal to nd */
            k = i + it->ao->nd - nd;
            if (k >= 0) {
                tmp = it->ao->dimensions[k];
                if (tmp == 1) {
                    continue;
                }
                if (mit->dimensions[i] == 1) {
                    mit->dimensions[i] = tmp;
                }
                else if (mit->dimensions[i] != tmp) {
                    NpyErr_SetString(NpyExc_ValueError,
                                    "shape mismatch: objects" \
                                    " cannot be broadcast" \
                                    " to a single shape");
                    return -1;
                }
            }
        }
    }

    /*
     * Reset the iterator dimensions and strides of each iterator
     * object -- using 0 valued strides for broadcasting
     * Need to check for overflow
     */
    tmp = NpyArray_OverflowMultiplyList(mit->dimensions, mit->nd);
    if (tmp < 0) {
        NpyErr_SetString(NpyExc_ValueError,
                        "broadcast dimensions too large.");
        return -1;
    }
    mit->size = tmp;
    for (i = 0; i < mit->numiter; i++) {
        it = mit->iters[i];
        it->nd_m1 = mit->nd - 1;
        it->size = tmp;
        nd = it->ao->nd;
        it->factors[mit->nd-1] = 1;
        for (j = 0; j < mit->nd; j++) {
            it->dims_m1[j] = mit->dimensions[j] - 1;
            k = j + nd - mit->nd;
            /*
             * If this dimension was added or shape of
             * underlying array was 1
             */
            if ((k < 0) ||
                it->ao->dimensions[k] != mit->dimensions[j]) {
                it->contiguous = 0;
                it->strides[j] = 0;
            }
            else {
                it->strides[j] = it->ao->strides[k];
            }
            it->backstrides[j] = it->strides[j] * it->dims_m1[j];
            if (j > 0)
                it->factors[mit->nd-j-1] =
                    it->factors[mit->nd-j] * mit->dimensions[mit->nd-j];
        }
        NpyArray_ITER_RESET(it);
    }
    return 0;
}

/*
 * Get MultiIterator from array of Python objects and any additional
 *
 * NpyArray **mps -- array of NpyArrays
 * int n - number of NpyArrays in the array
 * int nadd - number of additional arrays to include in the iterator.
 *
 * Returns a multi-iterator object.
 */
NpyArrayMultiIterObject *
NpyArray_MultiIterFromArrays(NpyArray **mps, int n, int nadd, ...)
{
    NpyArrayMultiIterObject* result;

    va_list va;
    va_start(va, nadd);
    result = NpyArray_vMultiIterFromArrays(mps, n, nadd, va);
    va_end(va);

    return result;
}

NpyArrayMultiIterObject *
NpyArray_vMultiIterFromArrays(NpyArray **mps, int n, int nadd, va_list va)
{
    PyArrayMultiIterObject *multi;
    NpyArray *current;
    PyObject *arr;

    int i, ntot, err=0;

    ntot = n + nadd;
    if (ntot < 2 || ntot > NPY_MAXARGS) {
        NpyErr_Format(NpyExc_ValueError,
                     "Need between 2 and (%d) "                 \
                     "array objects (inclusive).", NPY_MAXARGS);
        return NULL;
    }
    multi = NpyArray_malloc(sizeof(PyArrayMultiIterObject));
    if (multi == NULL) {
        NpyErr_NoMemory();
        return NULL;
    }
    NpyObject_Init((NpyObject *)multi, &NpyArrayMultiIter_Type);

    for (i = 0; i < ntot; i++) {
        multi->iters[i] = NULL;
    }
    multi->numiter = ntot;
    multi->index = 0;

    for (i = 0; i < ntot; i++) {
        if (i < n) {
            current = mps[i];
        }
        else {
            current = va_arg(va, NpyArray *);
            if (!PyArray_Check(current)) {
                err = 1;
                break;
            }
        }
        multi->iters[i] = NpyArray_IterNew(current);
    }

    if (!err && NpyArray_Broadcast(multi) < 0) {
        err = 1;
    }
    if (err) {
        Npy_DECREF(multi);
        return NULL;
    }
    NpyArray_MultiIter_RESET(multi);
    return multi;
}
