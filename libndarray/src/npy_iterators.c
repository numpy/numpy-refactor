#define _MULTIARRAYMODULE
#include <stdlib.h>
#include <memory.h>
#include <stdarg.h>
#include "npy_config.h"
#include "numpy_api.h"
#include "npy_arrayobject.h"
#include "npy_iterators.h"
#include "npy_internal.h"
#include "npy_index.h"



/* XXX: We should be getting this from an include. */
#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif

/* get the dataptr from its current coordinates for simple iterator */
static char*
get_ptr_simple(NpyArrayIterObject* iter, npy_intp *coordinates)
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
 * NpyArrayNeighborhoodIterObject
 *
 * Increase ao refcount
 */
static int
array_iter_base_init(NpyArrayIterObject *it, NpyArray *ao)
{
    int nd, i;

    Npy_INTERFACE(it) = NULL;
    it->magic_number = NPY_VALID_MAGIC;
    nd = ao->nd;
    NpyArray_UpdateFlags(ao, NPY_CONTIGUOUS);
    if (NpyArray_ISCONTIGUOUS(ao)) {
        it->contiguous = 1;
    }
    else {
        it->contiguous = 0;
    }
    _Npy_INCREF(ao);
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


static void
array_iter_base_dealloc(NpyArrayIterObject *it)
{
    Npy_INTERFACE(it) = NULL;
    _Npy_XDECREF(it->ao);
    it->magic_number = NPY_INVALID_MAGIC;
}

/*
 * Get Iterator.
 */
NpyArrayIterObject *
NpyArray_IterNew(NpyArray *ao)
{
    NpyArrayIterObject *it;

    it = (NpyArrayIterObject *)NpyArray_malloc(sizeof(NpyArrayIterObject));
    _NpyObject_Init((_NpyObject *)it, &NpyArrayIter_Type);
    if (it == NULL) {
        return NULL;
    }
    it->magic_number = NPY_VALID_MAGIC;

    array_iter_base_init(it, ao);
    if (NPY_FALSE == NpyInterface_IterNewWrapper(it, &it->nob_interface)) {
        Npy_INTERFACE(it) = NULL;
        _Npy_DECREF(it);
        return NULL;
    }
    return it;
}

/*
 * Get Iterator broadcast to a particular shape
 */
NpyArrayIterObject *
NpyArray_BroadcastToShape(NpyArray *ao, npy_intp *dims, int nd)
{
    NpyArrayIterObject *it;
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
    it = (NpyArrayIterObject *) NpyArray_malloc(sizeof(NpyArrayIterObject));
    if (it == NULL) {
        return NULL;
    }
    _NpyObject_Init((_NpyObject *)it, &NpyArrayIter_Type);
    it->magic_number = NPY_VALID_MAGIC;
    if (NPY_FALSE == NpyInterface_IterNewWrapper(it, &it->nob_interface)) {
        Npy_INTERFACE(it) = NULL;
        _Npy_DECREF(it);
        return NULL;
    }

    NpyArray_UpdateFlags(ao, NPY_CONTIGUOUS);
    if (NpyArray_ISCONTIGUOUS(ao)) {
        it->contiguous = 1;
    }
    else {
        it->contiguous = 0;
    }
    _Npy_INCREF(ao);
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

NpyArray *
NpyArray_IterSubcript(NpyArrayIterObject* self,
                      NpyIndex *indexes, int n)
{
    NpyIndex *index;

    if (n == 0 || n == 1 && indexes[0].type == NPY_INDEX_ELLIPSIS) {
        _Npy_INCREF(self->ao);
        return self->ao;
    }

    if (n > 1) {
        NpyErr_SetString(NpyExc_IndexError, "unsupported iterator index.");
        return NULL;
    }

    index = &indexes[0];

    return NULL;
}


/*
 * Get Iterator that iterates over all but one axis (don't use this with
 * NpyArray_ITER_GOTO1D).  The axis will be over-written if negative
 * with the axis having the smallest stride.
 */
NpyArrayIterObject *
NpyArray_IterAllButAxis(NpyArray* obj, int *inaxis)
{
    NpyArrayIterObject* it;
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
     * NpyArray_ITER_GOTO1D with this iterator)
     */
    return it;
}

/*
 * Adjusts previously broadcasted iterators so that the axis with
 * the smallest sum of iterator strides is not iterated over.
 * Returns dimension which is smallest in the range [0,multi->nd).
 * A -1 is returned if multi->nd == 0.
 *
 * don't use with NpyArray_ITER_GOTO1D because factors are not adjusted
 */
int
NpyArray_RemoveSmallest(NpyArrayMultiIterObject *multi)
{
    NpyArrayIterObject *it;
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
    NpyArrayIterObject *it;

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

static void
arrayiter_dealloc(NpyArrayIterObject *it)
{
    assert(0 == it->nob_refcnt);

    array_iter_base_dealloc(it);
    NpyArray_free(it);
}


_NpyTypeObject NpyArrayIter_Type = {
    (npy_destructor)arrayiter_dealloc,
};

static void
arraymultiter_dealloc(NpyArrayMultiIterObject *multi)
{
    int i;

    assert(0 == multi->nob_refcnt);

    for (i = 0; i < multi->numiter; i++) {
        _Npy_XDECREF(multi->iters[i]);
    }
    multi->magic_number = NPY_INVALID_MAGIC;
    NpyArray_free(multi);
}

_NpyTypeObject NpyArrayMultiIter_Type =   {
    (npy_destructor)arraymultiter_dealloc,
};



NpyArrayMultiIterObject *
NpyArray_vMultiIterFromArrays(NpyArray **mps, int n, int nadd, va_list va)
{
    NpyArrayMultiIterObject *multi;
    NpyArray *current;
    int i, ntot, err=0;
    char msg[1024];

    ntot = n + nadd;
    if (ntot < 2 || ntot > NPY_MAXARGS) {
        sprintf(msg, "Need between 2 and (%d) array objects (inclusive).",
                NPY_MAXARGS);
        NpyErr_SetString(NpyExc_ValueError, msg);
        return NULL;
    }
    multi = NpyArray_malloc(sizeof(NpyArrayMultiIterObject));
    if (multi == NULL) {
        NpyErr_SetString(NpyExc_MemoryError, "no memory");
        return NULL;
    }
    _NpyObject_Init((_NpyObject *)multi, &NpyArrayMultiIter_Type);
    multi->magic_number = NPY_VALID_MAGIC;

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
        }
        multi->iters[i] = NpyArray_IterNew(current);
    }

    if (!err && NpyArray_Broadcast(multi) < 0) {
        err = 1;
    }
    if (err) {
        _Npy_DECREF(multi);
        return NULL;
    }
    NpyArray_MultiIter_RESET(multi);
    if (NPY_FALSE == NpyInterface_MultiIterNewWrapper(multi, &multi->nob_interface)) {
        Npy_INTERFACE(multi) = NULL;
        _Npy_DECREF(multi);
        return NULL;
    }
    return multi;
}


NpyArrayMultiIterObject *
NpyArray_MultiIterNew()
{
    NpyArrayMultiIterObject *ret;

    ret = NpyArray_malloc(sizeof(NpyArrayMultiIterObject));
    if (NULL == ret) {
        NpyErr_SetString(NpyExc_MemoryError, "no memory");
        return NULL;
    }
    _NpyObject_Init(ret, &NpyArrayMultiIter_Type);
    ret->magic_number = NPY_VALID_MAGIC;
    if (NPY_FALSE == NpyInterface_MultiIterNewWrapper(ret, &ret->nob_interface)) {
        Npy_INTERFACE(ret) = NULL;
        _Npy_DECREF(ret);
        return NULL;
    }
    return ret;
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




/*========================= Neighborhood iterator ======================*/

#define _INF_SET_PTR(c) \
bd = coordinates[c] + p->coordinates[c]; \
if (bd < p->limits[c][0] || bd > p->limits[c][1]) { \
return niter->constant; \
} \
_coordinates[c] = bd;

/* set the dataptr from its current coordinates */
static char*
get_ptr_constant(NpyArrayIterObject* _iter, npy_intp *coordinates)
{
    int i;
    npy_intp bd, _coordinates[NPY_MAXDIMS];
    NpyArrayNeighborhoodIterObject *niter =
        (NpyArrayNeighborhoodIterObject*)_iter;
    NpyArrayIterObject *p = niter->_internal_iter;

    for(i = 0; i < niter->nd; ++i) {
        _INF_SET_PTR(i)
    }

    return p->translate(p, _coordinates);
}
#undef _INF_SET_PTR

#define _NPY_IS_EVEN(x) ((x) % 2 == 0)

/* For an array x of dimension n, and given index i, returns j, 0 <= j < n
 * such as x[i] = x[j], with x assumed to be mirrored. For example, for x =
 * {1, 2, 3} (n = 3)
 *
 * index -5 -4 -3 -2 -1 0 1 2 3 4 5 6
 * value  2  3  3  2  1 1 2 3 3 2 1 1
 *
 * _npy_pos_index_mirror(4, 3) will return 1, because x[4] = x[1]*/
static inline npy_intp
__npy_pos_remainder(npy_intp i, npy_intp n)
{
    npy_intp k, l, j;

    /* Mirror i such as it is guaranteed to be positive */
    if (i < 0) {
        i = - i - 1;
    }

    /* compute k and l such as i = k * n + l, 0 <= l < k */
    k = i / n;
    l = i - k * n;

    if (_NPY_IS_EVEN(k)) {
        j = l;
    } else {
        j = n - 1 - l;
    }
    return j;
}
#undef _NPY_IS_EVEN

#define _INF_SET_PTR_MIRROR(c) \
lb = p->limits[c][0]; \
bd = coordinates[c] + p->coordinates[c] - lb; \
_coordinates[c] = lb + __npy_pos_remainder(bd, p->limits_sizes[c]);

/* set the dataptr from its current coordinates */
static char*
get_ptr_mirror(NpyArrayIterObject* _iter, npy_intp *coordinates)
{
    int i;
    npy_intp bd, _coordinates[NPY_MAXDIMS], lb;
    NpyArrayNeighborhoodIterObject *niter =
        (NpyArrayNeighborhoodIterObject*)_iter;
    NpyArrayIterObject *p = niter->_internal_iter;

    for(i = 0; i < niter->nd; ++i) {
        _INF_SET_PTR_MIRROR(i)
    }

    return p->translate(p, _coordinates);
}
#undef _INF_SET_PTR_MIRROR

/* compute l such as i = k * n + l, 0 <= l < |k| */
static inline npy_intp
__npy_euclidean_division(npy_intp i, npy_intp n)
{
    npy_intp l;

    l = i % n;
    if (l < 0) {
        l += n;
    }
    return l;
}

#define _INF_SET_PTR_CIRCULAR(c) \
lb = p->limits[c][0]; \
bd = coordinates[c] + p->coordinates[c] - lb; \
_coordinates[c] = lb + __npy_euclidean_division(bd, p->limits_sizes[c]);

static char*
get_ptr_circular(NpyArrayIterObject* _iter, npy_intp *coordinates)
{
    int i;
    npy_intp bd, _coordinates[NPY_MAXDIMS], lb;
    NpyArrayNeighborhoodIterObject *niter =
        (NpyArrayNeighborhoodIterObject*)_iter;
    NpyArrayIterObject *p = niter->_internal_iter;

    for(i = 0; i < niter->nd; ++i) {
        _INF_SET_PTR_CIRCULAR(i)
    }
    return p->translate(p, _coordinates);
}

#undef _INF_SET_PTR_CIRCULAR

/*
 * fill and x->ao should have equivalent types
 */
/*
 * A Neighborhood Iterator object.
 */
NpyArrayNeighborhoodIterObject*
NpyArray_NeighborhoodIterNew(NpyArrayIterObject *x, npy_intp *bounds,
                             int mode, void* fill, npy_free_func fillfree)
{
    int i;
    NpyArrayNeighborhoodIterObject *ret = NULL;

    ret = NpyArray_malloc(sizeof(*ret));
    if (ret == NULL) {
        goto fail;
    }
    _NpyObject_Init((_NpyObject *)ret, &NpyArrayNeighborhoodIter_Type);
    ret->magic_number = NPY_VALID_MAGIC;

    array_iter_base_init((NpyArrayIterObject *)ret, x->ao);
    _Npy_INCREF(x);
    ret->_internal_iter = x;

    ret->nd = x->ao->nd;

    for (i = 0; i < ret->nd; ++i) {
        ret->dimensions[i] = x->ao->dimensions[i];
    }

    /* Compute the neighborhood size and copy the shape */
    ret->size = 1;
    for (i = 0; i < ret->nd; ++i) {
        ret->bounds[i][0] = bounds[2 * i];
        ret->bounds[i][1] = bounds[2 * i + 1];
        ret->size *= (ret->bounds[i][1] - ret->bounds[i][0]) + 1;

        /* limits keep track of valid ranges for the neighborhood: if a bound
         * of the neighborhood is outside the array, then limits is the same as
         * boundaries. On the contrary, if a bound is strictly inside the
         * array, then limits correspond to the array range. For example, for
         * an array [1, 2, 3], if bounds are [-1, 3], limits will be [-1, 3],
         * but if bounds are [1, 2], then limits will be [0, 2].
         *
         * This is used by neighborhood iterators stacked on top of this one */
        ret->limits[i][0] = ret->bounds[i][0] < 0 ? ret->bounds[i][0] : 0;
        ret->limits[i][1] = ret->bounds[i][1] >= ret->dimensions[i] - 1 ?
        ret->bounds[i][1] :
        ret->dimensions[i] - 1;
        ret->limits_sizes[i] = (ret->limits[i][1] - ret->limits[i][0]) + 1;
    }

    ret->constant = fill;
    ret->constant_free = fillfree;
    ret->mode = mode;

    switch (mode) {
        case NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING:
            ret->translate = &get_ptr_constant;
            break;
        case NPY_NEIGHBORHOOD_ITER_MIRROR_PADDING:
            ret->translate = &get_ptr_mirror;
            break;
        case NPY_NEIGHBORHOOD_ITER_CIRCULAR_PADDING:
            ret->translate = &get_ptr_circular;
            break;
        case NPY_NEIGHBORHOOD_ITER_ZERO_PADDING:
        case NPY_NEIGHBORHOOD_ITER_ONE_PADDING:
        default:
            NpyErr_SetString(NpyExc_ValueError, "Unsupported padding mode");
            goto fail;
    }

    /*
     * XXX: we force x iterator to be non contiguous because we need
     * coordinates... Modifying the iterator here is not great
     */
    x->contiguous = 0;

    NpyArrayNeighborhoodIter_Reset(ret);
    if (NPY_FALSE == NpyInterface_NeighborhoodIterNewWrapper(ret, &ret->nob_interface)) {
        if (fill && fillfree) {
            (*fillfree)(fill);
        }
        Npy_INTERFACE(ret) = NULL;
        _Npy_DECREF(ret);
        return NULL;
    }

    return ret;

 fail:
    if (fill && fillfree) {
        (*fillfree)(fill);
    }
    if (ret) {
        _Npy_DECREF(ret->_internal_iter);
        /* TODO: Free ret here once we add a level of indirection */
    }
    return NULL;
}

static void neighiter_dealloc(NpyArrayNeighborhoodIterObject* iter)
{
    assert(0 == iter->nob_refcnt);

    _Npy_DECREF(iter->_internal_iter);

    if (iter->constant && iter->constant_free) {
        (*iter->constant_free)(iter->constant);
    }

    array_iter_base_dealloc((NpyArrayIterObject*)iter);
    NpyArray_free(iter);
}

_NpyTypeObject NpyArrayNeighborhoodIter_Type = {
    (npy_destructor)neighiter_dealloc,
};
