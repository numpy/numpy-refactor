#include <stdlib.h>
#include <memory.h>
#include <stdarg.h>
#include "npy_config.h"
#include "npy_api.h"
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


static void
array_iter_base_dealloc(NpyArrayIterObject *it)
{
    Npy_INTERFACE(it) = NULL;
    Npy_XDECREF(it->ao);
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
    NpyObject_Init((_NpyObject *)it, &NpyArrayIter_Type);
    if (it == NULL) {
        return NULL;
    }
    it->magic_number = NPY_VALID_MAGIC;

    array_iter_base_init(it, ao);
    if (NPY_FALSE == NpyInterface_IterNewWrapper(it, &it->nob_interface)) {
        Npy_INTERFACE(it) = NULL;
        Npy_DECREF(it);
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
    NpyObject_Init((_NpyObject *)it, &NpyArrayIter_Type);
    it->magic_number = NPY_VALID_MAGIC;
    if (NPY_FALSE == NpyInterface_IterNewWrapper(it, &it->nob_interface)) {
        Npy_INTERFACE(it) = NULL;
        Npy_DECREF(it);
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

static NpyArray *
NpyArray_IterSubscriptBool(NpyArrayIterObject *self, npy_bool index)
{
    NpyArray *result;
    int swap;

    NpyArray_ITER_RESET(self);

    if (index) {
        /* Returns a 0-d array with the value. */
        Npy_INCREF(self->ao->descr);
        result = NpyArray_Alloc(self->ao->descr, 0, NULL,
                                NPY_FALSE, Npy_INTERFACE(self->ao));
        if (result == NULL) {
            return NULL;
        }

        swap = (NpyArray_ISNOTSWAPPED(self->ao) !=
                NpyArray_ISNOTSWAPPED(result));
        result->descr->f->copyswap(result->data, self->dataptr, swap,
                                   self->ao);
        return result;
    } else {
        /* Make an empty array. */
        npy_intp ii = 0;
        Npy_INCREF(self->ao->descr);
        result = NpyArray_Alloc(self->ao->descr, 7, &ii,
                                NPY_FALSE, Npy_INTERFACE(self->ao));
        return result;
    }
}

static NpyArray *
NpyArray_IterSubscriptIntp(NpyArrayIterObject *self, npy_intp index)
{
    NpyArray *result;
    int swap;

    Npy_INCREF(self->ao->descr);
    result = NpyArray_Alloc(self->ao->descr, 0, NULL,
                            NPY_FALSE, Npy_INTERFACE(self->ao));
    if (result == NULL) {
        return NULL;
    }

    swap = (NpyArray_ISNOTSWAPPED(self->ao) != NpyArray_ISNOTSWAPPED(result));
    NpyArray_ITER_RESET(self);
    NpyArray_ITER_GOTO1D(self, index);
    result->descr->f->copyswap(result->data, self->dataptr, swap,
                               self->ao);
    NpyArray_ITER_RESET(self);
    return result;
}

static int
NpyArray_IterSubscriptAssignIntp(NpyArrayIterObject *self, npy_intp index,
                                 NpyArray *value)
{
    NpyArray* converted_value;
    int swap;

    if (NpyArray_SIZE(value) == 0) {
        NpyErr_SetString(NpyExc_ValueError,
                         "Error setting single item of array");
        return -1;
    }

    Npy_INCREF(self->ao->descr);
    converted_value = NpyArray_FromArray(value, self->ao->descr, 0);
    if (converted_value == NULL) {
        return -1;
    }

    swap = (NpyArray_ISNOTSWAPPED(self->ao) !=
            NpyArray_ISNOTSWAPPED(converted_value));

    NpyArray_ITER_RESET(self);
    NpyArray_ITER_GOTO1D(self, index);
    self->ao->descr->f->copyswap(self->dataptr, converted_value->data,
                                 swap, self->ao);
    NpyArray_ITER_RESET(self);

    Npy_DECREF(converted_value);
    return 0;
}

static NpyArray *
NpyArray_IterSubscriptSlice(NpyArrayIterObject *self, NpyIndexSlice *slice)
{
    NpyArray *result;
    npy_intp steps, start, step_size;
    int elsize, swap;
    char *dptr;
    NpyArray_CopySwapFunc *copyswap;

    /* Build the result. */
    steps = NpyArray_SliceSteps(slice);

    Npy_INCREF(self->ao->descr);
    result = NpyArray_Alloc(self->ao->descr, 1, &steps,
                            NPY_FALSE, Npy_INTERFACE(self->ao));
    if (result == NULL) {
        return result;
    }

    /* Copy in the data. */
    copyswap = result->descr->f->copyswap;
    start = slice->start;
    step_size = slice->step;
    elsize = result->descr->elsize;
    swap = (NpyArray_ISNOTSWAPPED(self->ao) != NpyArray_ISNOTSWAPPED(result));
    dptr = result->data;

    NpyArray_ITER_RESET(self);
    while (steps--) {
        NpyArray_ITER_GOTO1D(self, start);
        copyswap(dptr, self->dataptr, swap, result);
        dptr += elsize;
        start += step_size;
    }
    NpyArray_ITER_RESET(self);

    return result;
}

static int
NpyArray_IterSubscriptAssignSlice(NpyArrayIterObject *self,
                                  NpyIndexSlice *slice,
                                  NpyArray *value)
{
    NpyArray *converted_value;
    npy_intp steps, start, step_size;
    int swap;
    NpyArray_CopySwapFunc *copyswap;
    NpyArrayIterObject *value_iter = NULL;

    Npy_INCREF(self->ao->descr);
    converted_value = NpyArray_FromArray(value, self->ao->descr, 0);
    if (converted_value == NULL) {
        return -1;
    }

    /* Copy in the data. */
    value_iter = NpyArray_IterNew(converted_value);
    if (value_iter == NULL) {
        Npy_DECREF(converted_value);
        return -1;
    }

    if (value_iter->size > 0) {
        steps = NpyArray_SliceSteps(slice);
        copyswap = self->ao->descr->f->copyswap;
        start = slice->start;
        step_size = slice->step;
        swap = (NpyArray_ISNOTSWAPPED(self->ao) !=
                NpyArray_ISNOTSWAPPED(converted_value));

        NpyArray_ITER_RESET(self);
        while (steps--) {
            NpyArray_ITER_GOTO1D(self, start);
            copyswap(self->dataptr, value_iter->dataptr, swap, self->ao);
            NpyArray_ITER_NEXT(value_iter);
            if (!NpyArray_ITER_NOTDONE(value_iter)) {
                NpyArray_ITER_RESET(value_iter);
            }
            start += step_size;
        }
        NpyArray_ITER_RESET(self);
    }

    Npy_DECREF(value_iter);
    Npy_DECREF(converted_value);

    return 0;
}

static NpyArray *
NpyArray_IterSubscriptBoolArray(NpyArrayIterObject *self, NpyArray *index)
{
    NpyArray *result;
    npy_intp bool_size, i;
    npy_intp result_size;
    npy_intp stride;
    npy_bool* dptr;
    char* optr;
    int elsize;
    NpyArray_CopySwapFunc *copyswap;
    int swap;

    if (index->nd != 1) {
        NpyErr_SetString(NpyExc_ValueError,
                         "boolean index array should have 1 dimension");
        return NULL;
    }

    bool_size = index->dimensions[0];
    if (bool_size > self->size) {
        NpyErr_SetString(NpyExc_ValueError,
                        "too many boolean indices");
        return NULL;
    }

    /* Get the size of the result by counting the Trues in the index. */
    stride = index->strides[0];
    dptr = (npy_bool *)index->data;
    assert(index->descr->elsize == 1);

    i = bool_size;
    result_size = 0;
    while (i--) {
        if (*dptr) {
            ++result_size;
        }
        dptr += stride;
    }

    /* Build the result. */
    Npy_INCREF(self->ao->descr);
    result = NpyArray_Alloc(self->ao->descr, 1, &result_size,
                            NPY_FALSE, Npy_INTERFACE(self->ao));
    if (result == NULL) {
        return NULL;
    }

    /* Copy in the data. */
    copyswap = result->descr->f->copyswap;
    swap = (NpyArray_ISNOTSWAPPED(self->ao) != NpyArray_ISNOTSWAPPED(result));
    elsize = result->descr->elsize;
    optr = result->data;
    dptr = (npy_bool *)index->data;
    NpyArray_ITER_RESET(self);
    i = bool_size;
    while (i--) {
        if (*dptr) {
            copyswap(optr, self->dataptr, swap, result);
            optr += elsize;
        }
        dptr += stride;
        NpyArray_ITER_NEXT(self);
    }
    assert(optr == result->data + result_size*elsize);
    NpyArray_ITER_RESET(self);

    return result;
}

static int
NpyArray_IterSubscriptAssignBoolArray(NpyArrayIterObject *self,
                                      NpyArray *index,
                                      NpyArray *value)
{
    NpyArray *converted_value;
    npy_intp bool_size, i;
    npy_intp stride;
    npy_bool* dptr;
    NpyArray_CopySwapFunc *copyswap;
    NpyArrayIterObject *value_iter;
    int swap;

    if (index->nd != 1) {
        NpyErr_SetString(NpyExc_ValueError,
                         "boolean index array should have 1 dimension");
        return -1;
    }

    bool_size = index->dimensions[0];
    if (bool_size > self->size) {
        NpyErr_SetString(NpyExc_ValueError,
                        "too many boolean indices");
        return -1;
    }

    Npy_INCREF(self->ao->descr);
    converted_value = NpyArray_FromArray(value, self->ao->descr, 0);
    if (converted_value == NULL) {
        return -1;
    }

    value_iter = NpyArray_IterNew(converted_value);
    if (value_iter == NULL) {
        Npy_DECREF(converted_value);
        return -1;
    }

    if (value_iter->size > 0) {
        /* Copy in the data. */
        stride = index->strides[0];
        dptr = (npy_bool *)index->data;
        assert(index->descr->elsize == 1);
        copyswap = self->ao->descr->f->copyswap;
        swap = (NpyArray_ISNOTSWAPPED(self->ao) !=
                NpyArray_ISNOTSWAPPED(converted_value));

        NpyArray_ITER_RESET(self);
        i = bool_size;
        while (i--) {
            if (*dptr) {
                copyswap(self->dataptr, value_iter->dataptr, swap,
                         self->ao);
                NpyArray_ITER_NEXT(value_iter);
                if (!NpyArray_ITER_NOTDONE(value_iter)) {
                    NpyArray_ITER_RESET(value_iter);
                }
            }
            dptr += stride;
            NpyArray_ITER_NEXT(self);
        }
        NpyArray_ITER_RESET(self);
    }

    Npy_DECREF(value_iter);
    Npy_DECREF(converted_value);

    return 0;
}

static NpyArray *
NpyArray_IterSubscriptIntpArray(NpyArrayIterObject *self,
                                NpyArray *index)
{
    NpyArray *result;
    NpyArray_CopySwapFunc *copyswap;
    NpyArrayIterObject *index_iter;
    npy_intp i, num;
    char* optr;
    int elsize;
    int swap;

    /* Build the result in the same shape as the index. */
    Npy_INCREF(self->ao->descr);
    result = NpyArray_Alloc(self->ao->descr,
                            index->nd, index->dimensions,
                            NPY_FALSE, Npy_INTERFACE(self->ao));
    if (result == NULL) {
        return NULL;
    }

    /* Copy in the data. */
    index_iter = NpyArray_IterNew(index);
    if (index_iter == NULL) {
        Npy_DECREF(result);
        return NULL;
    }
    copyswap = result->descr->f->copyswap;
    i = index_iter->size;
    swap = (NpyArray_ISNOTSWAPPED(self->ao) != NpyArray_ISNOTSWAPPED(result));
    optr = result->data;
    elsize = result->descr->elsize;

    NpyArray_ITER_RESET(self);
    while (i--) {
        num = *((npy_intp *)index_iter->dataptr);
        if (num < 0) {
            num += self->size;
        }
        if (num < 0 || num >= self->size) {
            char msg[1024];
            sprintf(msg, "index %"NPY_INTP_FMT" out of bounds"
                    " 0<=index<%"NPY_INTP_FMT,
                    num, self->size);
            NpyErr_SetString(NpyExc_IndexError, msg);
            Npy_DECREF(index_iter);
            Npy_DECREF(result);
            NpyArray_ITER_RESET(self);
            return NULL;
        }
        NpyArray_ITER_GOTO1D(self, num);
        copyswap(optr, self->dataptr, swap, result);
        optr += elsize;
        NpyArray_ITER_NEXT(index_iter);
    }
    Npy_DECREF(index_iter);
    NpyArray_ITER_RESET(self);

    return result;
}

static int
NpyArray_IterSubscriptAssignIntpArray(NpyArrayIterObject *self,
                                      NpyArray *index, NpyArray *value)
{
    NpyArray *converted_value;
    NpyArray_CopySwapFunc *copyswap;
    NpyArrayIterObject *index_iter, *value_iter;
    npy_intp i, num;
    int swap;

    Npy_INCREF(self->ao->descr);
    converted_value = NpyArray_FromArray(value, self->ao->descr, 0);
    if (converted_value == NULL) {
        return -1;
    }

    index_iter = NpyArray_IterNew(index);
    if (index_iter == NULL) {
        Npy_DECREF(converted_value);
        return -1;
    }

    value_iter = NpyArray_IterNew(converted_value);
    if (value_iter == NULL) {
        Npy_DECREF(index_iter);
        Npy_DECREF(converted_value);
        return -1;
    }
    Npy_DECREF(converted_value);

    if (value_iter->size > 0) {

        copyswap = self->ao->descr->f->copyswap;
        i = index_iter->size;
        swap = (NpyArray_ISNOTSWAPPED(self->ao) !=
                NpyArray_ISNOTSWAPPED(converted_value));

        NpyArray_ITER_RESET(self);
        while (i--) {
            num = *((npy_intp *)index_iter->dataptr);
            if (num < 0) {
                num += self->size;
            }
            if (num < 0 || num >= self->size) {
                char msg[1024];
                sprintf(msg, "index %"NPY_INTP_FMT" out of bounds"
                        " 0<=index<%"NPY_INTP_FMT,
                        num, self->size);
                NpyErr_SetString(NpyExc_IndexError, msg);
                Npy_DECREF(index_iter);
                Npy_DECREF(value_iter);
                NpyArray_ITER_RESET(self);
                return -1;
            }
            NpyArray_ITER_GOTO1D(self, num);
            copyswap(self->dataptr, value_iter->dataptr, swap, self->ao);
            NpyArray_ITER_NEXT(value_iter);
            if (!NpyArray_ITER_NOTDONE(value_iter)) {
                NpyArray_ITER_RESET(value_iter);
            }
            NpyArray_ITER_NEXT(index_iter);
        }
        NpyArray_ITER_RESET(self);
    }
    Npy_DECREF(index_iter);
    Npy_DECREF(value_iter);

    return 0;
}

NpyArray *
NpyArray_IterSubscript(NpyArrayIterObject* self,
                      NpyIndex *indexes, int n)
{
    NpyIndex *index;

    if (n == 0 || (n == 1 && indexes[0].type == NPY_INDEX_ELLIPSIS)) {
        Npy_INCREF(self->ao);
        return self->ao;
    }

    if (n > 1) {
        NpyErr_SetString(NpyExc_IndexError, "unsupported iterator index.");
        return NULL;
    }

    index = &indexes[0];

    switch (index->type) {

    case NPY_INDEX_BOOL:
        return NpyArray_IterSubscriptBool(self, index->index.boolean);
        break;

    case NPY_INDEX_INTP:
        /* Return a 0-d array with the value at the index. */
        return NpyArray_IterSubscriptIntp(self, index->index.intp);
        break;

    case NPY_INDEX_SLICE_NOSTOP:
    case NPY_INDEX_SLICE:
        {
            NpyIndex new_index;

            /* Bind the slice. */
            if (NpyArray_IndexBind(index, 1,
                                   &self->size, 1,
                                   &new_index) < 0) {
                return NULL;
            }
            assert(new_index.type == NPY_INDEX_SLICE);

            return NpyArray_IterSubscriptSlice(self, &new_index.index.slice);
        }
        break;

    case NPY_INDEX_BOOL_ARRAY:
        return NpyArray_IterSubscriptBoolArray(self, index->index.bool_array);
        break;

    case NPY_INDEX_INTP_ARRAY:
        return NpyArray_IterSubscriptIntpArray(self, index->index.intp_array);
        break;

    case NPY_INDEX_NEWAXIS:
    case NPY_INDEX_ELLIPSIS:
        NpyErr_SetString(NpyExc_IndexError,
                         "cannot use Ellipsis or newaxes here");
        return NULL;
        break;

    default:
        NpyErr_SetString(NpyExc_IndexError, "unsupported iterator index");
        return NULL;
        break;
    }
}

int
NpyArray_IterSubscriptAssign(NpyArrayIterObject *self,
                             NpyIndex *indexes, int n,
                             NpyArray *value)
{
    NpyIndex *index;

    if (n > 1) {
        NpyErr_SetString(NpyExc_IndexError, "unsupported iterator index.");
        return -1;
    }

    if (n == 0 || (n == 1 && indexes[0].type == NPY_INDEX_ELLIPSIS)) {
        /* Assign to the whole iter using a slice. */
        NpyIndexSlice slice;

        slice.start = 0;
        slice.stop = self->size;
        slice.step = 1;
        return NpyArray_IterSubscriptAssignSlice(self, &slice, value);
    }

    index = &indexes[0];

    switch (index->type) {

    case NPY_INDEX_BOOL:
        if (index->index.boolean) {
            return NpyArray_IterSubscriptAssignIntp(self, 0, value);
        } else {
            return 0;
        }
        break;

    case NPY_INDEX_INTP:
        return NpyArray_IterSubscriptAssignIntp(self, index->index.intp, value);
        break;

    case NPY_INDEX_SLICE:
    case NPY_INDEX_SLICE_NOSTOP:
        {
            NpyIndex new_index;

            /* Bind the slice. */
            if (NpyArray_IndexBind(index, 1,
                                   &self->size, 1,
                                   &new_index) < 0) {
                return -1;
            }
            assert(new_index.type == NPY_INDEX_SLICE);

            return NpyArray_IterSubscriptAssignSlice(self,
                                    &new_index.index.slice, value);
        }
        break;

    case NPY_INDEX_BOOL_ARRAY:
        return NpyArray_IterSubscriptAssignBoolArray(self,
                                                     index->index.bool_array,
                                                     value);
        break;

    case NPY_INDEX_INTP_ARRAY:
        return NpyArray_IterSubscriptAssignIntpArray(self,
                                                     index->index.intp_array,
                                                     value);
        break;

    case NPY_INDEX_NEWAXIS:
    case NPY_INDEX_ELLIPSIS:
        NpyErr_SetString(NpyExc_IndexError,
                         "cannot use Ellipsis or newaxes here");
        return -1;
        break;

    default:
        NpyErr_SetString(NpyExc_IndexError, "unsupported iterator index");
        return -1;
        break;
    }

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


NpyTypeObject NpyArrayIter_Type = {
    (npy_destructor)arrayiter_dealloc,
};

static void
arraymultiter_dealloc(NpyArrayMultiIterObject *multi)
{
    int i;

    assert(0 == multi->nob_refcnt);

    for (i = 0; i < multi->numiter; i++) {
        Npy_XDECREF(multi->iters[i]);
    }
    multi->magic_number = NPY_INVALID_MAGIC;
    NpyArray_free(multi);
}

NpyTypeObject NpyArrayMultiIter_Type =   {
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
    NpyObject_Init((_NpyObject *)multi, &NpyArrayMultiIter_Type);
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
        Npy_DECREF(multi);
        return NULL;
    }
    NpyArray_MultiIter_RESET(multi);
    if (NPY_FALSE == NpyInterface_MultiIterNewWrapper(multi, &multi->nob_interface)) {
        Npy_INTERFACE(multi) = NULL;
        Npy_DECREF(multi);
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
    NpyObject_Init(ret, &NpyArrayMultiIter_Type);
    ret->magic_number = NPY_VALID_MAGIC;
    if (NPY_FALSE == NpyInterface_MultiIterNewWrapper(ret, &ret->nob_interface)) {
        Npy_INTERFACE(ret) = NULL;
        Npy_DECREF(ret);
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
    NpyObject_Init((_NpyObject *)ret, &NpyArrayNeighborhoodIter_Type);
    ret->magic_number = NPY_VALID_MAGIC;

    array_iter_base_init((NpyArrayIterObject *)ret, x->ao);
    Npy_INCREF(x);
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
        Npy_DECREF(ret);
        return NULL;
    }

    return ret;

 fail:
    if (fill && fillfree) {
        (*fillfree)(fill);
    }
    if (ret) {
        Npy_DECREF(ret->_internal_iter);
        /* TODO: Free ret here once we add a level of indirection */
    }
    return NULL;
}

static void neighiter_dealloc(NpyArrayNeighborhoodIterObject* iter)
{
    assert(0 == iter->nob_refcnt);

    Npy_DECREF(iter->_internal_iter);

    if (iter->constant && iter->constant_free) {
        (*iter->constant_free)(iter->constant);
    }

    array_iter_base_dealloc((NpyArrayIterObject*)iter);
    NpyArray_free(iter);
}

NpyTypeObject NpyArrayNeighborhoodIter_Type = {
    (npy_destructor)neighiter_dealloc,
};
