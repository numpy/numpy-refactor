#define _MULTIARRAYMODULE
#include <stdlib.h>
#include <memory.h>
#include "npy_config.h"
#include <numpy/numpy_api.h>
#include <numpy/npy_iterators.h>
#include <numpy/npy_arrayobject.h>
#include <numpy/npy_index.h>
#include "npy_internal.h"




static void
arraymapiter_dealloc(NpyArrayMapIterObject *mit);


_NpyTypeObject NpyArrayMapIter_Type = {
    (npy_destructor)arraymapiter_dealloc,
};



NpyArrayMapIterObject *
NpyArray_MapIterNew(NpyIndex *indexes, int n)
{
    NpyArrayMapIterObject *mit;
    int i, j;

    /* Allocates the Python object wrapper around the map iterator. */
    mit = (NpyArrayMapIterObject *)NpyArray_malloc(sizeof(NpyArrayMapIterObject));
    _NpyObject_Init((_NpyObject *)mit, &NpyArrayMapIter_Type);
    if (mit == NULL) {
        return NULL;
    }
    for (i = 0; i < NPY_MAXDIMS; i++) {
        mit->iters[i] = NULL;
    }
    mit->index = 0;
    mit->ait = NULL;
    mit->subspace = NULL;
    mit->numiter = 0;
    mit->consec = 1;
    mit->n_indexes = 0;

    if (NPY_FALSE == NpyInterface_MapIterNewWrapper(mit, &mit->nob_interface)) {
        Npy_INTERFACE(mit) = NULL;
        _Npy_DECREF(mit);
        return NULL;
    }

    /* Expand the boolean arrays in indexes. */
    mit->n_indexes = NpyArray_IndexExpandBool(indexes, n,
                                              mit->indexes);
    if (mit->n_indexes < 0) {
        _Npy_DECREF(mit);
        return NULL;
    }

    /* Make iterators from any intp arrays in the index. */
    j = 0;
    for (i=0; i<mit->n_indexes; i++) {
        NpyIndex* index = &mit->indexes[i];

        if (index->type == NPY_INDEX_INTP_ARRAY) {
            mit->iters[j] = NpyArray_IterNew(index->index.intp_array);
            if (mit->iters[j] == NULL) {
                mit->numiter = j-1;
                _Npy_DECREF(mit);
                return NULL;
            }
            j++;
        }
    }
    mit->numiter = j;

    /* Broadcast the index iterators. */
    if (NpyArray_Broadcast((NpyArrayMultiIterObject *)mit) < 0) {
        _Npy_DECREF(mit);
        return NULL;
    }

    return mit;
}


static void
arraymapiter_dealloc(NpyArrayMapIterObject *mit)
{
    int i;

    assert(0 == mit->nob_refcnt);

    Npy_INTERFACE(mit) = NULL;
    _Npy_XDECREF(mit->ait);
    _Npy_XDECREF(mit->subspace);
    for (i = 0; i < mit->numiter; i++) {
        _Npy_XDECREF(mit->iters[i]);
    }
    NpyArray_IndexDealloc(mit->indexes, mit->n_indexes);
    NpyArray_free(mit);
}

void
NpyArray_MapIterBind(NpyArrayMapIterObject *mit, NpyArray *arr)
{
    NpyArrayIterObject *it;
    int subnd;
    int i, j, n;
    npy_intp dimsize;
    npy_intp *indptr;
    NpyIndex bound_indexes[NPY_MAXDIMS];
    int nbound = 0;

    subnd = arr->nd - mit->numiter;
    if (subnd < 0) {
        NpyErr_SetString(NpyExc_ValueError,
                        "too many indices for array");
        return;
    }

    mit->ait = NpyArray_IterNew(arr);
    if (mit->ait == NULL) {
        return;
    }
    /* no subspace iteration needed.  Finish up and Return */
    if (subnd == 0) {
        n = arr->nd;
        for (i = 0; i < n; i++) {
            mit->iteraxes[i] = i;
        }
        goto finish;
    }


    /* Bind the indexes to the array. */
    nbound = NpyArray_IndexBind(arr, mit->indexes, mit->n_indexes, 
                                bound_indexes);
    if (nbound < 0) {
        nbound = 0;
        goto fail;
    }

    /* Fill in iteraxes and bscoord from the bound indexes. */
    j = 0;
    for (i=0; i<nbound; i++) {
        NpyIndex *index = &bound_indexes[i];

        switch (index->type) {
        case NPY_INDEX_INTP_ARRAY:
            mit->iteraxes[j++] = i;
            mit->bscoord[i] = 0;
            break;
        case NPY_INDEX_INTP:
            mit->bscoord[i] = index->index.intp;
            break;
        case NPY_INDEX_SLICE:
            mit->bscoord[i] = index->index.slice.start;
            break;
        default:
            mit->bscoord[i] = 0;
        }
    }

    /*
     * Make the subspace iterator.
     */
    {
        npy_intp dimensions[NPY_MAXDIMS];
        npy_intp strides[NPY_MAXDIMS];
        npy_intp offset;
        int n2;
        NpyArray *view;

        /* Convert to dimensions and strides. */
        n2 = NpyArray_IndexToDimsEtc(arr, bound_indexes, nbound,
                                     dimensions, strides, &offset, 
                                     NPY_TRUE);
        if (n2 < 0) {
            goto fail;
        }

        view =  NpyArray_NewFromDescr(arr->descr, n2,
                                      dimensions, strides,
                                      arr->data + offset,
                                      arr->flags, NPY_TRUE,
                                      NULL, Npy_INTERFACE(arr));
        if (view == NULL) {
            goto fail;
        }
        mit->subspace = NpyArray_IterNew(view);
        _Npy_DECREF(view);
        if (mit->subspace == NULL) {
            goto fail;
        }
    }

    /* Expand dimensions of result */
    n = mit->subspace->ao->nd;
    for (i = 0; i < n; i++) {
        mit->dimensions[mit->nd+i] = mit->subspace->ao->dimensions[i];
    }
    mit->nd += n;

    /* Free the indexes. */
    NpyArray_IndexDealloc(bound_indexes, nbound);
    nbound = 0;

 finish:
    /* Here check the indexes (now that we have iteraxes) */
    mit->size = NpyArray_OverflowMultiplyList(mit->dimensions, mit->nd);
    if (mit->size < 0) {
        NpyErr_SetString(NpyExc_ValueError,
                        "dimensions too large in fancy indexing");
        goto fail;
    }
    if (mit->ait->size == 0 && mit->size != 0) {
        NpyErr_SetString(NpyExc_ValueError,
                        "invalid index into a 0-size array");
        goto fail;
    }

    for (i = 0; i < mit->numiter; i++) {
        npy_intp indval;
        it = mit->iters[i];
        NpyArray_ITER_RESET(it);
        dimsize = NpyArray_DIM(arr, mit->iteraxes[i]);
        while (it->index < it->size) {
            indptr = ((npy_intp *)it->dataptr);
            indval = *indptr;
            if (indval < 0) {
                indval += dimsize;
            }
            if (indval < 0 || indval >= dimsize) {
                char msg[1024];
                sprintf(msg,
                        "index (%"NPY_INTP_FMT") out of range "
                        "(0<=index<%"NPY_INTP_FMT") in dimension %d",
                        indval, (dimsize-1), mit->iteraxes[i]);
                NpyErr_SetString(NpyExc_IndexError, msg);
                goto fail;
            }
            NpyArray_ITER_NEXT(it);
        }
        NpyArray_ITER_RESET(it);
    }
    return;

 fail:
    NpyArray_IndexDealloc(bound_indexes, nbound);
    _Npy_XDECREF(mit->subspace);
    _Npy_XDECREF(mit->ait);
    mit->subspace = NULL;
    mit->ait = NULL;
    return;
}



/* Reset the map iterator to the beginning */
void
NpyArray_MapIterReset(NpyArrayMapIterObject *mit)
{
    NpyArrayIterObject *it;
    int i,j;
    npy_intp coord[NPY_MAXDIMS];
    NpyArray_CopySwapFunc *copyswap;

    mit->index = 0;

    copyswap = mit->iters[0]->ao->descr->f->copyswap;

    if (mit->subspace != NULL) {
        memcpy(coord, mit->bscoord, sizeof(npy_intp)*mit->ait->ao->nd);
        NpyArray_ITER_RESET(mit->subspace);
        for (i = 0; i < mit->numiter; i++) {
            it = mit->iters[i];
            NpyArray_ITER_RESET(it);
            j = mit->iteraxes[i];
            copyswap(coord+j,it->dataptr, !NpyArray_ISNOTSWAPPED(it->ao),
                     it->ao);
        }
        NpyArray_ITER_GOTO(mit->ait, coord);
        mit->subspace->dataptr = mit->ait->dataptr;
        mit->dataptr = mit->subspace->dataptr;
    }
    else {
        for (i = 0; i < mit->numiter; i++) {
            it = mit->iters[i];
            if (it->size != 0) {
                NpyArray_ITER_RESET(it);
                copyswap(coord+i,it->dataptr, !NpyArray_ISNOTSWAPPED(it->ao),
                         it->ao);
            }
            else {
                coord[i] = 0;
            }
        }
        NpyArray_ITER_GOTO(mit->ait, coord);
        mit->dataptr = mit->ait->dataptr;
    }
    return;
}

/*
 * This function needs to update the state of the map iterator
 * and point mit->dataptr to the memory-location of the next object
 */
void
NpyArray_MapIterNext(NpyArrayMapIterObject *mit)
{
    NpyArrayIterObject *it;
    int i, j;
    npy_intp coord[NPY_MAXDIMS];
    NpyArray_CopySwapFunc *copyswap;

    mit->index += 1;
    if (mit->index >= mit->size) {
        return;
    }
    copyswap = mit->iters[0]->ao->descr->f->copyswap;
    /* Sub-space iteration */
    if (mit->subspace != NULL) {
        NpyArray_ITER_NEXT(mit->subspace);
        if (mit->subspace->index >= mit->subspace->size) {
            /* reset coord to coordinates of beginning of the subspace */
            memcpy(coord, mit->bscoord, sizeof(npy_intp)*mit->ait->ao->nd);
            NpyArray_ITER_RESET(mit->subspace);
            for (i = 0; i < mit->numiter; i++) {
                it = mit->iters[i];
                NpyArray_ITER_NEXT(it);
                j = mit->iteraxes[i];
                copyswap(coord+j,it->dataptr, !NpyArray_ISNOTSWAPPED(it->ao),
                         it->ao);
            }
            NpyArray_ITER_GOTO(mit->ait, coord);
            mit->subspace->dataptr = mit->ait->dataptr;
        }
        mit->dataptr = mit->subspace->dataptr;
    }
    else {
        for (i = 0; i < mit->numiter; i++) {
            it = mit->iters[i];
            NpyArray_ITER_NEXT(it);
            copyswap(coord+i,it->dataptr,
                     !NpyArray_ISNOTSWAPPED(it->ao),
                     it->ao);
        }
        NpyArray_ITER_GOTO(mit->ait, coord);
        mit->dataptr = mit->ait->dataptr;
    }
    return;
}

static void
_swap_axes(NpyArrayMapIterObject *mit, NpyArray **ret, int getmap)
{
    NpyArray *new;
    int n1, n2, n3, val, bnd;
    int i;
    NpyArray_Dims permute;
    npy_intp d[NPY_MAXDIMS];
    NpyArray *arr;

    permute.ptr = d;
    permute.len = mit->nd;

    /*
     * arr might not have the right number of dimensions
     * and need to be reshaped first by pre-pending ones
     */
    arr = *ret;
    if (arr->nd != mit->nd) {
        for (i = 1; i <= arr->nd; i++) {
            permute.ptr[mit->nd-i] = arr->dimensions[arr->nd-i];
        }
        for (i = 0; i < mit->nd-arr->nd; i++) {
            permute.ptr[i] = 1;
        }
        new = NpyArray_Newshape(arr, &permute, NPY_ANYORDER);
        _Npy_DECREF(arr);
        *ret = new;
        if (new == NULL) {
            return;
        }
    }

    /*
     * Setting and getting need to have different permutations.
     * On the get we are permuting the returned object, but on
     * setting we are permuting the object-to-be-set.
     * The set permutation is the inverse of the get permutation.
     */

    /*
     * For getting the array the tuple for transpose is
     * (n1,...,n1+n2-1,0,...,n1-1,n1+n2,...,n3-1)
     * n1 is the number of dimensions of the broadcast index array
     * n2 is the number of dimensions skipped at the start
     * n3 is the number of dimensions of the result
     */

    /*
     * For setting the array the tuple for transpose is
     * (n2,...,n1+n2-1,0,...,n2-1,n1+n2,...n3-1)
     */
    n1 = mit->iters[0]->nd_m1 + 1;
    n2 = mit->iteraxes[0];
    n3 = mit->nd;

    /* use n1 as the boundary if getting but n2 if setting */
    bnd = getmap ? n1 : n2;
    val = bnd;
    i = 0;
    while (val < n1 + n2) {
        permute.ptr[i++] = val++;
    }
    val = 0;
    while (val < bnd) {
        permute.ptr[i++] = val++;
    }
    val = n1 + n2;
    while (val < n3) {
        permute.ptr[i++] = val++;
    }
    new = NpyArray_Transpose(*ret, &permute);
    _Npy_DECREF(*ret);
    *ret = new;
}

NpyArray *
NpyArray_GetMap(NpyArrayMapIterObject *mit)
{
    NpyArrayIterObject *it;
    NpyArray *ret, *temp;
    int index;
    int swap;
    NpyArray_CopySwapFunc *copyswap;

    /* Unbound map iterator --- Bind should have been called */
    if (mit->ait == NULL) {
        return NULL;
    }

    /* This relies on the map iterator object telling us the shape
       of the new array in nd and dimensions.
    */
    temp = mit->ait->ao;
    _Npy_INCREF(temp->descr);
    ret = NpyArray_NewFromDescr(temp->descr,
                                mit->nd, mit->dimensions,
                                NULL, NULL,
                                NpyArray_ISFORTRAN(temp),
                                NPY_FALSE, NULL, 
                                Npy_INTERFACE(temp));
    if (ret == NULL) {
        return NULL;
    }

    /*
     * Now just iterate through the new array filling it in
     * with the next object from the original array as
     * defined by the mapping iterator
     */

    if ((it = NpyArray_IterNew(ret)) == NULL) {
        _Npy_DECREF(ret);
        return NULL;
    }
    index = it->size;
    swap = (NpyArray_ISNOTSWAPPED(temp) != NpyArray_ISNOTSWAPPED(ret));
    copyswap = ret->descr->f->copyswap;
    NpyArray_MapIterReset(mit);
    while (index--) {
        copyswap(it->dataptr, mit->dataptr, swap, ret);
        NpyArray_MapIterNext(mit);
        NpyArray_ITER_NEXT(it);
    }
    _Npy_DECREF(it);

    /* check for consecutive axes */
    if ((mit->subspace != NULL) && (mit->consec)) {
        if (mit->iteraxes[0] > 0) {  /* then we need to swap */
            _swap_axes(mit, &ret, 1);
        }
    }
    return ret;
}


int
NpyArray_SetMap(NpyArrayMapIterObject *mit, NpyArray *arr)
{
    NpyArrayIterObject *it;
    int index;
    int swap;
    NpyArray_CopySwapFunc *copyswap;

    /* Unbound Map Iterator */
    if (mit->ait == NULL) {
        return -1;
    }
    if ((mit->subspace != NULL) && (mit->consec)) {
        if (mit->iteraxes[0] > 0) {  /* then we need to swap */
            _swap_axes(mit, &arr, 0);
            if (arr == NULL) {
                return -1;
            }
        }
    }

    /* Be sure values array is "broadcastable"
       to shape of mit->dimensions, mit->nd */

    if ((it = NpyArray_BroadcastToShape(arr, mit->dimensions,
                                        mit->nd))==NULL) {
        _Npy_DECREF(arr);
        return -1;
    }

    index = mit->size;
    swap = (NpyArray_ISNOTSWAPPED(mit->ait->ao) !=
            (NpyArray_ISNOTSWAPPED(arr)));
    copyswap = arr->descr->f->copyswap;
    NpyArray_MapIterReset(mit);
    /* Need to decref arrays with objects in them */
    if (NpyDataType_FLAGCHK(arr->descr, NPY_ITEM_HASOBJECT)) {
        while (index--) {
            NpyArray_Item_INCREF(it->dataptr, arr->descr);
            NpyArray_Item_XDECREF(mit->dataptr, arr->descr);
            memmove(mit->dataptr, it->dataptr, NpyArray_ITEMSIZE(arr));
            /* ignored unless VOID array with object's */
            if (swap) {
                copyswap(mit->dataptr, NULL, swap, arr);
            }
            NpyArray_MapIterNext(mit);
            NpyArray_ITER_NEXT(it);
        }
        _Npy_DECREF(arr);
        _Npy_DECREF(it);
        return 0;
    }
    while(index--) {
        memmove(mit->dataptr, it->dataptr, NpyArray_ITEMSIZE(arr));
        if (swap) {
            copyswap(mit->dataptr, NULL, swap, arr);
        }
        NpyArray_MapIterNext(mit);
        NpyArray_ITER_NEXT(it);
    }
    _Npy_DECREF(arr);
    _Npy_DECREF(it);
    return 0;
}


/*
 * Indexes the first dimenstion of the array and returns the
 * item as an array.
 */
NpyArray *
NpyArray_ArrayItem(NpyArray *self, npy_intp i)
{
    char *item;
    NpyArray *r;

    if(NpyArray_NDIM(self) == 0) {
        NpyErr_SetString(NpyExc_IndexError,
                        "0-d arrays can't be indexed");
        return NULL;
    }
    if ((item = NpyArray_Index2Ptr(self, i)) == NULL) {
        return NULL;
    }
    _Npy_INCREF(NpyArray_DESCR(self));
    r = NpyArray_NewFromDescr(NpyArray_DESCR(self),
                              NpyArray_NDIM(self)-1,
                              NpyArray_DIMS(self)+1,
                              NpyArray_STRIDES(self)+1, item,
                              NpyArray_FLAGS(self),
                              NPY_FALSE, NULL, Npy_INTERFACE(self));
    if (r == NULL) {
        return NULL;
    }
    NpyArray_BASE_ARRAY(r) = self;
    _Npy_INCREF(self);
    assert(r->base_obj == NULL);
    NpyArray_UpdateFlags(r, NPY_CONTIGUOUS | NPY_FORTRAN);
    return r;
}


NpyArray * NpyArray_IndexSimple(NpyArray* self, NpyIndex* indexes, int n)
{
    NpyIndex new_indexes[NPY_MAXDIMS];
    npy_intp dimensions[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];
    npy_intp offset;
    int n2;
    NpyArray *result;

    /* Bind the index to the array. */
    n2 = NpyArray_IndexBind(self, indexes, n, new_indexes);
    if (n2 < 0) {
        return NULL;
    }

    /* Convert to dimensions and strides. */
    n2 = NpyArray_IndexToDimsEtc(self, new_indexes, n2,
                                 dimensions, strides, &offset, NPY_FALSE);
    if (n2 < 0) {
        return NULL;
    }

    /* Make the result. */
    _Npy_INCREF(self->descr);
    result = NpyArray_NewFromDescr(self->descr, n2,
                                   dimensions, strides,
                                   self->data + offset,
                                   self->flags, NPY_FALSE,
                                   NULL, Npy_INTERFACE(self));

    if (result == NULL) {
        return NULL;
    }

    /* Set the base_arr on result. */
    result->base_arr = self;
    _Npy_INCREF(self);

    return result;
}
