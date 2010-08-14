#include <stdlib.h>
#include <memory.h>
#include "npy_config.h"
#include "npy_api.h"
#include "npy_iterators.h"
#include "npy_arrayobject.h"
#include "npy_index.h"
#include "npy_internal.h"
#include "npy_dict.h"



static void
arraymapiter_dealloc(NpyArrayMapIterObject *mit);

int
NpyArray_IndexExpandBool(NpyIndex *indexes, int n, NpyIndex *out_indexes);


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
    mit->nob_interface = NULL;

    /* Expand the boolean arrays in indexes. */
    mit->n_indexes = NpyArray_IndexExpandBool(indexes, n,
                                              mit->indexes);
    if (mit->n_indexes < 0) {
        _Npy_DECREF(mit);
        return NULL;
    }

    /* Make iterators from any intp arrays and intp in the index. */
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
        } else if (index->type == NPY_INDEX_INTP) {
            NpyArray_Descr *indtype;
            NpyArray *indarray;

            /* Make a 0-d array for the index. */
            indtype = NpyArray_DescrFromType(NPY_INTP);
            indarray = NpyArray_Alloc(indtype, 0, NULL, NPY_FALSE, NULL);
            if (indarray == NULL) {
                mit->numiter = j-1;
                _Npy_DECREF(mit);
                return NULL;
            }
            memcpy(indarray->data, &index->index.intp, sizeof(npy_intp));
            mit->iters[j] = NpyArray_IterNew(indarray);
            _Npy_DECREF(indarray);
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

int
NpyArray_MapIterBind(NpyArrayMapIterObject *mit, NpyArray *arr,
                     NpyArray *true_array)
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
        return -1;
    }

    mit->ait = NpyArray_IterNew(arr);
    if (mit->ait == NULL) {
        return -1;
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
    nbound = NpyArray_IndexBind(mit->indexes, mit->n_indexes,
                                arr->dimensions, arr->nd,
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

    /* Check for non-consecutive axes. */
    mit->consec = 1;
    j=mit->iteraxes[0];
    for (i=1; i<mit->numiter; i++) {
        if (mit->iteraxes[i] != j+i) {
            mit->consec = 0;
            break;
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

        _Npy_INCREF(arr->descr);
        view = NpyArray_NewView(arr->descr, n2,
                                dimensions, strides,
                                arr, offset, NPY_TRUE);
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
        mit->bscoord[mit->nd+i] = 0;
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
    return 0;

 fail:
    NpyArray_IndexDealloc(bound_indexes, nbound);
    _Npy_XDECREF(mit->subspace);
    _Npy_XDECREF(mit->ait);
    mit->subspace = NULL;
    mit->ait = NULL;
    return -1;
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
    ret = NpyArray_Alloc(temp->descr, mit->nd, mit->dimensions,
                         NpyArray_ISFORTRAN(temp), Npy_INTERFACE(temp));
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
    r = NpyArray_NewView(NpyArray_DESCR(self),
                         NpyArray_NDIM(self)-1,
                         NpyArray_DIMS(self)+1,
                         NpyArray_STRIDES(self)+1,
                         self, item-self->data,
                         NPY_FALSE);
    return r;
}


NpyArray * NpyArray_IndexSimple(NpyArray* self, NpyIndex* indexes, int n)
{
    NpyIndex new_indexes[NPY_MAXDIMS];
    npy_intp dimensions[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];
    npy_intp offset;
    int n2, n_new;
    NpyArray *result;

    /* Bind the index to the array. */
    n_new = NpyArray_IndexBind(indexes, n,
                               self->dimensions, self->nd,
                               new_indexes);
    if (n_new < 0) {
        return NULL;
    }

    /* Convert to dimensions and strides. */
    n2 = NpyArray_IndexToDimsEtc(self, new_indexes, n_new,
                                 dimensions, strides, &offset, NPY_FALSE);
    NpyArray_IndexDealloc(new_indexes, n_new);
    if (n2 < 0) {
        return NULL;
    }

    /* Make the result. */
    _Npy_INCREF(self->descr);
    result = NpyArray_NewView(self->descr, n2, dimensions, strides,
                              self, offset, NPY_FALSE);

    return result;
}

static NpyArray *
NpyArray_SubscriptField(NpyArray *self, char* field)
{
    NpyArray_DescrField *value = NULL;

    if (self->descr->names) {
        value = NpyDict_Get(self->descr->fields, field);
    }

    if (value != NULL) {
        _Npy_INCREF(value->descr);
        return NpyArray_GetField(self, value->descr, value->offset);
    } else {
        char msg[1024];

        sprintf(msg, "field named %s not found.", field);
        NpyErr_SetString(NpyExc_ValueError, msg);
        return NULL;
    }
}

static int
NpyArray_SubscriptAssignField(NpyArray *self, char* field,
                              NpyArray *v)
{
    NpyArray_DescrField *value = NULL;

    if (self->descr->names) {
        value = NpyDict_Get(self->descr->fields, field);
    }

    if (value != NULL) {
        _Npy_INCREF(value->descr);
        return NpyArray_SetField(self, value->descr, value->offset, v);
    } else {
        char msg[1024];

        sprintf(msg, "field named %s not found.", field);
        NpyErr_SetString(NpyExc_ValueError, msg);
        return -1;
    }
}

static NpyArray *
NpyArray_Subscript0d(NpyArray *self, NpyIndex *indexes, int n)
{
    NpyArray *result;
    npy_intp dimensions[NPY_MAXDIMS];
    npy_bool has_ellipsis = NPY_FALSE;
    int nd_new = 0;
    int i;

    for (i=0; i<n; i++) {
        switch (indexes[i].type) {
        case NPY_INDEX_NEWAXIS:
            dimensions[nd_new++] = 1;
            break;
        case NPY_INDEX_ELLIPSIS:
            if (has_ellipsis) {
                goto err;
            }
            has_ellipsis = NPY_TRUE;
            break;
        default:
            goto err;
            break;
        }
    }

    _Npy_INCREF(self->descr);
    result = NpyArray_NewView(self->descr, nd_new, dimensions, NULL,
                              self, 0, NPY_FALSE);
    return result;

 err:
    NpyErr_SetString(NpyExc_IndexError,
                     "0-d arrays can only use a single ()"
                     " or a list of newaxes (and a single ...)"
                     " as an index");
    return NULL;
}


static NpyArray *
NpyArray_IndexFancy(NpyArray *self, NpyIndex *indexes, int n)
{
    NpyArray *result;

    if (self->nd == 1 && n ==  1) {
        /* Special case for 1-d arrays. */
        NpyArrayIterObject *iter = NpyArray_IterNew(self);
        if (iter == NULL) {
            return NULL;
        }
        result = NpyArray_IterSubscript(iter, indexes, n);
        _Npy_DECREF(iter);
        return result;
    } else {
        NpyArrayMapIterObject *mit = NpyArray_MapIterNew(indexes, n);
        if (mit == NULL) {
            return NULL;
        }
        if (NpyArray_MapIterBind(mit, self, NULL) < 0) {
            _Npy_DECREF(mit);
            return NULL;
        }

        result = NpyArray_GetMap(mit);
        _Npy_DECREF(mit);
        return result;
    }
}

int
NpyArray_IndexFancyAssign(NpyArray *self, NpyIndex *indexes, int n,
                          NpyArray *value)
{
    int result;

    if (self->nd == 1 && n ==  1) {
        /* Special case for 1-d arrays. */
        NpyArrayIterObject *iter = NpyArray_IterNew(self);
        if (iter == NULL) {
            return -1;
        }
        result = NpyArray_IterSubscriptAssign(iter, indexes, n, value);
        _Npy_DECREF(iter);
        return result;
    } else {
        NpyArrayMapIterObject *mit = NpyArray_MapIterNew(indexes, n);
        if (mit == NULL) {
            return -1;
        }
        if (NpyArray_MapIterBind(mit, self, NULL) < 0) {
            _Npy_DECREF(mit);
            return -1;
        }

        result = NpyArray_SetMap(mit, value);
        _Npy_DECREF(mit);
        return result;
    }
}

/*
 * Determine if this is a simple index.
 */
static npy_bool
is_simple(NpyIndex *indexes, int n)
{
    int i;

    for (i=0; i<n; i++) {
        switch (indexes[i].type) {
        case NPY_INDEX_INTP_ARRAY:
        case NPY_INDEX_BOOL_ARRAY:
        case NPY_INDEX_STRING:
            return NPY_FALSE;
            break;
        default:
            break;
        }
    }

    return NPY_TRUE;
}

NpyArray*
NpyArray_Subscript(NpyArray *self, NpyIndex *indexes, int n)
{
    /* Handle cases where we just return this array. */
    if (n == 0 || (n == 1 && indexes[0].type == NPY_INDEX_ELLIPSIS)) {
        _Npy_INCREF(self);
        return self;
    }

    /* Handle returning a single field. */
    if (n == 1 && indexes[0].type == NPY_INDEX_STRING) {
        return NpyArray_SubscriptField(self, indexes[0].index.string);
    }

    /* Handle the simple item case. */
    if (n == 1 && indexes[0].type == NPY_INDEX_INTP) {
        return NpyArray_ArrayItem(self, indexes[0].index.intp);
    }

    /* Treat 0-d indexes as a special case. */
    if (self->nd == 0) {
        return NpyArray_Subscript0d(self, indexes, n);
    }

    /* Either do simple or fancy indexing. */
    if (is_simple(indexes, n)) {
        return NpyArray_IndexSimple(self, indexes, n);
    } else {
        return NpyArray_IndexFancy(self, indexes, n);
    }
}

int
NpyArray_SubscriptAssign(NpyArray *self, NpyIndex *indexes, int n,
                         NpyArray *value)
{
    NpyArray *view;
    int result;

    if (!NpyArray_ISWRITEABLE(self)) {
        NpyErr_SetString(NpyExc_RuntimeError,
                        "array is not writeable");
        return -1;
    }

    /* Handle cases where we have the whole array */
    if (n == 0 || (n == 1 && indexes[0].type == NPY_INDEX_ELLIPSIS)) {
        return NpyArray_MoveInto(self, value);
    }

    /* Handle returning a single field. */
    if (n == 1 && indexes[0].type == NPY_INDEX_STRING) {
        return NpyArray_SubscriptAssignField(self, indexes[0].index.string,
                                             value);
    }

    /* Handle the simple item case. */
    if (n == 1 && indexes[0].type == NPY_INDEX_INTP) {
        view = NpyArray_ArrayItem(self, indexes[0].index.intp);
        if (view == NULL) {
            return -1;
        }
        result = NpyArray_MoveInto(view, value);
        _Npy_DECREF(view);
        return result;
    }

    /* Either do simple or fancy indexing. */
    if (is_simple(indexes, n)) {
        view = NpyArray_IndexSimple(self, indexes, n);
        if (view == NULL) {
            return -1;
        }
        result = NpyArray_MoveInto(view, value);
        _Npy_DECREF(view);
        return result;
    } else {
        return NpyArray_IndexFancyAssign(self, indexes, n, value);
    }
}



