#define _MULTIARRAYMODULE
#include <numpy/npy_iterators.h>
#include "npy_config.h"
/* TODO: Get rid of this include once we've split PyArrayObject. */
#include <numpy/ndarraytypes.h>
#include <numpy/numpy_api.h>

/* XXX: We should be getting this from an include. */
#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif


static void
arraymapiter_dealloc(NpyArrayMapIterObject *mit);


_NpyTypeObject NpyArrayMapIter_Type = {
    (npy_destructor)arraymapiter_dealloc,
};



NpyArrayMapIterObject *
NpyArray_MapIterNew()
{
    NpyArrayMapIterObject *mit;
    int i;
    
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
    mit->indexobj = NULL;
    
    if (NPY_FALSE == NpyInterface_MapIterNewWrapper(mit, &mit->nob_interface)) {
        Npy_INTERFACE(mit) = NULL;
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
    NpyArray_free(mit);
}



/* Reset the map iterator to the beginning */
NPY_NO_EXPORT void
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
NPY_NO_EXPORT void
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
                copyswap(coord+j,it->dataptr, !PyArray_ISNOTSWAPPED(it->ao),
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
                     !PyArray_ISNOTSWAPPED(it->ao),
                     it->ao);
        }
        NpyArray_ITER_GOTO(mit->ait, coord);
        mit->dataptr = mit->ait->dataptr;
    }
    return;
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
    Npy_INCREF(NpyArray_DESCR(self));
    r = NpyArray_NewFromDescr(PyArray_DESCR(self),
                              PyArray_NDIM(self)-1,
                              PyArray_DIMS(self)+1,
                              PyArray_STRIDES(self)+1, item,
                              PyArray_FLAGS(self),
                              NPY_FALSE, NULL, self);
    if (r == NULL) {
        return NULL;
    }
    NpyArray_BASE_ARRAY(r) = self;
    Npy_INCREF(self); 
    assert(r->base_obj == NULL);
    NpyArray_UpdateFlags(r, NPY_CONTIGUOUS | NPY_FORTRAN);
    return r;
}
