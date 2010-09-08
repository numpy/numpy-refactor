/* npy_arrayobject.c */

#include <stdlib.h>
#include "npy_config.h"
#include "npy_api.h"
#include "npy_arrayobject.h"
#include "npy_internal.h"


/* TODO: Make these into interface functions */
extern int PyArray_INCREF(void *);
extern int PyArray_XDECREF(void *);

/*
 * Compute the size of an array (in number of items)
 */
NDARRAY_API npy_intp
NpyArray_Size(NpyArray *op)
{
    return NpyArray_SIZE(op);
}

NDARRAY_API int
NpyArray_CompareUCS4(npy_ucs4 *s1, npy_ucs4 *s2, size_t len)
{
    npy_ucs4 c1, c2;
    while(len-- > 0) {
        c1 = *s1++;
        c2 = *s2++;
        if (c1 != c2) {
            return (c1 < c2) ? -1 : 1;
        }
    }
    return 0;
}

NDARRAY_API int
NpyArray_CompareString(char *s1, char *s2, size_t len)
{
    const unsigned char *c1 = (unsigned char *)s1;
    const unsigned char *c2 = (unsigned char *)s2;
    size_t i;

    for(i = 0; i < len; ++i) {
        if (c1[i] != c2[i]) {
            return (c1[i] > c2[i]) ? 1 : -1;
        }
    }
    return 0;
}

NDARRAY_API int
NpyArray_ElementStrides(NpyArray *arr)
{
    int itemsize = NpyArray_ITEMSIZE(arr);
    int i, N = NpyArray_NDIM(arr);
    npy_intp *strides = NpyArray_STRIDES(arr);

    for (i = 0; i < N; i++) {
        if ((strides[i] % itemsize) != 0) {
            return 0;
        }
    }
    return 1;
}


/*
 * This routine checks to see if newstrides (of length nd) will not
 * ever be able to walk outside of the memory implied numbytes and offset.
 *
 * The available memory is assumed to start at -offset and proceed
 * to numbytes-offset.  The strides are checked to ensure
 * that accessing memory using striding will not try to reach beyond
 * this memory for any of the axes.
 *
 * If numbytes is 0 it will be calculated using the dimensions and
 * element-size.
 *
 * This function checks for walking beyond the beginning and right-end
 * of the buffer and therefore works for any integer stride (positive
 * or negative).
 */
NDARRAY_API npy_bool
NpyArray_CheckStrides(int elsize, int nd, npy_intp numbytes, npy_intp offset,
                      npy_intp *dims, npy_intp *newstrides)
{
    int i;
    npy_intp byte_begin;
    npy_intp begin;
    npy_intp end;

    if (numbytes == 0) {
        numbytes = NpyArray_MultiplyList(dims, nd) * elsize;
    }
    begin = -offset;
    end = numbytes - offset - elsize;
    for (i = 0; i < nd; i++) {
        byte_begin = newstrides[i]*(dims[i] - 1);
        if ((byte_begin < begin) || (byte_begin > end)) {
            return NPY_FALSE;
        }
    }
    return NPY_TRUE;
}


/* Deallocs & destroy's the array object.
 *  Returns whether or not we did an artificial incref
 *  so we can keep track of the total refcount for debugging.
 */
/* TODO: For now caller is expected to call _array_dealloc_buffer_info
         and clear weak refs.  Need to revisit. */
NDARRAY_API int
NpyArray_dealloc(NpyArray *self)
{
    int result = 0;

    assert(NPY_VALID_MAGIC == self->nob_magic_number);
    assert(NULL == self->base_arr ||
           NPY_VALID_MAGIC == self->base_arr->nob_magic_number);

    if (self->base_arr) {
        /*
         * UPDATEIFCOPY means that base points to an
         * array that should be updated with the contents
         * of this array upon destruction.
         * self->base->flags must have been WRITEABLE
         * (checked previously) and it was locked here
         * thus, unlock it.
         */
        if (self->flags & NPY_UPDATEIFCOPY) {
            self->base_arr->flags |= NPY_WRITEABLE;
            Npy_INCREF(self); /* hold on to self in next call */
            if (NpyArray_CopyAnyInto(self->base_arr, self) < 0) {
                /* NpyErr_Print(); */
                NpyErr_Clear();
            }
            /*
             * Don't need to DECREF -- because we are deleting
             *self already...
             */
            result = 1;
        }
        /*
         * In any case base is pointing to something that we need
         * to DECREF -- either a view or a buffer object
         */
        Npy_DECREF(self->base_arr);
        self->base_arr = NULL;
    } else if (NULL != self->base_obj) {
        NpyInterface_DECREF(self->base_obj);
        self->base_obj = NULL;
    }

    if ((self->flags & NPY_OWNDATA) && self->data) {
        /* Free internal references if an Object array */
        if (NpyDataType_FLAGCHK(self->descr, NPY_ITEM_REFCOUNT)) {
            Npy_INCREF(self); /* hold on to self in next call */
            NpyArray_XDECREF(self);
            /*
             * Don't need to DECREF -- because we are deleting
             * self already...
             */
            if (self->nob_refcnt == 1) {
                result = 1;
            }
        }
        NpyDataMem_FREE(self->data);
    }

    NpyDimMem_FREE(self->dimensions);
    Npy_DECREF(self->descr);
    /* Flag that this object is now deallocated. */
    self->nob_magic_number = NPY_INVALID_MAGIC;

    NpyArray_free(self);

    return result;
}


NpyTypeObject NpyArray_Type = {
    (npy_destructor)NpyArray_dealloc,
};
