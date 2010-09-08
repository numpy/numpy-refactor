/*
 *  npy_methods.c -
 *
 */

#include "npy_config.h"
#include "npy_api.h"
#include "npy_arrayobject.h"




/* steals typed reference */
/*
 Get a subset of bytes from each element of the array
 */
NDARRAY_API NpyArray *
NpyArray_GetField(NpyArray *self, NpyArray_Descr *typed, int offset)
{
    NpyArray *ret = NULL;
    char msg[1024];

    if (offset < 0 || (offset + typed->elsize) > self->descr->elsize) {
        sprintf(msg,
                "Need 0 <= offset <= %d for requested type "
                "but received offset = %d",
                self->descr->elsize-typed->elsize, offset);
        NpyErr_SetString(NpyExc_ValueError, msg);
        Npy_DECREF(typed);
        return NULL;
    }
    ret = NpyArray_NewView(typed, self->nd, self->dimensions, self->strides,
                           self, offset, NPY_FALSE);
    if (ret == NULL) {
        return NULL;
    }
    return ret;
}




/*
 Set a subset of bytes from each element of the array
 *
 * Steals a reference to dtype.
 */
NDARRAY_API int
NpyArray_SetField(NpyArray *self, NpyArray_Descr *dtype,
                  int offset, NpyArray *val)
{
    NpyArray *ret = NULL;
    int retval = 0;
    char msg[1024];

    if (offset < 0 || (offset + dtype->elsize) > self->descr->elsize) {
        sprintf(msg, "Need 0 <= offset <= %d for requested type "
                     "but received offset = %d",
                      self->descr->elsize-dtype->elsize, offset);
        NpyErr_SetString(NpyExc_ValueError, msg);
        Npy_DECREF(dtype);
        return -1;
    }
    ret = NpyArray_NewView(dtype, self->nd, self->dimensions, self->strides,
                           self, offset, NPY_FALSE);
    if (ret == NULL) {
        return -1;
    }
    retval = NpyArray_MoveInto(ret, val);
    Npy_DECREF(ret);
    return retval;
}




/* This doesn't change the descriptor just the actual data...
 */
NDARRAY_API NpyArray *
NpyArray_Byteswap(NpyArray *self, npy_bool inplace)
{
    NpyArray *ret;
    npy_intp size;
    NpyArray_CopySwapNFunc *copyswapn;
    NpyArrayIterObject *it;

    copyswapn = self->descr->f->copyswapn;
    if (inplace) {
        if (!NpyArray_ISWRITEABLE(self)) {
            NpyErr_SetString(NpyExc_RuntimeError,
                             "Cannot byte-swap in-place on a " \
                             "read-only array");
            return NULL;
        }
        size = NpyArray_SIZE(self);
        if (NpyArray_ISONESEGMENT(self)) {
            copyswapn(self->data, self->descr->elsize, NULL, -1, size, 1, self);
        }
        else { /* Use iterator */
            int axis = -1;
            npy_intp stride;
            it = NpyArray_IterAllButAxis(self, &axis);
            stride = self->strides[axis];
            size = self->dimensions[axis];
            while (it->index < it->size) {
                copyswapn(it->dataptr, stride, NULL, -1, size, 1, self);
                NpyArray_ITER_NEXT(it);
            }
            Npy_DECREF(it);
        }

        Npy_INCREF(self);
        return self;
    }
    else {
        NpyArray *new;
        if ((ret = NpyArray_NewCopy(self,-1)) == NULL) {
            return NULL;
        }
        new = NpyArray_Byteswap(ret, NPY_TRUE);
        Npy_DECREF(new);
        return ret;
    }
}
