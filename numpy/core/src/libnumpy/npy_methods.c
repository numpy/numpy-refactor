/*
 *  npy_methods.c -
 *
 */

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include "npy_config.h"
#include "numpy/numpy_api.h"





/* steals typed reference */
/*NUMPY_API
 Get a subset of bytes from each element of the array
 */
NpyArray *
NpyArray_GetField(NpyArray *self, NpyArray_Descr *typed, int offset)
{
    NpyArray *ret = NULL;

    if (offset < 0 || (offset + typed->elsize) > self->descr->elsize) {
        NpyErr_Format(NpyExc_ValueError,
                      "Need 0 <= offset <= %d for requested type "  \
                      "but received offset = %d",
                      self->descr->elsize-typed->elsize, offset);
        _Npy_DECREF(typed);
        return NULL;
    }
    ret = NpyArray_NewFromDescr(typed,
                                self->nd, self->dimensions,
                                self->strides,
                                self->data + offset,
                                self->flags, NPY_FALSE, NULL,
                                Npy_INTERFACE(self));
    if (ret == NULL) {
        return NULL;
    }
    _Npy_INCREF(self);
    ret->base_arr = self;
    assert(NULL == ret->base_arr || NULL == ret->base_obj);

    NpyArray_UpdateFlags(ret, NPY_UPDATE_ALL);
    return ret;
}




/*NUMPY_API
 Set a subset of bytes from each element of the array
 *
 * Steals a reference to dtype.
 */
int
NpyArray_SetField(NpyArray *self, NpyArray_Descr *dtype,
                  int offset, NpyArray *val)
{
    NpyArray *ret = NULL;
    int retval = 0;

    if (offset < 0 || (offset + dtype->elsize) > self->descr->elsize) {
        NpyErr_Format(NpyExc_ValueError,
                      "Need 0 <= offset <= %d for requested type "  \
                      "but received offset = %d",
                      self->descr->elsize-dtype->elsize, offset);
        _Npy_DECREF(dtype);
        return -1;
    }
    ret = NpyArray_NewFromDescr(dtype, self->nd, self->dimensions,
                                self->strides, self->data + offset,
                                self->flags, NPY_FALSE, NULL,
                                Npy_INTERFACE(self));
    if (ret == NULL) {
        return -1;
    }
    _Npy_INCREF(self);
    ret->base_arr = self;
    assert(NULL == ret->base_arr || NULL == ret->base_obj);

    NpyArray_UpdateFlags(ret, NPY_UPDATE_ALL);
    retval = NpyArray_MoveInto(ret, val);
    _Npy_DECREF(ret);
    return retval;
}




/* This doesn't change the descriptor just the actual data...
 */

/*NUMPY_API*/
NpyArray *
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
            _Npy_DECREF(it);
        }

        _Npy_INCREF(self);
        return self;
    }
    else {
        NpyArray *new;
        if ((ret = NpyArray_NewCopy(self,-1)) == NULL) {
            return NULL;
        }
        new = NpyArray_Byteswap(ret, NPY_TRUE);
        _Npy_DECREF(new);
        return ret;
    }
}
