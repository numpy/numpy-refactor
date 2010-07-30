/*
 *  npy_getset.c -
 *
 */

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include "npy_config.h"
#include "numpy/numpy_api.h"



/*NUMPY_API
 */
int
NpyArray_SetShape(NpyArray *self, NpyArray_Dims *newdims)
{
    int nd;
    NpyArray *ret;

    ret = NpyArray_Newshape(self, newdims, NPY_CORDER);
    if (ret == NULL) {
        return -1;
    }
    if (NpyArray_DATA(ret) != NpyArray_DATA(self)) {
        _Npy_XDECREF(ret);
        NpyErr_SetString(NpyExc_AttributeError,
                         "incompatible shape for a non-contiguous array");
        return -1;
    }

    /* Free old dimensions and strides */
    NpyDimMem_FREE(NpyArray_DIMS(self));
    nd = NpyArray_NDIM(ret);
    NpyArray_NDIM(self) = nd;
    if (nd > 0) {
        /* create new dimensions and strides */
        NpyArray_DIMS(self) = NpyDimMem_NEW(2 * nd);
        if (NpyArray_DIMS(self) == NULL) {
            _Npy_XDECREF(ret);
            NpyErr_SetString(NpyExc_MemoryError,"");
            return -1;
        }
        NpyArray_STRIDES(self) = NpyArray_DIMS(self) + nd;
        memcpy(NpyArray_DIMS(self), NpyArray_DIMS(ret), nd * sizeof(intp));
        memcpy(NpyArray_STRIDES(self), NpyArray_STRIDES(ret), nd * sizeof(intp));
    }
    else {
        NpyArray_DIMS(self) = NULL;
        NpyArray_STRIDES(self) = NULL;
    }
    _Npy_XDECREF(ret);
    NpyArray_UpdateFlags(self, NPY_CONTIGUOUS | NPY_FORTRAN);
    return 0;
}
