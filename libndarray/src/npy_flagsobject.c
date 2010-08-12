/*
 *  npy_flagsobject.c -
 *
 */

#include "npy_config.h"
#include "numpy_api.h"
#include "npy_arrayobject.h"



/*
 * Check whether the given array is stored contiguously
 * (row-wise) in memory.
 *
 * 0-strided arrays are not contiguous (even if dimension == 1)
 */
static int
_IsContiguous(NpyArray *ap)
{
    npy_intp sd;
    npy_intp dim;
    int i;

    if (ap->nd == 0) {
        return 1;
    }
    sd = ap->descr->elsize;
    if (ap->nd == 1) {
        return ap->dimensions[0] == 1 || sd == ap->strides[0];
    }
    for (i = ap->nd - 1; i >= 0; --i) {
        dim = ap->dimensions[i];
        /* contiguous by definition */
        if (dim == 0) {
            return 1;
        }
        if (ap->strides[i] != sd) {
            return 0;
        }
        sd *= dim;
    }
    return 1;
}


/* 0-strided arrays are not contiguous (even if dimension == 1) */
static int
_IsFortranContiguous(NpyArray *ap)
{
    npy_intp sd;
    npy_intp dim;
    int i;

    if (ap->nd == 0) {
        return 1;
    }
    sd = ap->descr->elsize;
    if (ap->nd == 1) {
        return ap->dimensions[0] == 1 || sd == ap->strides[0];
    }
    for (i = 0; i < ap->nd; ++i) {
        dim = ap->dimensions[i];
        /* fortran contiguous by definition */
        if (dim == 0) {
            return 1;
        }
        if (ap->strides[i] != sd) {
            return 0;
        }
        sd *= dim;
    }
    return 1;
}




/*NUMPY_API
 * Update Several Flags at once.
 */
void
NpyArray_UpdateFlags(NpyArray *ret, int flagmask)
{
    if (flagmask & NPY_FORTRAN) {
        if (_IsFortranContiguous(ret)) {
            ret->flags |= NPY_FORTRAN;
            if (ret->nd > 1) {
                ret->flags &= ~NPY_CONTIGUOUS;
            }
        }
        else {
            ret->flags &= ~NPY_FORTRAN;
        }
    }
    if (flagmask & NPY_CONTIGUOUS) {
        if (_IsContiguous(ret)) {
            ret->flags |= NPY_CONTIGUOUS;
            if (ret->nd > 1) {
                ret->flags &= ~NPY_FORTRAN;
            }
        }
        else {
            ret->flags &= ~NPY_CONTIGUOUS;
        }
    }
    if (flagmask & NPY_ALIGNED) {
        if (Npy_IsAligned(ret)) {
            ret->flags |= NPY_ALIGNED;
        }
        else {
            ret->flags &= ~NPY_ALIGNED;
        }
    }
    /*
     * This is not checked by default WRITEABLE is not
     * part of UPDATE_ALL
     */
    if (flagmask & NPY_WRITEABLE) {
        if (Npy_IsWriteable(ret)) {
            ret->flags |= NPY_WRITEABLE;
        }
        else {
            ret->flags &= ~NPY_WRITEABLE;
        }
    }
    return;
}
