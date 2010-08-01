/*
 *  npy_conversion_util.c -
 *
 */

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include "npy_config.h"
#include "numpy/numpy_api.h"




/*NUMPY_API
 * Typestr converter
 */
int
NpyArray_TypestrConvert(int itemsize, int gentype)
{
    int newtype = gentype;

    if (gentype == NPY_GENBOOLLTR) {
        if (itemsize == 1) {
            newtype = NPY_BOOL;
        }
        else {
            newtype = NPY_NOTYPE;
        }
    }
    else if (gentype == NPY_SIGNEDLTR) {
        switch(itemsize) {
            case 1:
                newtype = NPY_INT8;
                break;
            case 2:
                newtype = NPY_INT16;
                break;
            case 4:
                newtype = NPY_INT32;
                break;
            case 8:
                newtype = NPY_INT64;
                break;
#ifdef NPY_INT128
            case 16:
                newtype = NPY_INT128;
                break;
#endif
            default:
                newtype = NPY_NOTYPE;
        }
    }
    else if (gentype == NPY_UNSIGNEDLTR) {
        switch(itemsize) {
            case 1:
                newtype = NPY_UINT8;
                break;
            case 2:
                newtype = NPY_UINT16;
                break;
            case 4:
                newtype = NPY_UINT32;
                break;
            case 8:
                newtype = NPY_UINT64;
                break;
#ifdef NPY_INT128
            case 16:
                newtype = NPY_UINT128;
                break;
#endif
            default:
                newtype = NPY_NOTYPE;
                break;
        }
    }
    else if (gentype == NPY_FLOATINGLTR) {
        switch(itemsize) {
            case 4:
                newtype = NPY_FLOAT32;
                break;
            case 8:
                newtype = NPY_FLOAT64;
                break;
#ifdef NPY_FLOAT80
            case 10:
                newtype = NPY_FLOAT80;
                break;
#endif
#ifdef NPY_FLOAT96
            case 12:
                newtype = NPY_FLOAT96;
                break;
#endif
#ifdef NPY_FLOAT128
            case 16:
                newtype = NPY_FLOAT128;
                break;
#endif
            default:
                newtype = NPY_NOTYPE;
        }
    }
    else if (gentype == NPY_COMPLEXLTR) {
        switch(itemsize) {
            case 8:
                newtype = NPY_COMPLEX64;
                break;
            case 16:
                newtype = NPY_COMPLEX128;
                break;
#ifdef NPY_FLOAT80
            case 20:
                newtype = NPY_COMPLEX160;
                break;
#endif
#ifdef NPY_FLOAT96
            case 24:
                newtype = NPY_COMPLEX192;
                break;
#endif
#ifdef NPY_FLOAT128
            case 32:
                newtype = NPY_COMPLEX256;
                break;
#endif
            default:
                newtype = NPY_NOTYPE;
        }
    }
    return newtype;
}
