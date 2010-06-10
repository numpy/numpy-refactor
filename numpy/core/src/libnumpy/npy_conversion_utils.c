/*
 *  npy_conversion_util.c - 
 *  
 */

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"




/*NUMPY_API
 * Typestr converter
 */
int 
NpyArray_TypestrConvert(int itemsize, int gentype)
{
    int newtype = gentype;
    
    if (gentype == NpyArray_GENBOOLLTR) {
        if (itemsize == 1) {
            newtype = NpyArray_BOOL;
        }
        else {
            newtype = NpyArray_NOTYPE;
        }
    }
    else if (gentype == NpyArray_SIGNEDLTR) {
        switch(itemsize) {
            case 1:
                newtype = NpyArray_INT8;
                break;
            case 2:
                newtype = NpyArray_INT16;
                break;
            case 4:
                newtype = NpyArray_INT32;
                break;
            case 8:
                newtype = NpyArray_INT64;
                break;
#ifdef PyArray_INT128
            case 16:
                newtype = NpyArray_INT128;
                break;
#endif
            default:
                newtype = NpyArray_NOTYPE;
        }
    }
    else if (gentype == NpyArray_UNSIGNEDLTR) {
        switch(itemsize) {
            case 1:
                newtype = NpyArray_UINT8;
                break;
            case 2:
                newtype = NpyArray_UINT16;
                break;
            case 4:
                newtype = NpyArray_UINT32;
                break;
            case 8:
                newtype = NpyArray_UINT64;
                break;
#ifdef PyArray_INT128
            case 16:
                newtype = NpyArray_UINT128;
                break;
#endif
            default:
                newtype = NpyArray_NOTYPE;
                break;
        }
    }
    else if (gentype == NpyArray_FLOATINGLTR) {
        switch(itemsize) {
            case 4:
                newtype = NpyArray_FLOAT32;
                break;
            case 8:
                newtype = NpyArray_FLOAT64;
                break;
#ifdef PyArray_FLOAT80
            case 10:
                newtype = NpyArray_FLOAT80;
                break;
#endif
#ifdef PyArray_FLOAT96
            case 12:
                newtype = NpyArray_FLOAT96;
                break;
#endif
#ifdef PyArray_FLOAT128
            case 16:
                newtype = NpyArray_FLOAT128;
                break;
#endif
            default:
                newtype = NpyArray_NOTYPE;
        }
    }
    else if (gentype == NpyArray_COMPLEXLTR) {
        switch(itemsize) {
            case 8:
                newtype = NpyArray_COMPLEX64;
                break;
            case 16:
                newtype = NpyArray_COMPLEX128;
                break;
#ifdef PyArray_FLOAT80
            case 20:
                newtype = NpyArray_COMPLEX160;
                break;
#endif
#ifdef PyArray_FLOAT96
            case 24:
                newtype = NpyArray_COMPLEX192;
                break;
#endif
#ifdef PyArray_FLOAT128
            case 32:
                newtype = NpyArray_COMPLEX256;
                break;
#endif
            default:
                newtype = NpyArray_NOTYPE;
        }
    }
    return newtype;
}
