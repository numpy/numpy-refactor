/*
 *  npy_ctors.c - 
 *  
 * */

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"




NpyArray *NpyArray_CheckAxis(NpyArray *arr, int *axis, int flags)
{
    NpyArray *temp1, *temp2;
    int n = arr->nd;
    
    if (*axis == NPY_MAXDIMS || n == 0) {
        if (n != 1) {
            temp1 = NpyArray_Ravel(arr,0);
            if (temp1 == NULL) {
                *axis = 0;
                return NULL;
            }
            if (*axis == NPY_MAXDIMS) {
                *axis = NpyArray_NDIM(temp1)-1;
            }
        }
        else {
            temp1 = arr;
            Npy_INCREF(temp1);
            *axis = 0;
        }
        if (!flags && *axis == 0) {
            return temp1;
        }
    }
    else {
        temp1 = arr;
        Npy_INCREF(temp1);
    }
    if (flags) {
        temp2 = NpyArray_CheckFromAny(temp1, NULL,
                                     0, 0, flags, NULL);
        Npy_DECREF(temp1);
        if (temp2 == NULL) {
            return NULL;
        }
    }
    else {
        temp2 = temp1;
    }
    n = NpyArray_NDIM(temp2);
    if (*axis < 0) {
        *axis += n;
    }
    if ((*axis < 0) || (*axis >= n)) {
        NpyErr_Format(NpyExc_ValueError,
                     "axis(=%d) out of bounds", *axis);
        Npy_DECREF(temp2);
        return NULL;
    }
    return temp2;
}



