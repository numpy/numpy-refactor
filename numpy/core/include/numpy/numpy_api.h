#ifndef _NUMPY_API_H_
#define _NUMPY_API_H_

#include "numpy/arrayobject.h"

typedef PyArrayObject NpyArray;

#define NpyArray_SIZE(a) PyArray_SIZE(a)


/*
 * API functions.
 */
npy_intp NpyArray_Size(NpyArray *op);

#endif
