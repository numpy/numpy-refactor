#ifndef _NUMPY_API_H_
#define _NUMPY_API_H_

#include "numpy/arrayobject.h"

typedef PyArrayObject NpyArray;

#define NpyArray_SIZE(a) PyArray_SIZE(a)
#define NpyArray_ITEMSIZE(a) PyArray_ITEMSIZE(a)
#define NpyArray_NDIM(a) PyArray_NDIM(a)
#define NpyArray_STRIDES(a) PyArray_STRIDES(a)


/*
 * API functions.
 */
npy_intp NpyArray_Size(NpyArray *op);
int NpyArray_CompareUCS4(npy_ucs4 *s1, npy_ucs4 *s2, size_t len);
int NpyArray_CompareString(char *s1, char *s2, size_t len);
int NpyArray_ElementStrides(NpyArray *arr);
npy_bool NpyArray_CheckStrides(int elsize, int nd, npy_intp numbytes, npy_intp offset,
                               npy_intp *dims, npy_intp *newstrides);

/*
 * TMP
 */
#define NpyArray_MultiplyList(a, b) PyArray_MultiplyList(a, b)

#endif
