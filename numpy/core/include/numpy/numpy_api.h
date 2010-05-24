#ifndef _NUMPY_API_H_
#define _NUMPY_API_H_

#include "numpy/arrayobject.h"

typedef PyArrayObject NpyArray;
typedef PyArray_Descr NpyArray_Descr;

typedef PyArray_CopySwapFunc NpyArray_CopySwapFunc;
typedef PyArray_ArrFuncs NpyArray_ArrFuncs;

#define NpyArray_SIZE(a) PyArray_SIZE(a)
#define NpyArray_ITEMSIZE(a) PyArray_ITEMSIZE(a)
#define NpyArray_NDIM(a) PyArray_NDIM(a)
#define NpyArray_STRIDES(a) PyArray_STRIDES(a)
#define NpyArray_DESCR(a) PyArray_DESCR(a)

#define NpyArray_NOTYPE PyArray_NOTYPE
#define NpyArray_NTYPES PyArray_NTYPES
#define NpyArray_NSORTS PyArray_NSORTS
#define NpyArray_USERDEF PyArray_USERDEF



/*
 * API functions.
 */
npy_intp NpyArray_Size(NpyArray *op);
int NpyArray_CompareUCS4(npy_ucs4 *s1, npy_ucs4 *s2, size_t len);
int NpyArray_CompareString(char *s1, char *s2, size_t len);
int NpyArray_ElementStrides(NpyArray *arr);
npy_bool NpyArray_CheckStrides(int elsize, int nd, npy_intp numbytes, npy_intp offset,
                               npy_intp *dims, npy_intp *newstrides);

void NpyArray_InitArrFuncs(NpyArray_ArrFuncs *f);
int NpyArray_RegisterDataType(NpyArray_Descr *descr);

/*
 * Error handling.
 */
#define NpyErr_SetString(exc, str) PyErr_SetString(exc, str)
#define NpyExc_ValueError PyExc_ValueError
#define NpyExc_MemoryError PyExc_MemoryError



/*
 * TMP
 */
#define NpyArray_MultiplyList(a, b) PyArray_MultiplyList(a, b)
#define npy_userdescrs userdescrs

#endif
