#ifndef _NUMPY_API_H_
#define _NUMPY_API_H_

#include "numpy/arrayobject.h"

typedef PyObject NpyObject;
typedef PyArrayObject NpyArray;
typedef PyArray_Descr NpyArray_Descr;

typedef PyArray_Dims NpyArray_Dims;

typedef PyArray_CopySwapFunc NpyArray_CopySwapFunc;
typedef PyArray_ArrFuncs NpyArray_ArrFuncs;
typedef PyArray_VectorUnaryFunc NpyArray_VectorUnaryFunc;

#define NpyArray_SIZE(a) PyArray_SIZE(a)
#define NpyArray_ITEMSIZE(a) PyArray_ITEMSIZE(a)
#define NpyArray_NDIM(a) PyArray_NDIM(a)
#define NpyArray_DIM(a, i) PyArray_DIM(a, i)
#define NpyArray_STRIDES(a) PyArray_STRIDES(a)
#define NpyArray_DESCR(a) PyArray_DESCR(a)
#define NpyArray_FLAGS(a) PyArray_FLAGS(a)
#define NpyArray_BASE(a) PyArray_BASE(a)

#define NpyArray_CHKFLAGS(a, flags) PyArray_CHKFLAGS(a, flags)
#define NpyArray_ISFORTRAN(a) PyArray_ISFORTRAN(a)
#define NpyArray_ISCONTIGUOUS(a) PyArray_ISCONTIGUOUS(a)
#define NpyArray_ISONESEGMENT(a) PyArray_ISONESEGMENT(a)
#define NpyArray_ISWRITEABLE(a) PyArray_ISWRITEABLE(a)

#define NpyArray_NOTYPE PyArray_NOTYPE
#define NpyArray_NTYPES PyArray_NTYPES
#define NpyArray_NSORTS PyArray_NSORTS
#define NpyArray_USERDEF PyArray_USERDEF
#define NpyTypeNum_ISUSERDEF(a) PyTypeNum_ISUSERDEF(a)

#define NpyArray_NOSCALAR PyArray_NOSCALAR
#define NpyArray_NSCALARKINDS PyArray_NSCALARKINDS



/*
 * API functions.
 */
npy_intp NpyArray_Size(NpyArray *op);
int NpyArray_CompareUCS4(npy_ucs4 *s1, npy_ucs4 *s2, size_t len);
int NpyArray_CompareString(char *s1, char *s2, size_t len);
int NpyArray_ElementStrides(NpyArray *arr);
npy_bool NpyArray_CheckStrides(int elsize, int nd, npy_intp numbytes,
                               npy_intp offset,
                               npy_intp *dims, npy_intp *newstrides);

NpyArray* NpyArray_Newshape(NpyArray* self, NpyArray_Dims *newdims,
                            NPY_ORDER fortran);
NpyArray* NpyArray_Squeeze(NpyArray *self);
NpyArray* NpyArray_SwapAxes(NpyArray *ap, int a1, int a2);
NpyArray* NpyArray_Transpose(NpyArray *ap, NpyArray_Dims *permute);
NpyArray* NpyArray_Ravel(NpyArray *a, NPY_ORDER fortran);
NpyArray* NpyArray_Flatten(NpyArray *a, NPY_ORDER order);

void NpyArray_InitArrFuncs(NpyArray_ArrFuncs *f);
int NpyArray_RegisterDataType(NpyArray_Descr *descr);
int NpyArray_RegisterCastFunc(NpyArray_Descr *descr, int totype,
                              NpyArray_VectorUnaryFunc *castfunc);
int NpyArray_RegisterCanCast(NpyArray_Descr *descr, int totype,
                             NPY_SCALARKIND scalar);
int NpyArray_TypeNumFromName(char *str);

/*
 * Reference counting.
 */

#define Npy_INCREF(a) Py_INCREF(a)
#define Npy_DECREF(a) Py_DECREF(a)
#define NpyArray_REFCOUNT(a) PyArray_REFCOUNT(a)

/*
 * Memory
 */
#define NpyDataMem_RENEW(p, sz) PyDataMem_RENEW(p, sz)

#define NpyDimMem_RENEW(p, sz) PyDimMem_RENEW(p, sz)

/*
 * Error handling.
 */
#define NpyErr_SetString(exc, str) PyErr_SetString(exc, str)
#define NpyErr_NoMemory() PyErr_NoMemory()
#define NpyExc_ValueError PyExc_ValueError
#define NpyExc_MemoryError PyExc_MemoryError
#define NpyExc_TypeError PyExc_TypeError



/*
 * TMP
 */
#define NpyArray_MultiplyList(a, b) PyArray_MultiplyList(a, b)
#define npy_userdescrs userdescrs
#define NpyArray_NewFromDescr(a, b, c, d, e, f, g, h) \
    ((NpyArray*) PyArray_NewFromDescr(a, b, c, d, e, f, g, h))
#define NpyArray_View(a, b, c) ((NpyArray*) PyArray_View(a,b,c))
#define NpyArray_NewCopy(a, order) ((NpyArray*) PyArray_NewCopy(a, order))
#define NpyArray_UpdateFlags(a, flags) PyArray_UpdateFlags(a, flags)

extern int _flat_copyinto(PyObject *dst, PyObject *src, NPY_ORDER order);

#endif
