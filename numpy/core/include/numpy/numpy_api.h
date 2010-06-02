#ifndef _NUMPY_API_H_
#define _NUMPY_API_H_

#include "numpy/arrayobject.h"

typedef PyObject NpyObject;
typedef PyArrayObject NpyArray;
typedef PyArray_Descr NpyArray_Descr;
typedef PyArrayMultiIterObject NpyArrayMultiIterObject;

typedef PyArray_Dims NpyArray_Dims;

typedef PyArray_CopySwapFunc NpyArray_CopySwapFunc;
typedef PyArray_ArrFuncs NpyArray_ArrFuncs;
typedef PyArray_ArgFunc NpyArray_ArgFunc;
typedef PyArray_VectorUnaryFunc NpyArray_VectorUnaryFunc;
typedef PyArray_FastTakeFunc NpyArray_FastTakeFunc;
typedef PyArray_FastPutmaskFunc NpyArray_FastPutmaskFunc;

#define Npy_TYPE(a) Py_TYPE(a)
#define NpyArray_SIZE(a) PyArray_SIZE(a)
#define NpyArray_ITEMSIZE(a) PyArray_ITEMSIZE(a)
#define NpyArray_NDIM(a) PyArray_NDIM(a)
#define NpyArray_DIM(a, i) PyArray_DIM(a, i)
#define NpyArray_DIMS(a) PyArray_DIMS(a)
#define NpyArray_STRIDES(a) PyArray_STRIDES(a)
#define NpyArray_DESCR(a) PyArray_DESCR(a)
#define NpyArray_FLAGS(a) PyArray_FLAGS(a)
#define NpyArray_BASE(a) PyArray_BASE(a)

#define NpyArray_CHKFLAGS(a, flags) PyArray_CHKFLAGS(a, flags)
#define NpyArray_ISFORTRAN(a) PyArray_ISFORTRAN(a)
#define NpyArray_ISCONTIGUOUS(a) PyArray_ISCONTIGUOUS(a)
#define NpyArray_ISONESEGMENT(a) PyArray_ISONESEGMENT(a)
#define NpyArray_ISFLEXIBLE(obj) PyTypeNum_ISFLEXIBLE(PyArray_TYPE(obj))
#define NpyArray_ISUNSIGNED(obj) PyArray_ISUNSIGNED(obj)
#define NpyArray_ISWRITEABLE(a) PyArray_ISWRITEABLE(a)

#define NpyArray_TYPE(obj) PyArray_TYPE(obj)
#define NpyArray_NOTYPE PyArray_NOTYPE
#define NpyArray_NTYPES PyArray_NTYPES
#define NpyArray_NSORTS PyArray_NSORTS
#define NpyArray_USERDEF PyArray_USERDEF
#define NpyTypeNum_ISUSERDEF(a) PyTypeNum_ISUSERDEF(a)
#define NpyArray_BOOL PyArray_BOOL

#define NpyArray_NOSCALAR PyArray_NOSCALAR
#define NpyArray_NSCALARKINDS PyArray_NSCALARKINDS

#define NpyArray_INTP PyArray_INTP
#define NpyArray_BOOL PyArray_BOOL

#define NpyDataType_REFCHK(a) PyDataType_REFCHK(a)

#define NpyArray_MultiIter_NOTDONE(i) PyArray_MultiIter_NOTDONE(i)
#define NpyArray_MultiIter_DATA(i, n) PyArray_MultiIter_DATA(i, n)
#define NpyArray_MultiIter_NEXT(i) PyArray_MultiIter_NEXT(i)

/*
 * Functions we need to convert.
 */

#define NpyArray_FromAny(op, newType, min_depth, max_depth, flags, context) \
        PyArray_FromAny(op, newType, min_depth, max_depth, flags, context)

#define NpyArray_DescrFromType(type) \
        PyArray_DescrFromType(type)

#define NpyArray_FromArray(arr, newtype, flags) \
    ((NpyArray *)PyArray_FromArray(arr, newtype, flags))


/* ctors.c */
#define NpyArray_New(subtype, nd, dims, type_num, strides, data, itemsize, flags, obj) \
        (NpyArray *)PyArray_New(subtype, nd, dims, type_num, strides, data, itemsize, flags, obj)

#define NpyArray_CheckFromAny(op, descr, min_depth, max_depth, requires, context) \
        PyArray_CheckFromAny(op, descr, min_depth, max_depth, requires, context)

#define NpyArray_EnsureAnyArray(op)  (PyObject *)PyArray_EnsureAnyArray(op)


/* number.c */
#define NpyArray_GenericReduceFunction(m1, op, axis, rtype, out) \
        PyArray_GenericReduceFunction(m1, op, axis, rtype, out)


/* Already exists as a macro */
#define NpyArray_ContiguousFromAny(op, type, min_depth, max_depth)      \
    ((NpyArray*)NpyArray_FromAny(op, NpyArray_DescrFromType(type), min_depth, \
                                 max_depth, NPY_DEFAULT, NULL))

#define NpyArray_ContiguousFromArray(op, type)                          \
    ((NpyArray*) NpyArray_FromArray(op, NpyArray_DescrFromType(type),   \
                                    NPY_DEFAULT))



/*
 * API functions.
 */
npy_intp NpyArray_Size(NpyArray *op);
NpyArray *NpyArray_ArgMax(NpyArray *op, int axis, NpyArray *out);
NpyArray *NpyArray_CheckAxis(NpyArray *arr, int *axis, int flags);
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


NpyArray* NpyArray_TakeFrom(NpyArray *self0, NpyArray *indices0, int axis,
                            NpyArray *ret, NPY_CLIPMODE clipmode);

int NpyArray_PutTo(NpyArray *self, NpyArray* values0, NpyArray *indices0,
                   NPY_CLIPMODE clipmode);
int NpyArray_PutMask(NpyArray *self, NpyArray* values0, NpyArray* mask0);
NpyArray * NpyArray_Repeat(NpyArray *aop, NpyArray *op, int axis);
NpyArray * NpyArray_Choose(NpyArray *ip, NpyArray** mps, int n, NpyArray *ret,
                           NPY_CLIPMODE clipmode);

void NpyArray_InitArrFuncs(NpyArray_ArrFuncs *f);
int NpyArray_RegisterDataType(NpyArray_Descr *descr);
int NpyArray_RegisterCastFunc(NpyArray_Descr *descr, int totype,
                              NpyArray_VectorUnaryFunc *castfunc);
int NpyArray_RegisterCanCast(NpyArray_Descr *descr, int totype,
                             NPY_SCALARKIND scalar);
int NpyArray_TypeNumFromName(char *str);
int NpyArray_TypeNumFromTypeObj(void* typeobj);
NpyArray_Descr* NpyArray_UserDescrFromTypeNum(int typenum);

/*
 * Reference counting.
 */

#define Npy_INCREF(a) Py_INCREF(a)
#define Npy_DECREF(a) Py_DECREF(a)
#define Npy_XDECREF(a) Py_XDECREF(a)
#define NpyArray_REFCOUNT(a) PyArray_REFCOUNT(a)
#define NpyArray_INCREF(a) PyArray_INCREF(a)
#define NpyArray_XDECREF_ERR(a) PyArray_XDECREF_ERR(a)
#define NpyArray_Item_INCREF(a, descr) PyArray_Item_INCREF(a, descr)
#define NpyArray_Item_XDECREF(a, descr) PyArray_Item_XDECREF(a, descr)

/*
 * Memory
 */
#define NpyDataMem_RENEW(p, sz) PyDataMem_RENEW(p, sz)
#define NpyDataMem_FREE(p) PyDataMem_FREE(p)

#define NpyDimMem_RENEW(p, sz) PyDimMem_RENEW(p, sz)

/*
 * Error handling.
 */
#define NpyErr_SetString(exc, str) PyErr_SetString(exc, str)
#define NpyErr_NoMemory() PyErr_NoMemory()
#define NpyExc_ValueError PyExc_ValueError
#define NpyExc_MemoryError PyExc_MemoryError
#define NpyExc_TypeError PyExc_TypeError
#define NpyExc_IndexError PyExc_IndexError
#define NpyErr_Format PyErr_Format



/*
 * TMP
 */
#define NpyArray_MultiplyList(a, b) PyArray_MultiplyList(a, b)
#define NpyArray_CompareLists(a, b, n) PyArray_CompareLists(a, b, n)
#define NpyArray_NewFromDescr(a, b, c, d, e, f, g, h) \
    ((NpyArray*) PyArray_NewFromDescr(a, b, c, d, e, f, g, h))
#define NpyArray_View(a, b, c) ((NpyArray*) PyArray_View(a,b,c))
#define NpyArray_NewCopy(a, order) ((NpyArray*) PyArray_NewCopy(a, order))
#define NpyArray_UpdateFlags(a, flags) PyArray_UpdateFlags(a, flags)

extern int _flat_copyinto(PyObject *dst, PyObject *src, NPY_ORDER order);

#endif
