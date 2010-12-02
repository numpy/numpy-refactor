#ifndef _NPY_CALCULATION_H_
#define _NPY_CALCULATION_H_

#if defined(__cplusplus)
extern "C" {
#endif


struct NpyArray;

NDARRAY_API struct NpyArray 
*NpyArray_Conjugate(struct NpyArray *self, struct NpyArray *out);

NDARRAY_API struct NpyArray 
*NpyArray_ArgMax(NpyArray *op, int axis, NpyArray *out);

NDARRAY_API struct NpyArray *
NpyArray_Max(struct NpyArray *self, int axis, struct NpyArray *out);

NDARRAY_API struct NpyArray *
NpyArray_Min(struct NpyArray *self, int axis, struct NpyArray *out);

NDARRAY_API struct NpyArray *
NpyArray_Sum(struct NpyArray *self, int axis, int rtype, struct NpyArray *out);

NDARRAY_API struct NpyArray *
NpyArray_Prod(struct NpyArray *self, int axis, int rtype, struct NpyArray *out);

NDARRAY_API struct NpyArray *
NpyArray_CumSum(struct NpyArray *self, int axis, int rtype, struct NpyArray *out);

NDARRAY_API struct NpyArray *
NpyArray_CumProd(struct NpyArray *self, int axis, int rtype, struct NpyArray *out);

NDARRAY_API struct NpyArray *
NpyArray_Any(struct NpyArray *self, int axis, struct NpyArray *out);

NDARRAY_API struct NpyArray *
NpyArray_All(struct NpyArray *self, int axis, struct NpyArray *out);

#if defined(__cplusplus)
}
#endif

#endif
