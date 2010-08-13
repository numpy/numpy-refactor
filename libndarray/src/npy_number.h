#ifndef _NPY_NUMBER_H_
#define _NPY_NUMBER_H_

#include "npy_object.h"
#include "npy_index.h"


struct NpyUFuncObject;



struct NumericOps {
    struct NpyUFuncObject *add;
    struct NpyUFuncObject *subtract;
    struct NpyUFuncObject *multiply;
    struct NpyUFuncObject *divide;
    struct NpyUFuncObject *remainder;
    struct NpyUFuncObject *power;
    struct NpyUFuncObject *square;
    struct NpyUFuncObject *reciprocal;
    struct NpyUFuncObject *ones_like;
    struct NpyUFuncObject *sqrt;
    struct NpyUFuncObject *negative;
    struct NpyUFuncObject *absolute;
    struct NpyUFuncObject *invert;
    struct NpyUFuncObject *left_shift;
    struct NpyUFuncObject *right_shift;
    struct NpyUFuncObject *bitwise_and;
    struct NpyUFuncObject *bitwise_xor;
    struct NpyUFuncObject *bitwise_or;
    struct NpyUFuncObject *less;
    struct NpyUFuncObject *less_equal;
    struct NpyUFuncObject *equal;
    struct NpyUFuncObject *not_equal;
    struct NpyUFuncObject *greater;
    struct NpyUFuncObject *greater_equal;
    struct NpyUFuncObject *floor_divide;
    struct NpyUFuncObject *true_divide;
    struct NpyUFuncObject *logical_or;
    struct NpyUFuncObject *logical_and;
    struct NpyUFuncObject *floor;
    struct NpyUFuncObject *ceil;
    struct NpyUFuncObject *maximum;
    struct NpyUFuncObject *minimum;
    struct NpyUFuncObject *rint;
    struct NpyUFuncObject *conjugate;
};
typedef struct NumericOps NumericOps;


#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT NumericOps n_ops;
//extern NPY_NO_EXPORT PyNumberMethods array_as_number;
#endif


#endif
