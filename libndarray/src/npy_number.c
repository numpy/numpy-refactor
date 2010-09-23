/*
 *  npy_number.c -
 *
 */

#include <stdlib.h>
#include "npy_config.h"
#include "npy_api.h"
#include "npy_ufunc_object.h"
#include "npy_number.h"
#include "npy_arrayobject.h"


NumericOps n_ops = {
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,
};


static NpyUFuncObject **get_op_loc(enum NpyArray_Ops op)
{
    NpyUFuncObject **loc = NULL;

    switch (op) {
        case npy_op_add:
            loc = &n_ops.add;
            break;
        case npy_op_subtract:
            loc = &n_ops.subtract;
            break;
        case npy_op_multiply:
            loc = &n_ops.multiply;
            break;
        case npy_op_divide:
            loc = &n_ops.divide;
            break;
        case npy_op_remainder:
            loc = &n_ops.remainder;
            break;
        case npy_op_power:
            loc = &n_ops.power;
            break;
        case npy_op_square:
            loc = &n_ops.square;
            break;
        case npy_op_reciprocal:
            loc = &n_ops.reciprocal;
            break;
        case npy_op_ones_like:
            loc = &n_ops.ones_like;
            break;
        case npy_op_sqrt:
            loc = &n_ops.sqrt;
            break;
        case npy_op_negative:
            loc = &n_ops.negative;
            break;
        case npy_op_absolute:
            loc = &n_ops.absolute;
            break;
        case npy_op_invert:
            loc = &n_ops.invert;
            break;
        case npy_op_left_shift:
            loc = &n_ops.left_shift;
            break;
        case npy_op_right_shift:
            loc = &n_ops.right_shift;
            break;
        case npy_op_bitwise_and:
            loc = &n_ops.bitwise_and;
            break;
        case npy_op_bitwise_xor:
            loc = &n_ops.bitwise_xor;
            break;
        case npy_op_bitwise_or:
            loc = &n_ops.bitwise_or;
            break;
        case npy_op_less:
            loc = &n_ops.less;
            break;
        case npy_op_less_equal:
            loc = &n_ops.less_equal;
            break;
        case npy_op_equal:
            loc = &n_ops.equal;
            break;
        case npy_op_not_equal:
            loc = &n_ops.not_equal;
            break;
        case npy_op_greater:
            loc = &n_ops.greater;
            break;
        case npy_op_greater_equal:
            loc = &n_ops.greater_equal;
            break;
        case npy_op_floor_divide:
            loc = &n_ops.floor_divide;
            break;
        case npy_op_true_divide:
            loc = &n_ops.true_divide;
            break;
        case npy_op_logical_or:
            loc = &n_ops.logical_or;
            break;
        case npy_op_logical_and:
            loc = &n_ops.logical_and;
            break;
        case npy_op_floor:
            loc = &n_ops.floor;
            break;
        case npy_op_ceil:
            loc = &n_ops.ceil;
            break;
        case npy_op_maximum:
            loc = &n_ops.maximum;
            break;
        case npy_op_minimum:
            loc = &n_ops.minimum;
            break;
        case npy_op_rint:
            loc = &n_ops.rint;
            break;
        case npy_op_conjugate:
            loc = &n_ops.conjugate;
            break;
        default:
            loc = NULL;
    }
    return loc;
}


/* Returns the ufunc function associated with the specified operator. */
NDARRAY_API NpyUFuncObject *
NpyArray_GetNumericOp(enum NpyArray_Ops op)
{
    NpyUFuncObject **loc = get_op_loc(op);
    return (NULL != loc) ? *loc : NULL;
}


/* Sets the provided function as the global implementation of the specified
   operation. Any existing operator is replaced. */
NDARRAY_API int 
NpyArray_SetNumericOp(enum NpyArray_Ops op, NpyUFuncObject *func)
{
    NpyUFuncObject **loc = get_op_loc(op);

    assert(NPY_VALID_MAGIC == func->nob_magic_number);

    if (NULL == loc) {
        return -1;
    }
    Npy_XDECREF(*loc);
    *loc = func;
    Npy_INCREF(*loc);
    return 0;
}


NDARRAY_API NpyArray *
NpyArray_GenericBinaryFunction(NpyArray *m1, NpyArray *m2, NpyUFuncObject *op,
                               NpyArray *out)
{
    NpyArray *mps[NPY_MAXARGS];

    assert(NULL != op && NPY_VALID_MAGIC == op->nob_magic_number);
    assert(NULL != m1 && NPY_VALID_MAGIC == m1->nob_magic_number);
    assert(NULL != m2 && NPY_VALID_MAGIC == m2->nob_magic_number);
    assert(out == NULL || out->nob_magic_number == NPY_VALID_MAGIC);

    mps[0] = m1;
    mps[1] = m2;
    mps[2] = out;
    if (0 > NpyUFunc_GenericFunction(op, 3, mps, NULL, NPY_FALSE, NULL, NULL)) {
        return NULL;
    }
    return mps[op->nin];
}


NDARRAY_API NpyArray *
NpyArray_GenericUnaryFunction(NpyArray *m1, NpyUFuncObject *op, NpyArray* out)
{
    NpyArray *mps[NPY_MAXARGS];

    assert(NULL != op && NPY_VALID_MAGIC == op->nob_magic_number);
    assert(NULL != m1 && NPY_VALID_MAGIC == m1->nob_magic_number);
    assert(out == NULL || out->nob_magic_number == NPY_VALID_MAGIC);

    mps[0] = m1;
    mps[1] = out;
    if (0 > NpyUFunc_GenericFunction(op, 2, mps, NULL, NPY_FALSE, NULL, NULL)) {
        return NULL;
    }
    return mps[op->nin];
}
