/*
 *  npy_number.c -
 *
 */

#include <stdlib.h>
#include <strings.h>
#include "npy_config.h"
#include "npy_api.h"
#include "npy_ufunc_object.h"
#include "npy_number.h"


NumericOps n_ops;


int NpyArray_SetNumericOp(enum NpyArray_Ops op, NpyUFuncObject *func)
{
    switch (op) {
        case npy_op_add:
            n_ops.add = func;
            break;
        case npy_op_subtract:
            n_ops.subtract = func;
            break;
        case npy_op_multiply:
            n_ops.multiply = func;
            break;
        case npy_op_divide:
            n_ops.divide = func;
            break;
        case npy_op_remainder:
            n_ops.remainder = func;
            break;
        case npy_op_power:
            n_ops.power = func;
            break;
        case npy_op_square:
            n_ops.square = func;
            break;
        case npy_op_reciprocal:
            n_ops.reciprocal = func;
            break;
        case npy_op_ones_like:
            n_ops.ones_like = func;
            break;
        case npy_op_sqrt:
            n_ops.sqrt = func;
            break;
        case npy_op_negative:
            n_ops.negative = func;
            break;
        case npy_op_absolute:
            n_ops.absolute = func;
            break;
        case npy_op_invert:
            n_ops.invert = func;
            break;
        case npy_op_left_shift:
            n_ops.left_shift = func;
            break;
        case npy_op_right_shift:
            n_ops.right_shift = func;
            break;
        case npy_op_bitwise_and:
            n_ops.bitwise_and = func;
            break;
        case npy_op_bitwise_xor:
            n_ops.bitwise_xor = func;
            break;
        case npy_op_bitwise_or:
            n_ops.bitwise_or = func;
            break;
        case npy_op_less:
            n_ops.less = func;
            break;
        case npy_op_less_equal:
            n_ops.less_equal = func;
            break;
        case npy_op_equal:
            n_ops.equal = func;
            break;
        case npy_op_not_equal:
            n_ops.not_equal = func;
            break;
        case npy_op_greater:
            n_ops.greater = func;
            break;
        case npy_op_greater_equal:
            n_ops.greater_equal = func;
            break;
        case npy_op_floor_divide:
            n_ops.floor_divide = func;
            break;
        case npy_op_true_divide:
            n_ops.true_divide = func;
            break;
        case npy_op_logical_or:
            n_ops.logical_or = func;
            break;
        case npy_op_logical_and:
            n_ops.logical_and = func;
            break;
        case npy_op_floor:
            n_ops.floor = func;
            break;
        case npy_op_ceil:
            n_ops.ceil = func;
            break;
        case npy_op_maximum:
            n_ops.maximum = func;
            break;
        case npy_op_minimum:
            n_ops.minimum = func;
            break;
        case npy_op_rint:
            n_ops.rint = func;
            break;
        case npy_op_conjugate:
            n_ops.conjugate = func;
            break;
        default:
            return -1;
    }
    return 0;
}