
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"

/*
 * Compute the size of an array (in number of items)
 */
npy_intp
NpyArray_Size(NpyArray *op)
{
    return NpyArray_SIZE(op);
}

