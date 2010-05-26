
#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"

/*
 * Ravel
 * Returns a contiguous array
 */
NpyArray*
NpyArray_Ravel(NpyArray *a, NPY_ORDER fortran)
{
    NpyArray_Dims newdim = {NULL,1};
    npy_intp val[1] = {-1};

    if (fortran == NPY_ANYORDER) {
        fortran = NpyArray_ISFORTRAN(a);
    }
    newdim.ptr = val;
    if (!fortran && NpyArray_ISCONTIGUOUS(a)) {
        return NpyArray_Newshape(a, &newdim, PyArray_CORDER);
    }
    else if (fortran && PyArray_ISFORTRAN(a)) {
        return NpyArray_Newshape(a, &newdim, PyArray_FORTRANORDER);
    }
    else {
        return PyArray_Flatten(a, fortran);
    }
}

/*
 * Flatten
 */
NpyArray *
NpyArray_Flatten(NpyArray *a, NPY_ORDER order)
{
    NpyArray *ret;
    npy_intp size;

    if (order == NPY_ANYORDER) {
        order = NpyArray_ISFORTRAN(a);
    }
    Npy_INCREF(a->descr);
    size = NpyArray_SIZE(a);
    ret = NpyArray_NewFromDescr(Py_TYPE(a),
                                a->descr,
                                1, &size,
                                NULL,
                                NULL,
                                0, (NpyObject *)a);

    if (ret == NULL) {
        return NULL;
    }
    if (_flat_copyinto(ret, (NpyObject *)a, order) < 0) {
        Npy_DECREF(ret);
        return NULL;
    }
    return ret;
}
