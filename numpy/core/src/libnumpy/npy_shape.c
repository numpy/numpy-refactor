
#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"

/*NUMPY_API
 * Return Transpose.
 */
NpyArray*
NpyArray_Transpose(NpyArray *ap, NpyArray_Dims *permute)
{
    npy_intp *axes, axis;
    npy_intp i, n;
    npy_intp permutation[NPY_MAXDIMS], reverse_permutation[NPY_MAXDIMS];
    NpyArray *ret = NULL;

    if (permute == NULL) {
        n = ap->nd;
        for (i = 0; i < n; i++) {
            permutation[i] = n-1-i;
        }
    }
    else {
        n = permute->len;
        axes = permute->ptr;
        if (n != ap->nd) {
            NpyErr_SetString(NpyExc_ValueError,
                            "axes don't match array");
            return NULL;
        }
        for (i = 0; i < n; i++) {
            reverse_permutation[i] = -1;
        }
        for (i = 0; i < n; i++) {
            axis = axes[i];
            if (axis < 0) {
                axis = ap->nd + axis;
            }
            if (axis < 0 || axis >= ap->nd) {
                NpyErr_SetString(NpyExc_ValueError,
                                "invalid axis for this array");
                return NULL;
            }
            if (reverse_permutation[axis] != -1) {
                NpyErr_SetString(NpyExc_ValueError,
                                "repeated axis in transpose");
                return NULL;
            }
            reverse_permutation[axis] = i;
            permutation[i] = axis;
        }
        for (i = 0; i < n; i++) {
        }
    }

    /*
     * this allocates memory for dimensions and strides (but fills them
     * incorrectly), sets up descr, and points data at ap->data.
     */
    Npy_INCREF(ap->descr);
    ret = NpyArray_NewFromDescr(Py_TYPE(ap),
                                ap->descr,
                                n, ap->dimensions,
                                NULL, ap->data, ap->flags,
                                (PyObject *)ap);
    if (ret == NULL) {
        return NULL;
    }
    /* point at true owner of memory: */
    ret->base = (PyObject *)ap;
    Npy_INCREF(ap);

    /* fix the dimensions and strides of the return-array */
    for (i = 0; i < n; i++) {
        ret->dimensions[i] = ap->dimensions[permutation[i]];
        ret->strides[i] = ap->strides[permutation[i]];
    }
    PyArray_UpdateFlags(ret, NPY_CONTIGUOUS | NPY_FORTRAN);
    return ret;
}

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
