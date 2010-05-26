
#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"

/*
 * return a new view of the array object with all of its unit-length
 * dimensions squeezed out if needed, otherwise
 * return the same array.
 */
NpyArray*
NpyArray_Squeeze(NpyArray *self)
{
    int nd = self->nd;
    int newnd = nd;
    npy_intp dimensions[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];
    int i, j;
    NpyArray *ret;

    if (nd == 0) {
        Npy_INCREF(self);
        return self;
    }
    for (j = 0, i = 0; i < nd; i++) {
        if (self->dimensions[i] == 1) {
            newnd -= 1;
        }
        else {
            dimensions[j] = self->dimensions[i];
            strides[j++] = self->strides[i];
        }
    }

    Npy_INCREF(self->descr);
    ret = NpyArray_NewFromDescr(Py_TYPE(self),
                                self->descr,
                                newnd, dimensions,
                                strides, self->data,
                                self->flags,
                                (NpyObject *)self);
    if (ret == NULL) {
        return NULL;
    }
    PyArray_FLAGS(ret) &= ~NPY_OWNDATA;
    PyArray_BASE(ret) = (NpyObject*)self;
    Npy_INCREF(self);
    return ret;
}

/*
 * SwapAxes
 */
NpyArray*
NpyArray_SwapAxes(NpyArray *ap, int a1, int a2)
{
    NpyArray_Dims new_axes;
    npy_intp dims[NPY_MAXDIMS];
    int n, i, val;
    NpyArray *ret;

    if (a1 == a2) {
        Npy_INCREF(ap);
        return ap;
    }

    n = ap->nd;
    if (n <= 1) {
        Npy_INCREF(ap);
        return ap;
    }

    if (a1 < 0) {
        a1 += n;
    }
    if (a2 < 0) {
        a2 += n;
    }
    if ((a1 < 0) || (a1 >= n)) {
        NpyErr_SetString(NpyExc_ValueError,
                        "bad axis1 argument to swapaxes");
        return NULL;
    }
    if ((a2 < 0) || (a2 >= n)) {
        NpyErr_SetString(NpyExc_ValueError,
                        "bad axis2 argument to swapaxes");
        return NULL;
    }
    new_axes.ptr = dims;
    new_axes.len = n;

    for (i = 0; i < n; i++) {
        if (i == a1) {
            val = a2;
        }
        else if (i == a2) {
            val = a1;
        }
        else {
            val = i;
        }
        new_axes.ptr[i] = val;
    }
    ret = NpyArray_Transpose(ap, &new_axes);
    return ret;
}

/*
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
