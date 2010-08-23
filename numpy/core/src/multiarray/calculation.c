#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include <npy_api.h>
#include <npy_calculation.h>
#include <npy_ufunc_object.h>

#include "numpy/arrayobject.h"

#include "npy_config.h"

#include "numpy_3kcompat.h"

#include "common.h"
#include "number.h"
#include "ctors.h"
#include "arrayobject.h"

#include "calculation.h"

/* FIXME: just remove _check_axis ? */
#define _check_axis PyArray_CheckAxis
#define PyAO PyArrayObject


NPY_NO_EXPORT NpyArray_Descr *
PyArray_DescrFromObjectUnwrap(PyObject *op, NpyArray_Descr *mintype);


static double
power_of_ten(int n)
{
    static const double p10[] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8};
    double ret;
    if (n < 9) {
        ret = p10[n];
    }
    else {
        ret = 1e9;
        while (n-- > 9) {
            ret *= 10.;
        }
    }
    return ret;
}


/*NUMPY_API
 * ArgMax
 */
NPY_NO_EXPORT PyObject *
PyArray_ArgMax(PyArrayObject *op, int axis, PyArrayObject *out)
{
    RETURN_PYARRAY(NpyArray_ArgMax(PyArray_ARRAY(op), axis,
                                   PyArray_ARRAY(out)));
}


/*NUMPY_API
 * ArgMin
 */
NPY_NO_EXPORT PyObject *
PyArray_ArgMin(PyArrayObject *ap, int axis, PyArrayObject *out)
{
    PyObject *obj, *new, *ret;

    if (PyArray_ISFLEXIBLE(ap)) {
        PyErr_SetString(PyExc_TypeError,
                        "argmax is unsupported for this type");
        return NULL;
    }
    else if (PyArray_ISUNSIGNED(ap)) {
        obj = PyInt_FromLong((long) -1);
    }
    else if (PyArray_TYPE(ap) == PyArray_BOOL) {
        obj = PyInt_FromLong((long) 1);
    }
    else {
        obj = PyInt_FromLong((long) 0);
    }
    new = PyArray_EnsureAnyArray(PyNumber_Subtract(obj, (PyObject *)ap));
    Py_DECREF(obj);
    if (new == NULL) {
        return NULL;
    }
    ret = PyArray_ArgMax((PyArrayObject *)new, axis, out);
    Py_DECREF(new);
    return ret;
}


/*NUMPY_API
 * Max
 */
NPY_NO_EXPORT PyObject *
PyArray_Max(PyArrayObject *ap, int axis, PyArrayObject *out)
{
    PyArrayObject *arr;
    PyObject *ret;

    if ((arr=(PyArrayObject *)_check_axis(ap, &axis, 0)) == NULL) {
        return NULL;
    }
    ret = PyArray_GenericReduceFunction(arr, PyArray_GetNumericOp(npy_op_maximum), 
                                        axis, PyArray_TYPE(arr), out);
    Py_DECREF(arr);
    return ret;
}

/*NUMPY_API
 * Min
 */
NPY_NO_EXPORT PyObject *
PyArray_Min(PyArrayObject *ap, int axis, PyArrayObject *out)
{
    PyArrayObject *arr;
    PyObject *ret;

    if ((arr=(PyArrayObject *)_check_axis(ap, &axis, 0)) == NULL) {
        return NULL;
    }
    ret = PyArray_GenericReduceFunction(arr, PyArray_GetNumericOp(npy_op_minimum), axis,
                                        PyArray_TYPE(arr), out);
    Py_DECREF(arr);
    return ret;
}

/*NUMPY_API
 * Ptp
 */
NPY_NO_EXPORT PyObject *
PyArray_Ptp(PyArrayObject *ap, int axis, PyArrayObject *out)
{
    PyArrayObject *arr;
    PyObject *ret;
    PyObject *obj1 = NULL, *obj2 = NULL;

    if ((arr=(PyArrayObject *)_check_axis(ap, &axis, 0)) == NULL) {
        return NULL;
    }
    obj1 = PyArray_Max(arr, axis, out);
    if (obj1 == NULL) {
        goto fail;
    }
    obj2 = PyArray_Min(arr, axis, NULL);
    if (obj2 == NULL) {
        goto fail;
    }
    Py_DECREF(arr);
    if (out) {
        ret = PyObject_CallFunction(PyArray_GetNumericOp(npy_op_subtract), "OOO", out, obj2, out);
    }
    else {
        ret = PyNumber_Subtract(obj1, obj2);
    }
    Py_DECREF(obj1);
    Py_DECREF(obj2);
    return ret;

 fail:
    Py_XDECREF(arr);
    Py_XDECREF(obj1);
    Py_XDECREF(obj2);
    return NULL;
}



/*NUMPY_API
 * Set variance to 1 to by-pass square-root calculation and return variance
 * Std
 */
NPY_NO_EXPORT PyObject *
PyArray_Std(PyArrayObject *self, int axis, int rtype, PyArrayObject *out,
            int variance)
{
    return __New_PyArray_Std(self, axis, rtype, out, variance, 0);
}

NPY_NO_EXPORT PyObject *
__New_PyArray_Std(PyArrayObject *self, int axis, int rtype, PyArrayObject *out,
                  int variance, int num)
{
    PyObject *obj1 = NULL, *obj2 = NULL, *obj3 = NULL, *new = NULL;
    PyObject *ret = NULL, *newshape = NULL;
    int i, n;
    intp val;

    if ((new = _check_axis(self, &axis, 0)) == NULL) {
        return NULL;
    }
    /* Compute and reshape mean */
    obj1 = PyArray_EnsureAnyArray(PyArray_Mean((PyAO *)new, axis, rtype, NULL));
    if (obj1 == NULL) {
        Py_DECREF(new);
        return NULL;
    }
    n = PyArray_NDIM(new);
    newshape = PyTuple_New(n);
    if (newshape == NULL) {
        Py_DECREF(obj1);
        Py_DECREF(new);
        return NULL;
    }
    for (i = 0; i < n; i++) {
        if (i == axis) {
            val = 1;
        }
        else {
            val = PyArray_DIM(new,i);
        }
        PyTuple_SET_ITEM(newshape, i, PyInt_FromLong((long)val));
    }
    obj2 = PyArray_Reshape((PyAO *)obj1, newshape);
    Py_DECREF(obj1);
    Py_DECREF(newshape);
    if (obj2 == NULL) {
        Py_DECREF(new);
        return NULL;
    }

    /* Compute x = x - mx */
    obj1 = PyArray_EnsureAnyArray(PyNumber_Subtract((PyObject *)new, obj2));
    Py_DECREF(obj2);
    if (obj1 == NULL) {
        Py_DECREF(new);
        return NULL;
    }
    /* Compute x * x */
    if (PyArray_ISCOMPLEX(obj1)) {
        obj3 = PyArray_Conjugate((PyAO *)obj1, NULL);
    }
    else {
        obj3 = obj1;
        Py_INCREF(obj1);
    }
    if (obj3 == NULL) {
        Py_DECREF(new);
        return NULL;
    }
    obj2 = PyArray_EnsureAnyArray                                      \
        (PyArray_GenericBinaryFunction((PyAO *)obj1, obj3, PyArray_GetNumericOp(npy_op_multiply)));
    Py_DECREF(obj1);
    Py_DECREF(obj3);
    if (obj2 == NULL) {
        Py_DECREF(new);
        return NULL;
    }
    if (PyArray_ISCOMPLEX(obj2)) {
        obj3 = PyObject_GetAttrString(obj2, "real");
        switch(rtype) {
        case NPY_CDOUBLE:
            rtype = NPY_DOUBLE;
            break;
        case NPY_CFLOAT:
            rtype = NPY_FLOAT;
            break;
        case NPY_CLONGDOUBLE:
            rtype = NPY_LONGDOUBLE;
            break;
        }
    }
    else {
        obj3 = obj2;
        Py_INCREF(obj2);
    }
    if (obj3 == NULL) {
        Py_DECREF(new);
        return NULL;
    }
    /* Compute add.reduce(x*x,axis) */
    obj1 = PyArray_GenericReduceFunction((PyAO *)obj3, PyArray_GetNumericOp(npy_op_add),
                                         axis, rtype, NULL);
    Py_DECREF(obj3);
    Py_DECREF(obj2);
    if (obj1 == NULL) {
        Py_DECREF(new);
        return NULL;
    }
    n = PyArray_DIM(new,axis);
    Py_DECREF(new);
    n = (n-num);
    if (n == 0) {
        n = 1;
    }
    obj2 = PyFloat_FromDouble(1.0/((double )n));
    if (obj2 == NULL) {
        Py_DECREF(obj1);
        return NULL;
    }
    ret = PyNumber_Multiply(obj1, obj2);
    Py_DECREF(obj1);
    Py_DECREF(obj2);

    if (!variance) {
        obj1 = PyArray_EnsureAnyArray(ret);
        /* sqrt() */
        ret = PyArray_GenericUnaryFunction((PyAO *)obj1, PyArray_GetNumericOp(npy_op_sqrt));
        Py_DECREF(obj1);
    }
    if (ret == NULL || PyArray_CheckExact(self)) {
        return ret;
    }
    if (PyArray_Check(self) && Py_TYPE(self) == Py_TYPE(ret)) {
        return ret;
    }
    obj1 = PyArray_EnsureArray(ret);
    if (obj1 == NULL) {
        return NULL;
    }
    ret = PyArray_View((PyAO *)obj1, NULL, Py_TYPE(self));
    Py_DECREF(obj1);
    if (out) {
        if (PyArray_CopyAnyInto(out, (PyArrayObject *)ret) < 0) {
            Py_DECREF(ret);
            return NULL;
        }
        Py_DECREF(ret);
        Py_INCREF(out);
        return (PyObject *)out;
    }
    return ret;
}


/*NUMPY_API
 *Sum
 */
NPY_NO_EXPORT PyObject *
PyArray_Sum(PyArrayObject *self, int axis, int rtype, PyArrayObject *out)
{
    PyObject *new, *ret;

    if ((new = _check_axis(self, &axis, 0)) == NULL) {
        return NULL;
    }
    ret = PyArray_GenericReduceFunction((PyAO *)new, PyArray_GetNumericOp(npy_op_add), axis,
                                        rtype, out);
    Py_DECREF(new);
    return ret;
}

/*NUMPY_API
 * Prod
 */
NPY_NO_EXPORT PyObject *
PyArray_Prod(PyArrayObject *self, int axis, int rtype, PyArrayObject *out)
{
    PyObject *new, *ret;

    if ((new = _check_axis(self, &axis, 0)) == NULL) {
        return NULL;
    }
    ret = PyArray_GenericReduceFunction((PyAO *)new, PyArray_GetNumericOp(npy_op_multiply), axis,
                                        rtype, out);
    Py_DECREF(new);
    return ret;
}

/*NUMPY_API
 *CumSum
 */
NPY_NO_EXPORT PyObject *
PyArray_CumSum(PyArrayObject *self, int axis, int rtype, PyArrayObject *out)
{
    PyArrayObject *ret = NULL;
    
    ASSIGN_TO_PYARRAY(ret, NpyArray_CumSum(PyArray_ARRAY(self), axis, rtype, 
                                           (NULL != out) ? PyArray_ARRAY(out) : NULL));
    return (PyObject *)ret;
}


/*NUMPY_API
 * CumProd
 */
NPY_NO_EXPORT PyObject *
PyArray_CumProd(PyArrayObject *self, int axis, int rtype, PyArrayObject *out)
{
    PyArrayObject *ret = NULL;
    
    ASSIGN_TO_PYARRAY(ret, NpyArray_CumProd(PyArray_ARRAY(self), axis, rtype, 
                                            (NULL != out) ? PyArray_ARRAY(out) : NULL));
    return (PyObject *)ret;
}

/*NUMPY_API
 * Round
 */
NPY_NO_EXPORT PyObject *
PyArray_Round(PyArrayObject *a, int decimals, PyArrayObject *out)
{
    PyObject *f, *ret = NULL, *tmp, *op1, *op2;
    int ret_int=0;

    if (out && (PyArray_SIZE(out) != PyArray_SIZE(a))) {
        PyErr_SetString(PyExc_ValueError,
                        "invalid output shape");
        return NULL;
    }
    if (PyArray_ISCOMPLEX(a)) {
        PyObject *part;
        PyObject *round_part;
        PyObject *new;
        int res;

        if (out) {
            new = (PyObject *)out;
            Py_INCREF(new);
        }
        else {
            new = PyArray_Copy(a);
            if (new == NULL) {
                return NULL;
            }
        }

        /* new.real = a.real.round(decimals) */
        part = PyObject_GetAttrString(new, "real");
        if (part == NULL) {
            Py_DECREF(new);
            return NULL;
        }
        part = PyArray_EnsureAnyArray(part);
        round_part = PyArray_Round((PyArrayObject *)part,
                                   decimals, NULL);
        Py_DECREF(part);
        if (round_part == NULL) {
            Py_DECREF(new);
            return NULL;
        }
        res = PyObject_SetAttrString(new, "real", round_part);
        Py_DECREF(round_part);
        if (res < 0) {
            Py_DECREF(new);
            return NULL;
        }

        /* new.imag = a.imag.round(decimals) */
        part = PyObject_GetAttrString(new, "imag");
        if (part == NULL) {
            Py_DECREF(new);
            return NULL;
        }
        part = PyArray_EnsureAnyArray(part);
        round_part = PyArray_Round((PyArrayObject *)part,
                                   decimals, NULL);
        Py_DECREF(part);
        if (round_part == NULL) {
            Py_DECREF(new);
            return NULL;
        }
        res = PyObject_SetAttrString(new, "imag", round_part);
        Py_DECREF(round_part);
        if (res < 0) {
            Py_DECREF(new);
            return NULL;
        }
        return new;
    }
    /* do the most common case first */
    if (decimals >= 0) {
        if (PyArray_ISINTEGER(a)) {
            if (out) {
                if (PyArray_CopyAnyInto(out, a) < 0) {
                    return NULL;
                }
                Py_INCREF(out);
                return (PyObject *)out;
            }
            else {
                Py_INCREF(a);
                return (PyObject *)a;
            }
        }
        if (decimals == 0) {
            if (out) {
                return PyObject_CallFunction(PyArray_GetNumericOp(npy_op_rint), "OO", a, out);
            }
            return PyObject_CallFunction(PyArray_GetNumericOp(npy_op_rint), "O", a);
        }
        op1 = PyArray_GetNumericOp(npy_op_multiply);
        op2 = PyArray_GetNumericOp(npy_op_true_divide);
    }
    else {
        op1 = PyArray_GetNumericOp(npy_op_true_divide);
        op2 = PyArray_GetNumericOp(npy_op_multiply);
        decimals = -decimals;
    }
    if (!out) {
        NpyArray_Descr *my_descr;
        PyArray_Descr *my_descr_wrap;

        if (PyArray_ISINTEGER(a)) {
            ret_int = 1;
            my_descr = NpyArray_DescrFromType(NPY_DOUBLE);
        }
        else {
            my_descr = PyArray_DESCR(a);
            Npy_INCREF(my_descr);
        }

        /* TODO: Ugly.
           Refactor PyArray_Empty into core once zero fill is refactored. */
        my_descr_wrap = PyArray_Descr_WRAP(my_descr);
        Py_INCREF(my_descr_wrap);
        Npy_DECREF(my_descr);
        out = (PyArrayObject *)PyArray_Empty(PyArray_NDIM(a), PyArray_DIMS(a),
                                              my_descr_wrap,
                                              PyArray_ISFORTRAN(a));
        if (out == NULL) {
            return NULL;
        }
    }
    else {
        Py_INCREF(out);
    }
    f = PyFloat_FromDouble(power_of_ten(decimals));
    if (f == NULL) {
        return NULL;
    }
    ret = PyObject_CallFunction(op1, "OOO", a, f, out);
    if (ret == NULL) {
        goto finish;
    }
    tmp = PyObject_CallFunction(PyArray_GetNumericOp(npy_op_rint), "OO", ret, ret);
    if (tmp == NULL) {
        Py_DECREF(ret);
        ret = NULL;
        goto finish;
    }
    Py_DECREF(tmp);
    tmp = PyObject_CallFunction(op2, "OOO", ret, f, ret);
    if (tmp == NULL) {
        Py_DECREF(ret);
        ret = NULL;
        goto finish;
    }
    Py_DECREF(tmp);

 finish:
    Py_DECREF(f);
    Py_DECREF(out);
    if (ret_int) {
        Npy_INCREF(PyArray_DESCR(a));
        tmp = Npy_INTERFACE(NpyArray_CastToType(PyArray_ARRAY(ret),
                                                PyArray_DESCR(a),
                                                PyArray_ISFORTRAN(a)));
        Py_DECREF(ret);
        return tmp;
    }
    return ret;
}


/*NUMPY_API
 * Mean
 */
NPY_NO_EXPORT PyObject *
PyArray_Mean(PyArrayObject *self, int axis, int rtype, PyArrayObject *out)
{
    PyObject *obj1 = NULL, *obj2 = NULL;
    PyObject *new, *ret;

    if ((new = _check_axis(self, &axis, 0)) == NULL) {
        return NULL;
    }
    obj1 = PyArray_GenericReduceFunction((PyAO *)new, PyArray_GetNumericOp(npy_op_add), axis,
                                         rtype, out);
    obj2 = PyFloat_FromDouble((double) PyArray_DIM(new,axis));
    Py_DECREF(new);
    if (obj1 == NULL || obj2 == NULL) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        return NULL;
    }
    if (!out) {
#if defined(NPY_PY3K)
        ret = PyNumber_TrueDivide(obj1, obj2);
#else
        ret = PyNumber_Divide(obj1, obj2);
#endif
    }
    else {
        ret = PyObject_CallFunction(PyArray_GetNumericOp(npy_op_divide), "OOO", out, obj2, out);
    }
    Py_DECREF(obj1);
    Py_DECREF(obj2);
    return ret;
}

/*NUMPY_API
 * Any
 */
NPY_NO_EXPORT PyObject *
PyArray_Any(PyArrayObject *self, int axis, PyArrayObject *out)
{
    PyArrayObject *ret = NULL;
    
    ASSIGN_TO_PYARRAY(ret, NpyArray_Any(PyArray_ARRAY(self), axis,
                                        (NULL != out) ? PyArray_ARRAY(out) : NULL));
    return (PyObject *)ret;
}

/*NUMPY_API
 * All
 */
NPY_NO_EXPORT PyObject *
PyArray_All(PyArrayObject *self, int axis, PyArrayObject *out)
{
    PyArrayObject *ret = NULL;
    
    ASSIGN_TO_PYARRAY(ret, NpyArray_All(PyArray_ARRAY(self), axis,
                                        (NULL != out) ? PyArray_ARRAY(out) : NULL));
    return (PyObject *)ret;
}




static PyObject *
_GenericBinaryOutFunction(PyArrayObject *m1, PyObject *m2, PyArrayObject *out,
                          PyObject *op)
{
    if (out == NULL) {
        return PyObject_CallFunction(op, "OO", m1, m2);
    }
    else {
        return PyObject_CallFunction(op, "OOO", m1, m2, out);
    }
}

static PyObject *
_slow_array_clip(PyArrayObject *self, PyObject *min, PyObject *max, PyArrayObject *out)
{
    PyObject *res1=NULL, *res2=NULL;

    if (max != NULL) {
        res1 = _GenericBinaryOutFunction(self, max, out, PyArray_GetNumericOp(npy_op_minimum));
        if (res1 == NULL) {
            return NULL;
        }
    }
    else {
        res1 = (PyObject *)self;
        Py_INCREF(res1);
    }

    if (min != NULL) {
        res2 = _GenericBinaryOutFunction((PyArrayObject *)res1,
                                         min, out, 
                                         PyArray_GetNumericOp(npy_op_maximum));
        if (res2 == NULL) {
            Py_XDECREF(res1);
            return NULL;
        }
    }
    else {
        res2 = res1;
        Py_INCREF(res2);
    }
    Py_DECREF(res1);
    return res2;
}

/*NUMPY_API
 * Clip
 */
NPY_NO_EXPORT PyObject *
PyArray_Clip(PyArrayObject *self, PyObject *min, PyObject *max,
             PyArrayObject *out)
{
    PyArray_FastClipFunc *func;
    int outgood = 0, ingood = 0;
    PyArrayObject *maxa = NULL;
    PyArrayObject *mina = NULL;
    PyArrayObject *newout = NULL, *newin = NULL;
    NpyArray_Descr *indescr, *newdescr;
    char *max_data, *min_data;
    PyObject *zero;

    if ((max == NULL) && (min == NULL)) {
        PyErr_SetString(PyExc_ValueError, "array_clip: must set either max "\
                        "or min");
        return NULL;
    }

    func = PyArray_DESCR(self)->f->fastclip;
    if (func == NULL || (min != NULL && !PyArray_CheckAnyScalar(min)) ||
        (max != NULL && !PyArray_CheckAnyScalar(max))) {
        return _slow_array_clip(self, min, max, out);
    }
    /* Use the fast scalar clip function */

    /* First we need to figure out the correct type */
    indescr = NULL;
    if (min != NULL) {
        indescr = PyArray_DescrFromObjectUnwrap(min, NULL);
        if (indescr == NULL) {
            return NULL;
        }
    }
    if (max != NULL) {
        newdescr = PyArray_DescrFromObjectUnwrap(max, indescr);
        Npy_XDECREF(indescr);
        if (newdescr == NULL) {
            return NULL;
        }
    }
    else {
        /* Steal the reference */
        newdescr = indescr;
    }


    /*
     * Use the scalar descriptor only if it is of a bigger
     * KIND than the input array (and then find the
     * type that matches both).
     */
    if (PyArray_ScalarKind(newdescr->type_num, NULL) >
        PyArray_ScalarKind(PyArray_TYPE(self), NULL)) {
        indescr = NpyArray_SmallType(newdescr, PyArray_DESCR(self));
        func = indescr->f->fastclip;
        if (func == NULL) {
            return _slow_array_clip(self, min, max, out);
        }
    }
    else {
        indescr = PyArray_DESCR(self);
        Npy_INCREF(indescr);
    }
    Npy_DECREF(newdescr);

    if (!NpyArray_ISNBO(indescr->byteorder)) {
        NpyArray_Descr *descr2;
        descr2 = NpyArray_DescrNewByteorder(indescr, '=');
        Npy_DECREF(indescr);
        if (descr2 == NULL) {
            goto fail;
        }
        indescr = descr2;
    }

    /* Convert max to an array */
    if (max != NULL) {
        maxa = (NPY_AO *)PyArray_FromAnyUnwrap(max, indescr, 0, 0,
                                               NPY_DEFAULT, NULL);
        if (maxa == NULL) {
            return NULL;
        }
    }
    else {
        /* Side-effect of PyArray_FromAny */
        Npy_DECREF(indescr);
    }

    /*
     * If we are unsigned, then make sure min is not < 0
     * This is to match the behavior of _slow_array_clip
     *
     * We allow min and max to go beyond the limits
     * for other data-types in which case they
     * are interpreted as their modular counterparts.
    */
    if (min != NULL) {
        if (PyArray_ISUNSIGNED(self)) {
            int cmp;
            zero = PyInt_FromLong(0);
            cmp = PyObject_RichCompareBool(min, zero, Py_LT);
            if (cmp == -1) {
                Py_DECREF(zero);
                goto fail;
            }
            if (cmp == 1) {
                min = zero;
            }
            else {
                Py_DECREF(zero);
                Py_INCREF(min);
            }
        }
        else {
            Py_INCREF(min);
        }

        /* Convert min to an array */
        Npy_INCREF(indescr);
        mina = (NPY_AO *)PyArray_FromAnyUnwrap(min, indescr, 0, 0,
                                               NPY_DEFAULT, NULL);
        Py_DECREF(min);
        if (mina == NULL) {
            goto fail;
        }
    }


    /*
     * Check to see if input is single-segment, aligned,
     * and in native byteorder
     */
    if (PyArray_ISONESEGMENT(self) && PyArray_CHKFLAGS(self, ALIGNED) &&
        PyArray_ISNOTSWAPPED(self) && (PyArray_DESCR(self) == indescr)) {
        ingood = 1;
    }
    if (!ingood) {
        int flags;

        if (PyArray_ISFORTRAN(self)) {
            flags = NPY_FARRAY;
        }
        else {
            flags = NPY_CARRAY;
        }
        Npy_INCREF(indescr);
        ASSIGN_TO_PYARRAY(newin,
                          NpyArray_FromArray(PyArray_ARRAY(self),
                                             indescr, flags));
        if (newin == NULL) {
            goto fail;
        }
    }
    else {
        newin = self;
        Py_INCREF(newin);
    }

    /*
     * At this point, newin is a single-segment, aligned, and correct
     * byte-order array of the correct type
     *
     * if ingood == 0, then it is a copy, otherwise,
     * it is the original input.
     */

    /*
     * If we have already made a copy of the data, then use
     * that as the output array
     */
    if (out == NULL && !ingood) {
        out = newin;
    }

    /*
     * Now, we know newin is a usable array for fastclip,
     * we need to make sure the output array is available
     * and usable
     */
    if (out == NULL) {
        Npy_INCREF(indescr);
        ASSIGN_TO_PYARRAY(out,
                          NpyArray_Alloc(indescr, PyArray_NDIM(self),
                                         PyArray_DIMS(self), 
                                         PyArray_ISFORTRAN(self),
                                         self));
        if (out == NULL) {
            goto fail;
        }
        outgood = 1;
    }
    else Py_INCREF(out);
    /* Input is good at this point */
    if (out == newin) {
        outgood = 1;
    }
    if (!outgood && PyArray_ISONESEGMENT(out) &&
        PyArray_CHKFLAGS(out, ALIGNED) && PyArray_ISNOTSWAPPED(out) &&
        NpyArray_EquivTypes(PyArray_DESCR(out), indescr)) {
        outgood = 1;
    }

    /*
     * Do we still not have a suitable output array?
     * Create one, now
     */
    if (!outgood) {
        int oflags;
        if (PyArray_ISFORTRAN(out))
            oflags = NPY_FARRAY;
        else
            oflags = NPY_CARRAY;
        oflags |= NPY_UPDATEIFCOPY | NPY_FORCECAST;
        Npy_INCREF(indescr);
        ASSIGN_TO_PYARRAY(newout,
                          NpyArray_FromArray(PyArray_ARRAY(out), indescr, oflags));
        if (newout == NULL) {
            goto fail;
        }
    }
    else {
        newout = out;
        Py_INCREF(newout);
    }

    /* make sure the shape of the output array is the same */
    if (!PyArray_SAMESHAPE(newin, newout)) {
        PyErr_SetString(PyExc_ValueError, "clip: Output array must have the"
                        "same shape as the input.");
        goto fail;
    }
    if (PyArray_BYTES(newout) != PyArray_BYTES(newin)) {
        memcpy(PyArray_BYTES(newout),
               PyArray_BYTES(newin),
               PyArray_NBYTES(newin));
    }

    /* Now we can call the fast-clip function */
    min_data = max_data = NULL;
    if (mina != NULL) {
        min_data = PyArray_BYTES(mina);
    }
    if (maxa != NULL) {
        max_data = PyArray_BYTES(maxa);
    }
    func(PyArray_BYTES(newin), PyArray_SIZE(newin),
         min_data, max_data, PyArray_BYTES(newout));

    /* Clean up temporary variables */
    Py_XDECREF(mina);
    Py_XDECREF(maxa);
    Py_DECREF(newin);
    /* Copy back into out if out was not already a nice array. */
    Py_DECREF(newout);
    return (PyObject *)out;

 fail:
    Py_XDECREF(maxa);
    Py_XDECREF(mina);
    Py_XDECREF(newin);
    PyArray_XDECREF_ERR(newout);
    return NULL;
}




/*NUMPY_API
 * Conjugate
 */
NPY_NO_EXPORT PyObject *
PyArray_Conjugate(PyArrayObject *self, PyArrayObject *out)
{
    RETURN_PYARRAY(NpyArray_Conjugate(PyArray_ARRAY(self), PyArray_ARRAY(out)));
}


/*NUMPY_API
 * Trace
 */
NPY_NO_EXPORT PyObject *
PyArray_Trace(PyArrayObject *self, int offset, int axis1, int axis2,
              int rtype, PyArrayObject *out)
{
    PyObject *diag = NULL, *ret = NULL;

    diag = PyArray_Diagonal(self, offset, axis1, axis2);
    if (diag == NULL) {
        return NULL;
    }
    ret = PyArray_GenericReduceFunction((PyAO *)diag,
                                        PyArray_GetNumericOp(npy_op_add), -1, rtype, out);
    Py_DECREF(diag);
    return ret;
}

