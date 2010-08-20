#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

/*#include <stdio.h>*/
#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include <npy_ufunc_object.h>

#include "numpy/ufuncobject.h"
#include "arrayobject.h"
#include "npy_api.h"

#include "npy_config.h"

#include "numpy_3kcompat.h"

#include "number.h"

/*************************************************************************
 ****************   Implement Number Protocol ****************************
 *************************************************************************/


/*
 * Dictionary can contain any of the numeric operations, by name.
 * Those not present will not be changed
 */

/* FIXME - macro contains a return */
#define SET(npyop, op)   temp = PyDict_GetItemString(dict, #op); \
    if (temp != NULL) {                                   \
        if (!(PyCallable_Check(temp))) {                     \
            return -1;                                    \
        }                                                 \
        NpyArray_SetNumericOp(npyop, PyUFunc_UFUNC((PyUFuncObject *)temp)); \
    }


/*NUMPY_API
 *Set internal structure with number functions that all arrays will use
 */
NPY_NO_EXPORT int
PyArray_SetNumericOps(PyObject *dict)
{
    PyObject *temp = NULL;
    
    SET(npy_op_add, add);
    SET(npy_op_subtract, subtract);
    SET(npy_op_multiply, multiply);
    SET(npy_op_divide, divide);
    SET(npy_op_remainder, remainder);
    SET(npy_op_power, power);
    SET(npy_op_square, square);
    SET(npy_op_reciprocal, reciprocal);
    SET(npy_op_ones_like, ones_like);
    SET(npy_op_sqrt, sqrt);
    SET(npy_op_negative, negative);
    SET(npy_op_absolute, absolute);
    SET(npy_op_invert, invert);
    SET(npy_op_left_shift, left_shift);
    SET(npy_op_right_shift, right_shift);
    SET(npy_op_bitwise_and, bitwise_and);
    SET(npy_op_bitwise_or, bitwise_or);
    SET(npy_op_bitwise_xor, bitwise_xor);
    SET(npy_op_less, less);
    SET(npy_op_less_equal, less_equal);
    SET(npy_op_equal, equal);
    SET(npy_op_not_equal, not_equal);
    SET(npy_op_greater, greater);
    SET(npy_op_greater_equal, greater_equal);
    SET(npy_op_floor_divide, floor_divide);
    SET(npy_op_true_divide, true_divide);
    SET(npy_op_logical_or, logical_or);
    SET(npy_op_logical_and, logical_and);
    SET(npy_op_floor, floor);
    SET(npy_op_ceil, ceil);
    SET(npy_op_maximum, maximum);
    SET(npy_op_minimum, minimum);
    SET(npy_op_rint, rint);
    SET(npy_op_conjugate, conjugate);
    return 0;
}

/* FIXME - macro contains goto */
#define GET(npyop, op) if (PyDict_SetItemString(dict, #op, (PyObject *)PyArray_GetNumericOp(npyop))==-1)    \
        goto fail;

/*NUMPY_API
  Get dictionary showing number functions that all arrays will use
*/
NPY_NO_EXPORT PyObject *
PyArray_GetNumericOps(void)
{
    PyObject *dict;
    if ((dict = PyDict_New())==NULL)
        return NULL;
    GET(npy_op_add, add);
    GET(npy_op_subtract, subtract);
    GET(npy_op_multiply, multiply);
    GET(npy_op_divide, divide);
    GET(npy_op_remainder, remainder);
    GET(npy_op_power, power);
    GET(npy_op_square, square);
    GET(npy_op_reciprocal, reciprocal);
    GET(npy_op_ones_like, ones_like);
    GET(npy_op_sqrt, sqrt);
    GET(npy_op_negative, negative);
    GET(npy_op_absolute, absolute);
    GET(npy_op_invert, invert);
    GET(npy_op_left_shift, left_shift);
    GET(npy_op_right_shift, right_shift);
    GET(npy_op_bitwise_and, bitwise_and);
    GET(npy_op_bitwise_or, bitwise_or);
    GET(npy_op_bitwise_xor, bitwise_xor);
    GET(npy_op_less, less);
    GET(npy_op_less_equal, less_equal);
    GET(npy_op_equal, equal);
    GET(npy_op_not_equal, not_equal);
    GET(npy_op_greater, greater);
    GET(npy_op_greater_equal, greater_equal);
    GET(npy_op_floor_divide, floor_divide);
    GET(npy_op_true_divide, true_divide);
    GET(npy_op_logical_or, logical_or);
    GET(npy_op_logical_and, logical_and);
    GET(npy_op_floor, floor);
    GET(npy_op_ceil, ceil);
    GET(npy_op_maximum, maximum);
    GET(npy_op_minimum, minimum);
    GET(npy_op_rint, rint);
    GET(npy_op_conjugate, conjugate);
    return dict;

 fail:
    Py_DECREF(dict);
    return NULL;
}



NPY_NO_EXPORT PyObject *
PyArray_GetNumericOp(enum NpyArray_Ops op)
{
    return (PyObject *)PyUFunc_WRAP( NpyArray_GetNumericOp(op) );
}



static PyObject *
_get_keywords(int rtype, PyArrayObject *out)
{
    PyObject *kwds = NULL;
    if (rtype != PyArray_NOTYPE || out != NULL) {
        kwds = PyDict_New();
        if (rtype != PyArray_NOTYPE) {
            PyArray_Descr *descr;
            descr = PyArray_DescrFromType(rtype);
            if (descr) {
                PyDict_SetItemString(kwds, "dtype", (PyObject *)descr);
                Py_DECREF(descr);
            }
        }
        if (out != NULL) {
            PyDict_SetItemString(kwds, "out", (PyObject *)out);
        }
    }
    return kwds;
}

NPY_NO_EXPORT PyObject *
PyArray_GenericReduceFunction(PyArrayObject *m1, PyObject *op, int axis,
                              int rtype, PyArrayObject *out)
{
    PyObject *args, *ret = NULL, *meth;
    PyObject *kwds;
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    args = Py_BuildValue("(Oi)", m1, axis);
    kwds = _get_keywords(rtype, out);
    meth = PyObject_GetAttrString(op, "reduce");
    if (meth && PyCallable_Check(meth)) {
        ret = PyObject_Call(meth, args, kwds);
    }
    Py_DECREF(args);
    Py_DECREF(meth);
    Py_XDECREF(kwds);
    return ret;
}


NPY_NO_EXPORT PyObject *
PyArray_GenericAccumulateFunction(PyArrayObject *m1, PyObject *op, int axis,
                                  int rtype, PyArrayObject *out)
{
    PyObject *args, *ret = NULL, *meth;
    PyObject *kwds;
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    args = Py_BuildValue("(Oi)", m1, axis);
    kwds = _get_keywords(rtype, out);
    meth = PyObject_GetAttrString(op, "accumulate");
    if (meth && PyCallable_Check(meth)) {
        ret = PyObject_Call(meth, args, kwds);
    }
    Py_DECREF(args);
    Py_DECREF(meth);
    Py_XDECREF(kwds);
    return ret;
}


NPY_NO_EXPORT PyObject *
PyArray_GenericBinaryFunction(PyArrayObject *m1, PyObject *m2, PyObject *op)
{
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    return PyObject_CallFunction(op, "OO", m1, m2);
}

NPY_NO_EXPORT PyObject *
PyArray_GenericUnaryFunction(PyArrayObject *m1, PyObject *op)
{
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    return PyObject_CallFunction(op, "(O)", m1);
}

static PyObject *
PyArray_GenericInplaceBinaryFunction(PyArrayObject *m1,
                                     PyObject *m2, PyObject *op)
{
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    return PyObject_CallFunction(op, "OOO", m1, m2, m1);
}

static PyObject *
PyArray_GenericInplaceUnaryFunction(PyArrayObject *m1, PyObject *op)
{
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    return PyObject_CallFunction(op, "OO", m1, m1);
}

static PyObject *
array_add(PyArrayObject *m1, PyObject *m2)
{
    NpyArray *ret = NULL;
    /* TODO: This version breaks some tests. Don't know why yet... */
#if 0    
    // If m2 is an array, use the faster core routines.
    if (PyArray_Check(m2)) {
        ret = NpyArray_GenericBinaryFunction(PyArray_ARRAY(m1), 
                                             PyArray_ARRAY(m2), 
                                             NpyArray_GetNumericOp(npy_op_add)));
        Py_INCREF(Npy_INTERFACE(ret));
        Npy_DECREF(ret);
        return (PyObject *)Npy_INTERFACE(ret);
    }
#endif
    
    // Fall back to calling into Python.
    return PyArray_GenericBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_add));
}

static PyObject *
array_subtract(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_subtract));
}

static PyObject *
array_multiply(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_multiply));
}

static PyObject *
array_divide(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_divide));
}

static PyObject *
array_remainder(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_remainder));
}

static int
array_power_is_scalar(PyObject *o2, double* exp)
{
    PyObject *temp;
    const int optimize_fpexps = 1;

    if (PyInt_Check(o2)) {
        *exp = (double)PyInt_AsLong(o2);
        return 1;
    }
    if (optimize_fpexps && PyFloat_Check(o2)) {
        *exp = PyFloat_AsDouble(o2);
        return 1;
    }
    if ((PyArray_IsZeroDim(o2) &&
         ((PyArray_ISINTEGER(o2) ||
           (optimize_fpexps && PyArray_ISFLOAT(o2))))) ||
        PyArray_IsScalar(o2, Integer) ||
        (optimize_fpexps && PyArray_IsScalar(o2, Floating))) {
        temp = Py_TYPE(o2)->tp_as_number->nb_float(o2);
        if (temp != NULL) {
            *exp = PyFloat_AsDouble(o2);
            Py_DECREF(temp);
            return 1;
        }
    }
#if (PY_VERSION_HEX >= 0x02050000)
    if (PyIndex_Check(o2)) {
        PyObject* value = PyNumber_Index(o2);
        Py_ssize_t val;
        if (value==NULL) {
            if (PyErr_Occurred()) {
                PyErr_Clear();
            }
            return 0;
        }
        val = PyInt_AsSsize_t(value);
        Py_DECREF(value);
        if (val == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return 0;
        }
        *exp = (double) val;
        return 1;
    }
#endif
    return 0;
}

/* optimize float array or complex array to a scalar power */
static PyObject *
fast_scalar_power(PyArrayObject *a1, PyObject *o2, int inplace)
{
    double exp;

    if (PyArray_Check(a1) && array_power_is_scalar(o2, &exp)) {
        PyObject *fastop = NULL;
        if (PyArray_ISFLOAT(a1) || PyArray_ISCOMPLEX(a1)) {
            if (exp == 1.0) {
                /* we have to do this one special, as the
                   "copy" method of array objects isn't set
                   up early enough to be added
                   by PyArray_SetNumericOps.
                */
                if (inplace) {
                    Py_INCREF(a1);
                    return (PyObject *)a1;
                } else {
                    return PyArray_Copy(a1);
                }
            }
            else if (exp == -1.0) {
                fastop = PyArray_GetNumericOp(npy_op_reciprocal);
            }
            else if (exp ==  0.0) {
                fastop = PyArray_GetNumericOp(npy_op_ones_like);
            }
            else if (exp ==  0.5) {
                fastop = PyArray_GetNumericOp(npy_op_sqrt);
            }
            else if (exp ==  2.0) {
                fastop = PyArray_GetNumericOp(npy_op_square);
            }
            else {
                return NULL;
            }

            if (inplace) {
                return PyArray_GenericInplaceUnaryFunction(a1, fastop);
            } else {
                return PyArray_GenericUnaryFunction(a1, fastop);
            }
        }
        else if (exp==2.0) {
            fastop = PyArray_GetNumericOp(npy_op_multiply);
            if (inplace) {
                return PyArray_GenericInplaceBinaryFunction
                    (a1, (PyObject *)a1, fastop);
            }
            else {
                return PyArray_GenericBinaryFunction
                    (a1, (PyObject *)a1, fastop);
            }
        }
    }
    return NULL;
}

static PyObject *
array_power(PyArrayObject *a1, PyObject *o2, PyObject *NPY_UNUSED(modulo))
{
    /* modulo is ignored! */
    PyObject *value;
    value = fast_scalar_power(a1, o2, 0);
    if (!value) {
        value = PyArray_GenericBinaryFunction(a1, o2, PyArray_GetNumericOp(npy_op_power));
    }
    return value;
}


static PyObject *
array_negative(PyArrayObject *m1)
{
    return PyArray_GenericUnaryFunction(m1, PyArray_GetNumericOp(npy_op_negative));
}

static PyObject *
array_absolute(PyArrayObject *m1)
{
    return PyArray_GenericUnaryFunction(m1, PyArray_GetNumericOp(npy_op_absolute));
}

static PyObject *
array_invert(PyArrayObject *m1)
{
    return PyArray_GenericUnaryFunction(m1, PyArray_GetNumericOp(npy_op_invert));
}

static PyObject *
array_left_shift(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_left_shift));
}

static PyObject *
array_right_shift(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_right_shift));
}

static PyObject *
array_bitwise_and(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_bitwise_and));
}

static PyObject *
array_bitwise_or(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_bitwise_or));
}

static PyObject *
array_bitwise_xor(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_bitwise_xor));
}

static PyObject *
array_inplace_add(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_add));
}

static PyObject *
array_inplace_subtract(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_subtract));
}

static PyObject *
array_inplace_multiply(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_multiply));
}

static PyObject *
array_inplace_divide(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_divide));
}

static PyObject *
array_inplace_remainder(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_remainder));
}

static PyObject *
array_inplace_power(PyArrayObject *a1, PyObject *o2,
                    PyObject *NPY_UNUSED(modulo))
{
    /* modulo is ignored! */
    PyObject *value;
    value = fast_scalar_power(a1, o2, 1);
    if (!value) {
        value = PyArray_GenericInplaceBinaryFunction(a1, o2, PyArray_GetNumericOp(npy_op_power));
    }
    return value;
}

static PyObject *
array_inplace_left_shift(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_left_shift));
}

static PyObject *
array_inplace_right_shift(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_right_shift));
}

static PyObject *
array_inplace_bitwise_and(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_bitwise_and));
}

static PyObject *
array_inplace_bitwise_or(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_bitwise_or));
}

static PyObject *
array_inplace_bitwise_xor(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_bitwise_xor));
}

static PyObject *
array_floor_divide(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_floor_divide));
}

static PyObject *
array_true_divide(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, PyArray_GetNumericOp(npy_op_true_divide));
}

static PyObject *
array_inplace_floor_divide(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2,
                                                PyArray_GetNumericOp(npy_op_floor_divide));
}

static PyObject *
array_inplace_true_divide(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2,
                                                PyArray_GetNumericOp(npy_op_true_divide));
}

static int
_array_nonzero(PyArrayObject *mp)
{
    intp n;

    n = PyArray_SIZE(mp);
    if (n == 1) {
        return PyArray_DESCR(mp)->f->nonzero(PyArray_BYTES(mp),
                                             PyArray_ARRAY(mp));
    }
    else if (n == 0) {
        return 0;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "The truth value of an array "
                        "with more than one element is ambiguous. "
                        "Use a.any() or a.all()");
        return -1;
    }
}



static PyObject *
array_divmod(PyArrayObject *op1, PyObject *op2)
{
    PyObject *divp, *modp, *result;

    divp = array_floor_divide(op1, op2);
    if (divp == NULL) {
        return NULL;
    }
    modp = array_remainder(op1, op2);
    if (modp == NULL) {
        Py_DECREF(divp);
        return NULL;
    }
    result = Py_BuildValue("OO", divp, modp);
    Py_DECREF(divp);
    Py_DECREF(modp);
    return result;
}


NPY_NO_EXPORT PyObject *
array_int(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can be"
                        " converted to Python scalars");
        return NULL;
    }
    pv = PyArray_DESCR(v)->f->getitem(PyArray_BYTES(v), PyArray_ARRAY(v));
    if (pv == NULL) {
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to an int; "
                        "scalar object is not a number");
        Py_DECREF(pv);
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number->nb_int == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "
                        "scalar number to int");
        Py_DECREF(pv);
        return NULL;
    }

    pv2 = Py_TYPE(pv)->tp_as_number->nb_int(pv);
    Py_DECREF(pv);
    return pv2;
}

static PyObject *
array_float(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can "
                        "be converted to Python scalars");
        return NULL;
    }
    pv = PyArray_DESCR(v)->f->getitem(PyArray_BYTES(v), PyArray_ARRAY(v));
    if (pv == NULL) {
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to a "
                        "float; scalar object is not a number");
        Py_DECREF(pv);
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number->nb_float == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "
                        "scalar number to float");
        Py_DECREF(pv);
        return NULL;
    }
    pv2 = Py_TYPE(pv)->tp_as_number->nb_float(pv);
    Py_DECREF(pv);
    return pv2;
}

#if !defined(NPY_PY3K)

static PyObject *
array_long(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can "
                        "be converted to Python scalars");
        return NULL;
    }
    pv = PyArray_DESCR(v)->f->getitem(PyArray_BYTES(v), PyArray_ARRAY(v));
    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to an int; "
                        "scalar object is not a number");
        Py_DECREF(pv);
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number->nb_long == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "
                        "scalar number to long");
        Py_DECREF(pv);
        return NULL;
    }
    pv2 = Py_TYPE(pv)->tp_as_number->nb_long(pv);
    Py_DECREF(pv);
    return pv2;
}

static PyObject *
array_oct(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can "
                        "be converted to Python scalars");
        return NULL;
    }
    pv = PyArray_DESCR(v)->f->getitem(PyArray_BYTES(v), PyArray_ARRAY(v));
    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to an int; "
                        "scalar object is not a number");
        Py_DECREF(pv);
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number->nb_oct == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "
                        "scalar number to oct");
        Py_DECREF(pv);
        return NULL;
    }
    pv2 = Py_TYPE(pv)->tp_as_number->nb_oct(pv);
    Py_DECREF(pv);
    return pv2;
}

static PyObject *
array_hex(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can "
                        "be converted to Python scalars");
        return NULL;
    }
    pv = PyArray_DESCR(v)->f->getitem(PyArray_BYTES(v), PyArray_ARRAY(v));
    if (Py_TYPE(pv)->tp_as_number == 0) {
        Py_DECREF(pv);
        PyErr_SetString(PyExc_TypeError, "cannot convert to an int; "
                        "scalar object is not a number");
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number->nb_hex == 0) {
        Py_DECREF(pv);
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "
                        "scalar number to hex");
        return NULL;
    }
    pv2 = Py_TYPE(pv)->tp_as_number->nb_hex(pv);
    Py_DECREF(pv);
    return pv2;
}

#endif

static PyObject *
_array_copy_nice(PyArrayObject *self)
{
    return PyArray_Return((PyArrayObject *) PyArray_Copy(self));
}

#if PY_VERSION_HEX >= 0x02050000
static PyObject *
array_index(PyArrayObject *v)
{
    if (!PyArray_ISINTEGER(v) || PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only integer arrays with "
                        "one element can be converted to an index");
        return NULL;
    }
    return PyArray_DESCR(v)->f->getitem(PyArray_BYTES(v), PyArray_ARRAY(v));
}
#endif


NPY_NO_EXPORT PyNumberMethods array_as_number = {
    (binaryfunc)array_add,                      /*nb_add*/
    (binaryfunc)array_subtract,                 /*nb_subtract*/
    (binaryfunc)array_multiply,                 /*nb_multiply*/
#if defined(NPY_PY3K)
#else
    (binaryfunc)array_divide,                   /*nb_divide*/
#endif
    (binaryfunc)array_remainder,                /*nb_remainder*/
    (binaryfunc)array_divmod,                   /*nb_divmod*/
    (ternaryfunc)array_power,                   /*nb_power*/
    (unaryfunc)array_negative,                  /*nb_neg*/
    (unaryfunc)_array_copy_nice,                /*nb_pos*/
    (unaryfunc)array_absolute,                  /*(unaryfunc)array_abs,*/
    (inquiry)_array_nonzero,                    /*nb_nonzero*/
    (unaryfunc)array_invert,                    /*nb_invert*/
    (binaryfunc)array_left_shift,               /*nb_lshift*/
    (binaryfunc)array_right_shift,              /*nb_rshift*/
    (binaryfunc)array_bitwise_and,              /*nb_and*/
    (binaryfunc)array_bitwise_xor,              /*nb_xor*/
    (binaryfunc)array_bitwise_or,               /*nb_or*/
#if defined(NPY_PY3K)
#else
    0,                                          /*nb_coerce*/
#endif
    (unaryfunc)array_int,                       /*nb_int*/
#if defined(NPY_PY3K)
    0,                                          /*nb_reserved*/
#else
    (unaryfunc)array_long,                      /*nb_long*/
#endif
    (unaryfunc)array_float,                     /*nb_float*/
#if defined(NPY_PY3K)
#else
    (unaryfunc)array_oct,                       /*nb_oct*/
    (unaryfunc)array_hex,                       /*nb_hex*/
#endif

    /*
     * This code adds augmented assignment functionality
     * that was made available in Python 2.0
     */
    (binaryfunc)array_inplace_add,              /*inplace_add*/
    (binaryfunc)array_inplace_subtract,         /*inplace_subtract*/
    (binaryfunc)array_inplace_multiply,         /*inplace_multiply*/
#if defined(NPY_PY3K)
#else
    (binaryfunc)array_inplace_divide,           /*inplace_divide*/
#endif
    (binaryfunc)array_inplace_remainder,        /*inplace_remainder*/
    (ternaryfunc)array_inplace_power,           /*inplace_power*/
    (binaryfunc)array_inplace_left_shift,       /*inplace_lshift*/
    (binaryfunc)array_inplace_right_shift,      /*inplace_rshift*/
    (binaryfunc)array_inplace_bitwise_and,      /*inplace_and*/
    (binaryfunc)array_inplace_bitwise_xor,      /*inplace_xor*/
    (binaryfunc)array_inplace_bitwise_or,       /*inplace_or*/

    (binaryfunc)array_floor_divide,             /*nb_floor_divide*/
    (binaryfunc)array_true_divide,              /*nb_true_divide*/
    (binaryfunc)array_inplace_floor_divide,     /*nb_inplace_floor_divide*/
    (binaryfunc)array_inplace_true_divide,      /*nb_inplace_true_divide*/

#if PY_VERSION_HEX >= 0x02050000
    (unaryfunc)array_index,                     /* nb_index */
#endif

};
