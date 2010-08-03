/*
  Python Multiarray Module -- A useful collection of functions for creating and
  using ndarrays

  Original file
  Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu

  Modified for numpy in 2005

  Travis E. Oliphant
  oliphant@ee.byu.edu
  Brigham Young University
*/

/* $Id: multiarraymodule.c,v 1.36 2005/09/14 00:14:00 teoliphant Exp $ */

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "numpy/numpy_api.h"
#include "numpy/npy_math.h"

#include "numpy/npy_descriptor.h"

#include "npy_config.h"

#include "npy_3kcompat.h"

NPY_NO_EXPORT int NPY_NUMUSERTYPES = 0;

#define PyAO PyArrayObject

/* Internal APIs */
#include "arraytypes.h"
#include "arrayobject.h"
#include "hashdescr.h"
#include "descriptor.h"
#include "calculation.h"
#include "number.h"
#include "scalartypes.h"
#include "numpymemoryview.h"


/* Defined by the core library. */
extern void initlibnumpy(struct NpyArray_FunctionDefs *,
                         npy_tp_error_set,
                         npy_tp_error_occurred,
                         npy_tp_error_clear);


/* Defind in arraytypes.c.src */
extern struct NpyArray_FunctionDefs _array_function_defs;



/*NUMPY_API
 * Get Priority from object
 */
NPY_NO_EXPORT double
PyArray_GetPriority(PyObject *obj, double default_)
{
    PyObject *ret;
    double priority = PyArray_PRIORITY;

    if (PyArray_CheckExact(obj))
        return priority;

    ret = PyObject_GetAttrString(obj, "__array_priority__");
    if (ret != NULL) {
        priority = PyFloat_AsDouble(ret);
    }
    if (PyErr_Occurred()) {
        PyErr_Clear();
        priority = default_;
    }
    Py_XDECREF(ret);
    return priority;
}


/*NUMPY_API
 * Multiply a List of ints
 */
NPY_NO_EXPORT int
PyArray_MultiplyIntList(int *l1, int n)
{
    return NpyArray_MultiplyIntList(l1, n);
}


/*NUMPY_API
 * Multiply a List
 */
NPY_NO_EXPORT intp
PyArray_MultiplyList(intp *l1, int n)
{
    return NpyArray_MultiplyList(l1, n);
}


/*NUMPY_API
 * Multiply a List of Non-negative numbers with over-flow detection.
 */
NPY_NO_EXPORT intp
PyArray_OverflowMultiplyList(intp *l1, int n)
{
    return NpyArray_OverflowMultiplyList(l1, n);
}


/*NUMPY_API
 * Produce a pointer into array
 */
NPY_NO_EXPORT void *
PyArray_GetPtr(PyArrayObject *obj, intp* ind)
{
    return NpyArray_GetPtr(PyArray_ARRAY(obj), ind);
}


/*NUMPY_API
 * Compare Lists
 */
NPY_NO_EXPORT int
PyArray_CompareLists(intp *l1, intp *l2, int n)
{
    return NpyArray_CompareLists(l1, l2, n);
}

/*
 * simulates a C-style 1-3 dimensional array which can be accesed using
 * ptr[i]  or ptr[i][j] or ptr[i][j][k] -- requires pointer allocation
 * for 2-d and 3-d.
 *
 * For 2-d and up, ptr is NOT equivalent to a statically defined
 * 2-d or 3-d array.  In particular, it cannot be passed into a
 * function that requires a true pointer to a fixed-size array.
 */

/*NUMPY_API
 * Simulate a C-array
 * steals a reference to typedescr -- can be NULL
 */
NPY_NO_EXPORT int
PyArray_AsCArray(PyObject **op, void *ptr, intp *dims, int nd,
                 PyArray_Descr* typedescr)
{
    PyArrayObject *ap, *oldAp;
    intp result;

    if ((nd < 1) || (nd > 3)) {
        NpyErr_SetString(NpyExc_ValueError,
                         "C arrays of only 1-3 dimensions available");
        Npy_XDECREF(typedescr);
        return -1;
    }
    ap = (PyArrayObject *) PyArray_FromAny(*op, typedescr, nd, nd,
                                           NPY_CARRAY, NULL);
    if (ap == NULL) {
        return -1;
    }

    /* TODO: LOTS of potential bugs here, code is ugly.  Above PyArray_FromAny steals a reference to
       typedescr, and so does NpyArray_AsCArray.  However, they steal references to _difference_ objects,
       first one is an interface object, second is a core object.

       Also, PyArray_FromAny creates a new Python object that gets thrown away - probably want to refactor
       PyArray_FromAny to return a core-only object.

       Lastly, I think this function leaks a reference. op comes in pointing to an object which gets
       converted to an array and *op is overwritten with the new array object but the original object
       is not decref'd.  This is how it was so I left it alone.  Either the caller must decref or it's
       a leak or I just need more coffee. */
    _Npy_INCREF(typedescr->descr);
    oldAp = ap;
    result = NpyArray_AsCArray(&PyArray_LARRAY(ap), ptr, dims, nd, typedescr->descr);
    Py_DECREF(oldAp);
    *op = (PyObject *) ap;
    return result;
}

/* Deprecated --- Use PyArray_AsCArray instead */

/*NUMPY_API
 * Convert to a 1D C-array
 */
NPY_NO_EXPORT int
PyArray_As1D(PyObject **op, char **ptr, int *d1, int typecode)
{
    intp newd1;
    PyArray_Descr *descr;
    char msg[] = "PyArray_As1D: use PyArray_AsCArray.";

    if (DEPRECATE(msg) < 0) {
        return -1;
    }
    descr = PyArray_DescrFromType(typecode);
    if (PyArray_AsCArray(op, (void *)ptr, &newd1, 1, descr) == -1) {
        return -1;
    }
    *d1 = (int) newd1;
    return 0;
}


/*NUMPY_API
 * Convert to a 2D C-array
 */
NPY_NO_EXPORT int
PyArray_As2D(PyObject **op, char ***ptr, int *d1, int *d2, int typecode)
{
    intp newdims[2];
    PyArray_Descr *descr;
    char msg[] = "PyArray_As1D: use PyArray_AsCArray.";

    if (DEPRECATE(msg) < 0) {
        return -1;
    }
    descr = PyArray_DescrFromType(typecode);
    if (PyArray_AsCArray(op, (void *)ptr, newdims, 2, descr) == -1) {
        return -1;
    }
    *d1 = (int ) newdims[0];
    *d2 = (int ) newdims[1];
    return 0;
}

/* End Deprecated */

/*NUMPY_API
 * Free pointers created if As2D is called
 */
NPY_NO_EXPORT int
PyArray_Free(PyObject *op, void *ptr)
{
    return NpyArray_Free(PyArray_ARRAY(op), ptr);
}


static PyObject *
_swap_and_concat(PyObject *op, int axis, int n)
{
    PyObject *newtup = NULL;
    PyObject *otmp, *arr;
    int i;

    newtup = PyTuple_New(n);
    if (newtup == NULL) {
        return NULL;
    }
    for (i = 0; i < n; i++) {
        otmp = PySequence_GetItem(op, i);
        arr = PyArray_FROM_O(otmp);
        Py_DECREF(otmp);
        if (arr == NULL) {
            goto fail;
        }
        otmp = PyArray_SwapAxes((PyArrayObject *)arr, axis, 0);
        Py_DECREF(arr);
        if (otmp == NULL) {
            goto fail;
        }
        PyTuple_SET_ITEM(newtup, i, otmp);
    }
    otmp = PyArray_Concatenate(newtup, 0);
    Py_DECREF(newtup);
    if (otmp == NULL) {
        return NULL;
    }
    arr = PyArray_SwapAxes((PyArrayObject *)otmp, axis, 0);
    Py_DECREF(otmp);
    return arr;

 fail:
    Py_DECREF(newtup);
    return NULL;
}


/*NUMPY_API
 * Concatenate
 *
 * Concatenate an arbitrary Python sequence into an array.
 * op is a python object supporting the sequence interface.
 * Its elements will be concatenated together to form a single
 * multidimensional array. If axis is MAX_DIMS or bigger, then
 * each sequence object will be flattened before concatenation
*/
NPY_NO_EXPORT PyObject *
PyArray_Concatenate(PyObject *op, int axis)
{
    PyArrayObject *ret, **mps;
    PyObject *otmp;
    int i, n, tmp, nd = 0, new_dim;
    char *data;
    PyTypeObject *subtype;
    double prior1, prior2;
    intp numbytes;

    n = PySequence_Length(op);
    if (n == -1) {
        return NULL;
    }
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "concatenation of zero-length sequences is "\
                        "impossible");
        return NULL;
    }

    if ((axis < 0) || ((0 < axis) && (axis < MAX_DIMS))) {
        return _swap_and_concat(op, axis, n);
    }
    mps = PyArray_ConvertToCommonType(op, &n);
    if (mps == NULL) {
        return NULL;
    }

    /*
     * Make sure these arrays are legal to concatenate.
     * Must have same dimensions except d0
     */
    prior1 = PyArray_PRIORITY;
    subtype = &PyArray_Type;
    ret = NULL;
    for (i = 0; i < n; i++) {
        if (axis >= MAX_DIMS) {
            otmp = PyArray_Ravel(mps[i],0);
            Py_DECREF(mps[i]);
            mps[i] = (PyArrayObject *)otmp;
        }
        if (Py_TYPE(mps[i]) != subtype) {
            prior2 = PyArray_GetPriority((PyObject *)(mps[i]), 0.0);
            if (prior2 > prior1) {
                prior1 = prior2;
                subtype = Py_TYPE(mps[i]);
            }
        }
    }

    new_dim = 0;
    for (i = 0; i < n; i++) {
        if (mps[i] == NULL) {
            goto fail;
        }
        if (i == 0) {
            nd = PyArray_NDIM(mps[i]);
        }
        else {
            if (nd != PyArray_NDIM(mps[i])) {
                PyErr_SetString(PyExc_ValueError,
                                "arrays must have same "\
                                "number of dimensions");
                goto fail;
            }
            if (!PyArray_CompareLists(PyArray_DIMS(mps[0])+1,
                                      PyArray_DIMS(mps[i])+1,
                                      nd-1)) {
                PyErr_SetString(PyExc_ValueError,
                                "array dimensions must "\
                                "agree except for d_0");
                goto fail;
            }
        }
        if (nd == 0) {
            PyErr_SetString(PyExc_ValueError,
                            "0-d arrays can't be concatenated");
            goto fail;
        }
        new_dim += PyArray_DIM(mps[i], 0);
    }
    tmp = PyArray_DIM(mps[0], 0);
    PyArray_DIM(mps[0], 0) = new_dim;
    _Npy_INCREF(PyArray_DESCR(mps[0]));
    ASSIGN_TO_PYARRAY(ret, NpyArray_NewFromDescr(PyArray_DESCR(mps[0]), nd,
                                                 PyArray_DIMS(mps[0]),
                                                 NULL, NULL, 0,
                                                 NPY_FALSE, subtype, ret));
    PyArray_DIM(mps[0], 0) = tmp;

    if (ret == NULL) {
        goto fail;
    }
    data = PyArray_BYTES(ret);
    for (i = 0; i < n; i++) {
        numbytes = PyArray_NBYTES(mps[i]);
        memcpy(data, PyArray_BYTES(mps[i]), numbytes);
        data += numbytes;
    }

    PyArray_INCREF(ret);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    PyDataMem_FREE(mps);
    return (PyObject *)ret;

 fail:
    Py_XDECREF(ret);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    PyDataMem_FREE(mps);
    return NULL;
}


/*NUMPY_API
 * ScalarKind
 */
NPY_NO_EXPORT NPY_SCALARKIND
PyArray_ScalarKind(int typenum, PyArrayObject **arr)
{
    if (arr != NULL) {
        return NpyArray_ScalarKind(typenum, &PyArray_LARRAY(*arr));
    } else {
        return NpyArray_ScalarKind(typenum, NULL);
    }
}


/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_CanCoerceScalar(int thistype, int neededtype,
                        NPY_SCALARKIND scalar)
{
    return NpyArray_CanCoerceScalar(thistype, neededtype, scalar);
}

/* Could perhaps be redone to not make contiguous arrays */

/*NUMPY_API
 * Numeric.innerproduct(a,v)
 */
NPY_NO_EXPORT PyObject *
PyArray_InnerProduct(PyObject *op1, PyObject *op2)
{
    /* TODO: wrap return value. */
    PyArrayObject *ap1, *ap2, *ret;
    NpyArray *prod = NULL;
    int typenum;
    PyArray_Descr *typec;

    typenum = PyArray_ObjectType(op1, 0);
    typenum = PyArray_ObjectType(op2, typenum);

    typec = PyArray_DescrFromType(typenum);
    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 0, 0, ALIGNED, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 0, 0, ALIGNED, NULL);
    if (ap2 == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(ap1) == 0 || PyArray_NDIM(ap2) == 0) {
        ret = (PyArray_NDIM(ap1) == 0 ? ap1 : ap2);
        ret = (PyArrayObject *)Py_TYPE(ret)->tp_as_number->nb_multiply(
                                            (PyObject *)ap1, (PyObject *)ap2);
        Py_DECREF(ap1);
        Py_DECREF(ap2);
        return (PyObject *)ret;
    }

    prod = NpyArray_InnerProduct(PyArray_ARRAY(ap1), 
                                 PyArray_ARRAY(ap2), typenum);

    Py_DECREF(ap1);
    Py_DECREF(ap2);

    /* Get the interface object and move the reference. */
    ret = Npy_INTERFACE(prod);
    Py_INCREF(ret);
    _Npy_DECREF(prod);

    return (PyObject *)ret;

 fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    _Npy_XDECREF(prod);
    return NULL;
}


/*NUMPY_API
 *Numeric.matrixproduct(a,v)
 * just like inner product but does the swapaxes stuff on the fly
 */
NPY_NO_EXPORT PyObject *
PyArray_MatrixProduct(PyObject *op1, PyObject *op2)
{
    PyArrayObject *ap1, *ap2, *ret = NULL;
    int typenum;
    PyArray_Descr *typec;

    typenum = PyArray_ObjectType(op1, 0);
    typenum = PyArray_ObjectType(op2, typenum);
    typec = PyArray_DescrFromType(typenum);
    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 0, 0, ALIGNED, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 0, 0, ALIGNED, NULL);
    if (ap2 == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(ap1) == 0 || PyArray_NDIM(ap2) == 0) {
        ret = (PyArray_NDIM(ap1) == 0 ? ap1 : ap2);
        ret = (PyArrayObject *)Py_TYPE(ret)->tp_as_number->nb_multiply(
                                        (PyObject *)ap1, (PyObject *)ap2);
        Py_DECREF(ap1);
        Py_DECREF(ap2);
        return (PyObject *)ret;
    }

    /* TODO: Wrap return value. */
    ASSIGN_TO_PYARRAY(ret,
                      NpyArray_MatrixProduct(PyArray_ARRAY(ap1),
                                             PyArray_ARRAY(ap2), typenum));
    Py_DECREF(ap1);
    Py_DECREF(ap2);
    return (PyObject *)ret;

 fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ret);
    return NULL;
}


/*NUMPY_API
 * Fast Copy and Transpose
 */
NPY_NO_EXPORT PyObject *
PyArray_CopyAndTranspose(PyObject *op)
{
    PyArrayObject *ret, *arr;

    /* make sure it is well-behaved */
    arr = (PyArrayObject *)PyArray_FromAny(op, NULL, 0, 0, CARRAY, NULL);
    if (arr == NULL) {
        return NULL;
    }

    ASSIGN_TO_PYARRAY(ret,
                      NpyArray_CopyAndTranspose(PyArray_ARRAY(arr)));
    Py_DECREF(arr);
    if (ret == NULL) {
        return NULL;
    }
    return (PyObject *)ret;
}


/*NUMPY_API
 * correlate(a1, a2, mode)
 *
 * This function computes the usual correlation (correlate(a1, a2) !=
 * correlate(a2, a1), and conjugate the second argument for complex inputs
 */
NPY_NO_EXPORT PyObject *
PyArray_Correlate2(PyObject *op1, PyObject *op2, int mode)
{
    PyArrayObject *ap1, *ap2;
    PyObject *ret = NULL;
    int typenum;
    PyArray_Descr *typec;

    typenum = PyArray_ObjectType(op1, 0);
    typenum = PyArray_ObjectType(op2, typenum);
    typec = PyArray_DescrFromType(typenum);
    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 1, 1, DEFAULT, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 1, 1, DEFAULT, NULL);
    if (ap2 == NULL) {
        goto fail_clean_ap1;
    }

    ASSIGN_TO_PYARRAY(ret,
                      NpyArray_Correlate2(PyArray_ARRAY(ap1),
                                          PyArray_ARRAY(ap2), typenum, mode));
    Py_DECREF(ap1);
    Py_DECREF(ap2);
    return ret;

fail_clean_ap1:
    Py_DECREF(ap1);
    return NULL;
}


/*NUMPY_API
 * Numeric.correlate(a1, a2, mode)
 */
NPY_NO_EXPORT PyObject *
PyArray_Correlate(PyObject *op1, PyObject *op2, int mode)
{
    PyArrayObject *ap1, *ap2;
    PyObject *ret = NULL;
    int typenum;
    PyArray_Descr *typec;

    typenum = PyArray_ObjectType(op1, 0);
    typenum = PyArray_ObjectType(op2, typenum);
    typec = PyArray_DescrFromType(typenum);
    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 1, 1, DEFAULT, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 1, 1, DEFAULT, NULL);
    if (ap2 == NULL) {
        Py_DECREF(typec);
        goto fail;
    }

    ASSIGN_TO_PYARRAY(ret,
                      NpyArray_Correlate(PyArray_ARRAY(ap1),
                                         PyArray_ARRAY(ap2), typenum, mode));
    Py_DECREF(ap1);
    Py_DECREF(ap2);
    return (PyObject *)ret;

fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ret);
    return NULL;
}


static PyObject *
array_putmask(PyObject *NPY_UNUSED(module), PyObject *args, PyObject *kwds)
{
    PyObject *mask, *values;
    PyObject *array;

    static char *kwlist[] = {"arr", "mask", "values", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!OO:putmask", kwlist,
                &PyArray_Type, &array, &mask, &values)) {
        return NULL;
    }
    return PyArray_PutMask((PyArrayObject *)array, values, mask);
}


/*NUMPY_API
 * Convert an object to FORTRAN / C / ANY
 */
NPY_NO_EXPORT int
PyArray_OrderConverter(PyObject *object, NPY_ORDER *val)
{
    char *str;
    if (object == NULL || object == Py_None) {
        *val = PyArray_ANYORDER;
    }
    else if (PyUnicode_Check(object)) {
        PyObject *tmp;
        int ret;
        tmp = PyUnicode_AsASCIIString(object);
        ret = PyArray_OrderConverter(tmp, val);
        Py_DECREF(tmp);
        return ret;
    }
    else if (!PyBytes_Check(object) || PyBytes_GET_SIZE(object) < 1) {
        if (PyObject_IsTrue(object)) {
            *val = PyArray_FORTRANORDER;
        }
        else {
            *val = PyArray_CORDER;
        }
        if (PyErr_Occurred()) {
            return PY_FAIL;
        }
        return PY_SUCCEED;
    }
    else {
        str = PyBytes_AS_STRING(object);
        if (str[0] == 'C' || str[0] == 'c') {
            *val = PyArray_CORDER;
        }
        else if (str[0] == 'F' || str[0] == 'f') {
            *val = PyArray_FORTRANORDER;
        }
        else if (str[0] == 'A' || str[0] == 'a') {
            *val = PyArray_ANYORDER;
        }
        else {
            PyErr_SetString(PyExc_TypeError, "order not understood");
            return PY_FAIL;
        }
    }
    return PY_SUCCEED;
}


/*NUMPY_API
 * Convert an object to NPY_RAISE / NPY_CLIP / NPY_WRAP
 */
NPY_NO_EXPORT int
PyArray_ClipmodeConverter(PyObject *object, NPY_CLIPMODE *val)
{
    if (object == NULL || object == Py_None) {
        *val = NPY_RAISE;
    }
    else if (PyBytes_Check(object)) {
        char *str;
        str = PyBytes_AS_STRING(object);
        if (str[0] == 'C' || str[0] == 'c') {
            *val = NPY_CLIP;
        }
        else if (str[0] == 'W' || str[0] == 'w') {
            *val = NPY_WRAP;
        }
        else if (str[0] == 'R' || str[0] == 'r') {
            *val = NPY_RAISE;
        }
        else {
            PyErr_SetString(PyExc_TypeError, "clipmode not understood");
            return PY_FAIL;
        }
    }
    else if (PyUnicode_Check(object)) {
        PyObject *tmp;
        int ret;
        tmp = PyUnicode_AsASCIIString(object);
        ret = PyArray_ClipmodeConverter(tmp, val);
        Py_DECREF(tmp);
        return ret;
    }
    else {
        int number = PyInt_AsLong(object);
        if (number == -1 && PyErr_Occurred()) {
            goto fail;
        }
        if (number <= (int) NPY_RAISE && number >= (int) NPY_CLIP) {
            *val = (NPY_CLIPMODE) number;
        }
        else {
            goto fail;
        }
    }
    return PY_SUCCEED;

 fail:
    PyErr_SetString(PyExc_TypeError, "clipmode not understood");
    return PY_FAIL;
}


/*NUMPY_API
 *
 * This function returns true if the two typecodes are
 * equivalent (same basic kind and same itemsize).
 */
NPY_NO_EXPORT unsigned char
PyArray_EquivTypes(PyArray_Descr *typ1, PyArray_Descr *typ2)
{
    return NpyArray_EquivTypes(typ1->descr, typ2->descr);
}


/*NUMPY_API*/
NPY_NO_EXPORT unsigned char
PyArray_EquivTypenums(int typenum1, int typenum2)
{
    return NpyArray_EquivTypenums(typenum1, typenum2);
}

/*** END C-API FUNCTIONS (actually not quite, there are a few more below) ***/

static PyObject *
_prepend_ones(PyArrayObject *arr, int nd, int ndmin)
{
    intp newdims[MAX_DIMS];
    intp newstrides[MAX_DIMS];
    int i, k, num;
    PyArrayObject *ret;

    num = ndmin - nd;
    for (i = 0; i < num; i++) {
        newdims[i] = 1;
        newstrides[i] = PyArray_ITEMSIZE(arr);
    }
    for (i = num; i < ndmin; i++) {
        k = i - num;
        newdims[i] = PyArray_DIM(arr, k);
        newstrides[i] = PyArray_STRIDE(arr, k);
    }
    _Npy_INCREF(PyArray_DESCR(arr));
    ASSIGN_TO_PYARRAY(ret, 
                      NpyArray_NewFromDescr(PyArray_DESCR(arr), ndmin,
                                            newdims, newstrides, 
                                            PyArray_BYTES(arr), 
                                            PyArray_FLAGS(arr), 
                                            NPY_FALSE, NULL, arr));
    PyArray_BASE_ARRAY(ret) = PyArray_ARRAY(arr);
    _Npy_INCREF(PyArray_ARRAY(arr));
    Py_DECREF(arr);
    assert(NULL == PyArray_BASE_ARRAY(ret) || NULL == PyArray_BASE(ret));
    return (PyObject *)ret;
}


#define _ARET(x) PyArray_Return((PyArrayObject *)(x))

#define STRIDING_OK(op, order) ((order) == PyArray_ANYORDER ||          \
                                ((order) == PyArray_CORDER &&           \
                                 PyArray_ISCONTIGUOUS(op)) ||           \
                                ((order) == PyArray_FORTRANORDER &&     \
                                 PyArray_ISFORTRAN(op)))

static PyObject *
_array_fromobject(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kws)
{
    PyObject *op, *ret = NULL;
    static char *kwd[]= {"object", "dtype", "copy", "order", "subok",
                         "ndmin", NULL};
    Bool subok = FALSE;
    Bool copy = TRUE;
    int ndmin = 0, nd;
    PyArray_Descr *type = NULL;
    NpyArray_Descr *oldtype = NULL;
    NPY_ORDER order=PyArray_ANYORDER;
    int flags = 0;

    if (PyTuple_GET_SIZE(args) > 2) {
        PyErr_SetString(PyExc_ValueError,
                        "only 2 non-keyword arguments accepted");
        return NULL;
    }
    if(!PyArg_ParseTupleAndKeywords(args, kws, "O|O&O&O&O&i", kwd, &op,
                PyArray_DescrConverter2, &type,
                PyArray_BoolConverter, &copy,
                PyArray_OrderConverter, &order,
                PyArray_BoolConverter, &subok,
                &ndmin)) {
        goto clean_type;
    }

    if (ndmin > NPY_MAXDIMS) {
        PyErr_Format(PyExc_ValueError,
                "ndmin bigger than allowable number of dimensions "\
                "NPY_MAXDIMS (=%d)", NPY_MAXDIMS);
        goto clean_type;
    }
    /* fast exit if simple call */
    if ((subok && PyArray_Check(op))
            || (!subok && PyArray_CheckExact(op))) {
        if (type == NULL) {
            if (!copy && STRIDING_OK(op, order)) {
                Py_INCREF(op);
                ret = op;
                goto finish;
            }
            else {
                ret = PyArray_NewCopy((PyArrayObject*)op, order);
                goto finish;
            }
        }
        /* One more chance */
        oldtype = PyArray_DESCR((PyArrayObject *)op);
        if (NpyArray_EquivTypes(oldtype, type->descr)) {
            if (!copy && STRIDING_OK(op, order)) {
                Py_INCREF(op);
                ret = op;
                goto finish;
            }
            else {
                ret = PyArray_NewCopy((PyArrayObject*)op, order);
                if (oldtype == type->descr) {
                    goto finish;
                }
                _Npy_INCREF(oldtype);
                _Npy_DECREF(PyArray_DESCR(ret));
                PyArray_DESCR(ret) = oldtype;
                goto finish;
            }
        }
    }

    if (copy) {
        flags = ENSURECOPY;
    }
    if (order == PyArray_CORDER) {
        flags |= CONTIGUOUS;
    }
    else if ((order == PyArray_FORTRANORDER)
             /* order == PyArray_ANYORDER && */
             || (PyArray_Check(op) && PyArray_ISFORTRAN(op))) {
        flags |= FORTRAN;
    }
    if (!subok) {
        flags |= ENSUREARRAY;
    }

    flags |= NPY_FORCECAST;
    Py_XINCREF(type);
    ret = PyArray_CheckFromAny(op, type, 0, 0, flags, NULL);

 finish:
    Py_XDECREF(type);
    if (!ret || (nd=PyArray_NDIM(ret)) >= ndmin) {
        return ret;
    }
    /*
     * create a new array from the same data with ones in the shape
     * steals a reference to ret
     */
    return _prepend_ones((PyArrayObject *)ret, nd, ndmin);

clean_type:
    Py_XDECREF(type);
    return NULL;
}

static PyObject *
array_empty(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{

    static char *kwlist[] = {"shape","dtype","order",NULL};
    PyArray_Descr *typecode = NULL;
    PyArray_Dims shape = {NULL, 0};
    NPY_ORDER order = PyArray_CORDER;
    Bool fortran;
    PyObject *ret = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&", kwlist,
                PyArray_IntpConverter, &shape,
                PyArray_DescrConverter, &typecode,
                PyArray_OrderConverter, &order)) {
        goto fail;
    }
    if (order == PyArray_FORTRANORDER) {
        fortran = TRUE;
    }
    else {
        fortran = FALSE;
    }
    ret = PyArray_Empty(shape.len, shape.ptr, typecode, fortran);
    PyDimMem_FREE(shape.ptr);
    return ret;

 fail:
    Py_XDECREF(typecode);
    PyDimMem_FREE(shape.ptr);
    return NULL;
}

/*
 * This function is needed for supporting Pickles of
 * numpy scalar objects.
 */
static PyObject *
array_scalar(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{

    static char *kwlist[] = {"dtype","obj", NULL};
    PyArray_Descr *typecode;
    PyObject *obj = NULL;
    int alloc = 0;
    void *dptr;
    PyObject *ret;


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|O", kwlist,
                &PyArrayDescr_Type, &typecode, &obj)) {
        return NULL;
    }
    if (typecode->descr->elsize == 0) {
        PyErr_SetString(PyExc_ValueError,
                "itemsize cannot be zero");
        return NULL;
    }

    if (PyDataType_FLAGCHK(typecode, NPY_ITEM_IS_POINTER)) {
        if (obj == NULL) {
            obj = Py_None;
        }
        dptr = &obj;
    }
    else {
        if (obj == NULL) {
            dptr = _pya_malloc(typecode->descr->elsize);
            if (dptr == NULL) {
                return PyErr_NoMemory();
            }
            memset(dptr, '\0', typecode->descr->elsize);
            alloc = 1;
        }
        else {
            if (!PyString_Check(obj)) {
                PyErr_SetString(PyExc_TypeError,
                        "initializing object must be a string");
                return NULL;
            }
            if (PyString_GET_SIZE(obj) < typecode->descr->elsize) {
                PyErr_SetString(PyExc_ValueError,
                        "initialization string is too small");
                return NULL;
            }
            dptr = PyString_AS_STRING(obj);
        }
    }
    ret = PyArray_Scalar(dptr, typecode, NULL);

    /* free dptr which contains zeros */
    if (alloc) {
        _pya_free(dptr);
    }
    return ret;
}

static PyObject *
array_zeros(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"shape","dtype","order",NULL}; /* XXX ? */
    PyArray_Descr *typecode = NULL;
    PyArray_Dims shape = {NULL, 0};
    NPY_ORDER order = PyArray_CORDER;
    Bool fortran = FALSE;
    PyObject *ret = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&", kwlist,
                PyArray_IntpConverter, &shape,
                PyArray_DescrConverter, &typecode,
                PyArray_OrderConverter, &order)) {
        goto fail;
    }
    if (order == PyArray_FORTRANORDER) {
        fortran = TRUE;
    }
    else {
        fortran = FALSE;
    }
    ret = PyArray_Zeros(shape.len, shape.ptr, typecode, (int) fortran);
    PyDimMem_FREE(shape.ptr);
    return ret;

 fail:
    Py_XDECREF(typecode);
    PyDimMem_FREE(shape.ptr);
    return ret;
}

static PyObject *
array_fromstring(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *keywds)
{
    char *data;
    Py_ssize_t nin = -1;
    char *sep = NULL;
    Py_ssize_t s;
    static char *kwlist[] = {"string", "dtype", "count", "sep", NULL};
    PyArray_Descr *descr = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, keywds,
                "s#|O&" NPY_SSIZE_T_PYFMT "s", kwlist,
                &data, &s, PyArray_DescrConverter, &descr, &nin, &sep)) {
        Py_XDECREF(descr);
        return NULL;
    }
    return PyArray_FromString(data, (intp)s, descr, (intp)nin, sep);
}



static PyObject *
array_fromfile(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *keywds)
{
    PyObject *file = NULL, *ret;
    FILE *fp;
    char *sep = "";
    Py_ssize_t nin = -1;
    static char *kwlist[] = {"file", "dtype", "count", "sep", NULL};
    PyArray_Descr *type = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, keywds,
                "O|O&" NPY_SSIZE_T_PYFMT "s", kwlist,
                &file, PyArray_DescrConverter, &type, &nin, &sep)) {
        Py_XDECREF(type);
        return NULL;
    }
    if (PyString_Check(file) || PyUnicode_Check(file)) {
        file = npy_PyFile_OpenFile(file, "rb");
        if (file == NULL) {
            return NULL;
        }
    }
    else {
        Py_INCREF(file);
    }
#if defined(NPY_PY3K)
    fp = npy_PyFile_Dup(file, "rb");
#else
    fp = PyFile_AsFile(file);
#endif
    if (fp == NULL) {
        PyErr_SetString(PyExc_IOError,
                "first argument must be an open file");
        Py_DECREF(file);
        return NULL;
    }
    if (type == NULL) {
        type = PyArray_DescrFromType(PyArray_DEFAULT);
    }
    ret = PyArray_FromFile(fp, type, (intp) nin, sep);
#if defined(NPY_PY3K)
    fclose(fp);
#endif
    Py_DECREF(file);
    return ret;
}

static PyObject *
array_fromiter(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *keywds)
{
    PyObject *iter;
    Py_ssize_t nin = -1;
    static char *kwlist[] = {"iter", "dtype", "count", NULL};
    PyArray_Descr *descr = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, keywds,
                "OO&|" NPY_SSIZE_T_PYFMT, kwlist,
                &iter, PyArray_DescrConverter, &descr, &nin)) {
        Py_XDECREF(descr);
        return NULL;
    }
    return PyArray_FromIter(iter, descr, (intp)nin);
}

static PyObject *
array_frombuffer(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *keywds)
{
    PyObject *obj = NULL;
    Py_ssize_t nin = -1, offset = 0;
    static char *kwlist[] = {"buffer", "dtype", "count", "offset", NULL};
    PyArray_Descr *type = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, keywds,
                "O|O&" NPY_SSIZE_T_PYFMT NPY_SSIZE_T_PYFMT, kwlist,
                &obj, PyArray_DescrConverter, &type, &nin, &offset)) {
        Py_XDECREF(type);
        return NULL;
    }
    if (type == NULL) {
        type = PyArray_DescrFromType(PyArray_DEFAULT);
    }
    return PyArray_FromBuffer(obj, type, (intp)nin, (intp)offset);
}

static PyObject *
array_concatenate(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *a0;
    int axis = 0;
    static char *kwlist[] = {"seq", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&", kwlist,
                &a0, PyArray_AxisConverter, &axis)) {
        return NULL;
    }
    return PyArray_Concatenate(a0, axis);
}

static PyObject *
array_innerproduct(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *b0, *a0;

    if (!PyArg_ParseTuple(args, "OO", &a0, &b0)) {
        return NULL;
    }
    return _ARET(PyArray_InnerProduct(a0, b0));
}

static PyObject *
array_matrixproduct(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *v, *a;

    if (!PyArg_ParseTuple(args, "OO", &a, &v)) {
        return NULL;
    }
    return _ARET(PyArray_MatrixProduct(a, v));
}

static PyObject *
array_fastCopyAndTranspose(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *a0;

    if (!PyArg_ParseTuple(args, "O", &a0)) {
        return NULL;
    }
    return _ARET(PyArray_CopyAndTranspose(a0));
}

static PyObject *
array_correlate(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *shape, *a0;
    int mode = 0;
    static char *kwlist[] = {"a", "v", "mode", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|i", kwlist,
                &a0, &shape, &mode)) {
        return NULL;
    }
    return PyArray_Correlate(a0, shape, mode);
}

static PyObject*
array_correlate2(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *shape, *a0;
    int mode = 0;
    static char *kwlist[] = {"a", "v", "mode", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|i", kwlist,
                &a0, &shape, &mode)) {
        return NULL;
    }
    return PyArray_Correlate2(a0, shape, mode);
}

static PyObject *
array_arange(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kws) {
    PyObject *o_start = NULL, *o_stop = NULL, *o_step = NULL;
    static char *kwd[]= {"start", "stop", "step", "dtype", NULL};
    PyArray_Descr *typecode = NULL;
    PyObject* result;

    if(!PyArg_ParseTupleAndKeywords(args, kws, "O|OOO&", kwd,
                &o_start, &o_stop, &o_step,
                PyArray_DescrConverter2, &typecode)) {
        Py_XDECREF(typecode);
        return NULL;
    }
    result = PyArray_ArangeObj(o_start, o_stop, o_step, typecode);
    Py_XDECREF(typecode);
    return result;
}

/*NUMPY_API
 *
 * Included at the very first so not auto-grabbed and thus not labeled.
 */
NPY_NO_EXPORT unsigned int
PyArray_GetNDArrayCVersion(void)
{
    return (unsigned int)NPY_VERSION;
}

/*NUMPY_API
 * Returns the built-in (at compilation time) C API version
 */
NPY_NO_EXPORT unsigned int
PyArray_GetNDArrayCFeatureVersion(void)
{
    return (unsigned int)NPY_FEATURE_VERSION;
}

static PyObject *
array__get_ndarray_c_version(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist )) {
        return NULL;
    }
    return PyInt_FromLong( (long) PyArray_GetNDArrayCVersion() );
}


/*NUMPY_API
*/
NPY_NO_EXPORT int
PyArray_GetEndianness(void)
{
    return NpyArray_GetEndianness();
}


static PyObject *
array__reconstruct(PyObject *NPY_UNUSED(dummy), PyObject *args)
{

    PyObject *ret;
    PyTypeObject *subtype;
    PyArray_Dims shape = {NULL, 0};
    PyArray_Descr *dtype = NULL;

    if (!PyArg_ParseTuple(args, "O!O&O&",
                &PyType_Type, &subtype,
                PyArray_IntpConverter, &shape,
                PyArray_DescrConverter, &dtype)) {
        goto fail;
    }
    if (!PyType_IsSubtype(subtype, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError,
                "_reconstruct: First argument must be a sub-type of ndarray");
        goto fail;
    }
    ret = PyArray_NewFromDescr(subtype, dtype,
            (int)shape.len, shape.ptr, NULL, NULL, 0, NULL);
    if (shape.ptr) {
        PyDimMem_FREE(shape.ptr);
    }
    return ret;

 fail:
    Py_XDECREF(dtype);
    if (shape.ptr) {
        PyDimMem_FREE(shape.ptr);
    }
    return NULL;
}

static PyObject *
array_set_string_function(PyObject *NPY_UNUSED(self), PyObject *args,
        PyObject *kwds)
{
    PyObject *op = NULL;
    int repr = 1;
    static char *kwlist[] = {"f", "repr", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "|Oi", kwlist, &op, &repr)) {
        return NULL;
    }
    /* reset the array_repr function to built-in */
    if (op == Py_None) {
        op = NULL;
    }
    if (op != NULL && !PyCallable_Check(op)) {
        PyErr_SetString(PyExc_TypeError,
                "Argument must be callable.");
        return NULL;
    }
    PyArray_SetStringFunction(op, repr);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
array_set_ops_function(PyObject *NPY_UNUSED(self), PyObject *NPY_UNUSED(args),
        PyObject *kwds)
{
    PyObject *oldops = NULL;

    if ((oldops = PyArray_GetNumericOps()) == NULL) {
        return NULL;
    }
    /*
     * Should probably ensure that objects are at least callable
     *  Leave this to the caller for now --- error will be raised
     *  later when use is attempted
     */
    if (kwds && PyArray_SetNumericOps(kwds) == -1) {
        Py_DECREF(oldops);
        PyErr_SetString(PyExc_ValueError,
                "one or more objects not callable");
        return NULL;
    }
    return oldops;
}

static PyObject *
array_set_datetimeparse_function(PyObject *NPY_UNUSED(self), PyObject *args,
        PyObject *kwds)
{
    PyObject *op = NULL;
    static char *kwlist[] = {"f", NULL};
    PyObject *_numpy_internal;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &op)) {
        return NULL;
    }
    /* reset the array_repr function to built-in */
    if (op == Py_None) {
        _numpy_internal = PyImport_ImportModule("numpy.core._internal");
        if (_numpy_internal == NULL) {
            return NULL;
        }
        op = PyObject_GetAttrString(_numpy_internal, "datetime_from_string");
    }
    else { /* Must balance reference count increment in both branches */
        if (!PyCallable_Check(op)) {
            PyErr_SetString(PyExc_TypeError,
                    "Argument must be callable.");
            return NULL;
        }
        Py_INCREF(op);
    }
    PyArray_SetDatetimeParseFunction(op);
    Py_DECREF(op);
    Py_INCREF(Py_None);
    return Py_None;
}


/*NUMPY_API
 * Where
 */
NPY_NO_EXPORT PyObject *
PyArray_Where(PyObject *condition, PyObject *x, PyObject *y)
{
    PyArrayObject *arr;
    PyObject *tup = NULL, *obj = NULL;
    PyObject *ret = NULL, *zero = NULL;

    arr = (PyArrayObject *)PyArray_FromAny(condition, NULL, 0, 0, 0, NULL);
    if (arr == NULL) {
        return NULL;
    }
    if ((x == NULL) && (y == NULL)) {
        ret = PyArray_Nonzero(arr);
        Py_DECREF(arr);
        return ret;
    }
    if ((x == NULL) || (y == NULL)) {
        Py_DECREF(arr);
        PyErr_SetString(PyExc_ValueError,
                "either both or neither of x and y should be given");
        return NULL;
    }

    zero = PyInt_FromLong((long) 0);
    obj = PyArray_EnsureAnyArray(PyArray_GenericBinaryFunction(arr, zero,
                n_ops.not_equal));
    Py_DECREF(zero);
    Py_DECREF(arr);
    if (obj == NULL) {
        return NULL;
    }
    tup = Py_BuildValue("(OO)", y, x);
    if (tup == NULL) {
        Py_DECREF(obj);
        return NULL;
    }
    ret = PyArray_Choose((PyAO *)obj, tup, NULL, NPY_RAISE);
    Py_DECREF(obj);
    Py_DECREF(tup);
    return ret;
}

static PyObject *
array_where(PyObject *NPY_UNUSED(ignored), PyObject *args)
{
    PyObject *obj = NULL, *x = NULL, *y = NULL;

    if (!PyArg_ParseTuple(args, "O|OO", &obj, &x, &y)) {
        return NULL;
    }
    return PyArray_Where(obj, x, y);
}

static PyObject *
array_lexsort(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    int axis = -1;
    PyObject *obj;
    static char *kwlist[] = {"keys", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", kwlist, &obj, &axis)) {
        return NULL;
    }
    return _ARET(PyArray_LexSort(obj, axis));
}

#undef _ARET

static PyObject *
array_can_cast_safely(PyObject *NPY_UNUSED(self), PyObject *args,
        PyObject *kwds)
{
    PyArray_Descr *d1 = NULL;
    PyArray_Descr *d2 = NULL;
    Bool ret;
    PyObject *retobj = NULL;
    static char *kwlist[] = {"from", "to", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&", kwlist,
                PyArray_DescrConverter, &d1, PyArray_DescrConverter, &d2)) {
        goto finish;
    }
    if (d1 == NULL || d2 == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "did not understand one of the types; 'None' not accepted");
        goto finish;
    }

    ret = PyArray_CanCastTo(d1, d2);
    retobj = ret ? Py_True : Py_False;
    Py_INCREF(retobj);

 finish:
    Py_XDECREF(d1);
    Py_XDECREF(d2);
    return retobj;
}

#if !defined(NPY_PY3K)
static PyObject *
new_buffer(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    int size;

    if(!PyArg_ParseTuple(args, "i", &size)) {
        return NULL;
    }
    return PyBuffer_New(size);
}

static PyObject *
buffer_buffer(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    Py_ssize_t offset = 0, n;
    Py_ssize_t size = Py_END_OF_BUFFER;
    void *unused;
    static char *kwlist[] = {"object", "offset", "size", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                "O|" NPY_SSIZE_T_PYFMT NPY_SSIZE_T_PYFMT, kwlist,
                &obj, &offset, &size)) {
        return NULL;
    }
    if (PyObject_AsWriteBuffer(obj, &unused, &n) < 0) {
        PyErr_Clear();
        return PyBuffer_FromObject(obj, offset, size);
    }
    else {
        return PyBuffer_FromReadWriteObject(obj, offset, size);
    }
}
#endif

#ifndef _MSC_VER
#include <setjmp.h>
#include <signal.h>
jmp_buf _NPY_SIGSEGV_BUF;
static void
_SigSegv_Handler(int signum)
{
    longjmp(_NPY_SIGSEGV_BUF, signum);
}
#endif

#define _test_code() {                          \
        test = *((char*)memptr);                \
        if (!ro) {                              \
            *((char *)memptr) = '\0';           \
            *((char *)memptr) = test;           \
        }                                       \
        test = *((char*)memptr+size-1);         \
        if (!ro) {                              \
            *((char *)memptr+size-1) = '\0';    \
            *((char *)memptr+size-1) = test;    \
        }                                       \
    }

static PyObject *
as_buffer(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *mem;
    Py_ssize_t size;
    Bool ro = FALSE, check = TRUE;
    void *memptr;
    static char *kwlist[] = {"mem", "size", "readonly", "check", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                "O" NPY_SSIZE_T_PYFMT "|O&O&", kwlist,
                &mem, &size, PyArray_BoolConverter, &ro,
                PyArray_BoolConverter, &check)) {
        return NULL;
    }
    memptr = PyLong_AsVoidPtr(mem);
    if (memptr == NULL) {
        return NULL;
    }
    if (check) {
        /*
         * Try to dereference the start and end of the memory region
         * Catch segfault and report error if it occurs
         */
        char test;
        int err = 0;

#ifdef _MSC_VER
        __try {
            _test_code();
        }
        __except(1) {
            err = 1;
        }
#else
        PyOS_sighandler_t _npy_sig_save;
        _npy_sig_save = PyOS_setsig(SIGSEGV, _SigSegv_Handler);
        if (setjmp(_NPY_SIGSEGV_BUF) == 0) {
            _test_code();
        }
        else {
            err = 1;
        }
        PyOS_setsig(SIGSEGV, _npy_sig_save);
#endif
        if (err) {
            PyErr_SetString(PyExc_ValueError,
                    "cannot use memory location as a buffer.");
            return NULL;
        }
    }


#if defined(NPY_PY3K)
    PyErr_SetString(PyExc_RuntimeError,
            "XXX -- not implemented!");
    return NULL;
#else
    if (ro) {
        return PyBuffer_FromMemory(memptr, size);
    }
    return PyBuffer_FromReadWriteMemory(memptr, size);
#endif
}

#undef _test_code

static PyObject *
format_longfloat(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    unsigned int precision;
    longdouble x;
    static char *kwlist[] = {"x", "precision", NULL};
    static char repr[100];

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OI", kwlist,
                &obj, &precision)) {
        return NULL;
    }
    if (!PyArray_IsScalar(obj, LongDouble)) {
        PyErr_SetString(PyExc_TypeError,
                "not a longfloat");
        return NULL;
    }
    x = ((PyLongDoubleScalarObject *)obj)->obval;
    if (precision > 70) {
        precision = 70;
    }
    format_longdouble(repr, 100, x, precision);
    return PyUString_FromString(repr);
}

static PyObject *
compare_chararrays(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *array;
    PyObject *other;
    PyArrayObject *newarr, *newoth;
    int cmp_op;
    Bool rstrip;
    char *cmp_str;
    Py_ssize_t strlen;
    PyObject *res = NULL;
    static char msg[] = "comparision must be '==', '!=', '<', '>', '<=', '>='";
    static char *kwlist[] = {"a1", "a2", "cmp", "rstrip", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOs#O&", kwlist,
                &array, &other, &cmp_str, &strlen,
                PyArray_BoolConverter, &rstrip)) {
        return NULL;
    }
    if (strlen < 1 || strlen > 2) {
        goto err;
    }
    if (strlen > 1) {
        if (cmp_str[1] != '=') {
            goto err;
        }
        if (cmp_str[0] == '=') {
            cmp_op = Py_EQ;
        }
        else if (cmp_str[0] == '!') {
            cmp_op = Py_NE;
        }
        else if (cmp_str[0] == '<') {
            cmp_op = Py_LE;
        }
        else if (cmp_str[0] == '>') {
            cmp_op = Py_GE;
        }
        else {
            goto err;
        }
    }
    else {
        if (cmp_str[0] == '<') {
            cmp_op = Py_LT;
        }
        else if (cmp_str[0] == '>') {
            cmp_op = Py_GT;
        }
        else {
            goto err;
        }
    }

    newarr = (PyArrayObject *)PyArray_FROM_O(array);
    if (newarr == NULL) {
        return NULL;
    }
    newoth = (PyArrayObject *)PyArray_FROM_O(other);
    if (newoth == NULL) {
        Py_DECREF(newarr);
        return NULL;
    }
    if (PyArray_ISSTRING(newarr) && PyArray_ISSTRING(newoth)) {
        res = _strings_richcompare(newarr, newoth, cmp_op, rstrip != 0);
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                "comparison of non-string arrays");
    }
    Py_DECREF(newarr);
    Py_DECREF(newoth);
    return res;

 err:
    PyErr_SetString(PyExc_ValueError, msg);
    return NULL;
}

static PyObject *
_vec_string_with_args(PyArrayObject* char_array, PyArray_Descr* type,
                      PyObject* method, PyObject* args)
{
    PyObject* broadcast_args[NPY_MAXARGS];
    PyArrayMultiIterObject* in_iter = NULL;
    PyArrayObject* result = NULL;
    PyArrayIterObject* out_iter = NULL;
    PyObject* args_tuple = NULL;
    Py_ssize_t i, n, nargs;

    nargs = PySequence_Size(args) + 1;
    if (nargs == -1 || nargs > NPY_MAXARGS) {
        PyErr_Format(PyExc_ValueError,
                "len(args) must be < %d", NPY_MAXARGS - 1);
        Py_DECREF(type);
        goto err;
    }

    broadcast_args[0] = (PyObject*)char_array;
    for (i = 1; i < nargs; i++) {
        PyObject* item = PySequence_GetItem(args, i-1);
        if (item == NULL) {
            Py_DECREF(type);
            goto err;
        }
        broadcast_args[i] = item;
        Py_DECREF(item);
    }
    in_iter = (PyArrayMultiIterObject*)PyArray_MultiIterFromObjects
        (broadcast_args, nargs, 0);
    if (in_iter == NULL) {
        Py_DECREF(type);
        goto err;
    }
    n = in_iter->iter->numiter;

    result = (PyArrayObject*)PyArray_SimpleNewFromDescr(in_iter->iter->nd,
            in_iter->iter->dimensions, type);
    if (result == NULL) {
        goto err;
    }

    out_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)result);
    if (out_iter == NULL) {
        goto err;
    }

    args_tuple = PyTuple_New(n);
    if (args_tuple == NULL) {
        goto err;
    }

    while (PyArray_MultiIter_NOTDONE(in_iter)) {
        PyObject* item_result;

        for (i = 0; i < n; i++) {
            NpyArrayIterObject* it = in_iter->iter->iters[i];
            PyObject* arg = PyArray_ToScalar(NpyArray_ITER_DATA(it), 
                                             Npy_INTERFACE(it->ao));
            if (arg == NULL) {
                goto err;
            }
            /* Steals ref to arg */
            PyTuple_SetItem(args_tuple, i, arg);
        }

        item_result = PyObject_CallObject(method, args_tuple);
        if (item_result == NULL) {
            goto err;
        }

        if (PyArray_SETITEM(result, PyArray_ITER_DATA(out_iter), 
                            item_result)) {
            Py_DECREF(item_result);
            PyErr_SetString( PyExc_TypeError,
                    "result array type does not match underlying function");
            goto err;
        }
        Py_DECREF(item_result);

        PyArray_MultiIter_NEXT(in_iter);
        PyArray_ITER_NEXT(out_iter);
    }

    Py_DECREF(in_iter);
    Py_DECREF(out_iter);
    Py_DECREF(args_tuple);

    return (PyObject*)result;

 err:
    Py_XDECREF(in_iter);
    Py_XDECREF(out_iter);
    Py_XDECREF(args_tuple);
    Py_XDECREF(result);

    return 0;
}

static PyObject *
_vec_string_no_args(PyArrayObject* char_array,
                                   PyArray_Descr* type, PyObject* method)
{
    /*
     * This is a faster version of _vec_string_args to use when there
     * are no additional arguments to the string method.  This doesn't
     * require a broadcast iterator (and broadcast iterators don't work
     * with 1 argument anyway).
     */
    NpyArrayIterObject* in_iter = NULL;
    PyArrayObject* result = NULL;
    NpyArrayIterObject* out_iter = NULL;

    in_iter = NpyArray_IterNew(PyArray_ARRAY(char_array));
    if (in_iter == NULL) {
        Py_DECREF(type);
        goto err;
    }

    result = (PyArrayObject*)PyArray_SimpleNewFromDescr(
            PyArray_NDIM(char_array), PyArray_DIMS(char_array), type);
    if (result == NULL) {
        goto err;
    }

    out_iter = NpyArray_IterNew(PyArray_ARRAY(result));
    if (out_iter == NULL) {
        goto err;
    }

    while (NpyArray_ITER_NOTDONE(in_iter)) {
        PyObject* item_result;
        PyObject* item = PyArray_ToScalar(in_iter->dataptr, 
                                          Npy_INTERFACE(in_iter->ao));
        if (item == NULL) {
            goto err;
        }

        item_result = PyObject_CallFunctionObjArgs(method, item, NULL);
        Py_DECREF(item);
        if (item_result == NULL) {
            goto err;
        }

        if (PyArray_SETITEM(result, NpyArray_ITER_DATA(out_iter), 
                            item_result)) {
            Py_DECREF(item_result);
            PyErr_SetString( PyExc_TypeError,
                "result array type does not match underlying function");
            goto err;
        }
        Py_DECREF(item_result);

        NpyArray_ITER_NEXT(in_iter);
        NpyArray_ITER_NEXT(out_iter);
    }

    _Npy_DECREF(in_iter);
    _Npy_DECREF(out_iter);

    return (PyObject*)result;

 err:
    _Npy_XDECREF(in_iter);
    _Npy_XDECREF(out_iter);
    Py_XDECREF(result);

    return 0;
}

static PyObject *
_vec_string(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyArrayObject* char_array = NULL;
    PyArray_Descr *type = NULL;
    PyObject* method_name;
    PyObject* args_seq = NULL;

    PyObject* method = NULL;
    PyObject* result = NULL;

    if (!PyArg_ParseTuple(args, "O&O&O|O",
                PyArray_Converter, &char_array,
                PyArray_DescrConverter, &type,
                &method_name, &args_seq)) {
        Py_XDECREF(type);
        goto err;
    }

    if (PyArray_TYPE(char_array) == NPY_STRING) {
        method = PyObject_GetAttr((PyObject *)&PyString_Type, method_name);
    }
    else if (PyArray_TYPE(char_array) == NPY_UNICODE) {
        method = PyObject_GetAttr((PyObject *)&PyUnicode_Type, method_name);
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                "string operation on non-string array");
        Py_DECREF(type);
        goto err;
    }
    if (method == NULL) {
        Py_DECREF(type);
        goto err;
    }

    if (args_seq == NULL
            || (PySequence_Check(args_seq) && PySequence_Size(args_seq) == 0)) {
        result = _vec_string_no_args(char_array, type, method);
    }
    else if (PySequence_Check(args_seq)) {
        result = _vec_string_with_args(char_array, type, method, args_seq);
    }
    else {
        Py_DECREF(type);
        PyErr_SetString(PyExc_TypeError,
                "'args' must be a sequence of arguments");
        goto err;
    }
    if (result == NULL) {
        goto err;
    }

    Py_DECREF(char_array);
    Py_DECREF(method);

    return (PyObject*)result;

 err:
    Py_XDECREF(char_array);
    Py_XDECREF(method);

    return 0;
}

#ifndef __NPY_PRIVATE_NO_SIGNAL

SIGJMP_BUF _NPY_SIGINT_BUF;


/*NUMPY_API
 */
NPY_NO_EXPORT void
_PyArray_SigintHandler(int signum)
{
    PyOS_setsig(signum, SIG_IGN);
    SIGLONGJMP(_NPY_SIGINT_BUF, signum);
}


/*NUMPY_API
 */
NPY_NO_EXPORT void*
_PyArray_GetSigintBuf(void)
{
    return (void *)&_NPY_SIGINT_BUF;
}

#else

NPY_NO_EXPORT void
_PyArray_SigintHandler(int signum)
{
    return;
}

NPY_NO_EXPORT void*
_PyArray_GetSigintBuf(void)
{
    return NULL;
}

#endif


static PyObject *
test_interrupt(PyObject *NPY_UNUSED(self), PyObject *args)
{
    int kind = 0;
    int a = 0;

    if (!PyArg_ParseTuple(args, "|i", &kind)) {
        return NULL;
    }
    if (kind) {
        Py_BEGIN_ALLOW_THREADS;
        while (a >= 0) {
            if ((a % 1000 == 0) && PyOS_InterruptOccurred()) {
                break;
            }
            a += 1;
        }
        Py_END_ALLOW_THREADS;
    }
    else {
        NPY_SIGINT_ON
        while(a >= 0) {
            a += 1;
        }
        NPY_SIGINT_OFF
    }
    return PyInt_FromLong(a);
}

static struct PyMethodDef array_module_methods[] = {
    {"_get_ndarray_c_version",
        (PyCFunction)array__get_ndarray_c_version,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"_reconstruct",
        (PyCFunction)array__reconstruct,
        METH_VARARGS, NULL},
    {"set_string_function",
        (PyCFunction)array_set_string_function,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_numeric_ops",
        (PyCFunction)array_set_ops_function,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_datetimeparse_function",
        (PyCFunction)array_set_datetimeparse_function,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_typeDict",
        (PyCFunction)array_set_typeDict,
        METH_VARARGS, NULL},
    {"array",
        (PyCFunction)_array_fromobject,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"arange",
        (PyCFunction)array_arange,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"zeros",
        (PyCFunction)array_zeros,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"empty",
        (PyCFunction)array_empty,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"scalar",
        (PyCFunction)array_scalar,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"where",
        (PyCFunction)array_where,
        METH_VARARGS, NULL},
    {"lexsort",
        (PyCFunction)array_lexsort,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"putmask",
        (PyCFunction)array_putmask,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"fromstring",
        (PyCFunction)array_fromstring,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"fromiter",
        (PyCFunction)array_fromiter,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"concatenate",
        (PyCFunction)array_concatenate,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"inner",
        (PyCFunction)array_innerproduct,
        METH_VARARGS, NULL},
    {"dot",
        (PyCFunction)array_matrixproduct,
        METH_VARARGS, NULL},
    {"_fastCopyAndTranspose",
        (PyCFunction)array_fastCopyAndTranspose,
        METH_VARARGS, NULL},
    {"correlate",
        (PyCFunction)array_correlate,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"correlate2",
        (PyCFunction)array_correlate2,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"frombuffer",
        (PyCFunction)array_frombuffer,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"fromfile",
        (PyCFunction)array_fromfile,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"can_cast",
        (PyCFunction)array_can_cast_safely,
        METH_VARARGS | METH_KEYWORDS, NULL},
#if !defined(NPY_PY3K)
    {"newbuffer",
        (PyCFunction)new_buffer,
        METH_VARARGS, NULL},
    {"getbuffer",
        (PyCFunction)buffer_buffer,
        METH_VARARGS | METH_KEYWORDS, NULL},
#endif
    {"int_asbuffer",
        (PyCFunction)as_buffer,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"format_longfloat",
        (PyCFunction)format_longfloat,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"compare_chararrays",
        (PyCFunction)compare_chararrays,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"_vec_string",
        (PyCFunction)_vec_string,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"test_interrupt",
        (PyCFunction)test_interrupt,
        METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}                /* sentinel */
};

#include "__multiarray_api.c"

/* Establish scalar-type hierarchy
 *
 *  For dual inheritance we need to make sure that the objects being
 *  inherited from have the tp->mro object initialized.  This is
 *  not necessarily true for the basic type objects of Python (it is
 *  checked for single inheritance but not dual in PyType_Ready).
 *
 *  Thus, we call PyType_Ready on the standard Python Types, here.
 */
static int
setup_scalartypes(PyObject *NPY_UNUSED(dict))
{
    initialize_numeric_types();

    if (PyType_Ready(&PyBool_Type) < 0) {
        return -1;
    }
#if !defined(NPY_PY3K)
    if (PyType_Ready(&PyInt_Type) < 0) {
        return -1;
    }
#endif
    if (PyType_Ready(&PyFloat_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyComplex_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyString_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyUnicode_Type) < 0) {
        return -1;
    }

#define SINGLE_INHERIT(child, parent)                                   \
    Py##child##ArrType_Type.tp_base = &Py##parent##ArrType_Type;        \
    if (PyType_Ready(&Py##child##ArrType_Type) < 0) {                   \
        PyErr_Print();                                                  \
        PyErr_Format(PyExc_SystemError,                                 \
                     "could not initialize Py%sArrType_Type",           \
                     #child);                                           \
        return -1;                                                      \
    }

    if (PyType_Ready(&PyGenericArrType_Type) < 0) {
        return -1;
    }
    SINGLE_INHERIT(Number, Generic);
    SINGLE_INHERIT(Integer, Number);
    SINGLE_INHERIT(Inexact, Number);
    SINGLE_INHERIT(SignedInteger, Integer);
    SINGLE_INHERIT(UnsignedInteger, Integer);
    SINGLE_INHERIT(Floating, Inexact);
    SINGLE_INHERIT(ComplexFloating, Inexact);
    SINGLE_INHERIT(Flexible, Generic);
    SINGLE_INHERIT(Character, Flexible);

#define DUAL_INHERIT(child, parent1, parent2)                           \
    Py##child##ArrType_Type.tp_base = &Py##parent2##ArrType_Type;       \
    Py##child##ArrType_Type.tp_bases =                                  \
        Py_BuildValue("(OO)", &Py##parent2##ArrType_Type,               \
                      &Py##parent1##_Type);                             \
    if (PyType_Ready(&Py##child##ArrType_Type) < 0) {                   \
        PyErr_Print();                                                  \
        PyErr_Format(PyExc_SystemError,                                 \
                     "could not initialize Py%sArrType_Type",           \
                     #child);                                           \
        return -1;                                                      \
    }                                                                   \
    Py##child##ArrType_Type.tp_hash = Py##parent1##_Type.tp_hash;

#if defined(NPY_PY3K)
#define DUAL_INHERIT_COMPARE(child, parent1, parent2)
#else
#define DUAL_INHERIT_COMPARE(child, parent1, parent2)                   \
    Py##child##ArrType_Type.tp_compare =                                \
        Py##parent1##_Type.tp_compare;
#endif

#define DUAL_INHERIT2(child, parent1, parent2)                          \
    Py##child##ArrType_Type.tp_base = &Py##parent1##_Type;              \
    Py##child##ArrType_Type.tp_bases =                                  \
        Py_BuildValue("(OO)", &Py##parent1##_Type,                      \
                      &Py##parent2##ArrType_Type);                      \
    Py##child##ArrType_Type.tp_richcompare =                            \
        Py##parent1##_Type.tp_richcompare;                              \
    DUAL_INHERIT_COMPARE(child, parent1, parent2)                       \
    Py##child##ArrType_Type.tp_hash = Py##parent1##_Type.tp_hash;       \
    if (PyType_Ready(&Py##child##ArrType_Type) < 0) {                   \
        PyErr_Print();                                                  \
        PyErr_Format(PyExc_SystemError,                                 \
                     "could not initialize Py%sArrType_Type",           \
                     #child);                                           \
        return -1;                                                      \
    }

    SINGLE_INHERIT(Bool, Generic);
    SINGLE_INHERIT(Byte, SignedInteger);
    SINGLE_INHERIT(Short, SignedInteger);
#if SIZEOF_INT == SIZEOF_LONG && !defined(NPY_PY3K)
    DUAL_INHERIT(Int, Int, SignedInteger);
#else
    SINGLE_INHERIT(Int, SignedInteger);
#endif
#if !defined(NPY_PY3K)
    DUAL_INHERIT(Long, Int, SignedInteger);
#else
    SINGLE_INHERIT(Long, SignedInteger);
#endif
#if SIZEOF_LONGLONG == SIZEOF_LONG && !defined(NPY_PY3K)
    DUAL_INHERIT(LongLong, Int, SignedInteger);
#else
    SINGLE_INHERIT(LongLong, SignedInteger);
#endif

    SINGLE_INHERIT(TimeInteger, SignedInteger);
    SINGLE_INHERIT(Datetime, TimeInteger);
    SINGLE_INHERIT(Timedelta, TimeInteger);

    /*
       fprintf(stderr,
        "tp_free = %p, PyObject_Del = %p, int_tp_free = %p, base.tp_free = %p\n",
         PyIntArrType_Type.tp_free, PyObject_Del, PyInt_Type.tp_free,
         PySignedIntegerArrType_Type.tp_free);
     */
    SINGLE_INHERIT(UByte, UnsignedInteger);
    SINGLE_INHERIT(UShort, UnsignedInteger);
    SINGLE_INHERIT(UInt, UnsignedInteger);
    SINGLE_INHERIT(ULong, UnsignedInteger);
    SINGLE_INHERIT(ULongLong, UnsignedInteger);

    SINGLE_INHERIT(Float, Floating);
    DUAL_INHERIT(Double, Float, Floating);
    SINGLE_INHERIT(LongDouble, Floating);

    SINGLE_INHERIT(CFloat, ComplexFloating);
    DUAL_INHERIT(CDouble, Complex, ComplexFloating);
    SINGLE_INHERIT(CLongDouble, ComplexFloating);

    DUAL_INHERIT2(String, String, Character);
    DUAL_INHERIT2(Unicode, Unicode, Character);

    SINGLE_INHERIT(Void, Flexible);

    SINGLE_INHERIT(Object, Generic);

    return 0;

#undef SINGLE_INHERIT
#undef DUAL_INHERIT

    /*
     * Clean up string and unicode array types so they act more like
     * strings -- get their tables from the standard types.
     */
}

/* place a flag dictionary in d */

static void
set_flaginfo(PyObject *d)
{
    PyObject *s;
    PyObject *newd;

    newd = PyDict_New();

#define _addnew(val, one)                                       \
    PyDict_SetItemString(newd, #val, s=PyInt_FromLong(val));    \
    Py_DECREF(s);                                               \
    PyDict_SetItemString(newd, #one, s=PyInt_FromLong(val));    \
    Py_DECREF(s)

#define _addone(val)                                            \
    PyDict_SetItemString(newd, #val, s=PyInt_FromLong(val));    \
    Py_DECREF(s)

    _addnew(OWNDATA, O);
    _addnew(FORTRAN, F);
    _addnew(CONTIGUOUS, C);
    _addnew(ALIGNED, A);
    _addnew(UPDATEIFCOPY, U);
    _addnew(WRITEABLE, W);
    _addone(C_CONTIGUOUS);
    _addone(F_CONTIGUOUS);

#undef _addone
#undef _addnew

    PyDict_SetItemString(d, "_flagdict", newd);
    Py_DECREF(newd);
    return;
}

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "multiarray",
        NULL,
        -1,
        array_module_methods,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif


static void
error_set(enum npyexc_type tp, const char *msg)
{
    PyObject *cls = NULL, *obj = NULL;

    switch (tp) {
    case NpyExc_MemoryError:
        PyErr_NoMemory();
        break;
    case NpyExc_IOError:
        PyErr_SetString(PyExc_IOError, msg);
        break;
    case NpyExc_ValueError:
        PyErr_SetString(PyExc_ValueError, msg);
        break;
    case NpyExc_TypeError:
        PyErr_SetString(PyExc_TypeError, msg);
        break;
    case NpyExc_IndexError:
        PyErr_SetString(PyExc_IndexError, msg);
        break;
    case NpyExc_RuntimeError:
        PyErr_SetString(PyExc_RuntimeError, msg);
        break;
    case NpyExc_AttributeError:
        PyErr_SetString(PyExc_AttributeError, msg);
        break;
    case NpyExc_ComplexWarning:
        obj = PyImport_ImportModule("numpy.core");
#if PY_VERSION_HEX >= 0x02050000
#define WarnEx(cls, msg, stackLevel) PyErr_WarnEx(cls, msg, stackLevel)
#else
#define WarnEx(obj, msg, stackLevel) PyErr_Warn(cls, msg)
#endif      
        if (obj) {
            cls = PyObject_GetAttrString(obj, "ComplexWarning");
            WarnEx(cls, msg, 0);
        }
#undef WarnEx
        Py_XDECREF(obj);
        Py_XDECREF(cls);
        break;
    default:
        PyErr_Format(PyExc_SystemError, "Hmm, didn't expect tp = %d", tp);
    }
}

static int
error_occurred(void)
{
    if (PyErr_Occurred() == NULL) {
        return 0;
    }
    else {
        return 1;
    }
}

static void
error_clear(void)
{
    PyErr_Clear();
}


/* Initialization function for the module */
#if defined(NPY_PY3K)
#define RETVAL m
PyObject *PyInit_multiarray(void) {
#else
#define RETVAL
PyMODINIT_FUNC initmultiarray(void) {
#endif
    PyObject *m, *d, *s;
    PyObject *c_api;

    
    /* Initialize the core libnumpy library. Basically this is just providing
       pointers to functions that it can use for coverting type-to-object,
       object-to-type, and similar operations. */
    initlibnumpy(&_array_function_defs,
                 (npy_tp_error_set) error_set,
                 (npy_tp_error_occurred) error_occurred,
                 (npy_tp_error_clear) error_clear);

    /* Create the module and add the functions */
#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("multiarray", array_module_methods);
#endif
    if (!m) {
        goto err;
    }

#if defined(MS_WIN64) && defined(__GNUC__)
    PyErr_WarnEx(PyExc_Warning,
        "Numpy built with MINGW-W64 on Windows 64 bits is experimental, "
        "and only available for \n"
        "testing. You are advised not to use it for production. \n\n"
        "CRASHES ARE TO BE EXPECTED - PLEASE REPORT THEM TO NUMPY DEVELOPERS",
        1);
#endif

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);
    if (!d) {
        goto err;
    }
    PyArray_Type.tp_free = _pya_free;
    if (PyType_Ready(&PyArray_Type) < 0) {
        return RETVAL;
    }
    if (setup_scalartypes(d) < 0) {
        goto err;
    }
    PyArrayIter_Type.tp_iter = PyObject_SelfIter;
    PyArrayMultiIter_Type.tp_iter = PyObject_SelfIter;
    PyArrayMultiIter_Type.tp_free = _pya_free;
    if (PyType_Ready(&PyArrayIter_Type) < 0) {
        return RETVAL;
    }
    if (PyType_Ready(&PyArrayMapIter_Type) < 0) {
        return RETVAL;
    }
    if (PyType_Ready(&PyArrayMultiIter_Type) < 0) {
        return RETVAL;
    }
    PyArrayNeighborhoodIter_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&PyArrayNeighborhoodIter_Type) < 0) {
        return RETVAL;
    }

    PyArrayDescr_Type.tp_hash = PyArray_DescrHash;
    if (PyType_Ready(&PyArrayDescr_Type) < 0) {
        return RETVAL;
    }
    if (PyType_Ready(&PyArrayFlags_Type) < 0) {
        return RETVAL;
    }
/* FIXME
 * There is no error handling here
 */
    c_api = PyCapsule_FromVoidPtr((void *)PyArray_API, NULL);
    PyDict_SetItemString(d, "_ARRAY_API", c_api);
    Py_DECREF(c_api);
    if (PyErr_Occurred()) {
        goto err;
    }

    /* Initialize types in numpymemoryview.c */
    if (_numpymemoryview_init(&s) < 0) {
        return RETVAL;
    }
    if (s != NULL) {
        PyDict_SetItemString(d, "memorysimpleview", s);
    }

    /*
     * PyExc_Exception should catch all the standard errors that are
     * now raised instead of the string exception "multiarray.error"

     * This is for backward compatibility with existing code.
     */
    PyDict_SetItemString (d, "error", PyExc_Exception);

    s = PyUString_FromString("3.1");
    PyDict_SetItemString(d, "__version__", s);
    Py_DECREF(s);

/* FIXME
 * There is no error handling here
 */
    s = PyCapsule_FromVoidPtr((void *)_datetime_strings, NULL);
    PyDict_SetItemString(d, "DATETIMEUNITS", s);
    Py_DECREF(s);

#define ADDCONST(NAME)                          \
    s = PyInt_FromLong(NPY_##NAME);             \
    PyDict_SetItemString(d, #NAME, s);          \
    Py_DECREF(s)


    ADDCONST(ALLOW_THREADS);
    ADDCONST(BUFSIZE);
    ADDCONST(CLIP);

    ADDCONST(ITEM_HASOBJECT);
    ADDCONST(LIST_PICKLE);
    ADDCONST(ITEM_IS_POINTER);
    ADDCONST(NEEDS_INIT);
    ADDCONST(NEEDS_PYAPI);
    ADDCONST(USE_GETITEM);
    ADDCONST(USE_SETITEM);

    ADDCONST(RAISE);
    ADDCONST(WRAP);
    ADDCONST(MAXDIMS);
#undef ADDCONST

    Py_INCREF(&PyArray_Type);
    PyDict_SetItemString(d, "ndarray", (PyObject *)&PyArray_Type);
    Py_INCREF(&PyArrayIter_Type);
    PyDict_SetItemString(d, "flatiter", (PyObject *)&PyArrayIter_Type);
    Py_INCREF(&PyArrayMultiIter_Type);
    PyDict_SetItemString(d, "broadcast",
                         (PyObject *)&PyArrayMultiIter_Type);
    Py_INCREF(&PyArrayDescr_Type);
    PyDict_SetItemString(d, "dtype", (PyObject *)&PyArrayDescr_Type);

    Py_INCREF(&PyArrayFlags_Type);
    PyDict_SetItemString(d, "flagsobj", (PyObject *)&PyArrayFlags_Type);

    set_flaginfo(d);

    if (set_typeinfo(d) != 0) {
        goto err;
    }
    return RETVAL;

 err:
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load multiarray module.");
    }
    return RETVAL;
}
