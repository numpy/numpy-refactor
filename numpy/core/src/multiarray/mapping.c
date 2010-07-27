#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

/*#include <stdio.h>*/
#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/numpy_api.h"
#include "npy_config.h"

#include "npy_3kcompat.h"

#include "common.h"
#include "ctors.h"
#include "iterators.h"
#include "arrayobject.h"
#include "mapping.h"

#define SOBJ_NOTFANCY 0
#define SOBJ_ISFANCY 1
#define SOBJ_BADARRAY 2
#define SOBJ_TOOMANY 3
#define SOBJ_LISTTUP 4

#define ASSERT_ONE_BASE(r) \
    assert(NULL == PyArray_BASE_ARRAY(r) || NULL == PyArray_BASE(r))

static PyObject *
array_subscript_simple(PyArrayObject *self, PyObject *op);


/* Callback from the core to construct the PyObject wrapper around an interator. */
NPY_NO_EXPORT int
NpyInterface_MapIterNewWrapper(NpyArrayMapIterObject *iter, void **interfaceRet)
{
    PyArrayMapIterObject *result;
    
    result = _pya_malloc(sizeof(*result));
    if (result == NULL) {
        *interfaceRet = NULL;
        return NPY_FALSE;
    }
    
    PyObject_Init((PyObject *)result, &PyArrayMapIter_Type);
    result->magic_number = NPY_VALID_MAGIC;
    result->iter = iter;
    
    *interfaceRet = result;
    return NPY_TRUE;
}




/******************************************************************************
 ***                    IMPLEMENT MAPPING PROTOCOL                          ***
 *****************************************************************************/

NPY_NO_EXPORT Py_ssize_t
array_length(PyArrayObject *self)
{
    if (PyArray_NDIM(self) != 0) {
        return PyArray_DIM(self, 0);
    } else {
        PyErr_SetString(PyExc_TypeError, "len() of unsized object");
        return -1;
    }
}

NPY_NO_EXPORT PyObject *
array_big_item(PyArrayObject *self, intp i)
{
    RETURN_PYARRAY(NpyArray_ArrayItem(PyArray_ARRAY(self), i));
}

NPY_NO_EXPORT int
_array_ass_item(PyArrayObject *self, Py_ssize_t i, PyObject *v)
{
    return array_ass_big_item(self, (intp) i, v);
}
/* contains optimization for 1-d arrays */
NPY_NO_EXPORT PyObject *
array_item_nice(PyArrayObject *self, Py_ssize_t i)
{
    if (PyArray_NDIM(self) == 1) {
        char *item;
        if ((item = index2ptr(self, i)) == NULL) {
            return NULL;
        }
        return PyArray_Scalar(item, PyArray_Descr_WRAP( PyArray_DESCR(self) ), (PyObject *)self);
    }
    else {
        return PyArray_Return(
                (PyArrayObject *) array_big_item(self, (intp) i));
    }
}

NPY_NO_EXPORT int
array_ass_big_item(PyArrayObject *self, intp i, PyObject *v)
{
    PyArrayObject *tmp;
    char *item;
    int ret;

    if (v == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "can't delete array elements");
        return -1;
    }
    if (!PyArray_ISWRITEABLE(self)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "array is not writeable");
        return -1;
    }
    if (PyArray_NDIM(self) == 0) {
        PyErr_SetString(PyExc_IndexError,
                        "0-d arrays can't be indexed.");
        return -1;
    }


    if (PyArray_NDIM(self) > 1) {
        if((tmp = (PyArrayObject *)array_big_item(self, i)) == NULL) {
            return -1;
        }
        ret = PyArray_CopyObject(tmp, v);
        Py_DECREF(tmp);
        return ret;
    }

    if ((item = index2ptr(self, i)) == NULL) {
        return -1;
    }
    if (PyArray_DESCR(self)->f->setitem(v, item, PyArray_ARRAY(self)) == -1) {
        return -1;
    }
    return 0;
}

/* -------------------------------------------------------------- */

static void
_swap_axes(NpyArrayMapIterObject *mit, PyArrayObject **ret, int getmap)
{
    PyObject *new;
    int n1, n2, n3, val, bnd;
    int i;
    PyArray_Dims permute;
    intp d[MAX_DIMS];
    PyArrayObject *arr;

    permute.ptr = d;
    permute.len = mit->nd;

    /*
     * arr might not have the right number of dimensions
     * and need to be reshaped first by pre-pending ones
     */
    arr = *ret;
    if (PyArray_NDIM(arr) != mit->nd) {
        for (i = 1; i <= PyArray_NDIM(arr); i++) {
            permute.ptr[mit->nd-i] = PyArray_DIM(arr, PyArray_NDIM(arr)-i);
        }
        for (i = 0; i < mit->nd-PyArray_NDIM(arr); i++) {
            permute.ptr[i] = 1;
        }
        new = PyArray_Newshape(arr, &permute, PyArray_ANYORDER);
        Py_DECREF(arr);
        *ret = (PyArrayObject *)new;
        if (new == NULL) {
            return;
        }
    }

    /*
     * Setting and getting need to have different permutations.
     * On the get we are permuting the returned object, but on
     * setting we are permuting the object-to-be-set.
     * The set permutation is the inverse of the get permutation.
     */

    /*
     * For getting the array the tuple for transpose is
     * (n1,...,n1+n2-1,0,...,n1-1,n1+n2,...,n3-1)
     * n1 is the number of dimensions of the broadcast index array
     * n2 is the number of dimensions skipped at the start
     * n3 is the number of dimensions of the result
     */

    /*
     * For setting the array the tuple for transpose is
     * (n2,...,n1+n2-1,0,...,n2-1,n1+n2,...n3-1)
     */
    n1 = mit->iters[0]->nd_m1 + 1;
    n2 = mit->iteraxes[0];
    n3 = mit->nd;

    /* use n1 as the boundary if getting but n2 if setting */
    bnd = getmap ? n1 : n2;
    val = bnd;
    i = 0;
    while (val < n1 + n2) {
        permute.ptr[i++] = val++;
    }
    val = 0;
    while (val < bnd) {
        permute.ptr[i++] = val++;
    }
    val = n1 + n2;
    while (val < n3) {
        permute.ptr[i++] = val++;
    }
    new = PyArray_Transpose(*ret, &permute);
    Py_DECREF(*ret);
    *ret = (PyArrayObject *)new;
}

static PyObject *
PyArray_GetMap(PyArrayMapIterObject *pyMit)
{
    NpyArrayMapIterObject *mit = pyMit->iter;
    NpyArrayIterObject *it;
    PyArrayObject *ret, *temp;
    int index;
    int swap;
    PyArray_CopySwapFunc *copyswap;

    /* Unbound map iterator --- Bind should have been called */
    if (mit->ait == NULL) {
        return NULL;
    }

    /* This relies on the map iterator object telling us the shape
       of the new array in nd and dimensions.
    */
    temp = Npy_INTERFACE( mit->ait->ao );
    _Npy_INCREF(PyArray_DESCR(temp));
    ASSIGN_TO_PYARRAY(ret, NpyArray_NewFromDescr(PyArray_DESCR(temp),
                                               mit->nd, mit->dimensions,
                                               NULL, NULL,
                                               PyArray_ISFORTRAN(temp),
                                               NPY_FALSE, NULL, temp));
    if (ret == NULL) {
        return NULL;
    }

    /*
     * Now just iterate through the new array filling it in
     * with the next object from the original array as
     * defined by the mapping iterator
     */

    if ((it = NpyArray_IterNew(PyArray_ARRAY(ret))) == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    index = it->size;
    swap = (PyArray_ISNOTSWAPPED(temp) != PyArray_ISNOTSWAPPED(ret));
    copyswap = PyArray_DESCR(ret)->f->copyswap;
    NpyArray_MapIterReset(mit);
    while (index--) {
        copyswap(it->dataptr, mit->dataptr, swap, PyArray_ARRAY(ret));
        NpyArray_MapIterNext(mit);
        NpyArray_ITER_NEXT(it);
    }
    _Npy_DECREF(it);

    /* check for consecutive axes */
    if ((mit->subspace != NULL) && (mit->consec)) {
        if (mit->iteraxes[0] > 0) {  /* then we need to swap */
            _swap_axes(mit, &ret, 1);
        }
    }
    return (PyObject *)ret;
}

static int
PyArray_SetMap(PyArrayMapIterObject *pyMit, PyObject *op)
{
    NpyArrayMapIterObject *mit = pyMit->iter;
    NpyArrayIterObject *it;
    PyObject *arr = NULL;
    int index;
    int swap;
    PyArray_CopySwapFunc *copyswap;
    NpyArray_Descr *descr;

    /* Unbound Map Iterator */
    if (mit->ait == NULL) {
        return -1;
    }
    descr = mit->ait->ao->descr;
    _Npy_INCREF(descr);
    arr = PyArray_FromAnyUnwrap(op, descr, 0, 0, FORCECAST, NULL);
    if (arr == NULL) {
        return -1;
    }
    if ((mit->subspace != NULL) && (mit->consec)) {
        if (mit->iteraxes[0] > 0) {  /* then we need to swap */
            _swap_axes(mit, (PyArrayObject **)&arr, 0);
            if (arr == NULL) {
                return -1;
            }
        }
    }

    /* Be sure values array is "broadcastable"
       to shape of mit->dimensions, mit->nd */

    if ((it = NpyArray_BroadcastToShape(PyArray_ARRAY(arr), mit->dimensions, 
                                        mit->nd))==NULL) { 
        Py_DECREF(arr);
        return -1;
    }

    index = mit->size;
    swap = (NpyArray_ISNOTSWAPPED(mit->ait->ao) !=
            (PyArray_ISNOTSWAPPED(arr)));
    copyswap = PyArray_DESCR(arr)->f->copyswap;
    NpyArray_MapIterReset(mit);
    /* Need to decref arrays with objects in them */
    if (NpyDataType_FLAGCHK(descr, NPY_ITEM_HASOBJECT)) {
        while (index--) {
            NpyArray_Item_INCREF(it->dataptr, PyArray_DESCR(arr));
            NpyArray_Item_XDECREF(mit->dataptr, PyArray_DESCR(arr));
            memmove(mit->dataptr, it->dataptr, PyArray_ITEMSIZE(arr));
            /* ignored unless VOID array with object's */
            if (swap) {
                copyswap(mit->dataptr, NULL, swap, PyArray_ARRAY(arr));
            }
            NpyArray_MapIterNext(mit);
            NpyArray_ITER_NEXT(it);
        }
        Py_DECREF(arr);
        _Npy_DECREF(it);
        return 0;
    }
    while(index--) {
        memmove(mit->dataptr, it->dataptr, PyArray_ITEMSIZE(arr));
        if (swap) {
            copyswap(mit->dataptr, NULL, swap, PyArray_ARRAY(arr));
        }
        NpyArray_MapIterNext(mit);
        NpyArray_ITER_NEXT(it);
    }
    Py_DECREF(arr);
    _Npy_DECREF(it);
    return 0;
}

NPY_NO_EXPORT int
count_new_axes_0d(PyObject *tuple)
{
    int i, argument_count;
    int ellipsis_count = 0;
    int newaxis_count = 0;

    argument_count = PyTuple_GET_SIZE(tuple);
    for (i = 0; i < argument_count; ++i) {
        PyObject *arg = PyTuple_GET_ITEM(tuple, i);
        if (arg == Py_Ellipsis && !ellipsis_count) {
            ellipsis_count++;
        }
        else if (arg == Py_None) {
            newaxis_count++;
        }
        else {
            break;
        }
    }
    if (i < argument_count) {
        PyErr_SetString(PyExc_IndexError,
                        "0-d arrays can only use a single ()"
                        " or a list of newaxes (and a single ...)"
                        " as an index");
        return -1;
    }
    if (newaxis_count > MAX_DIMS) {
        PyErr_SetString(PyExc_IndexError, "too many dimensions");
        return -1;
    }
    return newaxis_count;
}

NPY_NO_EXPORT PyObject *
add_new_axes_0d(PyArrayObject *arr,  int newaxis_count)
{
    PyArrayObject *other;
    intp dimensions[MAX_DIMS];
    int i;

    for (i = 0; i < newaxis_count; ++i) {
        dimensions[i]  = 1;
    }
    _Npy_INCREF(PyArray_DESCR(arr));
    ASSIGN_TO_PYARRAY(other, 
                      NpyArray_NewFromDescr(PyArray_DESCR(arr),
                                            newaxis_count, dimensions,
                                            NULL, PyArray_BYTES(arr),
                                            PyArray_FLAGS(arr), NPY_FALSE,
                                            NULL, arr));
    if (NULL == other) {
        return NULL;
    }
    
    PyArray_BASE_ARRAY(other) = PyArray_ARRAY(arr);
    _Npy_INCREF(PyArray_BASE_ARRAY(other));
    ASSERT_ONE_BASE(other);
    return (PyObject *)other;
}


/* This checks the args for any fancy indexing objects */

static int
fancy_indexing_check(PyObject *args)
{
    int i, n;
    PyObject *obj;
    int retval = SOBJ_NOTFANCY;

    if (PyTuple_Check(args)) {
        n = PyTuple_GET_SIZE(args);
        if (n >= MAX_DIMS) {
            return SOBJ_TOOMANY;
        }
        for (i = 0; i < n; i++) {
            obj = PyTuple_GET_ITEM(args,i);
            if (PyArray_Check(obj)) {
                if (PyArray_ISINTEGER(obj) ||
                    PyArray_ISBOOL(obj)) {
                    retval = SOBJ_ISFANCY;
                }
                else {
                    retval = SOBJ_BADARRAY;
                    break;
                }
            }
            else if (PySequence_Check(obj)) {
                retval = SOBJ_ISFANCY;
            }
        }
    }
    else if (PyArray_Check(args)) {
        if ((PyArray_TYPE(args)==PyArray_BOOL) ||
            (PyArray_ISINTEGER(args))) {
            return SOBJ_ISFANCY;
        }
        else {
            return SOBJ_BADARRAY;
        }
    }
    else if (PySequence_Check(args)) {
        /*
         * Sequences < MAX_DIMS with any slice objects
         * or newaxis, or Ellipsis is considered standard
         * as long as there are also no Arrays and or additional
         * sequences embedded.
         */
        retval = SOBJ_ISFANCY;
        n = PySequence_Size(args);
        if (n < 0 || n >= MAX_DIMS) {
            return SOBJ_ISFANCY;
        }
        for (i = 0; i < n; i++) {
            obj = PySequence_GetItem(args, i);
            if (obj == NULL) {
                return SOBJ_ISFANCY;
            }
            if (PyArray_Check(obj)) {
                if (PyArray_ISINTEGER(obj) || PyArray_ISBOOL(obj)) {
                    retval = SOBJ_LISTTUP;
                }
                else {
                    retval = SOBJ_BADARRAY;
                }
            }
            else if (PySequence_Check(obj)) {
                retval = SOBJ_LISTTUP;
            }
            else if (PySlice_Check(obj) || obj == Py_Ellipsis ||
                    obj == Py_None) {
                retval = SOBJ_NOTFANCY;
            }
            Py_DECREF(obj);
            if (retval > SOBJ_ISFANCY) {
                return retval;
            }
        }
    }
    return retval;
}

/*
 * Called when treating array object like a mapping -- called first from
 * Python when using a[object] unless object is a standard slice object
 * (not an extended one).
 *
 * There are two situations:
 *
 *   1 - the subscript is a standard view and a reference to the
 *   array can be returned
 *
 *   2 - the subscript uses Boolean masks or integer indexing and
 *   therefore a new array is created and returned.
 */

NPY_NO_EXPORT PyObject *
array_subscript_simple(PyArrayObject *self, PyObject *op)
{
    intp dimensions[MAX_DIMS], strides[MAX_DIMS];
    intp offset;
    int nd;
    PyArrayObject *other;
    intp value;

    value = PyArray_PyIntAsIntp(op);
    if (!PyErr_Occurred()) {
        return array_big_item(self, value);
    }
    PyErr_Clear();

    /* Standard (view-based) Indexing */
    if ((nd = parse_index(self, op, dimensions, strides, &offset)) == -1) {
        return NULL;
    }
    /* This will only work if new array will be a view */
    _Npy_INCREF(PyArray_DESCR(self));
    ASSIGN_TO_PYARRAY(other, NpyArray_NewFromDescr(PyArray_DESCR(self),
                                                   nd, dimensions,
                                                   strides, PyArray_BYTES(self)+offset,
                                                   PyArray_FLAGS(self), NPY_FALSE,
                                                   NULL, self));
    if (NULL == other) {
        return NULL;
    }
    PyArray_BASE_ARRAY(other) = PyArray_ARRAY(self);
    _Npy_INCREF(PyArray_BASE_ARRAY(other));
    PyArray_UpdateFlags(other, UPDATE_ALL);
    ASSERT_ONE_BASE(other);
    return (PyObject *)other;
}

NPY_NO_EXPORT PyObject *
array_subscript(PyArrayObject *self, PyObject *op)
{
    int nd, fancy;
    PyArrayObject *other;
    NpyArray_DescrField *value;
    PyObject *obj;

    if (PyString_Check(op) || PyUnicode_Check(op)) {
        PyObject *temp;

        if (NULL != PyArray_DESCR(self)->names) {
            value = NpyDict_Get(PyArray_DESCR(self)->fields, PyString_AsString(op));
            if (NULL != value) {
                PyArrayObject *result;
                
                _Npy_INCREF(value->descr);  /* NpyArray_GetField steal ref. */
                ASSIGN_TO_PYARRAY(result, 
                                  NpyArray_GetField(PyArray_ARRAY(self), 
                                                    value->descr, value->offset));
                return (PyObject *)result;
            }
        }

        temp = op;
        if (PyUnicode_Check(op)) {
            temp = PyUnicode_AsUnicodeEscapeString(op);
        }
        PyErr_Format(PyExc_ValueError,
                     "field named %s not found.",
                     PyBytes_AsString(temp));
        if (temp != op) {
            Py_DECREF(temp);
        }
        return NULL;
    }

    /* Check for multiple field access */
    if (PyArray_DESCR(self)->names && PySequence_Check(op) && !PyTuple_Check(op)) {
        int seqlen, i;
        seqlen = PySequence_Size(op);
        for (i = 0; i < seqlen; i++) {
            obj = PySequence_GetItem(op, i);
            if (!PyString_Check(obj) && !PyUnicode_Check(obj)) {
                Py_DECREF(obj);
                break;
            }
            Py_DECREF(obj);
        }
        /*
         * extract multiple fields if all elements in sequence
         * are either string or unicode (i.e. no break occurred).
         */
        fancy = ((seqlen > 0) && (i == seqlen));
        if (fancy) {
            PyObject *_numpy_internal;
            _numpy_internal = PyImport_ImportModule("numpy.core._internal");
            if (_numpy_internal == NULL) {
                return NULL;
            }
            obj = PyObject_CallMethod(_numpy_internal,
                    "_index_fields", "OO", self, op);
            Py_DECREF(_numpy_internal);
            return obj;
        }
    }

    if (op == Py_Ellipsis) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    if (PyArray_NDIM(self) == 0) {
        if (op == Py_None) {
            return add_new_axes_0d(self, 1);
        }
        if (PyTuple_Check(op)) {
            if (0 == PyTuple_GET_SIZE(op))  {
                Py_INCREF(self);
                return (PyObject *)self;
            }
            if ((nd = count_new_axes_0d(op)) == -1) {
                return NULL;
            }
            return add_new_axes_0d(self, nd);
        }
        /* Allow Boolean mask selection also */
        if ((PyArray_Check(op) && (PyArray_DIMS(op)==0)
                    && PyArray_ISBOOL(op))) {
            if (PyObject_IsTrue(op)) {
                Py_INCREF(self);
                return (PyObject *)self;
            }
            else {
                PyArrayObject *result;
                
                intp oned = 0;
                _Npy_INCREF(PyArray_DESCR(self));
                ASSIGN_TO_PYARRAY(result, 
                                  NpyArray_NewFromDescr(PyArray_DESCR(self),
                                                        1, &oned,
                                                        NULL, NULL,
                                                        NPY_DEFAULT,
                                                        NPY_FALSE, Py_TYPE(self), NULL));
                return (PyObject *)result;
            }
        }
        PyErr_SetString(PyExc_IndexError, "0-d arrays can't be indexed.");
        return NULL;
    }

    fancy = fancy_indexing_check(op);
    if (fancy != SOBJ_NOTFANCY) {
        PyArrayMapIterObject *mit;
        int oned;

        oned = ((PyArray_NDIM(self) == 1) &&
                !(PyTuple_Check(op) && PyTuple_GET_SIZE(op) > 1));

        /* wrap arguments into a mapiter object */
        mit = (PyArrayMapIterObject *) PyArray_MapIterNew(op, oned, fancy);
        if (mit == NULL) {
            return NULL;
        }
        if (oned) {
            PyArrayIterObject *it;
            PyObject *rval;
            it = (PyArrayIterObject *) PyArray_IterNew((PyObject *)self);
            if (it == NULL) {
                Py_DECREF(mit);
                return NULL;
            }
            rval = npy_iter_subscript(it->iter, mit->iter->indexobj);
            Py_DECREF(it);
            Py_DECREF(mit);
            return rval;
        }
        PyArray_MapIterBind(mit, self);
        other = (PyArrayObject *)PyArray_GetMap(mit);
        Py_DECREF(mit);
        return (PyObject *)other;
    }

    return array_subscript_simple(self, op);
}


/*
 * Another assignment hacked by using CopyObject.
 * This only works if subscript returns a standard view.
 * Again there are two cases.  In the first case, PyArray_CopyObject
 * can be used.  In the second case, a new indexing function has to be
 * used.
 */

static int
array_ass_sub_simple(PyArrayObject *self, PyObject *index, PyObject *op)
{
    int ret;
    PyArrayObject *tmp;
    intp value;

    value = PyArray_PyIntAsIntp(index);
    if (!error_converting(value)) {
        return array_ass_big_item(self, value, op);
    }
    PyErr_Clear();

    /* Rest of standard (view-based) indexing */

    if (PyArray_CheckExact(self)) {
        tmp = (PyArrayObject *)array_subscript_simple(self, index);
        if (tmp == NULL) {
            return -1;
        }
    }
    else {
        PyObject *tmp0;
        tmp0 = PyObject_GetItem((PyObject *)self, index);
        if (tmp0 == NULL) {
            return -1;
        }
        if (!PyArray_Check(tmp0)) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Getitem not returning array.");
            Py_DECREF(tmp0);
            return -1;
        }
        tmp = (PyArrayObject *)tmp0;
    }

    if (PyArray_ISOBJECT(self) && (PyArray_NDIM(tmp) == 0)) {
        ret = PyArray_DESCR(tmp)->f->setitem(op, PyArray_BYTES(tmp), 
                                             PyArray_ARRAY(tmp));
    }
    else {
        ret = PyArray_CopyObject(tmp, op);
    }
    Py_DECREF(tmp);
    return ret;
}


/* return -1 if tuple-object seq is not a tuple of integers.
   otherwise fill vals with converted integers
*/
static int
_tuple_of_integers(PyObject *seq, intp *vals, int maxvals)
{
    int i;
    PyObject *obj;
    intp temp;

    for(i=0; i<maxvals; i++) {
        obj = PyTuple_GET_ITEM(seq, i);
        if ((PyArray_Check(obj) && PyArray_NDIM(obj) > 0)
                || PyList_Check(obj)) {
            return -1;
        }
        temp = PyArray_PyIntAsIntp(obj);
        if (error_converting(temp)) {
            return -1;
        }
        vals[i] = temp;
    }
    return 0;
}


static int
array_ass_sub(PyArrayObject *self, PyObject *index, PyObject *op)
{
    int ret, oned, fancy;
    intp vals[MAX_DIMS];

    if (op == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot delete array elements");
        return -1;
    }
    if (!PyArray_ISWRITEABLE(self)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "array is not writeable");
        return -1;
    }

    if (PyInt_Check(index) || PyArray_IsScalar(index, Integer) ||
        PyLong_Check(index) || (PyIndex_Check(index) &&
                                !PySequence_Check(index))) {
        intp value;
        value = PyArray_PyIntAsIntp(index);
        if (PyErr_Occurred()) {
            PyErr_Clear();
        }
        else {
            return array_ass_big_item(self, value, op);
        }
    }

    if (PyString_Check(index) || PyUnicode_Check(index)) {
        if (PyArray_DESCR(self)->names) {
            NpyArray_DescrField *value;
            
            value = NpyDict_Get(PyArray_DESCR(self)->fields, PyString_AsString(index));
            if (NULL != value) {
                PyArray_Descr *descrWrap = PyArray_Descr_WRAP(value->descr);
                Py_INCREF(descrWrap);
                return PyArray_SetField(self, descrWrap, 
                                        value->offset, op);
            }
        }

        PyErr_Format(PyExc_ValueError,
                     "field named %s not found.",
                     PyString_AsString(index));
        return -1;
    }

    if (PyArray_NDIM(self) == 0) {
        /*
         * Several different exceptions to the 0-d no-indexing rule
         *
         *  1) ellipses
         *  2) empty tuple
         *  3) Using newaxis (None)
         *  4) Boolean mask indexing
         */
        if (index == Py_Ellipsis || index == Py_None ||
            (PyTuple_Check(index) && (0 == PyTuple_GET_SIZE(index) ||
                                      count_new_axes_0d(index) > 0))) {
            return PyArray_DESCR(self)->f->setitem(op, PyArray_BYTES(self), 
                                                   PyArray_ARRAY(self));
        }
        if (PyBool_Check(index) || PyArray_IsScalar(index, Bool) ||
            (PyArray_Check(index) && (PyArray_DIMS(index)==0) &&
             PyArray_ISBOOL(index))) {
            if (PyObject_IsTrue(index)) {
                return PyArray_DESCR(self)->f->setitem(op, 
                            PyArray_BYTES(self), PyArray_ARRAY(self));
            }
            else { /* don't do anything */
                return 0;
            }
        }
        PyErr_SetString(PyExc_IndexError, "0-d arrays can't be indexed.");
        return -1;
    }

    /* optimization for integer-tuple */
    if (PyArray_NDIM(self) > 1 &&
        (PyTuple_Check(index) && (PyTuple_GET_SIZE(index) == PyArray_NDIM(self)))
        && (_tuple_of_integers(index, vals, PyArray_NDIM(self)) >= 0)) {
        int i;
        char *item;

        for (i = 0; i < PyArray_NDIM(self); i++) {
            if (vals[i] < 0) {
                vals[i] += PyArray_DIM(self, i);
            }
            if ((vals[i] < 0) || (vals[i] >= PyArray_DIM(self, i))) {
                PyErr_Format(PyExc_IndexError,
                             "index (%"INTP_FMT") out of range "\
                             "(0<=index<%"INTP_FMT") in dimension %d",
                             vals[i], PyArray_DIM(self, i), i);
                return -1;
            }
        }
        item = PyArray_GetPtr(self, vals);
        return PyArray_DESCR(self)->f->setitem(op, item, 
                                               PyArray_ARRAY(self));
    }
    PyErr_Clear();

    fancy = fancy_indexing_check(index);
    if (fancy != SOBJ_NOTFANCY) {
        PyArrayMapIterObject *mit;
        
        oned = ((PyArray_NDIM(self) == 1) &&
                !(PyTuple_Check(index) && PyTuple_GET_SIZE(index) > 1));
        mit = (PyArrayMapIterObject *) PyArray_MapIterNew(index, oned, fancy);
        if (mit == NULL) {
            return -1;
        }
        if (oned) {
            NpyArrayIterObject *it;
            int rval;

            it = NpyArray_IterNew(PyArray_ARRAY(self));
            if (it == NULL) {
                Py_DECREF(mit);
                return -1;
            }
            rval = npy_iter_ass_subscript(it, (PyObject *)mit->iter->indexobj, op);
            _Npy_DECREF(it);
            Py_DECREF(mit);
            return rval;
        }
        PyArray_MapIterBind(mit, self);
        ret = PyArray_SetMap(mit, op);
        Py_DECREF(mit);
        return ret;
    }

    return array_ass_sub_simple(self, index, op);
}


/*
 * There are places that require that array_subscript return a PyArrayObject
 * and not possibly a scalar.  Thus, this is the function exposed to
 * Python so that 0-dim arrays are passed as scalars
 */


static PyObject *
array_subscript_nice(PyArrayObject *self, PyObject *op)
{

    PyArrayObject *mp;
    intp vals[MAX_DIMS];

    if (PyInt_Check(op) || PyArray_IsScalar(op, Integer) ||
        PyLong_Check(op) || (PyIndex_Check(op) &&
                             !PySequence_Check(op))) {
        intp value;
        value = PyArray_PyIntAsIntp(op);
        if (PyErr_Occurred()) {
            PyErr_Clear();
        }
        else {
            return array_item_nice(self, (Py_ssize_t) value);
        }
    }
    /* optimization for a tuple of integers */
    if (PyArray_NDIM(self) > 1 && PyTuple_Check(op) &&
        (PyTuple_GET_SIZE(op) == PyArray_NDIM(self))
        && (_tuple_of_integers(op, vals, PyArray_NDIM(self)) >= 0)) {
        int i;
        char *item;

        for (i = 0; i < PyArray_NDIM(self); i++) {
            if (vals[i] < 0) {
                vals[i] += PyArray_DIM(self, i);
            }
            if ((vals[i] < 0) || (vals[i] >= PyArray_DIM(self, i))) {
                PyErr_Format(PyExc_IndexError,
                             "index (%"INTP_FMT") out of range "\
                             "(0<=index<%"INTP_FMT") in dimension %d",
                             vals[i], PyArray_DIM(self, i), i);
                return NULL;
            }
        }
        item = PyArray_GetPtr(self, vals);
        return PyArray_Scalar(item, PyArray_Descr_WRAP(PyArray_DESCR(self)), (PyObject *)self);
    }
    PyErr_Clear();

    mp = (PyArrayObject *)array_subscript(self, op);
    /*
     * mp could be a scalar if op is not an Int, Scalar, Long or other Index
     * object and still convertable to an integer (so that the code goes to
     * array_subscript_simple).  So, this cast is a bit dangerous..
     */

    /*
     * The following is just a copy of PyArray_Return with an
     * additional logic in the nd == 0 case.
     */

    if (mp == NULL) {
        return NULL;
    }
    if (PyErr_Occurred()) {
        Py_XDECREF(mp);
        return NULL;
    }
    if (PyArray_Check(mp) && PyArray_NDIM(mp) == 0) {
        Bool noellipses = TRUE;
        if ((op == Py_Ellipsis) || PyString_Check(op) || PyUnicode_Check(op)) {
            noellipses = FALSE;
        }
        else if (PyBool_Check(op) || PyArray_IsScalar(op, Bool) ||
                 (PyArray_Check(op) && (PyArray_DIMS(op)==0) &&
                  PyArray_ISBOOL(op))) {
            noellipses = FALSE;
        }
        else if (PySequence_Check(op)) {
            Py_ssize_t n, i;
            PyObject *temp;

            n = PySequence_Size(op);
            i = 0;
            while (i < n && noellipses) {
                temp = PySequence_GetItem(op, i);
                if (temp == Py_Ellipsis) {
                    noellipses = FALSE;
                }
                Py_DECREF(temp);
                i++;
            }
        }
        if (noellipses) {
            PyObject *ret;
            ret = PyArray_ToScalar(PyArray_BYTES(mp), mp);
            Py_DECREF(mp);
            return ret;
        }
    }
    return (PyObject *)mp;
}


NPY_NO_EXPORT PyMappingMethods array_as_mapping = {
#if PY_VERSION_HEX >= 0x02050000
    (lenfunc)array_length,              /*mp_length*/
#else
    (inquiry)array_length,              /*mp_length*/
#endif
    (binaryfunc)array_subscript_nice,       /*mp_subscript*/
    (objobjargproc)array_ass_sub,       /*mp_ass_subscript*/
};

/****************** End of Mapping Protocol ******************************/

/*********************** Subscript Array Iterator *************************
 *                                                                        *
 * This object handles subscript behavior for array objects.              *
 *  It is an iterator object with a next method                           *
 *  It abstracts the n-dimensional mapping behavior to make the looping   *
 *     code more understandable (maybe)                                   *
 *     and so that indexing can be set up ahead of time                   *
 */

/*
 * This function takes a Boolean array and constructs index objects and
 * iterators as if nonzero(Bool) had been called
 */
static int
_nonzero_indices(PyObject *myBool, NpyArrayIterObject **iters)
{
    NpyArray_Descr *typecode;
    PyArrayObject *ba = NULL, *new = NULL;
    int nd, j;
    intp size, i, count;
    Bool *ptr;
    intp coords[MAX_DIMS], dims_m1[MAX_DIMS];
    intp *dptr[MAX_DIMS];

    typecode = NpyArray_DescrFromType(PyArray_BOOL);
    ba = (PyArrayObject *)PyArray_FromAnyUnwrap(myBool, typecode, 0, 0,
                                                CARRAY, NULL);
    if (ba == NULL) {
        return -1;
    }
    nd = PyArray_NDIM(ba);
    for (j = 0; j < nd; j++) {
        iters[j] = NULL;
    }
    size = PyArray_SIZE(ba);
    ptr = (Bool *)PyArray_BYTES(ba);
    count = 0;

    /* pre-determine how many nonzero entries there are */
    for (i = 0; i < size; i++) {
        if (*(ptr++)) {
            count++;
        }
    }

    /* create count-sized index arrays for each dimension */
    for (j = 0; j < nd; j++) {
        new = (PyArrayObject *)PyArray_New(&PyArray_Type, 1, &count,
                                           PyArray_INTP, NULL, NULL,
                                           0, 0, NULL);
        if (new == NULL) {
            goto fail;
        }
        iters[j] = NpyArray_IterNew(PyArray_ARRAY(new));
        Py_DECREF(new);
        if (iters[j] == NULL) {
            goto fail;
        }
        dptr[j] = (intp *)iters[j]->ao->data;
        coords[j] = 0;
        dims_m1[j] = PyArray_DIM(ba, j)-1;
    }
    ptr = (Bool *)PyArray_BYTES(ba);
    if (count == 0) {
        goto finish;
    }

    /*
     * Loop through the Boolean array  and copy coordinates
     * for non-zero entries
     */
    for (i = 0; i < size; i++) {
        if (*(ptr++)) {
            for (j = 0; j < nd; j++) {
                *(dptr[j]++) = coords[j];
            }
        }
        /* Borrowed from ITER_NEXT macro */
        for (j = nd - 1; j >= 0; j--) {
            if (coords[j] < dims_m1[j]) {
                coords[j]++;
                break;
            }
            else {
                coords[j] = 0;
            }
        }
    }

 finish:
    Py_DECREF(ba);
    return nd;

 fail:
    for (j = 0; j < nd; j++) {
        _Npy_XDECREF(iters[j]);
    }
    Py_XDECREF(ba);
    return -1;
}

/* convert an indexing object to an INTP indexing array iterator
   if possible -- otherwise, it is a Slice or Ellipsis object
   and has to be interpreted on bind to a particular
   array so leave it NULL for now.
*/
static int
_convert_obj(PyObject *obj, NpyArrayIterObject **iter)
{
    NpyArray_Descr *indtype;
    PyObject *arr;

    if (PySlice_Check(obj) || (obj == Py_Ellipsis)) {
        return 0;
    }
    else if (PyArray_Check(obj) && PyArray_ISBOOL(obj)) {
        return _nonzero_indices(obj, iter);
    }
    else {
        indtype = NpyArray_DescrFromType(PyArray_INTP);
        arr = PyArray_FromAnyUnwrap(obj, indtype, 0, 0, FORCECAST, NULL);
        if (arr == NULL) {
            return -1;
        }
        *iter = NpyArray_IterNew(PyArray_ARRAY(arr));
        Py_DECREF(arr);
        if (*iter == NULL) {
            return -1;
        }
    }
    return 1;
}

/* Reset the map iterator to the beginning */
NPY_NO_EXPORT void
PyArray_MapIterReset(PyArrayMapIterObject *pyMit)
{
    NpyArray_MapIterReset(pyMit->iter);
}
#if 0
/* Not called from anywhere, and .h says it's not part of the C API.  So
 * exclude it from the build.
 */

/*
 * This function needs to update the state of the map iterator
 * and point mit->dataptr to the memory-location of the next object
 */
NPY_NO_EXPORT void
PyArray_MapIterNext(PyArrayMapIterObject *pyMit)
{
    NpyArray_MapIterNext(pyMit->iter);
}
#endif
/*
 * Bind a mapiteration to a particular array
 *
 *  Determine if subspace iteration is necessary.  If so,
 *  1) Fill in mit->iteraxes
 *  2) Create subspace iterator
 *  3) Update nd, dimensions, and size.
 *
 *  Subspace iteration is necessary if:  arr->nd > mit->numiter
 *
 * Need to check for index-errors somewhere.
 *
 * Let's do it at bind time and also convert all <0 values to >0 here
 * as well.
 */
NPY_NO_EXPORT void
PyArray_MapIterBind(PyArrayMapIterObject *pyMit, PyArrayObject *arr)
{
    NpyArrayMapIterObject *mit = pyMit->iter;
    NpyArrayIterObject *it;
    int subnd;
    PyObject *sub, *obj = NULL;
    int i, j, n, curraxis, ellipexp, noellip;
    intp dimsize;
    intp *indptr;

    subnd = PyArray_NDIM(arr) - mit->numiter;
    if (subnd < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "too many indices for array");
        return;
    }

    mit->ait = NpyArray_IterNew(PyArray_ARRAY(arr));
    if (mit->ait == NULL) {
        return;
    }
    /* no subspace iteration needed.  Finish up and Return */
    if (subnd == 0) {
        n = PyArray_NDIM(arr);
        for (i = 0; i < n; i++) {
            mit->iteraxes[i] = i;
        }
        goto finish;
    }

    /*
     * all indexing arrays have been converted to 0
     * therefore we can extract the subspace with a simple
     * getitem call which will use view semantics
     *
     * But, be sure to do it with a true array.
     */
    if (PyArray_CheckExact(arr)) {
        sub = array_subscript_simple(arr, mit->indexobj);
    }
    else {
        Py_INCREF(arr);
        obj = PyArray_EnsureArray((PyObject *)arr);
        if (obj == NULL) {
            goto fail;
        }
        sub = array_subscript_simple((PyArrayObject *)obj, mit->indexobj);
        Py_DECREF(obj);
    }

    if (sub == NULL) {
        goto fail;
    }
    mit->subspace = NpyArray_IterNew(PyArray_ARRAY(sub));
    Py_DECREF(sub);
    if (mit->subspace == NULL) {
        goto fail;
    }
    /* Expand dimensions of result */
    n = mit->subspace->ao->nd;
    for (i = 0; i < n; i++) {
        mit->dimensions[mit->nd+i] = mit->subspace->ao->dimensions[i];
    }
    mit->nd += n;

    /*
     * Now, we still need to interpret the ellipsis and slice objects
     * to determine which axes the indexing arrays are referring to
     */
    n = PyTuple_GET_SIZE(mit->indexobj);
    /* The number of dimensions an ellipsis takes up */
    ellipexp = PyArray_NDIM(arr) - n + 1;
    /*
     * Now fill in iteraxes -- remember indexing arrays have been
     * converted to 0's in mit->indexobj
     */
    curraxis = 0;
    j = 0;
    /* Only expand the first ellipsis */
    noellip = 1;
    memset(mit->bscoord, 0, sizeof(intp)*PyArray_NDIM(arr));
    for (i = 0; i < n; i++) {
        /*
         * We need to fill in the starting coordinates for
         * the subspace
         */
        obj = PyTuple_GET_ITEM(mit->indexobj, i);
        if (PyInt_Check(obj) || PyLong_Check(obj)) {
            mit->iteraxes[j++] = curraxis++;
        }
        else if (noellip && obj == Py_Ellipsis) {
            curraxis += ellipexp;
            noellip = 0;
        }
        else {
            intp start = 0;
            intp stop, step;
            /* Should be slice object or another Ellipsis */
            if (obj == Py_Ellipsis) {
                mit->bscoord[curraxis] = 0;
            }
            else if (!PySlice_Check(obj) ||
                     (slice_GetIndices((PySliceObject *)obj,
                                       PyArray_DIM(arr, curraxis),
                                       &start, &stop, &step,
                                       &dimsize) < 0)) {
                PyErr_Format(PyExc_ValueError,
                             "unexpected object "       \
                             "(%s) in selection position %d",
                             Py_TYPE(obj)->tp_name, i);
                goto fail;
            }
            else {
                mit->bscoord[curraxis] = start;
            }
            curraxis += 1;
        }
    }

    
 finish:
    /* Here check the indexes (now that we have iteraxes) */
    mit->size = PyArray_OverflowMultiplyList(mit->dimensions, mit->nd);
    if (mit->size < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "dimensions too large in fancy indexing");
        goto fail;
    }
    if (mit->ait->size == 0 && mit->size != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "invalid index into a 0-size array");
        goto fail;
    }

    for (i = 0; i < mit->numiter; i++) {
        intp indval;
        it = mit->iters[i];
        NpyArray_ITER_RESET(it);
        dimsize = PyArray_DIM(arr, mit->iteraxes[i]);
        while (it->index < it->size) {
            indptr = ((intp *)it->dataptr);
            indval = *indptr;
            if (indval < 0) {
                indval += dimsize;
            }
            if (indval < 0 || indval >= dimsize) {
                PyErr_Format(PyExc_IndexError,
                             "index (%"INTP_FMT") out of range "\
                             "(0<=index<%"INTP_FMT") in dimension %d",
                             indval, (dimsize-1), mit->iteraxes[i]);
                goto fail;
            }
            NpyArray_ITER_NEXT(it);
        }
        NpyArray_ITER_RESET(it);
    }
    return;

 fail:
    _Npy_XDECREF(mit->subspace);
    _Npy_XDECREF(mit->ait);
    mit->subspace = NULL;
    mit->ait = NULL;
    return;
}


NPY_NO_EXPORT PyObject *
PyArray_MapIterNew(PyObject *indexobj, int oned, int fancy)
{
    PyArrayMapIterObject *pyMit;
    NpyArrayMapIterObject *mit;
    PyObject *arr = NULL;
    int i, n, started, nonindex;

    if (fancy == SOBJ_BADARRAY) {
        PyErr_SetString(PyExc_IndexError,                       \
                        "arrays used as indices must be of "    \
                        "integer (or boolean) type");
        return NULL;
    }
    if (fancy == SOBJ_TOOMANY) {
        PyErr_SetString(PyExc_IndexError, "too many indices");
        return NULL;
    }

    /* This is the core iterator object - not a Python object. */
    mit = NpyArray_MapIterNew();
    if (NULL == mit) {
        _Npy_DECREF(mit);
        return NULL;
    }
    pyMit = Npy_INTERFACE(mit);
    
    /* Move the held reference from the core obj to the interface obj. */
    Py_INCREF(pyMit);
    _Npy_DECREF(mit);
    
    /* TODO: Refactor away the use of Py object for indexobj. */
    Py_INCREF(indexobj);
    mit->indexobj = indexobj;

    if (fancy == SOBJ_LISTTUP) {
        PyObject *newobj;
        newobj = PySequence_Tuple(indexobj);
        if (newobj == NULL) {
            goto fail;
        }
        Py_DECREF(indexobj);
        indexobj = newobj;
        mit->indexobj = indexobj;
    }

#undef SOBJ_NOTFANCY
#undef SOBJ_ISFANCY
#undef SOBJ_BADARRAY
#undef SOBJ_TOOMANY
#undef SOBJ_LISTTUP

    if (oned) {
        return (PyObject *)pyMit;
    }
    /*
     * Must have some kind of fancy indexing if we are here
     * indexobj is either a list, an arrayobject, or a tuple
     * (with at least 1 list or arrayobject or Bool object)
     */

    /* convert all inputs to iterators */
    if (PyArray_Check(indexobj) && (PyArray_TYPE(indexobj) == PyArray_BOOL)) {
        mit->numiter = _nonzero_indices(indexobj, mit->iters);
        if (mit->numiter < 0) {
            goto fail;
        }
        mit->nd = 1;
        mit->dimensions[0] = mit->iters[0]->dims_m1[0]+1;
        Py_DECREF(mit->indexobj);
        mit->indexobj = PyTuple_New(mit->numiter);
        if (mit->indexobj == NULL) {
            goto fail;
        }
        for (i = 0; i < mit->numiter; i++) {
            PyTuple_SET_ITEM(mit->indexobj, i, PyInt_FromLong(0));
        }
    }

    else if (PyArray_Check(indexobj) || !PyTuple_Check(indexobj)) {
        NpyArray_Descr *indtype;

        mit->numiter = 1;
        indtype = NpyArray_DescrFromType(PyArray_INTP);
        arr = PyArray_FromAnyUnwrap(indexobj, indtype, 0, 0, FORCECAST, NULL);
        if (arr == NULL) {
            goto fail;
        }
        mit->iters[0] = NpyArray_IterNew(PyArray_ARRAY(arr));
        if (mit->iters[0] == NULL) {
            Py_DECREF(arr);
            goto fail;
        }
        mit->nd = PyArray_NDIM(arr);
        memcpy(mit->dimensions, PyArray_DIMS(arr), mit->nd*sizeof(intp));
        mit->size = PyArray_SIZE(arr);
        Py_DECREF(arr);
        Py_DECREF(mit->indexobj);
        mit->indexobj = Py_BuildValue("(N)", PyInt_FromLong(0));
    }
    else {
        /* must be a tuple */
        PyObject *obj;
        NpyArrayIterObject **iterp;
        PyObject *new;
        int numiters, j, n2;
        /*
         * Make a copy of the tuple -- we will be replacing
         * index objects with 0's
         */
        n = PyTuple_GET_SIZE(indexobj);
        n2 = n;
        new = PyTuple_New(n2);
        if (new == NULL) {
            goto fail;
        }
        started = 0;
        nonindex = 0;
        j = 0;
        for (i = 0; i < n; i++) {
            obj = PyTuple_GET_ITEM(indexobj,i);
            iterp = mit->iters + mit->numiter;
            if ((numiters=_convert_obj(obj, iterp)) < 0) {
                Py_DECREF(new);
                goto fail;
            }
            if (numiters > 0) {
                started = 1;
                if (nonindex) {
                    mit->consec = 0;
                }
                mit->numiter += numiters;
                if (numiters == 1) {
                    PyTuple_SET_ITEM(new,j++, PyInt_FromLong(0));
                }
                else {
                    /*
                     * we need to grow the new indexing object and fill
                     * it with 0s for each of the iterators produced
                     */
                    int k;
                    n2 += numiters - 1;
                    if (_PyTuple_Resize(&new, n2) < 0) {
                        goto fail;
                    }
                    for (k = 0; k < numiters; k++) {
                        PyTuple_SET_ITEM(new, j++, PyInt_FromLong(0));
                    }
                }
            }
            else {
                if (started) {
                    nonindex = 1;
                }
                Py_INCREF(obj);
                PyTuple_SET_ITEM(new,j++,obj);
            }
        }
        Py_DECREF(mit->indexobj);
        mit->indexobj = new;
        /*
         * Store the number of iterators actually converted
         * These will be mapped to actual axes at bind time
         */
        if (NpyArray_Broadcast((NpyArrayMultiIterObject *)mit) < 0) {
            goto fail;
        }
    }

    return (PyObject *)pyMit;

 fail:
    Py_DECREF(pyMit);
    return NULL;
}


static void
arraymapiter_dealloc(PyArrayMapIterObject *mit)
{
    Npy_DEALLOC(mit->iter);
    _pya_free(mit);
}

/*
 * The mapiter object must be created new each time.  It does not work
 * to bind to a new array, and continue.
 *
 * This was the orginal intention, but currently that does not work.
 * Do not expose the MapIter_Type to Python.
 *
 * It's not very useful anyway, since mapiter(indexobj); mapiter.bind(a);
 * mapiter is equivalent to a[indexobj].flat but the latter gets to use
 * slice syntax.
 */
NPY_NO_EXPORT PyTypeObject PyArrayMapIter_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.mapiter",                            /* tp_name */
    sizeof(PyArrayIterObject),                  /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)arraymapiter_dealloc,           /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
    0,                                          /* tp_version_tag */
#endif
};

/** END of Subscript Iterator **/


