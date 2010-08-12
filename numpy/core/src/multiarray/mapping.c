#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

/*#include <stdio.h>*/
#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/numpy_api.h"
#include "numpy/npy_iterators.h"
#include "npy_dict.h"
#include "numpy/npy_index.h"

#include "common.h"
#include "ctors.h"
#include "iterators.h"
#include "arrayobject.h"
#include "mapping.h"
#include "conversion_utils.h"

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
    result->indexobj = NULL;

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


static PyObject *
PyArray_GetMap(PyArrayMapIterObject *pyMit)
{
    NpyArray* result = NpyArray_GetMap(pyMit->iter);
    RETURN_PYARRAY(result);
}

static int
PyArray_SetMap(PyArrayMapIterObject *pyMit, PyObject *op)
{
    PyObject *arr;
    int result;
    NpyArray_Descr *descr = pyMit->iter->ait->ao->descr;

    _Npy_INCREF(descr);
    arr = PyArray_FromAnyUnwrap(op, descr, 0, 0, FORCECAST, NULL);
    if (arr == NULL) {
        return -1;
    }

    result = NpyArray_SetMap(pyMit->iter, PyArray_ARRAY(arr));
    Py_DECREF(arr);
    return result;
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

static PyObject *
fancy_index(PyObject* index, npy_bool* pfancy)
{
    int fancy = fancy_indexing_check(index);

    switch (fancy) {
    case SOBJ_BADARRAY:
        *pfancy = NPY_FALSE;
        PyErr_SetString(PyExc_IndexError,
                        "arrays used as indices must be of "
                        "integer (or boolean) type");
        return NULL;
        break;
    case SOBJ_TOOMANY:
        *pfancy = NPY_FALSE;
        PyErr_SetString(PyExc_IndexError, "too many indices");
        return NULL;
        break;
    case SOBJ_ISFANCY:
        *pfancy = NPY_TRUE;
        Py_INCREF(index);
        return index;
        break;
    case SOBJ_LISTTUP:
        *pfancy = NPY_TRUE;
        return PySequence_Tuple(index);
        break;
    default:
        assert(NPY_FALSE);
        /*FALLTHROUGH*/
    case SOBJ_NOTFANCY:
        *pfancy = NPY_FALSE;
        Py_INCREF(index);
        return index;
    }
}

#undef SOBJ_NOTFANCY
#undef SOBJ_ISFANCY
#undef SOBJ_BADARRAY
#undef SOBJ_TOOMANY
#undef SOBJ_LISTTUP

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
    NpyIndex indexes[NPY_MAXDIMS];
    int n;

    n = PyArray_IndexConverter(op, indexes);
    if (n < 0) {
        return NULL;
    }

    RETURN_PYARRAY(NpyArray_IndexSimple(PyArray_ARRAY(self), indexes, n));
}

static npy_bool
is_multi_fields(NpyIndex *indexes, int n)
{
    if (n <= 1) {
        return NPY_FALSE;
    } else {
        int i;

        for (i=0; i<n; i++) {
            if (indexes[i].type != NPY_INDEX_STRING) {
                return NPY_FALSE;
            }
        }
        return NPY_TRUE;
    }
}

NPY_NO_EXPORT PyObject *
array_subscript(PyArrayObject *self, PyObject *op)
{
    NpyArray *result;
    NpyIndex indexes[NPY_MAXDIMS];
    int n;

    n = PyArray_IndexConverter(op, indexes);
    if (n < 0) {
        return NULL;
    }

    /*
     * Special case for multiple fields since we call into python.
     * NOTE: For backwards compatibility we check to make sure
     * op is not a tuple before treating it as multiple fields.
     * As far as I can tell there is no good reason for this check.
     */
    if (is_multi_fields(indexes, n) && !PyTuple_Check(op)) {
        PyObject *_numpy_internal;
        PyObject *obj;

        _numpy_internal = PyImport_ImportModule("numpy.core._internal");
        if (_numpy_internal == NULL) {
            return NULL;
        }
        obj = PyObject_CallMethod(_numpy_internal,
                                  "_index_fields", "OO", self, op);
        Py_DECREF(_numpy_internal);
        NpyArray_IndexDealloc(indexes, n);
        return obj;
    }

    /* Otherwise call NpyArray_Subscript. */
    result = NpyArray_Subscript(PyArray_ARRAY(self), indexes, n);
    NpyArray_IndexDealloc(indexes, n);
    RETURN_PYARRAY(result);
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
    int ret, oned;
    intp vals[MAX_DIMS];
    PyObject* new_index;
    npy_bool is_fancy;

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

    new_index = fancy_index(index, &is_fancy);
    if (new_index == NULL) {
        return -1;
    }

    if (is_fancy) {

        oned = ((PyArray_NDIM(self) == 1) &&
                !(PyTuple_Check(new_index) && PyTuple_GET_SIZE(new_index) > 1));
        if (oned) {
            NpyArrayIterObject *it;
            int rval;

            it = NpyArray_IterNew(PyArray_ARRAY(self));
            if (it == NULL) {
                Py_DECREF(new_index);
                return -1;
            }
            rval = npy_iter_ass_subscript(it, new_index, op);
            _Npy_DECREF(it);
            Py_DECREF(new_index);
            return rval;
        }
        else {
            PyArrayMapIterObject *mit;

            mit = (PyArrayMapIterObject *) PyArray_MapIterNew(new_index);
            if (mit == NULL) {
                Py_DECREF(new_index);
                return -1;
            }
            if (PyArray_MapIterBind(mit, self) < 0) {
                Py_DECREF(mit);
                Py_DECREF(new_index);
                return -1;
            }
            ret = PyArray_SetMap(mit, op);
            Py_DECREF(mit);
            Py_DECREF(new_index);
            return ret;
        }
    }
    else {
        int result;

        result = array_ass_sub_simple(self, new_index, op);
        Py_DECREF(new_index);
        return result;
    }
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
NPY_NO_EXPORT int
PyArray_MapIterBind(PyArrayMapIterObject *pyMit, PyArrayObject *arr)
{
    int result;
    PyObject* true_array;

    if (PyArray_CheckExact(arr)) {
        true_array = (PyObject*) arr;
        Py_INCREF(arr);
    } else {
        Py_INCREF(arr);
        true_array = PyArray_EnsureArray((PyObject *)arr);
        if (true_array == NULL) {
            return -1;
        }
    }
    result = NpyArray_MapIterBind(pyMit->iter, PyArray_ARRAY(arr),
                                  PyArray_ARRAY(true_array));
    Py_DECREF(true_array);
    return result;
}

/*
 * Creates a new MapIter from an indexob.  Assumes that the
 * index has already been processed with fancy_index.
 *
 * Sets iteraxes to the indexes in indexobj that has been converted
 * to iterators.  PyArray_MapIterBind will change these into
 * indexes into the array, taking into account ellipses.
 */
NPY_NO_EXPORT PyObject *
PyArray_MapIterNew(PyObject *indexobj)
{
    NpyIndex indexes[NPY_MAXDIMS];
    NpyArrayMapIterObject *mit;
    PyArrayMapIterObject *pyMit;
    int n;

    n = PyArray_IndexConverter(indexobj, indexes);
    if (n < 0) {
        return NULL;
    }
    mit = NpyArray_MapIterNew(indexes, n);
    if (mit == NULL) {
        return NULL;
    }

    /* Move the held reference and return the python object. */
    pyMit = Npy_INTERFACE(mit);
    Py_INCREF(pyMit);
    _Npy_DECREF(mit);

    return (PyObject *)pyMit;
}


static void
arraymapiter_dealloc(PyArrayMapIterObject *mit)
{
    Py_XDECREF(mit->indexobj);
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


