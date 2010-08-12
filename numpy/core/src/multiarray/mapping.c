#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

/*#include <stdio.h>*/
#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "npy_api.h"
#include "numpy/npy_iterators.h"
#include "npy_dict.h"
#include "numpy/npy_index.h"

#include "common.h"
#include "ctors.h"
#include "iterators.h"
#include "arrayobject.h"
#include "mapping.h"
#include "conversion_utils.h"

#define ASSERT_ONE_BASE(r) \
    assert(NULL == PyArray_BASE_ARRAY(r) || NULL == PyArray_BASE(r))


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


static int try_single_assign(NpyArray  *self, NpyIndex*  indexes,
                             int n, PyObject* val)
{
    int i, j, n_bound, result;
    NpyIndex bound_indexes[NPY_MAXDIMS];
    npy_intp offset;

    /* Check to make sure we have only indexes and newaxis/ellipses. */
    for (i=0; i<n; i++) {
        switch (indexes[i].type) {
        case NPY_INDEX_NEWAXIS:
        case NPY_INDEX_ELLIPSIS:
        case NPY_INDEX_INTP:
        case NPY_INDEX_BOOL:
            break;
        default:
            return 0;
        }
    }

    /* Bind the indexes. */
    n_bound = NpyArray_IndexBind(indexes, n,
                                 self->dimensions, self->nd,
                                 bound_indexes);
    if (n_bound < 0) {
        return -1;
    }

    /* Loop through and calculate the offset of the data. */
    offset = 0;
    j = 0;
    for (i=0; i<n_bound; i++) {
        NpyIndex *index = &bound_indexes[i];
        switch (index->type) {
        case NPY_INDEX_NEWAXIS:
            break;
        case NPY_INDEX_INTP:
#undef intp
            offset += self->strides[j++]*index->index.intp;
#define intp npy_intp
            break;
        case NPY_INDEX_SLICE:
            /* Not a single index. */
            NpyArray_IndexDealloc(bound_indexes, n_bound);
            return 0;
            break;
        default:
            assert(NPY_FALSE);
            NpyArray_IndexDealloc(bound_indexes, n_bound);
            return -1;
        }
    }
    NpyArray_IndexDealloc(bound_indexes, n_bound);
    if (j != self->nd) {
        /* This is not a single assignment. */
        return 0;
    }

    /* Now use setitem to set the item. */
    result = self->descr->f->setitem(val, self->data+offset, self);
    if (result < 0) {
        return -1;
    } else {
        return 1;
    }
}

/*
 * Determine if this is a simple index.
 */
static npy_bool
is_simple(NpyIndex *indexes, int n)
{
    int i;

    for (i=0; i<n; i++) {
        switch (indexes[i].type) {
        case NPY_INDEX_INTP_ARRAY:
        case NPY_INDEX_BOOL_ARRAY:
        case NPY_INDEX_STRING:
            return NPY_FALSE;
            break;
        default:
            break;
        }
    }

    return NPY_TRUE;
}


static int
array_ass_sub(PyArrayObject *self, PyObject *index, PyObject *op)
{
    NpyIndex indexes[NPY_MAXDIMS];
    int n, result;
    NpyArray *view = NULL;

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

    n = PyArray_IndexConverter(index, indexes);
    if (n < 0) {
        return -1;
    }

    if (n == 1 && indexes[0].type == NPY_INDEX_STRING) {
        /* This is an assignment to a field. Use
           PyArray_SetField to handle compatability
           with numeric in the case of string assignment. */
        if (PyArray_DESCR(self)->names) {
            NpyArray_DescrField *value;

            value = NpyDict_Get(PyArray_DESCR(self)->fields, 
                                indexes[0].index.string);
            if (NULL != value) {
                PyArray_Descr *descrWrap = PyArray_Descr_WRAP(value->descr);
                Py_INCREF(descrWrap);
                result = PyArray_SetField(self, descrWrap,
                                          value->offset, op);
                goto finish;
            }
        }

        PyErr_Format(PyExc_ValueError,
                     "field named %s not found.",
                     indexes[0].index.string);
        result = -1;
        goto finish;
    }


    /* Special case for bool index on 0-d arrays. */
    if (PyArray_NDIM(self) == 0 &&
        n == 1 && indexes[0].type == NPY_INDEX_BOOL) {
        NpyArray_IndexDealloc(indexes, n);
        if (indexes[0].index.boolean) {
            result = PyArray_DESCR(self)->f->setitem(op, PyArray_BYTES(self),
                                                     PyArray_ARRAY(self));
            goto finish;
        } else {
            /* Do nothing. */
            result = 0;
            goto finish;
        }
    }

    /* Look for a single item assignment. */
    result = try_single_assign(PyArray_ARRAY(self), indexes, n, op);
    if (result < 0) {
        goto finish;
    }
    if (result == 1) {
        /* Assignement succeeded. */
        result = 0;
        goto finish;
    }


    if (is_simple(indexes, n)) {
        if (PyArray_CheckExact(self)) {
            /* Do a PyArray_CopyObject onto a view into the array */
            view = NpyArray_IndexSimple(PyArray_ARRAY(self),
                                        indexes, n);
            if (view == NULL) {
                result = -1;
                goto finish;
            }
        } else {
            PyObject *tmp0;
            /* TODO: Why only in this case to get call PyObject_GetItem?
               It seems inconsistent. */
            tmp0 = PyObject_GetItem((PyObject *)self, index);
            if (tmp0 == NULL) {
                result = -1;
                goto finish;
            }
            if (!PyArray_Check(tmp0)) {
                PyErr_SetString(PyExc_RuntimeError,
                                "Getitem not returning array.");
                Py_DECREF(tmp0);
                result = -1;
                goto finish;
            }
            view = PyArray_ARRAY(tmp0);
            _Npy_INCREF(view);
            Py_DECREF(tmp0);
        }

        result = PyArray_CopyObject(Npy_INTERFACE(view), op);

    } else {
        /* Use a MapIter to do the fancy indexing. */
        PyObject *converted_value;

        _Npy_INCREF(PyArray_DESCR(self));
        converted_value = PyArray_FromAnyUnwrap(op, PyArray_DESCR(self),
                                                0, 0, NPY_FORCECAST, NULL);
        if (converted_value == NULL) {
            result = -1;
            goto finish;
        }

        result = NpyArray_IndexFancyAssign(PyArray_ARRAY(self),
                                           indexes, n,
                                           PyArray_ARRAY(converted_value));
        Py_DECREF(converted_value);
    }

 finish:
    NpyArray_IndexDealloc(indexes, n);
    _Npy_XDECREF(view);

    return result;
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


