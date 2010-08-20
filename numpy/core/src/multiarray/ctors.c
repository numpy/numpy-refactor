#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "npy_api.h"

#include "npy_math.h"

#include "npy_config.h"

#include "numpy_3kcompat.h"

#include "common.h"

#include "ctors.h"

#include "buffer.h"

#include "arrayobject.h"

#include "numpymemoryview.h"


NPY_NO_EXPORT NpyArray *
PyArray_FromScalarUnwrap(PyObject *scalar, NpyArray_Descr *outcode);


#if PY_VERSION_HEX < 0x02070000
int
PyCapsule_CheckExact(PyObject *ptr)
{
    return PyCObject_Check(ptr);
}

void *
PyCapsule_GetPointer(void *ptr, void *notused)
{
    return PyCObject_AsVoidPtr(ptr);
}

#endif


/*
 * If s is not a list, return 0
 * Otherwise:
 *
 * run object_depth_and_dimension on all the elements
 * and make sure the returned shape and size is the
 * same for each element
 */
static int
object_depth_and_dimension(PyObject *s, int max, intp *dims)
{
    intp *newdims, *test_dims;
    int nd, test_nd;
    int i, islist, istuple;
    intp size;
    PyObject *obj;

    islist = PyList_Check(s);
    istuple = PyTuple_Check(s);
    if (!(islist || istuple)) {
        return 0;
    }

    size = PySequence_Size(s);
    if (size == 0) {
        return 0;
    }
    if (max < 1) {
        return 0;
    }
    if (max < 2) {
        dims[0] = size;
        return 1;
    }

    newdims = PyDimMem_NEW(2 * (max - 1));
    test_dims = newdims + (max - 1);
    if (islist) {
        obj = PyList_GET_ITEM(s, 0);
    }
    else {
        obj = PyTuple_GET_ITEM(s, 0);
    }
    nd = object_depth_and_dimension(obj, max - 1, newdims);

    for (i = 1; i < size; i++) {
        if (islist) {
            obj = PyList_GET_ITEM(s, i);
        }
        else {
            obj = PyTuple_GET_ITEM(s, i);
        }
        test_nd = object_depth_and_dimension(obj, max - 1, test_dims);

        if ((nd != test_nd) ||
            (!PyArray_CompareLists(newdims, test_dims, nd))) {
            nd = 0;
            break;
        }
    }

    for (i = 1; i <= nd; i++) {
        dims[i] = newdims[i - 1];
    }
    dims[0] = size;
    PyDimMem_FREE(newdims);
    return nd + 1;
}


/* If numitems > 1, then dst must be contiguous */
NPY_NO_EXPORT void
copy_and_swap(void *dst, void *src, int itemsize, intp numitems,
              intp srcstrides, int swap)
{
    intp i;
    char *s1 = (char *)src;
    char *d1 = (char *)dst;

    if ((numitems == 1) || (itemsize == srcstrides)) {
        memcpy(d1, s1, itemsize * numitems);
    }
    else {
        for (i = 0; i < numitems; i++) {
            memcpy(d1, s1, itemsize);
            d1 += itemsize;
            s1 += srcstrides;
        }
    }

    if (swap) {
        npy_byte_swap_vector(d1, numitems, itemsize);
    }
}





/*NUMPY_API
 * Move the memory of one array into another.
 */
NPY_NO_EXPORT int
PyArray_MoveInto(PyArrayObject *dest, PyArrayObject *src)
{
    return NpyArray_MoveInto(PyArray_ARRAY(dest),
                             PyArray_ARRAY(src));
}



/* adapted from Numarray */
static int
setArrayFromSequence(PyArrayObject *a, PyObject *s, int dim, intp offset)
{
    Py_ssize_t i, slen;
    int res = -1;

    /*
     * This code is to ensure that the sequence access below will
     * return a lower-dimensional sequence.
     */

    /* INCREF on entry DECREF on exit */
    Py_INCREF(s);

    if (PyArray_Check(s) && !(PyArray_CheckExact(s))) {
      /*
       * FIXME:  This could probably copy the entire subarray at once here using
       * a faster algorithm.  Right now, just make sure a base-class array is
       * used so that the dimensionality reduction assumption is correct.
       */
        /* This will DECREF(s) if replaced */
        s = PyArray_EnsureArray(s);
    }

    if (dim > PyArray_NDIM(a)) {
        PyErr_Format(PyExc_ValueError, "setArrayFromSequence: "
                     "sequence/array dimensions mismatch.");
        goto fail;
    }

    slen = PySequence_Length(s);
    if (slen != PyArray_DIM(a, dim)) {
        PyErr_Format(PyExc_ValueError,
                     "setArrayFromSequence: sequence/array shape mismatch.");
        goto fail;
    }

    for (i = 0; i < slen; i++) {
        PyObject *o = PySequence_GetItem(s, i);
        if ((PyArray_NDIM(a) - dim) > 1) {
            res = setArrayFromSequence(a, o, dim + 1, offset);
        }
        else {
            res = PyArray_DESCR(a)->f->setitem(o, (PyArray_BYTES(a) + offset),
                                               PyArray_ARRAY(a));
        }
        Py_DECREF(o);
        if (res < 0) {
            goto fail;
        }
        offset += PyArray_STRIDE(a, dim);
    }

    Py_DECREF(s);
    return 0;

 fail:
    Py_DECREF(s);
    return res;
}

static int
Assign_Array(PyArrayObject *self, PyObject *v)
{
    if (!PySequence_Check(v)) {
        PyErr_SetString(PyExc_ValueError, "assignment from non-sequence");
        return -1;
    }
    if (PyArray_NDIM(self) == 0) {
        PyErr_SetString(PyExc_ValueError, "assignment to 0-d array");
        return -1;
    }
    return setArrayFromSequence(self, v, 0, 0);
}

/*
 * "Array Scalars don't call this code"
 * steals reference to typecode -- no NULL
 */
static PyObject *
Array_FromPyScalar(PyObject *op, NpyArray_Descr *typecode)
{
    PyArrayObject *ret;
    int itemsize;
    int type;

    itemsize = typecode->elsize;
    type = typecode->type_num;

    if (itemsize == 0 && NpyTypeNum_ISEXTENDED(type)) {
        itemsize = PyObject_Length(op);
        if (type == PyArray_UNICODE) {
            itemsize *= 4;
        }
        if (itemsize != typecode->elsize) {
            NpyArray_DESCR_REPLACE(typecode);
            typecode->elsize = itemsize;
        }
    }

    ASSIGN_TO_PYARRAY(ret,
                      NpyArray_Alloc(typecode, 0, NULL, NPY_FALSE, NULL));
    if (NULL == ret) {
        return NULL;
    }

    if (PyArray_NDIM(ret) > 0) {
        PyErr_SetString(PyExc_ValueError,
                        "shape-mismatch on array construction");
        Py_DECREF(ret);
        return NULL;
    }

    PyArray_DESCR(ret)->f->setitem(op, PyArray_BYTES(ret), PyArray_ARRAY(ret));
    if (PyErr_Occurred()) {
        Py_DECREF(ret);
        return NULL;
    }
    else {
        return (PyObject *)ret;
    }
}


static PyObject *
ObjectArray_FromNestedList(PyObject *s, NpyArray_Descr *typecode, int fortran)
{
    int nd;
    intp d[MAX_DIMS];
    PyArrayObject *r;
    NpyArray *arr;

    /* Get the depth and the number of dimensions */
    nd = object_depth_and_dimension(s, MAX_DIMS, d);
    if (nd < 0) {
        return NULL;
    }
    if (nd == 0) {
        return Array_FromPyScalar(s, typecode);
    }
    arr = NpyArray_Alloc(typecode, nd, d, fortran, NULL);
    if (!arr) {
        return NULL;
    }
    r = Npy_INTERFACE(arr);
    Py_INCREF(r);
    Npy_DECREF(arr);

    if(Assign_Array(r,s) == -1) {
        Py_DECREF(r);
        return NULL;
    }
    return (PyObject *)r;
}

/*
 * The rest of this code is to build the right kind of array
 * from a python object.
 */

static int
discover_depth(PyObject *s, int max, int stop_at_string, int stop_at_tuple)
{
    int d = 0;
    PyObject *e;
#if PY_VERSION_HEX >= 0x02060000
    Py_buffer buffer_view;
#endif

    if(max < 1) {
        return -1;
    }

    /* Fast path for lists. */
    if (PyList_CheckExact(s)) {
        if (PyList_GET_SIZE(s) == 0) {
            return 1;
        }
        else {
            e = PyList_GET_ITEM(s, 0);
            if (e == NULL) {
                return -1;
            }
            if (e != s) {
                d = discover_depth(e, max - 1, stop_at_string, stop_at_tuple);
                if (d >= 0) {
                    d++;
                }
            }
            return d;
        }
    }

    /* Fast path for tuples. */
    if (PyTuple_CheckExact(s)) {
        if (stop_at_tuple) {
            return 0;
        }
        if (PyTuple_GET_SIZE(s) == 0) {
            return 1;
        }
        else {
            e = PyTuple_GET_ITEM(s, 0);
            if (e == NULL) {
                return -1;
            }
            if (e != s) {
                d = discover_depth(e, max - 1, stop_at_string, stop_at_tuple);
                if (d >= 0) {
                    d++;
                }
            }
            return d;
        }
    }


    if(!PySequence_Check(s) ||
#if defined(NPY_PY3K)
       /* FIXME: XXX -- what is the correct thing to do here? */
#else
       PyInstance_Check(s) ||
#endif
       PySequence_Length(s) < 0) {
        PyErr_Clear();
        return 0;
    }
    if (PyArray_Check(s)) {
        return PyArray_NDIM(s);
    }
    if (PyArray_IsScalar(s, Generic)) {
        return 0;
    }
    if (PyString_Check(s) ||
#if defined(NPY_PY3K)
#else
        PyBuffer_Check(s) ||
#endif
        PyUnicode_Check(s)) {
        return stop_at_string ? 0:1;
    }
    if (stop_at_tuple && PyTuple_Check(s)) {
        return 0;
    }
#if PY_VERSION_HEX >= 0x02060000
    /* PEP 3118 buffer interface */
    memset(&buffer_view, 0, sizeof(Py_buffer));
    if (PyObject_GetBuffer(s, &buffer_view, PyBUF_STRIDES) == 0 ||
        PyObject_GetBuffer(s, &buffer_view, PyBUF_ND) == 0) {
        d = buffer_view.ndim;
        PyBuffer_Release(&buffer_view);
        return d;
    }
    else if (PyObject_GetBuffer(s, &buffer_view, PyBUF_SIMPLE) == 0) {
        PyBuffer_Release(&buffer_view);
        return 1;
    }
    else {
        PyErr_Clear();
    }
#endif
    if ((e = PyObject_GetAttrString(s, "__array_struct__")) != NULL) {
        d = -1;
        if (PyCapsule_Check(e)) {
            PyArrayInterface *inter;
            inter = (PyArrayInterface *)PyCapsule_AsVoidPtr(e);
            if (inter->two == 2) {
                d = inter->nd;
            }
        }
        Py_DECREF(e);
        if (d > -1) {
            return d;
        }
    }
    else {
        PyErr_Clear();
    }
    if ((e=PyObject_GetAttrString(s, "__array_interface__")) != NULL) {
        d = -1;
        if (PyDict_Check(e)) {
            PyObject *new;
            new = PyDict_GetItemString(e, "shape");
            if (new && PyTuple_Check(new)) {
                d = PyTuple_GET_SIZE(new);
            }
        }
        Py_DECREF(e);
        if (d > -1) {
            return d;
        }
    }
    else PyErr_Clear();

    if (PySequence_Length(s) == 0) {
        return 1;
    }
    if ((e=PySequence_GetItem(s, 0)) == NULL) {
        return -1;
    }
    if (e != s) {
        d = discover_depth(e, max - 1, stop_at_string, stop_at_tuple);
        if (d >= 0) {
            d++;
        }
    }
    Py_DECREF(e);
    return d;
}


static int
discover_itemsize(PyObject *s, int nd, int *itemsize)
{
    int n, r, i;
    PyObject *e;

    if (PyArray_Check(s)) {
        *itemsize = MAX(*itemsize, PyArray_ITEMSIZE(s));
        return 0;
    }

    n = PyObject_Length(s);
    if ((nd == 0) || PyString_Check(s) ||
#if defined(NPY_PY3K)
        PyMemoryView_Check(s) ||
#else
        PyBuffer_Check(s) ||
#endif
        PyUnicode_Check(s)) {

        *itemsize = MAX(*itemsize, n);
        return 0;
    }
    for (i = 0; i < n; i++) {
        if ((e = PySequence_GetItem(s,i))==NULL) {
            return -1;
        }
        r = discover_itemsize(e, nd - 1, itemsize);
        Py_DECREF(e);
        if (r == -1) {
            return -1;
        }
    }
    return 0;
}


/*
 * Take an arbitrary object known to represent
 * an array of ndim nd, and determine the size in each dimension
 */
static int
discover_dimensions(PyObject *s, int nd, intp *d, int check_it)
{
    PyObject *e;
    int r, n, i, n_lower;

    if (PyArray_Check(s)) {
        /*
         * XXXX: we handle the case of scalar arrays (0 dimensions) separately.
         * This is an hack, the function discover_dimensions needs to be
         * improved.
         */
        if (PyArray_NDIM(s) == 0) {
            d[0] = 0;
        } else {
            for (i=0; i<nd; i++) {
                d[i] = PyArray_DIM(s,i);
            }
        }
        return 0;
    }
    n = PyObject_Length(s);
    *d = n;
    if (*d < 0) {
        return -1;
    }
    if (nd <= 1) {
        return 0;
    }
    n_lower = 0;
    for(i = 0; i < n; i++) {
        if ((e = PySequence_GetItem(s,i)) == NULL) {
            return -1;
        }
        r = discover_dimensions(e, nd - 1, d + 1, check_it);
        Py_DECREF(e);

        if (r == -1) {
            return -1;
        }
        if (check_it && n_lower != 0 && n_lower != d[1]) {
            PyErr_SetString(PyExc_ValueError,
                            "inconsistent shape in sequence");
            return -1;
        }
        if (d[1] > n_lower) {
            n_lower = d[1];
        }
    }
    d[1] = n_lower;

    return 0;
}

/*
 * isobject means that we are constructing an
 * object array on-purpose with a nested list.
 * Only a list is interpreted as a sequence with these rules
 * steals reference to typecode
 */
static PyObject *
Array_FromSequence(PyObject *s, NpyArray_Descr *typecode, int fortran,
                   int min_depth, int max_depth)
{
    PyArrayObject *r;
    NpyArray *arr;
    int nd;
    int err;
    intp d[MAX_DIMS];
    int stop_at_string;
    int stop_at_tuple;
    int check_it;
    int type = typecode->type_num;
    int itemsize = typecode->elsize;

    check_it = (typecode->type != PyArray_CHARLTR);
    stop_at_string = (type != PyArray_STRING) ||
                     (typecode->type == PyArray_STRINGLTR);
    stop_at_tuple = (type == PyArray_VOID && (typecode->names ||
                                              typecode->subarray));

    nd = discover_depth(s, MAX_DIMS + 1, stop_at_string, stop_at_tuple);
    if (nd == 0) {
        return Array_FromPyScalar(s, typecode);
    }
    else if (nd < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid input sequence");
        goto fail;
    }
    if (max_depth && NpyTypeNum_ISOBJECT(type) && (nd > max_depth)) {
        nd = max_depth;
    }
    if ((max_depth && nd > max_depth) || (min_depth && nd < min_depth)) {
        PyErr_SetString(PyExc_ValueError, "invalid number of dimensions");
        goto fail;
    }

    err = discover_dimensions(s, nd, d, check_it);
    if (err == -1) {
        goto fail;
    }
    if (typecode->type == PyArray_CHARLTR && nd > 0 && d[nd - 1] == 1) {
        nd = nd - 1;
    }

    if (itemsize == 0 && NpyTypeNum_ISEXTENDED(type)) {
        err = discover_itemsize(s, nd, &itemsize);
        if (err == -1) {
            goto fail;
        }
        if (type == PyArray_UNICODE) {
            itemsize *= 4;
        }
    }
    if (itemsize != typecode->elsize) {
        NpyArray_DESCR_REPLACE(typecode);
        typecode->elsize = itemsize;
    }

    arr = NpyArray_Alloc(typecode, nd, d, fortran, NULL);
    if (!arr) {
        return NULL;
    }

    r = Npy_INTERFACE(arr);
    Py_INCREF(r);
    Npy_DECREF(arr);

    err = Assign_Array(r,s);
    if (err == -1) {
        Py_DECREF(r);
        return NULL;
    }
    return (PyObject *)r;

 fail:
    Npy_DECREF(typecode);
    return NULL;
}


/*NUMPY_API
 * Generic new array creation routine.
 *
 * steals a reference to descr (even on failure)
 */
NPY_NO_EXPORT PyObject *
PyArray_NewFromDescr(PyTypeObject *subtype, PyArray_Descr *descr, int nd,
                     intp *dims, intp *strides, void *data,
                     int flags, PyObject *obj)
{
    NpyArray_Descr *descrCore;
    PyArrayObject *ret;

    /* Move stolen reference to core about. */
    PyArray_Descr_REF_TO_CORE(descr, descrCore);

    ASSIGN_TO_PYARRAY(ret,
        NpyArray_NewFromDescr(descrCore, nd, dims, strides,
                              data, flags, NPY_FALSE, subtype, obj));
    return (PyObject *)ret;
}


/* This isn't part of the API but is a support function for the core.
   It is called by NpyArray_NewFromDescr to initialize the Python wrapper
   object.
   TODO: I don't understand the function of customStrides and why the flag
   only gets set
   if __array_finalize__ is defined. */
NPY_NO_EXPORT int
NpyInterface_ArrayNewWrapper(NpyArray *newArray, int ensureArray,
                             int customStrides, void *subtypeTmp,
                             void *interfaceData, void **interfaceRet)
{
    PyTypeObject *subtype;
    PyArrayObject *wrapper = NULL;
    PyObject *obj = (PyObject *)interfaceData;


    /* Figures out the subtype, defaulting to a basic array.  Ugly, overly
       complex because the core doesn't know about PyArray_Type but needs
       to be able to force it. */
    if (NPY_TRUE == ensureArray) {
        subtype = &PyArray_Type;
    } else if (NULL != subtypeTmp) {
        subtype = (PyTypeObject *)subtypeTmp;
        assert(PyType_Check((PyObject *)subtype));
    } else if (NULL != interfaceData) {
        assert(PyArray_Check(interfaceData));
        subtype = Py_TYPE((PyObject *)interfaceData);
    } else {
        subtype = &PyArray_Type;
    }


    wrapper = (PyArrayObject *) subtype->tp_alloc(subtype, 0);
    if (wrapper == NULL) {
        goto fail;
    }
    wrapper->magic_number = NPY_VALID_MAGIC;
    wrapper->weakreflist = NULL;
    wrapper->array = newArray;

    /* For subclasses of array allows the classes to do type-specific
       initialization. */
    if ((subtype != &PyArray_Type)) {
        PyObject *res, *func, *args;

        func = PyObject_GetAttrString((PyObject *)wrapper,
                                      "__array_finalize__");
        if (func && func != Py_None) {
            if (customStrides) {
                /*
                 * did not allocate own data or funny strides
                 * update flags before finalize function
                 */
                PyArray_UpdateFlags(wrapper, NPY_UPDATE_ALL);
            }
            if (PyCapsule_Check(func)) {
                /* A C-function is stored here */
                PyArray_FinalizeFunc *cfunc;
                cfunc = PyCapsule_AsVoidPtr(func);
                Py_DECREF(func);
                if (cfunc(wrapper, obj) < 0) {
                    goto fail;
                }
            }
            else {
                args = PyTuple_New(1);
                if (obj == NULL) {
                    obj=Py_None;
                }
                Py_INCREF(obj);
                PyTuple_SET_ITEM(args, 0, obj);
                res = PyObject_Call(func, args, NULL);
                Py_DECREF(args);
                Py_DECREF(func);
                if (res == NULL) {
                    goto fail;
                }
                else {
                    Py_DECREF(res);
                }
            }
        }
        else Py_XDECREF(func);
    }
    *interfaceRet = wrapper;
    return NPY_TRUE;

fail:
    *interfaceRet = NULL;
    return NPY_FALSE;
}


/*NUMPY_API
 * Generic new array creation routine.
 */
NPY_NO_EXPORT PyObject *
PyArray_New(PyTypeObject *subtype, int nd, intp *dims, int type_num,
            intp *strides, void *data, int itemsize, int flags,
            PyObject *obj)
{
    NpyArray *arr;
    PyArrayObject *ret;

    arr = NpyArray_New(subtype, nd, dims, type_num,
                       strides, data, itemsize, flags, obj);
    if (arr == NULL) {
        return NULL;
    }
    ret = Npy_INTERFACE(arr);
    Py_INCREF(ret);
    Npy_DECREF(arr);

    return (PyObject *)ret;
}


NPY_NO_EXPORT int
_array_from_buffer_3118(PyObject *obj, PyObject **out)
{
#if PY_VERSION_HEX >= 0x02060000
    /* PEP 3118 */
    PyObject *memoryview;
    Py_buffer *view;
    NpyArray_Descr *descr = NULL;
    PyObject *r;
    int nd, flags, k;
    Py_ssize_t d;
    npy_intp shape[NPY_MAXDIMS], strides[NPY_MAXDIMS];

    memoryview = PyMemoryView_FromObject(obj);
    if (memoryview == NULL) {
        PyErr_Clear();
        return -1;
    }

    view = PyMemoryView_GET_BUFFER(memoryview);
    if (view->format != NULL) {
        descr = _descriptor_from_pep3118_format(view->format);
        if (descr == NULL) {
            PyObject *msg;
            msg = PyBytes_FromFormat("Invalid PEP 3118 format string: '%s'",
                                     view->format);
            PyErr_WarnEx(PyExc_RuntimeWarning, PyBytes_AS_STRING(msg), 0);
            Py_DECREF(msg);
            goto fail;
        }

        /* Sanity check */
        if (descr->elsize != view->itemsize) {
            PyErr_WarnEx(PyExc_RuntimeWarning,
                         "Item size computed from the PEP 3118 buffer format "
                         "string does not match the actual item size.",
                         0);
            goto fail;
        }
    }
    else {
        descr = NpyArray_DescrNewFromType(PyArray_STRING);
        descr->elsize = view->itemsize;
    }

    if (view->shape != NULL) {
        nd = view->ndim;
        if (nd >= NPY_MAXDIMS || nd < 0) {
            goto fail;
        }
        for (k = 0; k < nd; ++k) {
            if (k >= NPY_MAXDIMS) {
                goto fail;
            }
            shape[k] = view->shape[k];
        }
        if (view->strides != NULL) {
            for (k = 0; k < nd; ++k) {
                strides[k] = view->strides[k];
            }
        }
        else {
            d = view->len;
            for (k = 0; k < nd; ++k) {
                d /= view->shape[k];
                strides[k] = d;
            }
        }
    }
    else {
        nd = 1;
        shape[0] = view->len / view->itemsize;
        strides[0] = view->itemsize;
    }

    flags = BEHAVED & (view->readonly ? ~NPY_WRITEABLE : ~0);
    ASSIGN_TO_PYARRAY(r,
                      NpyArray_NewFromDescr(descr,
                                            nd, shape, strides, view->buf,
                                            flags, NPY_TRUE, NULL, NULL));
    PyArray_BASE(r) = memoryview;
    PyArray_UpdateFlags((PyArrayObject *)r, UPDATE_ALL);

    *out = r;
    return 0;

fail:
    Npy_XDECREF(descr);
    Py_DECREF(memoryview);
    return -1;

#else
    return -1;
#endif
}


/*NUMPY_API
 * Does not check for ENSURECOPY and NOTSWAPPED in flags
 * Steals a reference to newtype --- which can be NULL
 *
 * Only here for compatibility with older API prior to
 * refactoring.
 */
NPY_NO_EXPORT PyObject *
PyArray_FromAny(PyObject *op, PyArray_Descr *newtype, int min_depth,
                int max_depth, int flags, PyObject *context)
{
    NpyArray_Descr *newtypeCore;

    /* Move stolen reference to core about. */
    PyArray_Descr_REF_TO_CORE(newtype, newtypeCore);

    return PyArray_FromAnyUnwrap(op, newtypeCore, min_depth,
                                 max_depth, flags, context);
}


/* Does not check for ENSURECOPY and NOTSWAPPED in flags
 * Steals a reference to newtype --- which can be NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_FromAnyUnwrap(PyObject *op, NpyArray_Descr *newtype, int min_depth,
                      int max_depth, int flags, PyObject *context)
{
    /*
     * This is the main code to make a NumPy array from a Python
     * Object.  It is called from lot's of different places which
     * is why there are so many checks.  The comments try to
     * explain some of the checks.
     */
    PyObject *r = NULL;
    int seq = FALSE;

    /* Fast path for lists and tuples. */
    if (PyList_CheckExact(op) || PyTuple_CheckExact(op)) {
        goto normal;
    }

    /*
     * Is input object already an array?
     * This is where the flags are used
     */
    if (PyArray_Check(op)) {
        ASSIGN_TO_PYARRAY(r, NpyArray_FromArray(PyArray_ARRAY(op),
                                                newtype, flags));
    }
    else if (PyArray_IsScalar(op, Generic)) {
        if (flags & UPDATEIFCOPY) {
            goto err;
        }
        ASSIGN_TO_PYARRAY(r, PyArray_FromScalarUnwrap(op, newtype));
    }
    else if (newtype == NULL &&
               (newtype = _array_find_python_scalar_type(op))) {
        if (flags & UPDATEIFCOPY) {
            goto err;
        }
        r = Array_FromPyScalar(op, newtype);
    }
    else if (!PyBytes_Check(op) && !PyUnicode_Check(op) &&
             _array_from_buffer_3118(op, &r) == 0) {
        /* PEP 3118 buffer -- but don't accept Bytes objects here */
        PyObject *new;
        if (newtype != NULL || flags != 0) {
            ASSIGN_TO_PYARRAY(new, NpyArray_FromArray(PyArray_ARRAY(r),
                                                      newtype, flags));
            Py_DECREF(r);
            r = new;
        }
    }
    else if (PyArray_HasArrayInterfaceTypeUnwrap(op, newtype, context, r)) {
        PyObject *new;
        if (r == NULL) {
            Npy_XDECREF(newtype);
            return NULL;
        }
        if (newtype != NULL || flags != 0) {
            ASSIGN_TO_PYARRAY(new, NpyArray_FromArray(PyArray_ARRAY(r),
                                                      newtype, flags));
            Py_DECREF(r);
            r = new;
        }
    }
    else {
        int isobject;

    normal:
        isobject = 0;

        if (flags & UPDATEIFCOPY) {
            goto err;
        }
        if (newtype == NULL) {
            newtype = _array_find_type(op, NULL, MAX_DIMS);
        }
        else if (newtype->type_num == NPY_OBJECT) {
            isobject = 1;
        }
        if (PySequence_Check(op)) {
            PyObject *thiserr;

            thiserr = NULL;
            /* necessary but not sufficient */
            Npy_INCREF(newtype);
            r = Array_FromSequence(op, newtype, flags & FORTRAN,
                                   min_depth, max_depth);
            if (r == NULL && (thiserr=PyErr_Occurred())) {
                if (PyErr_GivenExceptionMatches(thiserr,
                                                PyExc_MemoryError)) {
                    return NULL;
                }
                /*
                 * If object was explicitly requested,
                 * then try nested list object array creation
                 */
                PyErr_Clear();
                if (isobject) {
                    Npy_INCREF(newtype);
                    r = ObjectArray_FromNestedList
                        (op, newtype, flags & FORTRAN);
                    seq = TRUE;
                    Npy_DECREF(newtype);
                }
            }
            else {
                seq = TRUE;
                Npy_DECREF(newtype);
            }
        }
        if (!seq) {
            r = Array_FromPyScalar(op, newtype);
        }
    }

    /* If we didn't succeed return NULL */
    if (r == NULL) {
        return NULL;
    }

    /* Be sure we succeed here */
    if(!PyArray_Check(r)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "internal error: PyArray_FromAny "
                        "not producing an array");
        Py_DECREF(r);
        return NULL;
    }

    if (min_depth != 0 && (PyArray_NDIM(r) < min_depth)) {
        PyErr_SetString(PyExc_ValueError,
                        "object of too small depth for desired array");
        Py_DECREF(r);
        return NULL;
    }
    if (max_depth != 0 && (PyArray_NDIM(r) > max_depth)) {
        PyErr_SetString(PyExc_ValueError,
                        "object too deep for desired array");
        Py_DECREF(r);
        return NULL;
    }
    return r;

 err:
    Npy_XDECREF(newtype);
    PyErr_SetString(PyExc_TypeError,
                    "UPDATEIFCOPY used for non-array input.");
    return NULL;
}

/*
 * flags is any of
 * CONTIGUOUS,
 * FORTRAN,
 * ALIGNED,
 * WRITEABLE,
 * NOTSWAPPED,
 * ENSURECOPY,
 * UPDATEIFCOPY,
 * FORCECAST,
 * ENSUREARRAY,
 * ELEMENTSTRIDES
 *
 * or'd (|) together
 *
 * Any of these flags present means that the returned array should
 * guarantee that aspect of the array.  Otherwise the returned array
 * won't guarantee it -- it will depend on the object as to whether or
 * not it has such features.
 *
 * Note that ENSURECOPY is enough
 * to guarantee CONTIGUOUS, ALIGNED and WRITEABLE
 * and therefore it is redundant to include those as well.
 *
 * BEHAVED == ALIGNED | WRITEABLE
 * CARRAY = CONTIGUOUS | BEHAVED
 * FARRAY = FORTRAN | BEHAVED
 *
 * FORTRAN can be set in the FLAGS to request a FORTRAN array.
 * Fortran arrays are always behaved (aligned,
 * notswapped, and writeable) and not (C) CONTIGUOUS (if > 1d).
 *
 * UPDATEIFCOPY flag sets this flag in the returned array if a copy is
 * made and the base argument points to the (possibly) misbehaved array.
 * When the new array is deallocated, the original array held in base
 * is updated with the contents of the new array.
 *
 * FORCECAST will cause a cast to occur regardless of whether or not
 * it is safe.
 */

/* steals a reference to descr -- accepts NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_CheckFromAnyUnwrap(PyObject *op, NpyArray_Descr *descr, int min_depth,
                           int max_depth, int requires, PyObject *context)
{
    PyObject *obj;
    if (requires & NOTSWAPPED) {
        if (!descr && PyArray_Check(op) &&
            !NpyArray_ISNBO(PyArray_DESCR(op)->byteorder)) {
            descr = NpyArray_DescrNew(PyArray_DESCR(op));
        }
        else if (descr && !NpyArray_ISNBO(descr->byteorder)) {
            NpyArray_DESCR_REPLACE(descr);
        }
        if (descr) {
            descr->byteorder = PyArray_NATIVE;
        }
    }

    obj = PyArray_FromAnyUnwrap(op, descr, min_depth, max_depth,
                                requires, context);
    if (obj == NULL) {
        return NULL;
    }
    if ((requires & ELEMENTSTRIDES) &&
        !PyArray_ElementStrides(obj)) {
        PyObject *new;
        new = PyArray_NewCopy((PyArrayObject *)obj, PyArray_ANYORDER);
        Py_DECREF(obj);
        obj = new;
    }
    return obj;
}


/*NUMPY_API
 * steals a reference to descr -- accepts NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_CheckFromAny(PyObject *op, PyArray_Descr *descr, int min_depth,
                     int max_depth, int requires, PyObject *context)
{
    NpyArray_Descr *descrCore;

    /* Move stolen reference to core about. */
    PyArray_Descr_REF_TO_CORE(descr, descrCore);

    return PyArray_CheckFromAnyUnwrap(op, descrCore, min_depth,
                                      max_depth, requires, context);
}


/*NUMPY_API
 * steals reference to newtype --- acc. NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_FromArray(PyArrayObject *arr, PyArray_Descr *newtype, int flags)
{
    NpyArray_Descr *newtypeCore;
    PyArrayObject *ret;

    /* Move stolen reference to core about. */
    PyArray_Descr_REF_TO_CORE(newtype, newtypeCore);

    ASSIGN_TO_PYARRAY(ret,
                      NpyArray_FromArray(PyArray_ARRAY(arr),
                                         newtypeCore, flags));
    return (PyObject *)ret;
}


/*NUMPY_API */
NPY_NO_EXPORT PyObject *
PyArray_FromStructInterface(PyObject *input)
{
    NpyArray_Descr *thetype = NULL;
    char buf[40];
    PyArrayInterface *inter;
    PyObject *attr;
    PyArrayObject *r;
    char endian = PyArray_NATBYTE;

    attr = PyObject_GetAttrString(input, "__array_struct__");
    if (attr == NULL) {
        PyErr_Clear();
        return Py_NotImplemented;
    }
    if (!PyCapsule_Check(attr)) {
        goto fail;
    }
    inter = PyCapsule_AsVoidPtr(attr);
    if (inter->two != 2) {
        goto fail;
    }
    if ((inter->flags & NOTSWAPPED) != NOTSWAPPED) {
        endian = PyArray_OPPBYTE;
        inter->flags &= ~NOTSWAPPED;
    }

    if (inter->flags & ARR_HAS_DESCR) {
        PyArray_Descr *thetypeWrapper = NULL;
        if (PyArray_DescrConverter(
                (PyObject *)PyArray_Descr_WRAP(PyArray_DESCR(inter) ),
                &thetypeWrapper) == PY_FAIL) {
            thetype = NULL;
            PyErr_Clear();
        }
        else {
            thetype = thetypeWrapper->descr;
        }
    }

    if (thetype == NULL) {
        PyOS_snprintf(buf, sizeof(buf),
                "%c%c%d", endian, inter->typekind, inter->itemsize);
        if (!(thetype=_array_typedescr_fromstr(buf))) {
            Py_DECREF(attr);
            return NULL;
        }
    }

    ASSIGN_TO_PYARRAY(r, NpyArray_NewFromDescr(thetype,
                              inter->nd, inter->shape,
                              inter->strides, PyArray_BYTES(inter),
                              inter->flags, NPY_TRUE, NULL, NULL));
    PyArray_BASE(r) = input;
    Py_INCREF(input);
    Py_DECREF(attr);
    PyArray_UpdateFlags((PyArrayObject *)r, UPDATE_ALL);
    return (PyObject *)r;

 fail:
    PyErr_SetString(PyExc_ValueError, "invalid __array_struct__");
    Py_DECREF(attr);
    return NULL;
}

#define PyIntOrLong_Check(obj) (PyInt_Check(obj) || PyLong_Check(obj))

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_FromInterface(PyObject *input)
{
    PyObject *attr = NULL, *item = NULL;
    PyObject *tstr = NULL, *shape = NULL;
    PyObject *inter = NULL;
    PyObject *base = NULL;
    PyArrayObject *ret;
    NpyArray_Descr *type=NULL;
    char *data;
    Py_ssize_t buffer_len;
    int res, i, n;
    intp dims[MAX_DIMS], strides[MAX_DIMS];
    int dataflags = BEHAVED;

    /* Get the memory from __array_data__ and __array_offset__ */
    /* Get the shape */
    /* Get the typestring -- ignore array_descr */
    /* Get the strides */

    inter = PyObject_GetAttrString(input, "__array_interface__");
    if (inter == NULL) {
        PyErr_Clear();
        return Py_NotImplemented;
    }
    if (!PyDict_Check(inter)) {
        Py_DECREF(inter);
        return Py_NotImplemented;
    }
    shape = PyDict_GetItemString(inter, "shape");
    if (shape == NULL) {
        Py_DECREF(inter);
        return Py_NotImplemented;
    }
    tstr = PyDict_GetItemString(inter, "typestr");
    if (tstr == NULL) {
        Py_DECREF(inter);
        return Py_NotImplemented;
    }

    attr = PyDict_GetItemString(inter, "data");
    base = input;
    if ((attr == NULL) || (attr==Py_None) || (!PyTuple_Check(attr))) {
        if (attr && (attr != Py_None)) {
            item = attr;
        }
        else {
            item = input;
        }
        res = PyObject_AsWriteBuffer(item, (void **)&data, &buffer_len);
        if (res < 0) {
            PyErr_Clear();
            res = PyObject_AsReadBuffer(
                    item, (const void **)&data, &buffer_len);
            if (res < 0) {
                goto fail;
            }
            dataflags &= ~WRITEABLE;
        }
        attr = PyDict_GetItemString(inter, "offset");
        if (attr) {
            longlong num = PyLong_AsLongLong(attr);
            if (error_converting(num)) {
                PyErr_SetString(PyExc_TypeError,
                                "offset must be an integer");
                goto fail;
            }
            data += num;
        }
        base = item;
    }
    else {
        PyObject *dataptr;
        if (PyTuple_GET_SIZE(attr) != 2) {
            PyErr_SetString(PyExc_TypeError,
                            "data must return a 2-tuple with (data pointer "
                            "integer, read-only flag)");
            goto fail;
        }
        dataptr = PyTuple_GET_ITEM(attr, 0);
        if (PyString_Check(dataptr)) {
            res = sscanf(PyString_AsString(dataptr),
                         "%p", (void **)&data);
            if (res < 1) {
                PyErr_SetString(PyExc_TypeError,
                                "data string cannot be converted");
                goto fail;
            }
        }
        else if (PyIntOrLong_Check(dataptr)) {
            data = PyLong_AsVoidPtr(dataptr);
        }
        else {
            PyErr_SetString(PyExc_TypeError, "first element "
                            "of data tuple must be integer or string.");
            goto fail;
        }
        if (PyObject_IsTrue(PyTuple_GET_ITEM(attr,1))) {
            dataflags &= ~WRITEABLE;
        }
    }
    attr = tstr;
#if defined(NPY_PY3K)
    if (PyUnicode_Check(tstr)) {
        /* Allow unicode type strings */
        attr = PyUnicode_AsASCIIString(tstr);
    }
#endif
    if (!PyBytes_Check(attr)) {
        PyErr_SetString(PyExc_TypeError, "typestr must be a string");
        goto fail;
    }
    type = _array_typedescr_fromstr(PyString_AS_STRING(attr));
#if defined(NPY_PY3K)
    if (attr != tstr) {
        Py_DECREF(attr);
    }
#endif
    if (type == NULL) {
        goto fail;
    }
    attr = shape;
    if (!PyTuple_Check(attr)) {
        PyErr_SetString(PyExc_TypeError, "shape must be a tuple");
        Py_DECREF(type);
        goto fail;
    }
    n = PyTuple_GET_SIZE(attr);
    for (i = 0; i < n; i++) {
        item = PyTuple_GET_ITEM(attr, i);
        dims[i] = PyArray_PyIntAsIntp(item);
        if (error_converting(dims[i])) {
            break;
        }
    }

    ASSIGN_TO_PYARRAY(ret,
                      NpyArray_NewFromDescr(type,
                                            n, dims,
                                            NULL, data,
                                            dataflags,
                                            NPY_TRUE, NULL, NULL));
    if (ret == NULL) {
        return NULL;
    }
    PyArray_BASE(ret) = base;
    Py_INCREF(base);

    attr = PyDict_GetItemString(inter, "strides");
    if (attr != NULL && attr != Py_None) {
        if (!PyTuple_Check(attr)) {
            PyErr_SetString(PyExc_TypeError,
                            "strides must be a tuple");
            Py_DECREF(ret);
            return NULL;
        }
        if (n != PyTuple_GET_SIZE(attr)) {
            PyErr_SetString(PyExc_ValueError,
                            "mismatch in length of strides and shape");
            Py_DECREF(ret);
            return NULL;
        }
        for (i = 0; i < n; i++) {
            item = PyTuple_GET_ITEM(attr, i);
            strides[i] = PyArray_PyIntAsIntp(item);
            if (error_converting(strides[i])) {
                break;
            }
        }
        if (PyErr_Occurred()) {
            PyErr_Clear();
        }
        memcpy(PyArray_STRIDES(ret), strides, n*sizeof(intp));
    }
    else PyErr_Clear();
    PyArray_UpdateFlags(ret, UPDATE_ALL);
    Py_DECREF(inter);
    return (PyObject *)ret;

 fail:
    Py_XDECREF(inter);
    return NULL;
}



/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_FromArrayAttr(PyObject *op, PyArray_Descr *typecode, PyObject *context)
{
    PyObject *new;
    PyObject *array_meth;

    array_meth = PyObject_GetAttrString(op, "__array__");
    if (array_meth == NULL) {
        PyErr_Clear();
        return Py_NotImplemented;
    }
    if (context == NULL) {
        if (typecode == NULL) {
            new = PyObject_CallFunction(array_meth, NULL);
        }
        else {
            new = PyObject_CallFunction(array_meth, "O", typecode);
        }
    }
    else {
        if (typecode == NULL) {
            new = PyObject_CallFunction(array_meth, "OO", Py_None, context);
            if (new == NULL && PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();
                new = PyObject_CallFunction(array_meth, "");
            }
        }
        else {
            new = PyObject_CallFunction(array_meth, "OO", typecode, context);
            if (new == NULL && PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();
                new = PyObject_CallFunction(array_meth, "O", typecode);
            }
        }
    }
    Py_DECREF(array_meth);
    if (new == NULL) {
        return NULL;
    }
    if (!PyArray_Check(new)) {
        PyErr_SetString(PyExc_ValueError,
                        "object __array__ method not producing an array");
        Py_DECREF(new);
        return NULL;
    }
    return new;
}



NPY_NO_EXPORT PyObject *
PyArray_FromArrayAttrUnwrap(PyObject *op, NpyArray_Descr *typecode,
                            PyObject *context)
{
    return PyArray_FromArrayAttr(op,
        (NULL != typecode) ? PyArray_Descr_WRAP(typecode) : NULL, context);
}



NPY_NO_EXPORT NpyArray_Descr *
PyArray_DescrFromObjectUnwrap(PyObject *op, NpyArray_Descr *mintype)
{
    return _array_find_type(op, mintype, MAX_DIMS);
}


/*NUMPY_API
* new reference -- accepts NULL for mintype
*/
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrFromObject(PyObject *op, PyArray_Descr *mintype)
{
    NpyArray_Descr *result =
        PyArray_DescrFromObjectUnwrap(op,
                        (NULL != mintype) ? mintype->descr : NULL);
    PyArray_Descr_RETURN( result );
}

/* These are also old calls (should use PyArray_NewFromDescr) */

/* They all zero-out the memory as previously done */

/* steals reference to descr -- and enforces native byteorder on it.*/
/*NUMPY_API
  Like FromDimsAndData but uses the Descr structure instead of typecode
  as input.
*/
NPY_NO_EXPORT PyObject *
PyArray_FromDimsAndDataAndDescr(int nd, int *d,
                                PyArray_Descr *descr,
                                char *data)
{
    /* TODO: Not converted as comment says it's 'old'. */
    PyObject *ret;
    int i;
    intp newd[MAX_DIMS];
    char msg[] = "PyArray_FromDimsAndDataAndDescr: use PyArray_NewFromDescr.";

    if (DEPRECATE(msg) < 0) {
        return NULL;
    }
    if (!NpyArray_ISNBO(descr->descr->byteorder))
        descr->descr->byteorder = '=';
    for (i = 0; i < nd; i++) {
        newd[i] = (intp) d[i];
    }
    ret = PyArray_NewFromDescr(&PyArray_Type, descr,
                               nd, newd,
                               NULL, data,
                               (data ? CARRAY : 0), NULL);
    return ret;
}

/*NUMPY_API
  Construct an empty array from dimensions and typenum
*/
NPY_NO_EXPORT PyObject *
PyArray_FromDims(int nd, int *d, int type)
{
    PyObject *ret;
    char msg[] = "PyArray_FromDims: use PyArray_SimpleNew.";

    if (DEPRECATE(msg) < 0) {
        return NULL;
    }
    ret = PyArray_FromDimsAndDataAndDescr(nd, d,
                                          PyArray_DescrFromType(type),
                                          NULL);
    /*
     * Old FromDims set memory to zero --- some algorithms
     * relied on that.  Better keep it the same. If
     * Object type, then it's already been set to zero, though.
     */
    if (ret && (PyArray_DESCR(ret)->type_num != PyArray_OBJECT)) {
        memset(PyArray_BYTES(ret), 0, PyArray_NBYTES(ret));
    }
    return ret;
}

/* end old calls */

/*NUMPY_API
 * This is a quick wrapper around PyArray_FromAny(op, NULL, 0, 0, ENSUREARRAY)
 * that special cases Arrays and PyArray_Scalars up front
 * It *steals a reference* to the object
 * It also guarantees that the result is PyArray_Type
 * Because it decrefs op if any conversion needs to take place
 * so it can be used like PyArray_EnsureArray(some_function(...))
 */
NPY_NO_EXPORT PyObject *
PyArray_EnsureArray(PyObject *op)
{
    PyObject *new;

    /* TODO: Can be pushed into core once PyArray_FromAny has been pushed
             down. */
    if ((op == NULL) || (PyArray_CheckExact(op))) {
        new = op;
        Py_XINCREF(new);
    }
    else if (PyArray_Check(op)) {
        new = PyArray_View((PyArrayObject *)op, NULL, &PyArray_Type);
    }
    else if (PyArray_IsScalar(op, Generic)) {
        new = PyArray_FromScalar(op, NULL);
    }
    else {
        new = PyArray_FromAny(op, NULL, 0, 0, ENSUREARRAY, NULL);
    }
    Py_XDECREF(op);
    return new;
}


/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_EnsureAnyArray(PyObject *op)
{
    /* TODO: Can be pushed to core once PyArray_FromAny, and thus
             PyArray_EnsureArray, are in the core. */
    if (op && PyArray_Check(op)) {
        return op;
    }
    return PyArray_EnsureArray(op);
}

/*NUMPY_API
 * Copy an Array into another array -- memory must not overlap
 * Does not require src and dest to have "broadcastable" shapes
 * (only the same number of elements).
 */
NPY_NO_EXPORT int
PyArray_CopyAnyInto(PyArrayObject *dest, PyArrayObject *src)
{
    return NpyArray_CopyAnyInto(PyArray_ARRAY(dest),
                                PyArray_ARRAY(src));
}

/*NUMPY_API
 * Copy an Array into another array -- memory must not overlap.
 */
NPY_NO_EXPORT int
PyArray_CopyInto(PyArrayObject *dest, PyArrayObject *src)
{
    return NpyArray_CopyInto(PyArray_ARRAY(dest),
                             PyArray_ARRAY(src));
}


/*NUMPY_API
  PyArray_CheckAxis

  check that axis is valid
  convert 0-d arrays to 1-d arrays
*/
NPY_NO_EXPORT PyObject *
PyArray_CheckAxis(PyArrayObject *arr, int *axis, int flags)
{
    NpyArray *narr;
    PyObject *ret;

    narr = NpyArray_CheckAxis(PyArray_ARRAY(arr), axis, flags);
    if (narr == NULL) {
        return NULL;
    }
    ret = Npy_INTERFACE(narr);
    Py_INCREF(ret);
    Npy_DECREF(narr);
    return ret;
}


/*NUMPY_API
 * Zeros
 *
 * steal a reference
 * accepts NULL type
 */
NPY_NO_EXPORT PyObject *
PyArray_Zeros(int nd, intp *dims, PyArray_Descr *type, int fortran)
{
    PyArrayObject *ret;

    /* TODO: Function probably needs to be split,
             but _zerofill needs a "0" int object to use for object arrays. */
    if (!type) {
        type = PyArray_DescrFromType(PyArray_DEFAULT);
    }
    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                                type,
                                                nd, dims,
                                                NULL, NULL,
                                                fortran, NULL);
    if (ret == NULL) {
        return NULL;
    }
    if (_zerofill(ret) < 0) {
        Py_DECREF(ret);
        return NULL;
    }
    return (PyObject *)ret;

}


/*NUMPY_API
 * Empty
 *
 * accepts NULL type
 * steals referenct to type
 */
NPY_NO_EXPORT PyObject *
PyArray_Empty(int nd, intp *dims, PyArray_Descr *type, int fortran)
{
    PyArrayObject *ret;

    /* TODO: Probably needs to be split, uses Py_None as the default value. */
    if (!type) type = PyArray_DescrFromType(PyArray_DEFAULT);
    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                                type, nd, dims,
                                                NULL, NULL,
                                                fortran, NULL);
    if (ret == NULL) {
        return NULL;
    }
    if (PyDataType_REFCHK(type)) {
        PyArray_FillObjectArray(ret, Py_None);
        if (PyErr_Occurred()) {
            Py_DECREF(ret);
            return NULL;
        }
    }
    return (PyObject *)ret;
}


/*
 * Like ceil(value), but check for overflow.
 *
 * Return 0 on success, -1 on failure. In case of failure, set a PyExc_Overflow
 * exception
 */
static int _safe_ceil_to_intp(double value, intp* ret)
{
    double ivalue;

    ivalue = npy_ceil(value);
    if (ivalue < NPY_MIN_INTP || ivalue > NPY_MAX_INTP) {
        return -1;
    }

    *ret = (intp)ivalue;
    return 0;
}


/*NUMPY_API
  Arange,
*/
NPY_NO_EXPORT PyObject *
PyArray_Arange(double start, double stop, double step, int type_num)
{
    intp length;
    PyObject *range;
    NpyArray_ArrFuncs *funcs;
    PyObject *obj;
    int ret;

    if (_safe_ceil_to_intp((stop - start) / step, &length)) {
        PyErr_SetString(PyExc_OverflowError,
                "arange: overflow while computing length");
    }

    if (length <= 0) {
        length = 0;
        return PyArray_New(&PyArray_Type, 1, &length, type_num,
                           NULL, NULL, 0, 0, NULL);
    }
    range = PyArray_New(&PyArray_Type, 1, &length, type_num,
                        NULL, NULL, 0, 0, NULL);
    if (range == NULL) {
        return NULL;
    }
    funcs = PyArray_DESCR(range)->f;

    /*
     * place start in the buffer and the next value in the second position
     * if length > 2, then call the inner loop, otherwise stop
     */
    obj = PyFloat_FromDouble(start);
    ret = funcs->setitem(obj, PyArray_BYTES(range), PyArray_ARRAY(range));
    Py_DECREF(obj);
    if (ret < 0) {
        goto fail;
    }
    if (length == 1) {
        return range;
    }
    obj = PyFloat_FromDouble(start + step);
    ret = funcs->setitem(obj, PyArray_BYTES(range) + PyArray_ITEMSIZE(range),
                         PyArray_ARRAY(range));
    Py_DECREF(obj);
    if (ret < 0) {
        goto fail;
    }
    if (length == 2) {
        return range;
    }
    if (!funcs->fill) {
        PyErr_SetString(PyExc_ValueError, "no fill-function for data-type.");
        Py_DECREF(range);
        return NULL;
    }
    funcs->fill(PyArray_BYTES(range), length, PyArray_ARRAY(range));
    if (PyErr_Occurred()) {
        goto fail;
    }
    return range;

 fail:
    Py_DECREF(range);
    return NULL;
}


/*
 * the formula is len = (intp) ceil((start - stop) / step);
 */
static intp
_calc_length(PyObject *start, PyObject *stop, PyObject *step,
             PyObject **next, int cmplx)
{
    intp len, tmp;
    PyObject *val;
    double value;

    *next = PyNumber_Subtract(stop, start);
    if (!(*next)) {
        if (PyTuple_Check(stop)) {
            PyErr_Clear();
            PyErr_SetString(PyExc_TypeError,
                            "arange: scalar arguments expected "
                            "instead of a tuple.");
        }
        return -1;
    }
    val = PyNumber_TrueDivide(*next, step);
    Py_DECREF(*next);
    *next = NULL;
    if (!val) {
        return -1;
    }
    if (cmplx && PyComplex_Check(val)) {
        value = PyComplex_RealAsDouble(val);
        if (error_converting(value)) {
            Py_DECREF(val);
            return -1;
        }
        if (_safe_ceil_to_intp(value, &len)) {
            Py_DECREF(val);
            PyErr_SetString(PyExc_OverflowError,
                    "arange: overflow while computing length");
            return -1;
        }
        value = PyComplex_ImagAsDouble(val);
        Py_DECREF(val);
        if (error_converting(value)) {
            return -1;
        }
        if (_safe_ceil_to_intp(value, &tmp)) {
            PyErr_SetString(PyExc_OverflowError,
                    "arange: overflow while computing length");
            return -1;
        }
        len = MIN(len, tmp);
    }
    else {
        value = PyFloat_AsDouble(val);
        Py_DECREF(val);
        if (error_converting(value)) {
            return -1;
        }
        if (_safe_ceil_to_intp(value, &len)) {
            PyErr_SetString(PyExc_OverflowError,
                    "arange: overflow while computing length");
            return -1;
        }
    }
    if (len > 0) {
        *next = PyNumber_Add(start, step);
        if (!next) {
            return -1;
        }
    }
    return len;
}


/* ArangeObj,
 *
 * this doesn't change the references
 */
NPY_NO_EXPORT PyObject *
PyArray_ArangeObjUnwrap(PyObject *start, PyObject *stop, PyObject *step,
                        NpyArray_Descr *dtype)
{
    PyObject *range;
    PyArray_ArrFuncs *funcs;
    PyObject *next, *err;
    PyArrayObject *rangeArr;
    intp length;
    NpyArray_Descr *native = NULL;
    int swap;

    if (!dtype) {
        NpyArray_Descr *deftype;
        NpyArray_Descr *newtype;
        /* intentionally made to be PyArray_LONG default */
        deftype = NpyArray_DescrFromType(PyArray_LONG);
        newtype = PyArray_DescrFromObjectUnwrap(start, deftype);
        Npy_DECREF(deftype);
        deftype = newtype;
        if (stop && stop != Py_None) {
            newtype = PyArray_DescrFromObjectUnwrap(stop, deftype);
            Npy_DECREF(deftype);
            deftype = newtype;
        }
        if (step && step != Py_None) {
            newtype = PyArray_DescrFromObjectUnwrap(step, deftype);
            Npy_DECREF(deftype);
            deftype = newtype;
        }
        dtype = deftype;
    }
    else {
        Npy_INCREF(dtype);
    }
    if (!step || step == Py_None) {
        step = PyInt_FromLong(1);
    }
    else {
        Py_XINCREF(step);
    }
    if (!stop || stop == Py_None) {
        stop = start;
        start = PyInt_FromLong(0);
    }
    else {
        Py_INCREF(start);
    }
    /* calculate the length and next = start + step*/
    length = _calc_length(start, stop, step, &next,
                          NpyTypeNum_ISCOMPLEX(dtype->type_num));
    err = PyErr_Occurred();
    if (err) {
        Npy_DECREF(dtype);
        if (err && PyErr_GivenExceptionMatches(err, PyExc_OverflowError)) {
            PyErr_SetString(PyExc_ValueError, "Maximum allowed size exceeded");
        }
        goto fail;
    }
    if (length <= 0) {
        length = 0;
        ASSIGN_TO_PYARRAY(rangeArr,
                          NpyArray_Alloc(dtype, 1, &length, NPY_FALSE, NULL));
        Py_DECREF(step);
        Py_DECREF(start);
        return (PyObject *)rangeArr;
    }

    /*
     * If dtype is not in native byte-order then get native-byte
     * order version.  And then swap on the way out.
     */
    if (!NpyArray_ISNBO(dtype->byteorder)) {
        native = NpyArray_DescrNewByteorder(dtype, PyArray_NATBYTE);
        swap = 1;
    }
    else {
        native = dtype;
        swap = 0;
    }

    ASSIGN_TO_PYARRAY(rangeArr,
                      NpyArray_Alloc(native, 1, &length, NPY_FALSE, NULL));
    if (rangeArr == NULL) {
        goto fail;
    }
    range = (PyObject *)rangeArr;

    /*
     * place start in the buffer and the next value in the second position
     * if length > 2, then call the inner loop, otherwise stop
     */
    funcs = PyArray_DESCR(range)->f;
    if (funcs->setitem(
           start, PyArray_BYTES(range), PyArray_ARRAY(range)) < 0) {
        goto fail;
    }
    if (length == 1) {
        goto finish;
    }
    if (funcs->setitem(next, PyArray_BYTES(range)+PyArray_ITEMSIZE(range),
                       PyArray_ARRAY(range)) < 0) {
        goto fail;
    }
    if (length == 2) {
        goto finish;
    }
    if (!funcs->fill) {
        PyErr_SetString(PyExc_ValueError, "no fill-function for data-type.");
        Py_DECREF(range);
        goto fail;
    }
    funcs->fill(PyArray_BYTES(range), length, PyArray_ARRAY(range));
    if (PyErr_Occurred()) {
        goto fail;
    }
 finish:
    if (swap) {
        PyObject *new;
        new = PyArray_Byteswap((PyArrayObject *)range, 1);
        Py_DECREF(new);
        Npy_DECREF(PyArray_DESCR(range));
        /* reference alloc'd earlier in this func */
        PyArray_DESCR(range) = dtype;
    }
    Py_DECREF(start);
    Py_DECREF(step);
    Py_DECREF(next);
    return range;

 fail:
    Py_DECREF(start);
    Py_DECREF(step);
    Py_XDECREF(next);
    return NULL;
}


/*NUMPY_API
 *
 * ArangeObj,
 *
 * this doesn't change the references
 */
NPY_NO_EXPORT PyObject *
PyArray_ArangeObj(PyObject *start, PyObject *stop, PyObject *step,
                  PyArray_Descr *dtype)
{
    return PyArray_ArangeObjUnwrap(start, stop, step,
                                   (NULL != dtype) ? dtype->descr : NULL);
}


/* Steals a reference to dtype. */
NPY_NO_EXPORT PyObject *
PyArray_FromTextFile(FILE *fp, PyArray_Descr *dtype, intp num, char *sep)
{
    PyArrayObject *ret;
    
    if (dtype == NULL) {
        return NULL;
    }
    Npy_INCREF(dtype->descr);
    ASSIGN_TO_PYARRAY(ret,
                      NpyArray_FromTextFile(fp, dtype->descr, num, sep));
    Py_XDECREF(dtype);

    return (PyObject *)ret;
}


/*NUMPY_API
 *
 * Given a ``FILE *`` pointer ``fp``, and a ``PyArray_Descr``, return an
 * array corresponding to the data encoded in that file.
 *
 * If the dtype is NULL, the default array type is used (double).
 * If non-null, the reference is stolen.
 *
 * The number of elements to read is given as ``num``; if it is < 0, then
 * then as many as possible are read.
 *
 * If ``sep`` is NULL or empty, then binary data is assumed, else
 * text data, with ``sep`` as the separator between elements.  Whitespace in
 * the separator matches any length of whitespace in the text, and a match
 * for whitespace around the separator is added.
 *
 * For memory-mapped files, use the buffer interface. No more data than
 * necessary is read by this routine.
 */
NPY_NO_EXPORT PyObject *
PyArray_FromFile(FILE *fp, PyArray_Descr *dtype, intp num, char *sep)
{
    PyArrayObject *ret = NULL;

    if (PyDataType_REFCHK(dtype)) {
        PyErr_SetString(PyExc_ValueError, "Cannot read into object array");
        Py_DECREF(dtype);
        return NULL;
    }
    if (dtype->descr->elsize == 0) {
        PyErr_SetString(PyExc_ValueError, "The elements are 0-sized.");
        Py_DECREF(dtype);
        return NULL;
    }

    if (NULL == sep || 0 == strlen(sep)) {
        /* Move reference from interface to core. */
        Npy_INCREF(dtype->descr);
        Py_DECREF(dtype);
        ASSIGN_TO_PYARRAY(ret, NpyArray_FromBinaryFile(fp, dtype->descr, num));
    }
    else {
        ret = (PyArrayObject* ) PyArray_FromTextFile(fp, dtype, num, sep);
    }

    return (PyObject *)ret;
}


/*NUMPY_API
 *
 * Steals a reference to type.
 */
NPY_NO_EXPORT PyObject *
PyArray_FromBuffer(PyObject *buf, PyArray_Descr *type,
                   intp count, intp offset)
{
    PyArrayObject *ret;
    char *data;
    Py_ssize_t ts;
    intp s, n;
    int itemsize;
    int write = 1;

    if (PyDataType_REFCHK(type)) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot create an OBJECT array from memory buffer");
        Py_DECREF(type);
        return NULL;
    }
    if (type->descr->elsize == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "itemsize cannot be zero in type");
        Py_DECREF(type);
        return NULL;
    }
    if (Py_TYPE(buf)->tp_as_buffer == NULL
#if defined(NPY_PY3K)
        || Py_TYPE(buf)->tp_as_buffer->bf_getbuffer == NULL
#else
        || (Py_TYPE(buf)->tp_as_buffer->bf_getwritebuffer == NULL
            && Py_TYPE(buf)->tp_as_buffer->bf_getreadbuffer == NULL)
#endif
        ) {
        PyObject *newbuf;
        newbuf = PyObject_GetAttrString(buf, "__buffer__");
        if (newbuf == NULL) {
            Py_DECREF(type);
            return NULL;
        }
        buf = newbuf;
    }
    else {
        Py_INCREF(buf);
    }

    if (PyObject_AsWriteBuffer(buf, (void *)&data, &ts) == -1) {
        write = 0;
        PyErr_Clear();
        if (PyObject_AsReadBuffer(buf, (void *)&data, &ts) == -1) {
            Py_DECREF(buf);
            Py_DECREF(type);
            return NULL;
        }
    }

    if ((offset < 0) || (offset >= ts)) {
        PyErr_Format(PyExc_ValueError,
                     "offset must be non-negative and smaller than buffer "
                     "lenth (%" INTP_FMT ")", (intp)ts);
        Py_DECREF(buf);
        Py_DECREF(type);
        return NULL;
    }

    data += offset;
    s = (intp)ts - offset;
    n = (intp)count;
    itemsize = type->descr->elsize;
    if (n < 0 ) {
        if (s % itemsize != 0) {
            PyErr_SetString(PyExc_ValueError,
                            "buffer size must be a multiple of element size");
            Py_DECREF(buf);
            Py_DECREF(type);
            return NULL;
        }
        n = s/itemsize;
    }
    else {
        if (s < n*itemsize) {
            PyErr_SetString(PyExc_ValueError,
                            "buffer is smaller than requested size");
            Py_DECREF(buf);
            Py_DECREF(type);
            return NULL;
        }
    }

    if ((ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                                     type,
                                                     1, &n,
                                                     NULL, data,
                                                     DEFAULT,
                                                     NULL)) == NULL) {
        Py_DECREF(buf);
        return NULL;
    }

    if (!write) {
        PyArray_FLAGS(ret) &= ~WRITEABLE;
    }
    /* Store a reference for decref on deallocation */
    PyArray_BASE(ret) = buf;
    PyArray_UpdateFlags(ret, ALIGNED);
    return (PyObject *)ret;
}


/*NUMPY_API
 *
 * Given a pointer to a string ``data``, a string length ``slen``, and
 * a ``PyArray_Descr``, return an array corresponding to the data
 * encoded in that string.
 *
 * If the dtype is NULL, the default array type is used (double).
 * If non-null, the reference is stolen.
 *
 * If ``slen`` is < 0, then the end of string is used for text data.
 * It is an error for ``slen`` to be < 0 for binary data (since embedded NULLs
 * would be the norm).
 *
 * The number of elements to read is given as ``num``; if it is < 0, then
 * then as many as possible are read.
 *
 * If ``sep`` is NULL or empty, then binary data is assumed, else
 * text data, with ``sep`` as the separator between elements.  Whitespace in
 * the separator matches any length of whitespace in the text, and a match
 * for whitespace around the separator is added.
 */
NPY_NO_EXPORT PyObject *
PyArray_FromString(char *data, intp slen, PyArray_Descr *dtype,
                   intp num, char *sep)
{
    PyArrayObject *ret;

    if (dtype == NULL) {
        dtype = PyArray_DescrFromType(PyArray_DEFAULT);
        if (dtype == NULL) {
            return NULL;
        }
    }
    Npy_INCREF(dtype->descr);
    ASSIGN_TO_PYARRAY(ret,
                      NpyArray_FromString(data, slen, dtype->descr, num, sep));
    Py_XDECREF(dtype);

    return (PyObject *)ret;
}


/* steals a reference to dtype (which cannot be NULL)
 */
NPY_NO_EXPORT PyObject *
PyArray_FromIterUnwrap(PyObject *obj, NpyArray_Descr *dtype, intp count)
{
    PyObject *value;
    PyObject *iter = PyObject_GetIter(obj);
    PyArrayObject *ret = NULL;
    intp i, elsize, elcount;
    char *item, *new_data;

    if (iter == NULL) {
        goto done;
    }
    elcount = (count < 0) ? 0 : count;
    if ((elsize=dtype->elsize) == 0) {
        PyErr_SetString(PyExc_ValueError, "Must specify length "
                        "when using variable-size data-type.");
        goto done;
    }

    /*
     * We would need to alter the memory RENEW code to decrement any
     * reference counts before throwing away any memory.
     */
    if (NpyDataType_REFCHK(dtype)) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot create object arrays from iterator");
        goto done;
    }

    ASSIGN_TO_PYARRAY(ret,
                      NpyArray_Alloc(dtype, 1, &elcount, NPY_FALSE, NULL));
    dtype = NULL;
    if (ret == NULL) {
        goto done;
    }
    for (i = 0; (i < count || count == -1) &&
             (value = PyIter_Next(iter)); i++) {
        if (i >= elcount) {
            /*
              Grow PyArray_BYTES(ret):
              this is similar for the strategy for PyListObject, but we use
              50% overallocation => 0, 4, 8, 14, 23, 36, 56, 86 ...
            */
            elcount = (i >> 1) + (i < 4 ? 4 : 2) + i;
            if (elcount <= NPY_MAX_INTP/elsize) {
                new_data = PyDataMem_RENEW(PyArray_BYTES(ret),
                                           elcount * elsize);
            }
            else {
                new_data = NULL;
            }
            if (new_data == NULL) {
                PyErr_SetString(PyExc_MemoryError,
                                "cannot allocate array memory");
                Py_DECREF(value);
                goto done;
            }
            PyArray_BYTES(ret) = new_data;
        }
        PyArray_DIM(ret, 0) = i + 1;

        if (((item = index2ptr(ret, i)) == NULL)
            || (PyArray_DESCR(ret)->f->setitem(value, item,
                                               PyArray_ARRAY(ret)) == -1)) {
            Py_DECREF(value);
            goto done;
        }
        Py_DECREF(value);
    }

    if (i < count) {
        PyErr_SetString(PyExc_ValueError, "iterator too short");
        goto done;
    }

    /*
     * Realloc the data so that don't keep extra memory tied up
     * (assuming realloc is reasonably good about reusing space...)
     */
    if (i == 0) {
        i = 1;
    }
    new_data = PyDataMem_RENEW(PyArray_BYTES(ret), i * elsize);
    if (new_data == NULL) {
        PyErr_SetString(PyExc_MemoryError, "cannot allocate array memory");
        goto done;
    }
    PyArray_BYTES(ret) = new_data;

 done:
    Py_XDECREF(iter);
    Npy_XDECREF(dtype);
    if (PyErr_Occurred()) {
        Py_XDECREF(ret);
        return NULL;
    }
    return (PyObject *)ret;
}



/*NUMPY_API
 *
 * steals a reference to dtype (which cannot be NULL)
 */
NPY_NO_EXPORT PyObject *
PyArray_FromIter(PyObject *obj, PyArray_Descr *dtype, intp count)
{
    NpyArray_Descr *dtypeCore;

    /* Move stolen reference to core about. */
    PyArray_Descr_REF_TO_CORE(dtype, dtypeCore);

    return PyArray_FromIterUnwrap(obj, dtypeCore, count);
}
