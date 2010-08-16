/* Array Descr Object */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "npy_api.h"

#include "npy_config.h"

#include "numpy_3kcompat.h"

#include "common.h"
#include "scalartypes.h"
#include "ctors.h"
#include "descriptor.h"
#include "arrayobject.h"
#include "getset.h"

#define ASSERT_ONE_BASE(r) \
    assert(NULL == PyArray_BASE_ARRAY(r) || NULL == PyArray_BASE(r))

/*******************  array attribute get and set routines ******************/

static PyObject *
array_ndim_get(PyArrayObject *self)
{
    return PyInt_FromLong(PyArray_NDIM(self));
}

static PyObject *
array_flags_get(PyArrayObject *self)
{
    return PyArray_NewFlagsObject((PyObject *)self);
}

static PyObject *
array_shape_get(PyArrayObject *self)
{
    return PyArray_IntTupleFromIntp(PyArray_NDIM(self), PyArray_DIMS(self));
}


static int
array_shape_set(PyArrayObject *self, PyObject *val)
{
    int res;
    PyArray_Dims newdims;

    if (!PyArray_IntpConverter(val, &newdims)) {
        return -1;
    }
    res = NpyArray_SetShape(PyArray_ARRAY(self), &newdims);
    NpyDimMem_FREE(newdims.ptr);
    return res;
}


static PyObject *
array_strides_get(PyArrayObject *self)
{
    return PyArray_IntTupleFromIntp(PyArray_NDIM(self), PyArray_STRIDES(self));
}

static int
array_strides_set(PyArrayObject *self, PyObject *obj)
{
    int res;
    PyArray_Dims newstrides = {NULL, 0};

    if (!PyArray_IntpConverter(obj, &newstrides) ||
        newstrides.ptr == NULL) {
        PyErr_SetString(PyExc_TypeError, "invalid strides");
        return -1;
    }
    res = NpyArray_SetStrides(PyArray_ARRAY(self), &newstrides);
    NpyDimMem_FREE(newstrides.ptr);
    return res;
}



static PyObject *
array_priority_get(PyArrayObject *self)
{
    if (PyArray_CheckExact(self)) {
        return PyFloat_FromDouble(PyArray_PRIORITY);
    }
    else {
        return PyFloat_FromDouble(PyArray_SUBTYPE_PRIORITY);
    }
}

static PyObject *
array_typestr_get(PyArrayObject *self)
{
    return npy_arraydescr_protocol_typestr_get(PyArray_DESCR(self));
}

static PyObject *
array_descr_get(PyArrayObject *self)
{
    PyArray_Descr *descr;

    descr = PyArray_Descr_WRAP(PyArray_DESCR(self));
    Py_INCREF(descr);
    return (PyObject *)descr;
}

static PyObject *
array_protocol_descr_get(PyArrayObject *self)
{
    PyObject *res;
    PyObject *dobj;

    res = npy_arraydescr_protocol_descr_get(PyArray_DESCR(self));
    if (res) {
        return res;
    }
    PyErr_Clear();

    /* get default */
    dobj = PyTuple_New(2);
    if (dobj == NULL) {
        return NULL;
    }
    PyTuple_SET_ITEM(dobj, 0, PyString_FromString(""));
    PyTuple_SET_ITEM(dobj, 1, array_typestr_get(self));
    res = PyList_New(1);
    if (res == NULL) {
        Py_DECREF(dobj);
        return NULL;
    }
    PyList_SET_ITEM(res, 0, dobj);
    return res;
}

static PyObject *
array_protocol_strides_get(PyArrayObject *self)
{
    if PyArray_ISCONTIGUOUS(self) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return PyArray_IntTupleFromIntp(PyArray_NDIM(self), PyArray_STRIDES(self));
}



static PyObject *
array_dataptr_get(PyArrayObject *self)
{
    return Py_BuildValue("NO",
                         PyLong_FromVoidPtr(PyArray_BYTES(self)),
                         (PyArray_FLAGS(self) & WRITEABLE ? Py_False :
                          Py_True));
}

static PyObject *
array_ctypes_get(PyArrayObject *self)
{
    PyObject *_numpy_internal;
    PyObject *ret;
    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
    if (_numpy_internal == NULL) {
        return NULL;
    }
    ret = PyObject_CallMethod(_numpy_internal, "_ctypes", "ON", self,
                              PyLong_FromVoidPtr(PyArray_BYTES(self)));
    Py_DECREF(_numpy_internal);
    return ret;
}

static PyObject *
array_interface_get(PyArrayObject *self)
{
    PyObject *dict;
    PyObject *obj;

    dict = PyDict_New();
    if (dict == NULL) {
        return NULL;
    }

    /* dataptr */
    obj = array_dataptr_get(self);
    PyDict_SetItemString(dict, "data", obj);
    Py_DECREF(obj);

    obj = array_protocol_strides_get(self);
    PyDict_SetItemString(dict, "strides", obj);
    Py_DECREF(obj);

    obj = array_protocol_descr_get(self);
    PyDict_SetItemString(dict, "descr", obj);
    Py_DECREF(obj);

    obj = npy_arraydescr_protocol_typestr_get(PyArray_DESCR(self));
    PyDict_SetItemString(dict, "typestr", obj);
    Py_DECREF(obj);

    obj = array_shape_get(self);
    PyDict_SetItemString(dict, "shape", obj);
    Py_DECREF(obj);

    obj = PyInt_FromLong(3);
    PyDict_SetItemString(dict, "version", obj);
    Py_DECREF(obj);

    return dict;
}

static PyObject *
array_data_get(PyArrayObject *self)
{
#if defined(NPY_PY3K)
    return PyMemoryView_FromObject(self);
#else
    intp nbytes;
    if (!(PyArray_ISONESEGMENT(self))) {
        PyErr_SetString(PyExc_AttributeError, "cannot get single-"\
                        "segment buffer for discontiguous array");
        return NULL;
    }
    nbytes = PyArray_NBYTES(self);
    if (PyArray_ISWRITEABLE(self)) {
        return PyBuffer_FromReadWriteObject((PyObject *)self, 0, (Py_ssize_t) nbytes);
    }
    else {
        return PyBuffer_FromObject((PyObject *)self, 0, (Py_ssize_t) nbytes);
    }
#endif
}

static int
array_data_set(PyArrayObject *self, PyObject *op)
{
    void *buf;
    Py_ssize_t buf_len;
    int writeable=1;

    if (PyObject_AsWriteBuffer(op, &buf, &buf_len) < 0) {
        writeable = 0;
        if (PyObject_AsReadBuffer(op, (const void **)&buf, &buf_len) < 0) {
            PyErr_SetString(PyExc_AttributeError,
                            "object does not have single-segment " \
                            "buffer interface");
            return -1;
        }
    }
    if (!PyArray_ISONESEGMENT(self)) {
        PyErr_SetString(PyExc_AttributeError, "cannot set single-" \
                        "segment buffer for discontiguous array");
        return -1;
    }
    if (PyArray_NBYTES(self) > buf_len) {
        PyErr_SetString(PyExc_AttributeError, "not enough data for array");
        return -1;
    }
    if (PyArray_FLAGS(self) & OWNDATA) {
        PyArray_XDECREF(self);
        PyDataMem_FREE(PyArray_BYTES(self));
    }
    if (PyArray_BASE_ARRAY(self)) {
        if (PyArray_FLAGS(self) & UPDATEIFCOPY) {
            PyArray_BASE_ARRAY(self)->flags |= WRITEABLE;
            PyArray_FLAGS(self) &= ~UPDATEIFCOPY;
        }
        Npy_DECREF(PyArray_BASE_ARRAY(self));
        PyArray_BASE_ARRAY(self) = NULL;
    }
    if (NULL != PyArray_BASE(self)) {
        Py_DECREF(PyArray_BASE(self));
        PyArray_BASE(self) = NULL;
    }

    PyArray_BASE(self) = op;
    Py_INCREF(op);
    PyArray_BYTES(self) = buf;
    PyArray_FLAGS(self) = CARRAY;
    if (!writeable) {
        PyArray_FLAGS(self) &= ~WRITEABLE;
    }
    return 0;
}


static PyObject *
array_itemsize_get(PyArrayObject *self)
{
    return PyInt_FromLong((long) PyArray_ITEMSIZE(self));
}

static PyObject *
array_size_get(PyArrayObject *self)
{
    intp size=PyArray_SIZE(self);
#if SIZEOF_INTP <= SIZEOF_LONG
    return PyInt_FromLong((long) size);
#else
    if (size > MAX_LONG || size < MIN_LONG) {
        return PyLong_FromLongLong(size);
    }
    else {
        return PyInt_FromLong((long) size);
    }
#endif
}

static PyObject *
array_nbytes_get(PyArrayObject *self)
{
    intp nbytes = PyArray_NBYTES(self);
#if SIZEOF_INTP <= SIZEOF_LONG
    return PyInt_FromLong((long) nbytes);
#else
    if (nbytes > MAX_LONG || nbytes < MIN_LONG) {
        return PyLong_FromLongLong(nbytes);
    }
    else {
        return PyInt_FromLong((long) nbytes);
    }
#endif
}


/*
 * If the type is changed.
 * Also needing change: strides, itemsize
 *
 * Either itemsize is exactly the same or the array is single-segment
 * (contiguous or fortran) with compatibile dimensions The shape and strides
 * will be adjusted in that case as well.
 */

static int
array_descr_set(PyArrayObject *self, PyObject *arg)
{
    NpyArray_Descr *newtype = NULL;
    PyArray_Descr *newtypeInterface = NULL;
    int result;

    if (!(PyArray_DescrConverter(arg, &newtypeInterface)) ||
        newtypeInterface == NULL) {
        Py_XDECREF(newtypeInterface);
        PyErr_SetString(PyExc_TypeError, "invalid data-type for array");
        return -1;
    }

    PyArray_Descr_REF_TO_CORE(newtypeInterface, newtype);
    result = NpyArray_SetDescr(PyArray_ARRAY(self), newtype);
    Npy_DECREF(newtype);
    return result;
}

static PyObject *
array_struct_get(PyArrayObject *self)
{
    PyArrayInterface *inter;
    PyObject *ret;

    inter = (PyArrayInterface *)_pya_malloc(sizeof(PyArrayInterface));
    if (inter==NULL) {
        return PyErr_NoMemory();
    }
    inter->two = 2;
    PyArray_NDIM(inter) = PyArray_NDIM(self);
    inter->typekind = PyArray_DESCR(self)->kind;
    inter->itemsize = PyArray_ITEMSIZE(self);
    PyArray_FLAGS(inter) = PyArray_FLAGS(self);
    /* reset unused flags */
    PyArray_FLAGS(inter) &= ~(UPDATEIFCOPY | OWNDATA);
    if (PyArray_ISNOTSWAPPED(self)) PyArray_FLAGS(inter) |= NOTSWAPPED;
    /*
     * Copy shape and strides over since these can be reset
     *when the array is "reshaped".
     */
    if (PyArray_NDIM(self) > 0) {
        inter->shape = (intp *)_pya_malloc(2*sizeof(intp)*PyArray_NDIM(self));
        if (inter->shape == NULL) {
            _pya_free(inter);
            return PyErr_NoMemory();
        }
        PyArray_STRIDES(inter) = inter->shape + PyArray_NDIM(self);
        memcpy(inter->shape, PyArray_DIMS(self),
               sizeof(intp)*PyArray_NDIM(self));
        memcpy(PyArray_STRIDES(inter), PyArray_STRIDES(self),
               sizeof(intp)*PyArray_NDIM(self));
    }
    else {
        inter->shape = NULL;
        PyArray_STRIDES(inter) = NULL;
    }
    PyArray_BYTES(inter) = PyArray_BYTES(self);
    if (PyArray_DESCR(self)->names) {
        PyArray_Descr *wrap = (PyArray_Descr *)
                       npy_arraydescr_protocol_descr_get(PyArray_DESCR(self));
        if (NULL != wrap) {
            PyArray_DESCR(inter) = wrap->descr;
            Npy_INCREF(wrap->descr);
            Py_DECREF(wrap);
            PyArray_FLAGS(inter) &= ARR_HAS_DESCR;
        } else {
            PyArray_DESCR(inter) = NULL;
        }
    }
    else {
        PyArray_DESCR(inter) = NULL;
    }
    Py_INCREF(self);
    ret = PyCapsule_FromVoidPtrAndDesc(inter, self, gentype_struct_free);
    return ret;
}

static PyObject *
array_base_get(PyArrayObject *self)
{
    if (NULL != PyArray_BASE_ARRAY(self)) {
        Npy_INCREF(PyArray_BASE_ARRAY(self));
        RETURN_PYARRAY(PyArray_BASE_ARRAY(self));
    } else if (NULL != PyArray_BASE(self)) {
        Py_INCREF(PyArray_BASE(self));
        return PyArray_BASE(self);
    } else {
        Py_INCREF(Py_None);
        return Py_None;
    }
}

/*
 * Create a view of a complex array with an equivalent data-type
 * except it is real instead of complex.
 */
static PyArrayObject *
_get_part(PyArrayObject *self, int imag)
{
    NpyArray_Descr *type;
    PyArrayObject *ret;
    int offset;

    type = NpyArray_DescrFromType(PyArray_TYPE(self) -
                                  PyArray_NUM_FLOATTYPE);
    offset = (imag ? type->elsize : 0);

    if (!NpyArray_ISNBO(PyArray_DESCR(self)->byteorder)) {
        NpyArray_Descr *new;
        new = NpyArray_DescrNew(type);
        new->byteorder = PyArray_DESCR(self)->byteorder;
        Npy_DECREF(type);
        type = new;
    }
    RETURN_PYARRAY(NpyArray_NewView(type,
                                    PyArray_NDIM(self), 
                                    PyArray_DIMS(self), 
                                    PyArray_STRIDES(self),
                                    PyArray_ARRAY(self), offset,
                                    NPY_FALSE));
}

/* For Object arrays, we need to get and set the
   real part of each element.
 */

static PyObject *
array_real_get(PyArrayObject *self)
{
    PyArrayObject *ret;

    if (PyArray_ISCOMPLEX(self)) {
        ret = _get_part(self, 0);
        return (PyObject *)ret;
    }
    else {
        Py_INCREF(self);
        return (PyObject *)self;
    }
}


static int
array_real_set(PyArrayObject *self, PyObject *val)
{
    PyArrayObject *ret;
    PyArrayObject *new;
    int rint;

    if (PyArray_ISCOMPLEX(self)) {
        ret = _get_part(self, 0);
        if (ret == NULL) {
            return -1;
        }
    }
    else {
        Py_INCREF(self);
        ret = self;
    }
    new = (PyArrayObject *)PyArray_FromAny(val, NULL, 0, 0, 0, NULL);
    if (new == NULL) {
        Py_DECREF(ret);
        return -1;
    }
    rint = PyArray_MoveInto(ret, new);
    Py_DECREF(ret);
    Py_DECREF(new);
    return rint;
}

/* For Object arrays we need to get
   and set the imaginary part of
   each element
*/

static PyObject *
array_imag_get(PyArrayObject *self)
{
    PyArrayObject *ret;

    if (PyArray_ISCOMPLEX(self)) {
        ret = _get_part(self, 1);
    }
    else {
        Npy_INCREF(PyArray_DESCR(self));
        ASSIGN_TO_PYARRAY(ret,
                          NpyArray_Alloc(PyArray_DESCR(self),
                                         PyArray_NDIM(self),
                                         PyArray_DIMS(self),
                                         PyArray_ISFORTRAN(self),
                                         self));
        if (ret == NULL) {
            return NULL;
        }
        if (_zerofill(ret) < 0) {
            return NULL;
        }
        PyArray_FLAGS(ret) &= ~WRITEABLE;
    }
    return (PyObject *) ret;
}

static int
array_imag_set(PyArrayObject *self, PyObject *val)
{
    if (PyArray_ISCOMPLEX(self)) {
        PyArrayObject *ret;
        PyArrayObject *new;
        int rint;

        ret = _get_part(self, 1);
        if (ret == NULL) {
            return -1;
        }
        new = (PyArrayObject *)PyArray_FromAny(val, NULL, 0, 0, 0, NULL);
        if (new == NULL) {
            Py_DECREF(ret);
            return -1;
        }
        rint = PyArray_MoveInto(ret, new);
        Py_DECREF(ret);
        Py_DECREF(new);
        return rint;
    }
    else {
        PyErr_SetString(PyExc_TypeError, "array does not have "\
                        "imaginary part to set");
        return -1;
    }
}

static PyObject *
array_flat_get(PyArrayObject *self)
{
    return PyArray_IterNew((PyObject *)self);
}

static int
array_flat_set(PyArrayObject *self, PyObject *val)
{
    PyObject *arr = NULL;
    int retval = -1;
    NpyArrayIterObject *selfit = NULL, *arrit = NULL;
    NpyArray_Descr *typecode;
    int swap;
    PyArray_CopySwapFunc *copyswap;

    typecode = PyArray_DESCR(self);
    Npy_INCREF(typecode);
    arr = PyArray_FromAnyUnwrap(val, typecode,
                                0, 0, FORCECAST | FORTRAN_IF(self), NULL);
    if (arr == NULL) {
        return -1;
    }
    arrit = NpyArray_IterNew(PyArray_ARRAY(arr));
    if (arrit == NULL) {
        goto exit;
    }
    selfit = NpyArray_IterNew(PyArray_ARRAY(self));
    if (selfit == NULL) {
        goto exit;
    }
    if (arrit->size == 0) {
        retval = 0;
        goto exit;
    }
    swap = PyArray_ISNOTSWAPPED(self) != PyArray_ISNOTSWAPPED(arr);
    copyswap = PyArray_DESCR(self)->f->copyswap;
    if (NpyDataType_REFCHK(PyArray_DESCR(self))) {
        while (selfit->index < selfit->size) {
            /* TODO: Fix this for garbage collected environments,
               new pointers will be written back. */
            NpyArray_Item_XDECREF(selfit->dataptr, PyArray_DESCR(self));
            NpyArray_Item_INCREF(arrit->dataptr, PyArray_DESCR(arr));
            memmove(selfit->dataptr, arrit->dataptr, sizeof(PyObject **));
            if (swap) {
                copyswap(selfit->dataptr, NULL, swap,
                         PyArray_ARRAY(self));
            }
            NpyArray_ITER_NEXT(selfit);
            NpyArray_ITER_NEXT(arrit);
            if (arrit->index == arrit->size) {
                NpyArray_ITER_RESET(arrit);
            }
        }
        retval = 0;
        goto exit;
    }

    while(selfit->index < selfit->size) {
        memmove(selfit->dataptr, arrit->dataptr, PyArray_ITEMSIZE(self));
        if (swap) {
            copyswap(selfit->dataptr, NULL, swap, PyArray_ARRAY(self));
        }
        NpyArray_ITER_NEXT(selfit);
        NpyArray_ITER_NEXT(arrit);
        if (arrit->index == arrit->size) {
            NpyArray_ITER_RESET(arrit);
        }
    }
    retval = 0;

 exit:
    Npy_XDECREF(selfit);
    Npy_XDECREF(arrit);
    Py_XDECREF(arr);
    return retval;
}

static PyObject *
array_transpose_get(PyArrayObject *self)
{
    return PyArray_Transpose(self, NULL);
}

/* If this is None, no function call is made
   --- default sub-class behavior
*/
static PyObject *
array_finalize_get(PyArrayObject *NPY_UNUSED(self))
{
    Py_INCREF(Py_None);
    return Py_None;
}

NPY_NO_EXPORT PyGetSetDef array_getsetlist[] = {
    {"ndim",
        (getter)array_ndim_get,
        NULL,
        NULL, NULL},
    {"flags",
        (getter)array_flags_get,
        NULL,
        NULL, NULL},
    {"shape",
        (getter)array_shape_get,
        (setter)array_shape_set,
        NULL, NULL},
    {"strides",
        (getter)array_strides_get,
        (setter)array_strides_set,
        NULL, NULL},
    {"data",
        (getter)array_data_get,
        (setter)array_data_set,
        NULL, NULL},
    {"itemsize",
        (getter)array_itemsize_get,
        NULL,
        NULL, NULL},
    {"size",
        (getter)array_size_get,
        NULL,
        NULL, NULL},
    {"nbytes",
        (getter)array_nbytes_get,
        NULL,
        NULL, NULL},
    {"base",
        (getter)array_base_get,
        NULL,
        NULL, NULL},
    {"dtype",
        (getter)array_descr_get,
        (setter)array_descr_set,
        NULL, NULL},
    {"real",
        (getter)array_real_get,
        (setter)array_real_set,
        NULL, NULL},
    {"imag",
        (getter)array_imag_get,
        (setter)array_imag_set,
        NULL, NULL},
    {"flat",
        (getter)array_flat_get,
        (setter)array_flat_set,
        NULL, NULL},
    {"ctypes",
        (getter)array_ctypes_get,
        NULL,
        NULL, NULL},
    {   "T",
        (getter)array_transpose_get,
        NULL,
        NULL, NULL},
    {"__array_interface__",
        (getter)array_interface_get,
        NULL,
        NULL, NULL},
    {"__array_struct__",
        (getter)array_struct_get,
        NULL,
        NULL, NULL},
    {"__array_priority__",
        (getter)array_priority_get,
        NULL,
        NULL, NULL},
    {"__array_finalize__",
        (getter)array_finalize_get,
        NULL,
        NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},  /* Sentinel */
};

/****************** end of attribute get and set routines *******************/
