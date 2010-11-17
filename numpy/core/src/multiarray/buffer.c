#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "npy_api.h"
#include "npy_dict.h"
#include "npy_buffer.h"
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "numpy/npy_3kcompat.h"

#include "buffer.h"
#include "npy_os.h"


static size_t
array_getsegcount(PyArrayObject *self, size_t *lenp)
{
    return npy_array_getsegcount(PyArray_ARRAY(self), lenp);
}

static size_t
array_getreadbuf(PyArrayObject *self, size_t segment, void **ptrptr)
{
    return npy_array_getreadbuf(PyArray_ARRAY(self), segment, ptrptr);
}


static size_t
array_getwritebuf(PyArrayObject *self, size_t segment, void **ptrptr)
{
    return npy_array_getwritebuf(PyArray_ARRAY(self), segment, ptrptr);
}


static size_t
array_getcharbuf(PyArrayObject *self, size_t segment, char **ptrptr)
{
    return npy_array_getcharbuf(PyArray_ARRAY(self), segment, (void **) ptrptr);
}

/*************************************************************************
 * PEP 3118 buffer protocol
 *
 * Implementing PEP 3118 is somewhat convoluted because of the desirata:
 *
 * - Don't add new members to ndarray or descr structs, to preserve binary
 *   compatibility. (Also, adding the items is actually not very useful,
 *   since mutability issues prevent an 1 to 1 relationship between arrays
 *   and buffer views.)
 *
 * - Don't use bf_releasebuffer, because it prevents PyArg_ParseTuple("s#", ...
 *   from working. Breaking this would cause several backward compatibility
 *   issues already on Python 2.6.
 *
 * - Behave correctly when array is reshaped in-place, or it's dtype is
 *   altered.
 *
 * The solution taken below is to manually track memory allocated for
 * Py_buffers.
 *************************************************************************/

#if PY_VERSION_HEX >= 0x02060000


/*
 * { id(array): [list of pointers to npy_buffer_info_t, the last one is latest] }
 *
 * Because shape, strides, and format can be different for different buffers,
 * we may need to keep track of multiple buffer infos for each array.
 *
 * However, when none of them has changed, the same buffer info may be reused.
 *
 * Thread-safety is provided by GIL.
 */
static PyObject *_buffer_info_cache = NULL;


/* Get buffer info from the global dictionary */
static npy_buffer_info_t*
_buffer_get_info(PyObject *arr)
{
    PyObject *key, *item_list, *item;
    npy_buffer_info_t *info = NULL, *old_info = NULL;

    if (_buffer_info_cache == NULL) {
        _buffer_info_cache = PyDict_New();
        if (_buffer_info_cache == NULL) {
            return NULL;
        }
    }

    /* Compute information */
    info = npy_buffer_info_new(PyArray_ARRAY((PyArrayObject*)arr));
    if (info == NULL) {
        return NULL;
    }

    /* Check if it is identical with an old one; reuse old one, if yes */
    key = PyLong_FromVoidPtr((void*)arr);
    item_list = PyDict_GetItem(_buffer_info_cache, key);

    if (item_list != NULL) {
        Py_INCREF(item_list);
        if (PyList_GET_SIZE(item_list) > 0) {
            item = PyList_GetItem(item_list, PyList_GET_SIZE(item_list) - 1);
            old_info = (npy_buffer_info_t*)PyLong_AsVoidPtr(item);

            if (npy_buffer_info_cmp(info, old_info) == 0) {
                npy_buffer_info_free(info);
                info = old_info;
            }
        }
    }
    else {
        item_list = PyList_New(0);
        PyDict_SetItem(_buffer_info_cache, key, item_list);
    }

    if (info != old_info) {
        /* Needs insertion */
        item = PyLong_FromVoidPtr((void*)info);
        PyList_Append(item_list, item);
        Py_DECREF(item);
    }

    Py_DECREF(item_list);
    Py_DECREF(key);
    return info;
}

/* Clear buffer info from the global dictionary */
static void
_buffer_clear_info(PyObject *arr)
{
    PyObject *key, *item_list, *item;
    npy_buffer_info_t *info;
    int k;

    if (_buffer_info_cache == NULL) {
        return;
    }

    key = PyLong_FromVoidPtr((void*)arr);
    item_list = PyDict_GetItem(_buffer_info_cache, key);
    if (item_list != NULL) {
        for (k = 0; k < PyList_GET_SIZE(item_list); ++k) {
            item = PyList_GET_ITEM(item_list, k);
            info = (npy_buffer_info_t*)PyLong_AsVoidPtr(item);
            npy_buffer_info_free(info);
        }
        PyDict_DelItem(_buffer_info_cache, key);
    }

    Py_DECREF(key);
}

/*
 * Retrieving buffers
 */

static int
array_getbuffer(PyObject *obj, Py_buffer *view, int flags)
{
    PyArrayObject *self;
    npy_buffer_info_t *info = NULL;

    self = (PyArrayObject*)obj;

    /* Check whether we can provide the wanted properties */
    if ((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS &&
        !PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)) {
        PyErr_SetString(PyExc_ValueError, "ndarray is not C-contiguous");
        goto fail;
    }
    if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS &&
        !PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)) {
        PyErr_SetString(PyExc_ValueError, "ndarray is not Fortran contiguous");
        goto fail;
    }
    if ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS
        && !PyArray_ISONESEGMENT(self)) {
        PyErr_SetString(PyExc_ValueError, "ndarray is not contiguous");
        goto fail;
    }
    if ((flags & PyBUF_STRIDES) != PyBUF_STRIDES &&
        (flags & PyBUF_ND) == PyBUF_ND &&
        !PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)) {
        /* Non-strided N-dim buffers must be C-contiguous */
        PyErr_SetString(PyExc_ValueError, "ndarray is not C-contiguous");
        goto fail;
    }
    if ((flags & PyBUF_WRITEABLE) == PyBUF_WRITEABLE &&
        !PyArray_ISWRITEABLE(self)) {
        PyErr_SetString(PyExc_ValueError, "ndarray is not writeable");
        goto fail;
    }

    if (view == NULL) {
        PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
        goto fail;
    }

    /* Fill in information */
    info = _buffer_get_info(obj);
    if (info == NULL) {
        goto fail;
    }

    view->buf = PyArray_DATA(self);
    view->suboffsets = NULL;
    view->itemsize = PyArray_ITEMSIZE(self);
    view->readonly = !PyArray_ISWRITEABLE(self);
    view->internal = NULL;
    view->len = PyArray_NBYTES(self);
    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
        view->format = info->format;
    } else {
        view->format = NULL;
    }
    if ((flags & PyBUF_ND) == PyBUF_ND) {
        view->ndim = info->ndim;
        view->shape = info->shape;
    }
    else {
        view->ndim = 0;
        view->shape = NULL;
    }
    if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
        view->strides = info->strides;
    }
    else {
        view->strides = NULL;
    }
    view->obj = (PyObject*)self;

    Py_INCREF(self);
    return 0;

fail:
    return -1;
}


/*
 * NOTE: for backward compatibility (esp. with PyArg_ParseTuple("s#", ...))
 * we do *not* define bf_releasebuffer at all.
 *
 * Instead, any extra data allocated with the buffer is released only in
 * array_dealloc.
 *
 * Ensuring that the buffer stays in place is taken care by refcounting;
 * ndarrays do not reallocate if there are references to them, and a buffer
 * view holds one reference.
 */

NPY_NO_EXPORT void
_array_dealloc_buffer_info(PyArrayObject *self)
{
    int reset_error_state = 0;
    PyObject *ptype, *pvalue, *ptraceback;

    /* This function may be called when processing an exception --
     * we need to stash the error state to avoid confusing PyDict
     */

    if (PyErr_Occurred()) {
        reset_error_state = 1;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    }

    _buffer_clear_info((PyObject*)self);

    if (reset_error_state) {
        PyErr_Restore(ptype, pvalue, ptraceback);
    }
}

#else

NPY_NO_EXPORT void
_array_dealloc_buffer_info(PyArrayObject *self)
{
}

#endif

/*************************************************************************/

NPY_NO_EXPORT PyBufferProcs array_as_buffer = {
#if !defined(NPY_PY3K)
#if PY_VERSION_HEX >= 0x02050000
    (readbufferproc)array_getreadbuf,       /*bf_getreadbuffer*/
    (writebufferproc)array_getwritebuf,     /*bf_getwritebuffer*/
    (segcountproc)array_getsegcount,        /*bf_getsegcount*/
    (charbufferproc)array_getcharbuf,       /*bf_getcharbuffer*/
#else
    (getreadbufferproc)array_getreadbuf,    /*bf_getreadbuffer*/
    (getwritebufferproc)array_getwritebuf,  /*bf_getwritebuffer*/
    (getsegcountproc)array_getsegcount,     /*bf_getsegcount*/
    (getcharbufferproc)array_getcharbuf,    /*bf_getcharbuffer*/
#endif
#endif
#if PY_VERSION_HEX >= 0x02060000
    (getbufferproc)array_getbuffer,
    (releasebufferproc)0,
#endif
};


/*************************************************************************
 * Convert PEP 3118 format string to PyArray_Descr
 */
#if PY_VERSION_HEX >= 0x02060000

NPY_NO_EXPORT NpyArray_Descr*
_descriptor_from_pep3118_format(char *s)
{
    char *buf, *p;
    int in_name = 0;
    PyObject *descr;
    PyObject *str;
    PyObject *_numpy_internal;
    NpyArray_Descr *descrCore;

    if (s == NULL) {
        return NpyArray_DescrNewFromType(PyArray_BYTE);
    }

    /* Strip whitespace, except from field names */
    buf = (char*)malloc(strlen(s) + 1);
    p = buf;
    while (*s != '\0') {
        if (*s == ':') {
            in_name = !in_name;
            *p = *s;
        }
        else if (in_name || !NpyOS_ascii_isspace(*s)) {
            *p = *s;
        }
        ++p;
        ++s;
    }
    *p = '\0';

    /* Convert */
    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
    if (_numpy_internal == NULL) {
        Py_DECREF(str);
        return NULL;
    }

    str = PyUString_FromStringAndSize(buf, strlen(buf));
    free(buf);
    if (str == NULL) {
        return NULL;
    }
    descr = PyObject_CallMethod(
        _numpy_internal, "_dtype_from_pep3118", "O", str);
    Py_DECREF(_numpy_internal);
    Py_DECREF(str);
    if (descr == NULL) {
        PyErr_Format(PyExc_ValueError,
                     "'%s' is not a valid PEP 3118 buffer format string", buf);
        return NULL;
    }
    if (!PyArray_DescrCheck(descr)) {
        PyErr_Format(PyExc_RuntimeError,
                     "internal error: numpy.core._internal._dtype_from_pep3118 "
                     "did not return a valid dtype, got %s", buf);
        return NULL;
    }
    PyArray_Descr_REF_TO_CORE((PyArray_Descr *)descr, descrCore);
    return descrCore;
}

#else

NPY_NO_EXPORT NpyArray_Descr*
_descriptor_from_pep3118_format(char *s)
{
    PyErr_SetString(PyExc_RuntimeError,
                    "PEP 3118 is not supported on Python versions < 2.6");
    return NULL;
}

#endif
