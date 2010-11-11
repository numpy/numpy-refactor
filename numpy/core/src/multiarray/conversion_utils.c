#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "npy_api.h"

#include "npy_config.h"
#include "numpy/npy_3kcompat.h"

#include "common.h"
#include "ctors.h"
#include "arraytypes.h"

#include "conversion_utils.h"



/****************************************************************
* Useful function for conversion when used with PyArg_ParseTuple
****************************************************************/

NPY_NO_EXPORT NpyArray *
PyArray_FromScalarUnwrap(PyObject *scalar, NpyArray_Descr *outcode);


/*NUMPY_API
 *
 * Useful to pass as converter function for O& processing in PyArgs_ParseTuple.
 *
 * This conversion function can be used with the "O&" argument for
 * PyArg_ParseTuple.  It will immediately return an object of array type
 * or will convert to a CARRAY any other object.
 *
 * If you use PyArray_Converter, you must DECREF the array when finished
 * as you get a new reference to it.
 */
NPY_NO_EXPORT int
PyArray_Converter(PyObject *object, PyObject **address)
{
    if (PyArray_Check(object)) {
        *address = object;
        Py_INCREF(object);
        return PY_SUCCEED;
    }
    else {
        *address = PyArray_FromAny(object, NULL, 0, 0, CARRAY, NULL);
        if (*address == NULL) {
            return PY_FAIL;
        }
        return PY_SUCCEED;
    }
}

/*NUMPY_API
 * Useful to pass as converter function for O& processing in
 * PyArgs_ParseTuple for output arrays
 */
NPY_NO_EXPORT int
PyArray_OutputConverter(PyObject *object, PyArrayObject **address)
{
    if (object == NULL || object == Py_None) {
        *address = NULL;
        return PY_SUCCEED;
    }
    if (PyArray_Check(object)) {
        *address = (PyArrayObject *)object;
        return PY_SUCCEED;
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                        "output must be an array");
        *address = NULL;
        return PY_FAIL;
    }
}

/*NUMPY_API
 * Get intp chunk from sequence
 *
 * This function takes a Python sequence object and allocates and
 * fills in an intp array with the converted values.
 *
 * Remember to free the pointer seq.ptr when done using
 * PyDimMem_FREE(seq.ptr)**
 */
NPY_NO_EXPORT int
PyArray_IntpConverter(PyObject *obj, PyArray_Dims *seq)
{
    int len;
    int nd;

    seq->ptr = NULL;
    seq->len = 0;
    if (obj == Py_None) {
        return PY_SUCCEED;
    }
    len = PySequence_Size(obj);
    if (len == -1) {
        /* Check to see if it is a number */
        if (PyNumber_Check(obj)) {
            len = 1;
        }
    }
    if (len < 0) {
        PyErr_SetString(PyExc_TypeError,
                        "expected sequence object with len >= 0");
        return PY_FAIL;
    }
    if (len > MAX_DIMS) {
        PyErr_Format(PyExc_ValueError, "sequence too large; "
                     "must be smaller than %d", MAX_DIMS);
        return PY_FAIL;
    }
    if (len > 0) {
        seq->ptr = PyDimMem_NEW(len);
        if (seq->ptr == NULL) {
            PyErr_NoMemory();
            return PY_FAIL;
        }
    }
    seq->len = len;
    nd = PyArray_IntpFromSequence(obj, (intp *)seq->ptr, len);
    if (nd == -1 || nd != len) {
        PyDimMem_FREE(seq->ptr);
        seq->ptr = NULL;
        return PY_FAIL;
    }
    return PY_SUCCEED;
}

/*NUMPY_API
 * Get buffer chunk from object
 *
 * this function takes a Python object which exposes the (single-segment)
 * buffer interface and returns a pointer to the data segment
 *
 * You should increment the reference count by one of buf->base
 * if you will hang on to a reference
 *
 * You only get a borrowed reference to the object. Do not free the
 * memory...
 */
NPY_NO_EXPORT int
PyArray_BufferConverter(PyObject *obj, PyArray_Chunk *buf)
{
    Py_ssize_t buflen;

    buf->ptr = NULL;
    buf->flags = BEHAVED;
    buf->base = NULL;
    if (obj == Py_None) {
        return PY_SUCCEED;
    }
    if (PyObject_AsWriteBuffer(obj, &(buf->ptr), &buflen) < 0) {
        PyErr_Clear();
        buf->flags &= ~WRITEABLE;
        if (PyObject_AsReadBuffer(obj, (const void **)&(buf->ptr),
                                  &buflen) < 0) {
            return PY_FAIL;
        }
    }
    buf->len = (intp) buflen;

    /* Point to the base of the buffer object if present */
#if defined(NPY_PY3K)
    if (PyMemoryView_Check(obj)) {
        buf->base = PyMemoryView_GET_BASE(obj);
    }
#else
    if (PyBuffer_Check(obj)) {
        buf->base = ((PyArray_Chunk *)obj)->base;
    }
#endif
    if (buf->base == NULL) {
        buf->base = obj;
    }
    return PY_SUCCEED;
}

/*NUMPY_API
 * Get axis from an object (possibly None) -- a converter function,
 */
NPY_NO_EXPORT int
PyArray_AxisConverter(PyObject *obj, int *axis)
{
    if (obj == Py_None) {
        *axis = MAX_DIMS;
    }
    else {
        *axis = (int) PyInt_AsLong(obj);
        if (PyErr_Occurred()) {
            return PY_FAIL;
        }
    }
    return PY_SUCCEED;
}

/*NUMPY_API
 * Convert an object to true / false
 */
NPY_NO_EXPORT int
PyArray_BoolConverter(PyObject *object, Bool *val)
{
    if (PyObject_IsTrue(object)) {
        *val = TRUE;
    }
    else {
        *val = FALSE;
    }
    if (PyErr_Occurred()) {
        return PY_FAIL;
    }
    return PY_SUCCEED;
}

/*NUMPY_API
 * Convert object to endian
 */
NPY_NO_EXPORT int
PyArray_ByteorderConverter(PyObject *obj, char *endian)
{
    char *str;
    PyObject *tmp = NULL;

    if (PyUnicode_Check(obj)) {
        obj = tmp = PyUnicode_AsASCIIString(obj);
    }

    *endian = PyArray_SWAP;
    str = PyBytes_AsString(obj);
    if (!str) {
        Py_XDECREF(tmp);
        return PY_FAIL;
    }
    if (strlen(str) < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "Byteorder string must be at least length 1");
        Py_XDECREF(tmp);
        return PY_FAIL;
    }
    *endian = str[0];
    if (str[0] != PyArray_BIG && str[0] != PyArray_LITTLE
        && str[0] != PyArray_NATIVE && str[0] != PyArray_IGNORE) {
        if (str[0] == 'b' || str[0] == 'B') {
            *endian = PyArray_BIG;
        }
        else if (str[0] == 'l' || str[0] == 'L') {
            *endian = PyArray_LITTLE;
        }
        else if (str[0] == 'n' || str[0] == 'N') {
            *endian = PyArray_NATIVE;
        }
        else if (str[0] == 'i' || str[0] == 'I') {
            *endian = PyArray_IGNORE;
        }
        else if (str[0] == 's' || str[0] == 'S') {
            *endian = PyArray_SWAP;
        }
        else {
            PyErr_Format(PyExc_ValueError,
                         "%s is an unrecognized byteorder",
                         str);
            Py_XDECREF(tmp);
            return PY_FAIL;
        }
    }
    Py_XDECREF(tmp);
    return PY_SUCCEED;
}

/*NUMPY_API
 * Convert object to sort kind
 */
NPY_NO_EXPORT int
PyArray_SortkindConverter(PyObject *obj, NPY_SORTKIND *sortkind)
{
    char *str;
    PyObject *tmp = NULL;

    if (PyUnicode_Check(obj)) {
        obj = tmp = PyUnicode_AsASCIIString(obj);
    }

    *sortkind = PyArray_QUICKSORT;
    str = PyBytes_AsString(obj);
    if (!str) {
        Py_XDECREF(tmp);
        return PY_FAIL;
    }
    if (strlen(str) < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "Sort kind string must be at least length 1");
        Py_XDECREF(tmp);
        return PY_FAIL;
    }
    if (str[0] == 'q' || str[0] == 'Q') {
        *sortkind = PyArray_QUICKSORT;
    }
    else if (str[0] == 'h' || str[0] == 'H') {
        *sortkind = PyArray_HEAPSORT;
    }
    else if (str[0] == 'm' || str[0] == 'M') {
        *sortkind = PyArray_MERGESORT;
    }
    else {
        PyErr_Format(PyExc_ValueError,
                     "%s is an unrecognized kind of sort",
                     str);
        Py_XDECREF(tmp);
        return PY_FAIL;
    }
    Py_XDECREF(tmp);
    return PY_SUCCEED;
}

/*NUMPY_API
 * Convert object to searchsorted side
 */
NPY_NO_EXPORT int
PyArray_SearchsideConverter(PyObject *obj, void *addr)
{
    NPY_SEARCHSIDE *side = (NPY_SEARCHSIDE *)addr;
    char *str;
    PyObject *tmp = NULL;

    if (PyUnicode_Check(obj)) {
        obj = tmp = PyUnicode_AsASCIIString(obj);
    }

    str = PyBytes_AsString(obj);
    if (!str || strlen(str) < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "expected nonempty string for keyword 'side'");
        Py_XDECREF(tmp);
        return PY_FAIL;
    }

    if (str[0] == 'l' || str[0] == 'L') {
        *side = NPY_SEARCHLEFT;
    }
    else if (str[0] == 'r' || str[0] == 'R') {
        *side = NPY_SEARCHRIGHT;
    }
    else {
        PyErr_Format(PyExc_ValueError,
                     "'%s' is an invalid value for keyword 'side'", str);
        Py_XDECREF(tmp);
        return PY_FAIL;
    }
    Py_XDECREF(tmp);
    return PY_SUCCEED;
}

/*****************************
* Other conversion functions
*****************************/

/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_PyIntAsInt(PyObject *o)
{
    long long_value = -1;
    PyObject *obj;
    static char *msg = "an integer is required";
    NpyArray *arr;
    NpyArray_Descr *descr;
    int ret;


    if (!o) {
        PyErr_SetString(PyExc_TypeError, msg);
        return -1;
    }
    if (PyInt_Check(o)) {
        long_value = (long) PyInt_AS_LONG(o);
        goto finish;
    } else if (PyLong_Check(o)) {
        long_value = (long) PyLong_AsLong(o);
        goto finish;
    }

    descr = &npy_INT_Descr;
    arr = NULL;
    if (PyArray_Check(o)) {
        if (PyArray_SIZE(o)!=1 || !PyArray_ISINTEGER(o)) {
            PyErr_SetString(PyExc_TypeError, msg);
            return -1;
        }
        Npy_INCREF(descr);
        arr = NpyArray_CastToType(PyArray_ARRAY((PyArrayObject *)o), descr, 0);
    }
    if (PyArray_IsScalar(o, Integer)) {
        Npy_INCREF(descr);
        arr = PyArray_FromScalarUnwrap(o, descr);
    }
    if (arr != NULL) {
        ret = *((int *)NpyArray_DATA(arr));
        Npy_DECREF(arr);
        return ret;
    }
#if (PY_VERSION_HEX >= 0x02050000)
    if (PyIndex_Check(o)) {
        PyObject* value = PyNumber_Index(o);
        long_value = (longlong) PyInt_AsSsize_t(value);
        goto finish;
    }
#endif
    if (Py_TYPE(o)->tp_as_number != NULL &&         \
        Py_TYPE(o)->tp_as_number->nb_int != NULL) {
        obj = Py_TYPE(o)->tp_as_number->nb_int(o);
        if (obj == NULL) {
            return -1;
        }
        long_value = (long) PyLong_AsLong(obj);
        Py_DECREF(obj);
    }
#if !defined(NPY_PY3K)
    else if (Py_TYPE(o)->tp_as_number != NULL &&                    \
             Py_TYPE(o)->tp_as_number->nb_long != NULL) {
        obj = Py_TYPE(o)->tp_as_number->nb_long(o);
        if (obj == NULL) {
            return -1;
        }
        long_value = (long) PyLong_AsLong(obj);
        Py_DECREF(obj);
    }
#endif
    else {
        PyErr_SetString(PyExc_NotImplementedError,"");
    }

 finish:
    if error_converting(long_value) {
            PyErr_SetString(PyExc_TypeError, msg);
            return -1;
        }

#if (NPY_SIZEOF_LONG > NPY_SIZEOF_INT)
    if ((long_value < INT_MIN) || (long_value > INT_MAX)) {
        PyErr_SetString(PyExc_ValueError, "integer won't fit into a C int");
        return -1;
    }
#endif
    return (int) long_value;
}

/*NUMPY_API*/
NPY_NO_EXPORT intp
PyArray_PyIntAsIntp(PyObject *o)
{
    longlong long_value = -1;
    PyObject *obj;
    static char *msg = "an integer is required";
    NpyArray *arr;
    NpyArray_Descr *descr;
    intp ret;

    if (!o) {
        PyErr_SetString(PyExc_TypeError, msg);
        return -1;
    }
    if (PyInt_Check(o)) {
        long_value = (longlong) PyInt_AS_LONG(o);
        goto finish;
    } else if (PyLong_Check(o)) {
        long_value = (longlong) PyLong_AsLongLong(o);
        goto finish;
    }

#if NPY_SIZEOF_PTR == NPY_SIZEOF_LONG
    descr = &npy_LONG_Descr;
#elif NPY_SIZEOF_PTR == NPY_SIZEOF_INT
    descr = &npy_INT_Descr;
#else
    descr = &npy_LONGLONG_Descr;
#endif
    arr = NULL;

    if (PyArray_Check(o)) {
        if (PyArray_SIZE(o)!=1 || !PyArray_ISINTEGER(o)) {
            PyErr_SetString(PyExc_TypeError, msg);
            return -1;
        }
        Npy_INCREF(descr);
        arr = NpyArray_CastToType(PyArray_ARRAY((PyArrayObject *)o), descr, 0);
    }
    else if (PyArray_IsScalar(o, Integer)) {
        Npy_INCREF(descr);
        arr = PyArray_FromScalarUnwrap(o, descr);
    }
    if (arr != NULL) {
        ret = *((intp *)NpyArray_DATA(arr));
        Npy_DECREF(arr);
        return ret;
    }

#if (PY_VERSION_HEX >= 0x02050000)
    if (PyIndex_Check(o)) {
        PyObject* value = PyNumber_Index(o);
        if (value == NULL) {
            return -1;
        }
        long_value = (longlong) PyInt_AsSsize_t(value);
        goto finish;
    }
#endif
#if !defined(NPY_PY3K)
    if (Py_TYPE(o)->tp_as_number != NULL &&                 \
        Py_TYPE(o)->tp_as_number->nb_long != NULL) {
        obj = Py_TYPE(o)->tp_as_number->nb_long(o);
        if (obj != NULL) {
            long_value = (longlong) PyLong_AsLongLong(obj);
            Py_DECREF(obj);
        }
    }
    else
#endif
    if (Py_TYPE(o)->tp_as_number != NULL &&                 \
             Py_TYPE(o)->tp_as_number->nb_int != NULL) {
        obj = Py_TYPE(o)->tp_as_number->nb_int(o);
        if (obj != NULL) {
            long_value = (longlong) PyLong_AsLongLong(obj);
            Py_DECREF(obj);
        }
    }
    else {
        PyErr_SetString(PyExc_NotImplementedError,"");
    }

 finish:
    if error_converting(long_value) {
            PyErr_SetString(PyExc_TypeError, msg);
            return -1;
        }

#if (NPY_SIZEOF_LONGLONG > NPY_SIZEOF_PTR)
    if ((long_value < MIN_INTP) || (long_value > MAX_INTP)) {
        PyErr_SetString(PyExc_ValueError,
                        "integer won't fit into a C intp");
        return -1;
    }
#endif
    return (intp) long_value;
}

/*NUMPY_API
 * PyArray_IntpFromSequence
 * Returns the number of dimensions or -1 if an error occurred.
 * vals must be large enough to hold maxvals
 */
NPY_NO_EXPORT int
PyArray_IntpFromSequence(PyObject *seq, intp *vals, int maxvals)
{
    int nd, i;
    PyObject *op, *err;

    /*
     * Check to see if sequence is a single integer first.
     * or, can be made into one
     */
    if ((nd=PySequence_Length(seq)) == -1) {
        if (PyErr_Occurred()) PyErr_Clear();
#if NPY_SIZEOF_LONG >= NPY_SIZEOF_PTR && !defined(NPY_PY3K)
        if (!(op = PyNumber_Int(seq))) {
            return -1;
        }
#else
        if (!(op = PyNumber_Long(seq))) {
            return -1;
        }
#endif
        nd = 1;
#if NPY_SIZEOF_LONG >= NPY_SIZEOF_PTR
        vals[0] = (intp ) PyInt_AsLong(op);
#else
        vals[0] = (intp ) PyLong_AsLongLong(op);
#endif
        Py_DECREF(op);

        /*
         * Check wether there was an error - if the error was an overflow, raise
         * a ValueError instead to be more helpful
         */
        if(vals[0] == -1) {
            err = PyErr_Occurred();
            if (err  &&
                PyErr_GivenExceptionMatches(err, PyExc_OverflowError)) {
                PyErr_SetString(PyExc_ValueError,
                        "Maximum allowed dimension exceeded");
            }
            if(err != NULL) {
                return -1;
            }
        }
    }
    else {
        for (i = 0; i < MIN(nd,maxvals); i++) {
            op = PySequence_GetItem(seq, i);
            if (op == NULL) {
                return -1;
            }
#if NPY_SIZEOF_LONG >= NPY_SIZEOF_PTR
            vals[i]=(intp )PyInt_AsLong(op);
#else
            vals[i]=(intp )PyLong_AsLongLong(op);
#endif
            Py_DECREF(op);

            /*
             * Check wether there was an error - if the error was an overflow,
             * raise a ValueError instead to be more helpful
             */
            if(vals[0] == -1) {
                err = PyErr_Occurred();
                if (err  &&
                    PyErr_GivenExceptionMatches(err, PyExc_OverflowError)) {
                    PyErr_SetString(PyExc_ValueError,
                            "Maximum allowed dimension exceeded");
                }
                if(err != NULL) {
                    return -1;
                }
            }
        }
    }
    return nd;
}

/*NUMPY_API
 * Typestr converter
 */
NPY_NO_EXPORT int
PyArray_TypestrConvert(int itemsize, int gentype)
{
    return NpyArray_TypestrConvert(itemsize, gentype);
}


/* Lifted from numarray */
/* TODO: not documented */
/*NUMPY_API
 PyArray_IntTupleFromIntp
 */
NPY_NO_EXPORT PyObject *
PyArray_IntTupleFromIntp(int len, intp *vals)
{
    int i;
    PyObject *intTuple = PyTuple_New(len);

    if (!intTuple) {
        goto fail;
    }
    for (i = 0; i < len; i++) {
#if NPY_SIZEOF_PTR <= NPY_SIZEOF_LONG
        PyObject *o = PyInt_FromLong((long) vals[i]);
#else
        PyObject *o = PyLong_FromLongLong((longlong) vals[i]);
#endif
        if (!o) {
            Py_DECREF(intTuple);
            intTuple = NULL;
            goto fail;
        }
        PyTuple_SET_ITEM(intTuple, i, o);
    }

 fail:
    return intTuple;
}


static int
convert_array(PyObject*obj, int type, NpyArray **parray)
{
    NpyArray_Descr *descr;
    NpyArray *array;

    descr = NpyArray_DescrFromType(type);
    array = NpyArray_FromArray(PyArray_ARRAY(obj), descr,
                               NPY_FORCECAST);
    if (array == NULL) {
        return -1;
    } else {
        *parray = array;
        return 0;
    }
}

static int
convert_slice_nostop(PySliceObject* slice, NpyIndexSliceNoStop* islice)
{
    if (slice->step == Py_None) {
        islice->step = 1;
    }
    else {
        islice->step = PyArray_PyIntAsIntp(slice->step);
        if (islice->step == -1 && PyErr_Occurred()) {
            return -1;
        }
        if (islice->step == 0) {
            PyErr_SetString(PyExc_ValueError,
                            "slice step cannot be zero");
            return -1;
        }
    }

    if (slice->start == Py_None) {
        if (islice->step > 0) {
            islice->start = 0;
        } else {
            islice->start = -1;
        }
    }
    else {
        islice->start = PyArray_PyIntAsIntp(slice->start);
        if (islice->start == -1 && PyErr_Occurred()) {
            return -1;
        }
    }

    return 0;
}

static int
convert_slice(PySliceObject* slice, NpyIndexSlice* islice)
{
    if (slice->step == Py_None) {
        islice->step = 1;
    }
    else {
        islice->step = PyArray_PyIntAsIntp(slice->step);
        if (islice->step == -1 && PyErr_Occurred()) {
            return -1;
        }
        if (islice->step == 0) {
            PyErr_SetString(PyExc_ValueError,
                            "slice step cannot be zero");
            return -1;
        }
    }

    if (slice->start == Py_None) {
        if (islice->step > 0) {
            islice->start = 0;
        } else {
            islice->start = -1;
        }
    }
    else {
        islice->start = PyArray_PyIntAsIntp(slice->start);
        if (islice->start == -1 && PyErr_Occurred()) {
            return -1;
        }
    }

    islice->stop = PyArray_PyIntAsIntp(slice->stop);
    if (islice->stop == -1 && PyErr_Occurred()) {
        return -1;
    }

    return 0;
}

static int
convert_sequence(PyObject* seq, NpyArray** parray)
{
    PyObject* pyArray;
    NpyArray_Descr* descr;

    descr = NpyArray_DescrFromType(NPY_INTP);
    pyArray = PyArray_FromAnyUnwrap(seq, descr, 0, 0, NPY_FORCECAST, NULL);
    if (pyArray == NULL) {
        return -1;
    }

    *parray = PyArray_ARRAY(pyArray);
    Npy_INCREF(*parray);
    Py_DECREF(pyArray);

    return 0;
}



static int
convert_single_index(PyObject* obj, NpyIndex* index)
{
    /* None is a newaxis. */
    if (obj == Py_None) {
        index->type = NPY_INDEX_NEWAXIS;
    }
    else if (obj == Py_Ellipsis) {
        index->type = NPY_INDEX_ELLIPSIS;
    }
    else if (obj == Py_True) {
        index->type = NPY_INDEX_BOOL;
        index->index.boolean = PyObject_IsTrue(obj);
    }
    else if (obj == Py_False) {
        index->type = NPY_INDEX_BOOL;
        index->index.boolean = PyObject_IsTrue(obj);
    }
    /* Try as int. */
    else if (PyInt_Check(obj)) {
        longlong long_value = (longlong) PyInt_AS_LONG(obj);
#if (NPY_SIZEOF_LONGLONG > NPY_SIZEOF_PTR)
        if ((long_value < MIN_INTP) || (long_value > MAX_INTP)) {
            PyErr_SetString(PyExc_ValueError,
                            "integer won't fit into a C intp");
            return -1;
        }
#endif
        index->type = NPY_INDEX_INTP;
#undef intp
        index->index.intp = (npy_intp) long_value;
#define intp npy_intp
    }
    /* Try as long. */
    else if (PyLong_Check(obj)) {
        longlong long_value = (longlong) PyLong_AsLongLong(obj);
#if (NPY_SIZEOF_LONGLONG > NPY_SIZEOF_PTR)
        if ((long_value < MIN_INTP) || (long_value > MAX_INTP)) {
            PyErr_SetString(PyExc_ValueError,
                            "integer won't fit into a C intp");
            return -1;
        }
#endif
        index->type = NPY_INDEX_INTP;
#undef intp
        index->index.intp = (npy_intp) long_value;
#define intp npy_intp
    }
    /* Slices are converted. */
    else if (PySlice_Check(obj)) {
        PySliceObject* slice = (PySliceObject*)obj;

        if (slice->stop == Py_None) {
            index->type = NPY_INDEX_SLICE_NOSTOP;
            if (convert_slice_nostop(slice, &index->index.slice_nostop) < 0) {
                return -1;
            }
        } else {
            index->type = NPY_INDEX_SLICE;
            if (convert_slice(slice, &index->index.slice) < 0) {
                return -1;
            }
        }
    }
    /* Try as a boolean scalar. */
    else if (PyArray_IsScalar(obj, Bool) ||
             (PyArray_Check(obj) && PyArray_NDIM(obj) == 0 &&
              PyArray_ISBOOL(obj))) {
        index->type = NPY_INDEX_BOOL;
        index->index.boolean = PyObject_IsTrue(obj);
    }
    /* Strings and unicode. */
    else if (PyString_Check(obj) || PyUnicode_Check(obj)) {
        index->type = NPY_INDEX_STRING;
        PRINT(obj);
#if defined(NPY_PY3K)
        index->index.string = PyBytes_AsString(PyUnicode_Check(obj) ?
                                          PyUnicode_AsASCIIString(obj) : obj);
#else
        index->index.string = PyString_AsString(x);
#endif
        if (index->index.string == NULL) {
            return -1;
        }
    }
    /* Arrays must be bool or integeter and will be converted to
       intp or bool arrays. */
    else if (PyArray_Check(obj)) {
        if (PyArray_NDIM(obj) == 0 &&
            PyArray_ISINTEGER(obj)) {
            /* We treat 0-d arrays as scalars.  */
            npy_intp val = PyArray_PyIntAsIntp(obj);
            if (val == -1 && PyErr_Occurred()) {
                return -1;
            }
            index->type = NPY_INDEX_INTP;
#undef intp
            index->index.intp = val;
#define intp npy_intp
        }
        else if (PyArray_ISINTEGER(obj)) {
            index->type = NPY_INDEX_INTP_ARRAY;
            if (convert_array(obj, NPY_INTP, &index->index.intp_array) < 0) {
                return -1;
            }
        } else if (PyArray_ISBOOL(obj)) {
            index->type = NPY_INDEX_BOOL_ARRAY;
            if (convert_array(obj, NPY_BOOL, &index->index.bool_array) < 0) {
                return -1;
            }
        } else {
            PyErr_SetString(PyExc_IndexError,
                            "arrays used as indices must be of "
                            "integer (or boolean) type");
            return -1;
        }
    }
    /* Try to convert other sequences to intp arrays. */
    else if (PySequence_Check(obj)) {
        index->type = NPY_INDEX_INTP_ARRAY;
        if (convert_sequence(obj, &index->index.intp_array) < 0) {
            return -1;
        }
    }
    /* Anything else we try to convert to an intp. */
    else {
        npy_intp val;
        val = PyArray_PyIntAsIntp(obj);
        if (val == -1 && PyErr_Occurred()) {
            return -1;
        }
        index->type = NPY_INDEX_INTP;
#undef intp
        index->index.intp = val;
#define intp npy_intp
    }

    return 0;
}

/*
 * Returns whether a sequence should be treated like a tuple.
 * Essentially it should be unless it looks like a sequence of
 * indexes.
 */
static npy_bool
sequence_tuple(PyObject* seq)
{
    Py_ssize_t i, n;
    PyObject *item;

    n = PySequence_Length(seq);
    if (n < 0 || n > NPY_MAXDIMS) {
        return NPY_FALSE;
    }

    for (i=0; i<n; i++) {
        item = PySequence_GetItem(seq, i);
        if (item == Py_None ||
            item == Py_Ellipsis ||
            PySlice_Check(item) ||
            PySequence_Check(item)) {
            Py_DECREF(item);
            return NPY_TRUE;
        }
        Py_DECREF(item);
    }
    return NPY_FALSE;
}

/*
 * Converts a python object into an array of indexes.
 */
NPY_NO_EXPORT int
PyArray_IndexConverter(PyObject *index, NpyIndex* indexes)
{
    Py_ssize_t i, n;
    PyObject *item;

    /* This is the simplest case. We have multiple args as a
     * tuple.  Just convert each one. */
    if (PyTuple_Check(index)) {
        n = PyTuple_GET_SIZE(index);
        if (n >= NPY_MAXDIMS) {
            PyErr_SetString(PyExc_IndexError, "too many indices");
            return -1;
        }
        else {
            for (i=0; i<n; i++) {
                item = PyTuple_GET_ITEM(index, i);
                if (convert_single_index(item, &indexes[i]) < 0) {
                    NpyArray_IndexDealloc(indexes, i);
                    return -1;
                }
            }
            return n;
        }
    }

    if (PyArray_Check(index)) {
        if (convert_single_index(index, &indexes[0]) < 0) {
            return -1;
        }
        return 1;
    }

    if (PyString_Check(index) || PyUnicode_Check(index)) {
        if (convert_single_index(index, &indexes[0]) < 0) {
            return -1;
        }
        return 1;
    }

    /* For sequences that don't look like a sequence of intp
     * treat them like a tuple. */
    if (PySequence_Check(index) && sequence_tuple(index)) {
        n = PySequence_Length(index);
        for (i=0; i<n; i++) {
            item = PySequence_GetItem(index, i);
            if (convert_single_index(item, &indexes[i]) < 0) {
                NpyArray_IndexDealloc(indexes, i);
                Py_DECREF(item);
                return -1;
            }
            Py_DECREF(item);
        }
        return n;
    }

    if (convert_single_index(index, &indexes[0]) < 0) {
        return -1;
    }

    return 1;
}
