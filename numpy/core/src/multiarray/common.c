#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "npy_api.h"

#include "npy_config.h"
#include "numpy_3kcompat.h"

#include "usertypes.h"

#include "common.h"
#include "buffer.h"

/*
 * new reference
 * doesn't alter refcount of chktype or mintype ---
 * unless one of them is returned
 */
NPY_NO_EXPORT PyArray_Descr *
_array_small_type(PyArray_Descr *chktype, PyArray_Descr* mintype)
{
    NpyArray_Descr *result = NpyArray_SmallType(chktype->descr, mintype->descr);
    PyArray_Descr_RETURN( result );
}

NPY_NO_EXPORT NpyArray_Descr *
PyArray_DescrFromScalarUnwrap(PyObject *sc);



NPY_NO_EXPORT NpyArray_Descr *
_array_find_python_scalar_type(PyObject *op)
{
    if (PyFloat_Check(op)) {
        return NpyArray_DescrFromType(PyArray_DOUBLE);
    }
    else if (PyComplex_Check(op)) {
        return NpyArray_DescrFromType(PyArray_CDOUBLE);
    }
    else if (PyInt_Check(op)) {
        /* bools are a subclass of int */
        if (PyBool_Check(op)) {
            return NpyArray_DescrFromType(PyArray_BOOL);
        }
        else {
            return NpyArray_DescrFromType(PyArray_LONG);
        }
    }
    else if (PyLong_Check(op)) {
        /* if integer can fit into a longlong then return that*/
        if ((PyLong_AsLongLong(op) == -1) && PyErr_Occurred()) {
            PyErr_Clear();
            return NpyArray_DescrFromType(PyArray_OBJECT);
        }
        return NpyArray_DescrFromType(PyArray_LONGLONG);
    }
    return NULL;
}

static NpyArray_Descr *
_use_default_type(PyObject *op)
{
    int typenum;

    typenum = PyArray_TypeNumFromTypeObj(Py_TYPE(op));
    if (typenum == NPY_NOTYPE) {
        typenum = PyArray_OBJECT;
    }
    return NpyArray_DescrFromType(typenum);
}


/*
 * op is an object to be converted to an ndarray.
 *
 * minitype is the minimum type-descriptor needed.
 *
 * max is the maximum number of dimensions -- used for recursive call
 * to avoid infinite recursion...
 */
NPY_NO_EXPORT NpyArray_Descr *
_array_find_type(PyObject *op, NpyArray_Descr *minitype, int max)
{
    int l;
    PyObject *ip;
    NpyArray_Descr *chktype = NULL;
    NpyArray_Descr *outtype;
#if PY_VERSION_HEX >= 0x02060000
    Py_buffer buffer_view;
#endif

    /*
     * These need to come first because if op already carries
     * a descr structure, then we want it to be the result if minitype
     * is NULL.
     */
    if (PyArray_Check(op)) {
        chktype = PyArray_DESCR(op);
        Npy_INCREF(chktype);
        if (minitype == NULL) {
            return chktype;
        }
        Npy_INCREF(minitype);
        goto finish;
    }

    if (PyArray_IsScalar(op, Generic)) {
        chktype = PyArray_DescrFromScalarUnwrap(op);
        if (minitype == NULL) {
            return chktype;
        }
        Npy_INCREF(minitype);
        goto finish;
    }

    if (minitype == NULL) {
        minitype = NpyArray_DescrFromType(PyArray_BOOL);
    }
    else {
        Npy_INCREF(minitype);
    }
    if (max < 0) {
        goto deflt;
    }
    chktype = _array_find_python_scalar_type(op);
    if (chktype) {
        goto finish;
    }

    if (PyBytes_Check(op)) {
        chktype = NpyArray_DescrNewFromType(PyArray_STRING);
        chktype->elsize = PyString_GET_SIZE(op);
        goto finish;
    }

    if (PyUnicode_Check(op)) {
        chktype = NpyArray_DescrNewFromType(PyArray_UNICODE);
        chktype->elsize = PyUnicode_GET_DATA_SIZE(op);
#ifndef Py_UNICODE_WIDE
        chktype->elsize <<= 1;
#endif
        goto finish;
    }

#if PY_VERSION_HEX >= 0x02060000
    /* PEP 3118 buffer interface */
    memset(&buffer_view, 0, sizeof(Py_buffer));
    if (PyObject_GetBuffer(op, &buffer_view, PyBUF_FORMAT|PyBUF_STRIDES) == 0 ||
        PyObject_GetBuffer(op, &buffer_view, PyBUF_FORMAT) == 0) {

        PyErr_Clear();
        chktype = _descriptor_from_pep3118_format(buffer_view.format);
        PyBuffer_Release(&buffer_view);
        if (chktype) {
            goto finish;
        }
    }
    else if (PyObject_GetBuffer(op, &buffer_view, PyBUF_STRIDES) == 0 ||
             PyObject_GetBuffer(op, &buffer_view, PyBUF_SIMPLE) == 0) {

        PyErr_Clear();
        chktype = NpyArray_DescrNewFromType(PyArray_VOID);
        chktype->elsize = buffer_view.itemsize;
        PyBuffer_Release(&buffer_view);
        goto finish;
    }
    else {
        PyErr_Clear();
    }
#endif

    if ((ip=PyObject_GetAttrString(op, "__array_interface__"))!=NULL) {
        if (PyDict_Check(ip)) {
            PyObject *new;
            new = PyDict_GetItemString(ip, "typestr");
            if (new && PyString_Check(new)) {
                chktype =_array_typedescr_fromstr(PyString_AS_STRING(new));
            }
        }
        Py_DECREF(ip);
        if (chktype) {
            goto finish;
        }
    }
    else {
        PyErr_Clear();
    }
    if ((ip=PyObject_GetAttrString(op, "__array_struct__")) != NULL) {
        PyArrayInterface *inter;
        char buf[40];

        if (PyCapsule_Check(ip)) {
            inter = (PyArrayInterface *)PyCapsule_AsVoidPtr(ip);
            if (inter->two == 2) {
                PyOS_snprintf(buf, sizeof(buf),
                        "|%c%d", inter->typekind, inter->itemsize);
                chktype = _array_typedescr_fromstr(buf);
            }
        }
        Py_DECREF(ip);
        if (chktype) {
            goto finish;
        }
    }
    else {
        PyErr_Clear();
    }

#if !defined(NPY_PY3K)
    if (PyBuffer_Check(op)) {
        chktype = NpyArray_DescrNewFromType(PyArray_VOID);
        chktype->elsize = Py_TYPE(op)->tp_as_sequence->sq_length(op);
        PyErr_Clear();
        goto finish;
    }
#endif

    if (PyObject_HasAttrString(op, "__array__")) {
        ip = PyObject_CallMethod(op, "__array__", NULL);
        if(ip && PyArray_Check(ip)) {
            chktype = PyArray_DESCR(ip);
            Npy_INCREF(chktype);
            Py_DECREF(ip);
            goto finish;
        }
        Py_XDECREF(ip);
        if (PyErr_Occurred()) PyErr_Clear();
    }

#if defined(NPY_PY3K)
    /* FIXME: XXX -- what is the correct thing to do here? */
#else
    if (PyInstance_Check(op)) {
        goto deflt;
    }
#endif
    if (PySequence_Check(op)) {
        l = PyObject_Length(op);
        if (l < 0 && PyErr_Occurred()) {
            PyErr_Clear();
            goto deflt;
        }
        if (l == 0 && minitype->type_num == PyArray_BOOL) {
            Npy_DECREF(minitype);
            minitype = NpyArray_DescrFromType(PyArray_DEFAULT);
        }
        while (--l >= 0) {
            NpyArray_Descr *newtype;
            ip = PySequence_GetItem(op, l);
            if (ip==NULL) {
                PyErr_Clear();
                goto deflt;
            }
            chktype = _array_find_type(ip, minitype, max-1);
            newtype = NpyArray_SmallType(chktype, minitype);
            Npy_DECREF(minitype);
            minitype = newtype;
            Npy_DECREF(chktype);
            Py_DECREF(ip);
        }
        chktype = minitype;
        Npy_INCREF(minitype);
        goto finish;
    }


 deflt:
    chktype = _use_default_type(op);

 finish:
    outtype = NpyArray_SmallType(chktype, minitype);
    Npy_DECREF(chktype);
    Npy_DECREF(minitype);
    /*
     * VOID Arrays should not occur by "default"
     * unless input was already a VOID
     */
    if (outtype->type_num == PyArray_VOID &&
        minitype->type_num != PyArray_VOID) {
        Npy_DECREF(outtype);
        return NpyArray_DescrFromType(PyArray_OBJECT);
    }
    return outtype;
}

/* new reference */
NPY_NO_EXPORT NpyArray_Descr *
_array_typedescr_fromstr(char *str)
{
    NpyArray_Descr *descr;
    int type_num;
    char typechar;
    int size;
    char msg[] = "unsupported typestring";
    int swap;
    char swapchar;

    swapchar = str[0];
    str += 1;

    typechar = str[0];
    size = atoi(str + 1);
    switch (typechar) {
    case 'b':
        if (size == sizeof(Bool)) {
            type_num = PyArray_BOOL;
        }
        else {
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        break;
    case 'u':
        if (size == sizeof(uintp)) {
            type_num = PyArray_UINTP;
        }
        else if (size == sizeof(char)) {
            type_num = PyArray_UBYTE;
        }
        else if (size == sizeof(short)) {
            type_num = PyArray_USHORT;
        }
        else if (size == sizeof(ulong)) {
            type_num = PyArray_ULONG;
        }
        else if (size == sizeof(int)) {
            type_num = PyArray_UINT;
        }
        else if (size == sizeof(ulonglong)) {
            type_num = PyArray_ULONGLONG;
        }
        else {
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        break;
    case 'i':
        if (size == sizeof(intp)) {
            type_num = PyArray_INTP;
        }
        else if (size == sizeof(char)) {
            type_num = PyArray_BYTE;
        }
        else if (size == sizeof(short)) {
            type_num = PyArray_SHORT;
        }
        else if (size == sizeof(long)) {
            type_num = PyArray_LONG;
        }
        else if (size == sizeof(int)) {
            type_num = PyArray_INT;
        }
        else if (size == sizeof(longlong)) {
            type_num = PyArray_LONGLONG;
        }
        else {
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        break;
    case 'f':
        if (size == sizeof(float)) {
            type_num = PyArray_FLOAT;
        }
        else if (size == sizeof(double)) {
            type_num = PyArray_DOUBLE;
        }
        else if (size == sizeof(longdouble)) {
            type_num = PyArray_LONGDOUBLE;
        }
        else {
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        break;
    case 'c':
        if (size == sizeof(float)*2) {
            type_num = PyArray_CFLOAT;
        }
        else if (size == sizeof(double)*2) {
            type_num = PyArray_CDOUBLE;
        }
        else if (size == sizeof(longdouble)*2) {
            type_num = PyArray_CLONGDOUBLE;
        }
        else {
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        break;
    case 'O':
        if (size == sizeof(PyObject *)) {
            type_num = PyArray_OBJECT;
        }
        else {
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }
        break;
    case PyArray_STRINGLTR:
        type_num = PyArray_STRING;
        break;
    case PyArray_UNICODELTR:
        type_num = PyArray_UNICODE;
        size <<= 2;
        break;
    case 'V':
        type_num = PyArray_VOID;
        break;
    default:
        PyErr_SetString(PyExc_ValueError, msg);
        return NULL;
    }

    descr = NpyArray_DescrFromType(type_num);
    if (descr == NULL) {
        return NULL;
    }
    swap = !NpyArray_ISNBO(swapchar);
    if (descr->elsize == 0 || swap) {
        /* Need to make a new PyArray_Descr */
        NpyArray_DESCR_REPLACE(descr);
        if (descr==NULL) {
            return NULL;
        }
        if (descr->elsize == 0) {
            descr->elsize = size;
        }
        if (swap) {
            descr->byteorder = swapchar;
        }
    }
    return descr;
}

NPY_NO_EXPORT char *
index2ptr(PyArrayObject *mp, intp i)
{
    return NpyArray_Index2Ptr(PyArray_ARRAY(mp), i);
}

NPY_NO_EXPORT int
_zerofill(PyArrayObject *ret)
{
    if (NpyDataType_REFCHK(PyArray_DESCR(ret))) {
        PyObject *zero = PyInt_FromLong(0);
        PyArray_FillObjectArray(ret, zero);
        Py_DECREF(zero);
        if (PyErr_Occurred()) {
            Py_DECREF(ret);
            return -1;
        }
    }
    else {
        intp n = PyArray_NBYTES(ret);
        memset(PyArray_BYTES(ret), 0, n);
    }
    return 0;
}

NPY_NO_EXPORT int
_IsAligned(PyArrayObject *ap)
{
    return Npy_IsAligned(PyArray_ARRAY(ap));
}

NPY_NO_EXPORT Bool
_IsWriteable(PyArrayObject *ap)
{
    return Npy_IsWriteable(PyArray_ARRAY(ap));
}
