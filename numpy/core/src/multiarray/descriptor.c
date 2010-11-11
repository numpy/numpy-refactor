/* Array Descr Object */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "npy_descriptor.h"

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "npy_api.h"
#include "descriptor.h"

#include "npy_config.h"

#include "numpy/npy_3kcompat.h"

#include "common.h"

#define _chk_byteorder(arg) (arg == '>' || arg == '<' ||        \
                             arg == '|' || arg == '=')

#if defined(NPY_PY3K)
#define TO_CSTRING(x)  PyBytes_AsString(PyUnicode_AsASCIIString(x))
#else
#define TO_CSTRING(x)  PyString_AsString(x)
#endif


static PyObject *typeDict = NULL;   /* Must be explicitly loaded */

static PyArray_Descr *
_use_inherit(PyArray_Descr *type, PyObject *newobj, int *errflag);


/** Returns the descriptor field names as a Python tuple. */
NPY_NO_EXPORT PyObject *
PyArrayDescr_GetNames(PyArray_Descr *self)
{
    PyObject *names;
    int i, n;

    if (NULL == self->descr->names) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    for (n = 0; NULL != self->descr->names[n]; n++) ;

    names = PyTuple_New(n);
    for (i = 0; i < n; i++) {
        PyTuple_SET_ITEM(names, i,
#if defined(NPY_PY3K)
                         PyUnicode_FromString(self->descr->names[i])
#else
                         PyString_FromString(self->descr->names[i])
#endif
            );
    }
    return names;
}


/** Returns the descriptor fields as a Python dictionary. */
NPY_NO_EXPORT PyObject *
PyArrayDescr_GetFields(PyArray_Descr *self)
{
    PyObject *dict = NULL;
    NpyDict_Iter pos;
    const char *key;
    NpyArray_DescrField *value;

    if (NULL == self->descr->names) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    dict = PyDict_New();
    NpyDict_IterInit(&pos);
    while (NpyDict_IterNext(self->descr->fields, &pos, (void **)&key,
                            (void **)&value)) {
        PyObject *tup = PyTuple_New( (NULL == value->title) ? 2 : 3 );
        PyArray_Descr *valueDescr = PyArray_Descr_WRAP(value->descr);

        PyTuple_SET_ITEM(tup, 0, (PyObject *)valueDescr);
        Py_INCREF(valueDescr);
        PyTuple_SET_ITEM(tup, 1, PyInt_FromLong(value->offset));
        if (NULL != value->title) {
            PyTuple_SET_ITEM(tup, 2, PyString_FromString(value->title));
        }

        PyDict_SetItemString(dict, key, tup);
        Py_DECREF(tup);
    }
    return dict;
}



/* Returns new reference */
NPY_NO_EXPORT PyArray_Descr *
_arraydescr_fromobj(PyObject *obj)
{
    PyObject *dtypedescr;
    PyArray_Descr *new;
    int ret;

    dtypedescr = PyObject_GetAttrString(obj, "dtype");
    PyErr_Clear();
    if (dtypedescr) {
        ret = PyArray_DescrConverter(dtypedescr, &new);
        Py_DECREF(dtypedescr);
        if (ret == PY_SUCCEED) {
            return new;
        }
        PyErr_Clear();
    }
    /* Understand basic ctypes */
    dtypedescr = PyObject_GetAttrString(obj, "_type_");
    PyErr_Clear();
    if (dtypedescr) {
        ret = PyArray_DescrConverter(dtypedescr, &new);
        Py_DECREF(dtypedescr);
        if (ret == PY_SUCCEED) {
            PyObject *length;
            length = PyObject_GetAttrString(obj, "_length_");
            PyErr_Clear();
            if (length) {
                /* derived type */
                PyObject *newtup;
                PyArray_Descr *derived;
                newtup = Py_BuildValue("NO", new, length);
                Py_DECREF(length);
                ret = PyArray_DescrConverter(newtup, &derived);
                Py_DECREF(newtup);
                if (ret == PY_SUCCEED) {
                    return derived;
                }
                PyErr_Clear();
                return NULL;
            }
            return new;
        }
        PyErr_Clear();
        return NULL;
    }
    /* Understand ctypes structures --
       bit-fields are not supported
       automatically aligns */
    dtypedescr = PyObject_GetAttrString(obj, "_fields_");
    PyErr_Clear();
    if (dtypedescr) {
        ret = PyArray_DescrAlignConverter(dtypedescr, &new);
        Py_DECREF(dtypedescr);
        if (ret == PY_SUCCEED) {
            return new;
        }
        PyErr_Clear();
    }
    return NULL;
}

NPY_NO_EXPORT PyObject *
array_set_typeDict(PyObject *NPY_UNUSED(ignored), PyObject *args)
{
    PyObject *dict;

    if (!PyArg_ParseTuple(args, "O", &dict)) {
        return NULL;
    }
    /* Decrement old reference (if any)*/
    Py_XDECREF(typeDict);
    typeDict = dict;

    Py_INCREF(Py_None);
    return Py_None;
}

static int
_check_for_commastring(char *type, int len)
{
    int i;

    /* Check for ints at start of string */
    if ((type[0] >= '0'
                && type[0] <= '9')
            || ((len > 1)
                && _chk_byteorder(type[0])
                && (type[1] >= '0'
                && type[1] <= '9'))) {
        return 1;
    }
    /* Check for empty tuple */
    if (((len > 1)
                && (type[0] == '('
                && type[1] == ')'))
            || ((len > 3)
                && _chk_byteorder(type[0])
                && (type[1] == '('
                && type[2] == ')'))) {
        return 1;
    }
    /* Check for presence of commas */
    for (i = 1; i < len; i++) {
        if (type[i] == ',') {
            return 1;
        }
    }
    return 0;
}

static int
_check_for_datetime(char *type, int len)
{
    if (len < 1) {
        return 0;
    }
    if (type[1] == '8' && (type[0] == 'M' || type[0] == 'm')) {
        return 1;
    }
    if (len < 10) {
        return 0;
    }
    if (strncmp(type, "datetime64", 10) == 0) {
        return 1;
    }
    if (len < 11) {
        return 0;
    }
    if (strncmp(type, "timedelta64", 11) == 0) {
        return 1;
    }
    return 0;
}



#undef _chk_byteorder

static PyArray_Descr *
_convert_from_tuple(PyObject *obj)
{
    PyArray_Descr *type, *res;
    PyObject *val;
    int errflag;

    if (PyTuple_GET_SIZE(obj) != 2) {
        return NULL;
    }
    if (!PyArray_DescrConverter(PyTuple_GET_ITEM(obj,0), &type)) {
        return NULL;
    }
    val = PyTuple_GET_ITEM(obj,1);
    /* try to interpret next item as a type */
    res = _use_inherit(type, val, &errflag);
    if (res || errflag) {
        Py_DECREF(type);
        if (res) {
            return res;
        }
        else {
            return NULL;
        }
    }
    PyErr_Clear();
    /*
     * We get here if res was NULL but errflag wasn't set
     * --- i.e. the conversion to a data-descr failed in _use_inherit
     */
    if (type->descr->elsize == 0) {
        /* interpret next item as a typesize */
        int itemsize = PyArray_PyIntAsInt(PyTuple_GET_ITEM(obj,1));

        if (error_converting(itemsize)) {
            PyErr_SetString(PyExc_ValueError,
                    "invalid itemsize in generic type tuple");
            goto fail;
        }
        PyArray_DESCR_REPLACE(type);
        if (type->descr->type_num == PyArray_UNICODE) {
            type->descr->elsize = itemsize << 2;
        }
        else {
            type->descr->elsize = itemsize;
        }
    }
    else if (PyDict_Check(val)) {
        /* Assume it's a metadata dictionary */
        /* FIXME: metadata removed from structure */
    }
    else {
        /*
         * interpret next item as shape (if it's a tuple)
         * and reset the type to PyArray_VOID with
         * a new fields attribute.
         */
        PyArray_Dims shape = {NULL, -1};
        NpyArray_Descr *newdescr;

        if (!(PyArray_IntpConverter(val, &shape)) || (shape.len > MAX_DIMS)) {
            PyDimMem_FREE(shape.ptr);
            PyErr_SetString(PyExc_ValueError,
                    "invalid shape in fixed-type tuple.");
            goto fail;
        }
        /*
         * If (type, 1) was given, it is equivalent to type...
         * or (type, ()) was given it is equivalent to type...
         */
        if ((shape.len == 1
                    && shape.ptr[0] == 1
                    && PyNumber_Check(val))
                || (shape.len == 0
                    && PyTuple_Check(val))) {
            PyDimMem_FREE(shape.ptr);
            return type;
        }
        newdescr = NpyArray_DescrNewFromType(PyArray_VOID);
        if (newdescr == NULL) {
            PyDimMem_FREE(shape.ptr);
            goto fail;
        }
        /* TODO: Review to see if this makes sense to push into the core.
           Can it be reused? */
        newdescr->elsize = type->descr->elsize;
        newdescr->elsize *= PyArray_MultiplyList(shape.ptr, shape.len);
        newdescr->subarray = NpyArray_malloc(sizeof(NpyArray_ArrayDescr));
        newdescr->subarray->base = type->descr;
        newdescr->subarray->shape_num_dims = shape.len;
        newdescr->subarray->shape_dims = shape.ptr;
        shape.ptr = NULL; /* Stole shape.ptr, do not free. */
        newdescr->flags = type->descr->flags;
        NpyArray_DescrDeallocNamesAndFields(newdescr);

        /* Move reference on type to core since that's what's stored in
           subarray->base */
        Npy_INCREF(type->descr);
        Py_DECREF(type);

        assert((0 == newdescr->subarray->shape_num_dims &&
                NULL == newdescr->subarray->shape_dims) ||
               (0 < newdescr->subarray->shape_num_dims &&
                NULL != newdescr->subarray->shape_dims));

        /* Move reference to newdescr to interface object for return. */
        type = PyArray_Descr_WRAP(newdescr);
        Py_INCREF(type);
        Npy_DECREF(newdescr);
    }
    return type;

 fail:
    Py_XDECREF(type);
    return NULL;
}

/*
 * obj is a list.  Each item is a tuple with
 *
 * (field-name, data-type (either a list or a string), and an optional
 * shape parameter).
 *
 * field-name can be a string or a 2-tuple data-type can now be a list,
 * string, or 2-tuple (string, metadata dictionary))
 */

static PyArray_Descr *
_convert_from_array_descr(PyObject *obj, int align)
{
    int n, i, totalsize;
    int ret;
    PyObject *item, *newobj;
    PyObject *name, *title;
    NpyDict *fields = NULL;
    char **nameslist = NULL;
    int offset = 0;
    NpyArray_Descr *new;
    PyArray_Descr *conv;
    int dtypeflags = 0;
    int maxalign = 0;

    n = PyList_GET_SIZE(obj);
    totalsize = 0;

    nameslist = NpyArray_DescrAllocNames(n);
    if (NULL == nameslist) {
        return NULL;
    }
    fields = NpyArray_DescrAllocFields();
    if (NULL == fields) {
        free(nameslist);
        return NULL;
    }

    for (i = 0; i < n; i++) {
        item = PyList_GET_ITEM(obj, i);
        if (!PyTuple_Check(item) || (PyTuple_GET_SIZE(item) < 2)) {
            goto fail;
        }
        name = PyTuple_GET_ITEM(item, 0);
        if (PyUString_Check(name)) {
            title = NULL;
        }
        else if (PyTuple_Check(name)) {
            if (PyTuple_GET_SIZE(name) != 2) {
                goto fail;
            }
            title = PyTuple_GET_ITEM(name, 0);
            name = PyTuple_GET_ITEM(name, 1);
            if (!PyUString_Check(name)) {
                goto fail;
            }
        }
        else {
            goto fail;
        }

        /* Insert name into nameslist */
        Py_INCREF(name);

        if (PyUString_GET_SIZE(name) == 0) {
            Py_DECREF(name);
            if (title == NULL) {
                name = PyUString_FromFormat("f%d", i);
            }
#if defined(NPY_PY3K)
            /* On Py3, allow only non-empty Unicode strings as field names */
            else if (PyUString_Check(title) && PyUString_GET_SIZE(title) > 0) {
                name = title;
                Py_INCREF(name);
            }
            else {
                goto fail;
            }
#else
            else {
                name = title;
                Py_INCREF(name);
            }
#endif
        }
        nameslist[i] = strdup(TO_CSTRING(name));
        Py_DECREF(name);

        /* Process rest */

        if (PyTuple_GET_SIZE(item) == 2) {
            ret = PyArray_DescrConverter(PyTuple_GET_ITEM(item, 1), &conv);
            if (ret == PY_FAIL) {
                PyObject_Print(PyTuple_GET_ITEM(item, 1), stderr, 0);
            }
        }
        else if (PyTuple_GET_SIZE(item) == 3) {
            newobj = PyTuple_GetSlice(item, 1, 3);
            ret = PyArray_DescrConverter(newobj, &conv);
            Py_DECREF(newobj);
        }
        else {
            goto fail;
        }
        if (ret == PY_FAIL) {
            goto fail;
        }

        if (NpyDict_ContainsKey(fields, nameslist[i])
            || (title
#if defined(NPY_PY3K)
                 && PyUString_Check(title)
#else
                 && (PyUString_Check(title) || PyUnicode_Check(title))
#endif
                 && NpyDict_ContainsKey(fields,
                                        (void *)TO_CSTRING(title)))) {
            PyErr_SetString(PyExc_ValueError, "two fields with the same name");
            Py_DECREF(conv);
            goto fail;
        }
        dtypeflags |= (conv->descr->flags & NPY_FROM_FIELDS);


        if (align) {
            int _align;

            _align = conv->descr->alignment;
            if (_align > 1) {
                totalsize = ((totalsize + _align - 1)/_align)*_align;
            }
            maxalign = MAX(maxalign, _align);
        }
        offset = totalsize;

        /*
         * Title can be "meta-data".  Only insert it
         * into the fields dictionary if it is a string
         * and if it is not the same as the name.
         */
        if (title != NULL) {
#if defined(NPY_PY3K)
            if (PyUString_Check(title))
#else
            if (PyUString_Check(title) || PyUnicode_Check(title))
#endif
            {
                NpyArray_Descr *descr;
                char *titleStr = TO_CSTRING(title);
                if (!strcmp(nameslist[i], titleStr) ||
                        NpyDict_ContainsKey(fields, titleStr)) {
                    PyErr_SetString(PyExc_ValueError,
                            "title already used as a name or title.");
                    Py_DECREF(conv);
                    goto fail;
                }
                PyArray_Descr_REF_TO_CORE(conv, descr);
                NpyArray_DescrSetField(fields, nameslist[i], descr, offset,
                                       titleStr);
                /* First DescrSetField call steals the reference,
                   need a second to steal. */
                Npy_INCREF(descr);
                NpyArray_DescrSetField(fields, titleStr, descr, offset,
                                       titleStr);
            }
        } else {
            NpyArray_Descr *descr;

            PyArray_Descr_REF_TO_CORE(conv, descr);
            NpyArray_DescrSetField(fields, nameslist[i], descr, offset, NULL);
        }
        totalsize += conv->descr->elsize;
    }
    new = NpyArray_DescrNewFromType(PyArray_VOID);
    new->fields = fields;
    new->names = nameslist;
    new->elsize = totalsize;
    new->flags=dtypeflags;
    if (maxalign > 1) {
        totalsize = ((totalsize + maxalign - 1)/maxalign)*maxalign;
    }
    if (align) {
        new->alignment = maxalign;
    }
    Py_INCREF( Npy_INTERFACE(new) );
    Npy_DECREF(new);
    return PyArray_Descr_WRAP(new);

 fail:
    if (NULL != nameslist) {
        for (i=0; i < n; i++) {
            if (NULL != nameslist[i]) {
                free(nameslist[i]);
            }
        }
        free(nameslist);
    }

    if (NULL != fields) {
        NpyDict_Destroy(fields);
    }
    return NULL;
}



/*
 * a list specifying a data-type can just be
 * a list of formats.  The names for the fields
 * will default to f0, f1, f2, and so forth.
 */
static PyArray_Descr *
_convert_from_list(PyObject *obj, int align)
{
    int n, i;
    int totalsize;
    PyArray_Descr *conv = NULL;
    NpyArray_Descr *new;
    PyObject *key;
    NpyDict *fields = NULL;
    char **nameslist = NULL;
    int ret;
    int maxalign = 0;
    int dtypeflags = 0;

    n = PyList_GET_SIZE(obj);
    /*
     * Ignore any empty string at end which _internal._commastring
     * can produce
     */
    key = PyList_GET_ITEM(obj, n-1);
    if (PyBytes_Check(key) && PyBytes_GET_SIZE(key) == 0) {
        n = n - 1;
    }
    /* End ignore code.*/
    totalsize = 0;
    if (n == 0) {
        return NULL;
    }
    nameslist = NpyArray_DescrAllocNames(n);
    if (NULL == nameslist) {
        return NULL;
    }
    fields = NpyArray_DescrAllocFields();
    if (NULL == fields) {
        free(nameslist);
        return NULL;
    }

    for (i = 0; i < n; i++) {
        NpyArray_Descr *descr;

        key = PyUString_FromFormat("f%d", i);
        ret = PyArray_DescrConverter(PyList_GET_ITEM(obj, i), &conv);
        if (ret == PY_FAIL) {
            Py_DECREF(key);
            goto fail;
        }
        dtypeflags |= (conv->descr->flags & NPY_FROM_FIELDS);
        if (align) {
            int _align;

            _align = conv->descr->alignment;
            if (_align > 1) {
                totalsize = ((totalsize + _align - 1)/_align)*_align;
            }
            maxalign = MAX(maxalign, _align);
        }
        PyArray_Descr_REF_TO_CORE(conv, descr);
        NpyArray_DescrSetField(fields, TO_CSTRING(key), descr,
                               totalsize, NULL);
        nameslist[i] = strdup(TO_CSTRING(key));
        totalsize += conv->descr->elsize;
        Py_DECREF(key);
    }
    new = NpyArray_DescrNewFromType(PyArray_VOID);
    new->fields = fields;
    new->names = nameslist;
    new->flags=dtypeflags;
    if (maxalign > 1) {
        totalsize = ((totalsize+maxalign-1)/maxalign)*maxalign;
    }
    if (align) {
        new->alignment = maxalign;
    }
    new->elsize = totalsize;

    Py_INCREF( Npy_INTERFACE(new) );
    Npy_DECREF( new );
    return PyArray_Descr_WRAP(new);

 fail:
    if (NULL != nameslist) {
        for (i=0; i < n; i++) {
            if (NULL != nameslist[i]) {
                free(nameslist[i]);
            }
        }
        free(nameslist);
    }

    if (NULL != fields) {
        NpyDict_Destroy(fields);
    }
    return NULL;
}


static PyObject *
_get_datetime_tuple_from_datetimeinfo(NpyArray_DateTimeInfo *dt_data)
{
    PyObject *dt_tuple;

    dt_tuple = PyTuple_New(4);

    PyTuple_SET_ITEM(dt_tuple, 0,
            PyBytes_FromString(_datetime_strings[dt_data->base]));
    PyTuple_SET_ITEM(dt_tuple, 1,
            PyInt_FromLong(dt_data->num));
    PyTuple_SET_ITEM(dt_tuple, 2,
            PyInt_FromLong(dt_data->den));
    PyTuple_SET_ITEM(dt_tuple, 3,
            PyInt_FromLong(dt_data->events));

    return dt_tuple;
}



static NpyArray_DateTimeInfo *
_convert_datetime_tuple_to_datetimeinfo(PyObject *tuple)
{
    return NpyArray_DateTimeInfoNew(
        PyBytes_AsString(PyTuple_GET_ITEM(tuple, 0)),
        PyInt_AS_LONG(PyTuple_GET_ITEM(tuple, 1)),
        PyInt_AS_LONG(PyTuple_GET_ITEM(tuple, 2)),
        PyInt_AS_LONG(PyTuple_GET_ITEM(tuple, 3)));
}

static PyArray_Descr *
_convert_from_datetime_tuple(PyObject *obj)
{
    PyArray_Descr *new;
    PyObject *dt_tuple;
    NpyArray_DateTimeInfo *dtinfo;
    PyObject *datetime;

    if (!PyTuple_Check(obj) || PyTuple_GET_SIZE(obj)!=2) {
        PyErr_SetString(PyExc_RuntimeError,
                "_datetimestring is not returning a tuple with length 2");
        return NULL;
    }

    dt_tuple = PyTuple_GET_ITEM(obj, 0);
    datetime = PyTuple_GET_ITEM(obj, 1);
    if (!PyTuple_Check(dt_tuple)
          || PyTuple_GET_SIZE(dt_tuple) != 4
          || !PyInt_Check(datetime)) {
      PyErr_SetString(PyExc_RuntimeError,
                      "_datetimestring is not returning a length 4 tuple"
                      " and an integer");
      return NULL;
    }

    /* Create new timedelta or datetime dtype */
    if (PyObject_IsTrue(datetime)) {
        new = PyArray_DescrNewFromType(PyArray_DATETIME);
    }
    else {
        new = PyArray_DescrNewFromType(PyArray_TIMEDELTA);
    }

    if (new == NULL) {
        return NULL;
    }

    dtinfo = _convert_datetime_tuple_to_datetimeinfo(dt_tuple);

    if (dtinfo == NULL) {
        /* Failure in conversion */
        Py_DECREF(new);
        return NULL;
    }

    new->descr->dtinfo = dtinfo;

    return new;
}


static PyArray_Descr *
_convert_from_datetime(PyObject *obj)
{
    PyObject *tupleobj;
    PyArray_Descr *res;
    PyObject *_numpy_internal;

    if (!PyBytes_Check(obj)) {
        return NULL;
    }
    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
    if (_numpy_internal == NULL) {
        return NULL;
    }
    tupleobj = PyObject_CallMethod(_numpy_internal,
            "_datetimestring", "O", obj);
    Py_DECREF(_numpy_internal);
    if (!tupleobj) {
        return NULL;
    }
    /*
     * tuple of a standard tuple (baseunit, num, den, events) and a timedelta
     * boolean
     */
    res = _convert_from_datetime_tuple(tupleobj);
    Py_DECREF(tupleobj);
    if (!res && !PyErr_Occurred()) {
        PyErr_SetString(PyExc_ValueError, "invalid data-type");
        return NULL;
    }
    return res;
}


/*
 * comma-separated string
 * this is the format developed by the numarray records module and implemented
 * by the format parser in that module this is an alternative implementation
 * found in the _internal.py file patterned after that one -- the approach is
 * to try to convert to a list (with tuples if any repeat information is
 * present) and then call the _convert_from_list)
 */
static PyArray_Descr *
_convert_from_commastring(PyObject *obj, int align)
{
    PyObject *listobj;
    PyArray_Descr *res;
    PyObject *_numpy_internal;

    if (!PyBytes_Check(obj)) {
        return NULL;
    }
    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
    if (_numpy_internal == NULL) {
        return NULL;
    }
    listobj = PyObject_CallMethod(_numpy_internal, "_commastring", "O", obj);
    Py_DECREF(_numpy_internal);
    if (!listobj) {
        return NULL;
    }
    if (!PyList_Check(listobj) || PyList_GET_SIZE(listobj) < 1) {
        PyErr_SetString(PyExc_RuntimeError,
                "_commastring is not returning a list with len >= 1");
        Py_DECREF(listobj);
        return NULL;
    }
    if (PyList_GET_SIZE(listobj) == 1) {
        if (PyArray_DescrConverter(
                    PyList_GET_ITEM(listobj, 0), &res) == NPY_FAIL) {
            res = NULL;
        }
    }
    else {
        res = _convert_from_list(listobj, align);
    }
    Py_DECREF(listobj);
    if (!res && !PyErr_Occurred()) {
        PyErr_SetString(PyExc_ValueError, "invalid data-type");
        return NULL;
    }
    return res;
}

static int
_is_tuple_of_integers(PyObject *obj)
{
    int i;

    if (!PyTuple_Check(obj)) {
        return 0;
    }
    for (i = 0; i < PyTuple_GET_SIZE(obj); i++) {
        if (!PyArray_IsIntegerScalar(PyTuple_GET_ITEM(obj, i))) {
            return 0;
        }
    }
    return 1;
}

/*
 * A tuple type would be either (generic typeobject, typesize)
 * or (fixed-length data-type, shape)
 *
 * or (inheriting data-type, new-data-type)
 * The new data-type must have the same itemsize as the inheriting data-type
 * unless the latter is 0
 *
 * Thus (int32, {'real':(int16,0),'imag',(int16,2)})
 *
 * is one way to specify a descriptor that will give
 * a['real'] and a['imag'] to an int32 array.
 *
 * leave type reference alone
 */
static PyArray_Descr *
_use_inherit(PyArray_Descr *type, PyObject *newobj, int *errflag)
{
    PyArray_Descr *new;
    PyArray_Descr *conv;

    *errflag = 0;
    if (PyArray_IsScalar(newobj, Integer)
            || _is_tuple_of_integers(newobj)
            || !PyArray_DescrConverter(newobj, &conv)) {
        return NULL;
    }
    *errflag = 1;
    new = PyArray_DescrNew(type);
    if (new == NULL) {
        goto fail;
    }
    if (new->descr->elsize && new->descr->elsize != conv->descr->elsize) {
        PyErr_SetString(PyExc_ValueError,
                        "mismatch in size of old and new data-descriptor");
        Py_DECREF(new);
        goto fail;
    }
    new->descr->elsize = conv->descr->elsize;
    if (NULL != conv->descr->names) {
        new->descr->names = NpyArray_DescrNamesCopy(conv->descr->names);
        new->descr->fields = NpyArray_DescrFieldsCopy(conv->descr->fields);
    }
    new->descr->flags = conv->descr->flags;
    Py_DECREF(conv);
    *errflag = 0;
    return new;

 fail:
    Py_DECREF(conv);
    return NULL;
}



/*
 * a dictionary specifying a data-type
 * must have at least two and up to four
 * keys These must all be sequences of the same length.
 *
 * can also have an additional key called "metadata" which can be any dictionary
 *
 * "names" --- field names
 * "formats" --- the data-type descriptors for the field.
 *
 * Optional:
 *
 * "offsets" --- integers indicating the offset into the
 * record of the start of the field.
 * if not given, then "consecutive offsets"
 * will be assumed and placed in the dictionary.
 *
 * "titles" --- Allows the use of an additional key
 * for the fields dictionary.(if these are strings
 * or unicode objects) or
 * this can also be meta-data to
 * be passed around with the field description.
 *
 * Attribute-lookup-based field names merely has to query the fields
 * dictionary of the data-descriptor.  Any result present can be used
 * to return the correct field.
 *
 * So, the notion of what is a name and what is a title is really quite
 * arbitrary.
 *
 * What does distinguish a title, however, is that if it is not None,
 * it will be placed at the end of the tuple inserted into the
 * fields dictionary.and can therefore be used to carry meta-data around.
 *
 * If the dictionary does not have "names" and "formats" entries,
 * then it will be checked for conformity and used directly.
 */
static PyArray_Descr *
_use_fields_dict(PyObject *obj, int align)
{
    PyObject *_numpy_internal;
    PyArray_Descr *res;

    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
    if (_numpy_internal == NULL) {
        return NULL;
    }
    res = (PyArray_Descr *)PyObject_CallMethod(_numpy_internal,
            "_usefields", "Oi", obj, align);
    Py_DECREF(_numpy_internal);
    return res;
}

static PyArray_Descr *
_convert_from_dict(PyObject *obj, int align)
{
    NpyArray_Descr *new;
    NpyDict *fields = NULL;
    char **nameslist = NULL;
    PyObject *names, *offsets, *descrs, *titles;
    int n, i;
    int totalsize;
    int maxalign = 0;
    int dtypeflags = 0;

    names = PyDict_GetItemString(obj, "names");
    descrs = PyDict_GetItemString(obj, "formats");
    if (!names || !descrs) {
        return _use_fields_dict(obj, align);
    }
    n = PyObject_Length(names);

    fields = NpyArray_DescrAllocFields();
    if (fields == NULL) {
        return (PyArray_Descr *)PyErr_NoMemory();
    }
    nameslist = NpyArray_DescrAllocNames(n);
    if (NULL == nameslist) {
        NpyDict_Destroy(fields);
        return NULL;
    }

    offsets = PyDict_GetItemString(obj, "offsets");
    titles = PyDict_GetItemString(obj, "titles");
    if ((n > PyObject_Length(descrs))
        || (offsets && (n > PyObject_Length(offsets)))
        || (titles && (n > PyObject_Length(titles)))) {
        PyErr_SetString(PyExc_ValueError,
                "all items in the dictionary must have the same length.");
        goto fail;
    }

    totalsize = 0;
    for (i = 0; i < n; i++) {
        PyObject *descr, *index, *item, *name, *off;
        long offset = 0;
        int len, ret, _align = 1;
        PyArray_Descr *newdescr = NULL;
        NpyArray_Descr *coredescr;

        /* Build item to insert (descr, offset, [title])*/
        len = 2;
        item = NULL;
        index = PyInt_FromLong(i);
        if (titles) {
            item=PyObject_GetItem(titles, index);
            if (item && item != Py_None) {
                len = 3;
            }
            else {
                Py_XDECREF(item);
            }
            PyErr_Clear();
        }

        descr = PyObject_GetItem(descrs, index);
        if (descr == NULL) {
            Py_DECREF(index);
            Py_XDECREF(newdescr);
            goto fail;
        }
        ret = PyArray_DescrConverter(descr, &newdescr);
        Py_DECREF(descr);
        if (ret == PY_FAIL) {
            Py_DECREF(index);
            Py_XDECREF(newdescr);
            goto fail;
        }
        if (align) {
            _align = newdescr->descr->alignment;
            maxalign = MAX(maxalign,_align);
        }
        if (offsets) {
            off = PyObject_GetItem(offsets, index);
            if (off == NULL) {
                Py_DECREF(index);
                Py_XDECREF(newdescr);
                goto fail;
            }
            offset = PyInt_AsLong(off);
            Py_DECREF(off);
            if (offset < totalsize) {
                PyErr_SetString(PyExc_ValueError,
                                "invalid offset (must be ordered)");
                ret = PY_FAIL;
            }
            if (offset > totalsize) {
                totalsize = offset;
            }
        }
        else {
            if (align && _align > 1) {
                totalsize = ((totalsize + _align - 1)/_align)*_align;
            }
            offset = totalsize;
        }

        name = PyObject_GetItem(names, index);
        if (name == NULL) {
            Py_DECREF(index);
            Py_XDECREF(newdescr);
            goto fail;
        }
        Py_DECREF(index);
#if defined(NPY_PY3K)
        if (!PyUString_Check(name))
#else
        if (!(PyUString_Check(name) || PyUnicode_Check(name)))
#endif
        {
            PyErr_SetString(PyExc_ValueError, "field names must be strings");
            ret = PY_FAIL;
        }
        else {
            nameslist[i] = strdup(TO_CSTRING(name));
        }

        /* Insert into dictionary */
        if (NpyDict_ContainsKey(fields, TO_CSTRING(name))) {
            PyErr_SetString(PyExc_ValueError,
                    "name already used as a name or title");
            ret = PY_FAIL;
        }
        PyArray_Descr_REF_TO_CORE(newdescr, coredescr);
        NpyArray_DescrSetField(fields, TO_CSTRING(name),
                               coredescr, offset,
                               (3 == len) ? TO_CSTRING(item) : NULL);
        Py_DECREF(name);
        if (len == 3) {
#if defined(NPY_PY3K)
            if (PyUString_Check(item))
#else
            if (PyUString_Check(item) || PyUnicode_Check(item))
#endif
            {
                if (NpyDict_ContainsKey(fields, TO_CSTRING(item))) {
                    PyErr_SetString(PyExc_ValueError,
                            "title already used as a name or title.");
                    ret=PY_FAIL;
                }
                /* First DescrSetField stole the ref, need a second */
                Npy_INCREF(coredescr);
                NpyArray_DescrSetField(fields, TO_CSTRING(item),
                                       coredescr, offset,
                                       TO_CSTRING(item));
            }
            Py_DECREF(item);
        }
        if ((ret == PY_FAIL) || (coredescr->elsize == 0)) {
            goto fail;
        }
        dtypeflags |= (coredescr->flags & NPY_FROM_FIELDS);
        totalsize += coredescr->elsize;
    }

    new = NpyArray_DescrNewFromType(PyArray_VOID);
    if (new == NULL) {
        goto fail;
    }
    if (maxalign > 1) {
        totalsize = ((totalsize + maxalign - 1)/maxalign) * maxalign;
    }
    if (align) {
        new->alignment = maxalign;
    }
    new->elsize = totalsize;
    new->names = nameslist;
    new->fields = fields;
    new->flags = dtypeflags;

    Py_INCREF( Npy_INTERFACE(new) );
    Npy_DECREF(new);
    return Npy_INTERFACE(new);

 fail:
    if (NULL != nameslist) {
        for (i=0; i < n; i++) {
            if (NULL != nameslist[i]) {
                free(nameslist[i]);
            }
        }
        free(nameslist);
    }

    if (NULL != fields) {
        NpyDict_Destroy(fields);
    }
    return NULL;
}


/*NUMPY_API*/
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrNewFromType(int type_num)
{
    NpyArray_Descr *result = NpyArray_DescrNewFromType(type_num);

    Py_INCREF( Npy_INTERFACE(result) );
    Npy_DECREF(result);
    return Npy_INTERFACE(result);
}


/*NUMPY_API
 * Get typenum from an object -- None goes to NULL
 */
NPY_NO_EXPORT int
PyArray_DescrConverter2(PyObject *obj, PyArray_Descr **at)
{
    if (obj == Py_None) {
        *at = NULL;
        return PY_SUCCEED;
    }
    else {
        return PyArray_DescrConverter(obj, at);
    }
}


/*NUMPY_API
 * Get typenum from an object -- None goes to PyArray_DEFAULT
 * This function takes a Python object representing a type and converts it
 * to a the correct PyArray_Descr * structure to describe the type.
 *
 * Many objects can be used to represent a data-type which in NumPy is
 * quite a flexible concept.
 *
 * This is the central code that converts Python objects to
 * Type-descriptor objects that are used throughout numpy.
 * new reference in *at
 */
NPY_NO_EXPORT int
PyArray_DescrConverter(PyObject *obj, PyArray_Descr **at)
{
    char *type;
    int check_num = PyArray_NOTYPE + 10;
    int len;
    PyObject *item;
    int elsize = 0;
    char endian = '=';

    *at = NULL;
    /* default */
    if (obj == Py_None) {
        *at = PyArray_DescrFromType(PyArray_DEFAULT);
        return PY_SUCCEED;
    }
    if (PyArray_DescrCheck(obj)) {
        *at = (PyArray_Descr *)obj;
        Py_INCREF(*at);
        return PY_SUCCEED;
    }

    if (PyType_Check(obj)) {
        if (PyType_IsSubtype((PyTypeObject *)obj, &PyGenericArrType_Type)) {
            *at = PyArray_DescrFromTypeObject(obj);
            if (*at) {
                return PY_SUCCEED;
            }
            else {
                return PY_FAIL;
            }
        }
        check_num = PyArray_OBJECT;
#if !defined(NPY_PY3K)
        if (obj == (PyObject *)(&PyInt_Type)) {
            check_num = PyArray_LONG;
        }
        else if (obj == (PyObject *)(&PyLong_Type)) {
            check_num = PyArray_LONGLONG;
        }
#else
        if (obj == (PyObject *)(&PyLong_Type)) {
            check_num = PyArray_LONG;
        }
#endif
        else if (obj == (PyObject *)(&PyFloat_Type)) {
            check_num = PyArray_DOUBLE;
        }
        else if (obj == (PyObject *)(&PyComplex_Type)) {
            check_num = PyArray_CDOUBLE;
        }
        else if (obj == (PyObject *)(&PyBool_Type)) {
            check_num = PyArray_BOOL;
        }
        else if (obj == (PyObject *)(&PyBytes_Type)) {
            check_num = PyArray_STRING;
        }
        else if (obj == (PyObject *)(&PyUnicode_Type)) {
            check_num = PyArray_UNICODE;
        }
#if defined(NPY_PY3K)
        else if (obj == (PyObject *)(&PyMemoryView_Type)) {
            check_num = PyArray_VOID;
        }
#else
        else if (obj == (PyObject *)(&PyBuffer_Type)) {
            check_num = PyArray_VOID;
        }
#endif
        else {
            *at = _arraydescr_fromobj(obj);
            if (*at) {
                return PY_SUCCEED;
            }
        }
        goto finish;
    }

    /* or a typecode string */

    if (PyUnicode_Check(obj)) {
        /* Allow unicode format strings: convert to bytes */
        int retval;
        PyObject *obj2;
        obj2 = PyUnicode_AsASCIIString(obj);
        if (obj2 == NULL) {
            goto fail;
        }
        retval = PyArray_DescrConverter(obj2, at);
        Py_DECREF(obj2);
        return retval;
    }

    if (PyBytes_Check(obj)) {
        /* Check for a string typecode. */
        type = PyBytes_AS_STRING(obj);
        len = PyBytes_GET_SIZE(obj);
        if (len <= 0) {
            goto fail;
        }
        /* check for datetime format */
        if ((len > 1) && _check_for_datetime(type, len)) {
            *at = _convert_from_datetime(obj);
            if (*at) {
                return PY_SUCCEED;
            }
            return PY_FAIL;
        }
        /* check for commas present or first (or second) element a digit */
        if (_check_for_commastring(type, len)) {
            *at = _convert_from_commastring(obj, 0);
            if (*at) {
                return PY_SUCCEED;
            }
            return PY_FAIL;
        }
        check_num = (int) type[0];
        if ((char) check_num == '>'
                || (char) check_num == '<'
                || (char) check_num == '|'
                || (char) check_num == '=') {
            if (len <= 1) {
                goto fail;
            }
            endian = (char) check_num;
            type++; len--;
            check_num = (int) type[0];
            if (endian == '|') {
                endian = '=';
            }
        }
        if (len > 1) {
            elsize = atoi(type + 1);
            if (elsize == 0) {
                check_num = PyArray_NOTYPE+10;
            }
            /*
             * When specifying length of UNICODE
             * the number of characters is given to match
             * the STRING interface.  Each character can be
             * more than one byte and itemsize must be
             * the number of bytes.
             */
            else if (check_num == PyArray_UNICODELTR) {
                elsize <<= 2;
            }
            /* Support for generic processing c4, i4, f8, etc...*/
            else if ((check_num != PyArray_STRINGLTR)
                     && (check_num != PyArray_VOIDLTR)
                     && (check_num != PyArray_STRINGLTR2)) {
                check_num = PyArray_TypestrConvert(elsize, check_num);
                if (check_num == PyArray_NOTYPE) {
                    check_num += 10;
                }
                elsize = 0;
            }
        }
    }
    else if (PyTuple_Check(obj)) {
        /* or a tuple */
        *at = _convert_from_tuple(obj);
        if (*at == NULL){
            if (PyErr_Occurred()) {
                return PY_FAIL;
            }
            goto fail;
        }
        return PY_SUCCEED;
    }
    else if (PyList_Check(obj)) {
        /* or a list */
        *at = _convert_from_array_descr(obj,0);
        if (*at == NULL) {
            if (PyErr_Occurred()) {
                return PY_FAIL;
            }
            goto fail;
        }
        return PY_SUCCEED;
    }
    else if (PyDict_Check(obj)) {
        /* or a dictionary */
        *at = _convert_from_dict(obj,0);
        if (*at == NULL) {
            if (PyErr_Occurred()) {
                return PY_FAIL;
            }
            goto fail;
        }
        return PY_SUCCEED;
    }
    else if (PyArray_Check(obj)) {
        goto fail;
    }
    else {
        *at = _arraydescr_fromobj(obj);
        if (*at) {
            return PY_SUCCEED;
        }
        if (PyErr_Occurred()) {
            return PY_FAIL;
        }
        goto fail;
    }
    if (PyErr_Occurred()) {
        goto fail;
    }
    /* if (check_num == PyArray_NOTYPE) {
           return PY_FAIL;
       }
    */

 finish:
    if ((check_num == PyArray_NOTYPE + 10)
        || (*at = PyArray_DescrFromType(check_num)) == NULL) {
        PyErr_Clear();
        /* Now check to see if the object is registered in typeDict */
        if (typeDict != NULL) {
            item = PyDict_GetItem(typeDict, obj);
#if defined(NPY_PY3K)
            if (!item && PyBytes_Check(obj)) {
                PyObject *tmp;
                tmp = PyUnicode_FromEncodedObject(obj, "ascii", "strict");
                if (tmp != NULL) {
                    item = PyDict_GetItem(typeDict, tmp);
                    Py_DECREF(tmp);
                }
            }
#endif
            if (item) {
                return PyArray_DescrConverter(item, at);
            }
        }
        goto fail;
    }

    if (((*at)->descr->elsize == 0) && (elsize != 0)) {
        PyArray_DESCR_REPLACE(*at);
        (*at)->descr->elsize = elsize;
    }
    if (endian != '=' && NpyArray_ISNBO(endian)) {
        endian = '=';
    }
    if (endian != '=' && (*at)->descr->byteorder != '|'
        && (*at)->descr->byteorder != endian) {
        PyArray_DESCR_REPLACE(*at);
        (*at)->descr->byteorder = endian;
    }
    return PY_SUCCEED;

 fail:
    PyErr_SetString(PyExc_TypeError, "data type not understood");
    Py_XDECREF(*at);
    *at = NULL;
    return PY_FAIL;
}





/** Array Descr Objects for dynamic types **/

/*
 * There are some statically-defined PyArray_Descr objects corresponding
 * to the basic built-in types.
 * These can and should be DECREF'd and INCREF'd as appropriate, anyway.
 * If a mistake is made in reference counting, deallocation on these
 * builtins will be attempted leading to problems.
 *
 * This let's us deal with all PyArray_Descr objects using reference
 * counting (regardless of whether they are statically or dynamically
 * allocated).
 */

/*NUMPY_API
 * base cannot be NULL
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrNew(PyArray_Descr *base)
{
    NpyArray_Descr *result = NpyArray_DescrNew(base->descr);

    Py_INCREF( Npy_INTERFACE(result) );
    Npy_DECREF(result);
    return Npy_INTERFACE(result);
}

/*
 * should never be called for builtin-types unless
 * there is a reference-count problem
 */
static void
arraydescr_dealloc(PyArray_Descr *self)
{
    assert(self->descr->nob_refcnt == 0);
    NpyArray_DescrDestroy(self->descr);
    Py_XDECREF(self->typeobj);
    Py_TYPE(self)->tp_free((PyObject *)self);
}


static PyObject *
arraydescr_subdescr_get(PyArray_Descr *self)
{
    PyObject *shape = NULL;
    PyObject *ret = NULL;

    if (self->descr->subarray == NULL) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    shape = PyArray_IntTupleFromIntp(self->descr->subarray->shape_num_dims,
                                     self->descr->subarray->shape_dims);
    ret = Py_BuildValue("OO",
                (PyObject *)PyArray_Descr_WRAP(self->descr->subarray->base),
                shape);
    Py_DECREF(shape);
    return ret;
}


static PyObject *
_append_to_datetime_typestr(NpyArray_Descr *self, PyObject *ret)
{
    PyObject *tmp;
    PyObject *res;
    int num, den, events;
    char *basestr;
    NpyArray_DateTimeInfo *dt_data;

    /* This shouldn't happen */
    if (self->dtinfo == NULL) {
        return ret;
    }
    dt_data = self->dtinfo;
    num = dt_data->num;
    den = dt_data->den;
    events = dt_data->events;
    basestr = _datetime_strings[dt_data->base];

    if (num == 1) {
        tmp = PyUString_FromString(basestr);
    }
    else {
        tmp = PyUString_FromFormat("%d%s", num, basestr);
    }
    if (den != 1) {
        res = PyUString_FromFormat("/%d", den);
        PyUString_ConcatAndDel(&tmp, res);
    }

    res = PyUString_FromString("[");
    PyUString_ConcatAndDel(&res, tmp);
    PyUString_ConcatAndDel(&res, PyUString_FromString("]"));
    if (events != 1) {
        tmp = PyUString_FromFormat("//%d", events);
        PyUString_ConcatAndDel(&res, tmp);
    }
    PyUString_ConcatAndDel(&ret, res);
    return ret;
}


NPY_NO_EXPORT PyObject *
arraydescr_protocol_typestr_get(PyArray_Descr *self)
{
    return npy_arraydescr_protocol_typestr_get(self->descr);
}

NPY_NO_EXPORT PyObject *
npy_arraydescr_protocol_typestr_get(NpyArray_Descr *self)
{
    char basic_ = self->kind;
    char endian = self->byteorder;
    int size = self->elsize;
    PyObject *ret;

    if (endian == '=') {
        endian = '<';
        if (!NpyArray_IsNativeByteOrder(endian)) {
            endian = '>';
        }
    }
    if (self->type_num == PyArray_UNICODE) {
        size >>= 2;
    }

    ret = PyUString_FromFormat("%c%c%d", endian, basic_, size);
    if (NpyTypeNum_ISDATETIME(self->type)) {
        ret = _append_to_datetime_typestr(self, ret);
    }

    return ret;
}

static PyObject *
arraydescr_typename_get(PyArray_Descr *self)
{
    int len;
    PyTypeObject *typeobj = (PyTypeObject *)self->typeobj;
    PyObject *res;
    char *s;
    int prefix_len = 0;

    if (NpyTypeNum_ISUSERDEF(self->descr->type_num)) {
        s = strrchr(typeobj->tp_name, '.');
        if (s == NULL) {
            res = PyUString_FromString(typeobj->tp_name);
        }
        else {
            res = PyUString_FromStringAndSize(s + 1, strlen(s) - 1);
        }
        return res;
    }
    else {
        if (prefix_len == 0) {
            prefix_len = strlen("numpy.");
        }
        len = strlen(typeobj->tp_name);
        if (*(typeobj->tp_name + (len-1)) == '_') {
            len -= 1;
        }
        len -= prefix_len;
        res = PyUString_FromStringAndSize(typeobj->tp_name+prefix_len, len);
    }
    if (NpyTypeNum_ISFLEXIBLE(self->descr->type_num) &&
              self->descr->elsize != 0) {
        PyObject *p;
        p = PyUString_FromFormat("%d", self->descr->elsize * 8);
        PyUString_ConcatAndDel(&res, p);
    }
    if (PyDataType_ISDATETIME(self)) {
        res = _append_to_datetime_typestr(self->descr, res);
    }

    return res;
}

static PyObject *
arraydescr_base_get(PyArray_Descr *self)
{
    PyArray_Descr *base = NULL;

    if (self->descr->subarray == NULL) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    base = Npy_INTERFACE(self->descr->subarray->base);
    Py_INCREF(base);
    return (PyObject *)base;
}

static PyObject *
arraydescr_shape_get(PyArray_Descr *self)
{
    if (self->descr->subarray == NULL) {
        return PyTuple_New(0);
    }
    return PyArray_IntTupleFromIntp(self->descr->subarray->shape_num_dims,
                                    self->descr->subarray->shape_dims);
}

NPY_NO_EXPORT PyObject *
npy_arraydescr_protocol_descr_get(NpyArray_Descr *self)
{
    PyObject *dobj, *res;
    PyObject *_numpy_internal;

    if (self->names == NULL) {
        /* get default */
        dobj = PyTuple_New(2);
        if (dobj == NULL) {
            return NULL;
        }
        PyTuple_SET_ITEM(dobj, 0, PyUString_FromString(""));
        PyTuple_SET_ITEM(dobj, 1, npy_arraydescr_protocol_typestr_get(self));
        res = PyList_New(1);
        if (res == NULL) {
            Py_DECREF(dobj);
            return NULL;
        }
        PyList_SET_ITEM(res, 0, dobj);
        return res;
    }

    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
    if (_numpy_internal == NULL) {
        return NULL;
    }
    res = PyObject_CallMethod(_numpy_internal, "_array_descr", "O",
                              Npy_INTERFACE(self));
    Py_DECREF(_numpy_internal);
    return res;
}

NPY_NO_EXPORT PyObject *
arraydescr_protocol_descr_get(PyArray_Descr *self)
{
    return npy_arraydescr_protocol_descr_get(self->descr);
}


/*
 * returns 1 for a builtin type
 * and 2 for a user-defined data-type descriptor
 * return 0 if neither (i.e. it's a copy of one)
 */
static PyObject *
arraydescr_isbuiltin_get(PyArray_Descr *self)
{
    long val;
    val = 0;
    if (NULL != self->descr->fields) {
        val = 1;
    }
    if (NpyTypeNum_ISUSERDEF(self->descr->type_num)) {
        val = 2;
    }
    return PyInt_FromLong(val);
}


/*
 * return Py_True if this data-type descriptor
 * has native byteorder if no fields are defined
 *
 * or if all sub-fields have native-byteorder if
 * fields are defined
 */
static PyObject *
arraydescr_isnative_get(PyArray_Descr *self)
{
    PyObject *ret;
    int retval;
    retval = npy_arraydescr_isnative(self->descr);
    if (retval == -1) {
        return NULL;
    }
    ret = retval ? Py_True : Py_False;
    Py_INCREF(ret);
    return ret;
}

static PyObject *
arraydescr_fields_get(PyArray_Descr *self)
{
    return PyArrayDescr_GetFields(self);
}

static PyObject *
arraydescr_dtinfo_get(PyArray_Descr *self)
{
    PyObject *res;

    if (self->descr->dtinfo == NULL) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    res = _get_datetime_tuple_from_datetimeinfo( self->descr->dtinfo );
    Py_INCREF( res );

    return PyDictProxy_New(res);
}


static PyObject *
arraydescr_hasobject_get(PyArray_Descr *self)
{
    PyObject *res;

    if (PyDataType_FLAGCHK(self, NPY_ITEM_HASOBJECT)) {
        res = Py_True;
    }
    else {
        res = Py_False;
    }
    Py_INCREF(res);
    return res;
}


static PyObject *
arraydescr_names_get(PyArray_Descr *self)
{
    return PyArrayDescr_GetNames(self);
}


static int
arraydescr_names_set(PyArray_Descr *self, PyObject *val)
{
    int n = 0;
    int i;
    char **nameslist = NULL;
    if (self->descr->names == NULL) {
        PyErr_SetString(PyExc_ValueError, "there are no fields defined");
        return -1;
    }

    for (n = 0; NULL != self->descr->names[n]; n++) ;
    if (!PySequence_Check(val) || PyObject_Size((PyObject *)val) != n) {
        PyErr_Format(PyExc_ValueError,
                "must replace all names at once with a sequence of length %d",
                n);
        return -1;
    }
    /* Make sure all entries are strings */
    for (i = 0; i < n; i++) {
        PyObject *item;
        int valid = 1;
        item = PySequence_GetItem(val, i);
        valid = PyUString_Check(item);
        Py_DECREF(item);
        if (!valid) {
            PyErr_Format(PyExc_ValueError,
                         "item #%d of names is of type %s and not string",
                         i, Py_TYPE(item)->tp_name);
            return -1;
        }
    }
    /* Update dictionary keys in fields */
    nameslist = arraydescr_seq_to_nameslist(val);
    NpyArray_DescrReplaceNames(self->descr, nameslist);

    return 0;
}


/* Takes a sequence of strings and returns an array of char**.
   Each string is allocated and must be free'd eventually. */
NPY_NO_EXPORT char **
arraydescr_seq_to_nameslist(PyObject *seq)
{
    char **nameslist = NULL;
    int n = PySequence_Length(seq);
    int i;

    nameslist = NpyArray_DescrAllocNames(n);
    if (NULL != nameslist) {
        for (i = 0; i < n; i++) {
            PyObject *key = PySequence_GetItem(seq, i);
            nameslist[i] = strdup(TO_CSTRING(key));
            Py_DECREF(key);
        }
        nameslist[i] = NULL;
    }
    return nameslist;
}


/* Converts a PyDict structure defining a set of PyArray_Descr fields into a
   NpyDict describing the same fields. The PyDict values are 2-tuples or
   3-tuples containing another descr object, an offset, and an optional
   title string. */
NPY_NO_EXPORT NpyDict *
arraydescr_fields_from_pydict(PyObject *dict)
{
    NpyDict *fields = NpyArray_DescrAllocFields();
    PyObject *value;
    PyObject *key = NULL;
    NpyArray_Descr *descr;
    int offset;
    const char *title;
    Py_ssize_t pos;

    /* Extract dict of tuples of { sub descriptor, offset, [title] } and set
       as fields on the descriptor */
    pos = 0;
    while (PyDict_Next(dict, &pos, &key, &value)) {
        /* TODO: Unwrap descr object. Do we need type checking? */
        descr = ((PyArray_Descr *) PyTuple_GetItem(value, 0))->descr;
        offset = PyInt_AsLong(PyTuple_GetItem(value, 1));
        title = (2 < PyTuple_Size(value)) ?
                  TO_CSTRING(PyTuple_GetItem(value, 2)) : NULL;

        /* DescrSetField will incref subDescr, copy strings */
        Npy_INCREF(descr);
        NpyArray_DescrSetField(fields, TO_CSTRING(key), descr,
                               offset, title);
    }
    return fields;
}


static PyObject *
arraydescr_type_get(PyArray_Descr *self)
{
    Py_INCREF(self->typeobj);
    return (PyObject *)self->typeobj;
}


static PyObject *
arraydescr_kind_get(PyArray_Descr *self)
{
#if defined(NPY_PY3K)
    return PyUnicode_FromStringAndSize(&self->descr->kind, 1);
#else
    return PyString_FromStringAndSize(&self->descr->kind, 1);
#endif
}


static PyObject *
arraydescr_char_get(PyArray_Descr *self)
{
#if defined(NPY_PY3K)
    return PyUnicode_FromStringAndSize(&self->descr->type, 1);
#else
    return PyString_FromStringAndSize(&self->descr->type, 1);
#endif
}


static PyObject *
arraydescr_num_get(PyArray_Descr *self)
{
    return PyInt_FromLong(self->descr->type_num);
}


static PyObject *
arraydescr_byteorder_get(PyArray_Descr *self)
{
#if defined(NPY_PY3K)
    return PyUnicode_FromStringAndSize(&self->descr->byteorder, 1);
#else
    return PyString_FromStringAndSize(&self->descr->byteorder, 1);
#endif
}


static PyObject *
arraydescr_itemsize_get(PyArray_Descr *self)
{
    return PyInt_FromLong(self->descr->elsize);
}


static PyObject *
arraydescr_alignment_get(PyArray_Descr *self)
{
    return PyInt_FromLong(self->descr->alignment);
}


static PyObject *
arraydescr_flags_get(PyArray_Descr *self)
{
    return PyInt_FromLong(self->descr->flags);
}


static PyGetSetDef arraydescr_getsets[] = {
    {"subdtype",
        (getter)arraydescr_subdescr_get,
        NULL, NULL, NULL},
    {"descr",
        (getter)arraydescr_protocol_descr_get,
        NULL, NULL, NULL},
    {"str",
        (getter)arraydescr_protocol_typestr_get,
        NULL, NULL, NULL},
    {"name",
        (getter)arraydescr_typename_get,
        NULL, NULL, NULL},
    {"base",
        (getter)arraydescr_base_get,
        NULL, NULL, NULL},
    {"shape",
        (getter)arraydescr_shape_get,
        NULL, NULL, NULL},
    {"isbuiltin",
        (getter)arraydescr_isbuiltin_get,
        NULL, NULL, NULL},
    {"isnative",
        (getter)arraydescr_isnative_get,
        NULL, NULL, NULL},
    {"fields",
        (getter)arraydescr_fields_get,
        NULL, NULL, NULL},
    {"dtinfo",
        (getter)arraydescr_dtinfo_get,
        NULL, NULL, NULL},
    {"names",
        (getter)arraydescr_names_get,
        (setter)arraydescr_names_set,
        NULL, NULL},
    {"hasobject",
        (getter)arraydescr_hasobject_get,
        NULL, NULL, NULL},
    {"type",
        (getter)arraydescr_type_get,
        NULL, NULL, NULL},
    {"kind",
        (getter)arraydescr_kind_get,
        NULL, NULL, NULL},
    {"char",
        (getter)arraydescr_char_get,
        NULL, NULL, NULL},
    {"num",
        (getter)arraydescr_num_get,
        NULL, NULL, NULL},
    {"byteorder",
        (getter)arraydescr_byteorder_get,
        NULL, NULL, NULL},
    {"itemsize",
        (getter)arraydescr_itemsize_get,
        NULL, NULL, NULL},
    {"alignment",
        (getter)arraydescr_alignment_get,
        NULL, NULL, NULL},
    {"flags",
        (getter)arraydescr_flags_get,
        NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};



static PyObject *
arraydescr_new(PyTypeObject *NPY_UNUSED(subtype), PyObject *args,
               PyObject *kwds)
{
    PyObject *odescr=NULL;
    PyArray_Descr *descr, *conv = NULL;
    Bool align = FALSE;
    Bool copy = FALSE;
    Bool copied = FALSE;
    static char *kwlist[] = {"dtype", "align", "copy", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&O&", kwlist,
                &odescr, PyArray_BoolConverter, &align,
                PyArray_BoolConverter, &copy )) {
        Py_XDECREF(odescr);
        return NULL;
    }

    if (align) {
        if (!PyArray_DescrAlignConverter(odescr, &conv)) {
            Py_XDECREF(conv);
            return NULL;
        }
    }
    else if (!PyArray_DescrConverter(odescr, &conv)) {
        Py_XDECREF(conv);
        return NULL;
    }
    /* Get a new copy of it unless it's already a copy */
    if (copy && NULL == conv->descr->fields) {
        descr = PyArray_DescrNew(conv);
        Py_DECREF(conv);
        conv = descr;
        copied = TRUE;
    }

    return (PyObject *)conv;
}


/* return a tuple of (callable object, args, state). */
static PyObject *
arraydescr_reduce(PyArray_Descr *self, PyObject *NPY_UNUSED(args))
{
    /*
     * version number of this pickle type. Increment if we need to
     * change the format. Be sure to handle the old versions in
     * arraydescr_setstate.
    */
    const int version = 4;
    PyObject *ret, *mod, *obj;
    PyObject *state;
    char endian;
    int elsize, alignment;
    NpyArray_Descr *selfCore = self->descr;     /* Core repr of descriptor */

    ret = PyTuple_New(3);
    if (ret == NULL) {
        return NULL;
    }
    mod = PyImport_ImportModule("numpy.core.multiarray");
    if (mod == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    obj = PyObject_GetAttrString(mod, "dtype");
    Py_DECREF(mod);
    if (obj == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    PyTuple_SET_ITEM(ret, 0, obj);
    if (NpyTypeNum_ISUSERDEF(selfCore->type_num)
            || ((selfCore->type_num == PyArray_VOID
                    && self->typeobj != &PyVoidArrType_Type))) {
        obj = (PyObject *)self->typeobj;
        Py_INCREF(obj);
    }
    else {
        elsize = selfCore->elsize;
        if (selfCore->type_num == PyArray_UNICODE) {
            elsize >>= 2;
        }
        obj = PyUString_FromFormat("%c%d",selfCore->kind, elsize);
    }
    PyTuple_SET_ITEM(ret, 1, Py_BuildValue("(Nii)", obj, 0, 1));

    /*
     * Now return the state which is at least byteorder,
     * subarray, and fields
     */
    endian = selfCore->byteorder;
    if (endian == '=') {
        endian = '<';
        if (!NpyArray_IsNativeByteOrder(endian)) {
            endian = '>';
        }
    }

    if (PyDataType_ISDATETIME(self)) {
        /* newobj is a tuple date_time info (str, num, den, events) */
        PyObject *newobj;

        state = PyTuple_New(9);
        PyTuple_SET_ITEM(state, 0, PyInt_FromLong(version));

        newobj = _get_datetime_tuple_from_datetimeinfo(selfCore->dtinfo);
        PyTuple_SET_ITEM(state, 8, newobj);
    }
    else { /* Use version 3 pickle format */
        state = PyTuple_New(8);
        PyTuple_SET_ITEM(state, 0, PyInt_FromLong(3));
    }

    PyTuple_SET_ITEM(state, 1, PyUString_FromFormat("%c", endian));
    PyTuple_SET_ITEM(state, 2, arraydescr_subdescr_get(self));
    if (NULL != self->descr->names) {
        PyTuple_SET_ITEM(state, 3, arraydescr_names_get(self));
        PyTuple_SET_ITEM(state, 4, arraydescr_fields_get(self));
    }
    else {
        PyTuple_SET_ITEM(state, 3, Py_None);
        PyTuple_SET_ITEM(state, 4, Py_None);
        Py_INCREF(Py_None);
        Py_INCREF(Py_None);
    }

    /* for extended types it also includes elsize and alignment */
    if (NpyTypeNum_ISEXTENDED(selfCore->type_num)) {
        elsize = selfCore->elsize;
        alignment = selfCore->alignment;
    }
    else {
        elsize = -1;
        alignment = -1;
    }
    PyTuple_SET_ITEM(state, 5, PyInt_FromLong(elsize));
    PyTuple_SET_ITEM(state, 6, PyInt_FromLong(alignment));
    PyTuple_SET_ITEM(state, 7, PyInt_FromLong(selfCore->flags));

    PyTuple_SET_ITEM(ret, 2, state);
    return ret;
}



/*
 * state is at least byteorder, subarray, and fields but could include elsize
 * and alignment for EXTENDED arrays
 */
static PyObject *
arraydescr_setstate(PyArray_Descr *self, PyObject *args)
{
    int elsize = -1, alignment = -1;
    int version = 4;
#if defined(NPY_PY3K)
    int endian;
#else
    char endian;
#endif
    PyObject *subarray, *fields, *names = NULL, *dtinfo=NULL;
    int incref_names = 1;
    int dtypeflags = 0;

/*    if (NULL == self->fields) {
        Py_INCREF(Py_None);
        return Py_None;
    } */
    if (PyTuple_GET_SIZE(args) != 1
            || !(PyTuple_Check(PyTuple_GET_ITEM(args, 0)))) {
        PyErr_BadInternalCall();
        return NULL;
    }
    switch (PyTuple_GET_SIZE(PyTuple_GET_ITEM(args,0))) {
    case 9:
#if defined(NPY_PY3K)
#define _ARGSTR_ "(iCOOOiiiO)"
#else
#define _ARGSTR_ "(icOOOiiiO)"
#endif
        if (!PyArg_ParseTuple(args, _ARGSTR_, &version, &endian,
                    &subarray, &names, &fields, &elsize,
                    &alignment, &dtypeflags, &dtinfo)) {
            return NULL;
#undef _ARGSTR_
        }
        break;
    case 8:
#if defined(NPY_PY3K)
#define _ARGSTR_ "(iCOOOiii)"
#else
#define _ARGSTR_ "(icOOOiii)"
#endif
        if (!PyArg_ParseTuple(args, _ARGSTR_, &version, &endian,
                    &subarray, &names, &fields, &elsize,
                    &alignment, &dtypeflags)) {
            return NULL;
#undef _ARGSTR_
        }
        break;
    case 7:
#if defined(NPY_PY3K)
#define _ARGSTR_ "(iCOOOii)"
#else
#define _ARGSTR_ "(icOOOii)"
#endif
        if (!PyArg_ParseTuple(args, _ARGSTR_, &version, &endian,
                    &subarray, &names, &fields, &elsize,
                    &alignment)) {
            return NULL;
#undef _ARGSTR_
        }
        break;
    case 6:
#if defined(NPY_PY3K)
#define _ARGSTR_ "(iCOOii)"
#else
#define _ARGSTR_ "(icOOii)"
#endif
        if (!PyArg_ParseTuple(args, _ARGSTR_, &version,
                    &endian, &subarray, &fields,
                    &elsize, &alignment)) {
            PyErr_Clear();
#undef _ARGSTR_
        }
        break;
    case 5:
        version = 0;
#if defined(NPY_PY3K)
#define _ARGSTR_ "(COOii)"
#else
#define _ARGSTR_ "(cOOii)"
#endif
        if (!PyArg_ParseTuple(args, _ARGSTR_,
                    &endian, &subarray, &fields, &elsize,
                    &alignment)) {
#undef _ARGSTR_
            return NULL;
        }
        break;
    default:
        /* raise an error */
        if (PyTuple_GET_SIZE(PyTuple_GET_ITEM(args,0)) > 5) {
            version = PyInt_AsLong(PyTuple_GET_ITEM(args, 0));
        }
        else {
            version = -1;
        }
    }

    /*
     * If we ever need another pickle format, increment the version
     * number. But we should still be able to handle the old versions.
     */
    if (version < 0 || version > 4) {
        PyErr_Format(PyExc_ValueError,
                     "can't handle version %d of numpy.dtype pickle",
                     version);
        return NULL;
    }

    if (version == 1 || version == 0) {
        if (fields != Py_None) {
            PyObject *key, *list;
            key = PyInt_FromLong(-1);
            list = PyDict_GetItem(fields, key);
            if (!list) {
                return NULL;
            }
            Py_INCREF(list);
            names = list;
            PyDict_DelItem(fields, key);
            incref_names = 0;
        }
        else {
            names = Py_None;
        }
    }

    if ((fields == Py_None && names != Py_None) ||
        (names == Py_None && fields != Py_None)) {
        PyErr_Format(PyExc_ValueError, "inconsistent fields and names");
        return NULL;
    }

    if (endian != '|' && NpyArray_IsNativeByteOrder(endian)) {
        endian = '=';
    }
    self->descr->byteorder = endian;
    if (self->descr->subarray) {
        NpyArray_DestroySubarray(self->descr->subarray);
    }
    self->descr->subarray = NULL;

    if (subarray != Py_None) {
        NpyArray_Descr *selfCore = self->descr;
        PyObject *shape = NULL;
        int len = 0;

        selfCore->subarray = NpyArray_malloc(sizeof(NpyArray_ArrayDescr));
        selfCore->subarray->base = ((PyArray_Descr *)
                                    PyTuple_GET_ITEM(subarray, 0))->descr;
        Npy_INCREF( selfCore->subarray->base );

        shape = PyTuple_GET_ITEM(subarray, 1);
        len = PySequence_Check(shape) ? PySequence_Length(shape) : 1;
        selfCore->subarray->shape_dims = NpyArray_malloc(len * sizeof(npy_intp));
        if (PyArray_IntpFromSequence(shape,
                               selfCore->subarray->shape_dims, len) == -1) {
            NpyArray_free(selfCore->subarray->shape_dims);
            NpyArray_free(selfCore->subarray);
            selfCore->subarray = NULL;
            return PY_FAIL;
        }
        selfCore->subarray->shape_num_dims = len;
        assert((selfCore->subarray->shape_num_dims == 0 &&
                selfCore->subarray->shape_dims == NULL) ||
               (selfCore->subarray->shape_num_dims > 0 &&
                selfCore->subarray->shape_dims != NULL));
    }

    if (fields != Py_None) {
        /* Convert sequence of strings to array of char *. self will managed
           memory in the end. */
        NpyArray_DescrDeallocNamesAndFields(self->descr);
        NpyArray_DescrSetNames(self->descr, arraydescr_seq_to_nameslist(names));
        self->descr->fields = arraydescr_fields_from_pydict(fields);
        if (!incref_names) {
            Py_DECREF(names);
        }
    }

    if (NpyTypeNum_ISEXTENDED(self->descr->type_num)) {
        self->descr->elsize = elsize;
        self->descr->alignment = alignment;
    }

    self->descr->flags = dtypeflags;
    if (version < 3) {
        self->descr->flags = npy_descr_find_object_flag(self->descr);
    }

    if (PyDataType_ISDATETIME(self)
        && (dtinfo != Py_None)
        && (dtinfo != NULL)) {

      self->descr->dtinfo = _convert_datetime_tuple_to_datetimeinfo( dtinfo );
    }

    Py_INCREF(Py_None);
    return Py_None;
}

/*NUMPY_API
 *
 * Get type-descriptor from an object forcing alignment if possible
 * None goes to DEFAULT type.
 *
 * any object with the .fields attribute and/or .itemsize attribute (if the
 *.fields attribute does not give the total size -- i.e. a partial record
 * naming).  If itemsize is given it must be >= size computed from fields
 *
 * The .fields attribute must return a convertible dictionary if present.
 * Result inherits from PyArray_VOID.
*/
NPY_NO_EXPORT int
PyArray_DescrAlignConverter(PyObject *obj, PyArray_Descr **at)
{
    if (PyDict_Check(obj)) {
        *at =  _convert_from_dict(obj, 1);
    }
    else if (PyBytes_Check(obj)) {
        *at = _convert_from_commastring(obj, 1);
    }
    else if (PyUnicode_Check(obj)) {
        PyObject *tmp;
        tmp = PyUnicode_AsASCIIString(obj);
        *at = _convert_from_commastring(tmp, 1);
        Py_DECREF(tmp);
    }
    else if (PyList_Check(obj)) {
        *at = _convert_from_array_descr(obj, 1);
    }
    else {
        return PyArray_DescrConverter(obj, at);
    }
    if (*at == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError,
                    "data-type-descriptor not understood");
        }
        return PY_FAIL;
    }
    return PY_SUCCEED;
}

/*NUMPY_API
 *
 * Get type-descriptor from an object forcing alignment if possible
 * None goes to NULL.
 */
NPY_NO_EXPORT int
PyArray_DescrAlignConverter2(PyObject *obj, PyArray_Descr **at)
{
    if (PyDict_Check(obj)) {
        *at =  _convert_from_dict(obj, 1);
    }
    else if (PyBytes_Check(obj)) {
        *at = _convert_from_commastring(obj, 1);
    }
    else if (PyUnicode_Check(obj)) {
        PyObject *tmp;
        tmp = PyUnicode_AsASCIIString(obj);
        *at = _convert_from_commastring(tmp, 1);
        Py_DECREF(tmp);
    }
    else if (PyList_Check(obj)) {
        *at = _convert_from_array_descr(obj, 1);
    }
    else {
        return PyArray_DescrConverter2(obj, at);
    }
    if (*at == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError,
                    "data-type-descriptor not understood");
        }
        return PY_FAIL;
    }
    return PY_SUCCEED;
}


/*NUMPY_API
 *
 * returns a copy of the PyArray_Descr structure with the byteorder
 * altered:
 * no arguments:  The byteorder is swapped (in all subfields as well)
 * single argument:  The byteorder is forced to the given state
 * (in all subfields as well)
 *
 * Valid states:  ('big', '>') or ('little' or '<')
 * ('native', or '=')
 *
 * If a descr structure with | is encountered it's own
 * byte-order is not changed but any fields are:
 *
 *
 * Deep bytorder change of a data-type descriptor
 * *** Leaves reference count of self unchanged --- does not DECREF self ***
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrNewByteorder(PyArray_Descr *self, char newendian)
{
    NpyArray_Descr *result = NpyArray_DescrNewByteorder(self->descr, newendian);

    Py_INCREF( Npy_INTERFACE(result) );
    Npy_DECREF(result);
    return Npy_INTERFACE(result);
}


static PyObject *
arraydescr_newbyteorder(PyArray_Descr *self, PyObject *args)
{
    char endian=PyArray_SWAP;

    if (!PyArg_ParseTuple(args, "|O&", PyArray_ByteorderConverter,
                &endian)) {
        return NULL;
    }
    return (PyObject *)PyArray_DescrNewByteorder(self, endian);
}

static PyMethodDef arraydescr_methods[] = {
    /* for pickling */
    {"__reduce__",
        (PyCFunction)arraydescr_reduce,
        METH_VARARGS, NULL},
    {"__setstate__",
        (PyCFunction)arraydescr_setstate,
        METH_VARARGS, NULL},
    {"newbyteorder",
        (PyCFunction)arraydescr_newbyteorder,
        METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}           /* sentinel */
};

static PyObject *
arraydescr_str(PyArray_Descr *self)
{
    PyObject *sub;

    if (self->descr->names) {
        PyObject *lst;
        lst = arraydescr_protocol_descr_get(self);
        if (!lst) {
            sub = PyUString_FromString("<err>");
            PyErr_Clear();
        }
        else {
            sub = PyObject_Str(lst);
        }
        Py_XDECREF(lst);
        if (self->descr->type_num != PyArray_VOID) {
            PyObject *p, *t;
            t=PyUString_FromString("'");
            p = npy_arraydescr_protocol_typestr_get(self->descr);
            PyUString_Concat(&p, t);
            PyUString_ConcatAndDel(&t, p);
            p = PyUString_FromString("(");
            PyUString_ConcatAndDel(&p, t);
            PyUString_ConcatAndDel(&p, PyUString_FromString(", "));
            PyUString_ConcatAndDel(&p, sub);
            PyUString_ConcatAndDel(&p, PyUString_FromString(")"));
            sub = p;
        }
    }
    else if (self->descr->subarray) {
        PyObject *p;
        PyObject *t = PyUString_FromString("(");
        PyObject *sh;
        p = arraydescr_str( PyArray_Descr_WRAP( self->descr->subarray->base) );
        if (!self->descr->subarray->base->names &&
                  !self->descr->subarray->base->subarray) {
            PyObject *t=PyUString_FromString("'");
            PyUString_Concat(&p, t);
            PyUString_ConcatAndDel(&t, p);
            p = t;
        }
        PyUString_ConcatAndDel(&t, p);
        PyUString_ConcatAndDel(&t, PyUString_FromString(","));
        sh = PyArray_IntTupleFromIntp(self->descr->subarray->shape_num_dims,
                                      self->descr->subarray->shape_dims);
        PyUString_ConcatAndDel(&t, PyObject_Str(sh));
        Py_DECREF(sh);
        PyUString_ConcatAndDel(&t, PyUString_FromString(")"));
        sub = t;
    }
    else if (NpyTypeNum_ISFLEXIBLE(self->descr->type_num) ||
             !NpyArray_ISNBO(self->descr->byteorder)) {
        sub = npy_arraydescr_protocol_typestr_get(self->descr);
    }
    else {
        sub = arraydescr_typename_get(self);
    }
    return sub;
}

static PyObject *
arraydescr_repr(PyArray_Descr *self)
{
    PyObject *sub, *s;
    s = PyUString_FromString("dtype(");
    sub = arraydescr_str(self);
    if (sub == NULL) {
        return sub;
    }
    if (!self->descr->names && !self->descr->subarray) {
        PyObject *t=PyUString_FromString("'");
        PyUString_Concat(&sub, t);
        PyUString_ConcatAndDel(&t, sub);
        sub = t;
    }
    PyUString_ConcatAndDel(&s, sub);
    sub = PyUString_FromString(")");
    PyUString_ConcatAndDel(&s, sub);
    return s;
}

static PyObject *
arraydescr_richcompare(PyArray_Descr *self, PyObject *other, int cmp_op)
{
    PyArray_Descr *new = NULL;
    PyObject *result = Py_NotImplemented;
    if (!PyArray_DescrCheck(other)) {
        if (PyArray_DescrConverter(other, &new) == PY_FAIL) {
            return NULL;
        }
    }
    else {
        new = (PyArray_Descr *)other;
        Py_INCREF(new);
    }
    switch (cmp_op) {
    case Py_LT:
        if (!PyArray_EquivTypes(self, new) && PyArray_CanCastTo(self, new)) {
            result = Py_True;
        }
        else {
            result = Py_False;
        }
        break;
    case Py_LE:
        if (PyArray_CanCastTo(self, new)) {
            result = Py_True;
        }
        else {
            result = Py_False;
        }
        break;
    case Py_EQ:
        if (PyArray_EquivTypes(self, new)) {
            result = Py_True;
        }
        else {
            result = Py_False;
        }
        break;
    case Py_NE:
        if (PyArray_EquivTypes(self, new))
            result = Py_False;
        else
            result = Py_True;
        break;
    case Py_GT:
        if (!PyArray_EquivTypes(self, new) && PyArray_CanCastTo(new, self)) {
            result = Py_True;
        }
        else {
            result = Py_False;
        }
        break;
    case Py_GE:
        if (PyArray_CanCastTo(new, self)) {
            result = Py_True;
        }
        else {
            result = Py_False;
        }
        break;
    default:
        result = Py_NotImplemented;
    }

    Py_XDECREF(new);
    Py_INCREF(result);
    return result;
}

/*************************************************************************
 ****************   Implement Mapping Protocol ***************************
 *************************************************************************/

static Py_ssize_t
descr_length(PyObject *self0)
{
    PyArray_Descr *self = (PyArray_Descr *)self0;

    if (NULL != self->descr->names) {
        int i;

        for (i=0; NULL != self->descr->names[i]; i++) ;
        return i;
    }
    return 0;
}

static PyObject *
descr_repeat(PyObject *self, Py_ssize_t length)
{
    PyObject *tup;
    PyArray_Descr *new;
    if (length < 0) {
        return PyErr_Format(PyExc_ValueError,
                "Array length must be >= 0, not %"INTP_FMT, length);
    }
    tup = Py_BuildValue("O" NPY_SSIZE_T_PYFMT, self, length);
    if (tup == NULL) {
        return NULL;
    }
    PyArray_DescrConverter(tup, &new);
    Py_DECREF(tup);
    return (PyObject *)new;
}

static PyObject *
descr_subscript(PyArray_Descr *self, PyObject *op)
{
    PyObject *retval;

    if (!self->descr->names) {
        PyObject *astr = arraydescr_str(self);
#if defined(NPY_PY3K)
        PyObject *bstr = PyUnicode_AsUnicodeEscapeString(astr);
        Py_DECREF(astr);
        astr = bstr;
#endif
        PyErr_Format(PyExc_KeyError,
                "There are no fields in dtype %s.", PyBytes_AsString(astr));
        Py_DECREF(astr);
        return NULL;
    }
#if defined(NPY_PY3K)
    if (PyUString_Check(op)) {
#else
    if (PyUString_Check(op) || PyUnicode_Check(op)) {
#endif
        NpyArray_DescrField *value = NpyDict_Get(self->descr->fields,
                                                 TO_CSTRING(op));
        PyObject *s;

        if (NULL == value) {
            if (PyUnicode_Check(op)) {
                s = PyUnicode_AsUnicodeEscapeString(op);
            }
            else {
                s = op;
            }

            PyErr_Format(PyExc_KeyError,
                    "Field named \'%s\' not found.", PyBytes_AsString(s));
            if (s != op) {
                Py_DECREF(s);
            }
            return NULL;
        }
        retval = (PyObject *)Npy_INTERFACE(value->descr);
        Py_INCREF(retval);
    }
    else if (PyInt_Check(op)) {
        NpyArray_DescrField *field;
        int size;
        int value = PyArray_PyIntAsInt(op);

        for (size=0; NULL != self->descr->names[size]; size++) ;

        if (PyErr_Occurred()) {
            return NULL;
        }
        if (value < 0) {
            value += size;
        }
        if (value < 0 || value >= size) {
            PyErr_Format(PyExc_IndexError, "Field index out of range.");
            return NULL;
        }

        field = NpyDict_Get(self->descr->fields, self->descr->names[value]);
        retval = (PyObject *)PyArray_Descr_WRAP(field->descr);
        Py_INCREF(retval);
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "Field key must be an integer, string, or unicode.");
        return NULL;
    }
    return retval;
}

static PySequenceMethods descr_as_sequence = {
    descr_length,
    (binaryfunc)NULL,
    descr_repeat,
    NULL, NULL,
    NULL,                              /* sq_ass_item */
    NULL,                              /* ssizessizeobjargproc sq_ass_slice */
    0,                                 /* sq_contains */
    0,                                 /* sq_inplace_concat */
    0,                                 /* sq_inplace_repeat */
};

static PyMappingMethods descr_as_mapping = {
    descr_length,                                /* mp_length*/
    (binaryfunc)descr_subscript,                 /* mp_subscript*/
    (objobjargproc)NULL,                         /* mp_ass_subscript*/
};

/****************** End of Mapping Protocol ******************************/

NPY_NO_EXPORT PyTypeObject PyArrayDescr_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.dtype",                              /* tp_name */
    sizeof(PyArray_Descr),                      /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)arraydescr_dealloc,             /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    (void *)0,                                  /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    (reprfunc)arraydescr_repr,                  /* tp_repr */
    0,                                          /* tp_as_number */
    &descr_as_sequence,                         /* tp_as_sequence */
    &descr_as_mapping,                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    (reprfunc)arraydescr_str,                   /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    (richcmpfunc)arraydescr_richcompare,        /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    arraydescr_methods,                         /* tp_methods */
    0,                                          /* tp_members */
    arraydescr_getsets,                         /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    arraydescr_new,                             /* tp_new */
    (freefunc)_pya_free,                        /* tp_free */
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
