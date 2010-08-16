#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define _MULTIARRAYMODULE
#include "npy_api.h"
#include "npy_dict.h"
#include <numpy/ndarrayobject.h>

#include "npy_config.h"

#include "numpy_3kcompat.h"

#include "hashdescr.h"

/*
 * How does this work ? The hash is computed from a list which contains all the
 * information specific to a type. The hard work is to build the list
 * (_array_descr_walk). The list is built as follows:
 *      * If the dtype is builtin (no fields, no subarray), then the list
 *      contains 6 items which uniquely define one dtype (_array_descr_builtin)
 *      * If the dtype is a compound array, one walk on each field. For each
 *      field, we append title, names, offset to the final list used for
 *      hashing, and then append the list recursively built for each
 *      corresponding dtype (_array_descr_walk_fields)
 *      * If the dtype is a subarray, one adds the shape tuple to the list, and
 *      then append the list recursively built for each corresponding dtype
 *      (_array_descr_walk_subarray)
 *
 */

static int _is_array_descr_builtin(NpyArray_Descr* descr);
static int _array_descr_walk(NpyArray_Descr* descr, PyObject *l);
static int _array_descr_walk_fields(NpyDict *fields, PyObject* l);
static int _array_descr_builtin(NpyArray_Descr* descr, PyObject *l);

/*
 * Return true if descr is a builtin type
 */
static int _is_array_descr_builtin(NpyArray_Descr* descr)
{
        if (NULL != descr->fields) {
                return 0;
        }
        if (descr->subarray != NULL) {
                return 0;
        }
        return 1;
}

/*
 * Add to l all the items which uniquely define a builtin type
 */
static int _array_descr_builtin(NpyArray_Descr* descr, PyObject *l)
{
    Py_ssize_t i;
    PyObject *t, *item;

    /*
     * For builtin type, hash relies on : kind + byteorder + flags +
     * type_num + elsize + alignment
     */
    t = Py_BuildValue("(cciiii)", descr->kind, descr->byteorder,
            descr->flags, descr->type_num, descr->elsize,
            descr->alignment);

    for(i = 0; i < PyTuple_Size(t); ++i) {
        item = PyTuple_GetItem(t, i);
        if (item == NULL) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) Error while computing builting hash");
            goto clean_t;
        }
        Py_INCREF(item);
        PyList_Append(l, item);
    }

    Py_DECREF(t);
    return 0;

clean_t:
    Py_DECREF(t);
    return -1;
}

/*
 * Walk inside the fields and add every item which will be used for hashing
 * into the list l
 *
 * Return 0 on success
 */
static int _array_descr_walk_fields(NpyDict *fields, PyObject* l)
{
    const char *key = NULL;
    NpyArray_DescrField *value = NULL;
    NpyDict_Iter pos;
    int st;

    NpyDict_IterInit(&pos);
    while (NpyDict_IterNext(fields, &pos, (void **)&key, (void **)&value)) {
        /*
         * For each field, add the key + descr + offset to l
         */

        PyList_Append(l, PyString_FromString(key));

        st = _array_descr_walk(value->descr, l);
        if (st) {
            return -1;
        }
        PyList_Append(l, PyInt_FromLong(value->offset));
    }

    return 0;
}


/*
 * Walk into subarray, and add items for hashing in l
 *
 * Return 0 on success
 */
static int _array_descr_walk_subarray(NpyArray_ArrayDescr* adescr, PyObject *l)
{
    Py_ssize_t i;
    int st;

    /*
     * Add shape and descr itself to the list of object to hash
     */
    for (i=0; i < adescr->shape_num_dims; i++) {
        PyList_Append(l, PyInt_FromLong(adescr->shape_dims[i]));
    }

    Npy_INCREF(adescr->base);
    st = _array_descr_walk(adescr->base, l);
    Npy_DECREF(adescr->base);

    return st;
}

/*
 * 'Root' function to walk into a dtype. May be called recursively
 */
static int _array_descr_walk(NpyArray_Descr* descr, PyObject *l)
{
    int st;

    if (_is_array_descr_builtin(descr)) {
        return _array_descr_builtin(descr, l);
    } else {
        if (NULL != descr->fields) {
            st = _array_descr_walk_fields(descr->fields, l);
            if (st) {
                return -1;
            }
        }
        if(descr->subarray != NULL) {
            st = _array_descr_walk_subarray(descr->subarray, l);
            if (st) {
                return -1;
            }
        }
    }

    return 0;
}

/*
 * Return 0 if successfull
 */
static int _PyArray_DescrHashImp(NpyArray_Descr *descr, long *hash)
{
    PyObject *l, *tl, *item;
    Py_ssize_t i;
    int st;

    l = PyList_New(0);
    if (l == NULL) {
        return -1;
    }

    st = _array_descr_walk(descr, l);
    if (st) {
        goto clean_l;
    }

    /*
     * Convert the list to tuple and compute the tuple hash using python
     * builtin function
     */
    tl = PyTuple_New(PyList_Size(l));
    for(i = 0; i < PyList_Size(l); ++i) {
        item = PyList_GetItem(l, i);
        if (item == NULL) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) Error while translating the list into a tuple " \
                    "(NULL item)");
            goto clean_tl;
        }
        PyTuple_SetItem(tl, i, item);
    }

    *hash = PyObject_Hash(tl);
    if (*hash == -1) {
        /* XXX: does PyObject_Hash set an exception on failure ? */
#if 0
        PyErr_SetString(PyExc_SystemError,
                "(Hash) Error while hashing final tuple");
#endif
        goto clean_tl;
    }
    Py_DECREF(tl);
    Py_DECREF(l);

    return 0;

clean_tl:
    Py_DECREF(tl);
clean_l:
    Py_DECREF(l);
    return -1;
}

NPY_NO_EXPORT long
PyArray_DescrHash(PyObject* odescr)
{
    PyArray_Descr *descr;
    int st;
    long hash;

    if (!PyArray_DescrCheck(odescr)) {
        PyErr_SetString(PyExc_ValueError,
                "PyArray_DescrHash argument must be a type descriptor");
        return -1;
    }
    descr = (PyArray_Descr*)odescr;

    st = _PyArray_DescrHashImp(descr->descr, &hash);
    if (st) {
        return -1;
    }

    return hash;
}
