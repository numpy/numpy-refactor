/*
 * Python Universal Functions Object -- Math for all types, plus fast
 * arrays math
 *
 * Full description
 *
 * This supports mathematical (and Boolean) functions on arrays and other python
 * objects.  Math on large arrays of basic C types is rather efficient.
 *
 * Travis E. Oliphant  2005, 2006 oliphant@ee.byu.edu (oliphant.travis@ieee.org)
 * Brigham Young University
 *
 * based on the
 *
 * Original Implementation:
 * Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu
 *
 * with inspiration and code from
 * Numarray
 * Space Science Telescope Institute
 * J. Todd Miller
 * Perry Greenfield
 * Rick White
 *
 */
#define _UMATHMODULE


#define NPY_ALLOW_THREADS 1


#include <Python.h>

#include "numpy_config.h"
#ifdef ENABLE_SEPARATE_COMPILATION
#define PY_ARRAY_UNIQUE_SYMBOL _npy_umathmodule_ARRAY_API
#define NO_IMPORT_ARRAY
#endif

#include "npy_api.h"
#include "npy_iterators.h"
#include <npy_dict.h>
#include <npy_ufunc_object.h>

#include "numpy/noprefix.h"
#include "numpy/ufuncobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy_3kcompat.h"

#include "ufunc_object.h"

#define USE_USE_DEFAULTS 1

#define ASSIGN_TO_PYARRAY(pya, arr)             \
    do {                                        \
        NpyArray* a_ = (arr);                   \
        if (a_ == NULL) {                       \
            pya = NULL;                         \
        } else {                                \
            pya = Npy_INTERFACE(a_);            \
            Py_INCREF(pya);                     \
            Npy_DECREF(a_);                    \
        }                                       \
    } while (0)

/* ---------------------------------------------------------------- */


/* TODO: Following three functions are duplicates of npy_ufunc_object.c and should go away 
   once we can build a shared library of the core code. */
static int npy_compare_ints(const void *a, const void *b)
{
    if (a < b ) return -1;
    else if ( a > b ) return 1;
    return 0;
}

static unsigned long npy_hash_int(const void *a)
{
    return (unsigned long)a;          /* Size change is safe - just a hash function */
}

/*
 * This frees the linked-list structure when the CObject
 * is destroyed (removed from the internal dictionary)
 */
static void npy_free_loop1d_list(PyUFunc_Loop1d *data)
{
    while (data != NULL) {
        PyUFunc_Loop1d *next = data->next;
        NpyArray_free(data->arg_types);
        NpyArray_free(data);
        data = next;
    }
}


NpyDict *npy_create_userloops_table()
{
    NpyDict *new = NpyDict_CreateTable(7);  /* 7 is a guess at enough */
    NpyDict_SetKeyComparisonFunction(new, (int (*)(const void *, const void *))npy_compare_ints);
    NpyDict_SetHashFunction(new, (unsigned long (*)(const void *))npy_hash_int);
    NpyDict_SetDeallocationFunctions(new, NULL, (void (*)(void *))npy_free_loop1d_list); 
    return new;
}

/* TODO: END of duplicated code */






static int
_does_loop_use_arrays(void *data);

/*
 * fpstatus is the ufunc_formatted hardware status
 * errmask is the handling mask specified by the user.
 * errobj is a Python object with (string, callable object or None)
 * or NULL
 */

/*
 * 2. for each of the flags
 * determine whether to ignore, warn, raise error, or call Python function.
 * If ignore, do nothing
 * If warn, print a warning and continue
 * If raise return an error
 * If call, call a user-defined function with string
 */

static int
_error_handler(int method, PyObject *errobj, char *errtype, int retstatus, int *first)
{
    PyObject *pyfunc, *ret, *args;
    char *name = PyBytes_AS_STRING(PyTuple_GET_ITEM(errobj,0));
    char msg[100];
    ALLOW_C_API_DEF;

    ALLOW_C_API;
    switch(method) {
    case UFUNC_ERR_WARN:
        PyOS_snprintf(msg, sizeof(msg), "%s encountered in %s", errtype, name);
        if (PyErr_Warn(PyExc_RuntimeWarning, msg) < 0) {
            goto fail;
        }
        break;
    case UFUNC_ERR_RAISE:
        PyErr_Format(PyExc_FloatingPointError, "%s encountered in %s",
                errtype, name);
        goto fail;
    case UFUNC_ERR_CALL:
        pyfunc = PyTuple_GET_ITEM(errobj, 1);
        if (pyfunc == Py_None) {
            PyErr_Format(PyExc_NameError,
                    "python callback specified for %s (in " \
                    " %s) but no function found.",
                    errtype, name);
            goto fail;
        }
        args = Py_BuildValue("NN", PyUString_FromString(errtype),
                PyInt_FromLong((long) retstatus));
        if (args == NULL) {
            goto fail;
        }
        ret = PyObject_CallObject(pyfunc, args);
        Py_DECREF(args);
        if (ret == NULL) {
            goto fail;
        }
        Py_DECREF(ret);
        break;
    case UFUNC_ERR_PRINT:
        if (*first) {
            fprintf(stderr, "Warning: %s encountered in %s\n", errtype, name);
            *first = 0;
        }
        break;
    case UFUNC_ERR_LOG:
        if (first) {
            *first = 0;
            pyfunc = PyTuple_GET_ITEM(errobj, 1);
            if (pyfunc == Py_None) {
                PyErr_Format(PyExc_NameError,
                        "log specified for %s (in %s) but no " \
                        "object with write method found.",
                        errtype, name);
                goto fail;
            }
            PyOS_snprintf(msg, sizeof(msg),
                    "Warning: %s encountered in %s\n", errtype, name);
            ret = PyObject_CallMethod(pyfunc, "write", "s", msg);
            if (ret == NULL) {
                goto fail;
            }
            Py_DECREF(ret);
        }
        break;
    }
    DISABLE_C_API;
    return 0;

fail:
    DISABLE_C_API;
    return -1;
}


/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_getfperr(void)
{
    int retstatus;
    UFUNC_CHECK_STATUS(retstatus);
    return retstatus;
}

#define HANDLEIT(NAME, str) {if (retstatus & UFUNC_FPE_##NAME) {        \
            handle = errmask & UFUNC_MASK_##NAME;                       \
            if (handle &&                                               \
                _error_handler(handle >> UFUNC_SHIFT_##NAME,            \
                               errobj, str, retstatus, first) < 0)      \
                return -1;                                              \
        }}

/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_handlefperr(int errmask, PyObject *errobj, int retstatus, int *first)
{
    int handle;
    if (errmask && retstatus) {
        HANDLEIT(DIVIDEBYZERO, "divide by zero");
        HANDLEIT(OVERFLOW, "overflow");
        HANDLEIT(UNDERFLOW, "underflow");
        HANDLEIT(INVALID, "invalid value");
    }
    return 0;
}

#undef HANDLEIT


/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_checkfperr(int errmask, PyObject *errobj, int *first)
{
    int retstatus;

    /* 1. check hardware flag --- this is platform dependent code */
    retstatus = PyUFunc_getfperr();
    return PyUFunc_handlefperr(errmask, errobj, retstatus, first);
}


/* Checking the status flag clears it */
/*UFUNC_API*/
NPY_NO_EXPORT void
PyUFunc_clearfperr()
{
    PyUFunc_getfperr();
}


#define NO_UFUNCLOOP        0
#define ZERO_EL_REDUCELOOP  0
#define ONE_UFUNCLOOP       1
#define ONE_EL_REDUCELOOP   1
#define NOBUFFER_UFUNCLOOP  2
#define NOBUFFER_REDUCELOOP 2
#define BUFFER_UFUNCLOOP    3
#define BUFFER_REDUCELOOP   3
#define SIGNATURE_NOBUFFER_UFUNCLOOP 4


static char
_lowest_type(char intype) 
{
    /* TODO: Ready to move */
    switch(intype) {
        /* case PyArray_BYTE */
    case NPY_SHORT:
    case NPY_INT:
    case NPY_LONG:
    case NPY_LONGLONG:
    case NPY_DATETIME:
    case NPY_TIMEDELTA:
        return NPY_BYTE;
        /* case NPY_UBYTE */
    case NPY_USHORT:
    case NPY_UINT:
    case NPY_ULONG:
    case NPY_ULONGLONG:
        return NPY_UBYTE;
        /* case PyArray_FLOAT:*/
    case NPY_DOUBLE:
    case NPY_LONGDOUBLE:
        return NPY_FLOAT;
        /* case PyArray_CFLOAT:*/
    case NPY_CDOUBLE:
    case NPY_CLONGDOUBLE:
        return NPY_CFLOAT;
    default:
        return intype;
    }
}

static char *_types_msg =  "function not supported for these types, "   \
    "and can't coerce safely to supported types";

/*
 * This function analyzes the input arguments
 * and determines an appropriate __array_prepare__ function to call
 * for the outputs.
 *
 * If an output argument is provided, then it is wrapped
 * with its own __array_prepare__ not with the one determined by
 * the input arguments.
 *
 * if the provided output argument is already an ndarray,
 * the wrapping function is None (which means no wrapping will
 * be done --- not even PyArray_Return).
 *
 * A NULL is placed in output_wrap for outputs that
 * should just have PyArray_Return called.
 */
static void
_find_array_prepare(PyObject *args, PyObject **output_wrap, int nin, int nout)
{
    Py_ssize_t nargs;
    int i;
    int np = 0;
    PyObject *with_wrap[NPY_MAXARGS], *wraps[NPY_MAXARGS];
    PyObject *obj, *wrap = NULL;

    nargs = PyTuple_GET_SIZE(args);
    for (i = 0; i < nin; i++) {
        obj = PyTuple_GET_ITEM(args, i);
        if (PyArray_CheckExact(obj) || PyArray_IsAnyScalar(obj)) {
            continue;
        }
        wrap = PyObject_GetAttrString(obj, "__array_prepare__");
        if (wrap) {
            if (PyCallable_Check(wrap)) {
                with_wrap[np] = obj;
                wraps[np] = wrap;
                ++np;
            }
            else {
                Py_DECREF(wrap);
                wrap = NULL;
            }
        }
        else {
            PyErr_Clear();
        }
    }
    if (np > 0) {
        /* If we have some wraps defined, find the one of highest priority */
        wrap = wraps[0];
        if (np > 1) {
            double maxpriority = PyArray_GetPriority(with_wrap[0],
                        PyArray_SUBTYPE_PRIORITY);
            for (i = 1; i < np; ++i) {
                double priority = PyArray_GetPriority(with_wrap[i],
                            PyArray_SUBTYPE_PRIORITY);
                if (priority > maxpriority) {
                    maxpriority = priority;
                    Py_DECREF(wrap);
                    wrap = wraps[i];
                }
                else {
                    Py_DECREF(wraps[i]);
                }
            }
        }
    }

    /*
     * Here wrap is the wrapping function determined from the
     * input arrays (could be NULL).
     *
     * For all the output arrays decide what to do.
     *
     * 1) Use the wrap function determined from the input arrays
     * This is the default if the output array is not
     * passed in.
     *
     * 2) Use the __array_prepare__ method of the output object.
     * This is special cased for
     * exact ndarray so that no PyArray_Return is
     * done in that case.
     */
    for (i = 0; i < nout; i++) {
        int j = nin + i;
        int incref = 1;
        output_wrap[i] = wrap;
        if (j < nargs) {
            obj = PyTuple_GET_ITEM(args, j);
            if (obj == Py_None) {
                continue;
            }
            if (PyArray_CheckExact(obj)) {
                output_wrap[i] = Py_None;
            }
            else {
                PyObject *owrap = PyObject_GetAttrString(obj,
                            "__array_prepare__");
                incref = 0;
                if (!(owrap) || !(PyCallable_Check(owrap))) {
                    Py_XDECREF(owrap);
                    owrap = wrap;
                    incref = 1;
                    PyErr_Clear();
                }
                output_wrap[i] = owrap;
            }
        }
        if (incref) {
            Py_XINCREF(output_wrap[i]);
        }
    }
    Py_XDECREF(wrap);
    return;
}

/*
 * Called for non-NULL user-defined functions.
 * The object should be a CObject pointing to a linked-list of functions
 * storing the function, data, and signature of all user-defined functions.
 * There must be a match with the input argument types or an error
 * will occur.
 */
static int
_find_matching_userloop(PyUFunc_Loop1d *funcdata, int *arg_types,
                        NPY_SCALARKIND *scalars,
                        NpyUFuncGenericFunction *function, void **data,
                        int nargs, int nin)
{
    /* TODO: Ready to move */
    int i;

    while (funcdata != NULL) {
        for (i = 0; i < nin; i++) {
            if (!NpyArray_CanCoerceScalar(arg_types[i],
                                          funcdata->arg_types[i],
                                          scalars[i]))
                break;
        }
        if (i == nin) {
            /* match found */
            *function = funcdata->func;
            *data = funcdata->data;
            /* Make sure actual arg_types supported by the loop are used */
            for (i = 0; i < nargs; i++) {
                arg_types[i] = funcdata->arg_types[i];
            }
            return 0;
        }
        funcdata = funcdata->next;
    }
    return -1;
}

/*
 * Converts type_tup into an array of typenums.
 */
static int
get_rtypenums(NpyUFuncObject *self, PyObject *type_tup, int *rtypenums)
{
    PyArray_Descr *dtype;
    int n, i;
    int nargs;
    int strtype = 0;

    nargs = self->nargs;

    if (PyTuple_Check(type_tup)) {
        n = PyTuple_GET_SIZE(type_tup);
        if (n != 1 && n != nargs) {
            PyErr_Format(PyExc_ValueError,
                         "a type-tuple must be specified " \
                         "of length 1 or %d for %s", nargs,
                         self->name ? self->name : "(unknown)");
            return -1;
        }
    }
    else if (PyString_Check(type_tup)) {
            Py_ssize_t slen;
            char *thestr;

            slen = PyString_GET_SIZE(type_tup);
            thestr = PyString_AS_STRING(type_tup);
            for (i = 0; i < slen - 2; i++) {
                if (thestr[i] == '-' && thestr[i+1] == '>') {
                    break;
                }
            }
            if (i < slen-2) {
                strtype = 1;
                n = slen - 2;
                if (i != self->nin
                    || slen - 2 - i != self->nout) {
                    PyErr_Format(PyExc_ValueError,
                                 "a type-string for %s, "   \
                                 "requires %d typecode(s) before " \
                                 "and %d after the -> sign",
                                 self->name ? self->name : "(unknown)",
                                 self->nin, self->nout);
                    return -1;
                }
            }
        }

    if (strtype) {
        char *ptr;
        ptr = PyString_AS_STRING(type_tup);
        i = 0;
        while (i < n) {
            if (*ptr == '-' || *ptr == '>') {
                ptr++;
                continue;
            }
            dtype = PyArray_DescrFromType((int) *ptr);
            if (dtype == NULL) {
                return -1;
            }
            rtypenums[i] = dtype->descr->type_num;
            Py_DECREF(dtype);
            ptr++;
            i++;
        }
    }
    else if (PyTuple_Check(type_tup)) {
        for (i = 0; i < n; i++) {
            if (PyArray_DescrConverter(PyTuple_GET_ITEM(type_tup, i),
                                       &dtype) == NPY_FAIL) {
                return -1;
            }
            rtypenums[i] = dtype->descr->type_num;
            Py_DECREF(dtype);
        }
    }
    else {
        if (PyArray_DescrConverter(type_tup, &dtype) == NPY_FAIL) {
            return -1;
        }
        rtypenums[0] = dtype->descr->type_num;
        Py_DECREF(dtype);
    }

    return 0;
}

/*
 * if only one type is specified then it is the "first" output data-type
 * and the first signature matching this output data-type is returned.
 *
 * if a tuple of types is specified then an exact match to the signature
 * is searched and it much match exactly or an error occurs
 */
static int
extract_specified_loop(NpyUFuncObject *self, int *arg_types,
                       NpyUFuncGenericFunction *function, void **data,
                       int *rtypenums, int userdef)
{
    /* TODO: Ready to move */
    Py_ssize_t n = 1;
    static char msg[] = "loop written to specified type(s) not found";
    int nargs;
    int i, j;

    nargs = self->nargs;
    if (userdef > 0) {
        /* search in the user-defined functions */
        PyUFunc_Loop1d *funcdata;

        funcdata = NpyDict_Get(self->userloops, (void *)(npy_intp)userdef);
        if (NULL == funcdata) {
            PyErr_SetString(PyExc_TypeError,
                            "user-defined type used in ufunc" \
                            " with no registered loops");
            return -1;
        }
        /*
         * extract the correct function
         * data and argtypes
         */
        while (funcdata != NULL) {
            if (n != 1) {
                for (i = 0; i < nargs; i++) {
                    if (rtypenums[i] != funcdata->arg_types[i]) {
                        break;
                    }
                }
            }
            else if (rtypenums[0] == funcdata->arg_types[self->nin]) {
                i = nargs;
            }
            else {
                i = -1;
            }
            if (i == nargs) {
                *function = funcdata->func;
                *data = funcdata->data;
                for(i = 0; i < nargs; i++) {
                    arg_types[i] = funcdata->arg_types[i];
                }
                return 0;
            }
            funcdata = funcdata->next;
        }
        PyErr_SetString(PyExc_TypeError, msg);
        return -1;
    }

    /* look for match in self->functions */
    for (j = 0; j < self->ntypes; j++) {
        if (n != 1) {
            for(i = 0; i < nargs; i++) {
                if (rtypenums[i] != self->types[j*nargs + i]) {
                    break;
                }
            }
        }
        else if (rtypenums[0] == self->types[j*nargs+self->nin]) {
            i = nargs;
        }
        else {
            i = -1;
        }
        if (i == nargs) {
            *function = self->functions[j];
            *data = self->data[j];
            for (i = 0; i < nargs; i++) {
                arg_types[i] = self->types[j*nargs+i];
            }
            return 0;
        }
    }
    PyErr_SetString(PyExc_TypeError, msg);

    return -1;
}


/*
 * Called to determine coercion
 * Can change arg_types.
 */
static int
select_types(NpyUFuncObject *self, int *arg_types,
             NpyUFuncGenericFunction *function, void **data,
             NPY_SCALARKIND *scalars,
             int *rtypenums)
{
    /* TODO: Ready to move */
    int i, j;
    char start_type;
    int userdef = -1;
    int userdef_ind = -1;

    if (self->userloops) {
        for(i = 0; i < self->nin; i++) {
            if (NpyTypeNum_ISUSERDEF(arg_types[i])) {
                userdef = arg_types[i];
                userdef_ind = i;
                break;
            }
        }
    }

    if (rtypenums != NULL)
        return extract_specified_loop(self, arg_types, function, data,
                                      rtypenums, userdef);

    if (userdef > 0) {
        int ret = -1;

        /*
         * Look through all the registered loops for all the user-defined
         * types to find a match.
         */
        while (ret == -1) {
            PyUFunc_Loop1d *funcdata;
            npy_intp userdefP;
            
            if (userdef_ind >= self->nin) {
                break;
            }
            userdef = arg_types[userdef_ind++];
            if (!(NpyTypeNum_ISUSERDEF(userdef))) {
                continue;
            }
            userdefP = (npy_intp)userdef;
            funcdata = NpyDict_Get(self->userloops, (void *)userdefP);
            /*
             * extract the correct function
             * data and argtypes for this user-defined type.
             */
            ret = _find_matching_userloop(funcdata, arg_types, scalars,
                                          function, data, self->nargs,
                                          self->nin);
        }
        if (ret == 0) {
            return ret;
        }
        PyErr_SetString(PyExc_TypeError, _types_msg);
        return ret;
    }

    start_type = arg_types[0];
    /*
     * If the first argument is a scalar we need to place
     * the start type as the lowest type in the class
     */
    if (scalars[0] != PyArray_NOSCALAR) {
        start_type = _lowest_type(start_type);
    }

    i = 0;
    while (i < self->ntypes && start_type > self->types[i*self->nargs]) {
        i++;
    }
    for (; i < self->ntypes; i++) {
        for (j = 0; j < self->nin; j++) {
            if (!NpyArray_CanCoerceScalar(arg_types[j],
                                          self->types[i*self->nargs + j],
                                          scalars[j]))
                break;
        }
        if (j == self->nin) {
            break;
        }
    }
    if (i >= self->ntypes) {
        PyErr_SetString(PyExc_TypeError, _types_msg);
        return -1;
    }
    for (j = 0; j < self->nargs; j++) {
        arg_types[j] = self->types[i*self->nargs+j];
    }
    if (self->data) {
        *data = self->data[i];
    }
    else {
        *data = NULL;
    }
    *function = self->functions[i];

    return 0;
}

#if USE_USE_DEFAULTS==1
static int PyUFunc_NUM_NODEFAULTS = 0;
#endif
static PyObject *PyUFunc_PYVALS_NAME = NULL;


static int
_extract_pyvals(PyObject *ref, char *name, int *bufsize,
                int *errmask, PyObject **errobj)
{
    PyObject *retval;

    *errobj = NULL;
    if (!PyList_Check(ref) || (PyList_GET_SIZE(ref)!=3)) {
        PyErr_Format(PyExc_TypeError, "%s must be a length 3 list.",
                     UFUNC_PYVALS_NAME);
        return -1;
    }

    *bufsize = PyInt_AsLong(PyList_GET_ITEM(ref, 0));
    if ((*bufsize == -1) && PyErr_Occurred()) {
        return -1;
    }
    if ((*bufsize < PyArray_MIN_BUFSIZE)
        || (*bufsize > PyArray_MAX_BUFSIZE)
        || (*bufsize % 16 != 0)) {
        PyErr_Format(PyExc_ValueError,
                     "buffer size (%d) is not in range "
                     "(%"INTP_FMT" - %"INTP_FMT") or not a multiple of 16",
                     *bufsize, (intp) PyArray_MIN_BUFSIZE,
                     (intp) PyArray_MAX_BUFSIZE);
        return -1;
    }

    *errmask = PyInt_AsLong(PyList_GET_ITEM(ref, 1));
    if (*errmask < 0) {
        if (PyErr_Occurred()) {
            return -1;
        }
        PyErr_Format(PyExc_ValueError,
                     "invalid error mask (%d)",
                     *errmask);
        return -1;
    }

    retval = PyList_GET_ITEM(ref, 2);
    if (retval != Py_None && !PyCallable_Check(retval)) {
        PyObject *temp;
        temp = PyObject_GetAttrString(retval, "write");
        if (temp == NULL || !PyCallable_Check(temp)) {
            PyErr_SetString(PyExc_TypeError,
                            "python object must be callable or have " \
                            "a callable write method");
            Py_XDECREF(temp);
            return -1;
        }
        Py_DECREF(temp);
    }

    *errobj = Py_BuildValue("NO", PyBytes_FromString(name), retval);
    if (*errobj == NULL) {
        return -1;
    }
    return 0;
}



/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_GetPyValues(char *name, int *bufsize, int *errmask, PyObject **errobj)
{
    PyObject *thedict;
    PyObject *ref = NULL;

#if USE_USE_DEFAULTS==1
    if (PyUFunc_NUM_NODEFAULTS != 0) {
#endif
        if (PyUFunc_PYVALS_NAME == NULL) {
            PyUFunc_PYVALS_NAME = PyUString_InternFromString(UFUNC_PYVALS_NAME);
        }
        thedict = PyThreadState_GetDict();
        if (thedict == NULL) {
            thedict = PyEval_GetBuiltins();
        }
        ref = PyDict_GetItem(thedict, PyUFunc_PYVALS_NAME);
#if USE_USE_DEFAULTS==1
    }
#endif
    if (ref == NULL) {
        *errmask = UFUNC_ERR_DEFAULT;
        *errobj = Py_BuildValue("NO", PyBytes_FromString(name), Py_None);
        *bufsize = PyArray_BUFSIZE;
        return 0;
    }
    return _extract_pyvals(ref, name, bufsize, errmask, errobj);
}

/*
 * Create copies for any arrays that are less than loop->bufsize
 * in total size (or core_enabled) and are mis-behaved or in need
 * of casting.
 */
static int
_create_copies(NpyUFuncLoopObject *loop, int *arg_types, PyArrayObject **mps)
{
    int nin = loop->ufunc->nin;
    int i;
    intp size;
    PyObject *new;
    NpyArray_Descr *ntype;
    NpyArray_Descr *atype;

    for (i = 0; i < nin; i++) {
        size = PyArray_SIZE(mps[i]);
        /*
         * if the type of mps[i] is equivalent to arg_types[i]
         * then set arg_types[i] equal to type of mps[i] for later checking....
         */
        if (PyArray_TYPE(mps[i]) != arg_types[i]) {
            ntype = PyArray_ARRAY(mps[i])->descr;
            atype = NpyArray_DescrFromType(arg_types[i]);
            if (NpyArray_EquivTypes(atype, ntype)) {
                arg_types[i] = ntype->type_num;
            }
            Npy_DECREF(atype);
        }
        if (size < loop->bufsize || loop->ufunc->core_enabled) {
            if (!(PyArray_ISBEHAVED_RO(mps[i]))
                || PyArray_TYPE(mps[i]) != arg_types[i]) {
                PyArray_Descr *ntypeWrap;
                ntype = NpyArray_DescrFromType(arg_types[i]);
                
                /* Move reference to interface. */
                PyArray_Descr_REF_FROM_CORE(ntype, ntypeWrap);
                new = PyArray_FromAny((PyObject *)mps[i],
                                      ntypeWrap, 0, 0,
                                      FORCECAST | ALIGNED, NULL);
                if (new == NULL) {
                        return -1;
                }
                Py_DECREF(mps[i]);
                mps[i] = (PyArrayObject *)new;
            }
        }
    }
    return 0;
}

#define _GETATTR_(str, rstr) do {if (strcmp(name, #str) == 0)     \
        return PyObject_HasAttrString(op, "__" #rstr "__");} while (0);

static int
_has_reflected_op(PyObject *op, char *name)
{
    _GETATTR_(add, radd);
    _GETATTR_(subtract, rsub);
    _GETATTR_(multiply, rmul);
    _GETATTR_(divide, rdiv);
    _GETATTR_(true_divide, rtruediv);
    _GETATTR_(floor_divide, rfloordiv);
    _GETATTR_(remainder, rmod);
    _GETATTR_(power, rpow);
    _GETATTR_(left_shift, rlshift);
    _GETATTR_(right_shift, rrshift);
    _GETATTR_(bitwise_and, rand);
    _GETATTR_(bitwise_xor, rxor);
    _GETATTR_(bitwise_or, ror);
    return 0;
}

#undef _GETATTR_


/* Return the position of next non-white-space char in the string */
static int
_next_non_white_space(const char* str, int offset)
{
    int ret = offset;
    while (str[ret] == ' ' || str[ret] == '\t') {
        ret++;
    }
    return ret;
}

static int
_is_alpha_underscore(char ch)
{
    return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch == '_';
}

static int
_is_alnum_underscore(char ch)
{
    return _is_alpha_underscore(ch) || (ch >= '0' && ch <= '9');
}

/*
 * Return the ending position of a variable name
 */
static int
_get_end_of_name(const char* str, int offset)
{
    int ret = offset;
    while (_is_alnum_underscore(str[ret])) {
        ret++;
    }
    return ret;
}

/*
 * Returns 1 if the dimension names pointed by s1 and s2 are the same,
 * otherwise returns 0.
 */
static int
_is_same_name(const char* s1, const char* s2)
{
    while (_is_alnum_underscore(*s1) && _is_alnum_underscore(*s2)) {
        if (*s1 != *s2) {
            return 0;
        }
        s1++;
        s2++;
    }
    return !_is_alnum_underscore(*s1) && !_is_alnum_underscore(*s2);
}

/*
 * Sets core_num_dim_ix, core_num_dims, core_dim_ixs, core_offsets,
 * and core_signature in PyUFuncObject "self".  Returns 0 unless an
 * error occured.
 */
static int
_parse_signature(NpyUFuncObject *self, const char *signature)
{
    /* TODO: Ready to move */
    size_t len;
    char const **var_names;
    int nd = 0;             /* number of dimension of the current argument */
    int cur_arg = 0;        /* index into core_num_dims&core_offsets */
    int cur_core_dim = 0;   /* index into core_dim_ixs */
    int i = 0;
    char *parse_error = NULL;

    if (signature == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "_parse_signature with NULL signature");
        return -1;
    }

    len = strlen(signature);
    self->core_signature = malloc(sizeof(char) * (len+1));
    if (self->core_signature) {
        strcpy(self->core_signature, signature);
    }
    /* Allocate sufficient memory to store pointers to all dimension names */
    var_names = malloc(sizeof(char const*) * len);
    if (var_names == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    self->core_enabled = 1;
    self->core_num_dim_ix = 0;
    self->core_num_dims = malloc(sizeof(int) * self->nargs);
    self->core_dim_ixs = malloc(sizeof(int) * len); /* shrink this later */
    self->core_offsets = malloc(sizeof(int) * self->nargs);
    if (self->core_num_dims == NULL || self->core_dim_ixs == NULL
        || self->core_offsets == NULL) {
        PyErr_NoMemory();
        goto fail;
    }

    i = _next_non_white_space(signature, 0);
    while (signature[i] != '\0') {
        /* loop over input/output arguments */
        if (cur_arg == self->nin) {
            /* expect "->" */
            if (signature[i] != '-' || signature[i+1] != '>') {
                parse_error = "expect '->'";
                goto fail;
            }
            i = _next_non_white_space(signature, i + 2);
        }

        /*
         * parse core dimensions of one argument,
         * e.g. "()", "(i)", or "(i,j)"
         */
        if (signature[i] != '(') {
            parse_error = "expect '('";
            goto fail;
        }
        i = _next_non_white_space(signature, i + 1);
        while (signature[i] != ')') {
            /* loop over core dimensions */
            int j = 0;
            if (!_is_alpha_underscore(signature[i])) {
                parse_error = "expect dimension name";
                goto fail;
            }
            while (j < self->core_num_dim_ix) {
                if (_is_same_name(signature+i, var_names[j])) {
                    break;
                }
                j++;
            }
            if (j >= self->core_num_dim_ix) {
                var_names[j] = signature+i;
                self->core_num_dim_ix++;
            }
            self->core_dim_ixs[cur_core_dim] = j;
            cur_core_dim++;
            nd++;
            i = _get_end_of_name(signature, i);
            i = _next_non_white_space(signature, i);
            if (signature[i] != ',' && signature[i] != ')') {
                parse_error = "expect ',' or ')'";
                goto fail;
            }
            if (signature[i] == ',')
            {
                i = _next_non_white_space(signature, i + 1);
                if (signature[i] == ')') {
                    parse_error = "',' must not be followed by ')'";
                    goto fail;
                }
            }
        }
        self->core_num_dims[cur_arg] = nd;
        self->core_offsets[cur_arg] = cur_core_dim-nd;
        cur_arg++;
        nd = 0;

        i = _next_non_white_space(signature, i + 1);
        if (cur_arg != self->nin && cur_arg != self->nargs) {
            /*
             * The list of input arguments (or output arguments) was
             * only read partially
             */
            if (signature[i] != ',') {
                parse_error = "expect ','";
                goto fail;
            }
            i = _next_non_white_space(signature, i + 1);
        }
    }
    if (cur_arg != self->nargs) {
        parse_error = "incomplete signature: not all arguments found";
        goto fail;
    }
    self->core_dim_ixs = _pya_realloc(self->core_dim_ixs,
            sizeof(int)*cur_core_dim);
    /* check for trivial core-signature, e.g. "(),()->()" */
    if (cur_core_dim == 0) {
        self->core_enabled = 0;
    }
    _pya_free((void*)var_names);
    return 0;

fail:
    _pya_free((void*)var_names);
    if (parse_error) {
        char *buf = malloc(sizeof(char) * (len + 200));
        if (buf) {
            sprintf(buf, "%s at position %d in \"%s\"",
                    parse_error, i, signature);
            PyErr_SetString(PyExc_ValueError, signature);
            _pya_free(buf);
        }
        else {
            PyErr_NoMemory();
        }
    }
    return -1;
}

/*
 * Concatenate the loop and core dimensions of
 * PyArrayMultiIterObject's iarg-th argument, to recover a full
 * dimension array (used for output arguments).
 */
static npy_intp*
_compute_output_dims(NpyUFuncLoopObject *loop, int iarg,
                     int *out_nd, npy_intp *tmp_dims)
{
    /* TODO: Ready to move */
    int i;
    NpyUFuncObject *ufunc = loop->ufunc;
    if (ufunc->core_enabled == 0) {
        /* case of ufunc with trivial core-signature */
        *out_nd = loop->iter->nd;
        return loop->iter->dimensions;
    }

    *out_nd = loop->iter->nd + ufunc->core_num_dims[iarg];
    if (*out_nd > NPY_MAXARGS) {
        PyErr_SetString(PyExc_ValueError,
                        "dimension of output variable exceeds limit");
        return NULL;
    }

    /* copy loop dimensions */
    memcpy(tmp_dims, loop->iter->dimensions, sizeof(npy_intp) * loop->iter->nd);

    /* copy core dimension */
    for (i = 0; i < ufunc->core_num_dims[iarg]; i++) {
        tmp_dims[loop->iter->nd + i] = loop->core_dim_sizes[1 +
            ufunc->core_dim_ixs[ufunc->core_offsets[iarg] + i]];
    }
    return tmp_dims;
}

/* Check and set core_dim_sizes and core_strides for the i-th argument. */
static int
_compute_dimension_size(NpyUFuncLoopObject *loop, PyArrayObject **mps, int i)
{
    NpyUFuncObject *ufunc = loop->ufunc;
    int j = ufunc->core_offsets[i];
    int k = PyArray_NDIM(mps[i]) - ufunc->core_num_dims[i];
    int ind;
    
    for (ind = 0; ind < ufunc->core_num_dims[i]; ind++, j++, k++) {
        npy_intp dim = k < 0 ? 1 : PyArray_DIM(mps[i], k);
        /* First element of core_dim_sizes will be used for looping */
        int dim_ix = ufunc->core_dim_ixs[j] + 1;
        if (loop->core_dim_sizes[dim_ix] == 1) {
            /* broadcast core dimension  */
            loop->core_dim_sizes[dim_ix] = dim;
        }
        else if (dim != 1 && dim != loop->core_dim_sizes[dim_ix]) {
            PyErr_SetString(PyExc_ValueError, "core dimensions mismatch");
            return -1;
        }
        /* First ufunc->nargs elements will be used for looping */
        loop->core_strides[ufunc->nargs + j] =
            dim == 1 ? 0 : PyArray_STRIDE(mps[i], k);
    }
    return 0;
}

/* Return a view of array "ap" with "core_nd" dimensions cut from tail. */
static PyArrayObject *
_trunc_coredim(PyArrayObject *ap, int core_nd)
{
    PyArrayObject *ret;
    NpyArray *arr;
    int nd = PyArray_NDIM(ap) - core_nd;

    if (nd < 0) {
        nd = 0;
    }
    /* The following code is basically taken from PyArray_Transpose */
    /* NewFromDescr will steal this reference */
    Npy_INCREF(PyArray_DESCR(ap));
    arr = NpyArray_NewView(PyArray_DESCR(ap),
                           nd, PyArray_DIMS(ap),
                           PyArray_STRIDES(ap),
                           PyArray_ARRAY(ap), 0, 
                           NPY_FALSE);
    if (arr == NULL) {
        return NULL;
    }
    ret = Npy_INTERFACE(arr);
    Py_INCREF(ret);
    Npy_DECREF(arr);
    return ret;
}

static int
convert_args(NpyUFuncObject *self, PyObject* args, PyArrayObject **mps)
{
    PyObject *obj, *context;
    Py_ssize_t nargs;
    int i;
    int result;

    /* Check number of arguments */
    nargs = PyTuple_Size(args);
    if ((nargs < self->nin) || (nargs > self->nargs)) {
        PyErr_SetString(PyExc_ValueError, "invalid number of arguments");
        return -1;
    }

    /* Clear mps. */
    for (i=0; i < self->nargs; i++) {
        mps[i] = NULL;
    }

    /* Convert input arguments to arrays. */
    for (i=0; i < self->nin; i++) {
        obj = PyTuple_GET_ITEM(args,i);
        if (!PyArray_Check(obj) && !PyArray_IsScalar(obj, Generic)) {
            context = Py_BuildValue("OOi", Npy_INTERFACE(self), args, i);
        }
        else {
            context = NULL;
        }
        mps[i] = (PyArrayObject *)PyArray_FromAny(obj, NULL, 0, 0, 0, context);
        Py_XDECREF(context);
        if (mps[i] == NULL) {
            result = -1;
            goto fail;
        }
    }


    /* Get any return arguments */
    for (i = self->nin; i < nargs; i++) {
        obj = PyTuple_GET_ITEM(args, i);
        if (obj == Py_None) {
            mps[i] = NULL;
        } else if (PyArray_Check(obj)) {
            mps[i] = (PyArrayObject *)obj;
            Py_INCREF(obj);
        } else if (PyArrayIter_Check(obj)) {
            PyObject *new = PyObject_CallMethod(obj, "__array__", NULL);
            if (new == NULL) {
                result = -1;
                goto fail;
            } else if (!PyArray_Check(new)) {
                PyErr_SetString(PyExc_TypeError,
                                "__array__ must return an array.");
                Py_DECREF(new);
                result = -1;
                goto fail;
            } else {
                mps[i] = (PyArrayObject *)new;
            }
        } else {
            PyErr_SetString(PyExc_TypeError,
                            "return arrays must be "    \
                            "of ArrayType");
            result = -1;
            goto fail;
        }

    }


    return 0;

 fail:
    /* DECREF mps. */
    for (i=0; i < self->nargs; i++) {
        Py_XDECREF(mps[i]);
        mps[i] = NULL;
    }
    return result;
}

typedef int (*prepare_outputs_func)(PyUFuncObject* self, PyArrayObject **mps,
                                    void* data);

static Py_ssize_t
construct_arrays(NpyUFuncLoopObject *loop, Py_ssize_t nargs, PyArrayObject **mps,
                 int *rtypenums, 
                 prepare_outputs_func prepare, void* prepare_data)
{
    int i;
    int arg_types[NPY_MAXARGS];
    PyArray_SCALARKIND scalars[NPY_MAXARGS];
    PyArray_SCALARKIND maxarrkind, maxsckind, new;
    NpyUFuncObject *self = loop->ufunc;
    Bool allscalars = TRUE;
    PyTypeObject *subtype = &PyArray_Type;
    int flexible = 0;
    int object = 0;

    npy_intp temp_dims[NPY_MAXDIMS];
    npy_intp *out_dims;
    int out_nd;

    /* Get each input argument */
    maxarrkind = PyArray_NOSCALAR;
    maxsckind = PyArray_NOSCALAR;
    for(i = 0; i < self->nin; i++) {
        arg_types[i] = PyArray_TYPE(mps[i]);
        if (!flexible && NpyTypeNum_ISFLEXIBLE(arg_types[i])) {
            flexible = 1;
        }
        if (!object && NpyTypeNum_ISOBJECT(arg_types[i])) {
            object = 1;
        }
        /*
         * debug
         * fprintf(stderr, "array %d has reference %d\n", i,
         * (mps[i])->ob_refcnt);
         */

        /*
         * Scalars are 0-dimensional arrays at this point
         */

        /*
         * We need to keep track of whether or not scalars
         * are mixed with arrays of different kinds.
         */

        if (PyArray_NDIM(mps[i]) > 0) {
            scalars[i] = PyArray_NOSCALAR;
            allscalars = FALSE;
            new = PyArray_ScalarKind(arg_types[i], NULL);
            maxarrkind = NPY_MAX(new, maxarrkind);
        }
        else {
            scalars[i] = PyArray_ScalarKind(arg_types[i], &(mps[i]));
            maxsckind = NPY_MAX(scalars[i], maxsckind);
        }
    }

    /* We don't do strings */
    if (flexible && !object) {
        loop->notimplemented = 1;
        return nargs;
    }

    /*
     * If everything is a scalar, or scalars mixed with arrays of
     * different kinds of lesser kinds then use normal coercion rules
     */
    if (allscalars || (maxsckind > maxarrkind)) {
        for (i = 0; i < self->nin; i++) {
            scalars[i] = PyArray_NOSCALAR;
        }
    }

    /* Select an appropriate function for these argument types. */
    if (select_types(loop->ufunc, arg_types, &(loop->function),
                     &(loop->funcdata), scalars, rtypenums) == -1) {
        return -1;
    }

    /*
     * Create copies for some of the arrays if they are small
     * enough and not already contiguous
     */
    if (_create_copies(loop, arg_types, mps) < 0) {
        return -1;
    }

    /*
     * Only use loop dimensions when constructing Iterator:
     * temporarily replace mps[i] (will be recovered below).
     */
    if (self->core_enabled) {
        for (i = 0; i < self->nin; i++) {
            PyArrayObject *ao;

            if (_compute_dimension_size(loop, mps, i) < 0) {
                return -1;
            }
            ao = _trunc_coredim(mps[i], self->core_num_dims[i]);
            if (ao == NULL) {
                return -1;
            }
            mps[i] = ao;
        }
    }

    /* Create Iterators for the Inputs */
    for (i = 0; i < self->nin; i++) {
        loop->iter->iters[i] = NpyArray_IterNew(PyArray_ARRAY(mps[i]));
        if (loop->iter->iters[i] == NULL) {
            return -1;
        }
    }

    /* Recover mps[i]. */
    if (self->core_enabled) {
        for (i = 0; i < self->nin; i++) {
            PyArrayObject *ao = mps[i];
            mps[i] = Npy_INTERFACE(PyArray_BASE_ARRAY(mps[i]));
            Py_DECREF(ao);
        }
    }

    /* Broadcast the result */
    loop->iter->numiter = self->nin;
    if (NpyArray_Broadcast(loop->iter) < 0) {
        return -1;
    }

    /* Get any return arguments */
    for (i = self->nin; i < nargs; i++) {
        if (self->core_enabled) {
            if (_compute_dimension_size(loop, mps, i) < 0) {
                return -1;
            }
        }
        out_dims = _compute_output_dims(loop, i, &out_nd, temp_dims);
        if (!out_dims) {
            return -1;
        }
        if (PyArray_NDIM(mps[i]) != out_nd
            || !PyArray_CompareLists(PyArray_DIMS(mps[i]), out_dims, out_nd)) {
            PyErr_SetString(PyExc_ValueError, "invalid return array shape");
            Py_DECREF(mps[i]);
            mps[i] = NULL;
            return -1;
        }
        if (!PyArray_ISWRITEABLE(mps[i])) {
            PyErr_SetString(PyExc_ValueError, "return array is not writeable");
            Py_DECREF(mps[i]);
            mps[i] = NULL;
            return -1;
        }
    }

    /* construct any missing return arrays and make output iterators */
    for(i = self->nin; i < self->nargs; i++) {
        NpyArray_Descr *ntype;

        if (mps[i] == NULL) {
            out_dims = _compute_output_dims(loop, i, &out_nd, temp_dims);
            if (!out_dims) {
                return -1;
            }
            mps[i] = (PyArrayObject *)PyArray_New(subtype,
                                                  out_nd,
                                                  out_dims,
                                                  arg_types[i],
                                                  NULL, NULL,
                                                  0, 0, NULL);
            if (mps[i] == NULL) {
                return -1;
            }
        }

        /*
         * reset types for outputs that are equivalent
         * -- no sense casting uselessly
         */
        else {
            if (PyArray_TYPE(mps[i]) != arg_types[i]) {
                NpyArray_Descr *atype;
                ntype = PyArray_DESCR(mps[i]);
                atype = NpyArray_DescrFromType(arg_types[i]);
                if (NpyArray_EquivTypes(atype, ntype)) {
                    arg_types[i] = ntype->type_num;
                }
                Npy_DECREF(atype);
            }

            /* still not the same -- or will we have to use buffers?*/
            if (PyArray_TYPE(mps[i]) != arg_types[i]
                || !PyArray_ISBEHAVED_RO(mps[i])) {
                if (loop->iter->size < loop->bufsize || self->core_enabled) {
                    PyArray_Descr *ntypeWrap;
                    PyObject *new;
                    /*
                     * Copy the array to a temporary copy
                     * and set the UPDATEIFCOPY flag
                     */
                    ntype = NpyArray_DescrFromType(arg_types[i]);
                    
                    /* Move reference to interface. */
                    PyArray_Descr_REF_FROM_CORE(ntype, ntypeWrap);
                    new = PyArray_FromAny((PyObject *)mps[i],
                                          ntypeWrap, 0, 0,
                                          FORCECAST | ALIGNED |
                                          UPDATEIFCOPY, NULL);
                    if (new == NULL) {
                        return -1;
                    }
                    Py_DECREF(mps[i]);
                    mps[i] = (PyArrayObject *)new;
                }
            }
        }

        if (self->core_enabled) {
            PyArrayObject *ao;

            /* computer for all output arguments, and set strides in "loop" */
            if (_compute_dimension_size(loop, mps, i) < 0) {
                return -1;
            }
            ao = _trunc_coredim(mps[i], self->core_num_dims[i]);
            if (ao == NULL) {
                return -1;
            }
            /* Temporarily modify mps[i] for constructing iterator. */
            mps[i] = ao;
        }

        loop->iter->iters[i] = NpyArray_IterNew(PyArray_ARRAY(mps[i]));
        if (loop->iter->iters[i] == NULL) {
            return -1;
        }

        /* Recover mps[i]. */
        if (self->core_enabled) {
            PyArrayObject *ao = mps[i];
            mps[i] = Npy_INTERFACE(PyArray_BASE_ARRAY(mps[i]));
            Py_DECREF(ao);
        }

    }

    /* wrap outputs */
    if (prepare) {
        if (prepare(PyUFunc_WRAP(self), mps, prepare_data) < 0) {
            return -1;
        }
    }

    /*
     * If any of different type, or misaligned or swapped
     * then must use buffers
     */
    loop->bufcnt = 0;
    loop->obj = 0;
    /* Determine looping method needed */
    loop->meth = NO_UFUNCLOOP;
    if (loop->iter->size == 0) {
        return nargs;
    }
    if (self->core_enabled) {
        loop->meth = SIGNATURE_NOBUFFER_UFUNCLOOP;
    }
    for (i = 0; i < self->nargs; i++) {
        loop->needbuffer[i] = 0;
        if (arg_types[i] != PyArray_TYPE(mps[i])
            || !PyArray_ISBEHAVED_RO(mps[i])) {
            if (self->core_enabled) {
                PyErr_SetString(PyExc_RuntimeError,
                                "never reached; copy should have been made");
                return -1;
            }
            loop->meth = BUFFER_UFUNCLOOP;
            loop->needbuffer[i] = 1;
        }
        if (!(loop->obj & UFUNC_OBJ_ISOBJECT)
            && ((PyArray_TYPE(mps[i]) == PyArray_OBJECT)
                    || (arg_types[i] == PyArray_OBJECT))) {
            loop->obj = UFUNC_OBJ_ISOBJECT|UFUNC_OBJ_NEEDS_API;
        }
        if (!(loop->obj & UFUNC_OBJ_NEEDS_API)
                && ((PyArray_TYPE(mps[i]) == PyArray_DATETIME)
                    || (PyArray_TYPE(mps[i]) == PyArray_TIMEDELTA)
                    || (arg_types[i] == PyArray_DATETIME)
                    || (arg_types[i] == PyArray_TIMEDELTA))) {
            loop->obj = UFUNC_OBJ_NEEDS_API;
        }
    }

    if (self->core_enabled && (loop->obj & UFUNC_OBJ_ISOBJECT)) {
        PyErr_SetString(PyExc_TypeError,
                        "Object type not allowed in ufunc with signature");
        return -1;
    }
    if (loop->meth == NO_UFUNCLOOP) {
        loop->meth = ONE_UFUNCLOOP;

        /* All correct type and BEHAVED */
        /* Check for non-uniform stridedness */
        for (i = 0; i < self->nargs; i++) {
            if (!(loop->iter->iters[i]->contiguous)) {
                /*
                 * May still have uniform stride
                 * if (broadcast result) <= 1-d
                 */
                if (PyArray_NDIM(mps[i]) != 0 &&                  \
                    (loop->iter->iters[i]->nd_m1 > 0)) {
                    loop->meth = NOBUFFER_UFUNCLOOP;
                    break;
                }
            }
        }
        if (loop->meth == ONE_UFUNCLOOP) {
            for (i = 0; i < self->nargs; i++) {
                loop->bufptr[i] = PyArray_BYTES(mps[i]);
            }
        }
    }

    loop->iter->numiter = self->nargs;

    /* Fill in steps  */
    if (loop->meth == SIGNATURE_NOBUFFER_UFUNCLOOP && loop->iter->nd == 0) {
        /* Use default core_strides */
    }
    else if (loop->meth != ONE_UFUNCLOOP) {
        int ldim;
        intp minsum;
        intp maxdim;
        NpyArrayIterObject *it;
        intp stride_sum[NPY_MAXDIMS];
        int j;

        /* Fix iterators */

        /*
         * Optimize axis the iteration takes place over
         *
         * The first thought was to have the loop go
         * over the largest dimension to minimize the number of loops
         *
         * However, on processors with slow memory bus and cache,
         * the slowest loops occur when the memory access occurs for
         * large strides.
         *
         * Thus, choose the axis for which strides of the last iterator is
         * smallest but non-zero.
         */
        for (i = 0; i < loop->iter->nd; i++) {
            stride_sum[i] = 0;
            for (j = 0; j < loop->iter->numiter; j++) {
                stride_sum[i] += loop->iter->iters[j]->strides[i];
            }
        }

        ldim = loop->iter->nd - 1;
        minsum = stride_sum[loop->iter->nd - 1];
        for (i = loop->iter->nd - 2; i >= 0; i--) {
            if (stride_sum[i] < minsum ) {
                ldim = i;
                minsum = stride_sum[i];
            }
        }
        maxdim = loop->iter->dimensions[ldim];
        loop->iter->size /= maxdim;
        loop->bufcnt = maxdim;
        loop->lastdim = ldim;

        /*
         * Fix the iterators so the inner loop occurs over the
         * largest dimensions -- This can be done by
         * setting the size to 1 in that dimension
         * (just in the iterators)
         */
        for (i = 0; i < loop->iter->numiter; i++) {
            it = loop->iter->iters[i];
            it->contiguous = 0;
            it->size /= (it->dims_m1[ldim] + 1);
            it->dims_m1[ldim] = 0;
            it->backstrides[ldim] = 0;

            /*
             * (won't fix factors because we
             * don't use PyArray_ITER_GOTO1D
             * so don't change them)
             *
             * Set the steps to the strides in that dimension
             */
            loop->steps[i] = it->strides[ldim];
        }

        /*
         * Set looping part of core_dim_sizes and core_strides.
         */
        if (loop->meth == SIGNATURE_NOBUFFER_UFUNCLOOP) {
            loop->core_dim_sizes[0] = maxdim;
            for (i = 0; i < self->nargs; i++) {
                loop->core_strides[i] = loop->steps[i];
            }
        }

        /*
         * fix up steps where we will be copying data to
         * buffers and calculate the ninnerloops and leftover
         * values -- if step size is already zero that is not changed...
         */
        if (loop->meth == BUFFER_UFUNCLOOP) {
            loop->leftover = maxdim % loop->bufsize;
            loop->ninnerloops = (maxdim / loop->bufsize) + 1;
            for (i = 0; i < self->nargs; i++) {
                if (loop->needbuffer[i] && loop->steps[i]) {
                    loop->steps[i] = PyArray_ITEMSIZE(mps[i]);
                }
                /* These are changed later if casting is needed */
            }
        }
    }
    else if (loop->meth == ONE_UFUNCLOOP) {
        /* uniformly-strided case */
        for (i = 0; i < self->nargs; i++) {
            if (PyArray_SIZE(mps[i]) == 1) {
                loop->steps[i] = 0;
            }
            else {
                loop->steps[i] = PyArray_STRIDE(mps[i], PyArray_NDIM(mps[i])-1);
            }
        }
    }


    /* Finally, create memory for buffers if we need them */

    /*
     * Buffers for scalars are specially made small -- scalars are
     * not copied multiple times
     */
    if (loop->meth == BUFFER_UFUNCLOOP) {
        int cnt = 0, cntcast = 0;
        int scnt = 0, scntcast = 0;
        char *castptr;
        char *bufptr;
        int last_was_scalar = 0;
        int last_cast_was_scalar = 0;
        int oldbufsize = 0;
        int oldsize = 0;
        int scbufsize = 4*sizeof(double);
        int memsize;
        NpyArray_Descr *descr;

        /* compute the element size */
        for (i = 0; i < self->nargs; i++) {
            if (!loop->needbuffer[i]) {
                continue;
            }
            if (arg_types[i] != PyArray_ARRAY(mps[i])->descr->type_num) {
                descr = NpyArray_DescrFromType(arg_types[i]);
                if (loop->steps[i]) {
                    cntcast += descr->elsize;
                }
                else {
                    scntcast += descr->elsize;
                }
                if (i < self->nin) {
                    loop->cast[i] = PyArray_GetCastFunc( PyArray_Descr_WRAP(PyArray_DESCR(mps[i])),
                                            arg_types[i]);
                }
                else {
                    loop->cast[i] = PyArray_GetCastFunc \
                        ( PyArray_Descr_WRAP(descr), PyArray_DESCR(mps[i])->type_num);
                }
                Npy_DECREF(descr);
                if (!loop->cast[i]) {
                    return -1;
                }
            }
            loop->swap[i] = !(PyArray_ISNOTSWAPPED(mps[i]));
            if (loop->steps[i]) {
                cnt += PyArray_ITEMSIZE(mps[i]);
            }
            else {
                scnt += PyArray_ITEMSIZE(mps[i]);
            }
        }
        memsize = loop->bufsize*(cnt+cntcast) + scbufsize*(scnt+scntcast);
        loop->buffer[0] = PyDataMem_NEW(memsize);

        /*
         * debug
         * fprintf(stderr, "Allocated buffer at %p of size %d, cnt=%d, cntcast=%d\n",
         *               loop->buffer[0], loop->bufsize * (cnt + cntcast), cnt, cntcast);
         */
        if (loop->buffer[0] == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        if (loop->obj & UFUNC_OBJ_ISOBJECT) {
            memset(loop->buffer[0], 0, memsize);
        }
        castptr = loop->buffer[0] + loop->bufsize*cnt + scbufsize*scnt;
        bufptr = loop->buffer[0];
        loop->objfunc = 0;
        for (i = 0; i < self->nargs; i++) {
            if (!loop->needbuffer[i]) {
                continue;
            }
            loop->buffer[i] = bufptr + (last_was_scalar ? scbufsize :
                                        loop->bufsize)*oldbufsize;
            last_was_scalar = (loop->steps[i] == 0);
            bufptr = loop->buffer[i];
            oldbufsize = PyArray_ITEMSIZE(mps[i]);
            /* fprintf(stderr, "buffer[%d] = %p\n", i, loop->buffer[i]); */
            if (loop->cast[i]) {
                NpyArray_Descr *descr;
                loop->castbuf[i] = castptr + (last_cast_was_scalar ? scbufsize :
                                              loop->bufsize)*oldsize;
                last_cast_was_scalar = last_was_scalar;
                /* fprintf(stderr, "castbuf[%d] = %p\n", i, loop->castbuf[i]); */
                descr = NpyArray_DescrFromType(arg_types[i]);
                oldsize = descr->elsize;
                Npy_DECREF(descr);
                loop->bufptr[i] = loop->castbuf[i];
                castptr = loop->castbuf[i];
                if (loop->steps[i]) {
                    loop->steps[i] = oldsize;
                }
            }
            else {
                loop->bufptr[i] = loop->buffer[i];
            }
            if (!loop->objfunc && (loop->obj & UFUNC_OBJ_ISOBJECT)) {
                if (arg_types[i] == PyArray_OBJECT) {
                    loop->objfunc = 1;
                }
            }
        }
    }

    if (_does_loop_use_arrays(loop->funcdata)) {
        loop->funcdata = (void*)mps;
    }

    return nargs;
}

static void
ufuncreduce_dealloc(NpyUFuncReduceObject *self)
{
    if (self->ufunc) {
        Npy_XDECREF(self->it);
        Npy_XDECREF(self->rit);
        Npy_XDECREF(self->ret);
        Py_XDECREF(self->errobj);
        Npy_XDECREF(self->decref_arr);
        if (self->buffer) {
            NpyDataMem_FREE(self->buffer);
        }
        Npy_DECREF(self->ufunc);
    }
    self->magic_number = NPY_INVALID_MAGIC;
    free(self);
}

static void
ufuncloop_dealloc(NpyUFuncLoopObject *self)
{
    /* TODO: Ready to move */
    if (self->ufunc != NULL) {
        if (self->core_dim_sizes) {
            free(self->core_dim_sizes);
        }
        if (self->core_strides) {
            free(self->core_strides);
        }
        self->iter->numiter = self->ufunc->nargs;
        Npy_DECREF(self->iter);
        if (self->buffer[0]) {
            NpyDataMem_FREE(self->buffer[0]);
        }
        Py_XDECREF(self->errobj);
        Npy_DECREF(self->ufunc);
    }
    self->magic_number = NPY_INVALID_MAGIC;
    free(self);
}

static NpyUFuncLoopObject *
construct_loop(NpyUFuncObject *self)
{
    /* TODO: Ready to move */
    NpyUFuncLoopObject *loop;
    int i;
    char *name;

    if (self == NULL) {
        PyErr_SetString(PyExc_ValueError, "function not supported");
        return NULL;
    }
    if ((loop = malloc(sizeof(NpyUFuncLoopObject))) == NULL) {
        PyErr_NoMemory();
        return loop;
    }

    loop->iter = NpyArray_MultiIterNew();
    if (loop->iter == NULL) {
        free(loop);
        return NULL;
    }

    loop->iter->index = 0;
    loop->iter->numiter = self->nargs;
    loop->ufunc = self;
    Npy_INCREF(loop->ufunc);
    loop->buffer[0] = NULL;
    for (i = 0; i < self->nargs; i++) {
        loop->iter->iters[i] = NULL;
        loop->cast[i] = NULL;
    }
    loop->errobj = NULL;
    loop->notimplemented = 0;
    loop->first = 1;
    loop->core_dim_sizes = NULL;
    loop->core_strides = NULL;

    if (self->core_enabled) {
        int num_dim_ix = 1 + self->core_num_dim_ix;
        int nstrides = self->nargs 
                        + self->core_offsets[self->nargs - 1]
                        + self->core_num_dims[self->nargs - 1];
        loop->core_dim_sizes = malloc(sizeof(npy_intp)*num_dim_ix);
        loop->core_strides = malloc(sizeof(npy_intp)*nstrides);
        if (loop->core_dim_sizes == NULL || loop->core_strides == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        memset(loop->core_strides, 0, sizeof(npy_intp) * nstrides);
        for (i = 0; i < num_dim_ix; i++) {
            loop->core_dim_sizes[i] = 1;
        }
    }
    name = self->name ? self->name : "";

    return loop;

fail:
    ufuncloop_dealloc(loop);
    return NULL;
}


/*
  static void
  _printbytebuf(NpyUFuncLoopObject *loop, int bufnum)
  {
  int i;

  fprintf(stderr, "Printing byte buffer %d\n", bufnum);
  for(i=0; i<loop->bufcnt; i++) {
  fprintf(stderr, "  %d\n", *(((byte *)(loop->buffer[bufnum]))+i));
  }
  }

  static void
  _printlongbuf(NpyUFuncLoopObject *loop, int bufnum)
  {
  int i;

  fprintf(stderr, "Printing long buffer %d\n", bufnum);
  for(i=0; i<loop->bufcnt; i++) {
  fprintf(stderr, "  %ld\n", *(((long *)(loop->buffer[bufnum]))+i));
  }
  }

  static void
  _printlongbufptr(NpyUFuncLoopObject *loop, int bufnum)
  {
  int i;

  fprintf(stderr, "Printing long buffer %d\n", bufnum);
  for(i=0; i<loop->bufcnt; i++) {
  fprintf(stderr, "  %ld\n", *(((long *)(loop->bufptr[bufnum]))+i));
  }
  }



  static void
  _printcastbuf(NpyUFuncLoopObject *loop, int bufnum)
  {
  int i;

  fprintf(stderr, "Printing long buffer %d\n", bufnum);
  for(i=0; i<loop->bufcnt; i++) {
  fprintf(stderr, "  %ld\n", *(((long *)(loop->castbuf[bufnum]))+i));
  }
  }

*/

static int
prepare_outputs(PyUFuncObject* self, PyArrayObject **mps, PyObject* args)
{
    PyObject *wraparr[NPY_MAXARGS];
    Py_ssize_t nargs;
    int i;
    int result = 0;

    /* Set up prepare functions. */
    nargs = PyTuple_Size(args);
    _find_array_prepare(args, wraparr, PyUFunc_UFUNC(self)->nin, PyUFunc_UFUNC(self)->nout);

    /* wrap outputs */
    for (i = 0; i < PyUFunc_UFUNC(self)->nout; i++) {
        int j = PyUFunc_UFUNC(self)->nin+i;
        PyObject *wrap;
        PyObject *res;
        wrap = wraparr[i];
        if (wrap != NULL) {
            if (wrap == Py_None) {
                continue;
            }
            res = PyObject_CallFunction(wrap, "O(OOi)",
                                        mps[j], 
                                        self, args, i);
            if ((res == NULL) || (res == Py_None)) {
                if (!PyErr_Occurred()){
                    PyErr_SetString(PyExc_TypeError,
                            "__array_prepare__ must return an ndarray or subclass thereof");
                }
                result = -1;
                break;
            }
            Py_DECREF(mps[j]);
            mps[j] = (PyArrayObject *)res;
        }
    }

    /* Cleanup. */
    for (i=0; i < PyUFunc_UFUNC(self)->nout; i++) {
        Py_XDECREF(wraparr[i]);
    }

    return result;
}


/*
 * currently generic ufuncs cannot be built for use on flexible arrays.
 *
 * The cast functions in the generic loop would need to be fixed to pass
 * in something besides NULL, NULL.
 *
 * Also the underlying ufunc loops would not know the element-size unless
 * that was passed in as data (which could be arranged).
 *
 */

/*UFUNC_API
 *
 * This generic function is called with the ufunc object, the arguments to it,
 * and an array of (pointers to) PyArrayObjects which are NULL.  The
 * arguments are parsed and placed in mps in construct_loop (construct_arrays)
 */
NPY_NO_EXPORT int
PyUFunc_GenericFunction(PyUFuncObject *pySelf, PyObject *args, PyObject *kwds,
                        PyArrayObject **mps)
{
    NpyUFuncObject *self;
    NpyUFuncLoopObject *loop;
    int i;
    NPY_BEGIN_THREADS_DEF;
    Py_ssize_t nargs;
    PyObject *extobj = NULL;
    PyObject *typetup = NULL;
    int typenumbuf[NPY_MAXARGS];
    int *rtypenums;
    int res;
    char* name = PyUFunc_UFUNC(pySelf)->name ? PyUFunc_UFUNC(pySelf)->name : "";
    int bufsize, errormask;
    PyObject *errobj;
    PyObject *obj;
    int originalArgWasObjArray = 0;

    /*
     * Extract sig= keyword and extobj= keyword if present.
     * Raise an error if anything else is present in the
     * keyword dictionary
     */
    if (kwds != NULL) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwds, &pos, &key, &value)) {
            char *keystring = PyString_AsString(key);

            if (keystring == NULL) {
                PyErr_Clear();
                PyErr_SetString(PyExc_TypeError, "invalid keyword");
                return -1;
            }
            if (strncmp(keystring,"extobj",6) == 0) {
                extobj = value;
            }
            else if (strncmp(keystring,"sig",3) == 0) {
                typetup = value;
            }
            else {
                char *format = "'%s' is an invalid keyword to %s";
                PyErr_Format(PyExc_TypeError,format,keystring, name);
                return -1;
            }
        }
    }

    /* Convert typetup to an array of typenums. */
    if (typetup != NULL) {
        if (get_rtypenums(PyUFunc_UFUNC(pySelf), typetup, typenumbuf) < 0) {
            return -1;
        }
        rtypenums = typenumbuf;
    } else {
        rtypenums = NULL;
    }
    Py_XDECREF(typetup);
    
    
    /* Setup error handling. */
    if (extobj == NULL) {
        if (PyUFunc_GetPyValues(name,
                                &bufsize, &errormask,
                                &errobj) < 0) {
            return -1;
        }
    }
    else {
        res =_extract_pyvals(extobj, name,
                             &bufsize, &errormask,
                             &errobj);
        Py_DECREF(extobj);
        if (res < 0) {
            return -1;
        }
    }
    PyUFunc_clearfperr();
    
    self = PyUFunc_UFUNC(pySelf);
    pySelf = NULL;
    
    /* Convert args to arrays in mps. */
    if ((i=convert_args(self, args, mps)) < 0) {
        return i;
    }

    /*
     * We will fail with NotImplemented if the other object has
     * the __r<op>__ method and has __array_priority__ as
     * an attribute (signalling it can handle ndarray's)
     * and is not already an ndarray or a subtype of the same type.
     * We can determin this yet, but need to check the original
     * argument here in the interface before calling the core.
     */
    /* TODO: This is ugly, need a better solution. */
    if (self->nin == 2 && self->nout == 1) {
        obj = PyTuple_GET_ITEM(args, 1);
        originalArgWasObjArray = (!PyArray_CheckExact(obj)
                                  /* If both are same subtype of object arrays, then proceed */
                                  && !(Py_TYPE(obj) == Py_TYPE(PyTuple_GET_ITEM(args, 0)))
                                  && PyObject_HasAttrString(obj, "__array_priority__")
                                  && _has_reflected_op(obj, self->name));
    } else originalArgWasObjArray = 0;
    
        
    /*
     * REFACTOR POINT - most of the rest can be pushed into the core.
     */
    
    /* Build the loop. */
    loop = construct_loop(self);
    if (loop == NULL) {
        return -1;
    }
    loop->bufsize = bufsize;
    loop->errormask = errormask;
    loop->errobj = errobj;
    
    /* Setup the arrays */
    nargs = PyTuple_Size(args);
    res = construct_arrays(loop, nargs, mps, rtypenums, 
                           (prepare_outputs_func)prepare_outputs,
                           (void*) args);

    if (res < 0) {
        ufuncloop_dealloc(loop);
        return -1;
    }

    /*
     * FAIL with NotImplemented if the other object has
     * the __r<op>__ method and has __array_priority__ as
     * an attribute (signalling it can handle ndarray's)
     * and is not already an ndarray or a subtype of the same type.
     */
    if (self->nin == 2 && self->nout == 1 &&
        PyArray_TYPE(mps[1]) == PyArray_OBJECT && originalArgWasObjArray) {
        /* Return -2 for notimplemented. */
        ufuncloop_dealloc(loop);
        return -2;
    }

    if (loop->notimplemented) {
        ufuncloop_dealloc(loop);
        return -2;
    }
    if (self->core_enabled && loop->meth != SIGNATURE_NOBUFFER_UFUNCLOOP) {
        PyErr_SetString(PyExc_RuntimeError,
                        "illegal loop method for ufunc with signature");
        goto fail;
    }

    NPY_LOOP_BEGIN_THREADS;
    switch(loop->meth) {
    case ONE_UFUNCLOOP:
        /*
         * Everything is contiguous, notswapped, aligned,
         * and of the right type.  -- Fastest.
         * Or if not contiguous, then a single-stride
         * increment moves through the entire array.
         */
        /*fprintf(stderr, "ONE...%d\n", loop->size);*/
        loop->function((char **)loop->bufptr, &(loop->iter->size),
                loop->steps, loop->funcdata);
        UFUNC_CHECK_ERROR(loop);
        break;
    case NOBUFFER_UFUNCLOOP:
        /*
         * Everything is notswapped, aligned and of the
         * right type but not contiguous. -- Almost as fast.
         */
        /*fprintf(stderr, "NOBUFFER...%d\n", loop->size);*/
        while (loop->iter->index < loop->iter->size) {
            for (i = 0; i < self->nargs; i++) {
                loop->bufptr[i] = loop->iter->iters[i]->dataptr;
            }
            loop->function((char **)loop->bufptr, &(loop->bufcnt),
                    loop->steps, loop->funcdata);
            UFUNC_CHECK_ERROR(loop);

            /* Adjust loop pointers */
            for (i = 0; i < self->nargs; i++) {
                NpyArray_ITER_NEXT(loop->iter->iters[i]);
            }
            loop->iter->index++;
        }
        break;
    case SIGNATURE_NOBUFFER_UFUNCLOOP:
        while (loop->iter->index < loop->iter->size) {
            for (i = 0; i < self->nargs; i++) {
                loop->bufptr[i] = loop->iter->iters[i]->dataptr;
            }
            loop->function((char **)loop->bufptr, loop->core_dim_sizes,
                    loop->core_strides, loop->funcdata);
            UFUNC_CHECK_ERROR(loop);

            /* Adjust loop pointers */
            for (i = 0; i < self->nargs; i++) {
                NpyArray_ITER_NEXT(loop->iter->iters[i]);
            }
            loop->iter->index++;
        }
        break;
    case BUFFER_UFUNCLOOP: {
        /* This should be a function */
        PyArray_CopySwapNFunc *copyswapn[NPY_MAXARGS];
        NpyArrayIterObject **iters=loop->iter->iters;
        int *swap=loop->swap;
        char **dptr=loop->dptr;
        int mpselsize[NPY_MAXARGS];
        intp laststrides[NPY_MAXARGS];
        int fastmemcpy[NPY_MAXARGS];
        int *needbuffer = loop->needbuffer;
        intp index=loop->iter->index, size=loop->iter->size;
        int bufsize;
        intp bufcnt;
        int copysizes[NPY_MAXARGS];
        char **bufptr = loop->bufptr;
        char **buffer = loop->buffer;
        char **castbuf = loop->castbuf;
        intp *steps = loop->steps;
        char *tptr[NPY_MAXARGS];
        int ninnerloops = loop->ninnerloops;
        Bool pyobject[NPY_MAXARGS];
        int datasize[NPY_MAXARGS];
        int j, k, stopcondition;
        char *myptr1, *myptr2;

        for (i = 0; i <self->nargs; i++) {
            copyswapn[i] = PyArray_DESCR(mps[i])->f->copyswapn;
            mpselsize[i] = PyArray_DESCR(mps[i])->elsize;
            pyobject[i] = ((loop->obj & UFUNC_OBJ_ISOBJECT)
                    && (PyArray_TYPE(mps[i]) == PyArray_OBJECT));
            laststrides[i] = iters[i]->strides[loop->lastdim];
            if (steps[i] && laststrides[i] != mpselsize[i]) {
                fastmemcpy[i] = 0;
            }
            else {
                fastmemcpy[i] = 1;
            }
        }
        /* Do generic buffered looping here (works for any kind of
         * arrays -- some need buffers, some don't.
         *
         *
         * New algorithm: N is the largest dimension.  B is the buffer-size.
         * quotient is loop->ninnerloops-1
         * remainder is loop->leftover
         *
         * Compute N = quotient * B + remainder.
         * quotient = N / B  # integer math
         * (store quotient + 1) as the number of innerloops
         * remainder = N % B # integer remainder
         *
         * On the inner-dimension we will have (quotient + 1) loops where
         * the size of the inner function is B for all but the last when the niter size is
         * remainder.
         *
         * So, the code looks very similar to NOBUFFER_LOOP except the inner-most loop is
         * replaced with...
         *
         * for(i=0; i<quotient+1; i++) {
         * if (i==quotient+1) make itersize remainder size
         * copy only needed items to buffer.
         * swap input buffers if needed
         * cast input buffers if needed
         * call loop_function()
         * cast outputs in buffers if needed
         * swap outputs in buffers if needed
         * copy only needed items back to output arrays.
         * update all data-pointers by strides*niter
         * }
         */


        /*
         * fprintf(stderr, "BUFFER...%d,%d,%d\n", loop->size,
         * loop->ninnerloops, loop->leftover);
         */
        /*
         * for(i=0; i<self->nargs; i++) {
         * fprintf(stderr, "iters[%d]->dataptr = %p, %p of size %d\n", i,
         * iters[i], iters[i]->ao->data, PyArray_NBYTES(iters[i]->ao));
         * }
         */
        stopcondition = ninnerloops;
        if (loop->leftover == 0) {
            stopcondition--;
        }
        while (index < size) {
            bufsize=loop->bufsize;
            for(i = 0; i<self->nargs; i++) {
                tptr[i] = loop->iter->iters[i]->dataptr;
                if (needbuffer[i]) {
                    dptr[i] = bufptr[i];
                    datasize[i] = (steps[i] ? bufsize : 1);
                    copysizes[i] = datasize[i] * mpselsize[i];
                }
                else {
                    dptr[i] = tptr[i];
                }
            }

            /* This is the inner function over the last dimension */
            for (k = 1; k<=stopcondition; k++) {
                if (k == ninnerloops) {
                    bufsize = loop->leftover;
                    for (i=0; i<self->nargs;i++) {
                        if (!needbuffer[i]) {
                            continue;
                        }
                        datasize[i] = (steps[i] ? bufsize : 1);
                        copysizes[i] = datasize[i] * mpselsize[i];
                    }
                }
                for (i = 0; i < self->nin; i++) {
                    if (!needbuffer[i]) {
                        continue;
                    }
                    if (fastmemcpy[i]) {
                        memcpy(buffer[i], tptr[i], copysizes[i]);
                    }
                    else {
                        myptr1 = buffer[i];
                        myptr2 = tptr[i];
                        for (j = 0; j < bufsize; j++) {
                            memcpy(myptr1, myptr2, mpselsize[i]);
                            myptr1 += mpselsize[i];
                            myptr2 += laststrides[i];
                        }
                    }

                    /* swap the buffer if necessary */
                    if (swap[i]) {
                        /* fprintf(stderr, "swapping...\n");*/
                        copyswapn[i](buffer[i], mpselsize[i], NULL, -1,
                                     (intp) datasize[i], 1,
                                     PyArray_ARRAY(mps[i]));
                    }
                    /* cast to the other buffer if necessary */
                    if (loop->cast[i]) {
                        /* fprintf(stderr, "casting... %d, %p %p\n", i, buffer[i]); */
                        loop->cast[i](buffer[i], castbuf[i],
                                (intp) datasize[i],
                                NULL, NULL);
                    }
                }

                bufcnt = (intp) bufsize;
                loop->function((char **)dptr, &bufcnt, steps, loop->funcdata);
                UFUNC_CHECK_ERROR(loop);

                for (i = self->nin; i < self->nargs; i++) {
                    if (!needbuffer[i]) {
                        continue;
                    }
                    if (loop->cast[i]) {
                        /* fprintf(stderr, "casting back... %d, %p", i, castbuf[i]); */
                        loop->cast[i](castbuf[i],
                                buffer[i],
                                (intp) datasize[i],
                                NULL, NULL);
                    }
                    if (swap[i]) {
                        copyswapn[i](buffer[i], mpselsize[i], NULL, -1,
                                     (intp) datasize[i], 1,
                                     PyArray_ARRAY(mps[i]));
                    }
                    /*
                     * copy back to output arrays
                     * decref what's already there for object arrays
                     */
                    if (pyobject[i]) {
                        myptr1 = tptr[i];
                        for (j = 0; j < datasize[i]; j++) {
                            Py_XDECREF(*((PyObject **)myptr1));
                            myptr1 += laststrides[i];
                        }
                    }
                    if (fastmemcpy[i]) {
                        memcpy(tptr[i], buffer[i], copysizes[i]);
                    }
                    else {
                        myptr2 = buffer[i];
                        myptr1 = tptr[i];
                        for (j = 0; j < bufsize; j++) {
                            memcpy(myptr1, myptr2, mpselsize[i]);
                            myptr1 += laststrides[i];
                            myptr2 += mpselsize[i];
                        }
                    }
                }
                if (k == stopcondition) {
                    continue;
                }
                for (i = 0; i < self->nargs; i++) {
                    tptr[i] += bufsize * laststrides[i];
                    if (!needbuffer[i]) {
                        dptr[i] = tptr[i];
                    }
                }
            }
            /* end inner function over last dimension */

            if (loop->objfunc) {
                /*
                 * DECREF castbuf when underlying function used
                 * object arrays and casting was needed to get
                 * to object arrays
                 */
                for (i = 0; i < self->nargs; i++) {
                    if (loop->cast[i]) {
                        if (steps[i] == 0) {
                            Py_XDECREF(*((PyObject **)castbuf[i]));
                        }
                        else {
                            int size = loop->bufsize;

                            PyObject **objptr = (PyObject **)castbuf[i];
                            /*
                             * size is loop->bufsize unless there
                             * was only one loop
                             */
                            if (ninnerloops == 1) {
                                size = loop->leftover;
                            }
                            for (j = 0; j < size; j++) {
                                Py_XDECREF(*objptr);
                                *objptr = NULL;
                                objptr += 1;
                            }
                        }
                    }
                }
                /* Prevent doing the decref twice on an error. */
                loop->objfunc = 0;
            }
            /* fixme -- probably not needed here*/
            UFUNC_CHECK_ERROR(loop);

            for (i = 0; i < self->nargs; i++) {
                NpyArray_ITER_NEXT(loop->iter->iters[i]);
            }
            index++;
        }
    } /* end of last case statement */
    }

    NPY_LOOP_END_THREADS;
    ufuncloop_dealloc(loop);
    return 0;

fail:
    NPY_LOOP_END_THREADS;
    if (loop) {
        if (loop->objfunc) {
            char **castbuf = loop->castbuf;
            intp *steps = loop->steps;
            int ninnerloops = loop->ninnerloops;
            int j;

            /*
             * DECREF castbuf when underlying function used
             * object arrays and casting was needed to get
             * to object arrays
             */
            for (i = 0; i < self->nargs; i++) {
                if (loop->cast[i]) {
                    if (steps[i] == 0) {
                        Py_XDECREF(*((PyObject **)castbuf[i]));
                    }
                    else {
                        int size = loop->bufsize;

                        PyObject **objptr = (PyObject **)castbuf[i];
                        /*
                         * size is loop->bufsize unless there
                         * was only one loop
                         */
                        if (ninnerloops == 1) {
                            size = loop->leftover;
                        }
                        for (j = 0; j < size; j++) {
                            Py_XDECREF(*objptr);
                            *objptr = NULL;
                            objptr += 1;
                        }
                    }
                }
            }
        }
        ufuncloop_dealloc(loop);
    }
    return -1;
}

static PyArrayObject *
_getidentity(NpyUFuncObject *self, int otype, char *str)
{
    /* TODO: Ready to move */
    PyObject *obj, *arr;
    PyArray_Descr *typecode = NULL;

    if (self->identity == PyUFunc_None) {
        PyErr_Format(PyExc_ValueError,
                     "zero-size array to ufunc.%s "      \
                     "without identity", str);
        return NULL;
    }
    if (self->identity == PyUFunc_One) {
        obj = PyInt_FromLong((long) 1);
    } else {
        obj = PyInt_FromLong((long) 0);
    }

    typecode = PyArray_DescrFromType(otype);
    
    arr = PyArray_FromAny(obj, typecode, 0, 0, CARRAY, NULL);
    Py_DECREF(obj);
    return (PyArrayObject *)arr;
}

static int
_create_reduce_copy(NpyUFuncReduceObject *loop, NpyArray **arr, int rtype)
{
    /* TODO: Ready to move */
    intp maxsize;
    NpyArray *new;
    NpyArray_Descr *ntype;

    maxsize = NpyArray_SIZE(*arr);

    if (maxsize < loop->bufsize) {
        if (!(NpyArray_ISBEHAVED_RO(*arr))
            || NpyArray_TYPE(*arr) != rtype) {
            ntype = NpyArray_DescrFromType(rtype);
            
            new = NpyArray_FromArray(*arr, ntype, FORCECAST | ALIGNED);
            if (new == NULL) {
                return -1;
            }
            *arr = new;
            loop->decref_arr = new;
        }
    }

    /*
     * Don't decref *arr before re-assigning
     * because it was not going to be DECREF'd anyway.
     *
     * If a copy is made, then the copy will be removed
     * on deallocation of the loop structure by setting
     * loop->decref_arr.
     */
    return 0;
}

static NpyUFuncReduceObject *
construct_reduce(NpyUFuncObject *self, NpyArray **arr, NpyArray *out,
                 int axis, int otype, int operation, intp ind_size, char *str)
{
    NpyUFuncReduceObject *loop;
    NpyArray *idarr;
    NpyArray *aar;
    npy_intp loop_i[MAX_DIMS], outsize = 0;
    int arg_types[3];
    NPY_SCALARKIND scalars[3] = { NPY_NOSCALAR, NPY_NOSCALAR,
                                  NPY_NOSCALAR };
    int i, j, nd;
    int flags;
    int bufsize, errormask;
    PyObject *errobj;

    /* Reduce type is the type requested of the input during reduction */
    if (self->core_enabled) {
        PyErr_Format(PyExc_RuntimeError,
                     "construct_reduce not allowed on ufunc with signature");
        return NULL;
    }
    /* get looping parameters from Python */
    if (PyUFunc_GetPyValues(str, &bufsize, &errormask,
                            &errobj) < 0) {
        return NULL;
    }
    
    nd = NpyArray_NDIM(*arr);
    arg_types[0] = otype;
    arg_types[1] = otype;
    arg_types[2] = otype;
    if ((loop = malloc(sizeof(NpyUFuncReduceObject))) == NULL) {
        PyErr_NoMemory();
        return loop;
    }
    loop->magic_number = NPY_VALID_MAGIC;
    
    loop->retbase = 0;
    loop->swap = 0;
    loop->index = 0;
    loop->ufunc = self;
    Npy_INCREF(self);
    loop->cast = NULL;
    loop->buffer = NULL;
    loop->ret = NULL;
    loop->it = NULL;
    loop->rit = NULL;
    loop->errobj = NULL;
    loop->first = 1;
    loop->decref_arr = NULL;
    loop->N = NpyArray_DIM(*arr,axis);
    loop->instrides = NpyArray_STRIDE(*arr, axis);
    loop->bufsize = bufsize;
    loop->errormask = errormask;
    loop->errobj = errobj;
    if (select_types(loop->ufunc, arg_types, &(loop->function),
                     &(loop->funcdata), scalars, NULL) == -1) {
        goto fail;
    }
    /*
     * output type may change -- if it does
     * reduction is forced into that type
     * and we need to select the reduction function again
     */
    if (otype != arg_types[2]) {
        otype = arg_types[2];
        arg_types[0] = otype;
        arg_types[1] = otype;
        if (select_types(loop->ufunc, arg_types, &(loop->function),
                         &(loop->funcdata), scalars, NULL) == -1) {
            goto fail;
        }
    }

    /* Make copy if misbehaved or not otype for small arrays */
    if (_create_reduce_copy(loop, arr, otype) < 0) {
        goto fail;
    }
    aar = *arr;

    if (loop->N == 0) {
        loop->meth = ZERO_EL_REDUCELOOP;
    }
    else if (NpyArray_ISBEHAVED_RO(aar) && (otype == NpyArray_TYPE(aar))) {
        if (loop->N == 1) {
            loop->meth = ONE_EL_REDUCELOOP;
        }
        else {
            loop->meth = NOBUFFER_UFUNCLOOP;
            loop->steps[1] = NpyArray_STRIDE(aar, axis);
            loop->N -= 1;
        }
    }
    else {
        loop->meth = BUFFER_UFUNCLOOP;
        loop->swap = !(NpyArray_ISNOTSWAPPED(aar));
    }

    /* Determine if object arrays are involved */
    if (otype == NPY_OBJECT || NpyArray_TYPE(aar) == NPY_OBJECT) {
        loop->obj = UFUNC_OBJ_ISOBJECT | UFUNC_OBJ_NEEDS_API;
    }
    else if ((otype == PyArray_DATETIME)
            || (NpyArray_TYPE(aar) == NPY_DATETIME)
            || (otype == PyArray_TIMEDELTA)
            || (NpyArray_TYPE(aar) == NPY_TIMEDELTA))
    {
        loop->obj = UFUNC_OBJ_NEEDS_API;
    }
    else {
        loop->obj = 0;
    }
    if ((loop->meth == ZERO_EL_REDUCELOOP)
        || ((operation == UFUNC_REDUCEAT)
            && (loop->meth == BUFFER_UFUNCLOOP))) {
        /* TODO: Refactor this, _getidentity calls PyArray_FromAny, need to make it core-safe. */    
        PyArrayObject *idarrWrap = (PyArrayObject *)_getidentity(self, otype, str);
        if (idarr == NULL) {
            goto fail;
        }
        idarr = PyArray_ARRAY(idarrWrap);
        Npy_INCREF(idarr);
        Py_DECREF(idarrWrap);
                
        if (NpyArray_ITEMSIZE(idarr) > UFUNC_MAXIDENTITY) {
            PyErr_Format(PyExc_RuntimeError,
                    "UFUNC_MAXIDENTITY (%d) is too small"\
                    "(needs to be at least %d)",
                    UFUNC_MAXIDENTITY, NpyArray_ITEMSIZE(idarr));
            Npy_DECREF(idarr);
            goto fail;
        }
        memcpy(loop->idptr, NpyArray_BYTES(idarr), NpyArray_ITEMSIZE(idarr));
        Npy_DECREF(idarr);
    }

    /* Construct return array */
    flags = NPY_CARRAY | NPY_UPDATEIFCOPY | NPY_FORCECAST;
    switch(operation) {
    case UFUNC_REDUCE:
        for (j = 0, i = 0; i < nd; i++) {
            if (i != axis) {
                loop_i[j++] = NpyArray_DIM(aar, i);
            }
        }
        if (out == NULL) {
            loop->ret = NpyArray_New(Py_TYPE(Npy_INTERFACE(aar)), NpyArray_NDIM(aar)-1, loop_i,
                            otype, NULL, NULL, 0, 0,
                            (PyObject *)Npy_INTERFACE(aar));
        }
        else {
            outsize = PyArray_MultiplyList(loop_i, NpyArray_NDIM(aar) - 1);
        }
        break;
    case UFUNC_ACCUMULATE:
        if (out == NULL) {
            loop->ret = NpyArray_New(Py_TYPE(Npy_INTERFACE(aar)), NpyArray_NDIM(aar), 
                            NpyArray_DIMS(aar),
                            otype, NULL, NULL, 0, 0, (PyObject *)Npy_INTERFACE(aar));
        }
        else {
            outsize = PyArray_MultiplyList(NpyArray_DIMS(aar), NpyArray_NDIM(aar));
        }
        break;
    case UFUNC_REDUCEAT:
        memcpy(loop_i, NpyArray_DIMS(aar), nd*sizeof(npy_intp));
        /* Index is 1-d array */
        loop_i[axis] = ind_size;
        if (out == NULL) {
            loop->ret = NpyArray_New(Py_TYPE(Npy_INTERFACE(aar)), NpyArray_NDIM(aar), loop_i, otype,
                                     NULL, NULL, 0, 0, (PyObject *)Npy_INTERFACE(aar));
        }
        else {
            outsize = PyArray_MultiplyList(loop_i, NpyArray_NDIM(aar));
        }
        if (ind_size == 0) {
            loop->meth = ZERO_EL_REDUCELOOP;
            return loop;
        }
        if (loop->meth == ONE_EL_REDUCELOOP) {
            loop->meth = NOBUFFER_REDUCELOOP;
        }
        break;
    }
    if (out) {
        if (NpyArray_SIZE(out) != outsize) {
            PyErr_SetString(PyExc_ValueError,
                    "wrong shape for output");
            goto fail;
        }
        loop->ret = NpyArray_FromArray(out, NpyArray_DescrFromType(otype), flags);
        if (loop->ret && loop->ret != out) {
            loop->retbase = 1;
        }
    }
    if (loop->ret == NULL) {
        goto fail;
    }
    loop->insize = NpyArray_ITEMSIZE(aar);
    loop->outsize = NpyArray_ITEMSIZE(loop->ret);
    loop->bufptr[0] = NpyArray_BYTES(loop->ret);

    if (loop->meth == ZERO_EL_REDUCELOOP) {
        loop->size = NpyArray_SIZE(loop->ret);
        return loop;
    }

    loop->it = NpyArray_IterNew(aar);
    if (loop->it == NULL) {
        return NULL;
    }
    if (loop->meth == ONE_EL_REDUCELOOP) {
        loop->size = loop->it->size;
        return loop;
    }

    /*
     * Fix iterator to loop over correct dimension
     * Set size in axis dimension to 1
     */
    loop->it->contiguous = 0;
    loop->it->size /= (loop->it->dims_m1[axis]+1);
    loop->it->dims_m1[axis] = 0;
    loop->it->backstrides[axis] = 0;
    loop->size = loop->it->size;
    if (operation == UFUNC_REDUCE) {
        loop->steps[0] = 0;
    }
    else {
        loop->rit = NpyArray_IterNew(loop->ret);
        if (loop->rit == NULL) {
            return NULL;
        }
        /*
         * Fix iterator to loop over correct dimension
         * Set size in axis dimension to 1
         */
        loop->rit->contiguous = 0;
        loop->rit->size /= (loop->rit->dims_m1[axis] + 1);
        loop->rit->dims_m1[axis] = 0;
        loop->rit->backstrides[axis] = 0;

        if (operation == UFUNC_ACCUMULATE) {
            loop->steps[0] = NpyArray_STRIDE(loop->ret, axis);
        }
        else {
            loop->steps[0] = 0;
        }
    }
    loop->steps[2] = loop->steps[0];
    loop->bufptr[2] = loop->bufptr[0] + loop->steps[2];
    if (loop->meth == BUFFER_UFUNCLOOP) {
        int _size;

        loop->steps[1] = loop->outsize;
        if (otype != NpyArray_TYPE(aar)) {
            _size=loop->bufsize*(loop->outsize + NpyArray_ITEMSIZE(aar));
            loop->buffer = NpyDataMem_NEW(_size);
            if (loop->buffer == NULL) {
                goto fail;
            }
            if (loop->obj & UFUNC_OBJ_ISOBJECT) {
                memset(loop->buffer, 0, _size);
            }
            loop->castbuf = loop->buffer + loop->bufsize*NpyArray_ITEMSIZE(aar);
            loop->bufptr[1] = loop->castbuf;
            loop->cast = NpyArray_GetCastFunc(NpyArray_DESCR(aar), otype);
            if (loop->cast == NULL) {
                goto fail;
            }
        }
        else {
            _size = loop->bufsize * loop->outsize;
            loop->buffer = NpyDataMem_NEW(_size);
            if (loop->buffer == NULL) {
                goto fail;
            }
            if (loop->obj & UFUNC_OBJ_ISOBJECT) {
                memset(loop->buffer, 0, _size);
            }
            loop->bufptr[1] = loop->buffer;
        }
    }
    PyUFunc_clearfperr();
    return loop;

 fail:
    ufuncreduce_dealloc(loop);
    return NULL;
}


/*
 * We have two basic kinds of loops. One is used when arr is not-swapped
 * and aligned and output type is the same as input type.  The other uses
 * buffers when one of these is not satisfied.
 *
 *  Zero-length and one-length axes-to-be-reduced are handled separately.
 */
static PyObject *
PyUFunc_Reduce(PyUFuncObject *self, PyArrayObject *arr, PyArrayObject *out,
        int axis, int otype)
{
    PyArrayObject *ret = NULL;
    NpyUFuncReduceObject *loop;
    intp i, n;
    char *dptr;
    NpyArray *arrCore = NULL;
    NPY_BEGIN_THREADS_DEF;

    assert(NULL == arr || NPY_VALID_MAGIC == arr->magic_number && NPY_VALID_MAGIC == PyArray_DESCR(arr)->magic_number);
    
    /* Construct loop object */
    arrCore = PyArray_ARRAY(arr);
    loop = construct_reduce(PyUFunc_UFUNC(self), &arrCore, PyArray_ARRAY(out), axis, otype, UFUNC_REDUCE, 0,
            "reduce");
    if (!loop) {
        return NULL;
    }

    NPY_LOOP_BEGIN_THREADS;
    switch(loop->meth) {
    case ZERO_EL_REDUCELOOP:
        /* fprintf(stderr, "ZERO..%d\n", loop->size); */
        for (i = 0; i < loop->size; i++) {
            if (loop->obj & UFUNC_OBJ_ISOBJECT) {
                Py_INCREF(*((PyObject **)loop->idptr));
            }
            memmove(loop->bufptr[0], loop->idptr, loop->outsize);
            loop->bufptr[0] += loop->outsize;
        }
        break;
    case ONE_EL_REDUCELOOP:
        /*fprintf(stderr, "ONEDIM..%d\n", loop->size); */
        while (loop->index < loop->size) {
            if (loop->obj & UFUNC_OBJ_ISOBJECT) {
                Py_INCREF(*((PyObject **)loop->it->dataptr));
            }
            memmove(loop->bufptr[0], loop->it->dataptr, loop->outsize);
            NpyArray_ITER_NEXT(loop->it);
            loop->bufptr[0] += loop->outsize;
            loop->index++;
        }
        break;
    case NOBUFFER_UFUNCLOOP:
        /*fprintf(stderr, "NOBUFFER..%d\n", loop->size); */
        while (loop->index < loop->size) {
            /* Copy first element to output */
            if (loop->obj & UFUNC_OBJ_ISOBJECT) {
                Py_INCREF(*((PyObject **)loop->it->dataptr));
            }
            memmove(loop->bufptr[0], loop->it->dataptr, loop->outsize);
            /* Adjust input pointer */
            loop->bufptr[1] = loop->it->dataptr+loop->steps[1];
            loop->function((char **)loop->bufptr, &(loop->N),
                    loop->steps, loop->funcdata);
            UFUNC_CHECK_ERROR(loop);
            NpyArray_ITER_NEXT(loop->it);
            loop->bufptr[0] += loop->outsize;
            loop->bufptr[2] = loop->bufptr[0];
            loop->index++;
        }
        break;
    case BUFFER_UFUNCLOOP:
        /*
         * use buffer for arr
         *
         * For each row to reduce
         * 1. copy first item over to output (casting if necessary)
         * 2. Fill inner buffer
         * 3. When buffer is filled or end of row
         * a. Cast input buffers if needed
         * b. Call inner function.
         * 4. Repeat 2 until row is done.
         */
        /* fprintf(stderr, "BUFFERED..%d %d\n", loop->size, loop->swap); */
        while(loop->index < loop->size) {
            loop->inptr = loop->it->dataptr;
            /* Copy (cast) First term over to output */
            if (loop->cast) {
                /* A little tricky because we need to cast it first */
                NpyArray_DESCR(arrCore)->f->copyswap(loop->buffer, loop->inptr,
                                                     loop->swap, NULL);
                loop->cast(loop->buffer, loop->castbuf, 1, NULL, NULL);
                if ((loop->obj & UFUNC_OBJ_ISOBJECT) &&
                    !NpyArray_ISOBJECT(arrCore)) {
                    /* 
                     * In this case the cast function is creating
                     * an object reference so we need to incref
                     * it since we care copying it to bufptr[0].
                     */
                    Py_XINCREF(*((PyObject **)loop->castbuf));
                }
                memcpy(loop->bufptr[0], loop->castbuf, loop->outsize);
            }
            else {
                /* Simple copy */
                NpyArray_DESCR(arrCore)->f->copyswap(loop->bufptr[0], loop->inptr,
                                                     loop->swap, NULL);
            }
            loop->inptr += loop->instrides;
            n = 1;
            while(n < loop->N) {
                /* Copy up to loop->bufsize elements to buffer */
                dptr = loop->buffer;
                for (i = 0; i < loop->bufsize; i++, n++) {
                    if (n == loop->N) {
                        break;
                    }
                    NpyArray_DESCR(arrCore)->f->copyswap(dptr, loop->inptr,
                                                         loop->swap, NULL);
                    loop->inptr += loop->instrides;
                    dptr += loop->insize;
                }
                if (loop->cast) {
                    loop->cast(loop->buffer, loop->castbuf, i, NULL, NULL);
                }
                loop->function((char **)loop->bufptr, &i,
                               loop->steps, loop->funcdata);
                loop->bufptr[0] += loop->steps[0]*i;
                loop->bufptr[2] += loop->steps[2]*i;
                UFUNC_CHECK_ERROR(loop);
            }
            NpyArray_ITER_NEXT(loop->it);
            loop->bufptr[0] += loop->outsize;
            loop->bufptr[2] = loop->bufptr[0];
            loop->index++;
        }

        if (loop->obj & UFUNC_OBJ_ISOBJECT) {
            /*
             * DECREF left-over objects if buffering was used.
             * There are 2 cases here.
             * 1. The output is an object. In this case the
             * castfunc will produce objects and castbuf needs
             * to be decrefed.
             * 2. The input is an object array.  In this case
             * the copyswap will produce object references and
             * the buffer needs to be decrefed.
             */
            if (!NpyArray_ISOBJECT(arrCore)) {
                for (i=0; i<loop->bufsize; i++) {
                    Py_CLEAR(((PyObject **)loop->castbuf)[i]);
                }
            } else {
                for (i=0; i<loop->bufsize; i++) {
                    Py_CLEAR(((PyObject **)loop->buffer)[i]);
                }
            }
        }

    }
    NPY_LOOP_END_THREADS;
    /* Hang on to this reference -- will be decref'd with loop */
    if (loop->retbase) {
        ret = PyArray_WRAP(loop->ret->base_arr);
    }
    else {
        ret = PyArray_WRAP(loop->ret);
    }
    Py_INCREF(ret);
    ufuncreduce_dealloc(loop);
    return (PyObject *)ret;

fail:
    NPY_LOOP_END_THREADS;
    if (loop) {
        ufuncreduce_dealloc(loop);
    }
    return NULL;
}


static PyObject *
PyUFunc_Accumulate(PyUFuncObject *self, PyArrayObject *arr, PyArrayObject *out,
                   int axis, int otype)
{
    PyArrayObject *ret = NULL;
    NpyUFuncReduceObject *loop;
    intp i, n;
    char *dptr;
    NpyArray *arrCore = PyArray_ARRAY(arr);
    NPY_BEGIN_THREADS_DEF;

    /* Construct loop object */
    loop = construct_reduce(PyUFunc_UFUNC(self), &arrCore, PyArray_ARRAY(out), axis, otype,
            UFUNC_ACCUMULATE, 0, "accumulate");
    if (!loop) {
        return NULL;
    }

    NPY_LOOP_BEGIN_THREADS;
    switch(loop->meth) {
    case ZERO_EL_REDUCELOOP:
        /* Accumulate */
        /* fprintf(stderr, "ZERO..%d\n", loop->size); */
        for (i = 0; i < loop->size; i++) {
            if (loop->obj & UFUNC_OBJ_ISOBJECT) {
                Py_INCREF(*((PyObject **)loop->idptr));
            }
            memcpy(loop->bufptr[0], loop->idptr, loop->outsize);
            loop->bufptr[0] += loop->outsize;
        }
        break;
    case ONE_EL_REDUCELOOP:
        /* Accumulate */
        /* fprintf(stderr, "ONEDIM..%d\n", loop->size); */
        while (loop->index < loop->size) {
            if (loop->obj & UFUNC_OBJ_ISOBJECT) {
                Py_INCREF(*((PyObject **)loop->it->dataptr));
            }
            memmove(loop->bufptr[0], loop->it->dataptr, loop->outsize);
            NpyArray_ITER_NEXT(loop->it);
            loop->bufptr[0] += loop->outsize;
            loop->index++;
        }
        break;
    case NOBUFFER_UFUNCLOOP:
        /* Accumulate */
        /* fprintf(stderr, "NOBUFFER..%d\n", loop->size); */
        while (loop->index < loop->size) {
            /* Copy first element to output */
            if (loop->obj & UFUNC_OBJ_ISOBJECT) {
                Py_INCREF(*((PyObject **)loop->it->dataptr));
            }
            memmove(loop->bufptr[0], loop->it->dataptr, loop->outsize);
            /* Adjust input pointer */
            loop->bufptr[1] = loop->it->dataptr + loop->steps[1];
            loop->function((char **)loop->bufptr, &(loop->N),
                           loop->steps, loop->funcdata);
            UFUNC_CHECK_ERROR(loop);
            NpyArray_ITER_NEXT(loop->it);
            NpyArray_ITER_NEXT(loop->rit);
            loop->bufptr[0] = loop->rit->dataptr;
            loop->bufptr[2] = loop->bufptr[0] + loop->steps[0];
            loop->index++;
        }
        break;
    case BUFFER_UFUNCLOOP:
        /* Accumulate
         *
         * use buffer for arr
         *
         * For each row to reduce
         * 1. copy identity over to output (casting if necessary)
         * 2. Fill inner buffer
         * 3. When buffer is filled or end of row
         * a. Cast input buffers if needed
         * b. Call inner function.
         * 4. Repeat 2 until row is done.
         */
        /* fprintf(stderr, "BUFFERED..%d %p\n", loop->size, loop->cast); */
        while (loop->index < loop->size) {
            loop->inptr = loop->it->dataptr;
            /* Copy (cast) First term over to output */
            if (loop->cast) {
                /* A little tricky because we need to
                   cast it first */
                NpyArray_DESCR(arrCore)->f->copyswap(loop->buffer, loop->inptr,
                                                     loop->swap, NULL);
                loop->cast(loop->buffer, loop->castbuf, 1, NULL, NULL);
                if (loop->obj & UFUNC_OBJ_ISOBJECT) {
                    Py_XINCREF(*((PyObject **)loop->castbuf));
                }
                memcpy(loop->bufptr[0], loop->castbuf, loop->outsize);
            }
            else {
                /* Simple copy */
                NpyArray_DESCR(arrCore)->f->copyswap(loop->bufptr[0], loop->inptr,
                                                     loop->swap, NULL);
            }
            loop->inptr += loop->instrides;
            n = 1;
            while (n < loop->N) {
                /* Copy up to loop->bufsize elements to buffer */
                dptr = loop->buffer;
                for (i = 0; i < loop->bufsize; i++, n++) {
                    if (n == loop->N) {
                        break;
                    }
                    NpyArray_DESCR(arrCore)->f->copyswap(dptr, loop->inptr,
                                                         loop->swap, NULL);
                    loop->inptr += loop->instrides;
                    dptr += loop->insize;
                }
                if (loop->cast) {
                    loop->cast(loop->buffer, loop->castbuf, i, NULL, NULL);
                }
                loop->function((char **)loop->bufptr, &i,
                               loop->steps, loop->funcdata);
                loop->bufptr[0] += loop->steps[0]*i;
                loop->bufptr[2] += loop->steps[2]*i;
                UFUNC_CHECK_ERROR(loop);
            }
            NpyArray_ITER_NEXT(loop->it);
            NpyArray_ITER_NEXT(loop->rit);
            loop->bufptr[0] = loop->rit->dataptr;
            loop->bufptr[2] = loop->bufptr[0] + loop->steps[0];
            loop->index++;
        }

        /*
         * DECREF left-over objects if buffering was used.
         * It is needed when casting created new objects in
         * castbuf.  Intermediate copying into castbuf (via
         * loop->function) decref'd what was already there.

         * It's the final copy into the castbuf that needs a DECREF.
         */

        /* Only when casting needed and it is from a non-object array */
        if ((loop->obj & UFUNC_OBJ_ISOBJECT) && loop->cast &&
            (!NpyArray_ISOBJECT(arrCore))) {
            for (i=0; i<loop->bufsize; i++) {
                Py_CLEAR(((PyObject **)loop->castbuf)[i]);
            }
        }

    }
    NPY_LOOP_END_THREADS;
    /* Hang on to this reference -- will be decref'd with loop */
    if (loop->retbase) {
        ret = PyArray_WRAP(PyArray_BASE_ARRAY(loop->ret));
    }
    else {
        ret = PyArray_WRAP(loop->ret);
    }
    Py_INCREF(ret);
    ufuncreduce_dealloc(loop);
    return (PyObject *)ret;

 fail:
    NPY_LOOP_END_THREADS;
    if (loop) {
        ufuncreduce_dealloc(loop);
    }
    return NULL;
}

/*
 * Reduceat performs a reduce over an axis using the indices as a guide
 *
 * op.reduceat(array,indices)  computes
 * op.reduce(array[indices[i]:indices[i+1]]
 * for i=0..end with an implicit indices[i+1]=len(array)
 * assumed when i=end-1
 *
 * if indices[i+1] <= indices[i]+1
 * then the result is array[indices[i]] for that value
 *
 * op.accumulate(array) is the same as
 * op.reduceat(array,indices)[::2]
 * where indices is range(len(array)-1) with a zero placed in every other sample
 * indices = zeros(len(array)*2-1)
 * indices[1::2] = range(1,len(array))
 *
 * output shape is based on the size of indices
 */
static PyObject *
PyUFunc_Reduceat(PyUFuncObject *self, PyArrayObject *arr, PyArrayObject *ind,
                 PyArrayObject *out, int axis, int otype)
{
    PyArrayObject *ret;
    NpyUFuncReduceObject *loop;
    intp *ptr = (intp *)PyArray_BYTES(ind);
    intp nn = PyArray_DIM(ind, 0);
    intp mm = PyArray_DIM(arr, axis) - 1;
    intp n, i, j;
    char *dptr;
    NpyArray *arrCore = PyArray_ARRAY(arr);
    NPY_BEGIN_THREADS_DEF;

    /* Check for out-of-bounds values in indices array */
    for (i = 0; i<nn; i++) {
        if ((*ptr < 0) || (*ptr > mm)) {
            PyErr_Format(PyExc_IndexError,
                    "index out-of-bounds (0, %d)", (int) mm);
            return NULL;
        }
        ptr++;
    }

    ptr = (npy_intp *)PyArray_BYTES(ind);
    /* Construct loop object */
    loop = construct_reduce(PyUFunc_UFUNC(self), &arrCore, PyArray_ARRAY(out), axis, otype,
            UFUNC_REDUCEAT, nn, "reduceat");
    if (!loop) {
        return NULL;
    }

    NPY_LOOP_BEGIN_THREADS;
    switch(loop->meth) {
    case ZERO_EL_REDUCELOOP:
        /* zero-length index -- return array immediately */
        /* fprintf(stderr, "ZERO..\n"); */
        break;
    case NOBUFFER_UFUNCLOOP:
        /* Reduceat
         * NOBUFFER -- behaved array and same type
         */
        /* fprintf(stderr, "NOBUFFER..%d\n", loop->size); */
        while (loop->index < loop->size) {
            ptr = (npy_intp *)PyArray_BYTES(ind);
            for (i = 0; i < nn; i++) {
                loop->bufptr[1] = loop->it->dataptr + (*ptr)*loop->steps[1];
                if (loop->obj & UFUNC_OBJ_ISOBJECT) {
                    Py_XINCREF(*((PyObject **)loop->bufptr[1]));
                }
                memcpy(loop->bufptr[0], loop->bufptr[1], loop->outsize);
                mm = (i == nn - 1 ? NpyArray_DIM(arrCore, axis) - *ptr :
                        *(ptr + 1) - *ptr) - 1;
                if (mm > 0) {
                    loop->bufptr[1] += loop->steps[1];
                    loop->bufptr[2] = loop->bufptr[0];
                    loop->function((char **)loop->bufptr, &mm,
                            loop->steps, loop->funcdata);
                    UFUNC_CHECK_ERROR(loop);
                }
                loop->bufptr[0] += NpyArray_STRIDE(loop->ret, axis);
                ptr++;
            }
            NpyArray_ITER_NEXT(loop->it);
            NpyArray_ITER_NEXT(loop->rit);
            loop->bufptr[0] = loop->rit->dataptr;
            loop->index++;
        }
        break;

    case BUFFER_UFUNCLOOP:
        /* Reduceat
         * BUFFER -- misbehaved array or different types
         */
        /* fprintf(stderr, "BUFFERED..%d\n", loop->size); */
        while (loop->index < loop->size) {
            ptr = (npy_intp *)PyArray_BYTES(ind);
            for (i = 0; i < nn; i++) {
                if (loop->obj & UFUNC_OBJ_ISOBJECT) {
                    Py_XINCREF(*((PyObject **)loop->idptr));
                }
                memcpy(loop->bufptr[0], loop->idptr, loop->outsize);
                n = 0;
                mm = (i == nn - 1 ? NpyArray_DIM(arrCore, axis) - *ptr :
                        *(ptr + 1) - *ptr);
                if (mm < 1) {
                    mm = 1;
                }
                loop->inptr = loop->it->dataptr + (*ptr)*loop->instrides;
                while (n < mm) {
                    /* Copy up to loop->bufsize elements to buffer */
                    dptr = loop->buffer;
                    for (j = 0; j < loop->bufsize; j++, n++) {
                        if (n == mm) {
                            break;
                        }
                        NpyArray_DESCR(arrCore)->f->copyswap(dptr, loop->inptr,
                             loop->swap, NULL);
                        loop->inptr += loop->instrides;
                        dptr += loop->insize;
                    }
                    if (loop->cast) {
                        loop->cast(loop->buffer, loop->castbuf, j, NULL, NULL);
                    }
                    loop->bufptr[2] = loop->bufptr[0];
                    loop->function((char **)loop->bufptr, &j,
                            loop->steps, loop->funcdata);
                    UFUNC_CHECK_ERROR(loop);
                    loop->bufptr[0] += j*loop->steps[0];
                }
                loop->bufptr[0] += NpyArray_STRIDE(loop->ret, axis);
                ptr++;
            }
            NpyArray_ITER_NEXT(loop->it);
            NpyArray_ITER_NEXT(loop->rit);
            loop->bufptr[0] = loop->rit->dataptr;
            loop->index++;
        }

        /*
         * DECREF left-over objects if buffering was used.
         * It is needed when casting created new objects in
         * castbuf.  Intermediate copying into castbuf (via
         * loop->function) decref'd what was already there.

         * It's the final copy into the castbuf that needs a DECREF.
         */

        /* Only when casting needed and it is from a non-object array */
        if ((loop->obj & UFUNC_OBJ_ISOBJECT) && loop->cast &&
            (!NpyArray_ISOBJECT(arrCore))) {
            for (i=0; i<loop->bufsize; i++) {
                Py_CLEAR(((PyObject **)loop->castbuf)[i]);
            }
        }

        break;
    }
    NPY_LOOP_END_THREADS;
    /* Hang on to this reference -- will be decref'd with loop */
    if (loop->retbase) {
        ret = PyArray_WRAP(PyArray_BASE_ARRAY(loop->ret));
    }
    else {
        ret = PyArray_WRAP(loop->ret);
    }
    Py_INCREF(ret);
    ufuncreduce_dealloc(loop);
    return (PyObject *)ret;

fail:
    NPY_LOOP_END_THREADS;
    if (loop) {
        ufuncreduce_dealloc(loop);
    }
    return NULL;
}


/*
 * This code handles reduce, reduceat, and accumulate
 * (accumulate and reduce are special cases of the more general reduceat
 * but they are handled separately for speed)
 */
static PyObject *
PyUFunc_GenericReduction(PyUFuncObject *self, PyObject *args,
                         PyObject *kwds, int operation)
{
    int axis=0;
    PyArrayObject *mp, *ret = NULL;
    PyObject *op, *res = NULL;
    PyObject *obj_ind, *context;
    PyArrayObject *indices = NULL;
    PyArray_Descr *otype = NULL;
    PyArrayObject *out = NULL;
    static char *kwlist1[] = {"array", "axis", "dtype", "out", NULL};
    static char *kwlist2[] = {"array", "indices", "axis", "dtype", "out", NULL};
    static char *_reduce_type[] = {"reduce", "accumulate", "reduceat", NULL};

    if (self == NULL) {
        PyErr_SetString(PyExc_ValueError, "function not supported");
        return NULL;
    }
    if (PyUFunc_UFUNC(self)->core_enabled) {
        PyErr_Format(PyExc_RuntimeError,
                     "Reduction not defined on ufunc with signature");
        return NULL;
    }
    if (PyUFunc_UFUNC(self)->nin != 2) {
        PyErr_Format(PyExc_ValueError,
                     "%s only supported for binary functions",
                     _reduce_type[operation]);
        return NULL;
    }
    if (PyUFunc_UFUNC(self)->nout != 1) {
        PyErr_Format(PyExc_ValueError,
                     "%s only supported for functions " \
                     "returning a single value",
                     _reduce_type[operation]);
        return NULL;
    }

    if (operation == UFUNC_REDUCEAT) {
        PyArray_Descr *indtype;
        indtype = PyArray_DescrFromType(PyArray_INTP);
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO|iO&O&", kwlist2,
                                        &op, &obj_ind, &axis,
                                        PyArray_DescrConverter2,
                                        &otype,
                                        PyArray_OutputConverter,
                                        &out)) {
            Py_XDECREF(otype);
            return NULL;
        }
        
        indices = (PyArrayObject *)PyArray_FromAny(obj_ind, indtype,
                                                   1, 1, CARRAY, NULL);
        if (indices == NULL) {
            Py_XDECREF(otype);
            return NULL;
        }
    }
    else {
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|iO&O&", kwlist1,
                                        &op, &axis,
                                        PyArray_DescrConverter2,
                                        &otype,
                                        PyArray_OutputConverter,
                                        &out)) {
            Py_XDECREF(otype);
            return NULL;
        }
    }
    /* Ensure input is an array */
    if (!PyArray_Check(op) && !PyArray_IsScalar(op, Generic)) {
        context = Py_BuildValue("O(O)i", self, op, 0);
    }
    else {
        context = NULL;
    }
    
    mp = (PyArrayObject *)PyArray_FromAny(op, NULL, 0, 0, 0, context);
    Py_XDECREF(context);
    if (mp == NULL) {
        return NULL;
    }
    assert( PyArray_ISVALID(mp) );
    
    /* Check to see if input is zero-dimensional */
    if (PyArray_NDIM(mp) == 0) {
        PyErr_Format(PyExc_TypeError, "cannot %s on a scalar",
                     _reduce_type[operation]);
        Py_XDECREF(otype);
        Py_DECREF(mp);
        return NULL;
    }
    /* Check to see that type (and otype) is not FLEXIBLE */
    if (PyArray_ISFLEXIBLE(mp) ||
        (otype && NpyTypeNum_ISFLEXIBLE(otype->descr->type_num))) {
        PyErr_Format(PyExc_TypeError,
                     "cannot perform %s with flexible type",
                     _reduce_type[operation]);
        Py_XDECREF(otype);
        Py_DECREF(mp);
        return NULL;
    }

    if (axis < 0) {
        axis += PyArray_NDIM(mp);
    }
    if (axis < 0 || axis >= PyArray_NDIM(mp)) {
        PyErr_SetString(PyExc_ValueError, "axis not in array");
        Py_XDECREF(otype);
        Py_DECREF(mp);
        return NULL;
    }
     /*
      * If out is specified it determines otype
      * unless otype already specified.
      */
    if (otype == NULL && out != NULL) {
        otype = PyArray_Descr_WRAP( PyArray_DESCR(out) );
        Py_INCREF(otype);
    }
    if (otype == NULL) {
        /*
         * For integer types --- make sure at least a long
         * is used for add and multiply reduction to avoid overflow
         */
        int typenum = PyArray_TYPE(mp);
        if ((typenum < NPY_FLOAT)
            && ((strcmp(PyUFunc_UFUNC(self)->name,"add") == 0)
                || (strcmp(PyUFunc_UFUNC(self)->name,"multiply") == 0))) {
            if (NpyTypeNum_ISBOOL(typenum)) {
                typenum = PyArray_LONG;
            }
            else if ((size_t)PyArray_ITEMSIZE(mp) < sizeof(long)) {
                if (NpyTypeNum_ISUNSIGNED(typenum)) {
                    typenum = PyArray_ULONG;
                }
                else {
                    typenum = PyArray_LONG;
                }
            }
        }
        otype = PyArray_DescrFromType(typenum);
    }


    switch(operation) {
    case UFUNC_REDUCE:
        ret = (PyArrayObject *)PyUFunc_Reduce(self, mp, out, axis,
                                              otype->descr->type_num);
        break;
    case UFUNC_ACCUMULATE:
        ret = (PyArrayObject *)PyUFunc_Accumulate(self, mp, out, axis,
                                                  otype->descr->type_num);
        break;
    case UFUNC_REDUCEAT:
        ret = (PyArrayObject *)PyUFunc_Reduceat(self, mp, indices, out,
                                                axis, otype->descr->type_num);
        Py_DECREF(indices);
        break;
    }
    Py_DECREF(mp);
    Py_DECREF(otype);
    if (ret == NULL) {
        return NULL;
    }
    if (Py_TYPE(op) != Py_TYPE(ret)) {
        res = PyObject_CallMethod(op, "__array_wrap__", "O", ret);
        if (res == NULL) {
            PyErr_Clear();
        }
        else if (res == Py_None) {
            Py_DECREF(res);
        }
        else {
            Py_DECREF(ret);
            return res;
        }
    }
    return PyArray_Return(ret);
}

/*
 * This function analyzes the input arguments
 * and determines an appropriate __array_wrap__ function to call
 * for the outputs.
 *
 * If an output argument is provided, then it is wrapped
 * with its own __array_wrap__ not with the one determined by
 * the input arguments.
 *
 * if the provided output argument is already an array,
 * the wrapping function is None (which means no wrapping will
 * be done --- not even PyArray_Return).
 *
 * A NULL is placed in output_wrap for outputs that
 * should just have PyArray_Return called.
 */
static void
_find_array_wrap(PyObject *args, PyObject **output_wrap, int nin, int nout)
{
    Py_ssize_t nargs;
    int i;
    int np = 0;
    PyObject *with_wrap[NPY_MAXARGS], *wraps[NPY_MAXARGS];
    PyObject *obj, *wrap = NULL;

    nargs = PyTuple_GET_SIZE(args);
    for (i = 0; i < nin; i++) {
        obj = PyTuple_GET_ITEM(args, i);
        if (PyArray_CheckExact(obj) || PyArray_IsAnyScalar(obj)) {
            continue;
        }
        wrap = PyObject_GetAttrString(obj, "__array_wrap__");
        if (wrap) {
            if (PyCallable_Check(wrap)) {
                with_wrap[np] = obj;
                wraps[np] = wrap;
                ++np;
            }
            else {
                Py_DECREF(wrap);
                wrap = NULL;
            }
        }
        else {
            PyErr_Clear();
        }
    }
    if (np > 0) {
        /* If we have some wraps defined, find the one of highest priority */
        wrap = wraps[0];
        if (np > 1) {
            double maxpriority = PyArray_GetPriority(with_wrap[0],
                        PyArray_SUBTYPE_PRIORITY);
            for (i = 1; i < np; ++i) {
                double priority = PyArray_GetPriority(with_wrap[i],
                            PyArray_SUBTYPE_PRIORITY);
                if (priority > maxpriority) {
                    maxpriority = priority;
                    Py_DECREF(wrap);
                    wrap = wraps[i];
                }
                else {
                    Py_DECREF(wraps[i]);
                }
            }
        }
    }

    /*
     * Here wrap is the wrapping function determined from the
     * input arrays (could be NULL).
     *
     * For all the output arrays decide what to do.
     *
     * 1) Use the wrap function determined from the input arrays
     * This is the default if the output array is not
     * passed in.
     *
     * 2) Use the __array_wrap__ method of the output object
     * passed in. -- this is special cased for
     * exact ndarray so that no PyArray_Return is
     * done in that case.
     */
    for (i = 0; i < nout; i++) {
        int j = nin + i;
        int incref = 1;
        output_wrap[i] = wrap;
        if (j < nargs) {
            obj = PyTuple_GET_ITEM(args, j);
            if (obj == Py_None) {
                continue;
            }
            if (PyArray_CheckExact(obj)) {
                output_wrap[i] = Py_None;
            }
            else {
                PyObject *owrap = PyObject_GetAttrString(obj,"__array_wrap__");
                incref = 0;
                if (!(owrap) || !(PyCallable_Check(owrap))) {
                    Py_XDECREF(owrap);
                    owrap = wrap;
                    incref = 1;
                    PyErr_Clear();
                }
                output_wrap[i] = owrap;
            }
        }
        if (incref) {
            Py_XINCREF(output_wrap[i]);
        }
    }
    Py_XDECREF(wrap);
    return;
}

static PyObject *
ufunc_generic_call(PyUFuncObject *self, PyObject *args, PyObject *kwds)
{
    int i;
    PyTupleObject *ret;
    PyArrayObject *mps[NPY_MAXARGS];
    PyObject *retobj[NPY_MAXARGS];
    PyObject *wraparr[NPY_MAXARGS];
    PyObject *res;
    int errval;

    /*
     * Initialize all array objects to NULL to make cleanup easier
     * if something goes wrong.
     */
    for(i = 0; i < PyUFunc_UFUNC(self)->nargs; i++) {
        mps[i] = NULL;
    }
    errval = PyUFunc_GenericFunction(self, args, kwds, mps);
    if (errval < 0) {
        for (i = 0; i < PyUFunc_UFUNC(self)->nargs; i++) {
            PyArray_XDECREF_ERR(mps[i]);
        }
        if (errval == -1)
            return NULL;
        else if (PyUFunc_UFUNC(self)->nin == 2 && PyUFunc_UFUNC(self)->nout == 1) {
          /* To allow the other argument to be given a chance
           */
            Py_INCREF(Py_NotImplemented);
            return Py_NotImplemented;
        }
        else {
            PyErr_SetString(PyExc_NotImplementedError, "Not implemented for this type");
            return NULL;
        }
    }

    for (i = 0; i < PyUFunc_UFUNC(self)->nin; i++) {
        Py_DECREF(mps[i]);
    }
    /*
     * Use __array_wrap__ on all outputs
     * if present on one of the input arguments.
     * If present for multiple inputs:
     * use __array_wrap__ of input object with largest
     * __array_priority__ (default = 0.0)
     *
     * Exception:  we should not wrap outputs for items already
     * passed in as output-arguments.  These items should either
     * be left unwrapped or wrapped by calling their own __array_wrap__
     * routine.
     *
     * For each output argument, wrap will be either
     * NULL --- call PyArray_Return() -- default if no output arguments given
     * None --- array-object passed in don't call PyArray_Return
     * method --- the __array_wrap__ method to call.
     */
    _find_array_wrap(args, wraparr, PyUFunc_UFUNC(self)->nin, PyUFunc_UFUNC(self)->nout);

    /* wrap outputs */
    for (i = 0; i < PyUFunc_UFUNC(self)->nout; i++) {
        int j = PyUFunc_UFUNC(self)->nin+i;
        PyObject *wrap;
        /*
         * check to see if any UPDATEIFCOPY flags are set
         * which meant that a temporary output was generated
         */
        if (PyArray_FLAGS(mps[j]) & UPDATEIFCOPY) {
            PyArrayObject *old;
            old = Npy_INTERFACE(PyArray_BASE_ARRAY(mps[j]));
            /* we want to hang on to this */
            Py_INCREF(old);
            /* should trigger the copyback into old */
            Py_DECREF(mps[j]);
            mps[j] = old;
        }
        wrap = wraparr[i];
        if (wrap != NULL) {
            if (wrap == Py_None) {
                Py_DECREF(wrap);
                retobj[i] = (PyObject *)mps[j];
                continue;
            }
            res = PyObject_CallFunction(wrap, "O(OOi)", mps[j], self, args, i);
            if (res == NULL && PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();
                res = PyObject_CallFunctionObjArgs(wrap, mps[j], NULL);
            }
            Py_DECREF(wrap);
            if (res == NULL) {
                goto fail;
            }
            else if (res == Py_None) {
                Py_DECREF(res);
            }
            else {
                Py_DECREF(mps[j]);
                retobj[i] = res;
                continue;
            }
        }
        /* default behavior */
        retobj[i] = PyArray_Return(mps[j]);
    }

    if (PyUFunc_UFUNC(self)->nout == 1) {
        return retobj[0];
    }
    else {
        ret = (PyTupleObject *)PyTuple_New( PyUFunc_UFUNC(self)->nout );
        for (i = 0; i < PyUFunc_UFUNC(self)->nout; i++) {
            PyTuple_SET_ITEM(ret, i, retobj[i]);
        }
        return (PyObject *)ret;
    }

fail:
    for (i = PyUFunc_UFUNC(self)->nin; i < PyUFunc_UFUNC(self)->nargs; i++) {
        Py_XDECREF(mps[i]);
    }
    return NULL;
}

NPY_NO_EXPORT PyObject *
ufunc_geterr(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *thedict;
    PyObject *res;

    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    if (PyUFunc_PYVALS_NAME == NULL) {
        PyUFunc_PYVALS_NAME = PyUString_InternFromString(UFUNC_PYVALS_NAME);
    }
    thedict = PyThreadState_GetDict();
    if (thedict == NULL) {
        thedict = PyEval_GetBuiltins();
    }
    res = PyDict_GetItem(thedict, PyUFunc_PYVALS_NAME);
    if (res != NULL) {
        Py_INCREF(res);
        return res;
    }
    /* Construct list of defaults */
    res = PyList_New(3);
    if (res == NULL) {
        return NULL;
    }
    PyList_SET_ITEM(res, 0, PyInt_FromLong(PyArray_BUFSIZE));
    PyList_SET_ITEM(res, 1, PyInt_FromLong(UFUNC_ERR_DEFAULT));
    PyList_SET_ITEM(res, 2, Py_None); Py_INCREF(Py_None);
    return res;
}

#if USE_USE_DEFAULTS==1
/*
 * This is a strategy to buy a little speed up and avoid the dictionary
 * look-up in the default case.  It should work in the presence of
 * threads.  If it is deemed too complicated or it doesn't actually work
 * it could be taken out.
 */
static int
ufunc_update_use_defaults(void)
{
    PyObject *errobj = NULL;
    int errmask, bufsize;
    int res;

    PyUFunc_NUM_NODEFAULTS += 1;
    res = PyUFunc_GetPyValues("test", &bufsize, &errmask, &errobj);
    PyUFunc_NUM_NODEFAULTS -= 1;
    if (res < 0) {
        Py_XDECREF(errobj);
        return -1;
    }
    if ((errmask != UFUNC_ERR_DEFAULT) || (bufsize != PyArray_BUFSIZE)
            || (PyTuple_GET_ITEM(errobj, 1) != Py_None)) {
        PyUFunc_NUM_NODEFAULTS += 1;
    }
    else if (PyUFunc_NUM_NODEFAULTS > 0) {
        PyUFunc_NUM_NODEFAULTS -= 1;
    }
    Py_XDECREF(errobj);
    return 0;
}
#endif

NPY_NO_EXPORT PyObject *
ufunc_seterr(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *thedict;
    int res;
    PyObject *val;
    static char *msg = "Error object must be a list of length 3";

    if (!PyArg_ParseTuple(args, "O", &val)) {
        return NULL;
    }
    if (!PyList_CheckExact(val) || PyList_GET_SIZE(val) != 3) {
        PyErr_SetString(PyExc_ValueError, msg);
        return NULL;
    }
    if (PyUFunc_PYVALS_NAME == NULL) {
        PyUFunc_PYVALS_NAME = PyUString_InternFromString(UFUNC_PYVALS_NAME);
    }
    thedict = PyThreadState_GetDict();
    if (thedict == NULL) {
        thedict = PyEval_GetBuiltins();
    }
    res = PyDict_SetItem(thedict, PyUFunc_PYVALS_NAME, val);
    if (res < 0) {
        return NULL;
    }
#if USE_USE_DEFAULTS==1
    if (ufunc_update_use_defaults() < 0) {
        return NULL;
    }
#endif
    Py_INCREF(Py_None);
    return Py_None;
}


int
NpyUFunc_ReplaceLoopBySignature(NpyUFuncObject *func,
                               NpyUFuncGenericFunction newfunc,
                               int *signature,
                               NpyUFuncGenericFunction *oldfunc)
{
    /* TODO: Ready to move */
    int i, j;
    int res = -1;
    /* Find the location of the matching signature */
    for (i = 0; i < func->ntypes; i++) {
        for (j = 0; j < func->nargs; j++) {
            if (signature[j] != func->types[i*func->nargs+j]) {
                break;
            }
        }
        if (j < func->nargs) {
            continue;
        }
        if (oldfunc != NULL) {
            *oldfunc = func->functions[i];
        }
        func->functions[i] = newfunc;
        res = 0;
        break;
    }
    return res;
}

/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_ReplaceLoopBySignature(PyUFuncObject *func,
                               NpyUFuncGenericFunction newfunc,
                               int *signature,
                               NpyUFuncGenericFunction *oldfunc)
{
    return NpyUFunc_ReplaceLoopBySignature(PyUFunc_UFUNC(func), newfunc, signature, oldfunc);
}

/*UFUNC_API*/
NPY_NO_EXPORT PyObject *
PyUFunc_FromFuncAndData(NpyUFuncGenericFunction *func, void **data,
                        char *types, int ntypes,
                        int nin, int nout, int identity,
                        char *name, char *doc, int check_return)
{
    return PyUFunc_FromFuncAndDataAndSignature(func, data, types, ntypes,
        nin, nout, identity, name, doc, check_return, NULL);
}

NpyUFuncObject *
NpyUFunc_FromFuncAndDataAndSignature(NpyUFuncGenericFunction *func, void **data,
                                     char *types, int ntypes,
                                     int nin, int nout, int identity,
                                     char *name, char *doc,
                                     int check_return, const char *signature)
{
    /* TODO: Ready to move */
    NpyUFuncObject *self;
    
    self = (NpyUFuncObject *)malloc(sizeof(NpyUFuncObject));
    if (NULL == self) {
        return NULL;
    }
    NpyObject_Init(self, &NpyUFunc_Type);
    self->magic_number = NPY_VALID_MAGIC;
    
    self->nin = nin;
    self->nout = nout;
    self->nargs = nin+nout;
    self->identity = identity;

    self->functions = func;
    self->data = data;
    self->types = types;
    self->ntypes = ntypes;
    self->check_return = check_return;
    self->ptr = NULL;
    self->userloops=NULL;

    if (name == NULL) {
        self->name = "?";
    }
    else {
        self->name = name;
    }
    if (doc == NULL) {
        self->doc = "NULL";
    }
    else {
        self->doc = doc;
    }

    /* generalized ufunc */
    self->core_enabled = 0;
    self->core_num_dim_ix = 0;
    self->core_num_dims = NULL;
    self->core_dim_ixs = NULL;
    self->core_offsets = NULL;
    self->core_signature = NULL;
    if (signature != NULL) {
        if (0 != _parse_signature(self, signature)) {
            Npy_DECREF(self);
            return NULL;
        }
    }
    return self;
}

NpyUFuncObject *
NpyUFunc_FromFuncAndData(NpyUFuncGenericFunction *func, void **data,
                        char *types, int ntypes,
                        int nin, int nout, int identity,
                        char *name, char *doc, int check_return)
{
    return NpyUFunc_FromFuncAndDataAndSignature(func, data, types, ntypes,
                                                nin, nout, identity, name, doc, check_return, NULL);
}


/*UFUNC_API*/
NPY_NO_EXPORT PyObject *
PyUFunc_FromFuncAndDataAndSignature(NpyUFuncGenericFunction *func, void **data,
                                    char *types, int ntypes,
                                    int nin, int nout, int identity,
                                    char *name, char *doc,
                                    int check_return, const char *signature)
{
    PyUFuncObject *self;
    NpyUFuncObject *selfCore;
    
    selfCore = NpyUFunc_FromFuncAndDataAndSignature(func, data, types, ntypes, nin, nout, 
                                                    identity, name, doc, check_return, 
                                                    signature);
    if (NULL == selfCore) {
        return NULL;
    }
    
    self = _pya_malloc(sizeof(PyUFuncObject));
    if (NULL == self) {
        return NULL;
    }
    PyObject_Init((PyObject *)self, &PyUFunc_Type);
    self->magic_number = NPY_VALID_MAGIC;
    self->ufunc = selfCore;
    self->func_obj = NULL;

    /* This is ugly. We need to move the reference from the core to the interface and set
       the interface object.  This must be done in exactly this order: First inc ref the
       interface to 2, then set the interface on the core object, and then decref the core.
       When we decref the core to 0, it will decref the interface back to 1. */
    Py_INCREF(self);
    selfCore->nob_interface = self;
    Npy_DECREF(selfCore);
    
    return (PyObject *)self;
}



/* Specify that the loop specified by the given index should use the array of
 * input and arrays as the data pointer to the loop.
 */
/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_SetUsesArraysAsData(void **data, size_t i)
{
    data[i] = (void*)PyUFunc_SetUsesArraysAsData;
    return 0;
}

/* Return 1 if the given data pointer for the loop specifies that it needs the
 * arrays as the data pointer.
 */
static int
_does_loop_use_arrays(void *data)
{
    return (data == PyUFunc_SetUsesArraysAsData);
}


/*
 * This is the first-part of the CObject structure.
 *
 * I don't think this will change, but if it should, then
 * this needs to be fixed.  The exposed C-API was insufficient
 * because I needed to replace the pointer and it wouldn't
 * let me with a destructor set (even though it works fine
 * with the destructor).
 */
typedef struct {
    PyObject_HEAD
    void *c_obj;
} _simple_cobj;

#define _SETCPTR(cobj, val) ((_simple_cobj *)(cobj))->c_obj = (val)

/* return 1 if arg1 > arg2, 0 if arg1 == arg2, and -1 if arg1 < arg2 */
static int
cmp_arg_types(int *arg1, int *arg2, int n)
{
    for (; n > 0; n--, arg1++, arg2++) {
        if (PyArray_EquivTypenums(*arg1, *arg2)) {
            continue;
        }
        if (PyArray_CanCastSafely(*arg1, *arg2)) {
            return -1;
        }
        return 1;
    }
    return 0;
}

/*
 * This frees the linked-list structure when the CObject
 * is destroyed (removed from the internal dictionary)
*/
static NPY_INLINE void
_free_loop1d_list(PyUFunc_Loop1d *data)
{
    while (data != NULL) {
        PyUFunc_Loop1d *next = data->next;
        _pya_free(data->arg_types);
        _pya_free(data);
        data = next;
    }
}

#if PY_VERSION_HEX >= 0x02070000
static void
_loop1d_list_free(PyObject *ptr)
{
    PyUFunc_Loop1d *data = (PyUFunc_Loop1d *)PyCapsule_GetPointer(ptr, NULL);
    _free_loop1d_list(data);
}

#else
static void
_loop1d_list_free(void *ptr)
{
    PyUFunc_Loop1d *data = (PyUFunc_Loop1d *)ptr;
    _free_loop1d_list(data);
}
#endif




int
NpyUFunc_RegisterLoopForType(NpyUFuncObject *ufunc,
                             int usertype,
                             NpyUFuncGenericFunction function,
                             int *arg_types,
                             void *data)
{
    NpyArray_Descr *descr;
    PyUFunc_Loop1d *funcdata, *current = NULL;
    int i;
    int *newtypes=NULL;
    
    descr = NpyArray_DescrFromType(usertype);
    if ((usertype < PyArray_USERDEF) || (descr==NULL)) {
        PyErr_SetString(PyExc_TypeError, "unknown user-defined type");
        return -1;
    }
    Npy_DECREF(descr);
    
    if (ufunc->userloops == NULL) {
        ufunc->userloops = npy_create_userloops_table();
    }
    funcdata = malloc(sizeof(PyUFunc_Loop1d));
    if (funcdata == NULL) {
        goto fail;
    }
    newtypes = malloc(sizeof(int)*ufunc->nargs);
    if (newtypes == NULL) {
        goto fail;
    }
    if (arg_types != NULL) {
        for (i = 0; i < ufunc->nargs; i++) {
            newtypes[i] = arg_types[i];
        }
    }
    else {
        for (i = 0; i < ufunc->nargs; i++) {
            newtypes[i] = usertype;
        }
    }
    
    funcdata->func = function;
    funcdata->arg_types = newtypes;
    funcdata->data = data;
    funcdata->next = NULL;
    
    /* Get entry for this user-defined type*/
    current = (PyUFunc_Loop1d *)NpyDict_Get(ufunc->userloops, (void *)(npy_intp)usertype);
    /* If it's not there, then make one and return. */
    if (NULL == current) {
        NpyDict_Put(ufunc->userloops, (void *)(npy_intp)usertype, funcdata);
        return 0;
    }
    else {
        PyUFunc_Loop1d *prev = NULL;
        int cmp = 1;
        /*
         * There is already at least 1 loop. Place this one in
         * lexicographic order.  If the next one signature
         * is exactly like this one, then just replace.
         * Otherwise insert.
         */
        while (current != NULL) {
            cmp = cmp_arg_types(current->arg_types, newtypes, ufunc->nargs);
            if (cmp >= 0) {
                break;
            }
            prev = current;
            current = current->next;
        }
        if (cmp == 0) {
            /* just replace it with new function */
            current->func = function;
            current->data = data;
            free(newtypes);
            free(funcdata);
        }
        else {
            /*
             * insert it before the current one by hacking the internals
             * of cobject to replace the function pointer --- can't use
             * CObject API because destructor is set.
             */
            funcdata->next = current;
            if (prev == NULL) {
                /* place this at front */
                NpyDict_ForceValue(ufunc->userloops, (void *)(npy_intp)usertype, funcdata);
            }
            else {
                prev->next = funcdata;
            }
        }
    }
    return 0;
    
fail:
    free(funcdata);
    free(newtypes);
    if (!PyErr_Occurred()) PyErr_NoMemory();
    return -1;
}



/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_RegisterLoopForType(PyUFuncObject *ufunc,
                            int usertype,
                            NpyUFuncGenericFunction function,
                            int *arg_types,
                            void *data)
{
    return NpyUFunc_RegisterLoopForType(PyUFunc_UFUNC(ufunc), usertype, function, arg_types, data);
}

#undef _SETCPTR

static void
npy_ufunc_dealloc(NpyUFuncObject *self)
{
    /* TODO: Ready to move */
    if (self->core_num_dims) {
        free(self->core_num_dims);
    }
    if (self->core_dim_ixs) {
        free(self->core_dim_ixs);
    }
    if (self->core_offsets) {
        free(self->core_offsets);
    }
    if (self->core_signature) {
        free(self->core_signature);
    }
    if (self->ptr) {
        free(self->ptr);
    }
    if (NULL != self->userloops) {
        NpyDict_Destroy(self->userloops);
    }
    self->magic_number = NPY_INVALID_MAGIC;
    free(self);
}

NpyTypeObject NpyUFunc_Type = {
    /* TODO: Ready to move */
    (void (*)(_NpyObject *))npy_ufunc_dealloc
};


static void
ufunc_dealloc(PyUFuncObject *self)
{
    npy_ufunc_dealloc(PyUFunc_UFUNC(self));
    Py_XDECREF(self->func_obj);
    self->magic_number = NPY_INVALID_MAGIC;
    _pya_free(self);
}


static PyObject *
ufunc_repr(PyUFuncObject *self)
{
    char buf[100];

    sprintf(buf, "<ufunc '%.50s'>", PyUFunc_UFUNC(self)->name);
    return PyUString_FromString(buf);
}


/******************************************************************************
 ***                          UFUNC METHODS                                 ***
 *****************************************************************************/


/*
 * op.outer(a,b) is equivalent to op(a[:,NewAxis,NewAxis,etc.],b)
 * where a has b.ndim NewAxis terms appended.
 *
 * The result has dimensions a.ndim + b.ndim
 */
static PyObject *
ufunc_outer(PyUFuncObject *self, PyObject *args, PyObject *kwds)
{
    int i;
    PyObject *ret;
    PyArrayObject *ap1 = NULL, *ap2 = NULL, *ap_new = NULL;
    PyObject *new_args, *tmp;
    PyObject *shape1, *shape2, *newshape;

    if (PyUFunc_UFUNC(self)->core_enabled) {
        PyErr_Format(PyExc_TypeError,
                     "method outer is not allowed in ufunc with non-trivial"\
                     " signature");
        return NULL;
    }

    if (PyUFunc_UFUNC(self)->nin != 2) {
        PyErr_SetString(PyExc_ValueError,
                        "outer product only supported "\
                        "for binary functions");
        return NULL;
    }

    if (PySequence_Length(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "exactly two arguments expected");
        return NULL;
    }

    tmp = PySequence_GetItem(args, 0);
    if (tmp == NULL) {
        return NULL;
    }
    ap1 = (PyArrayObject *) PyArray_FromObject(tmp, PyArray_NOTYPE, 0, 0);
    Py_DECREF(tmp);
    if (ap1 == NULL) {
        return NULL;
    }
    tmp = PySequence_GetItem(args, 1);
    if (tmp == NULL) {
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromObject(tmp, PyArray_NOTYPE, 0, 0);
    Py_DECREF(tmp);
    if (ap2 == NULL) {
        Py_DECREF(ap1);
        return NULL;
    }
    /* Construct new shape tuple */
    shape1 = PyTuple_New(PyArray_NDIM(ap1));
    if (shape1 == NULL) {
        goto fail;
    }
    for (i = 0; i < PyArray_NDIM(ap1); i++) {
        PyTuple_SET_ITEM(shape1, i,
                PyLong_FromLongLong((longlong)PyArray_DIM(ap1, i)));
    }
    shape2 = PyTuple_New(PyArray_NDIM(ap2));
    for (i = 0; i < PyArray_NDIM(ap2); i++) {
        PyTuple_SET_ITEM(shape2, i, PyInt_FromLong((long) 1));
    }
    if (shape2 == NULL) {
        Py_DECREF(shape1);
        goto fail;
    }
    newshape = PyNumber_Add(shape1, shape2);
    Py_DECREF(shape1);
    Py_DECREF(shape2);
    if (newshape == NULL) {
        goto fail;
    }
    ap_new = (PyArrayObject *)PyArray_Reshape(ap1, newshape);
    Py_DECREF(newshape);
    if (ap_new == NULL) {
        goto fail;
    }
    new_args = Py_BuildValue("(OO)", ap_new, ap2);
    Py_DECREF(ap1);
    Py_DECREF(ap2);
    Py_DECREF(ap_new);
    ret = ufunc_generic_call(self, new_args, kwds);
    Py_DECREF(new_args);
    return ret;

 fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ap_new);
    return NULL;
}


static PyObject *
ufunc_reduce(PyUFuncObject *self, PyObject *args, PyObject *kwds)
{
    return PyUFunc_GenericReduction(self, args, kwds, UFUNC_REDUCE);
}

static PyObject *
ufunc_accumulate(PyUFuncObject *self, PyObject *args, PyObject *kwds)
{
    return PyUFunc_GenericReduction(self, args, kwds, UFUNC_ACCUMULATE);
}

static PyObject *
ufunc_reduceat(PyUFuncObject *self, PyObject *args, PyObject *kwds)
{
    return PyUFunc_GenericReduction(self, args, kwds, UFUNC_REDUCEAT);
}


static struct PyMethodDef ufunc_methods[] = {
    {"reduce",
        (PyCFunction)ufunc_reduce,
        METH_VARARGS | METH_KEYWORDS, NULL },
    {"accumulate",
        (PyCFunction)ufunc_accumulate,
        METH_VARARGS | METH_KEYWORDS, NULL },
    {"reduceat",
        (PyCFunction)ufunc_reduceat,
        METH_VARARGS | METH_KEYWORDS, NULL },
    {"outer",
        (PyCFunction)ufunc_outer,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}           /* sentinel */
};


/******************************************************************************
 ***                           UFUNC GETSET                                 ***
 *****************************************************************************/


/* construct the string y1,y2,...,yn */
static PyObject *
_makeargs(int num, char *ltr, int null_if_none)
{
    PyObject *str;
    int i;

    switch (num) {
    case 0:
        if (null_if_none) {
            return NULL;
        }
        return PyString_FromString("");
    case 1:
        return PyString_FromString(ltr);
    }
    str = PyString_FromFormat("%s1, %s2", ltr, ltr);
    for (i = 3; i <= num; ++i) {
        PyString_ConcatAndDel(&str, PyString_FromFormat(", %s%d", ltr, i));
    }
    return str;
}

static char
_typecharfromnum(int num) {
    NpyArray_Descr *descr;
    char ret;

    descr = NpyArray_DescrFromType(num);
    ret = descr->type;
    Npy_DECREF(descr);
    return ret;
}

static PyObject *
ufunc_get_doc(PyUFuncObject *self)
{
    /*
     * Put docstring first or FindMethod finds it... could so some
     * introspection on name and nin + nout to automate the first part
     * of it the doc string shouldn't need the calling convention
     * construct name(x1, x2, ...,[ out1, out2, ...]) __doc__
     */
    PyObject *outargs, *inargs, *doc;
    outargs = _makeargs(PyUFunc_UFUNC(self)->nout, "out", 1);
    inargs = _makeargs(PyUFunc_UFUNC(self)->nin, "x", 0);
    if (outargs == NULL) {
        doc = PyUString_FromFormat("%s(%s)\n\n%s",
                                   PyUFunc_UFUNC(self)->name,
                                   PyString_AS_STRING(inargs),
                                   PyUFunc_UFUNC(self)->doc);
    }
    else {
        doc = PyUString_FromFormat("%s(%s[, %s])\n\n%s",
                                   PyUFunc_UFUNC(self)->name,
                                   PyString_AS_STRING(inargs),
                                   PyString_AS_STRING(outargs),
                                   PyUFunc_UFUNC(self)->doc);
        Py_DECREF(outargs);
    }
    Py_DECREF(inargs);
    return doc;
}

static PyObject *
ufunc_get_nin(PyUFuncObject *self)
{
    return PyInt_FromLong(PyUFunc_UFUNC(self)->nin);
}

static PyObject *
ufunc_get_nout(PyUFuncObject *self)
{
    return PyInt_FromLong(PyUFunc_UFUNC(self)->nout);
}

static PyObject *
ufunc_get_nargs(PyUFuncObject *self)
{
    return PyInt_FromLong(PyUFunc_UFUNC(self)->nargs);
}

static PyObject *
ufunc_get_ntypes(PyUFuncObject *self)
{
    return PyInt_FromLong(PyUFunc_UFUNC(self)->ntypes);
}

static PyObject *
ufunc_get_types(PyUFuncObject *self)
{
    /* return a list with types grouped input->output */
    PyObject *list;
    PyObject *str;
    int k, j, n, nt = PyUFunc_UFUNC(self)->ntypes;
    int ni = PyUFunc_UFUNC(self)->nin;
    int no = PyUFunc_UFUNC(self)->nout;
    char *t;
    list = PyList_New(nt);
    if (list == NULL) {
        return NULL;
    }
    t = _pya_malloc(no+ni+2);
    n = 0;
    for (k = 0; k < nt; k++) {
        for (j = 0; j<ni; j++) {
            t[j] = _typecharfromnum(PyUFunc_UFUNC(self)->types[n]);
            n++;
        }
        t[ni] = '-';
        t[ni+1] = '>';
        for (j = 0; j < no; j++) {
            t[ni + 2 + j] = _typecharfromnum(PyUFunc_UFUNC(self)->types[n]);
            n++;
        }
        str = PyUString_FromStringAndSize(t, no + ni + 2);
        PyList_SET_ITEM(list, k, str);
    }
    _pya_free(t);
    return list;
}

static PyObject *
ufunc_get_name(PyUFuncObject *self)
{
    return PyUString_FromString(PyUFunc_UFUNC(self)->name);
}

static PyObject *
ufunc_get_identity(PyUFuncObject *self)
{
    switch(PyUFunc_UFUNC(self)->identity) {
    case PyUFunc_One:
        return PyInt_FromLong(1);
    case PyUFunc_Zero:
        return PyInt_FromLong(0);
    }
    return Py_None;
}

static PyObject *
ufunc_get_signature(PyUFuncObject *self)
{
    if (!PyUFunc_UFUNC(self)->core_enabled) {
        Py_RETURN_NONE;
    }
    return PyUString_FromString(PyUFunc_UFUNC(self)->core_signature);
}

#undef _typecharfromnum

/*
 * Docstring is now set from python
 * static char *Ufunctype__doc__ = NULL;
 */
static PyGetSetDef ufunc_getset[] = {
    {"__doc__",
        (getter)ufunc_get_doc,
        NULL, NULL, NULL},
    {"nin",
        (getter)ufunc_get_nin,
        NULL, NULL, NULL},
    {"nout",
        (getter)ufunc_get_nout,
        NULL, NULL, NULL},
    {"nargs",
        (getter)ufunc_get_nargs,
        NULL, NULL, NULL},
    {"ntypes",
        (getter)ufunc_get_ntypes,
        NULL, NULL, NULL},
    {"types",
        (getter)ufunc_get_types,
        NULL, NULL, NULL},
    {"__name__",
        (getter)ufunc_get_name,
        NULL, NULL, NULL},
    {"identity",
        (getter)ufunc_get_identity,
        NULL, NULL, NULL},
    {"signature",
        (getter)ufunc_get_signature,
        NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},  /* Sentinel */
};


/******************************************************************************
 ***                        UFUNC TYPE OBJECT                               ***
 *****************************************************************************/

NPY_NO_EXPORT PyTypeObject PyUFunc_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.ufunc",                              /* tp_name */
    sizeof(PyUFuncObject),                      /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)ufunc_dealloc,                  /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    (reprfunc)ufunc_repr,                       /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    (ternaryfunc)ufunc_generic_call,            /* tp_call */
    (reprfunc)ufunc_repr,                       /* tp_str */
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
    ufunc_methods,                              /* tp_methods */
    0,                                          /* tp_members */
    ufunc_getset,                               /* tp_getset */
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

/* End of code for ufunc objects */
