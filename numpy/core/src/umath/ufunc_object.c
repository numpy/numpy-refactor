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


#define PY_ARRAY_UNIQUE_SYMBOL _npy_umathmodule_ARRAY_API
#define NO_IMPORT_ARRAY

#include "npy_api.h"
#include "npy_iterators.h"
#include <npy_dict.h>
#include <npy_ufunc_object.h>

#include "numpy/noprefix.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/npy_3kcompat.h"

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
_error_handler(int method, char* name, PyObject *errobj, char *errtype, int retstatus, int *first)
{
    PyObject *pyfunc, *ret, *args;
    char msg[100];
    ALLOW_C_API_DEF;

    ALLOW_C_API;
    switch(method) {
    case NPY_UFUNC_ERR_WARN:
        PyOS_snprintf(msg, sizeof(msg), "%s encountered in %s", errtype, name);
        if (PyErr_Warn(PyExc_RuntimeWarning, msg) < 0) {
            goto fail;
        }
        break;
    case NPY_UFUNC_ERR_RAISE:
        PyErr_Format(PyExc_FloatingPointError, "%s encountered in %s",
                errtype, name);
        goto fail;
    case NPY_UFUNC_ERR_CALL:
        pyfunc = errobj;
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
    case NPY_UFUNC_ERR_PRINT:
        if (*first) {
            fprintf(stderr, "Warning: %s encountered in %s\n", errtype, name);
            *first = 0;
        }
        break;
    case NPY_UFUNC_ERR_LOG:
        if (first) {
            *first = 0;
            pyfunc = errobj;
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
    return NpyUFunc_getfperr();
}

#define HANDLEIT(NAME, str) {if (retstatus & NPY_UFUNC_FPE_##NAME) {        \
            handle = errmask & NPY_UFUNC_MASK_##NAME;                       \
            if (handle &&                                               \
                _error_handler(handle >> NPY_UFUNC_SHIFT_##NAME,            \
                               name, errobj, str, retstatus, first) < 0) \
                return -1;                                              \
        }}

/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_handlefperr(char* name, int errmask, PyObject *errobj, int retstatus, int *first)
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
PyUFunc_checkfperr(char* name, int errmask, PyObject *errobj, int *first)
{
    return NpyUFunc_checkfperr(name, errmask, errobj, first);
}


/* Checking the status flag clears it */
/*UFUNC_API*/
NPY_NO_EXPORT void
PyUFunc_clearfperr()
{
    NpyUFunc_clearfperr();
}






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
        return n;
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
        return n;
    }
    else {
        if (PyArray_DescrConverter(type_tup, &dtype) == NPY_FAIL) {
            return -1;
        }
        rtypenums[0] = dtype->descr->type_num;
        Py_DECREF(dtype);
        return 1;
    }
}





#if USE_USE_DEFAULTS==1
static int PyUFunc_NUM_NODEFAULTS = 0;
#endif
static PyObject *PyUFunc_PYVALS_NAME = NULL;


static int
_extract_pyvals(PyObject *ref, int *bufsize,
                int *errmask, PyObject **errobj)
{
    PyObject *retval;

    *errobj = NULL;
    if (!PyList_Check(ref) || (PyList_GET_SIZE(ref)!=3)) {
        PyErr_Format(PyExc_TypeError, "%s must be a length 3 list.",
                     NPY_UFUNC_PYVALS_NAME);
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

    *errobj = retval;
    Py_INCREF(retval);
    return 0;
}



/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_GetPyValues(int *bufsize, int *errmask, PyObject **errobj)
{
    PyObject *thedict;
    PyObject *ref = NULL;

#if USE_USE_DEFAULTS==1
    if (PyUFunc_NUM_NODEFAULTS != 0) {
#endif
        if (PyUFunc_PYVALS_NAME == NULL) {
            PyUFunc_PYVALS_NAME = PyUString_InternFromString(NPY_UFUNC_PYVALS_NAME);
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
        *errmask = NPY_UFUNC_ERR_DEFAULT;
        *errobj = Py_None;
        Py_INCREF(Py_None);
        *bufsize = PyArray_BUFSIZE;
        return 0;
    }
    return _extract_pyvals(ref, bufsize, errmask, errobj);
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
prepare_outputs(NpyUFuncObject *self, NpyArray **mps, PyObject* args)
{
    PyObject *wraparr[NPY_MAXARGS];
    Py_ssize_t nargs;
    int i;
    int result = 0;

    /* Set up prepare functions. */
    nargs = PyTuple_Size(args);
    _find_array_prepare(args, wraparr, self->nin, self->nout);

    /* wrap outputs */
    for (i = 0; i < self->nout; i++) {
        int j = self->nin+i;
        PyObject *wrap;
        PyObject *res;
        wrap = wraparr[i];
        if (wrap != NULL) {
            if (wrap == Py_None) {
                continue;
            }
            res = PyObject_CallFunction(wrap, "O(OOi)",
                                        Npy_INTERFACE(mps[j]), 
                                        Npy_INTERFACE(self), args, i);
            if ((res == NULL) || (res == Py_None) || !PyArray_Check(res)) {
                if (!PyErr_Occurred()){
                    PyErr_SetString(PyExc_TypeError,
                            "__array_prepare__ must return an ndarray or subclass thereof");
                }
                result = -1;
                break;
            }
            Npy_DECREF(mps[j]);
            mps[j] = PyArray_ARRAY(res);
            Npy_INCREF(mps[j]);
            Py_DECREF(res);
        }
    }

    /* Cleanup. */
    for (i=0; i < self->nout; i++) {
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
    int i, result;
    NpyArray *mpsCore[NPY_MAXARGS];
    NPY_BEGIN_THREADS_DEF
    Py_ssize_t nargs=0;
    PyObject *extobj = NULL;
    PyObject *typetup = NULL;
    NpyUFuncObject *self;
    int typenumbuf[NPY_MAXARGS];
    int *rtypenums = NULL;
    int ntypenums = 0;
    char* name = PyUFunc_UFUNC(pySelf)->name ? PyUFunc_UFUNC(pySelf)->name : "";
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
        int ntypenums = get_rtypenums(PyUFunc_UFUNC(pySelf), typetup, 
                                      typenumbuf);
        if (ntypenums < 0) {
            return -1;
        }
        rtypenums = typenumbuf;
    }
    Py_XDECREF(typetup);

    /* Setup error handling. */
    /* TODO: Is extobj ever used? */
    assert(NULL == extobj);
/*    if (extobj == NULL) {
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
    PyUFunc_clearfperr(); */
    
    self = PyUFunc_UFUNC(pySelf);
    
    /* Convert args to arrays in mps. */
    if ((i=convert_args(self, args, mps)) < 0) {
/*        Py_XDECREF(errobj); */
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

    /* Strip the python wrappers from the arrays before passing down. */
    nargs = PyTuple_Size(args);
    for (i=0; i < self->nargs; i++) {
        if (NULL == mps[i]) mpsCore[i] = NULL;
        else {
            mpsCore[i] = PyArray_ARRAY(mps[i]);
            Npy_INCREF(mpsCore[i]);
            Py_DECREF(mps[i]);
        }
    }

    result = NpyUFunc_GenericFunction(self, nargs, mpsCore, 
                                      ntypenums, rtypenums,
                                      originalArgWasObjArray,
                                      (npy_prepare_outputs_func)prepare_outputs, args);

    for (i=0; i < self->nargs; i++) {
        if (NULL == mpsCore[i]) mps[i] = NULL;
        else {
            mps[i] = PyArray_WRAP(mpsCore[i]);
            Py_INCREF(mps[i]);
            Npy_DECREF(mpsCore[i]);
        }
    }
    return result;
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

    if (operation == NPY_UFUNC_REDUCEAT) {
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
    assert(PyArray_ISVALID(mp));
    
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

    if (indices) {
        /* Move reference to core */
        Npy_INCREF(PyArray_ARRAY(indices));
        Py_DECREF(indices);
    }
    ASSIGN_TO_PYARRAY(ret,
        NpyUFunc_GenericReduction(PyUFunc_UFUNC(self), 
                                  PyArray_ARRAY(mp), 
                                  (NULL != indices ? PyArray_ARRAY(indices) : NULL), 
                                  (NULL != out ? PyArray_ARRAY(out) : NULL), 
                                  axis, (NULL != otype) ? otype->descr : NULL, operation));
    
    Py_DECREF(mp);
    Py_XDECREF(otype);
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
        PyUFunc_PYVALS_NAME = PyUString_InternFromString(NPY_UFUNC_PYVALS_NAME);
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
    PyList_SET_ITEM(res, 1, PyInt_FromLong(NPY_UFUNC_ERR_DEFAULT));
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
    res = PyUFunc_GetPyValues(&bufsize, &errmask, &errobj);
    PyUFunc_NUM_NODEFAULTS -= 1;
    if (res < 0) {
        Py_XDECREF(errobj);
        return -1;
    }
    if ((errmask != NPY_UFUNC_ERR_DEFAULT) || (bufsize != PyArray_BUFSIZE)
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
        PyUFunc_PYVALS_NAME = PyUString_InternFromString(NPY_UFUNC_PYVALS_NAME);
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
    Py_INCREF(Npy_INTERFACE(selfCore));
    Npy_DECREF(selfCore);    
    return (PyObject *)Npy_INTERFACE(selfCore);
}



/* Specify that the loop specified by the given index should use the array of
 * input and arrays as the data pointer to the loop.
 */
/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_SetUsesArraysAsData(void **data, size_t i)
{
    return NpyUFunc_SetUsesArraysAsData(data, i);
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


/*
 * This frees the linked-list structure when the CObject
 * is destroyed (removed from the internal dictionary)
*/
static NPY_INLINE void
_free_loop1d_list(NpyUFunc_Loop1d *data)
{
    while (data != NULL) {
        NpyUFunc_Loop1d *next = data->next;
        _pya_free(data->arg_types);
        _pya_free(data);
        data = next;
    }
}

#if PY_VERSION_HEX >= 0x02070000
static void
_loop1d_list_free(PyObject *ptr)
{
    NpyUFunc_Loop1d *data = (NpyUFunc_Loop1d *)PyCapsule_GetPointer(ptr, NULL);
    _free_loop1d_list(data);
}

#else
static void
_loop1d_list_free(void *ptr)
{
    NpyUFunc_Loop1d *data = (NpyUFunc_Loop1d *)ptr;
    _free_loop1d_list(data);
}
#endif



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
    return PyUFunc_GenericReduction(self, args, kwds, NPY_UFUNC_REDUCE);
}

static PyObject *
ufunc_accumulate(PyUFuncObject *self, PyObject *args, PyObject *kwds)
{
    return PyUFunc_GenericReduction(self, args, kwds, NPY_UFUNC_ACCUMULATE);
}

static PyObject *
ufunc_reduceat(PyUFuncObject *self, PyObject *args, PyObject *kwds)
{
    return PyUFunc_GenericReduction(self, args, kwds, NPY_UFUNC_REDUCEAT);
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
    case NpyUFunc_One:
        return PyInt_FromLong(1);
    case NpyUFunc_Zero:
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
