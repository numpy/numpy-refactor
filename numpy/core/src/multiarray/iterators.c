#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "npy_api.h"
#include "npy_iterators.h"
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "conversion_utils.h"

#include "arrayobject.h"
#include "iterators.h"
#include "ctors.h"
#include "common.h"

#define PseudoIndex -1
#define RubberIndex -2
#define SingleIndex -3





/* Callback from the core to construct the PyObject wrapper around an interator. */
NPY_NO_EXPORT int
NpyInterface_IterNewWrapper(NpyArrayIterObject *iter, void **interfaceRet)
{
    PyArrayIterObject *result;

    result = _pya_malloc(sizeof(*result));
    if (result == NULL) {
        *interfaceRet = NULL;
        return NPY_FALSE;
    }

    PyObject_Init((PyObject *)result, &PyArrayIter_Type);
    result->magic_number = NPY_VALID_MAGIC;
    result->iter = iter;

    *interfaceRet = result;
    return NPY_TRUE;
}


NPY_NO_EXPORT int
NpyInterface_MultiIterNewWrapper(NpyArrayMultiIterObject *iter, void **interfaceRet)
{
    PyArrayMultiIterObject *result;

    result = _pya_malloc(sizeof(*result));
    if (result == NULL) {
        *interfaceRet = NULL;
        return NPY_FALSE;
    }

    PyObject_Init((PyObject *)result, &PyArrayMultiIter_Type);
    result->magic_number = NPY_VALID_MAGIC;
    result->iter = iter;

    *interfaceRet = result;
    return NPY_TRUE;
}


NPY_NO_EXPORT int
NpyInterface_NeighborhoodIterNewWrapper(NpyArrayNeighborhoodIterObject *iter, void **interfaceRet)
{
    PyArrayNeighborhoodIterObject *result;

    result = _pya_malloc(sizeof(*result));
    if (result == NULL) {
        *interfaceRet = NULL;
        return NPY_FALSE;
    }

    PyObject_Init((PyObject *)result, &PyArrayNeighborhoodIter_Type);
    result->magic_number = NPY_VALID_MAGIC;
    result->iter = iter;

    *interfaceRet = result;
    return NPY_TRUE;
}

/*NUMPY_API
 * Get Iterator.
 */
NPY_NO_EXPORT PyObject *
PyArray_IterNew(PyObject *obj)
{
    PyArrayObject *ao = (PyArrayObject *)obj;
    NpyArrayIterObject *iter = NULL;

    if (!PyArray_Check(ao)) {
        PyErr_BadInternalCall();
        return NULL;
    }

    iter = NpyArray_IterNew(PyArray_ARRAY(ao));
    if (NULL == iter) {
        return NULL;
    }

    /* Move reference from iter to Npy_INTERFACE(iter) since we are returning the
       interface object. Decref before incref would be unfortunate. */
    Py_INCREF( (PyObject *)Npy_INTERFACE(iter) );
    Npy_DECREF( iter );

    return (PyObject *)Npy_INTERFACE(iter);
}




/*NUMPY_API
 * Get Iterator broadcast to a particular shape
 */
NPY_NO_EXPORT PyObject *
PyArray_BroadcastToShape(PyObject *obj, intp *dims, int nd)
{
    NpyArrayIterObject *iter = NpyArray_BroadcastToShape(PyArray_ARRAY(obj), dims, nd);

    /* Move reference from iter to Npy_INTERFACE(iter) since we are returning the
     interface object. Decref before incref would be unfortunate. */
    Py_INCREF( (PyObject *)Npy_INTERFACE(iter) );
    Npy_DECREF( iter );

    return (PyObject*)Npy_INTERFACE(iter);
}





/*NUMPY_API
 * Get Iterator that iterates over all but one axis (don't use this with
 * PyArray_ITER_GOTO1D).  The axis will be over-written if negative
 * with the axis having the smallest stride.
 */
NPY_NO_EXPORT PyObject *
PyArray_IterAllButAxis(PyObject *obj, int *inaxis)
{
    NpyArrayIterObject *iter;

    if (!PyArray_Check(obj)) {
        PyErr_BadInternalCall();
        return NULL;
    }
    iter = NpyArray_IterAllButAxis(PyArray_ARRAY(obj), inaxis);

    /* Move reference from iter to Npy_INTERFACE(iter) since we are returning the
     interface object. Decref before incref would be unfortunate. */
    Py_INCREF( (PyArrayIterObject *)Npy_INTERFACE(iter) );
    Npy_DECREF( iter );

    return (PyObject*)Npy_INTERFACE(iter);
}



/*NUMPY_API
 * Adjusts previously broadcasted iterators so that the axis with
 * the smallest sum of iterator strides is not iterated over.
 * Returns dimension which is smallest in the range [0,multi->nd).
 * A -1 is returned if multi->nd == 0.
 *
 * don't use with PyArray_ITER_GOTO1D because factors are not adjusted
 */
NPY_NO_EXPORT int
PyArray_RemoveSmallest(PyArrayMultiIterObject *multi)
{
    return NpyArray_RemoveSmallest(multi->iter);
}

/* Returns an array scalar holding the element desired */

static PyObject *
arrayiter_next(PyArrayIterObject *pit)
{
    NpyArrayIterObject *it = pit->iter;
    PyObject *ret;

    if (it->index < it->size) {
        ret = PyArray_ToScalar(it->dataptr, Npy_INTERFACE(it->ao));
        NpyArray_ITER_NEXT(it);
        return ret;
    }
    return NULL;
}

static void
arrayiter_dealloc(PyArrayIterObject *it)
{
    Npy_DEALLOC(it->iter);

    it->magic_number = NPY_INVALID_MAGIC;
    _pya_free(it);
}

static Py_ssize_t
iter_length(PyArrayIterObject *self)
{
    return self->iter->size;
}


NPY_NO_EXPORT PyObject* npy_iter_subscript(NpyArrayIterObject* self, PyObject* ind);

/* Always returns arrays */
NPY_NO_EXPORT PyObject *
iter_subscript(PyArrayIterObject *self, PyObject *ind)
{
    return npy_iter_subscript(self->iter, ind);
}

NPY_NO_EXPORT PyObject*
npy_iter_subscript(NpyArrayIterObject* self, PyObject* ind)
{
    NpyIndex indexes[NPY_MAXDIMS];
    int n;
    NpyArray *result;

    n = PyArray_IndexConverter(ind, indexes);
    if (n < 0) {
        return NULL;
    }

    result = NpyArray_IterSubscript(self, indexes, n);
    NpyArray_IndexDealloc(indexes, n);

    if (result == NULL) {
        return NULL;
    }

    /* Move the ref and return. */
    Py_INCREF(Npy_INTERFACE(result));
    Npy_DECREF(result);

    return PyArray_Return(Npy_INTERFACE(result));
}


NPY_NO_EXPORT int
iter_ass_subscript(PyArrayIterObject *self, PyObject *ind, PyObject *val)
{
    return npy_iter_ass_subscript(self->iter, ind, val);
}

static int
npy_iter_ass_single(NpyArrayIterObject *self, npy_intp index,
                    PyObject *val)
{
    if (index < 0) {
        index += self->size;
    }
    if (index < 0 || index >= self->size) {
        PyErr_Format(PyExc_IndexError,
                     "index out of bounds 0<=index<%"NPY_INTP_FMT, self->size);
        return -1;
    }
    NpyArray_ITER_RESET(self);
    NpyArray_ITER_GOTO1D(self, index);
    return self->ao->descr->f->setitem(val, self->dataptr, self->ao);
}

NPY_NO_EXPORT int
npy_iter_ass_subscript(NpyArrayIterObject* self, PyObject* ind, PyObject* val)
{
    int result;
    NpyIndex indexes[NPY_MAXDIMS];
    PyArrayObject *arr_val = NULL;
    int n;

    n = PyArray_IndexConverter(ind, indexes);
    if (n < 0) {
        return -1;
    }

    /* Special cases for single assignment. */
    if (n == 1) {
        if (indexes[0].type == NPY_INDEX_INTP) {
#undef intp
            result = npy_iter_ass_single(self, indexes[0].index.intp, val);
#define intp npy_intp
            goto finish;
        } else if (indexes[0].type == NPY_INDEX_BOOL) {
            if (indexes[0].index.boolean) {
                result = npy_iter_ass_single(self, 0, val);
            } else {
                result = 0;
            }
            goto finish;
        }
    }

    Npy_INCREF(self->ao->descr);
    arr_val = (PyArrayObject *)
        PyArray_FromAnyUnwrap(val, self->ao->descr, 0, 0, 0, NULL);
    if (arr_val == NULL) {
        result = -1;
        goto finish;
    }

    result = NpyArray_IterSubscriptAssign(self, indexes, n, 
                                          PyArray_ARRAY(arr_val));

 finish:
    Py_XDECREF(arr_val);
    NpyArray_IndexDealloc(indexes, n);

    return result;
}


static PyMappingMethods iter_as_mapping = {
#if PY_VERSION_HEX >= 0x02050000
    (lenfunc)iter_length,                   /*mp_length*/
#else
    (inquiry)iter_length,                   /*mp_length*/
#endif
    (binaryfunc)iter_subscript,             /*mp_subscript*/
    (objobjargproc)iter_ass_subscript,      /*mp_ass_subscript*/
};



static PyObject *
iter_array(PyArrayIterObject *pit, PyObject *NPY_UNUSED(op))
{
    RETURN_PYARRAY(NpyArray_FlatView(pit->iter->ao));
}

static PyObject *
iter_copy(PyArrayIterObject *it, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    RETURN_PYARRAY(NpyArray_Flatten(it->iter->ao, 0));
}

static PyMethodDef iter_methods[] = {
    /* to get array */
    {"__array__",
        (PyCFunction)iter_array,
        METH_VARARGS, NULL},
    {"copy",
        (PyCFunction)iter_copy,
        METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}           /* sentinel */
};

static PyObject *
iter_richcompare(PyArrayIterObject *self, PyObject *other, int cmp_op)
{
    PyArrayObject *new;
    PyObject *ret;
    new = (PyArrayObject *)iter_array(self, NULL);
    if (new == NULL) {
        return NULL;
    }
    ret = array_richcompare(new, other, cmp_op);
    Py_DECREF(new);
    return ret;
}


static PyObject *
iter_coords_get(PyArrayIterObject *pself)
{
    NpyArrayIterObject *self = pself->iter;
    int nd;
    nd = self->ao->nd;
    if (self->contiguous) {
        /*
         * coordinates not kept track of ---
         * need to generate from index
         */
        intp val;
        int i;
        val = self->index;
        for (i = 0; i < nd; i++) {
            if (self->factors[i] != 0) {
                self->coordinates[i] = val / self->factors[i];
                val = val % self->factors[i];
            } else {
                self->coordinates[i] = 0;
            }
        }
    }
    return PyArray_IntTupleFromIntp(nd, self->coordinates);
}

static PyObject *
iter_base_get(PyArrayIterObject* self)
{
    Npy_INCREF(self->iter->ao);
    return (PyObject *)self->iter->ao;
}

static PyObject *
iter_index_get(PyArrayIterObject *self)
{
#if NPY_SIZEOF_INTP <= NPY_SIZEOF_LONG
    return PyInt_FromLong((long) self->iter->index);
#else
    if (self->size < MAX_LONG) {
        return PyInt_FromLong((long) self->iter->index);
    }
    else {
        return PyLong_FromLongLong((longlong) self->iter->index);
    }
#endif
}


static PyGetSetDef iter_getsets[] = {
    {"coords",
        (getter)iter_coords_get,
        NULL,
        NULL, NULL},
    {"base",
        (getter)iter_base_get,
        NULL,
        NULL, NULL},
    {"index",
        (getter)iter_index_get,
        NULL,
        NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

NPY_NO_EXPORT PyTypeObject PyArrayIter_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.flatiter",                           /* tp_name */
    sizeof(PyArrayIterObject),                  /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)arrayiter_dealloc,              /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    &iter_as_mapping,                           /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    (richcmpfunc)iter_richcompare,              /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    (iternextfunc)arrayiter_next,               /* tp_iternext */
    iter_methods,                               /* tp_methods */
    0,                                          /* tp_members */
    iter_getsets,                               /* tp_getset */
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

/** END of Array Iterator **/

/* Adjust dimensionality and strides for index object iterators
   --- i.e. broadcast
*/
/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_Broadcast(PyArrayMultiIterObject *mit)
{
    return NpyArray_Broadcast(mit->iter);
}


static PyObject*
PyArray_vMultiIterFromObjects(PyObject **mps, int n, int nadd, va_list va)
{
    PyArrayMultiIterObject* result = NULL;
    NpyArrayMultiIterObject* multi;
    NpyArray* arrays[NPY_MAXARGS];
    int ntot, i;
    int err = 0;

    /* Check the arg count. */
    ntot = n + nadd;
    if (ntot < 2 || ntot > NPY_MAXARGS) {
        PyErr_Format(PyExc_ValueError,
                     "Need between 2 and (%d) "                 \
                     "array objects (inclusive).", NPY_MAXARGS);
        return NULL;
    }

    /* Convert to arrays. */
    for (i=0; i<ntot; i++) {
        if (err) {
            arrays[i] = NULL;
        }
        else {
            if (i < n) {
                PyObject *tmp = PyArray_FROM_O(mps[i]);
                arrays[i] = PyArray_ARRAY(tmp);
                Npy_INCREF(arrays[i]);
                Py_DECREF(tmp);
            } else {
                PyObject* arg = va_arg(va, PyObject*);
                PyObject *tmp = PyArray_FROM_O(arg);
                arrays[i] = PyArray_ARRAY(tmp);
                Npy_INCREF(arrays[i]);
                Py_DECREF(tmp);
            }
            if (arrays[i] == NULL) {
                err = 1;
            }
        }
    }

    if (err) {
        goto finish;
    }

    multi = NpyArray_MultiIterFromArrays(arrays, ntot, 0);
    if (multi == NULL) {
        goto finish;
    }

    /* Move the reference from the core object to the interface. */
    result = (PyArrayMultiIterObject *)Npy_INTERFACE(multi);
    Py_INCREF(result);
    Npy_DECREF(multi);


  finish:
    for (i=0; i<ntot; i++) {
        Npy_XDECREF(arrays[i]);
    }
    return (PyObject*) result;
}

/*NUMPY_API
 * Get MultiIterator from array of Python objects and any additional
 *
 * PyObject **mps -- array of PyObjects
 * int n - number of PyObjects in the array
 * int nadd - number of additional arrays to include in the iterator.
 *
 * Returns a multi-iterator object.
 */
NPY_NO_EXPORT PyObject *
PyArray_MultiIterFromObjects(PyObject **mps, int n, int nadd, ...)
{
    PyObject* result;

    va_list va;
    va_start(va, nadd);
    result = PyArray_vMultiIterFromObjects(mps, n, nadd, va);
    va_end(va);

    return result;
}


/*NUMPY_API
 * Get MultiIterator,
 */
NPY_NO_EXPORT PyObject *
PyArray_MultiIterNew(int n, ...)
{
    PyObject* result;
    va_list va;

    va_start(va, n);
    result = PyArray_vMultiIterFromObjects(NULL, 0, n, va);
    va_end(va);

    return result;
}

static PyObject *
arraymultiter_new(PyTypeObject *NPY_UNUSED(subtype), PyObject *args, PyObject *kwds)
{

    Py_ssize_t n, i;
    NpyArrayMultiIterObject *multi;
    PyArrayObject *arr;

    if (kwds != NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "keyword arguments not accepted.");
        return NULL;
    }

    n = PyTuple_Size(args);
    if (n < 2 || n > NPY_MAXARGS) {
        if (PyErr_Occurred()) {
            return NULL;
        }
        PyErr_Format(PyExc_ValueError,
                     "Need at least two and fewer than (%d) "   \
                     "array objects.", NPY_MAXARGS);
        return NULL;
    }

    /* TODO: Shouldn't this just call PyArray_MultiIterFromObjects? */
    multi = NpyArray_MultiIterNew();
    multi->numiter = n;
    multi->index = 0;
    for (i = 0; i < n; i++) {
        multi->iters[i] = NULL;
    }
    for (i = 0; i < n; i++) {
        arr = (PyArrayObject* )PyArray_FromAny(PyTuple_GET_ITEM(args, i),
                                               NULL, 0, 0, 0, NULL);
        if (arr == NULL) {
            goto fail;
        }
        if ((multi->iters[i] = NpyArray_IterNew(PyArray_ARRAY(arr)))
                == NULL) {
            goto fail;
        }
        Py_DECREF(arr);
    }
    if (NpyArray_Broadcast(multi) < 0) {
        goto fail;
    }
    NpyArray_MultiIter_RESET(multi);

    /* Move the reference from the core to the interface. */
    Py_INCREF( (PyObject *)Npy_INTERFACE(multi) );
    Npy_DECREF( multi );
    return (PyObject *)Npy_INTERFACE(multi);

 fail:
    Npy_DECREF(multi);
    return NULL;
}

static PyObject *
arraymultiter_next(PyArrayMultiIterObject *pmulti)
{
    NpyArrayMultiIterObject* multi = pmulti->iter;
    PyObject *ret;
    int i, n;

    n = multi->numiter;
    ret = PyTuple_New(n);
    if (ret == NULL) {
        return NULL;
    }
    if (multi->index < multi->size) {
        for (i = 0; i < n; i++) {
            NpyArrayIterObject *it=multi->iters[i];
            PyTuple_SET_ITEM(ret, i,
                             PyArray_ToScalar(it->dataptr,
                                              Npy_INTERFACE(it->ao)));
            NpyArray_ITER_NEXT(it);
        }
        multi->index++;
        return ret;
    }
    Py_DECREF(ret);
    return NULL;
}

static void
arraymultiter_dealloc(PyArrayMultiIterObject *multi)
{
    Npy_DEALLOC(multi->iter);
    multi->magic_number = NPY_INVALID_MAGIC;
    Py_TYPE(multi)->tp_free((PyObject *)multi);
}

static PyObject *
arraymultiter_size_get(PyArrayMultiIterObject *self)
{
#if NPY_SIZEOF_INTP <= NPY_SIZEOF_LONG
    return PyInt_FromLong((long) self->iter->size);
#else
    if (self->size < MAX_LONG) {
        return PyInt_FromLong((long) self->iter->size);
    }
    else {
        return PyLong_FromLongLong((longlong) self->iter->size);
    }
#endif
}

static PyObject *
arraymultiter_index_get(PyArrayMultiIterObject *self)
{
#if NPY_SIZEOF_INTP <= NPY_SIZEOF_LONG
    return PyInt_FromLong((long) self->iter->index);
#else
    if (self->size < MAX_LONG) {
        return PyInt_FromLong((long) self->iter->index);
    }
    else {
        return PyLong_FromLongLong((longlong) self->iter->index);
    }
#endif
}

static PyObject *
arraymultiter_shape_get(PyArrayMultiIterObject *self)
{
    return PyArray_IntTupleFromIntp(self->iter->nd, self->iter->dimensions);
}

static PyObject *
arraymultiter_iters_get(PyArrayMultiIterObject *self)
{
    PyObject *res;
    int i, n;

    n = self->iter->numiter;
    res = PyTuple_New(n);
    if (res == NULL) {
        return res;
    }
    for (i = 0; i < n; i++) {
        /* Add to tuple. */
        PyTuple_SET_ITEM(res, i, (PyObject *)Npy_INTERFACE(self->iter->iters[i]));
    }

    return res;
}

static PyObject*
arraymultiter_numiter_get(PyArrayMultiIterObject* multi)
{
    return PyInt_FromLong((long) multi->iter->numiter);
}

static PyObject*
arraymultiter_nd_get(PyArrayMultiIterObject* multi)
{
    return PyInt_FromLong((long) multi->iter->nd);
}

static PyGetSetDef arraymultiter_getsetlist[] = {
    {"size",
        (getter)arraymultiter_size_get,
        NULL,
        NULL, NULL},
    {"index",
        (getter)arraymultiter_index_get,
        NULL,
        NULL, NULL},
    {"shape",
        (getter)arraymultiter_shape_get,
        NULL,
        NULL, NULL},
    {"iters",
        (getter)arraymultiter_iters_get,
        NULL,
        NULL, NULL},
    {"numiter",
        (getter)arraymultiter_numiter_get,
        NULL,
        NULL, NULL},
    {"nd",
        (getter)arraymultiter_nd_get,
        NULL,
        NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyObject *
arraymultiter_reset(PyArrayMultiIterObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    PyArray_MultiIter_RESET(self);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef arraymultiter_methods[] = {
    {"reset",
        (PyCFunction) arraymultiter_reset,
        METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL},      /* sentinal */
};

NPY_NO_EXPORT PyTypeObject PyArrayMultiIter_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.broadcast",                          /* tp_name */
    sizeof(PyArrayMultiIterObject),             /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)arraymultiter_dealloc,          /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
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
    (iternextfunc)arraymultiter_next,           /* tp_iternext */
    arraymultiter_methods,                      /* tp_methods */
    0,                                          /* tp_members */
    arraymultiter_getsetlist,                   /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)0,                                /* tp_init */
    0,                                          /* tp_alloc */
    arraymultiter_new,                          /* tp_new */
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

/*========================= Neighborhood iterator ======================*/

/* TODO: Duplicate of code in npy_iterator.c - need to refactor */
static char* _set_constant(NpyArray* ao, NpyArray *fill)
{
    char *ret;
    int storeflags, st;

    ret = (char *)PyDataMem_NEW(ao->descr->elsize);
    if (ret == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }

    if (NpyArray_ISOBJECT(ao)) {                /* TODO: Double check this case, memcpy of a pointer? Is ISOBJECT still correct? */
        memcpy(ret, fill->data, sizeof(PyObject*));
        Py_INCREF(*(PyObject**)ret);     /* TODO: What are the possible object types for ret? */
    } else {
        /* Non-object types */
        storeflags = ao->flags;
        ao->flags |= NPY_BEHAVED;
        st = ao->descr->f->setitem(Npy_INTERFACE(fill), ret, ao);
        ao->flags = storeflags;

        if (st < 0) {
            PyDataMem_FREE(ret);
            return NULL;
        }
    }

    return ret;
}


static void decref_and_free(void* p)
{
    Py_DECREF(*(PyObject **)p);
    free(p);
}

/*
 * fill and x->ao should have equivalent types
 */
/*NUMPY_API
 * A Neighborhood Iterator object.
*/
NPY_NO_EXPORT PyObject*
PyArray_NeighborhoodIterNew(PyArrayIterObject *x, intp *bounds,
                            int mode, PyArrayObject* fill)
{
    NpyArrayNeighborhoodIterObject *coreRet;
    void *fillptr = NULL;
    npy_free_func freefill = NULL;

    switch (mode) {
        case NPY_NEIGHBORHOOD_ITER_ZERO_PADDING:
            fillptr = PyArray_Zero(Npy_INTERFACE(x->iter->ao));
            freefill = free;
            mode = NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING;
            break;
        case NPY_NEIGHBORHOOD_ITER_ONE_PADDING:
            fillptr = PyArray_One(Npy_INTERFACE(x->iter->ao));
            freefill = free;
            mode = NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING;
            break;
        case NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING:
            fillptr = _set_constant(x->iter->ao, PyArray_ARRAY(fill));
            if (!NpyArray_ISOBJECT(x->iter->ao)) {
                freefill = free;
            } else {
                freefill = decref_and_free;
            }
            break;
        case NPY_NEIGHBORHOOD_ITER_MIRROR_PADDING:
        case NPY_NEIGHBORHOOD_ITER_CIRCULAR_PADDING:
            break;
        default:
            PyErr_SetString(PyExc_ValueError, "Unsupported padding mode");
            return NULL;
    }
    coreRet = NpyArray_NeighborhoodIterNew(x->iter, bounds, mode,
                                           fillptr, freefill);
    if (NULL == coreRet) {
        return NULL;
    }

    /* Move the reference from the core object to the interface obj. */
    Py_INCREF( (PyObject *)Npy_INTERFACE(coreRet) );
    Npy_DECREF( coreRet );

    return (PyObject *)Npy_INTERFACE(coreRet);
}


static void neighiter_dealloc(PyArrayNeighborhoodIterObject* iter)
{
    Npy_DEALLOC(iter->iter);
    iter->magic_number = NPY_INVALID_MAGIC;
    _pya_free((PyArrayObject*)iter);
}

NPY_NO_EXPORT PyTypeObject PyArrayNeighborhoodIter_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.neigh_internal_iter",                /* tp_name*/
    sizeof(PyArrayNeighborhoodIterObject),      /* tp_basicsize*/
    0,                                          /* tp_itemsize*/
    (destructor)neighiter_dealloc,              /* tp_dealloc*/
    0,                                          /* tp_print*/
    0,                                          /* tp_getattr*/
    0,                                          /* tp_setattr*/
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    0,                                          /* tp_repr*/
    0,                                          /* tp_as_number*/
    0,                                          /* tp_as_sequence*/
    0,                                          /* tp_as_mapping*/
    0,                                          /* tp_hash */
    0,                                          /* tp_call*/
    0,                                          /* tp_str*/
    0,                                          /* tp_getattro*/
    0,                                          /* tp_setattro*/
    0,                                          /* tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,                         /* tp_flags*/
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    (iternextfunc)0,                            /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)0,                                /* tp_init */
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
