#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "numpy/numpy_api.h"

#include "npy_config.h"

#include "npy_3kcompat.h"

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






NPY_NO_EXPORT intp
parse_subindex(PyObject *op, intp *step_size, intp *n_steps, intp max)
{
    intp index;

    if (op == Py_None) {
        *n_steps = PseudoIndex;
        index = 0;
    }
    else if (op == Py_Ellipsis) {
        *n_steps = RubberIndex;
        index = 0;
    }
    else if (PySlice_Check(op)) {
        intp stop;
        if (slice_GetIndices((PySliceObject *)op, max,
                             &index, &stop, step_size, n_steps) < 0) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_IndexError,
                                "invalid slice");
            }
            goto fail;
        }
        if (*n_steps <= 0) {
            *n_steps = 0;
            *step_size = 1;
            index = 0;
        }
    }
    else {
        index = PyArray_PyIntAsIntp(op);
        if (error_converting(index)) {
            PyErr_SetString(PyExc_IndexError,
                            "each subindex must be either a "\
                            "slice, an integer, Ellipsis, or "\
                            "newaxis");
            goto fail;
        }
        *n_steps = SingleIndex;
        *step_size = 0;
        if (index < 0) {
            index += max;
        }
        if (index >= max || index < 0) {
            PyErr_SetString(PyExc_IndexError, "invalid index");
            goto fail;
        }
    }
    return index;

 fail:
    return -1;
}


NPY_NO_EXPORT int
parse_index(PyArrayObject *self, PyObject *op,
            intp *dimensions, intp *strides, intp *offset_ptr)
{
    int i, j, n;
    int nd_old, nd_new, n_add, n_pseudo;
    intp n_steps, start, offset, step_size;
    PyObject *op1 = NULL;
    int is_slice;

    if (PySlice_Check(op) || op == Py_Ellipsis || op == Py_None) {
        n = 1;
        op1 = op;
        Py_INCREF(op);
        /* this relies on the fact that n==1 for loop below */
        is_slice = 1;
    }
    else {
        if (!PySequence_Check(op)) {
            PyErr_SetString(PyExc_IndexError,
                            "index must be either an int "\
                            "or a sequence");
            return -1;
        }
        n = PySequence_Length(op);
        is_slice = 0;
    }

    nd_old = nd_new = 0;

    offset = 0;
    for (i = 0; i < n; i++) {
        if (!is_slice) {
            if (!(op1=PySequence_GetItem(op, i))) {
                PyErr_SetString(PyExc_IndexError,
                                "invalid index");
                return -1;
            }
        }
        start = parse_subindex(op1, &step_size, &n_steps,
                               nd_old < PyArray_NDIM(self) ?
                               PyArray_DIM(self, nd_old) : 0);
        Py_DECREF(op1);
        if (start == -1) {
            break;
        }
        if (n_steps == PseudoIndex) {
            dimensions[nd_new] = 1; strides[nd_new] = 0;
            nd_new++;
        }
        else {
            if (n_steps == RubberIndex) {
                for (j = i + 1, n_pseudo = 0; j < n; j++) {
                    op1 = PySequence_GetItem(op, j);
                    if (op1 == Py_None) {
                        n_pseudo++;
                    }
                    Py_DECREF(op1);
                }
                n_add = PyArray_NDIM(self)-(n-i-n_pseudo-1+nd_old);
                if (n_add < 0) {
                    PyErr_SetString(PyExc_IndexError,
                                    "too many indices");
                    return -1;
                }
                for (j = 0; j < n_add; j++) {
                    dimensions[nd_new] = \
                        PyArray_DIM(self, nd_old);
                    strides[nd_new] = \
                        PyArray_STRIDE(self, nd_old);
                    nd_new++; nd_old++;
                }
            }
            else {
                if (nd_old >= PyArray_NDIM(self)) {
                    PyErr_SetString(PyExc_IndexError,
                                    "too many indices");
                    return -1;
                }
                offset += PyArray_STRIDE(self, nd_old)*start;
                nd_old++;
                if (n_steps != SingleIndex) {
                    dimensions[nd_new] = n_steps;
                    strides[nd_new] = step_size * \
                        PyArray_STRIDE(self, nd_old-1);
                    nd_new++;
                }
            }
        }
    }
    if (i < n) {
        return -1;
    }
    n_add = PyArray_NDIM(self)-nd_old;
    for (j = 0; j < n_add; j++) {
        dimensions[nd_new] = PyArray_DIM(self, nd_old);
        strides[nd_new] = PyArray_STRIDE(self, nd_old);
        nd_new++;
        nd_old++;
    }
    *offset_ptr = offset;
    return nd_new;
}

static int
slice_coerce_index(PyObject *o, intp *v)
{
    *v = PyArray_PyIntAsIntp(o);
    if (error_converting(*v)) {
        PyErr_Clear();
        return 0;
    }
    return 1;
}

/* This is basically PySlice_GetIndicesEx, but with our coercion
 * of indices to integers (plus, that function is new in Python 2.3) */
NPY_NO_EXPORT int
slice_GetIndices(PySliceObject *r, intp length,
                 intp *start, intp *stop, intp *step,
                 intp *slicelength)
{
    intp defstop;

    if (r->step == Py_None) {
        *step = 1;
    }
    else {
        if (!slice_coerce_index(r->step, step)) {
            return -1;
        }
        if (*step == 0) {
            PyErr_SetString(PyExc_ValueError,
                            "slice step cannot be zero");
            return -1;
        }
    }
    /* defstart = *step < 0 ? length - 1 : 0; */
    defstop = *step < 0 ? -1 : length;
    if (r->start == Py_None) {
        *start = *step < 0 ? length-1 : 0;
    }
    else {
        if (!slice_coerce_index(r->start, start)) {
            return -1;
        }
        if (*start < 0) {
            *start += length;
        }
        if (*start < 0) {
            *start = (*step < 0) ? -1 : 0;
        }
        if (*start >= length) {
            *start = (*step < 0) ? length - 1 : length;
        }
    }

    if (r->stop == Py_None) {
        *stop = defstop;
    }
    else {
        if (!slice_coerce_index(r->stop, stop)) {
            return -1;
        }
        if (*stop < 0) {
            *stop += length;
        }
        if (*stop < 0) {
            *stop = -1;
        }
        if (*stop > length) {
            *stop = length;
        }
    }

    if ((*step < 0 && *stop >= *start) ||
        (*step > 0 && *start >= *stop)) {
        *slicelength = 0;
    }
    else if (*step < 0) {
        *slicelength = (*stop - *start + 1) / (*step) + 1;
    }
    else {
        *slicelength = (*stop - *start - 1) / (*step) + 1;
    }

    return 0;
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
    Py_INCREF( Npy_INTERFACE(iter) );
    _Npy_DECREF( iter );
    
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
    Py_INCREF( Npy_INTERFACE(iter) );
    _Npy_DECREF( iter );
    
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
    Py_INCREF( Npy_INTERFACE(iter) );
    _Npy_DECREF( iter );
    
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


static PyObject *
iter_subscript_Bool(NpyArrayIterObject *self, PyArrayObject *ind)
{
    intp index, strides;
    int itemsize;
    intp count = 0;
    char *dptr, *optr;
    PyObject *r;
    int swap;
    PyArray_CopySwapFunc *copyswap;


    if (PyArray_NDIM(ind) != 1) {
        PyErr_SetString(PyExc_ValueError,
                        "boolean index array should have 1 dimension");
        return NULL;
    }
    index = PyArray_DIM(ind, 0);
    if (index > self->size) {
        PyErr_SetString(PyExc_ValueError,
                        "too many boolean indices");
        return NULL;
    }

    strides = PyArray_STRIDE(ind, 0);
    dptr = PyArray_BYTES(ind);
    /* Get size of return array */
    while (index--) {
        if (*((Bool *)dptr) != 0) {
            count++;
        }
        dptr += strides;
    }
    itemsize = self->ao->descr->elsize;
    _Npy_INCREF(self->ao->descr);
    ASSIGN_TO_PYARRAY(r, 
                      NpyArray_NewFromDescr(self->ao->descr, 1, &count,
                                            NULL, NULL,
                                            0, NPY_FALSE, NULL, 
                                            Npy_INTERFACE(self->ao)));
    if (r == NULL) {
        return NULL;
    }
    /* Set up loop */
    optr = PyArray_DATA(r);
    index = PyArray_DIM(ind, 0);
    dptr = PyArray_BYTES(ind);
    copyswap = self->ao->descr->f->copyswap;
    /* Loop over Boolean array */
    swap = (NpyArray_ISNOTSWAPPED(self->ao) != PyArray_ISNOTSWAPPED(r));
    while (index--) {
        if (*((Bool *)dptr) != 0) {
            copyswap(optr, self->dataptr, swap, self->ao);
            optr += itemsize;
        }
        dptr += strides;
        NpyArray_ITER_NEXT(self);
    }
    NpyArray_ITER_RESET(self);
    return r;
}

static PyObject *
iter_subscript_int(NpyArrayIterObject *self, PyArrayObject *ind)
{
    intp num;
    PyObject *r;
    NpyArrayIterObject *ind_it;
    int itemsize;
    int swap;
    char *optr;
    intp index;
    PyArray_CopySwapFunc *copyswap;

    itemsize = self->ao->descr->elsize;
    if (PyArray_NDIM(ind) == 0) {
        num = *((intp *)PyArray_BYTES(ind));
        if (num < 0) {
            num += self->size;
        }
        if (num < 0 || num >= self->size) {
            PyErr_Format(PyExc_IndexError,
                         "index %"INTP_FMT" out of bounds"   \
                         " 0<=index<%"INTP_FMT,
                         num, self->size);
            r = NULL;
        }
        else {
            NpyArray_ITER_GOTO1D(self, num);
            r = PyArray_ToScalar(self->dataptr, Npy_INTERFACE(self->ao));
        }
        NpyArray_ITER_RESET(self);
        return r;
    }

    _Npy_INCREF(self->ao->descr);
    ASSIGN_TO_PYARRAY(r, NpyArray_NewFromDescr(self->ao->descr,
                              PyArray_NDIM(ind), PyArray_DIMS(ind),
                              NULL, NULL,
                              0, NPY_FALSE, NULL, Npy_INTERFACE(self->ao)));
    if (r == NULL) {
        return NULL;
    }
    optr = PyArray_DATA(r);
    ind_it = NpyArray_IterNew(PyArray_ARRAY(ind));
    if (ind_it == NULL) {
        Py_DECREF(r);
        return NULL;
    }
    index = ind_it->size;
    copyswap = PyArray_DESCR(r)->f->copyswap;
    swap = (PyArray_ISNOTSWAPPED(r) != NpyArray_ISNOTSWAPPED(self->ao));
    while (index--) {
        num = *((intp *)(ind_it->dataptr));
        if (num < 0) {
            num += self->size;
        }
        if (num < 0 || num >= self->size) {
            PyErr_Format(PyExc_IndexError,
                         "index %"INTP_FMT" out of bounds" \
                         " 0<=index<%"INTP_FMT,
                         num, self->size);
            _Npy_DECREF(ind_it);
            Py_DECREF(r);
            NpyArray_ITER_RESET(self);
            return NULL;
        }
        NpyArray_ITER_GOTO1D(self, num);
        copyswap(optr, self->dataptr, swap, PyArray_ARRAY(r));
        optr += itemsize;
        NpyArray_ITER_NEXT(ind_it);
    }
    _Npy_DECREF(ind_it);
    NpyArray_ITER_RESET(self);
    return r;
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
    NpyArray_Descr *indtype = NULL;
    intp start, step_size;
    intp n_steps;
    PyObject *r;
    char *dptr;
    int size;
    PyObject *obj = NULL;
    PyArray_CopySwapFunc *copyswap;

    if (ind == Py_Ellipsis) {
        ind = PySlice_New(NULL, NULL, NULL);
        obj = npy_iter_subscript(self, ind);
        Py_DECREF(ind);
        return obj;
    }
    if (PyTuple_Check(ind)) {
        int len;
        len = PyTuple_GET_SIZE(ind);
        if (len > 1) {
            goto fail;
        }
        if (len == 0) {
            _Npy_INCREF(self->ao);
            RETURN_PYARRAY(self->ao);
        }
        ind = PyTuple_GET_ITEM(ind, 0);
    }

    /*
     * Tuples >1d not accepted --- i.e. no newaxis
     * Could implement this with adjusted strides and dimensions in iterator
     * Check for Boolean -- this is first becasue Bool is a subclass of Int
     */
    NpyArray_ITER_RESET(self);

    if (PyBool_Check(ind)) {
        if (PyObject_IsTrue(ind)) {
            return PyArray_ToScalar(self->dataptr, Npy_INTERFACE(self->ao));
        }
        else { /* empty array */
            intp ii = 0;
            _Npy_INCREF(self->ao->descr);
            ASSIGN_TO_PYARRAY(r, NpyArray_NewFromDescr(self->ao->descr,      /* TODO: Wrap array return */
                                      1, &ii,
                                      NULL, NULL, 0,
                                      NPY_FALSE, NULL,
                                      Npy_INTERFACE(self->ao)));
            return r;
        }
    }

    /* Check for Integer or Slice */
    if (PyLong_Check(ind) || PyInt_Check(ind) || PySlice_Check(ind)) {
        start = parse_subindex(ind, &step_size, &n_steps,
                               self->size);
        if (start == -1) {
            goto fail;
        }
        if (n_steps == RubberIndex || n_steps == PseudoIndex) {
            PyErr_SetString(PyExc_IndexError,
                            "cannot use Ellipsis or newaxes here");
            goto fail;
        }
        NpyArray_ITER_GOTO1D(self, start)
            if (n_steps == SingleIndex) { /* Integer */
                r = PyArray_ToScalar(self->dataptr, Npy_INTERFACE(self->ao));
                NpyArray_ITER_RESET(self);
                return r;
            }
        size = self->ao->descr->elsize;
        _Npy_INCREF(self->ao->descr);
        ASSIGN_TO_PYARRAY(r, NpyArray_NewFromDescr(self->ao->descr,
                                  1, &n_steps,
                                  NULL, NULL,
                                  0, NPY_FALSE, 
                                  NULL, Npy_INTERFACE(self->ao)));
        if (r == NULL) {
            goto fail;
        }
        dptr = PyArray_DATA(r);
        copyswap = PyArray_DESCR(r)->f->copyswap;
        while (n_steps--) {
            copyswap(dptr, self->dataptr, 0, PyArray_ARRAY(r));
            start += step_size;
            NpyArray_ITER_GOTO1D(self, start)
                dptr += size;
        }
        NpyArray_ITER_RESET(self);
        return r;
    }

    /* convert to INTP array if Integer array scalar or List */
    indtype = NpyArray_DescrFromType(PyArray_INTP);
    if (PyArray_IsScalar(ind, Integer) || PyList_Check(ind)) {
        _Npy_INCREF(indtype);
        obj = PyArray_FromAnyUnwrap(ind, indtype, 0, 0, FORCECAST, NULL);
        if (obj == NULL) {
            goto fail;
        }
    }
    else {
        Py_INCREF(ind);
        obj = ind;
    }

    if (PyArray_Check(obj)) {
        /* Check for Boolean object */
        if (PyArray_TYPE(obj)==PyArray_BOOL) {
            r = iter_subscript_Bool(self, (PyArrayObject *)obj);
            _Npy_DECREF(indtype);
        }
        /* Check for integer array */
        else if (PyArray_ISINTEGER(obj)) {
            PyObject *new;
            new = PyArray_FromAnyUnwrap(obj, indtype, 0, 0,
                                        FORCECAST | ALIGNED, NULL);
            if (new == NULL) {
                goto fail;
            }
            Py_DECREF(obj);
            obj = new;
            r = iter_subscript_int(self, (PyArrayObject *)obj);
        }
        else {
            goto fail;
        }
        Py_DECREF(obj);
        return r;
    }
    else {
        _Npy_DECREF(indtype);
    }


 fail:
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_IndexError, "unsupported iterator index");
    }
    _Npy_XDECREF(indtype);
    Py_XDECREF(obj);
    return NULL;

}


static int
iter_ass_sub_Bool(NpyArrayIterObject *self, PyArrayObject *ind,
                  NpyArrayIterObject *val, int swap)
{
    intp index, strides;
    char *dptr;
    PyArray_CopySwapFunc *copyswap;

    if (PyArray_NDIM(ind) != 1) {
        PyErr_SetString(PyExc_ValueError,
                        "boolean index array should have 1 dimension");
        return -1;
    }

    index = PyArray_DIM(ind, 0);
    if (index > self->size) {
        PyErr_SetString(PyExc_ValueError,
                        "boolean index array has too many values");
        return -1;
    }

    strides = PyArray_STRIDE(ind, 0);
    dptr = PyArray_BYTES(ind);
    NpyArray_ITER_RESET(self);
    /* Loop over Boolean array */
    copyswap = self->ao->descr->f->copyswap;
    while (index--) {
        if (*((Bool *)dptr) != 0) {
            copyswap(self->dataptr, val->dataptr, swap, self->ao);
            NpyArray_ITER_NEXT(val);
            if (val->index == val->size) {
                NpyArray_ITER_RESET(val);
            }
        }
        dptr += strides;
        NpyArray_ITER_NEXT(self);
    }
    NpyArray_ITER_RESET(self);
    return 0;
}

static int
iter_ass_sub_int(NpyArrayIterObject *self, PyArrayObject *ind,
                 NpyArrayIterObject *val, int swap)
{
    NpyArray_Descr *typecode;
    intp num;
    NpyArrayIterObject *ind_it;
    intp index;
    PyArray_CopySwapFunc *copyswap;

    typecode = self->ao->descr;
    copyswap = self->ao->descr->f->copyswap;
    if (PyArray_NDIM(ind) == 0) {
        num = *((intp *)PyArray_BYTES(ind));
        NpyArray_ITER_GOTO1D(self, num);
        copyswap(self->dataptr, val->dataptr, swap, self->ao);
        return 0;
    }
    ind_it = NpyArray_IterNew(PyArray_ARRAY(ind));
    if (ind_it == NULL) {
        return -1;
    }
    index = ind_it->size;
    while (index--) {
        num = *((intp *)(ind_it->dataptr));
        if (num < 0) {
            num += self->size;
        }
        if ((num < 0) || (num >= self->size)) {
            PyErr_Format(PyExc_IndexError,
                         "index %"INTP_FMT" out of bounds"           \
                         " 0<=index<%"INTP_FMT, num,
                         self->size);
            _Npy_DECREF(ind_it);
            return -1;
        }
        NpyArray_ITER_GOTO1D(self, num);
        copyswap(self->dataptr, val->dataptr, swap, self->ao);
        NpyArray_ITER_NEXT(ind_it);
        NpyArray_ITER_NEXT(val);
        if (val->index == val->size) {
            NpyArray_ITER_RESET(val);
        }
    }
    _Npy_DECREF(ind_it);
    return 0;
}

    
NPY_NO_EXPORT int
iter_ass_subscript(PyArrayIterObject *self, PyObject *ind, PyObject *val)
{
    return npy_iter_ass_subscript(self->iter, ind, val);
}

NPY_NO_EXPORT int
npy_iter_ass_subscript(NpyArrayIterObject* self, PyObject* ind, PyObject* val)
{
    PyArrayObject *arrval = NULL;
    NpyArrayIterObject *val_it = NULL;
    NpyArray_Descr *type;
    NpyArray_Descr *indtype = NULL;
    int swap, retval = -1;
    intp start, step_size;
    intp n_steps;
    PyObject *obj = NULL;
    PyArray_CopySwapFunc *copyswap;


    if (ind == Py_Ellipsis) {
        ind = PySlice_New(NULL, NULL, NULL);
        retval = npy_iter_ass_subscript(self, ind, val);
        Py_DECREF(ind);
        return retval;
    }

    if (PyTuple_Check(ind)) {
        int len;
        len = PyTuple_GET_SIZE(ind);
        if (len > 1) {
            goto finish;
        }
        ind = PyTuple_GET_ITEM(ind, 0);
    }

    type = self->ao->descr;

    /*
     * Check for Boolean -- this is first becasue
     * Bool is a subclass of Int
     */
    if (PyBool_Check(ind)) {
        retval = 0;
        if (PyObject_IsTrue(ind)) {
            retval = type->f->setitem(val, self->dataptr, self->ao);
        }
        goto finish;
    }

    if (PySequence_Check(ind) || PySlice_Check(ind)) {
        goto skip;
    }
    start = PyArray_PyIntAsIntp(ind);
    if (start==-1 && PyErr_Occurred()) {
        PyErr_Clear();
    }
    else {
        if (start < -self->size || start >= self->size) {
            PyErr_Format(PyExc_ValueError,
                         "index (%" NPY_INTP_FMT \
                         ") out of range", start);
            goto finish;
        }
        retval = 0;
        NpyArray_ITER_GOTO1D(self, start);
        retval = type->f->setitem(val, self->dataptr, self->ao);
        NpyArray_ITER_RESET(self);
        if (retval < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "Error setting single item of array.");
        }
        goto finish;
    }

 skip:
    _Npy_INCREF(type);
    arrval = (PyArrayObject *)PyArray_FromAnyUnwrap(val, type, 0, 0, 0, NULL);
    if (arrval == NULL) {
        return -1;
    }
    val_it = NpyArray_IterNew(PyArray_ARRAY(arrval));
    if (val_it == NULL) {
        goto finish;
    }
    if (val_it->size == 0) {
        retval = 0;
        goto finish;
    }

    copyswap = PyArray_DESCR(arrval)->f->copyswap;
    swap = (NpyArray_ISNOTSWAPPED(self->ao)!=PyArray_ISNOTSWAPPED(arrval));

    /* Check Slice */
    if (PySlice_Check(ind)) {
        start = parse_subindex(ind, &step_size, &n_steps, self->size);
        if (start == -1) {
            goto finish;
        }
        if (n_steps == RubberIndex || n_steps == PseudoIndex) {
            PyErr_SetString(PyExc_IndexError,
                            "cannot use Ellipsis or newaxes here");
            goto finish;
        }
        NpyArray_ITER_GOTO1D(self, start);
        if (n_steps == SingleIndex) {
            /* Integer */
            copyswap(self->dataptr, PyArray_DATA(arrval), swap, 
                     PyArray_ARRAY(arrval));
            NpyArray_ITER_RESET(self);
            retval = 0;
            goto finish;
        }
        while (n_steps--) {
            copyswap(self->dataptr, val_it->dataptr, swap, 
                     PyArray_ARRAY(arrval));
            start += step_size;
            NpyArray_ITER_GOTO1D(self, start);
            NpyArray_ITER_NEXT(val_it);
            if (val_it->index == val_it->size) {
                NpyArray_ITER_RESET(val_it);
            }
        }
        NpyArray_ITER_RESET(self);
        retval = 0;
        goto finish;
    }

    /* convert to INTP array if Integer array scalar or List */
    indtype = NpyArray_DescrFromType(PyArray_INTP);
    if (PyList_Check(ind)) {
        _Npy_INCREF(indtype);
        obj = PyArray_FromAnyUnwrap(ind, indtype, 0, 0, FORCECAST, NULL);
    }
    else {
        Py_INCREF(ind);
        obj = ind;
    }

    if (obj != NULL && PyArray_Check(obj)) {
        /* Check for Boolean object */
        if (PyArray_TYPE(obj)==PyArray_BOOL) {
            if (iter_ass_sub_Bool(self, (PyArrayObject *)obj,
                                  val_it, swap) < 0) {
                goto finish;
            }
            retval=0;
        }
        /* Check for integer array */
        else if (PyArray_ISINTEGER(obj)) {
            PyObject *new;
            _Npy_INCREF(indtype);
            new = PyArray_CheckFromAnyUnwrap(obj, indtype, 0, 0,
                                             FORCECAST | BEHAVED_NS, NULL);
            Py_DECREF(obj);
            obj = new;
            if (new == NULL) {
                goto finish;
            }
            if (iter_ass_sub_int(self, (PyArrayObject *)obj,
                                 val_it, swap) < 0) {
                goto finish;
            }
            retval = 0;
        }
    }

 finish:
    if (!PyErr_Occurred() && retval < 0) {
        PyErr_SetString(PyExc_IndexError, "unsupported iterator index");
    }
    _Npy_XDECREF(indtype);
    Py_XDECREF(obj);
    _Npy_XDECREF(val_it);
    Py_XDECREF(arrval);
    return retval;

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
    NpyArrayIterObject *it = pit->iter;
    NpyArray *r;
    PyArrayObject *result = NULL;
    intp size;

    /* Any argument ignored */

    /* Two options:
     *  1) underlying array is contiguous
     *  -- return 1-d wrapper around it
     * 2) underlying array is not contiguous
     * -- make new 1-d contiguous array with updateifcopy flag set
     * to copy back to the old array
     */
    size = NpyArray_SIZE(it->ao);
    _Npy_INCREF(it->ao->descr);
    if (NpyArray_ISCONTIGUOUS(it->ao)) {
        r = NpyArray_NewFromDescr(it->ao->descr,
                                  1, &size,
                                  NULL, it->ao->data,
                                  it->ao->flags,
                                  NPY_TRUE, NULL, Npy_INTERFACE(it->ao));
        if (r == NULL) {
            return NULL;
        }
    }
    else {
        r = NpyArray_NewFromDescr(it->ao->descr,
                                  1, &size,
                                  NULL, NULL,
                                  0, NPY_TRUE, NULL, Npy_INTERFACE(it->ao));
        if (r == NULL) {
            return NULL;
        }
        if (_flat_copyinto(r, it->ao,
                           PyArray_CORDER) < 0) {
            _Npy_DECREF(r);
            return NULL;
        }
        NpyArray_FLAGS(r) |= UPDATEIFCOPY;
        it->ao->flags &= ~WRITEABLE;
    }
    _Npy_INCREF(it->ao);
    NpyArray_BASE_ARRAY(r) = it->ao;
    assert(NULL == NpyArray_BASE_ARRAY(r) || NULL == NpyArray_BASE(r));
    
    ASSIGN_TO_PYARRAY(result, r);
    return (PyObject *)result;
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
    _Npy_INCREF(self->iter->ao);
    return (PyObject *)self->iter->ao;
}

static PyObject *
iter_index_get(PyArrayIterObject *self)
{
#if SIZEOF_INTP <= SIZEOF_LONG
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
                _Npy_INCREF(arrays[i]);
                Py_DECREF(tmp);
            } else {
                PyObject* arg = va_arg(va, PyObject*);
                PyObject *tmp = PyArray_FROM_O(arg);
                arrays[i] = PyArray_ARRAY(tmp);
                _Npy_INCREF(arrays[i]);
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
    _Npy_DECREF(multi);


  finish:
    for (i=0; i<ntot; i++) {
        _Npy_XDECREF(arrays[i]);
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
        Npy_DECREF(arr);
    }
    if (NpyArray_Broadcast(multi) < 0) {
        goto fail;
    }
    NpyArray_MultiIter_RESET(multi);

    /* Move the reference from the core to the interface. */
    Py_INCREF( Npy_INTERFACE(multi) );
    _Npy_DECREF( multi );
    return (PyObject *)Npy_INTERFACE(multi);

 fail:
    _Npy_DECREF(multi);
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
            PyArray_ITER_NEXT(it);
        }
        multi->index++;
        return ret;
    }
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
#if SIZEOF_INTP <= SIZEOF_LONG
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
#if SIZEOF_INTP <= SIZEOF_LONG
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
    Py_DECREF((PyObject*)p);
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
    Py_INCREF( Npy_INTERFACE(coreRet) );
    _Npy_DECREF( coreRet );
    
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
