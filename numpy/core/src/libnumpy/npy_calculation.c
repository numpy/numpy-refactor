/*
 *  npy_calculation.c - 
 *  
 */

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"



NpyArray *
NpyArray_ArgMax(NpyArray *op, int axis, NpyArray *out)
{
    NpyArray *ap = NULL, *rp = NULL;
    NpyArray_ArgFunc *arg_func;
    char *ip;
    npy_intp *rptr;
    npy_intp i, n, m;
    int elsize;
    int copyret = 0;
    NPY_BEGIN_THREADS_DEF;
    
    if ((ap=NpyArray_CheckAxis(op, &axis, 0)) == NULL) {
        return NULL;
    }
    
    /*
     * We need to permute the array so that axis is placed at the end.
     * And all other dimensions are shifted left.
     */
    if (axis != ap->nd-1) {
        NpyArray_Dims newaxes;
        npy_intp dims[NPY_MAXDIMS];
        int i;
        
        newaxes.ptr = dims;
        newaxes.len = ap->nd;
        for (i = 0; i < axis; i++) dims[i] = i;
        for (i = axis; i < ap->nd - 1; i++) dims[i] = i + 1;
        dims[ap->nd - 1] = axis;
        op = NpyArray_Transpose(ap, &newaxes);
        Py_DECREF(ap);
        if (op == NULL) {
            return NULL;
        }
    }
    else {
        op = ap;
    }
    
    /* Will get native-byte order contiguous copy. */
    /* TODO: ContiguousFromAny calls PyArray_FromAny which is currently an interface function. Ugh. */
    ap = NpyArray_ContiguousFromAny(op, op->descr->type_num, 1, 0);
    Npy_DECREF(op);
    if (ap == NULL) {
        return NULL;
    }
    arg_func = ap->descr->f->argmax;
    if (arg_func == NULL) {
        NpyErr_SetString(NpyExc_TypeError, "data type not ordered");
        goto fail;
    }
    elsize = ap->descr->elsize;
    m = ap->dimensions[ap->nd-1];
    if (m == 0) {
        NpyErr_SetString(NpyExc_ValueError,
                         "attempt to get argmax/argmin "\
                         "of an empty sequence");
        goto fail;
    }
    
    if (!out) {
        rp = NpyArray_New(Npy_TYPE(ap), ap->nd-1,
                          ap->dimensions, NPY_INTP,
                          NULL, NULL, 0, 0, ap);
        if (rp == NULL) {
            goto fail;
        }
    }
    else {
        if (NpyArray_SIZE(out) !=
            NpyArray_MultiplyList(ap->dimensions, ap->nd - 1)) {
            PyErr_SetString(NpyExc_TypeError,
                            "invalid shape for output array.");
        }
        rp = NpyArray_FromArray(out,
                          NpyArray_DescrFromType(NPY_INTP),
                          NPY_CARRAY | NPY_UPDATEIFCOPY);
        if (rp == NULL) {
            goto fail;
        }
        if (rp != out) {
            copyret = 1;
        }
    }
    
    NPY_BEGIN_THREADS_DESCR(ap->descr);
    n = NpyArray_SIZE(ap)/m;
    rptr = (npy_intp *)rp->data;
    for (ip = ap->data, i = 0; i < n; i++, ip += elsize*m) {
        arg_func(ip, m, rptr, ap);
        rptr += 1;
    }
    NPY_END_THREADS_DESCR(ap->descr);
    
    Npy_DECREF(ap);
    if (copyret) {
        NpyArray *obj;
        obj = (NpyArray *)rp->base;
        Npy_INCREF(obj);
        Npy_DECREF(rp);
        rp = obj;
    }
    return rp;
    
fail:
    Npy_DECREF(ap);
    Npy_XDECREF(rp);
    return NULL;
}




NpyArray *
NpyArray_ArgMin(NpyArray *ap, int axis, NpyArray *out)
{
    PyObject *obj, *new, *ret;
    
    /* TODO: Code still uses PyInt_FromLong and PyNumber_Subtract.  
       Either need to move this function to the interface layer or 
       find a solution for these. */
    if (NpyArray_ISFLEXIBLE(ap)) {
        NpyErr_SetString(NpyExc_TypeError,
                         "argmax is unsupported for this type");
        return NULL;
    }
    else if (NpyArray_ISUNSIGNED(ap)) {
        obj = PyInt_FromLong((long) -1);
    }
    else if (NpyArray_TYPE(ap) == NpyArray_BOOL) {
        obj = PyInt_FromLong((long) 1);
    }
    else {
        obj = PyInt_FromLong((long) 0);
    }
    new = NpyArray_EnsureAnyArray(PyNumber_Subtract(obj, ap));
    Npy_DECREF(obj);
    if (new == NULL) {
        return NULL;
    }
    ret = NpyArray_ArgMax(new, axis, out);
    Npy_DECREF(new);
    return ret;
}

