#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"
#include "npy_3kcompat.h"

NpyArray *
NpyArray_TakeFrom(NpyArray *self0, NpyArray *indices0, int axis,
                  NpyArray *ret, NPY_CLIPMODE clipmode)
{
    NpyArray_FastTakeFunc *func;
    NpyArray *self, *indices;
    npy_intp nd, i, j, n, m, max_item, tmp, chunk, nelem;
    npy_intp shape[NPY_MAXDIMS];
    char *src, *dest;
    int copyret = 0;
    int err;

    indices = NULL;
    self = NpyArray_CheckAxis(self0, &axis, NPY_CARRAY);
    if (self == NULL) {
        return NULL;
    }
    indices = NpyArray_ContiguousFromArray(indices0, NpyArray_INTP);
    if (indices == NULL) {
        goto fail;
    }
    n = m = chunk = 1;
    nd = self->nd + indices->nd - 1;
    for (i = 0; i < nd; i++) {
        if (i < axis) {
            shape[i] = self->dimensions[i];
            n *= shape[i];
        }
        else {
            if (i < axis+indices->nd) {
                shape[i] = indices->dimensions[i-axis];
                m *= shape[i];
            }
            else {
                shape[i] = self->dimensions[i-indices->nd+1];
                chunk *= shape[i];
            }
        }
    }
    Npy_INCREF(self->descr);
    if (!ret) {
        ret = NpyArray_NewFromDescr(Npy_TYPE(self),
                                    self->descr,
                                    nd, shape,
                                    NULL, NULL, 0,
                                    (NpyObject *)self);

        if (ret == NULL) {
            goto fail;
        }
    }
    else {
        NpyArray *obj;
        int flags = NPY_CARRAY | NPY_UPDATEIFCOPY;

        if ((ret->nd != nd) ||
            !NpyArray_CompareLists(ret->dimensions, shape, nd)) {
            NpyErr_SetString(NpyExc_ValueError,
                             "bad shape in output array");
            ret = NULL;
            Npy_DECREF(self->descr);
            goto fail;
        }

        if (clipmode == NPY_RAISE) {
            /*
             * we need to make sure and get a copy
             * so the input array is not changed
             * before the error is called
             */
            flags |= NPY_ENSURECOPY;
        }
        obj = NpyArray_FromArray(ret, self->descr,
                                 flags);
        if (obj != ret) {
            copyret = 1;
        }
        ret = obj;
        if (ret == NULL) {
            goto fail;
        }
    }

    max_item = self->dimensions[axis];
    nelem = chunk;
    chunk = chunk * ret->descr->elsize;
    src = self->data;
    dest = ret->data;

    func = self->descr->f->fasttake;
    if (func == NULL) {
        switch(clipmode) {
        case NPY_RAISE:
            for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                    tmp = ((npy_intp *)(indices->data))[j];
                    if (tmp < 0) {
                        tmp = tmp + max_item;
                    }
                    if ((tmp < 0) || (tmp >= max_item)) {
                        NpyErr_SetString(NpyExc_IndexError,
                                "index out of range "\
                                "for array");
                        goto fail;
                    }
                    memmove(dest, src + tmp*chunk, chunk);
                    dest += chunk;
                }
                src += chunk*max_item;
            }
            break;
        case NPY_WRAP:
            for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                    tmp = ((npy_intp *)(indices->data))[j];
                    if (tmp < 0) {
                        while (tmp < 0) {
                            tmp += max_item;
                        }
                    }
                    else if (tmp >= max_item) {
                        while (tmp >= max_item) {
                            tmp -= max_item;
                        }
                    }
                    memmove(dest, src + tmp*chunk, chunk);
                    dest += chunk;
                }
                src += chunk*max_item;
            }
            break;
        case NPY_CLIP:
            for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                    tmp = ((npy_intp *)(indices->data))[j];
                    if (tmp < 0) {
                        tmp = 0;
                    }
                    else if (tmp >= max_item) {
                        tmp = max_item - 1;
                    }
                    memmove(dest, src+tmp*chunk, chunk);
                    dest += chunk;
                }
                src += chunk*max_item;
            }
            break;
        }
    }
    else {
        err = func(dest, src, (npy_intp *)(indices->data),
                    max_item, n, m, nelem, clipmode);
        if (err) {
            goto fail;
        }
    }

    NpyArray_INCREF(ret);
    Npy_XDECREF(indices);
    Npy_XDECREF(self);
    if (copyret) {
        PyObject *obj;
        obj = ret->base;
        Npy_INCREF(obj);
        Npy_DECREF(ret);
        ret = (NpyArray *)obj;
    }
    return ret;

 fail:
    NpyArray_XDECREF_ERR(ret);
    Npy_XDECREF(indices);
    Npy_XDECREF(self);
    return NULL;
}

/*
 * Put values into an array
 */
int
NpyArray_PutTo(NpyArray *self, NpyArray* values0, NpyArray *indices0,
               NPY_CLIPMODE clipmode)
{
    NpyArray  *indices, *values;
    npy_intp i, chunk, ni, max_item, nv, tmp;
    char *src, *dest;
    int copied = 0;

    indices = NULL;
    values = NULL;
    if (!NpyArray_ISCONTIGUOUS(self)) {
        NpyArray *obj;
        int flags = NPY_CARRAY | NPY_UPDATEIFCOPY;

        if (clipmode == NPY_RAISE) {
            flags |= NPY_ENSURECOPY;
        }
        Npy_INCREF(self->descr);
        obj = NpyArray_FromArray(self, self->descr, flags);
        if (obj != self) {
            copied = 1;
        }
        self = obj;
    }
    max_item = NpyArray_SIZE(self);
    dest = self->data;
    chunk = self->descr->elsize;
    indices = NpyArray_ContiguousFromArray(indices0, NpyArray_INTP);
    if (indices == NULL) {
        goto fail;
    }
    ni = NpyArray_SIZE(indices);
    Npy_INCREF(self->descr);
    values = NpyArray_FromArray(values0, self->descr, NPY_DEFAULT | NPY_FORCECAST);
    if (values == NULL) {
        goto fail;
    }
    nv = NpyArray_SIZE(values);
    if (nv <= 0) {
        goto finish;
    }
    if (NpyDataType_REFCHK(self->descr)) {
        switch(clipmode) {
        case NPY_RAISE:
            for (i = 0; i < ni; i++) {
                src = values->data + chunk*(i % nv);
                tmp = ((npy_intp *)(indices->data))[i];
                if (tmp < 0) {
                    tmp = tmp + max_item;
                }
                if ((tmp < 0) || (tmp >= max_item)) {
                    NpyErr_SetString(NpyExc_IndexError,
                            "index out of " \
                            "range for array");
                    goto fail;
                }
                NpyArray_Item_INCREF(src, self->descr);
                NpyArray_Item_XDECREF(dest+tmp*chunk, self->descr);
                memmove(dest + tmp*chunk, src, chunk);
            }
            break;
        case NPY_WRAP:
            for (i = 0; i < ni; i++) {
                src = values->data + chunk * (i % nv);
                tmp = ((npy_intp *)(indices->data))[i];
                if (tmp < 0) {
                    while (tmp < 0) {
                        tmp += max_item;
                    }
                }
                else if (tmp >= max_item) {
                    while (tmp >= max_item) {
                        tmp -= max_item;
                    }
                }
                NpyArray_Item_INCREF(src, self->descr);
                NpyArray_Item_XDECREF(dest+tmp*chunk, self->descr);
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        case NPY_CLIP:
            for (i = 0; i < ni; i++) {
                src = values->data + chunk * (i % nv);
                tmp = ((npy_intp *)(indices->data))[i];
                if (tmp < 0) {
                    tmp = 0;
                }
                else if (tmp >= max_item) {
                    tmp = max_item - 1;
                }
                NpyArray_Item_INCREF(src, self->descr);
                NpyArray_Item_XDECREF(dest+tmp*chunk, self->descr);
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        }
    }
    else {
        switch(clipmode) {
        case NPY_RAISE:
            for (i = 0; i < ni; i++) {
                src = values->data + chunk * (i % nv);
                tmp = ((npy_intp *)(indices->data))[i];
                if (tmp < 0) {
                    tmp = tmp + max_item;
                }
                if ((tmp < 0) || (tmp >= max_item)) {
                    NpyErr_SetString(NpyExc_IndexError,
                            "index out of " \
                            "range for array");
                    goto fail;
                }
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        case NPY_WRAP:
            for (i = 0; i < ni; i++) {
                src = values->data + chunk * (i % nv);
                tmp = ((npy_intp *)(indices->data))[i];
                if (tmp < 0) {
                    while (tmp < 0) {
                        tmp += max_item;
                    }
                }
                else if (tmp >= max_item) {
                    while (tmp >= max_item) {
                        tmp -= max_item;
                    }
                }
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        case NPY_CLIP:
            for (i = 0; i < ni; i++) {
                src = values->data + chunk * (i % nv);
                tmp = ((npy_intp *)(indices->data))[i];
                if (tmp < 0) {
                    tmp = 0;
                }
                else if (tmp >= max_item) {
                    tmp = max_item - 1;
                }
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        }
    }

 finish:
    Npy_XDECREF(values);
    Npy_XDECREF(indices);
    if (copied) {
        Npy_DECREF(self);
    }
    return 0;

 fail:
    Npy_XDECREF(indices);
    Npy_XDECREF(values);
    if (copied) {
        NpyArray_XDECREF_ERR(self);
    }
    return -1;
}

/*
 * Put values into an array according to a mask.
 */
int
NpyArray_PutMask(NpyArray *self, NpyArray* values0, NpyArray* mask0)
{
    NpyArray_FastPutmaskFunc *func;
    NpyArray  *mask, *values;
    npy_intp i, chunk, ni, max_item, nv, tmp;
    char *src, *dest;
    int copied = 0;

    mask = NULL;
    values = NULL;

    if (!NpyArray_ISCONTIGUOUS(self)) {
        NpyArray *obj;
        int flags = NPY_CARRAY | NPY_UPDATEIFCOPY;

        Npy_INCREF(self->descr);
        obj = NpyArray_FromArray(self, self->descr, flags);
        if (obj != self) {
            copied = 1;
        }
        self = obj;
    }

    max_item = NpyArray_SIZE(self);
    dest = self->data;
    chunk = self->descr->elsize;
    mask = NpyArray_FromArray(mask0, NpyArray_DescrFromType(NpyArray_BOOL),
                              NPY_CARRAY | NPY_FORCECAST);
    if (mask == NULL) {
        goto fail;
    }
    ni = NpyArray_SIZE(mask);
    if (ni != max_item) {
        NpyErr_SetString(NpyExc_ValueError,
                        "putmask: mask and data must be "\
                        "the same size");
        goto fail;
    }
    Npy_INCREF(self->descr);
    values = NpyArray_FromArray(values0, self->descr, NPY_CARRAY);
    if (values == NULL) {
        goto fail;
    }
    nv = NpyArray_SIZE(values); /* zero if null array */
    if (nv <= 0) {
        Npy_XDECREF(values);
        Npy_XDECREF(mask);
        return 0;
    }
    if (NpyDataType_REFCHK(self->descr)) {
        for (i = 0; i < ni; i++) {
            tmp = ((npy_bool *)(mask->data))[i];
            if (tmp) {
                src = values->data + chunk * (i % nv);
                NpyArray_Item_INCREF(src, self->descr);
                NpyArray_Item_XDECREF(dest+i*chunk, self->descr);
                memmove(dest + i * chunk, src, chunk);
            }
        }
    }
    else {
        func = self->descr->f->fastputmask;
        if (func == NULL) {
            for (i = 0; i < ni; i++) {
                tmp = ((npy_bool *)(mask->data))[i];
                if (tmp) {
                    src = values->data + chunk*(i % nv);
                    memmove(dest + i*chunk, src, chunk);
                }
            }
        }
        else {
            func(dest, mask->data, ni, values->data, nv);
        }
    }

    Npy_XDECREF(values);
    Npy_XDECREF(mask);
    if (copied) {
        Npy_DECREF(self);
    }
    return 0;

 fail:
    Npy_XDECREF(mask);
    Npy_XDECREF(values);
    if (copied) {
        NpyArray_XDECREF_ERR(self);
    }
    return -1;
}

/*
 * Repeat the array.
 */
NpyArray *
NpyArray_Repeat(NpyArray *aop, NpyArray *op, int axis)
{
    npy_intp *counts;
    npy_intp n, n_outer, i, j, k, chunk, total;
    npy_intp tmp;
    int nd;
    NpyArray *repeats = NULL;
    NpyArray *ret = NULL;
    char *new_data, *old_data;

    repeats = NpyArray_ContiguousFromArray(op, NpyArray_INTP);
    if (repeats == NULL) {
        return NULL;
    }
    nd = repeats->nd;
    counts = (npy_intp *)repeats->data;

    aop = NpyArray_CheckAxis(aop, &axis, NPY_CARRAY);
    if (aop == NULL) {
        Npy_DECREF(repeats);
        return NULL;
    }

    if (nd == 1) {
        n = repeats->dimensions[0];
    }
    else {
        /* nd == 0 */
        n = aop->dimensions[axis];
    }
    if (aop->dimensions[axis] != n) {
        NpyErr_SetString(NpyExc_ValueError,
                        "a.shape[axis] != len(repeats)");
        goto fail;
    }

    if (nd == 0) {
        total = counts[0]*n;
    }
    else {

        total = 0;
        for (j = 0; j < n; j++) {
            if (counts[j] < 0) {
                NpyErr_SetString(NpyExc_ValueError, "count < 0");
                goto fail;
            }
            total += counts[j];
        }
    }


    /* Construct new array */
    aop->dimensions[axis] = total;
    Npy_INCREF(aop->descr);
    ret = NpyArray_NewFromDescr(Npy_TYPE(aop),
                                aop->descr,
                                aop->nd,
                                aop->dimensions,
                                NULL, NULL, 0,
                                (NpyObject *)aop);
    aop->dimensions[axis] = n;
    if (ret == NULL) {
        goto fail;
    }
    new_data = ret->data;
    old_data = aop->data;

    chunk = aop->descr->elsize;
    for(i = axis + 1; i < aop->nd; i++) {
        chunk *= aop->dimensions[i];
    }

    n_outer = 1;
    for (i = 0; i < axis; i++) {
        n_outer *= aop->dimensions[i];
    }
    for (i = 0; i < n_outer; i++) {
        for (j = 0; j < n; j++) {
            tmp = nd ? counts[j] : counts[0];
            for (k = 0; k < tmp; k++) {
                memcpy(new_data, old_data, chunk);
                new_data += chunk;
            }
            old_data += chunk;
        }
    }

    Npy_DECREF(repeats);
    NpyArray_INCREF(ret);
    Npy_XDECREF(aop);
    return ret;

 fail:
    Npy_DECREF(repeats);
    Npy_XDECREF(aop);
    Npy_XDECREF(ret);
    return NULL;
}

/*
 */
NpyArray *
NpyArray_Choose(NpyArray *ip, NpyArray** mps, int n, NpyArray *ret,
               NPY_CLIPMODE clipmode)
{
    int elsize;
    npy_intp i;
    char *ret_data;
    NpyArray *ap;
    NpyArrayMultiIterObject *multi = NULL;
    npy_intp mi;
    int copyret = 0;
    ap = NULL;

    ap = NpyArray_FromArray(ip, NpyArray_DescrFromType(NPY_INTP), 0);
    if (ap == NULL) {
        goto fail;
    }
    /* Broadcast all arrays to each other, index array at the end. */ 
    /* XXX: This needs to be changed when we convert MultiIters.
       We can't replace witha  macro due to variable args. */
    multi = (NpyArrayMultiIterObject *)
        PyArray_MultiIterFromObjects((PyObject **)mps, n, 1, ap);
    if (multi == NULL) {
        goto fail;
    }
    /* Set-up return array */
    if (!ret) {
        Npy_INCREF(mps[0]->descr);
        ret = NpyArray_NewFromDescr(Npy_TYPE(ap),
                                    mps[0]->descr,
                                    multi->nd,
                                    multi->dimensions,
                                    NULL, NULL, 0,
                                    (NpyObject *)ap);
    }
    else {
        NpyArray *obj;
        int flags = NPY_CARRAY | NPY_UPDATEIFCOPY | NPY_FORCECAST;

        if ((NpyArray_NDIM(ret) != multi->nd)
                || !NpyArray_CompareLists(
                    NpyArray_DIMS(ret), multi->dimensions, multi->nd)) {
            NpyErr_SetString(NpyExc_TypeError,
                            "invalid shape for output array.");
            ret = NULL;
            goto fail;
        }
        if (clipmode == NPY_RAISE) {
            /*
             * we need to make sure and get a copy
             * so the input array is not changed
             * before the error is called
             */
            flags |= NPY_ENSURECOPY;
        }
        Npy_INCREF(mps[0]->descr);
        obj = NpyArray_FromArray(ret, mps[0]->descr, flags);
        if (obj != ret) {
            copyret = 1;
        }
        ret = obj;
    }

    if (ret == NULL) {
        goto fail;
    }
    elsize = ret->descr->elsize;
    ret_data = ret->data;

    while (NpyArray_MultiIter_NOTDONE(multi)) {
        mi = *((npy_intp *)NpyArray_MultiIter_DATA(multi, n));
        if (mi < 0 || mi >= n) {
            switch(clipmode) {
            case NPY_RAISE:
                NpyErr_SetString(NpyExc_ValueError,
                        "invalid entry in choice "\
                        "array");
                goto fail;
            case NPY_WRAP:
                if (mi < 0) {
                    while (mi < 0) {
                        mi += n;
                    }
                }
                else {
                    while (mi >= n) {
                        mi -= n;
                    }
                }
                break;
            case NPY_CLIP:
                if (mi < 0) {
                    mi = 0;
                }
                else if (mi >= n) {
                    mi = n - 1;
                }
                break;
            }
        }
        memmove(ret_data, NpyArray_MultiIter_DATA(multi, mi), elsize);
        ret_data += elsize;
        NpyArray_MultiIter_NEXT(multi);
    }

    NpyArray_INCREF(ret);
    Npy_DECREF(multi);
    Npy_DECREF(ap);
    if (copyret) {
        PyObject *obj;
        obj = ret->base;
        Npy_INCREF(obj);
        Npy_DECREF(ret);
        ret = obj;
    }
    return ret;

 fail:
    Npy_XDECREF(multi);
    Npy_XDECREF(ap);
    NpyArray_XDECREF_ERR(ret);
    return NULL;
}

