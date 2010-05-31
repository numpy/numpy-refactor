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
    indices = NpyArray_ContiguousFromArray(indices0,
                                           NpyArray_INTP,
                                           1, 0);
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
        ret = NpyArray_NewFromDescr(Py_TYPE(self),
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
