#include <stdlib.h>
#include <memory.h>
#include "npy_config.h"
#include "npy_api.h"
#include "npy_arrayobject.h"


/* TODO: Get rid of use of PyArray_INCREF here */
extern int PyArray_INCREF(void *);


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
    indices = NpyArray_ContiguousFromArray(indices0, NPY_INTP);
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
    _Npy_INCREF(self->descr);
    if (!ret) {
        ret = NpyArray_NewFromDescr(self->descr,
                                    nd, shape,
                                    NULL, NULL, 0,
                                    NPY_FALSE, NULL,
                                    Npy_INTERFACE(self));

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
            _Npy_DECREF(self->descr);
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
        obj = NpyArray_FromArray(ret, self->descr, flags);
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
    _Npy_XDECREF(indices);
    _Npy_XDECREF(self);
    if (copyret) {
        NpyArray *obj;
        obj = ret->base_arr;
        _Npy_INCREF(obj);
        _Npy_DECREF(ret);
        ret = obj;
    }
    return ret;

 fail:
    NpyArray_XDECREF_ERR(ret);
    _Npy_XDECREF(indices);
    _Npy_XDECREF(self);
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
        _Npy_INCREF(self->descr);
        obj = NpyArray_FromArray(self, self->descr, flags);
        if (obj != self) {
            copied = 1;
        }
        self = obj;
    }
    max_item = NpyArray_SIZE(self);
    dest = self->data;
    chunk = self->descr->elsize;
    indices = NpyArray_ContiguousFromArray(indices0, NPY_INTP);
    if (indices == NULL) {
        goto fail;
    }
    ni = NpyArray_SIZE(indices);
    _Npy_INCREF(self->descr);
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
    _Npy_XDECREF(values);
    _Npy_XDECREF(indices);
    if (copied) {
        _Npy_DECREF(self);
    }
    return 0;

 fail:
    _Npy_XDECREF(indices);
    _Npy_XDECREF(values);
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

        _Npy_INCREF(self->descr);
        obj = NpyArray_FromArray(self, self->descr, flags);
        if (obj != self) {
            copied = 1;
        }
        self = obj;
    }

    max_item = NpyArray_SIZE(self);
    dest = self->data;
    chunk = self->descr->elsize;
    mask = NpyArray_FromArray(mask0, NpyArray_DescrFromType(NPY_BOOL),
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
    _Npy_INCREF(self->descr);
    values = NpyArray_FromArray(values0, self->descr, NPY_CARRAY);
    if (values == NULL) {
        goto fail;
    }
    nv = NpyArray_SIZE(values); /* zero if null array */
    if (nv <= 0) {
        _Npy_XDECREF(values);
        _Npy_XDECREF(mask);
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

    _Npy_XDECREF(values);
    _Npy_XDECREF(mask);
    if (copied) {
        _Npy_DECREF(self);
    }
    return 0;

 fail:
    _Npy_XDECREF(mask);
    _Npy_XDECREF(values);
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

    repeats = NpyArray_ContiguousFromArray(op, NPY_INTP);
    if (repeats == NULL) {
        return NULL;
    }
    nd = repeats->nd;
    counts = (npy_intp *)repeats->data;

    aop = NpyArray_CheckAxis(aop, &axis, NPY_CARRAY);
    if (aop == NULL) {
        _Npy_DECREF(repeats);
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
    _Npy_INCREF(aop->descr);
    ret = NpyArray_NewFromDescr(aop->descr,
                                aop->nd,
                                aop->dimensions,
                                NULL, NULL, 0,
                                NPY_FALSE, NULL,
                                Npy_INTERFACE(aop));
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

    _Npy_DECREF(repeats);
    NpyArray_INCREF(ret);
    _Npy_XDECREF(aop);
    return ret;

 fail:
    _Npy_DECREF(repeats);
    _Npy_XDECREF(aop);
    _Npy_XDECREF(ret);
    return NULL;
}

/*
 */
NpyArray *
NpyArray_Choose(NpyArray *ip, NpyArray** mps, int n, NpyArray *ret,
               NPY_CLIPMODE clipmode)
{
    int elsize;
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
    multi = NpyArray_MultiIterFromArrays(mps, n, 1, ap);
    if (multi == NULL) {
        goto fail;
    }
    /* Set-up return array */
    if (!ret) {
        _Npy_INCREF(mps[0]->descr);
        ret = NpyArray_NewFromDescr(mps[0]->descr,
                                    multi->nd,
                                    multi->dimensions,
                                    NULL, NULL, 0,
                                    NPY_FALSE, NULL,
                                    Npy_INTERFACE(ap));
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
        _Npy_INCREF(mps[0]->descr);
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
    _Npy_DECREF(multi);
    _Npy_DECREF(ap);
    if (copyret) {
        NpyArray *obj;
        obj = ret->base_arr;
        _Npy_INCREF(obj);
        _Npy_DECREF(ret);
        ret = obj;
    }
    return ret;

 fail:
    _Npy_XDECREF(multi);
    _Npy_XDECREF(ap);
    NpyArray_XDECREF_ERR(ret);
    return NULL;
}


/*
 * These algorithms use special sorting.  They are not called unless the
 * underlying sort function for the type is available.  Note that axis is
 * already valid. The sort functions require 1-d contiguous and well-behaved
 * data.  Therefore, a copy will be made of the data if needed before handing
 * it to the sorting routine.  An iterator is constructed and adjusted to walk
 * over all but the desired sorting axis.
 */
static int
_new_sort(NpyArray *op, int axis, NPY_SORTKIND which)
{
    NpyArrayIterObject *it;
    int needcopy = 0, swap;
    npy_intp N, size;
    int elsize;
    npy_intp astride;
    NpyArray_SortFunc *sort;
    NPY_BEGIN_THREADS_DEF;

    it = NpyArray_IterAllButAxis(op, &axis);
    swap = !NpyArray_ISNOTSWAPPED(op);
    if (it == NULL) {
        return -1;
    }

    NPY_BEGIN_THREADS_DESCR(op->descr);
    sort = op->descr->f->sort[which];
    size = it->size;
    N = op->dimensions[axis];
    elsize = op->descr->elsize;
    astride = op->strides[axis];

    needcopy = !(op->flags & NPY_ALIGNED) || (astride != (npy_intp) elsize) ||
                                                                         swap;
    if (needcopy) {
        char *buffer = NpyDataMem_NEW(N * elsize);

        while (size--) {
            _unaligned_strided_byte_copy(buffer, (npy_intp) elsize, it->dataptr,
                                         astride, N, elsize);
            if (swap) {
                _strided_byte_swap(buffer, (npy_intp) elsize, N, elsize);
            }
            if (sort(buffer, N, op) < 0) {
                NpyDataMem_FREE(buffer);
                goto fail;
            }
            if (swap) {
                _strided_byte_swap(buffer, (npy_intp) elsize, N, elsize);
            }
            _unaligned_strided_byte_copy(it->dataptr, astride, buffer,
                                         (npy_intp) elsize, N, elsize);
            NpyArray_ITER_NEXT(it);
        }
        NpyDataMem_FREE(buffer);
    }
    else {
        while (size--) {
            if (sort(it->dataptr, N, op) < 0) {
                goto fail;
            }
            NpyArray_ITER_NEXT(it);
        }
    }
    NPY_END_THREADS_DESCR(op->descr);
    _Npy_DECREF(it);
    return 0;

 fail:
    NPY_END_THREADS;
    _Npy_DECREF(it);
    return 0;
}

static NpyArray*
_new_argsort(NpyArray *op, int axis, NPY_SORTKIND which)
{

    NpyArrayIterObject *it = NULL;
    NpyArrayIterObject *rit = NULL;
    NpyArray *ret;
    int needcopy = 0, i;
    npy_intp N, size;
    int elsize, swap;
    npy_intp astride, rstride, *iptr;
    NpyArray_ArgSortFunc *argsort;
    NPY_BEGIN_THREADS_DEF;

    ret = NpyArray_New(NULL, op->nd,
                       op->dimensions, NPY_INTP,
                       NULL, NULL, 0, 0, Npy_INTERFACE(op));
    if (ret == NULL) {
        return NULL;
    }
    it = NpyArray_IterAllButAxis(op, &axis);
    rit = NpyArray_IterAllButAxis(ret, &axis);
    if (rit == NULL || it == NULL) {
        goto fail;
    }
    swap = !NpyArray_ISNOTSWAPPED(op);

    NPY_BEGIN_THREADS_DESCR(op->descr);
    argsort = op->descr->f->argsort[which];
    size = it->size;
    N = op->dimensions[axis];
    elsize = op->descr->elsize;
    astride = op->strides[axis];
    rstride = NpyArray_STRIDE(ret,axis);

    needcopy = swap || !(op->flags & NPY_ALIGNED) ||
        (astride != (npy_intp) elsize) || (rstride != sizeof(npy_intp));
    if (needcopy) {
        char *valbuffer, *indbuffer;

        valbuffer = NpyDataMem_NEW(N*elsize);
        indbuffer = NpyDataMem_NEW(N*sizeof(npy_intp));
        while (size--) {
            _unaligned_strided_byte_copy(valbuffer, (npy_intp) elsize,
                                         it->dataptr, astride, N, elsize);
            if (swap) {
                _strided_byte_swap(valbuffer, (npy_intp) elsize, N, elsize);
            }
            iptr = (npy_intp *)indbuffer;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            if (argsort(valbuffer, (npy_intp *)indbuffer, N, op) < 0) {
                NpyDataMem_FREE(valbuffer);
                NpyDataMem_FREE(indbuffer);
                goto fail;
            }
            _unaligned_strided_byte_copy(rit->dataptr, rstride, indbuffer,
                                         sizeof(npy_intp), N, sizeof(npy_intp));
            NpyArray_ITER_NEXT(it);
            NpyArray_ITER_NEXT(rit);
        }
        NpyDataMem_FREE(valbuffer);
        NpyDataMem_FREE(indbuffer);
    }
    else {
        while (size--) {
            iptr = (npy_intp *)rit->dataptr;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            if (argsort(it->dataptr, (npy_intp *)rit->dataptr, N, op) < 0) {
                goto fail;
            }
            NpyArray_ITER_NEXT(it);
            NpyArray_ITER_NEXT(rit);
        }
    }

    NPY_END_THREADS_DESCR(op->descr);

    _Npy_DECREF(it);
    _Npy_DECREF(rit);
    return ret;

 fail:
    NPY_END_THREADS;
    _Npy_DECREF(ret);
    _Npy_XDECREF(it);
    _Npy_XDECREF(rit);
    return NULL;
}


/* Be sure to save this global_compare when necessary */
/* XXX: This may be an issue in an MT case. */
static NpyArray* global_obj;

static int
qsortCompare (const void *a, const void *b)
{
    return global_obj->descr->f->compare(a,b,global_obj);
}

/*
 * Consumes reference to ap (op gets it) op contains a version of
 * the array with axes swapped if local variable axis is not the
 * last dimension.  Origin must be defined locally.
 */
#define SWAPAXES(op, ap) {                                      \
        orign = (ap)->nd-1;                                     \
        if (axis != orign) {                                    \
            (op) = NpyArray_SwapAxes((ap), axis, orign);        \
            _Npy_DECREF((ap));                                   \
            if ((op) == NULL) return NULL;                      \
        }                                                       \
        else (op) = (ap);                                       \
    }

/*
 * Consumes reference to ap (op gets it) origin must be previously
 * defined locally.  SWAPAXES must have been called previously.
 * op contains the swapped version of the array.
 */
#define SWAPBACK(op, ap) {                                      \
        if (axis != orign) {                                    \
            (op) = NpyArray_SwapAxes((ap), axis, orign);        \
            _Npy_DECREF((ap));                                   \
            if ((op) == NULL) return NULL;                      \
        }                                                       \
        else (op) = (ap);                                       \
    }

/* These swap axes in-place if necessary */
#define SWAPINTP(a,b) {npy_intp c; c=(a); (a) = (b); (b) = c;}
#define SWAPAXES2(ap) {                                                 \
        orign = (ap)->nd-1;                                             \
        if (axis != orign) {                                            \
            SWAPINTP(ap->dimensions[axis], ap->dimensions[orign]);      \
            SWAPINTP(ap->strides[axis], ap->strides[orign]);            \
            NpyArray_UpdateFlags(ap, NPY_CONTIGUOUS | NPY_FORTRAN);     \
        }                                                               \
    }

#define SWAPBACK2(ap) {                                                 \
        if (axis != orign) {                                            \
            SWAPINTP(ap->dimensions[axis], ap->dimensions[orign]);      \
            SWAPINTP(ap->strides[axis], ap->strides[orign]);            \
            NpyArray_UpdateFlags(ap, NPY_CONTIGUOUS | NPY_FORTRAN);     \
        }                                                               \
    }

/*
 * Sort an array in-place
 */
int
NpyArray_Sort(NpyArray *op, int axis, NPY_SORTKIND which)
{
    NpyArray *ap = NULL, *store_arr = NULL;
    char *ip;
    int i, n, m, elsize, orign;
    char msg[1024];

    n = op->nd;
    if ((n == 0) || (NpyArray_SIZE(op) == 1)) {
        return 0;
    }
    if (axis < 0) {
        axis += n;
    }
    if ((axis < 0) || (axis >= n)) {
        sprintf(msg, "axis(=%d) out of bounds", axis);
        NpyErr_SetString(NpyExc_ValueError, msg);
        return -1;
    }
    if (!NpyArray_ISWRITEABLE(op)) {
        NpyErr_SetString(NpyExc_RuntimeError,
                        "attempted sort on unwriteable array.");
        return -1;
    }

    /* Determine if we should use type-specific algorithm or not */
    if (op->descr->f->sort[which] != NULL) {
        return _new_sort(op, axis, which);
    }
    if ((which != NPY_QUICKSORT)
        || op->descr->f->compare == NULL) {
        NpyErr_SetString(NpyExc_TypeError,
                        "desired sort not supported for this type");
        return -1;
    }

    SWAPAXES2(op);

    ap = NpyArray_FromArray(op, NULL, NPY_DEFAULT | NPY_UPDATEIFCOPY);
    if (ap == NULL) {
        goto fail;
    }
    elsize = ap->descr->elsize;
    m = ap->dimensions[ap->nd-1];
    if (m == 0) {
        goto finish;
    }
    n = NpyArray_SIZE(ap)/m;

    /* Store global -- allows re-entry -- restore before leaving*/
    store_arr = global_obj;
    global_obj = ap;
    for (ip = ap->data, i = 0; i < n; i++, ip += elsize*m) {
        qsort(ip, m, elsize, qsortCompare);
    }
    global_obj = store_arr;

    if (NpyErr_Occurred()) {
        goto fail;
    }

 finish:
    _Npy_DECREF(ap);  /* Should update op if needed */
    SWAPBACK2(op);
    return 0;

 fail:
    _Npy_XDECREF(ap);
    SWAPBACK2(op);
    return -1;
}


/* XXX: This could be a problem for MT. */
static char *global_data;

static int
argsort_static_compare(const void *ip1, const void *ip2)
{
    int isize = global_obj->descr->elsize;
    const npy_intp *ipa = ip1;
    const npy_intp *ipb = ip2;
    return global_obj->descr->f->compare(global_data + (isize * *ipa),
                                         global_data + (isize * *ipb),
                                         global_obj);
}

/*
 * ArgSort an array
 */
NpyArray *
NpyArray_ArgSort(NpyArray *op, int axis, NPY_SORTKIND which)
{
    NpyArray *ap = NULL, *ret = NULL, *store, *op2;
    npy_intp *ip;
    npy_intp i, j, n, m, orign;
    int argsort_elsize;
    char *store_ptr;

    n = op->nd;
    if ((n == 0) || (NpyArray_SIZE(op) == 1)) {
        ret = NpyArray_New(NULL, op->nd,
                           op->dimensions,
                           NPY_INTP,
                           NULL, NULL, 0, 0,
                           Npy_INTERFACE(op));
        if (ret == NULL) {
            return NULL;
        }
        *((npy_intp *)ret->data) = 0;
        return ret;
    }

    /* Creates new reference op2 */
    if ((op2=NpyArray_CheckAxis(op, &axis, 0)) == NULL) {
        return NULL;
    }
    /* Determine if we should use new algorithm or not */
    if (op2->descr->f->argsort[which] != NULL) {
        ret = _new_argsort(op2, axis, which);
        _Npy_DECREF(op2);
        return ret;
    }

    if ((which != NPY_QUICKSORT) || op2->descr->f->compare == NULL) {
        NpyErr_SetString(NpyExc_TypeError,
                        "requested sort not available for type");
        _Npy_DECREF(op2);
        op = NULL;
        goto fail;
    }

    /* ap will contain the reference to op2 */
    SWAPAXES(ap, op2);
    op = NpyArray_ContiguousFromArray(ap, NPY_NOTYPE);
    _Npy_DECREF(ap);
    if (op == NULL) {
        return NULL;
    }
    ret = NpyArray_New(NULL, op->nd,
                       op->dimensions, NPY_INTP,
                       NULL, NULL, 0, 0, Npy_INTERFACE(op));
    if (ret == NULL) {
        goto fail;
    }
    ip = (npy_intp *)ret->data;
    argsort_elsize = op->descr->elsize;
    m = op->dimensions[op->nd-1];
    if (m == 0) {
        goto finish;
    }
    n = NpyArray_SIZE(op)/m;
    store_ptr = global_data;
    global_data = op->data;
    store = global_obj;
    global_obj = op;
    for (i = 0; i < n; i++, ip += m, global_data += m*argsort_elsize) {
        for (j = 0; j < m; j++) {
            ip[j] = j;
        }
        qsort((char *)ip, m, sizeof(npy_intp), argsort_static_compare);
    }
    global_data = store_ptr;
    global_obj = store;

 finish:
    _Npy_DECREF(op);
    SWAPBACK(op, ret);
    return op;

 fail:
    _Npy_XDECREF(op);
    _Npy_XDECREF(ret);
    return NULL;

}

/*
 * LexSort an array providing indices that will sort a collection of arrays
 * lexicographically.  The first key is sorted on first, followed by the
 * second key -- requires that arg"merge"sort is available for each sort_key
 *
 * Returns an index array that shows the indexes for the lexicographic sort along
 * the given axis.
 */
NpyArray *
NpyArray_LexSort(NpyArray** mps, int n, int axis)
{
    NpyArrayIterObject **its;
    NpyArray *ret = NULL;
    NpyArrayIterObject *rit = NULL;
    int nd;
    int needcopy = 0, i,j;
    npy_intp N, size;
    int elsize;
    int maxelsize;
    npy_intp astride, rstride, *iptr;
    int object = 0;
    NpyArray_ArgSortFunc *argsort;
    NPY_BEGIN_THREADS_DEF;
    char msg[1024];

    its = (NpyArrayIterObject **) NpyDataMem_NEW(n*sizeof(NpyArrayIterObject*));
    if (its == NULL) {
        NpyErr_SetString(NpyExc_MemoryError, "no memory");
        return NULL;
    }
    for (i = 0; i < n; i++) {
        its[i] = NULL;
    }
    for (i = 0; i < n; i++) {
        if (i > 0) {
            if ((mps[i]->nd != mps[0]->nd)
                || (!NpyArray_CompareLists(mps[i]->dimensions,
                                       mps[0]->dimensions,
                                       mps[0]->nd))) {
                NpyErr_SetString(NpyExc_ValueError,
                                "all keys need to be the same shape");
                goto fail;
            }
        }
        if (!mps[i]->descr->f->argsort[NPY_MERGESORT]) {
            sprintf(msg, "merge sort not available for item %d", i);
            NpyErr_SetString(NpyExc_TypeError, msg);
            goto fail;
        }
        /* XXX: What do we do about this NPY_NEEDS_PYAPI? */
        if (!object
            && NpyDataType_FLAGCHK(mps[i]->descr, NPY_NEEDS_PYAPI)) {
            object = 1;
        }
        its[i] = NpyArray_IterAllButAxis(mps[i], &axis);
        if (its[i] == NULL) {
            goto fail;
        }
    }

    /* Now we can check the axis */
    nd = mps[0]->nd;
    if ((nd == 0) || (NpyArray_SIZE(mps[0]) == 1)) {
        /* single element case */
        ret = NpyArray_New(NULL, mps[0]->nd,
                           mps[0]->dimensions,
                           NPY_INTP,
                           NULL, NULL, 0, 0, NULL);

        if (ret == NULL) {
            goto fail;
        }
        *((npy_intp *)(ret->data)) = 0;
        goto finish;
    }
    if (axis < 0) {
        axis += nd;
    }
    if ((axis < 0) || (axis >= nd)) {
        sprintf(msg, "axis(=%d) out of bounds", axis);
        NpyErr_SetString(NpyExc_ValueError, msg);
        goto fail;
    }

    /* Now do the sorting */
    ret = NpyArray_New(NULL, mps[0]->nd,
                       mps[0]->dimensions, NPY_INTP,
                       NULL, NULL, 0, 0, NULL);
    if (ret == NULL) {
        goto fail;
    }
    rit = NpyArray_IterAllButAxis(ret, &axis);
    if (rit == NULL) {
        goto fail;
    }
    if (!object) {
        NPY_BEGIN_THREADS;
    }
    size = rit->size;
    N = mps[0]->dimensions[axis];
    rstride = NpyArray_STRIDE(ret, axis);
    maxelsize = mps[0]->descr->elsize;
    needcopy = (rstride != sizeof(npy_intp));
    for (j = 0; j < n; j++) {
        needcopy = needcopy
            || NpyArray_ISBYTESWAPPED(mps[j])
            || !(mps[j]->flags & NPY_ALIGNED)
            || (mps[j]->strides[axis] != (npy_intp)mps[j]->descr->elsize);
        if (mps[j]->descr->elsize > maxelsize) {
            maxelsize = mps[j]->descr->elsize;
        }
    }

    if (needcopy) {
        char *valbuffer, *indbuffer;
        int *swaps;

        valbuffer = NpyDataMem_NEW(N*maxelsize);
        indbuffer = NpyDataMem_NEW(N*sizeof(npy_intp));
        swaps = malloc(n*sizeof(int));
        for (j = 0; j < n; j++) {
            swaps[j] = NpyArray_ISBYTESWAPPED(mps[j]);
        }
        while (size--) {
            iptr = (npy_intp *)indbuffer;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            for (j = 0; j < n; j++) {
                elsize = mps[j]->descr->elsize;
                astride = mps[j]->strides[axis];
                argsort = mps[j]->descr->f->argsort[NPY_MERGESORT];
                _unaligned_strided_byte_copy(valbuffer, (npy_intp) elsize,
                                             its[j]->dataptr, astride,
                                             N, elsize);
                if (swaps[j]) {
                    _strided_byte_swap(valbuffer, (npy_intp) elsize, N, elsize);
                }
                if (argsort(valbuffer, (npy_intp *)indbuffer, N, mps[j]) < 0) {
                    NpyDataMem_FREE(valbuffer);
                    NpyDataMem_FREE(indbuffer);
                    free(swaps);
                    goto fail;
                }
                NpyArray_ITER_NEXT(its[j]);
            }
            _unaligned_strided_byte_copy(rit->dataptr, rstride, indbuffer,
                                         sizeof(npy_intp), N, sizeof(npy_intp));
            NpyArray_ITER_NEXT(rit);
        }
        NpyDataMem_FREE(valbuffer);
        NpyDataMem_FREE(indbuffer);
        free(swaps);
    }
    else {
        while (size--) {
            iptr = (npy_intp *)rit->dataptr;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            for (j = 0; j < n; j++) {
                argsort = mps[j]->descr->f->argsort[NPY_MERGESORT];
                if (argsort(its[j]->dataptr, (npy_intp *)rit->dataptr,
                            N, mps[j]) < 0) {
                    goto fail;
                }
                NpyArray_ITER_NEXT(its[j]);
            }
            NpyArray_ITER_NEXT(rit);
        }
    }

    if (!object) {
        NPY_END_THREADS;
    }

 finish:
    for (i = 0; i < n; i++) {
        _Npy_XDECREF(its[i]);
    }
    _Npy_XDECREF(rit);
    NpyDataMem_FREE(its);
    return ret;

 fail:
    NPY_END_THREADS;
    _Npy_XDECREF(rit);
    _Npy_XDECREF(ret);
    for (i = 0; i < n; i++) {
        _Npy_XDECREF(its[i]);
    }
    NpyDataMem_FREE(its);
    return NULL;
}


/** @brief Use bisection of sorted array to find first entries >= keys.
 *
 * For each key use bisection to find the first index i s.t. key <= arr[i].
 * When there is no such index i, set i = len(arr). Return the results in ret.
 * All arrays are assumed contiguous on entry and both arr and key must be of
 * the same comparable type.
 *
 * @param arr contiguous sorted array to be searched.
 * @param key contiguous array of keys.
 * @param ret contiguous array of intp for returned indices.
 * @return void
 */
static void
local_search_left(NpyArray *arr, NpyArray *key, NpyArray *ret)
{
    NpyArray_CompareFunc *compare = key->descr->f->compare;
    npy_intp nelts = arr->dimensions[arr->nd - 1];
    npy_intp nkeys = NpyArray_SIZE(key);
    char *parr = arr->data;
    char *pkey = key->data;
    npy_intp *pret = (npy_intp *)ret->data;
    int elsize = arr->descr->elsize;
    npy_intp i;

    for (i = 0; i < nkeys; ++i) {
        npy_intp imin = 0;
        npy_intp imax = nelts;
        while (imin < imax) {
            npy_intp imid = imin + ((imax - imin) >> 1);
            if (compare(parr + elsize*imid, pkey, key) < 0) {
                imin = imid + 1;
            }
            else {
                imax = imid;
            }
        }
        *pret = imin;
        pret += 1;
        pkey += elsize;
    }
}


/** @brief Use bisection of sorted array to find first entries > keys.
 *
 * For each key use bisection to find the first index i s.t. key < arr[i].
 * When there is no such index i, set i = len(arr). Return the results in ret.
 * All arrays are assumed contiguous on entry and both arr and key must be of
 * the same comparable type.
 *
 * @param arr contiguous sorted array to be searched.
 * @param key contiguous array of keys.
 * @param ret contiguous array of intp for returned indices.
 * @return void
 */
static void
local_search_right(NpyArray *arr, NpyArray *key, NpyArray *ret)
{
    NpyArray_CompareFunc *compare = key->descr->f->compare;
    npy_intp nelts = arr->dimensions[arr->nd - 1];
    npy_intp nkeys = NpyArray_SIZE(key);
    char *parr = arr->data;
    char *pkey = key->data;
    npy_intp *pret = (npy_intp *)ret->data;
    int elsize = arr->descr->elsize;
    npy_intp i;

    for(i = 0; i < nkeys; ++i) {
        npy_intp imin = 0;
        npy_intp imax = nelts;
        while (imin < imax) {
            npy_intp imid = imin + ((imax - imin) >> 1);
            if (compare(parr + elsize*imid, pkey, key) <= 0) {
                imin = imid + 1;
            }
            else {
                imax = imid;
            }
        }
        *pret = imin;
        pret += 1;
        pkey += elsize;
    }
}


/*
 * Numeric.searchsorted(a,v)
 */
NpyArray *
NpyArray_SearchSorted(NpyArray *op1, NpyArray *op2, NPY_SEARCHSIDE side)
{
    NpyArray *ap1 = NULL;
    NpyArray *ap2 = NULL;
    NpyArray *ret = NULL;
    NpyArray_Descr *dtype;
    NPY_BEGIN_THREADS_DEF;

    dtype = NpyArray_DescrFromArray(op2, op1->descr);
    /* need ap1 as contiguous array and of right type */
    _Npy_INCREF(dtype);
    ap1 = NpyArray_FromArray(op1, dtype, NPY_DEFAULT);
    if (ap1 == NULL) {
        _Npy_DECREF(dtype);
        return NULL;
    }

    /* need ap2 as contiguous array and of right type */
    ap2 = NpyArray_FromArray(op2, dtype, NPY_DEFAULT);
    if (ap2 == NULL) {
        goto fail;
    }
    /* ret is a contiguous array of intp type to hold returned indices */
    ret = NpyArray_New(NULL, ap2->nd,
                       ap2->dimensions, NPY_INTP,
                       NULL, NULL, 0, 0, Npy_INTERFACE(ap2));
    if (ret == NULL) {
        goto fail;
    }
    /* check that comparison function exists */
    if (ap2->descr->f->compare == NULL) {
        NpyErr_SetString(NpyExc_TypeError,
                         "compare not supported for type");
        goto fail;
    }

    if (side == NPY_SEARCHLEFT) {
        NPY_BEGIN_THREADS_DESCR(ap2->descr);
        local_search_left(ap1, ap2, ret);
        NPY_END_THREADS_DESCR(ap2->descr);
    }
    else if (side == NPY_SEARCHRIGHT) {
        NPY_BEGIN_THREADS_DESCR(ap2->descr);
        local_search_right(ap1, ap2, ret);
        NPY_END_THREADS_DESCR(ap2->descr);
    }
    _Npy_DECREF(ap1);
    _Npy_DECREF(ap2);
    return ret;

 fail:
    _Npy_XDECREF(ap1);
    _Npy_XDECREF(ap2);
    _Npy_XDECREF(ret);
    return NULL;
}


/*
 * Fills on index_arrays with 1-d arrays giving the indexes
 * along each dimension of the non-zero elements in self.
 * index_arrays must be an array of size self->nd.  Each
 * index array will be the length of the number of non-zero
 * elements in self.
 */
int
NpyArray_NonZero(NpyArray* self, NpyArray** index_arrays, void* obj)
{
    int n = self->nd, j;
    npy_intp count = 0, i, size;
    NpyArrayIterObject *it = NULL;
    NpyArray *item;
    npy_intp *dptr[NPY_MAXDIMS];
    NpyArray_NonzeroFunc *nonzero = self->descr->f->nonzero;

    for (i=0; i<n; i++) {
        index_arrays[i] = NULL;
    }

    it = NpyArray_IterNew(self);
    if (it == NULL) {
        return -1;
    }
    size = it->size;
    for (i = 0; i < size; i++) {
        if (nonzero(it->dataptr, self)) {
            count++;
        }
        NpyArray_ITER_NEXT(it);
    }

    NpyArray_ITER_RESET(it);
    for (j = 0; j < n; j++) {
        item = NpyArray_New(NULL, 1, &count,
                            NPY_INTP, NULL, NULL, 0, 0,
                            obj);
        if (item == NULL) {
            goto fail;
        }
        index_arrays[j] = item;
        dptr[j] = (npy_intp *)NpyArray_DATA(item);
    }
    if (n == 1) {
        for (i = 0; i < size; i++) {
            if (nonzero(it->dataptr, self)) {
                *(dptr[0])++ = i;
            }
            NpyArray_ITER_NEXT(it);
        }
    }
    else {
        /* reset contiguous so that coordinates gets updated */
        it->contiguous = 0;
        for (i = 0; i < size; i++) {
            if (nonzero(it->dataptr, self)) {
                for (j = 0; j < n; j++) {
                    *(dptr[j])++ = it->coordinates[j];
                }
            }
            NpyArray_ITER_NEXT(it);
        }
    }

    _Npy_DECREF(it);
    return 0;

 fail:
    for (i=0; i<n; i++) {
        _Npy_XDECREF(index_arrays[i]);
    }
    _Npy_XDECREF(it);
    return -1;
}
