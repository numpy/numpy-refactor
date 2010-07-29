/*
 *  npy_convert.c - 
 *  
 */

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include "npy_config.h"
#include "numpy/numpy_api.h"




NpyArray *
NpyArray_View(NpyArray *self, NpyArray_Descr *type, void *subtype)
{
    NpyArray *new = NULL;

    _Npy_INCREF(NpyArray_DESCR(self));
    new = NpyArray_NewFromDescr(NpyArray_DESCR(self),
                                NpyArray_NDIM(self), NpyArray_DIMS(self),
                                NpyArray_STRIDES(self),
                                NpyArray_BYTES(self),
                                NpyArray_FLAGS(self), 
                                NPY_FALSE,
                                subtype, Npy_INTERFACE(self));
    if (new == NULL) {
        return NULL;
    }
    
    new->base_arr = self;
    _Npy_INCREF(self);
    assert(NULL == new->base_obj);

    if (type != NULL) {
        /* TODO: unwrap type. */
        if (NpyArray_SetDescr(new, type) < 0) {
            _Npy_DECREF(new);
            _Npy_DECREF(type);
            return NULL;
        }
        _Npy_DECREF(type);
    }
    return new;
}


int
NpyArray_SetDescr(NpyArray *self, NpyArray_Descr *newtype)
{
    npy_intp newdim;
    int index;
    char *msg = "new type not compatible with array.";

    _Npy_INCREF(newtype);

    if (NpyDataType_FLAGCHK(newtype, NPY_ITEM_HASOBJECT) ||
        NpyDataType_FLAGCHK(newtype, NPY_ITEM_IS_POINTER) ||
        NpyDataType_FLAGCHK(NpyArray_DESCR(self), NPY_ITEM_HASOBJECT) ||
        NpyDataType_FLAGCHK(NpyArray_DESCR(self), NPY_ITEM_IS_POINTER)) {
        NpyErr_SetString(NpyExc_TypeError,                      \
                        "Cannot change data-type for object " \
                        "array.");
        _Npy_DECREF(newtype);
        return -1;
    }

    if (newtype->elsize == 0) {
        NpyErr_SetString(NpyExc_TypeError,
                        "data-type must not be 0-sized");
        _Npy_DECREF(newtype);
        return -1;
    }


    if ((newtype->elsize != NpyArray_ITEMSIZE(self)) &&
        (NpyArray_NDIM(self) == 0 || !NpyArray_ISONESEGMENT(self) ||
         newtype->subarray)) {
        goto fail;
    }
    if (NpyArray_ISCONTIGUOUS(self)) {
        index = NpyArray_NDIM(self) - 1;
    }
    else {
        index = 0;
    }
    if (newtype->elsize < NpyArray_ITEMSIZE(self)) {
        /*
         * if it is compatible increase the size of the
         * dimension at end (or at the front for FORTRAN)
         */
        if (NpyArray_ITEMSIZE(self) % newtype->elsize != 0) {
            goto fail;
        }
        newdim = NpyArray_ITEMSIZE(self) / newtype->elsize;
        NpyArray_DIM(self, index) *= newdim;
        NpyArray_STRIDE(self, index) = newtype->elsize;
    }
    else if (newtype->elsize > NpyArray_ITEMSIZE(self)) {
        /*
         * Determine if last (or first if FORTRAN) dimension
         * is compatible
         */
        newdim = NpyArray_DIM(self, index) * NpyArray_ITEMSIZE(self);
        if ((newdim % newtype->elsize) != 0) {
            goto fail;
        }
        NpyArray_DIM(self, index) = newdim / newtype->elsize;
        NpyArray_STRIDE(self, index) = newtype->elsize;
    }

    /* fall through -- adjust type*/
    _Npy_DECREF(NpyArray_DESCR(self));
    if (newtype->subarray) {
        /*
         * create new array object from data and update
         * dimensions, strides and descr from it
         */
        NpyArray *temp;
        /*
         * We would decref newtype here.
         * temp will steal a reference to it
         */
        temp = 
            NpyArray_NewFromDescr(newtype, NpyArray_NDIM(self),
                                  NpyArray_DIMS(self), NpyArray_STRIDES(self),
                                  NpyArray_BYTES(self), NpyArray_FLAGS(self), 
                                  NPY_TRUE, NULL, 
                                  NULL);
        if (temp == NULL) {
            return -1;
        }
        NpyDimMem_FREE(NpyArray_DIMS(self));
        NpyArray_DIMS(self) = NpyArray_DIMS(temp);
        NpyArray_NDIM(self) = NpyArray_NDIM(temp);
        NpyArray_STRIDES(self) = NpyArray_STRIDES(temp);
        newtype = NpyArray_DESCR(temp);
        _Npy_INCREF(newtype);
        /* Fool deallocator not to delete these*/
        NpyArray_NDIM(temp) = 0;
        NpyArray_DIMS(temp) = NULL;
        _Npy_DECREF(temp);
    }

    NpyArray_DESCR(self) = newtype;
    NpyArray_UpdateFlags(self, NPY_UPDATE_ALL);
    return 0;

 fail:
    NpyErr_SetString(NpyExc_ValueError, msg);
    _Npy_DECREF(newtype);
    return -1;
}


/*
  Copy an array.
*/
NpyArray *
NpyArray_NewCopy(NpyArray *m1, NPY_ORDER fortran)
{
    NpyArray *ret;
    if (fortran == NPY_ANYORDER)
        fortran = NpyArray_ISFORTRAN(m1);

    _Npy_INCREF(NpyArray_DESCR(m1));
    ret = NpyArray_NewFromDescr(NpyArray_DESCR(m1),
                                NpyArray_NDIM(m1),
                                NpyArray_DIMS(m1),
                                NULL, NULL,
                                fortran,
                                NPY_FALSE, NULL,
                                Npy_INTERFACE(m1));
    if (ret == NULL) {
        return NULL;
    }
    if (NpyArray_CopyInto(ret, m1) == -1) {
        _Npy_DECREF(ret);
        return NULL;
    }

    return ret;
}


int 
NpyArray_ToBinaryFile(NpyArray *self, FILE *fp)
{
    npy_intp size;
    npy_intp n;
    NpyArrayIterObject *it;
        
    /* binary data */
    if (NpyDataType_FLAGCHK(self->descr, NPY_LIST_PICKLE)) {
        NpyErr_SetString(NpyExc_ValueError, "cannot write " \
                         "object arrays to a file in "   \
                         "binary mode");
        return -1;
    }
    
    if (NpyArray_ISCONTIGUOUS(self)) {
        size = NpyArray_SIZE(self);
        NPY_BEGIN_ALLOW_THREADS;
        n = fwrite((const void *)self->data,
                   (size_t) self->descr->elsize,
                   (size_t) size, fp);
        NPY_END_ALLOW_THREADS;
        if (n < size) {
            NpyErr_Format(NpyExc_ValueError,
                          "%ld requested and %ld written",
                          (long) size, (long) n);
            return -1;
        }
    }
    else {
        NPY_BEGIN_THREADS_DEF;
        
        it = NpyArray_IterNew(self);
        NPY_BEGIN_THREADS;
        while (it->index < it->size) {
            if (fwrite((const void *)it->dataptr,
                       (size_t) self->descr->elsize,
                       1, fp) < 1) {
                NPY_END_THREADS;
                NpyErr_Format(NpyExc_IOError,
                              "problem writing element"\
                              " %"NPY_INTP_FMT" to file",
                              it->index);
                _Npy_DECREF(it);
                return -1;
            }
            NpyArray_ITER_NEXT(it);
        }
        NPY_END_THREADS;
        _Npy_DECREF(it);
    }
    return 0;
}
