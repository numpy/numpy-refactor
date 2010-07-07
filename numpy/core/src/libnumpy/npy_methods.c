/*
 *  npy_methods.c - 
 *  
 */

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"





/* steals typed reference */
/*NUMPY_API
 Get a subset of bytes from each element of the array
 */
NpyArray *
NpyArray_GetField(NpyArray *self, NpyArray_Descr *typed, int offset)
{
    NpyArray *ret = NULL;
    
    if (offset < 0 || (offset + typed->elsize) > self->descr->elsize) {
        NpyErr_Format(NpyExc_ValueError,
                      "Need 0 <= offset <= %d for requested type "  \
                      "but received offset = %d",
                      self->descr->elsize-typed->elsize, offset);
        _Npy_DECREF(typed);
        return NULL;
    }
    ret = NpyArray_NewFromDescr(Npy_TYPE(self),
                                typed,
                                self->nd, self->dimensions,
                                self->strides,
                                self->data + offset,
                                self->flags, (NpyObject *)self);
    if (ret == NULL) {
        return NULL;
    }
    Npy_INCREF(self);
    ret->base_arr = self;
    assert(NULL == ret->base_arr || NULL == ret->base_obj);
    
    NpyArray_UpdateFlags(ret, NPY_UPDATE_ALL);
    return ret;
}




/*NUMPY_API
 Set a subset of bytes from each element of the array
 */
int 
NpyArray_SetField(NpyArray *self, NpyArray_Descr *dtype,
                  int offset, NpyObject *val)
{
    NpyArray *ret = NULL;
    int retval = 0;
    
    if (offset < 0 || (offset + dtype->elsize) > self->descr->elsize) {
        NpyErr_Format(NpyExc_ValueError,
                      "Need 0 <= offset <= %d for requested type "  \
                      "but received offset = %d",
                      self->descr->elsize-dtype->elsize, offset);
        _Npy_DECREF(dtype);
        return -1;
    }
    ret = NpyArray_NewFromDescr(Npy_TYPE(self),
                                dtype, self->nd, self->dimensions,
                                self->strides, self->data + offset,
                                self->flags, (NpyObject *)self);
    if (ret == NULL) {
        return -1;
    }
    Npy_INCREF(self);
    ret->base_arr = self;
    assert(NULL == ret->base_arr || NULL == ret->base_obj);
    
    NpyArray_UpdateFlags(ret, NPY_UPDATE_ALL);
    retval = NpyArray_CopyObject(ret, val);
    Npy_DECREF(ret);
    return retval;
}




/* This doesn't change the descriptor just the actual data...
 */

/*NUMPY_API*/
NpyArray *
NpyArray_Byteswap(NpyArray *self, npy_bool inplace)
{
    NpyArray *ret;
    npy_intp size;
    NpyArray_CopySwapNFunc *copyswapn;
    NpyArrayIterObject *it;
    
    copyswapn = self->descr->f->copyswapn;
    if (inplace) {
        if (!NpyArray_ISWRITEABLE(self)) {
            NpyErr_SetString(NpyExc_RuntimeError,
                             "Cannot byte-swap in-place on a " \
                             "read-only array");
            return NULL;
        }
        size = NpyArray_SIZE(self);
        if (NpyArray_ISONESEGMENT(self)) {
            copyswapn(self->data, self->descr->elsize, NULL, -1, size, 1, self);
        }
        else { /* Use iterator */
            int axis = -1;
            npy_intp stride;
            it = NpyArray_IterAllButAxis(self, &axis);
            stride = self->strides[axis];
            size = self->dimensions[axis];
            while (it->index < it->size) {
                copyswapn(it->dataptr, stride, NULL, -1, size, 1, self);
                NpyArray_ITER_NEXT(it);
            }
            _Npy_DECREF(it);
        }
        
        Npy_INCREF(self);
        return self;
    }
    else {
        NpyArray *new;
        if ((ret = NpyArray_NewCopy(self,-1)) == NULL) {
            return NULL;
        }
        new = NpyArray_Byteswap(ret, NPY_TRUE);
        Npy_DECREF(new);
        return ret;
    }
}





/*
 * compare the field dictionary for two types
 * return 1 if the same or 0 if not
 */
static int
_equivalent_fields(NpyDict *field1, NpyDict *field2) 
{
    NpyDict_Iter pos;
    NpyArray_DescrField *value1, *value2;
    const char *key;
    int same=1;
    
    if (field1 == field2) {
        return 1;
    }
    if (field1 == NULL || field2 == NULL) {
        return 0;
    }
    if (NpyDict_Size(field1) != NpyDict_Size(field2)) {
        same = 0;
    }
    
    NpyDict_IterInit(&pos);
    while (same && NpyDict_IterNext(field1, &pos, (void **)&key, (void **)&value1)) {
        value2 = NpyDict_Get(field2, key);
        if (NULL == value2 || value1->offset != value2->offset ||
            ((NULL == value1->title && NULL != value2->title) ||
             (NULL != value1->title && NULL == value2->title) ||
             (NULL != value1->title && NULL != value2->title && 
              strcmp(value1->title, value2->title)))) {
            same = 0;
        } else if (!NpyArray_EquivTypes(value1->descr, value2->descr)) {
            same = 0;
        }
    }
    return same;
}

/*
 * compare the metadata for two date-times
 * return 1 if they are the same
 * or 0 if not
 */
static int
_equivalent_units(NpyObject *meta1, NpyObject *meta2)
{
    /* TODO: Refactor this once metadata is converted to non-Python structures. */
    NpyObject *cobj1, *cobj2;
    NpyArray_DatetimeMetaData *data1, *data2;
    
    /* Same meta object */
    if (meta1 == meta2) {
        return 1;
    }
    
    cobj1 = PyDict_GetItemString(meta1, NPY_METADATA_DTSTR);
    cobj2 = PyDict_GetItemString(meta2, NPY_METADATA_DTSTR);
    if (cobj1 == cobj2) {
        return 1;
    }
    
    /* FIXME
     * There is no err handling here.
     */
    data1 = NpyCapsule_AsVoidPtr(cobj1);
    data2 = NpyCapsule_AsVoidPtr(cobj2);
    return ((data1->base == data2->base)
            && (data1->num == data2->num)
            && (data1->den == data2->den)
            && (data1->events == data2->events));
}


/*NUMPY_API
 *
 * This function returns true if the two typecodes are
 * equivalent (same basic kind and same itemsize).
 */
NPY_NO_EXPORT unsigned char
NpyArray_EquivTypes(NpyArray_Descr *typ1, NpyArray_Descr *typ2)
{
    int typenum1 = typ1->type_num;
    int typenum2 = typ2->type_num;
    int size1 = typ1->elsize;
    int size2 = typ2->elsize;
    
    if (size1 != size2) {
        return NPY_FALSE;
    }
    if (NpyArray_ISNBO(typ1->byteorder) != NpyArray_ISNBO(typ2->byteorder)) {
        return NPY_FALSE;
    }
    if (typenum1 == NPY_VOID
        || typenum2 == NPY_VOID) {
        return ((typenum1 == typenum2)
                && _equivalent_fields(typ1->fields, typ2->fields));
    }
    if (typenum1 == NPY_DATETIME
        || typenum1 == NPY_DATETIME
        || typenum2 == NPY_TIMEDELTA
        || typenum2 == NPY_TIMEDELTA) {
        return ((typenum1 == typenum2)
                && _equivalent_units(typ1->metadata, typ2->metadata));
    }
    return typ1->kind == typ2->kind;
}


