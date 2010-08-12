/*
 *  npy_refcount.c -
 *
 */

#include <stdlib.h>
#include <strings.h>
#include "npy_config.h"
#include "numpy_api.h"
#include "npy_dict.h"
#include "npy_iterators.h"
#include "npy_arrayobject.h"
#include "npy_descriptor.h"


/* Incref all objects found at this record */
/*NUMPY_API
 */
void
NpyArray_Item_INCREF(char *data, NpyArray_Descr *descr)
{
    void *temp;

    if (!NpyDataType_REFCHK(descr)) {
        return;
    }
    if (descr->type_num == NPY_OBJECT) {
        NPY_COPY_PYOBJECT_PTR(&temp, data);
        temp = NpyInterface_INCREF(temp);
        NPY_COPY_PYOBJECT_PTR(data, &temp);
    }
    else if (NpyDataType_HASFIELDS(descr)) {
        const char *key;
        NpyArray_DescrField *value;
        NpyDict_Iter pos;

        NpyDict_IterInit(&pos);
        while (NpyDict_IterNext(descr->fields, &pos, (void **)&key, (void **)&value)) {
            if (NULL != value->title && !strcmp(value->title, key)) {
                continue;
            }
            NpyArray_Item_INCREF(data + value->offset, value->descr);
        }
    }
    return;
}


void
NpyArray_Item_XDECREF(char *data, NpyArray_Descr *descr)
{
    void *temp;

    if (!NpyDataType_REFCHK(descr)) {
        return;
    }

    if (descr->type_num == NPY_OBJECT) {
        NPY_COPY_VOID_PTR(&temp, data);
        NpyInterface_DECREF(temp);
    }
    else if (NpyDataType_HASFIELDS(descr)) {
        const char *key;
        NpyArray_DescrField *value;
        NpyDict_Iter pos;

        NpyDict_IterInit(&pos);
        while (NpyDict_IterNext(descr->fields, &pos, (void **)&key, (void **)&value)) {
            if (NULL != value->title && !strcmp(value->title, key)) {
                continue;
            }
            NpyArray_Item_XDECREF(data + value->offset, value->descr);
        }
    }
    return;
}




/* Used for arrays of python objects to increment the reference count of */
/* every python object in the array. */
int
NpyArray_INCREF(NpyArray *mp)
{
    npy_intp i, n;
    void **data;
    void *temp;
    NpyArrayIterObject *it;
    
    if (!NpyDataType_REFCHK(NpyArray_DESCR(mp))) {
        return 0;
    }
    if (NpyArray_TYPE(mp) != NPY_OBJECT) {
        it = NpyArray_IterNew(mp);
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            NpyArray_Item_INCREF(it->dataptr, NpyArray_DESCR(mp));
            NpyArray_ITER_NEXT(it);
        }
        _Npy_DECREF(it);
    } else if (NpyArray_ISONESEGMENT(mp)) {
        data = (void **)NpyArray_BYTES(mp);
        n = NpyArray_SIZE(mp);
        if (NpyArray_ISALIGNED(mp)) {
            for (i = 0; i < n; i++, data++) {
                *data = NpyInterface_INCREF(*data);
            }
        }
        else {
            for( i = 0; i < n; i++, data++) {
                NPY_COPY_PYOBJECT_PTR(&temp, data);
                temp = NpyInterface_INCREF(temp);
                NPY_COPY_PYOBJECT_PTR(data, &temp);
            }
        }
    }
    else { /* handles misaligned data too */
        it = NpyArray_IterNew(mp);
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            NPY_COPY_PYOBJECT_PTR(&temp, it->dataptr);
            temp = NpyInterface_INCREF(temp);
            NPY_COPY_PYOBJECT_PTR(it->dataptr, &temp);
            NpyArray_ITER_NEXT(it);
        }
        _Npy_DECREF(it);
    }
    return 0;
}

/*Decrement all internal references for object arrays.
 (or arrays with object fields)
 */
int
NpyArray_XDECREF(NpyArray *mp)
{
    npy_intp i, n;
    void **data;
    void *temp;
    NpyArrayIterObject *it;
    
    if (!NpyDataType_REFCHK(NpyArray_DESCR(mp))) {
        return 0;
    }
    if (NpyArray_TYPE(mp) != NPY_OBJECT) {
        it = NpyArray_IterNew(mp);
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            NpyArray_Item_XDECREF(it->dataptr, NpyArray_DESCR(mp));
            NpyArray_ITER_NEXT(it);
        }
        _Npy_DECREF(it);
        
    } else if (NpyArray_ISONESEGMENT(mp)) {
        data = (void **)NpyArray_BYTES(mp);
        n = NpyArray_SIZE(mp);
        if (NpyArray_ISALIGNED(mp)) {
            for (i = 0; i < n; i++, data++) NpyInterface_DECREF(*data);
        }
        else {
            for (i = 0; i < n; i++, data++) {
                NPY_COPY_PYOBJECT_PTR(&temp, data);
                NpyInterface_DECREF(temp);
            }
        }
    }
    else { /* handles misaligned data too */
        it = NpyArray_IterNew(mp);
        if (it == NULL) {
            return -1;
        }
        while(it->index < it->size) {
            NPY_COPY_PYOBJECT_PTR(&temp, it->dataptr);
            NpyInterface_DECREF(temp);
            NpyArray_ITER_NEXT(it);
        }
        _Npy_DECREF(it);
    }
    return 0;
}

