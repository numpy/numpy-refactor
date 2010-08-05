/*
 *  npy_refcount.c -
 *
 */

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include <stdlib.h>
#include <strings.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"
#include "numpy/npy_dict.h"


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
        NpyInterface_Incref(temp);
        /* TODO: Fix for garbage collected environments - needs to store possibly new pointer */
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
        NpyInterface_XDecref(temp);
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

