/*
 *  npy_refcount.c -
 *
 */

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN

#include "npy_config.h"
#include "numpy/numpy_api.h"


/* Incref all objects found at this record */
/*NUMPY_API
 */
NPY_NO_EXPORT void
NpyArray_Item_INCREF(char *data, NpyArray_Descr *descr)
{
    void *temp;

    if (!NpyDataType_REFCHK(descr)) {
        return;
    }
    if (descr->type_num == NPY_OBJECT) {
        NPY_COPY_PYOBJECT_PTR(&temp, data);
        Npy_Interface_XINCREF(temp);
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


NPY_NO_EXPORT void
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

