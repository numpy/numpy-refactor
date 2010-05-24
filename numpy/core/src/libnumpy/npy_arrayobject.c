
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"

/*
 * Compute the size of an array (in number of items)
 */
npy_intp
NpyArray_Size(NpyArray *op)
{
    return NpyArray_SIZE(op);
}

int
NpyArray_TypeNumFromName(char *str)
{
    int i;
    NpyArray_Descr *descr;

    for (i = 0; i < NPY_NUMUSERTYPES; i++) {
        descr = userdescrs[i];
        /* XXX: We are looking at the python type for the name. This 
           will need to be fixed. */
        if (strcmp(descr->typeobj->tp_name, str) == 0) {
            return descr->type_num;
        }
    }
    return PyArray_NOTYPE;
}
