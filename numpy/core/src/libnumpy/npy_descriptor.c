/*
 *  npy_descriptor.c - 
 *  
 */

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"





/*NUMPY_API*/
NpyArray_Descr *
NpyArray_DescrNewFromType(int type_num)
{
    NpyArray_Descr *old;
    NpyArray_Descr *new;
    
    old = NpyArray_DescrFromType(type_num);
    new = NpyArray_DescrNew(old);
    Npy_DECREF(old);
    return new;
}



/** Array Descr Objects for dynamic types **/

/*
 * There are some statically-defined PyArray_Descr objects corresponding
 * to the basic built-in types.
 * These can and should be DECREF'd and INCREF'd as appropriate, anyway.
 * If a mistake is made in reference counting, deallocation on these
 * builtins will be attempted leading to problems.
 *
 * This let's us deal with all PyArray_Descr objects using reference
 * counting (regardless of whether they are statically or dynamically
 * allocated).
 */

/*NUMPY_API
 * base cannot be NULL
 */
NpyArray_Descr *
NpyArray_DescrNew(NpyArray_Descr *base)
{
    NpyArray_Descr *new = NpyObject_New(NpyArray_Descr, &NpyArrayDescr_Type);
    
    if (new == NULL) {
        return NULL;
    }
    /* Don't copy PyObject_HEAD part */
    /* TODO: Fix memory allocation once PyObject head structure is removed. */
    memcpy((char *)new + sizeof(PyObject),
           (char *)base + sizeof(PyObject),
           sizeof(NpyArray_Descr) - sizeof(PyObject));
    
    /* TODO: Fix once NpyDescr fields no longer use python objects. */
    if (new->fields == Py_None) {
        new->fields = NULL;
    }
    Npy_XINCREF(new->fields);
    Npy_XINCREF(new->names);
    if (new->subarray) {
        new->subarray = NpyArray_malloc(sizeof(NpyArray_ArrayDescr));
        memcpy(new->subarray, base->subarray, sizeof(NpyArray_ArrayDescr));
        Npy_INCREF(new->subarray->shape);
        Npy_INCREF(new->subarray->base);
    }
    Npy_XINCREF(new->typeobj);
    Npy_XINCREF(new->metadata);
    
    return new;
}
