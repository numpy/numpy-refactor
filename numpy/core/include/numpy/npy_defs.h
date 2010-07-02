#ifndef _NPY_DEFS_H_
#define _NPY_DEFS_H_


/* TODO: This needs to be fixed so we are not dependent on Python. */
#include <Python.h>

#include "npy_common.h"

/* 
 * This file contains defines and basic types used by the core
 * library.
 */

/* VALID indicates a currently-allocated object, INVALID means object has
   been deallocated. */
#define NPY_VALID_MAGIC 1234567
#define NPY_INVALID_MAGIC 0xdeadbeef


/*
 * This is to typedef npy_intp to the appropriate pointer size for
 * this platform.  Py_intptr_t, Py_uintptr_t are defined in pyport.h.
 */
typedef Py_intptr_t npy_intp;
typedef Py_uintptr_t npy_uintp;
#define NPY_SIZEOF_INTP NPY_SIZEOF_PY_INTPTR_T
#define NPY_SIZEOF_UINTP NPY_SIZEOF_PY_INTPTR_T


/*
 * There are several places in the code where an array of dimensions
 * is allocated statically.  This is the size of that static
 * allocation.
 *
 * The array creation itself could have arbitrary dimensions but all
 * the places where static allocation is used would need to be changed
 * to dynamic (including inside of several structures)
 */

#define NPY_MAXDIMS 32
#define NPY_MAXARGS 32


/* Forward structure declarations */

/* Temporary forward structure declarations.  These will be removed as objects
 are refactored into two layers */
struct PyArrayObject;
struct PyArrayFlagsObject;
struct _PyArray_Descr;
struct _arr_descr;
struct PyArray_DatetimeMetaData;
struct PyArray_Dims;


typedef PyObject NpyObject;                             /* An object opaque to core but understood by the interface layer */
typedef struct PyArrayObject NpyArray;
typedef struct _PyArray_Descr NpyArray_Descr;
typedef struct _arr_descr NpyArray_ArrayDescr;
typedef struct PyArray_DatetimeMetaData NpyArray_DatetimeMetaData;

typedef struct PyArray_Dims NpyArray_Dims;




#endif
