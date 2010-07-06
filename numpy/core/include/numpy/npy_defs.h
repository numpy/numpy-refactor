#ifndef _NPY_DEFS_H_
#define _NPY_DEFS_H_

/* TODO: This needs to be fixed so we are not dependent on Python. */
#include <Python.h>

#include "npy_common.h"
#include "npy_endian.h"


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


/* 
 * TODO: Move this to npy_descriptor.h.
 */
enum NPY_TYPES {    NPY_BOOL=0,
                    NPY_BYTE, NPY_UBYTE,
                    NPY_SHORT, NPY_USHORT,
                    NPY_INT, NPY_UINT,
                    NPY_LONG, NPY_ULONG,
                    NPY_LONGLONG, NPY_ULONGLONG,
                    NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
                    NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
                    NPY_DATETIME, NPY_TIMEDELTA,
                    NPY_OBJECT=19,
                    NPY_STRING, NPY_UNICODE,
                    NPY_VOID,
                    NPY_NTYPES,
                    NPY_NOTYPE,
                    NPY_CHAR,      /* special flag */
                    NPY_USERDEF=256  /* leave room for characters */
};

#define NpyTypeNum_ISBOOL(type) ((type) == NPY_BOOL)

#define NpyTypeNum_ISUNSIGNED(type) (((type) == NPY_UBYTE) ||  \
                                     ((type) == NPY_USHORT) || \
                                     ((type) == NPY_UINT) ||   \
                                     ((type) == NPY_ULONG) ||  \
                                     ((type) == NPY_ULONGLONG))

#define NpyTypeNum_ISSIGNED(type) (((type) == NPY_BYTE) ||      \
                               ((type) == NPY_SHORT) ||        \
                               ((type) == NPY_INT) ||          \
                               ((type) == NPY_LONG) ||         \
                               ((type) == NPY_LONGLONG))

#define NpyTypeNum_ISINTEGER(type) (((type) >= NPY_BYTE) &&     \
                                    ((type) <= NPY_ULONGLONG))

#define NpyTypeNum_ISFLOAT(type) (((type) >= NPY_FLOAT) &&      \
                                  ((type) <= NPY_LONGDOUBLE))

#define NpyTypeNum_ISNUMBER(type) ((type) <= NPY_CLONGDOUBLE)

#define NpyTypeNum_ISSTRING(type) (((type) == NPY_STRING) ||    \
                                   ((type) == NPY_UNICODE))

#define NpyTypeNum_ISCOMPLEX(type) (((type) >= NPY_CFLOAT) &&   \
                                    ((type) <= NPY_CLONGDOUBLE))

#define NpyTypeNum_ISPYTHON(type) (((type) == NPY_LONG) ||      \
                                   ((type) == NPY_DOUBLE) ||    \
                                   ((type) == NPY_CDOUBLE) ||   \
                                   ((type) == NPY_BOOL) ||      \
                                   ((type) == NPY_OBJECT ))

#define NpyTypeNum_ISFLEXIBLE(type) (((type) >=NPY_STRING) &&  \
                                     ((type) <=NPY_VOID))

#define NpyTypeNum_ISDATETIME(type) (((type) >=NPY_DATETIME) &&  \
                                     ((type) <=NPY_TIMEDELTA))

#define NpyTypeNum_ISUSERDEF(type) (((type) >= NPY_USERDEF) && \
                                    ((type) < NPY_USERDEF+     \
                                     NPY_NUMUSERTYPES))

#define NpyTypeNum_ISEXTENDED(type) (NpyTypeNum_ISFLEXIBLE(type) ||  \
                                     NpyTypeNum_ISUSERDEF(type))

#define NpyTypeNum_ISOBJECT(type) ((type) == NPY_OBJECT)


#define NPY_LITTLE '<'
#define NPY_BIG '>'
#define NPY_NATIVE '='
#define NPY_SWAP 's'
#define NPY_IGNORE '|'

#if NPY_BYTE_ORDER == NPY_BIG_ENDIAN
#define NPY_NATBYTE NPY_BIG
#define NPY_OPPBYTE NPY_LITTLE
#else
#define NPY_NATBYTE NPY_LITTLE
#define NPY_OPPBYTE NPY_BIG
#endif

#define NpyArray_ISNBO(arg) ((arg) != NPY_OPPBYTE)
#define NpyArray_IsNativeByteOrder NpyArray_ISNBO

#endif
