#ifndef _NPY_DEFS_H_
#define _NPY_DEFS_H_

#include <Python.h>
#include <stdio.h>
#include <stdint.h>
#include <numpy/npy_common.h>
#include <numpy/npy_endian.h>


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
/* TODO: Need a platform-dependent size for npy_intp */
typedef Py_intptr_t npy_intp;
typedef Py_uintptr_t npy_uintp;


/*typedef Py_intptr_t npy_intp;
typedef Py_uintptr_t npy_uintp; */
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




/* Forward type declarations */
struct NpyDict_struct;      /* From npy_dict.c, numpy_api.h */

struct NpyArray_ArrayDescr;
struct NpyArray_ArrFuncs;
struct NpyArray_CastFuncsItem;

typedef struct NpyArray NpyArray;
typedef struct NpyArray_ArrayDescr NpyArray_ArrayDescr;
typedef struct NpyArray_DatetimeMetaData NpyArray_DatetimeMetaData;
typedef struct NpyArray_DescrField NpyArray_DescrField;
typedef struct NpyArray_Descr NpyArray_Descr;
typedef struct NpyArray_ArrFuncs NpyArray_ArrFuncs;
typedef struct NpyArray_DateTimeInfo NpyArray_DateTimeInfo;
typedef struct NpyArray_Dims NpyArray_Dims;
typedef struct NpyArray_CastFuncsItem NpyArray_CastFuncsItem;



struct NpyArray_Dims {
    npy_intp *ptr;
    int len;
};

typedef struct {
    npy_longlong year;
    int month, day, hour, min, sec, us, ps, as;
} npy_datetimestruct;

typedef struct {
    npy_longlong day;
    int sec, us, ps, as;
} npy_timedeltastruct;




/*
 * These characters correspond to the array type and the struct
 * module
 */

/*  except 'p' -- signed integer for pointer type */

enum NPY_TYPECHAR { NPY_BOOLLTR = '?',
    NPY_BYTELTR = 'b',
    NPY_UBYTELTR = 'B',
    NPY_SHORTLTR = 'h',
    NPY_USHORTLTR = 'H',
    NPY_INTLTR = 'i',
    NPY_UINTLTR = 'I',
    NPY_LONGLTR = 'l',
    NPY_ULONGLTR = 'L',
    NPY_LONGLONGLTR = 'q',
    NPY_ULONGLONGLTR = 'Q',
    NPY_FLOATLTR = 'f',
    NPY_DOUBLELTR = 'd',
    NPY_LONGDOUBLELTR = 'g',
    NPY_CFLOATLTR = 'F',
    NPY_CDOUBLELTR = 'D',
    NPY_CLONGDOUBLELTR = 'G',
    NPY_OBJECTLTR = 'O',
    NPY_STRINGLTR = 'S',
    NPY_STRINGLTR2 = 'a',
    NPY_UNICODELTR = 'U',
    NPY_VOIDLTR = 'V',
    NPY_DATETIMELTR = 'M',
    NPY_TIMEDELTALTR = 'm',
    NPY_CHARLTR = 'c',

    /*
     * No Descriptor, just a define -- this let's
     * Python users specify an array of integers
     * large enough to hold a pointer on the
     * platform
     */
    NPY_INTPLTR = 'p',
    NPY_UINTPLTR = 'P',

    NPY_GENBOOLLTR ='b',
    NPY_SIGNEDLTR = 'i',
    NPY_UNSIGNEDLTR = 'u',
    NPY_FLOATINGLTR = 'f',
    NPY_COMPLEXLTR = 'c'
};



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



typedef enum {
    NPY_QUICKSORT=0,
    NPY_HEAPSORT=1,
    NPY_MERGESORT=2
} NPY_SORTKIND;
#define NPY_NSORTS (NPY_MERGESORT + 1)


typedef enum {
    NPY_SEARCHLEFT=0,
    NPY_SEARCHRIGHT=1
} NPY_SEARCHSIDE;
#define NPY_NSEARCHSIDES (NPY_SEARCHRIGHT + 1)


typedef enum {
    NPY_NOSCALAR=-1,
    NPY_BOOL_SCALAR,
    NPY_INTPOS_SCALAR,
    NPY_INTNEG_SCALAR,
    NPY_FLOAT_SCALAR,
    NPY_COMPLEX_SCALAR,
    NPY_OBJECT_SCALAR
} NPY_SCALARKIND;
#define NPY_NSCALARKINDS (NPY_OBJECT_SCALAR + 1)

typedef enum {
    NPY_ANYORDER=-1,
    NPY_CORDER=0,
    NPY_FORTRANORDER=1
} NPY_ORDER;


typedef enum {
    NPY_CLIP=0,
    NPY_WRAP=1,
    NPY_RAISE=2
} NPY_CLIPMODE;

typedef enum {
    NPY_FR_Y,
    NPY_FR_M,
    NPY_FR_W,
    NPY_FR_B,
    NPY_FR_D,
    NPY_FR_h,
    NPY_FR_m,
    NPY_FR_s,
    NPY_FR_ms,
    NPY_FR_us,
    NPY_FR_ns,
    NPY_FR_ps,
    NPY_FR_fs,
    NPY_FR_as
} NPY_DATETIMEUNIT;

#define NPY_DATETIME_NUMUNITS (NPY_FR_as + 1)
#define NPY_DATETIME_DEFAULTUNIT NPY_FR_us

#define NPY_STR_Y "Y"
#define NPY_STR_M "M"
#define NPY_STR_W "W"
#define NPY_STR_B "B"
#define NPY_STR_D "D"
#define NPY_STR_h "h"
#define NPY_STR_m "m"
#define NPY_STR_s "s"
#define NPY_STR_ms "ms"
#define NPY_STR_us "us"
#define NPY_STR_ns "ns"
#define NPY_STR_ps "ps"
#define NPY_STR_fs "fs"
#define NPY_STR_as "as"





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

typedef void (*npy_free_func)(void*);





/* These must deal with unaligned and swapped data if necessary */
typedef void * (NpyArray_GetItemFunc) (void *, struct NpyArray *);
typedef int (NpyArray_SetItemFunc)(void *, void *, struct NpyArray *);

typedef void (NpyArray_CopySwapNFunc)(void *, npy_intp, void *, npy_intp,
                                      npy_intp, int, struct NpyArray *);

typedef void (NpyArray_CopySwapFunc)(void *, void *, int, struct NpyArray *);
typedef npy_bool (NpyArray_NonzeroFunc)(void *, struct NpyArray *);


/*
 * These assume aligned and notswapped data -- a buffer will be used
 * before or contiguous data will be obtained
 */

typedef int (NpyArray_CompareFunc)(const void *, const void *, struct NpyArray *);
typedef int (NpyArray_ArgFunc)(void*, npy_intp, npy_intp*, struct NpyArray *);

typedef void (NpyArray_DotFunc)(void *, npy_intp, void *, npy_intp, void *,
                                npy_intp, struct NpyArray *);

typedef void (NpyArray_VectorUnaryFunc)(void *, void *, npy_intp, void *,
                                        void *);

/*
 * XXX the ignore argument should be removed next time the API version
 * is bumped. It used to be the separator.
 */
typedef int (NpyArray_ScanFunc)(FILE *fp, void *dptr,
                                char *ignore, struct NpyArray_Descr *);
typedef int (NpyArray_FromStrFunc)(char *s, void *dptr, char **endptr,
                                   struct NpyArray_Descr *);

typedef int (NpyArray_FillFunc)(void *, npy_intp, struct NpyArray *);

typedef int (NpyArray_SortFunc)(void *, npy_intp, struct NpyArray *);
typedef int (NpyArray_ArgSortFunc)(void *, npy_intp *, npy_intp, struct NpyArray *);

typedef int (NpyArray_FillWithScalarFunc)(void *, npy_intp, void *, struct NpyArray *);

typedef int (NpyArray_ScalarKindFunc)(struct NpyArray *);

typedef void (NpyArray_FastClipFunc)(void *in, npy_intp n_in, void *min,
                                     void *max, void *out);
typedef void (NpyArray_FastPutmaskFunc)(void *in, void *mask, npy_intp n_in,
                                        void *values, npy_intp nv);
typedef int  (NpyArray_FastTakeFunc)(void *dest, void *src, npy_intp *indarray,
                                     npy_intp nindarray, npy_intp n_outer,
                                     npy_intp m_middle, npy_intp nelem,
                                     NPY_CLIPMODE clipmode);


#endif
