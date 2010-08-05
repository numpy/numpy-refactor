#ifndef _NPY_DEFS_H_
#define _NPY_DEFS_H_

#include <stdio.h>
#include <stdint.h>
#include <numpy/npy_common.h>
#include <numpy/npy_endian.h>
#include <pyconfig.h>


/*
 * This file contains defines and basic types used by the core
 * library.
 */

/* VALID indicates a currently-allocated object, INVALID means object has
   been deallocated. */
#define NPY_VALID_MAGIC 1234567
#define NPY_INVALID_MAGIC 0xdeadbeef


/* uintptr_t is the C9X name for an unsigned integral type such that a
 * legitimate void* can be cast to uintptr_t and then back to void* again
 * without loss of information.  Similarly for intptr_t, wrt a signed
 * integral type.
 */
#ifdef HAVE_UINTPTR_T
typedef uintptr_t	npy_uintp;
typedef intptr_t	npy_intp;

#elif NPY_SIZEOF_PY_INTPTR_T <= NPY_SIZEOF_INT
typedef unsigned int	npy_uintp;
typedef int		npy_intp;

#elif NPY_SIZEOF_PY_INTPTR_T <= NPY_SIZEOF_LONG
typedef unsigned long	npy_uintp;
typedef long		npy_intp;

#elif defined(HAVE_LONG_LONG) && (NPY_SIZEOF_PY_INTPTR_T <= NPY_SIZEOF_PY_LONG_LONG)
typedef unsigned long long	npy_uintp;
typedef long long		npy_intp;


#else
#   error "NumPy needs a typedef for npy_uintp and npy_intp."
#endif /* HAVE_UINTPTR_T */


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



#if NPY_SIZEOF_PY_INTPTR_T == NPY_SIZEOF_INT
#define NPY_INTP NPY_INT
#define NPY_UINTP NPY_UINT
#define PyIntpArrType_Type PyIntArrType_Type
#define PyUIntpArrType_Type PyUIntArrType_Type
#define NPY_MAX_INTP NPY_MAX_INT
#define NPY_MIN_INTP NPY_MIN_INT
#define NPY_MAX_UINTP NPY_MAX_UINT
#define NPY_INTP_FMT "d"
#elif NPY_SIZEOF_PY_INTPTR_T == NPY_SIZEOF_LONG
#define NPY_INTP NPY_LONG
#define NPY_UINTP NPY_ULONG
#define PyIntpArrType_Type PyLongArrType_Type
#define PyUIntpArrType_Type PyULongArrType_Type
#define NPY_MAX_INTP NPY_MAX_LONG
#define NPY_MIN_INTP MIN_LONG
#define NPY_MAX_UINTP NPY_MAX_ULONG
#define NPY_INTP_FMT "ld"
#elif defined(PY_LONG_LONG) && (NPY_SIZEOF_PY_INTPTR_T == NPY_SIZEOF_LONGLONG)
#define NPY_INTP NPY_LONGLONG
#define NPY_UINTP NPY_ULONGLONG
#define PyIntpArrType_Type PyLongLongArrType_Type
#define PyUIntpArrType_Type PyULongLongArrType_Type
#define NPY_MAX_INTP NPY_MAX_LONGLONG
#define NPY_MIN_INTP NPY_MIN_LONGLONG
#define NPY_MAX_UINTP NPY_MAX_ULONGLONG
#define NPY_INTP_FMT "Ld"
#endif


/* The item must be reference counted when it is inserted or extracted. */
#define NPY_ITEM_REFCOUNT   0x01
/* Same as needing REFCOUNT */
#define NPY_ITEM_HASOBJECT  0x01
/* Convert to list for pickling */
#define NPY_LIST_PICKLE     0x02
/* The item is a POINTER  */
#define NPY_ITEM_IS_POINTER 0x04
/* memory needs to be initialized for this data-type */
#define NPY_NEEDS_INIT      0x08
/* operations need Python C-API so don't give-up thread. */
#define NPY_NEEDS_PYAPI     0x10
/* Use f.getitem when extracting elements of this data-type */
#define NPY_USE_GETITEM     0x20
/* Use f.setitem when setting creating 0-d array from this data-type.*/
#define NPY_USE_SETITEM     0x40


/* Data-type needs extra initialization on creation */
#define NPY_EXTRA_DTYPE_INIT 0x80

/* When creating an array of this type -- call extra function */
#define NPY_UFUNC_OUTPUT_CREATION 0x100

/*
 *These are inherited for global data-type if any data-types in the
 * field have them
 */
#define NPY_FROM_FIELDS    (NPY_NEEDS_INIT | NPY_LIST_PICKLE |             \
NPY_ITEM_REFCOUNT | NPY_NEEDS_PYAPI)

#define NPY_OBJECT_DTYPE_FLAGS (NPY_LIST_PICKLE | NPY_USE_GETITEM |       \
NPY_ITEM_IS_POINTER | NPY_ITEM_REFCOUNT | \
NPY_NEEDS_INIT | NPY_NEEDS_PYAPI)



/*
 * Means c-style contiguous (last index varies the fastest). The data
 * elements right after each other.
 */
#define NPY_CONTIGUOUS    0x0001

/*
 * set if array is a contiguous Fortran array: the first index varies
 * the fastest in memory (strides array is reverse of C-contiguous
 * array)
 */
#define NPY_FORTRAN       0x0002

#define NPY_C_CONTIGUOUS NPY_CONTIGUOUS
#define NPY_F_CONTIGUOUS NPY_FORTRAN

/*
 * Note: all 0-d arrays are CONTIGUOUS and FORTRAN contiguous. If a
 * 1-d array is CONTIGUOUS it is also FORTRAN contiguous
 */

/*
 * If set, the array owns the data: it will be free'd when the array
 * is deleted.
 */
#define NPY_OWNDATA       0x0004

/*
 * An array never has the next four set; they're only used as parameter
 * flags to the the various FromAny functions
 */

/* Cause a cast to occur regardless of whether or not it is safe. */
#define NPY_FORCECAST     0x0010

/*
 * Always copy the array. Returned arrays are always CONTIGUOUS,
 * ALIGNED, and WRITEABLE.
 */
#define NPY_ENSURECOPY    0x0020

/* Make sure the returned array is a base-class ndarray */
#define NPY_ENSUREARRAY   0x0040

/*
 * Make sure that the strides are in units of the element size Needed
 * for some operations with record-arrays.
 */
#define NPY_ELEMENTSTRIDES 0x0080

/*
 * Array data is aligned on the appropiate memory address for the type
 * stored according to how the compiler would align things (e.g., an
 * array of integers (4 bytes each) starts on a memory address that's
 * a multiple of 4)
 */
#define NPY_ALIGNED       0x0100

/* Array data has the native endianness */
#define NPY_NOTSWAPPED    0x0200

/* Array data is writeable */
#define NPY_WRITEABLE     0x0400

/*
 * If this flag is set, then base contains a pointer to an array of
 * the same size that should be updated with the current contents of
 * this array when this array is deallocated
 */
#define NPY_UPDATEIFCOPY  0x1000

/* This flag is for the array interface */
#define NPY_ARR_HAS_DESCR  0x0800


#define NPY_BEHAVED (NPY_ALIGNED | NPY_WRITEABLE)
#define NPY_BEHAVED_NS (NPY_ALIGNED | NPY_WRITEABLE | NPY_NOTSWAPPED)
#define NPY_CARRAY (NPY_CONTIGUOUS | NPY_BEHAVED)
#define NPY_CARRAY_RO (NPY_CONTIGUOUS | NPY_ALIGNED)
#define NPY_FARRAY (NPY_FORTRAN | NPY_BEHAVED)
#define NPY_FARRAY_RO (NPY_FORTRAN | NPY_ALIGNED)
#define NPY_DEFAULT NPY_CARRAY
#define NPY_IN_ARRAY NPY_CARRAY_RO
#define NPY_OUT_ARRAY NPY_CARRAY
#define NPY_INOUT_ARRAY (NPY_CARRAY | NPY_UPDATEIFCOPY)
#define NPY_IN_FARRAY NPY_FARRAY_RO
#define NPY_OUT_FARRAY NPY_FARRAY
#define NPY_INOUT_FARRAY (NPY_FARRAY | NPY_UPDATEIFCOPY)

#define NPY_UPDATE_ALL (NPY_CONTIGUOUS | NPY_FORTRAN | NPY_ALIGNED)


/*
 * Size of internal buffers used for alignment Make BUFSIZE a multiple
 * of sizeof(cdouble) -- ususally 16 so that ufunc buffers are aligned
 */
#define NPY_MIN_BUFSIZE ((int)sizeof(cdouble))
#define NPY_MAX_BUFSIZE (((int)sizeof(cdouble))*1000000)
#define NPY_BUFSIZE 10000
/* #define NPY_BUFSIZE 80*/



/* TODO: Need to generalize the threading interface. */
#if NPY_ALLOW_THREADS
#define NPY_BEGIN_ALLOW_THREADS Py_BEGIN_ALLOW_THREADS
#define NPY_END_ALLOW_THREADS Py_END_ALLOW_THREADS
#define NPY_BEGIN_THREADS_DEF PyThreadState *_save=NULL;
#define NPY_BEGIN_THREADS _save = PyEval_SaveThread();
#define NPY_END_THREADS   do {if (_save) PyEval_RestoreThread(_save);} while (0);

#define NPY_BEGIN_THREADS_DESCR(dtype)                          \
do {if (!(NpyDataType_FLAGCHK(dtype, NPY_NEEDS_PYAPI)))      \
NPY_BEGIN_THREADS;} while (0);

#define NPY_END_THREADS_DESCR(dtype)                            \
do {if (!(NpyDataType_FLAGCHK(dtype, NPY_NEEDS_PYAPI)))      \
NPY_END_THREADS; } while (0);

#define NPY_ALLOW_C_API_DEF  PyGILState_STATE __save__;
#define NPY_ALLOW_C_API      __save__ = PyGILState_Ensure();
#define NPY_DISABLE_C_API    PyGILState_Release(__save__);
#else
#define NPY_BEGIN_ALLOW_THREADS
#define NPY_END_ALLOW_THREADS
#define NPY_BEGIN_THREADS_DEF
#define NPY_BEGIN_THREADS
#define NPY_END_THREADS
#define NPY_BEGIN_THREADS_DESCR(dtype)
#define NPY_END_THREADS_DESCR(dtype)
#define NPY_ALLOW_C_API_DEF
#define NPY_ALLOW_C_API
#define NPY_DISABLE_C_API
#endif


#define NpyArray_MAX(a,b) (((a)>(b))?(a):(b))
#define NpyArray_MIN(a,b) (((a)<(b))?(a):(b))

/* The default array type */
#define NPY_DEFAULT_TYPE NPY_DOUBLE


#define NpyArray_CLT(p,q) ((((p).real==(q).real) ? ((p).imag < (q).imag) : \
    ((p).real < (q).real)))
#define NpyArray_CGT(p,q) ((((p).real==(q).real) ? ((p).imag > (q).imag) : \
    ((p).real > (q).real)))
#define NpyArray_CLE(p,q) ((((p).real==(q).real) ? ((p).imag <= (q).imag) : \
    ((p).real <= (q).real)))
#define NpyArray_CGE(p,q) ((((p).real==(q).real) ? ((p).imag >= (q).imag) : \
    ((p).real >= (q).real)))
#define NpyArray_CEQ(p,q) (((p).real==(q).real) && ((p).imag == (q).imag))
#define NpyArray_CNE(p,q) (((p).real!=(q).real) || ((p).imag != (q).imag))



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

typedef int (NpyArray_CompareFunc)(const void *, const void *,
                                   struct NpyArray *);
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
typedef int (NpyArray_ArgSortFunc)(void *, npy_intp *, npy_intp,
                                   struct NpyArray *);

typedef int (NpyArray_FillWithScalarFunc)(void *, npy_intp, void *,
                                          struct NpyArray *);

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
