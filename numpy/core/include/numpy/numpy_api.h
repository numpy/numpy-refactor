#ifndef _NUMPY_API_H_
#define _NUMPY_API_H_

#include "numpy/arrayobject.h"
#include "npy_3kcompat.h"
#include "assert.h"


typedef PyArray_CopySwapFunc NpyArray_CopySwapFunc;
typedef PyArray_CopySwapNFunc NpyArray_CopySwapNFunc;
typedef PyArray_ArrFuncs NpyArray_ArrFuncs;
typedef PyArray_ArgFunc NpyArray_ArgFunc;
typedef PyArray_VectorUnaryFunc NpyArray_VectorUnaryFunc;
typedef PyArray_FastTakeFunc NpyArray_FastTakeFunc;
typedef PyArray_FastPutmaskFunc NpyArray_FastPutmaskFunc;
typedef PyArray_SortFunc NpyArray_SortFunc;
typedef PyArray_ArgSortFunc NpyArray_ArgSortFunc;
typedef PyArray_CompareFunc NpyArray_CompareFunc;

typedef void (NpyArray_DotFunc)(void *, npy_intp, void *, npy_intp, void *, npy_intp, void *);

#define NpyTypeObject PyTypeObject
#define NpyArray_Type PyArray_Type
#define NpyArrayDescr_Type PyArrayDescr_Type

#define NpyArray_UCS4 npy_ucs4

#define Npy_TYPE(a) Py_TYPE(a)
#define NpyArray_SIZE(a) PyArray_SIZE(a)
#define NpyArray_BUFSIZE PyArray_BUFSIZE
#define NpyArray_ITEMSIZE(a) PyArray_ITEMSIZE(a)
#define NpyArray_NDIM(a) PyArray_NDIM(a)
#define NpyArray_DIM(a, i) PyArray_DIM(a, i)
#define NpyArray_DIMS(a) PyArray_DIMS(a)
#define NpyArray_STRIDES(a) PyArray_STRIDES(a)
#define NpyArray_STRIDE(obj, n) PyArray_STRIDE(obj,n)
#define NpyArray_DESCR(a) PyArray_DESCR(a)
#define NpyArray_FLAGS(a) PyArray_FLAGS(a)
#define NpyArray_BASE(a) PyArray_BASE(a)
#define NpyArray_BYTES(obj) PyArray_BYTES(obj) 
#define NpyArray_NBYTES(m) (NpyArray_ITEMSIZE(m) * NpyArray_SIZE(m))
#define NpyArray_CHKFLAGS(a, flags) PyArray_CHKFLAGS(a, flags)
#define NpyArray_ISFORTRAN(a) PyArray_ISFORTRAN(a)
#define NpyArray_ISCONTIGUOUS(a) PyArray_ISCONTIGUOUS(a)
#define NpyArray_ISONESEGMENT(a) PyArray_ISONESEGMENT(a)
#define NpyArray_ISFLEXIBLE(obj) PyTypeNum_ISFLEXIBLE(PyArray_TYPE(obj))
#define NpyArray_ISWRITEABLE(a) PyArray_ISWRITEABLE(a)
#define NpyArray_SAMESHAPE(a1, a2) PyArray_SAMESHAPE(a1,a2)
#define NpyTypeNum_ISCOMPLEX(a) PyTypeNum_ISCOMPLEX(a)
#define NpyTypeNum_ISNUMBER(a) PyTypeNum_ISNUMBER(a)
#define NpyTypeNum_ISBOOL(a) PyTypeNum_ISBOOL(a)
#define NpyTypeNum_ISOBJECT(a) PyTypeNum_ISOBJECT(a)
#define NpyTypeNum_ISINTEGER(a) PyTypeNum_ISINTEGER(a)
#define NpyTypeNum_ISSIGNED(a) PyTypeNum_ISSIGNED(a)
#define NpyTypeNum_ISUNSIGNED(a) PyTypeNum_ISUNSIGNED(a)
#define NpyTypeNum_ISFLOAT(a) PyTypeNum_ISFLOAT(a)
#define NpyArray_ISOBJECT(a) PyArray_ISOBJECT(a)
#define NpyArray_ISNUMBER(a) PyArray_ISNUMBER(a)
#define NpyArray_ISUNSIGNED(a) PyArray_ISUNSIGNED(a)

#define NpyDataType_FLAGCHK(dtype, flag)                                   \
        (((dtype)->flags & (flag)) == (flag))

#define NpyArray_DESCR_REPLACE(descr) PyArray_DESCR_REPLACE(descr)
#define NpyArray_ISNBO(arg) ((arg) != NPY_OPPBYTE)
#define NpyArray_IsNativeByteOrder NpyArray_ISNBO
#define NpyArray_ISNOTSWAPPED(m) NpyArray_ISNBO(PyArray_DESCR(m)->byteorder)
#define NpyArray_ISBYTESWAPPED(m) (!NpyArray_ISNOTSWAPPED(m))

#define NpyArray_FLAGSWAP(m, flags) (NpyArray_CHKFLAGS(m, flags) &&       \
        NpyArray_ISNOTSWAPPED(m))
#define NpyArray_EquivByteorders(b1, b2) PyArray_EquivByteorders(b1, b2)

#define NpyArray_SAFEALIGNEDCOPY(obj) PyArray_SAFEALIGNEDCOPY(obj)
#define NpyArray_ISCARRAY(m) PyArray_FLAGSWAP(m, NPY_CARRAY)
#define NpyArray_ISCARRAY_RO(m) PyArray_FLAGSWAP(m, NPY_CARRAY_RO)
#define NpyArray_ISFARRAY(m) PyArray_FLAGSWAP(m, NPY_FARRAY)
#define NpyArray_ISFARRAY_RO(m) PyArray_FLAGSWAP(m, NPY_FARRAY_RO)
#define NpyArray_ISBEHAVED(m) PyArray_FLAGSWAP(m, NPY_BEHAVED)
#define NpyArray_ISBEHAVED_RO(m) PyArray_FLAGSWAP(m, NPY_ALIGNED)
#define NpyArray_ISALIGNED(m) PyArray_ISALIGNED(m)

#define NpyArray_TYPE(obj) PyArray_TYPE(obj)
#define NpyArray_NOTYPE PyArray_NOTYPE
#define NpyArray_NTYPES PyArray_NTYPES
#define NpyArray_NSORTS PyArray_NSORTS
#define NpyArray_USERDEF PyArray_USERDEF
#define NpyTypeNum_ISUSERDEF(a) PyTypeNum_ISUSERDEF(a)
#define NpyArray_BOOL PyArray_BOOL
#define NpyArray_GENBOOLLTR PyArray_GENBOOLLTR
#define NpyArray_SIGNEDLTR PyArray_SIGNEDLTR
#define NpyArray_SHORT PyArray_SHORT
#define NpyArray_INT PyArray_INT
#define NpyArray_INT8 PyArray_INT8
#define NpyArray_INT16 PyArray_INT16
#define NpyArray_INT32 PyArray_INT32
#define NpyArray_INT64 PyArray_INT64
#define NpyArray_INTP PyArray_INTP
#define NpyArray_UNSIGNEDLTR PyArray_UNSIGNEDLTR
#define NpyArray_UINT8 PyArray_UINT8
#define NpyArray_UINT16 PyArray_UINT16
#define NpyArray_UINT32 PyArray_UINT32
#define NpyArray_UINT64 PyArray_UINT64
#define NpyArray_UINT  PyArray_UINT
#define NpyArray_LONG PyArray_LONG
#define NpyArray_LONGLONG PyArray_LONGLONG
#define NpyArray_ULONG PyArray_ULONG
#define NpyArray_ULONGLONG PyArray_ULONGLONG
#define NpyArray_FLOATINGLTR PyArray_FLOATINGLTR
#define NpyArray_FLOAT PyArray_FLOAT
#define NpyArray_DOUBLE PyArray_DOUBLE
#define NpyArray_LONGDOUBLE PyArray_LONGDOUBLE
#define NpyArray_CFLOAT PyArray_CFLOAT
#define NpyArray_CDOUBLE PyArray_CDOUBLE
#define NpyArray_CLONGDOUBLE PyArray_CLONGDOUBLE
#define NpyArray_FLOAT32 PyArray_FLOAT32
#define NpyArray_FLOAT64 PyArray_FLOAT64
#ifdef PyArray_FLOAT80
#define NpyArray_FLOAT80 PyArray_FLOAT80
#define NpyArray_COMPLEX160 PyArray_COMPLEX160
#endif
#ifdef PyArray_FLOAT96
#define NpyArray_FLOAT96 PyArray_FLOAT96
#define NpyArray_COMPLEX192 PyArray_COMPLEX192
#endif
#ifdef PyArray_FLOAT128
#define NpyArray_FLOAT128 PyArray_FLOAT128
#define NpyArray_COMPLEX256 PyArray_COMPLEX256
#endif
#define NpyArray_COMPLEXLTR PyArray_COMPLEXLTR
#define NpyArray_COMPLEX64 PyArray_COMPLEX64
#define NpyArray_COMPLEX128 PyArray_COMPLEX128
#define NpyArray_COMPLEX256 PyArray_COMPLEX256
#define NpyArray_STRING PyArray_STRING
#define NpyArray_UNICODE PyArray_UNICODE
#define NpyArray_VOID PyArray_VOID
#define NpyArray_BYTE PyArray_BYTE
#define NpyArray_UBYTE PyArray_UBYTE
#define NpyArray_USHORT PyArray_USHORT

#define NpyArray_NOSCALAR PyArray_NOSCALAR
#define NpyArray_NSCALARKINDS PyArray_NSCALARKINDS
#define NpyArray_FORTRANORDER NPY_FORTRANORDER

#define NpyDataType_ISSTRING(obj) PyDataType_ISSTRING(obj)
#define NpyArray_CheckExact(op) PyArray_CheckExact(op)
#define NpyArray_Check(op) PyArray_Check(op)

#define NpyDataType_REFCHK(a) PyDataType_REFCHK(a)


typedef struct NpyDict_KVPair_struct {
    const void *key;
    void *value;
    struct NpyDict_KVPair_struct *next;
} NpyDict_KVPair;

typedef struct NpyDict_struct {
    long numOfBuckets;
    long numOfElements;
    NpyDict_KVPair **bucketArray;
    float idealRatio, lowerRehashThreshold, upperRehashThreshold;
    int (*keycmp)(const void *key1, const void *key2);
    int (*valuecmp)(const void *value1, const void *value2);
    unsigned long (*hashFunction)(const void *key);
    void (*keyDeallocator)(void *key);
    void (*valueDeallocator)(void *value);
} NpyDict;

typedef struct {
    long bucket;
    NpyDict_KVPair *element;
} NpyDict_Iter;



/* Used as the value of an NpyDict to record the fields in an NpyArray_Descr object */
typedef struct {
    NpyArray_Descr *descr;
    int offset;
    char *title;                /* String owned/managed by each instance */
} NpyArray_DescrField;



/*
 * Functions we need to convert.
 */

/* arraytypes.c.src */
#define NpyArray_CopyObject(d, s) PyArray_CopyObject(d, s)  /* TODO: Needs to call back to interface layer */

#define NpyArray_DescrFromType(type) \
        PyArray_DescrFromType(type)

void NpyArray_dealloc(NpyArray *self);


/* common.c */
#define NpyString_Check(a) PyString_Check(a)        /* TODO: Npy_IsWriteable() need callback to interface for base of string, buffer */
#define NpyObject_AsWriteBuffer(a, b, c) PyObject_AsWriteBuffer(a, b, c) 
int Npy_IsAligned(NpyArray *ap);
npy_bool Npy_IsWriteable(NpyArray *ap);




/* ctors.c */
#define NpyArray_EnsureAnyArray(op)  (NpyArray *)PyArray_EnsureAnyArray(op)
size_t _array_fill_strides(npy_intp *strides, npy_intp *dims, int nd, size_t itemsize,
                           int inflag, int *objflags);


/* descriptor.c */
NpyArray_Descr *NpyArray_DescrNewFromType(int type_num);
NpyArray_Descr *NpyArray_DescrNew(NpyArray_Descr *base);
void NpyArray_DescrDestroy(NpyArray_Descr *);
char **NpyArray_DescrAllocNames(int n);
NpyDict *NpyArray_DescrAllocFields(void);
NpyArray_ArrayDescr *NpyArray_DupSubarray(NpyArray_ArrayDescr *src);
void NpyArray_DestroySubarray(NpyArray_ArrayDescr *);
void NpyArray_DescrDeallocNamesAndFields(NpyArray_Descr *base);
NpyArray_Descr *NpyArray_DescrNewByteorder(NpyArray_Descr *self, char newendian);
void NpyArray_DescrSetField(NpyDict *self, const char *key, NpyArray_Descr *descr,
                            int offset, const char *title);
NpyDict *NpyArray_DescrFieldsCopy(NpyDict *fields);
char **NpyArray_DescrNamesCopy(char **names);
int NpyArray_DescrReplaceNames(NpyArray_Descr *self, char **nameslist);
void NpyArray_DescrSetNames(NpyArray_Descr *self, char **nameslist);

/* npy_dict.c */
NpyDict *NpyDict_CreateTable(long numOfBuckets);
void NpyDict_Destroy(NpyDict *hashTable);
NpyDict *NpyDict_Copy(const NpyDict *orig, void *(*copyKey)(void *), void *(*copyValue)(void *));
int NpyDict_ContainsKey(const NpyDict *hashTable, const void *key);
int NpyDict_ContainsValue(const NpyDict *hashTable, const void *value);
int NpyDict_Put(NpyDict *hashTable, const void *key, void *value);
void *NpyDict_Get(const NpyDict *hashTable, const void *key);
void NpyDict_Rekey(NpyDict *hashTable, const void *oldKey, const void *newKey);
void NpyDict_Remove(NpyDict *hashTable, const void *key);
void NpyDict_RemoveAll(NpyDict *hashTable);
void NpyDict_IterInit(NpyDict_Iter *iter);
int NpyDict_IterNext(NpyDict *hashTable, NpyDict_Iter *iter, void **key, void **value);
int NpyDict_IsEmpty(const NpyDict *hashTable);
long NpyDict_Size(const NpyDict *hashTable);
long NpyDict_GetNumBuckets(const NpyDict *hashTable);
void NpyDict_SetKeyComparisonFunction(NpyDict *hashTable,
                                      int (*keycmp)(const void *key1, const void *key2));
void NpyDict_SetValueComparisonFunction(NpyDict *hashTable,
                                        int (*valuecmp)(const void *value1, const void *value2));
void NpyDict_SetHashFunction(NpyDict *hashTable,
                             unsigned long (*hashFunction)(const void *key));
void NpyDict_Rehash(NpyDict *hashTable, long numOfBuckets);
void NpyDict_SetIdealRatio(NpyDict *hashTable, float idealRatio,
                           float lowerRehashThreshold, float upperRehashThreshold);
void NpyDict_SetDeallocationFunctions(NpyDict *hashTable,
                                      void (*keyDeallocator)(void *key),
                                      void (*valueDeallocator)(void *value));
unsigned long NpyDict_StringHashFunction(const void *key);



/* flagsobject.c */
void NpyArray_UpdateFlags(NpyArray *ret, int flagmask);


#include <numpy/npy_iterators.h>


/* methods.c */
NpyArray *NpyArray_GetField(NpyArray *self, NpyArray_Descr *typed, int offset);
int NpyArray_SetField(NpyArray *self, NpyArray_Descr *dtype, int offset, NpyObject *val);
NpyArray *NpyArray_Byteswap(NpyArray *self, npy_bool inplace);
unsigned char NpyArray_EquivTypes(NpyArray_Descr *typ1, NpyArray_Descr *typ2);




/* mapping.c */
NpyArrayMapIterObject *NpyArray_MapIterNew(void);
void NpyArray_MapIterNext(NpyArrayMapIterObject *mit);
void NpyArray_MapIterReset(NpyArrayMapIterObject *mit);


/* multiarraymodule.c */
#define NpyArray_GetPriority(obj, def) PyArray_GetPriority(obj, def);       /* TODO: Needs to be callback to interface layer */

int NpyArray_MultiplyIntList(int *l1, int n);
npy_intp NpyArray_MultiplyList(npy_intp *l1, int n);
npy_intp NpyArray_OverflowMultiplyList(npy_intp *l1, int n);
void *NpyArray_GetPtr(NpyArray *obj, npy_intp *ind);
int NpyArray_CompareLists(npy_intp *l1, npy_intp *l2, int n);
int NpyArray_AsCArray(NpyArray **op, void *ptr, npy_intp *dims, int nd,
                      NpyArray_Descr* typedescr);
int NpyArray_Free(NpyArray *ap, void *ptr);
NPY_SCALARKIND NpyArray_ScalarKind(int typenum, NpyArray **arr);
int NpyArray_CanCoerceScalar(int thistype, int neededtype, NPY_SCALARKIND scalar);
NpyArray *NpyArray_InnerProduct(NpyArray *ap1, NpyArray *ap2, int typenum);


/* number.c */
#define NpyArray_GenericReduceFunction(m1, op, axis, rtype, out) \
        PyArray_GenericReduceFunction(m1, op, axis, rtype, out)


/* Already exists as a macro */
#define NpyArray_ContiguousFromAny(op, type, min_depth, max_depth)             \
        PyArray_FromAny(op, NpyArray_DescrFromType(type), min_depth,           \
        max_depth, NPY_DEFAULT, NULL)

#define NpyArray_ContiguousFromArray(op, type)                          \
    ((NpyArray*) NpyArray_FromArray(op, NpyArray_DescrFromType(type),   \
                                    NPY_DEFAULT))

#define NpyArray_EquivArrTypes(a1, a2)                                         \
        NpyArray_EquivTypes(NpyArray_DESCR(a1), NpyArray_DESCR(a2))


/*
 * API functions.
 */
npy_intp NpyArray_Size(NpyArray *op);
NpyArray *NpyArray_ArgMax(NpyArray *op, int axis, NpyArray *out);
NpyArray *NpyArray_ArgMin(NpyArray *op, int axis, NpyArray *out);
NpyArray *NpyArray_CheckAxis(NpyArray *arr, int *axis, int flags);
int NpyArray_CompareUCS4(npy_ucs4 *s1, npy_ucs4 *s2, size_t len);
int NpyArray_CompareString(char *s1, char *s2, size_t len);
int NpyArray_ElementStrides(NpyArray *arr);
npy_bool NpyArray_CheckStrides(int elsize, int nd, npy_intp numbytes,
                               npy_intp offset,
                               npy_intp *dims, npy_intp *newstrides);
NpyArray *NpyArray_FromArray(NpyArray *arr, NpyArray_Descr *newtype, int flags);
NpyArray *NpyArray_FromBinaryFile(FILE *fp, PyArray_Descr *dtype, npy_intp num);
NpyArray *NpyArray_FromBinaryString(char *data, npy_intp slen, PyArray_Descr *dtype, npy_intp num);
NpyArray *NpyArray_CheckFromArray(NpyArray *arr, PyArray_Descr *descr, int requires);
int NpyArray_ToBinaryFile(NpyArray *self, FILE *fp);

int NpyArray_MoveInto(NpyArray *dest, NpyArray *src);

NpyArray* NpyArray_Newshape(NpyArray* self, NpyArray_Dims *newdims,
                            NPY_ORDER fortran);
NpyArray* NpyArray_Squeeze(NpyArray *self);
NpyArray* NpyArray_SwapAxes(NpyArray *ap, int a1, int a2);
NpyArray* NpyArray_Transpose(NpyArray *ap, NpyArray_Dims *permute);
int NpyArray_TypestrConvert(int itemsize, int gentype);
NpyArray* NpyArray_Ravel(NpyArray *a, NPY_ORDER fortran);
NpyArray* NpyArray_Flatten(NpyArray *a, NPY_ORDER order);

NpyArray *NpyArray_CastToType(NpyArray *mp, NpyArray_Descr *at, int fortran);
NpyArray_VectorUnaryFunc *NpyArray_GetCastFunc(NpyArray_Descr *descr, int type_num);
int NpyArray_CastTo(NpyArray *out, NpyArray *mp);
int NpyArray_CastAnyTo(NpyArray *out, NpyArray *mp);
int NpyArray_CanCastSafely(int fromtype, int totype);
npy_bool NpyArray_CanCastTo(NpyArray_Descr *from, NpyArray_Descr *to);
npy_bool NpyArray_CanCastScalar(NpyTypeObject *from, NpyTypeObject *to);
int NpyArray_ValidType(int type);

NpyArray* NpyArray_TakeFrom(NpyArray *self0, NpyArray *indices0, int axis,
                            NpyArray *ret, NPY_CLIPMODE clipmode);

int NpyArray_PutTo(NpyArray *self, NpyArray* values0, NpyArray *indices0,
                   NPY_CLIPMODE clipmode);
int NpyArray_PutMask(NpyArray *self, NpyArray* values0, NpyArray* mask0);
NpyArray * NpyArray_Repeat(NpyArray *aop, NpyArray *op, int axis);
NpyArray * NpyArray_Choose(NpyArray *ip, NpyArray** mps, int n, NpyArray *ret,
                           NPY_CLIPMODE clipmode);
int NpyArray_Sort(NpyArray *op, int axis, NPY_SORTKIND which);
NpyArray * NpyArray_ArgSort(NpyArray *op, int axis, NPY_SORTKIND which);
NpyArray * NpyArray_LexSort(NpyArray** mps, int n, int axis);
NpyArray * NpyArray_SearchSorted(NpyArray *op1, NpyArray *op2, NPY_SEARCHSIDE side);

void NpyArray_InitArrFuncs(NpyArray_ArrFuncs *f);
int NpyArray_RegisterDataType(NpyArray_Descr *descr);
int NpyArray_RegisterCastFunc(NpyArray_Descr *descr, int totype,
                              NpyArray_VectorUnaryFunc *castfunc);
int NpyArray_RegisterCanCast(NpyArray_Descr *descr, int totype,
                             NPY_SCALARKIND scalar);
int NpyArray_TypeNumFromName(char *str);
int NpyArray_TypeNumFromTypeObj(void* typeobj);
NpyArray_Descr* NpyArray_UserDescrFromTypeNum(int typenum);

NpyArray *NpyArray_NewFromDescr(NpyTypeObject *subtype, 
                                NpyArray_Descr *descr, int nd,
                                npy_intp *dims, npy_intp *strides, void *data,
                                int flags, NpyObject *obj);
NpyArray *NpyArray_New(NpyTypeObject *subtype, int nd, npy_intp *dims, int type_num,
                       npy_intp *strides, void *data, int itemsize, int flags,
                       NpyObject *obj);
int NpyArray_CopyInto(NpyArray *dest, NpyArray *src);
int NpyArray_CopyAnyInto(NpyArray *dest, NpyArray *src);
int NpyArray_Resize(NpyArray *self, NpyArray_Dims *newshape, int refcheck, NPY_ORDER fortran);

npy_datetime NpyArray_DatetimeStructToDatetime(NPY_DATETIMEUNIT fr, npy_datetimestruct *d);
npy_datetime NpyArray_TimedeltaStructToTimedelta(NPY_DATETIMEUNIT fr, npy_timedeltastruct *d);
void NpyArray_DatetimeToDatetimeStruct(npy_datetime val, NPY_DATETIMEUNIT fr, npy_datetimestruct *result);
void NpyArray_TimedeltaToTimedeltaStruct(npy_timedelta val, NPY_DATETIMEUNIT fr, npy_timedeltastruct *result);


/* TODO: Check this, defined in npy_ctors.c. */
int NpyCapsule_Check(PyObject *ptr);


/*
 * Reference counting.
 */

/* These operate on core data structures, NOT interface objects. */
#define Npy_INCREF(a) {                                 \
    assert(NPY_VALID_MAGIC == (a)->magic_number);       \
    (a)->ob_refcnt = (a)->ob_refcnt;                    \
    Py_INCREF(a);   }

#define Npy_DECREF(a) {                                 \
    assert(NPY_VALID_MAGIC == (a)->magic_number);       \
    (a)->ob_refcnt = (a)->ob_refcnt;                    \
    Py_DECREF(a); }

#define Npy_XINCREF(a) {                                                \
    assert(NULL == (a) || NPY_VALID_MAGIC == (a)->magic_number);        \
    (a)->ob_refcnt = (a)->ob_refcnt;                    \
    Py_XINCREF(a); }

#define Npy_XDECREF(a) {                                                \
    assert(NULL == (a) || NPY_VALID_MAGIC == (a)->magic_number);        \
    (a)->ob_refcnt = (a)->ob_refcnt;                    \
    Py_XDECREF(a); }


/* These operate on interface objects and will be replaced with callbacks to the interface layer. */
#define Npy_Interface_INCREF(a) Py_INCREF(a)
#define Npy_Interface_DECREF(a) Py_DECREF(a)
#define Npy_Interface_XINCREF(a) Py_XINCREF(a)
#define Npy_Interface_XDECREF(a) Py_XDECREF(a)


/* These operate on the elements IN the array, not the array itself. */
/* TODO: Would love to rename these, easy to misread NpyArray_XX and Npy_XX */
#define NpyArray_REFCOUNT(a) PyArray_REFCOUNT(a)
#define NpyArray_INCREF(a) PyArray_INCREF(a)
#define NpyArray_DECREF(a) PyArray_DECREF(a)
#define NpyArray_XDECREF(a) PyArray_XDECREF(a)

#define NpyArray_XDECREF_ERR(a) PyArray_XDECREF_ERR(a)
#define NpyArray_Item_INCREF(a, descr) PyArray_Item_INCREF(a, descr)
#define NpyArray_Item_XDECREF(a, descr) PyArray_Item_XDECREF(a, descr)

#define NpyObject_New(a, b) PyObject_New(a, b)

/*
 * Object model. 
 */
#define NpyObject_Init(object, type) PyObject_Init(object, type)

/*
 * Memory
 */
#define NpyDataMem_NEW(sz) PyDataMem_NEW(sz)
#define NpyDataMem_RENEW(p, sz) PyDataMem_RENEW(p, sz)
#define NpyDataMem_FREE(p) PyDataMem_FREE(p)

#define NpyDimMem_NEW(size) PyDimMem_NEW(size)
#define NpyDimMem_RENEW(p, sz) PyDimMem_RENEW(p, sz)
#define NpyDimMem_FREE(ptr) PyDimMem_FREE(ptr)

#define NpyArray_malloc(size) PyArray_malloc(size)
#define NpyArray_free(ptr) PyArray_free(ptr)


/*
 * scalar
 */
int _typenum_fromtypeobj(NpyObject *type, int user);


/*
 * Error handling.
 */
#define NpyErr_SetString(exc, str) PyErr_SetString(exc, str)
#define NpyErr_SetNone(e) PyErr_SetNone(e)
#define NpyErr_NoMemory() PyErr_NoMemory()
#define NpyErr_Occurred() PyErr_Occurred()
#define NpyExc_ValueError PyExc_ValueError
#define NpyExc_MemoryError PyExc_MemoryError
#define NpyExc_IOError PyExc_IOError
#define NpyExc_TypeError PyExc_TypeError
#define NpyExc_IndexError PyExc_IndexError
#define NpyExc_RuntimeError PyExc_RuntimeError
#define NpyErr_Format PyErr_Format
#define NpyExc_RuntimeError PyExc_RuntimeError
#define NpyErr_Clear() PyErr_Clear()
#define NpyErr_Print() PyErr_Print()

#if PY_VERSION_HEX >= 0x02050000
#define NpyErr_WarnEx(cls, msg, stackLevel) PyErr_WarnEx(cls, msg, stackLevel) 
#else
#define NpyErr_WarnEx(obj, msg, stackLevel) PyErr_Warn(cls, msg)
#endif



/*
 * TMP
 */
#define NpyArray_View(a, b, c) ((NpyArray*) PyArray_View(a,b,c))
#define NpyArray_NewCopy(a, order) ((NpyArray*) PyArray_NewCopy(a, order))
#define NpyArray_DescrFromArray(a, dtype) PyArray_DescrFromObject((PyObject*)(a), dtype)

extern int _flat_copyinto(NpyArray *dst, NpyArray *src, NPY_ORDER order);
extern void _unaligned_strided_byte_copy(char *dst, npy_intp outstrides, char *src,
                                         npy_intp instrides, npy_intp N, int elsize);
extern void _strided_byte_swap(void *p, npy_intp stride, npy_intp n, int size);

//extern NpyArray_Descr * _array_small_type(NpyArray_Descr *chktype, NpyArray_Descr* mintype);

#endif

