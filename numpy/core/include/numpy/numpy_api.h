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
typedef PyArray_DateTimeInfo NpyArray_DateTimeInfo;
typedef PyArray_CastFuncsItem NpyArray_CastFuncsItem;
typedef PyArray_DotFunc NpyArray_DotFunc;

#define NpyTypeObject PyTypeObject
#define NpyArrayDescr_Type PyArrayDescr_Type

#define NpyArray_UCS4 npy_ucs4

#define NpyDataType_FLAGCHK(dtype, flag)                                   \
        (((dtype)->flags & (flag)) == (flag))

#define NpyArray_DESCR_REPLACE(descr) PyArray_DESCR_REPLACE(descr)
#define NpyArray_EquivByteorders(b1, b2) PyArray_EquivByteorders(b1, b2)

#define NpyDataType_ISSTRING(obj) PyDataType_ISSTRING(obj)
#define NpyDataType_ISOBJECT(obj) PyDataType_ISOBJECT(obj)
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


/* Really internal to the core, but required for now by PyArray_TypeNumFromString */
/* TODO: Refactor and add an accessor for npy_userdescrs */
extern NpyArray_Descr **npy_userdescrs;


/*
 * External interface-provided functions 
 */
void NpyInterface_Incref(void *);
void NpyInterface_Decref(void *);
int NpyInterface_ArrayNewWrapper(NpyArray *newArray, int ensureArray, int customStrides, 
                                 void *subtype, void *interfaceData, void **interfaceRet);
int NpyInterface_IterNewWrapper(NpyArrayIterObject *iter, void **interfaceRet);
int NpyInterface_MultiIterNewWrapper(NpyArrayMultiIterObject *iter, void **interfaceRet);
int NpyInterface_NeighborhoodIterNewWrapper(NpyArrayNeighborhoodIterObject *iter, void **interfaceRet);
int NpyInterface_MapIterNewWrapper(NpyArrayMapIterObject *iter, void **interfaceRet);



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
NpyArray_Descr *
NpyArray_SmallType(NpyArray_Descr *chktype, NpyArray_Descr *mintype);
char *
NpyArray_Index2Ptr(NpyArray *self, npy_intp i);


/* npy_convert.c */
NpyArray *
NpyArray_View(NpyArray *self, PyArray_Descr *type, void *pytype);
int 
NpyArray_SetDescr(NpyArray *self, PyArray_Descr *newtype);
NpyArray *
NpyArray_NewCopy(NpyArray *m1, NPY_ORDER fortran);


/* ctors.c */
size_t _array_fill_strides(npy_intp *strides, npy_intp *dims, int nd, size_t itemsize,
                           int inflag, int *objflags);
NpyArray_Descr *
NpyArray_DescrFromArray(NpyArray* array, NpyArray_Descr* mintype);

NPY_NO_EXPORT void
byte_swap_vector(void *p, npy_intp n, int size);



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
int NpyArray_SetField(NpyArray *self, NpyArray_Descr *dtype, int offset, NpyArray *val);
NpyArray *NpyArray_Byteswap(NpyArray *self, npy_bool inplace);
unsigned char NpyArray_EquivTypes(NpyArray_Descr *typ1, NpyArray_Descr *typ2);




/* mapping.c */
NpyArrayMapIterObject *NpyArray_MapIterNew(void);
void NpyArray_MapIterNext(NpyArrayMapIterObject *mit);
void NpyArray_MapIterReset(NpyArrayMapIterObject *mit);
NpyArray * NpyArray_ArrayItem(NpyArray *self, npy_intp i);


/* multiarraymodule.c */
#define NpyArray_GetPriority(obj, def) PyArray_GetPriority(obj, def);       /* TODO: Needs to be callback to interface layer */

int NpyArray_MultiplyIntList(int *l1, int n);
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

#define NpyArray_ContiguousFromArray(op, type)                  \
    NpyArray_FromArray(op, NpyArray_DescrFromType(type),        \
                       NPY_DEFAULT)

#define NpyArray_EquivArrTypes(a1, a2)                                         \
        NpyArray_EquivTypes(NpyArray_DESCR(a1), NpyArray_DESCR(a2))


/*
 * API functions.
 */
npy_intp NpyArray_Size(NpyArray *op);
NpyArray *NpyArray_ArgMax(NpyArray *op, int axis, NpyArray *out);
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

NpyArray *NpyArray_NewFromDescr(NpyArray_Descr *descr, int nd,
                                npy_intp *dims, npy_intp *strides, void *data,
                                int flags, int ensureArray, void *subtype, 
                                void *interfaceData);
NpyArray *NpyArray_New(void *subtype, int nd, npy_intp *dims, int type_num,
                       npy_intp *strides, void *data, int itemsize, int flags,
                       void *obj);
int NpyArray_CopyInto(NpyArray *dest, NpyArray *src);
int NpyArray_CopyAnyInto(NpyArray *dest, NpyArray *src);
int NpyArray_Resize(NpyArray *self, NpyArray_Dims *newshape, int refcheck, NPY_ORDER fortran);

npy_datetime NpyArray_DatetimeStructToDatetime(NPY_DATETIMEUNIT fr, npy_datetimestruct *d);
npy_datetime NpyArray_TimedeltaStructToTimedelta(NPY_DATETIMEUNIT fr, npy_timedeltastruct *d);
void NpyArray_DatetimeToDatetimeStruct(npy_datetime val, NPY_DATETIMEUNIT fr, npy_datetimestruct *result);
void NpyArray_TimedeltaToTimedeltaStruct(npy_timedelta val, NPY_DATETIMEUNIT fr, npy_timedeltastruct *result);



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
    if (NULL != (a)) (a)->ob_refcnt = (a)->ob_refcnt;                   \
    Py_XINCREF(a); }

#define Npy_XDECREF(a) {                                                \
    assert(NULL == (a) || NPY_VALID_MAGIC == (a)->magic_number);        \
    if (NULL != (a)) (a)->ob_refcnt = (a)->ob_refcnt;                   \
    Py_XDECREF(a); }


/* These operate on interface objects and will be replaced with callbacks to the interface layer. */
#define Npy_Interface_INCREF(a) Py_INCREF(a)
#define Npy_Interface_DECREF(a) Py_DECREF(a)
#define Npy_Interface_XINCREF(a) Py_XINCREF(a)
#define Npy_Interface_XDECREF(a) Py_XDECREF(a)


/* These operate on the elements IN the array, not the array itself. */
/* TODO: Would love to rename these, easy to misread NpyArray_XX and Npy_XX */
#define NpyArray_REFCOUNT(a) PyArray_REFCOUNT(a)
#define NpyArray_INCREF(a) PyArray_INCREF(Npy_INTERFACE(a))
#define NpyArray_DECREF(a) PyArray_DECREF(Npy_INTERFACE(a))
#define NpyArray_XDECREF(a) PyArray_XDECREF( a ? Npy_INTERFACE(a) : NULL)

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
extern int _flat_copyinto(NpyArray *dst, NpyArray *src, NPY_ORDER order);
extern void _unaligned_strided_byte_copy(char *dst, npy_intp outstrides, char *src,
                                         npy_intp instrides, npy_intp N, int elsize);
extern void _strided_byte_swap(void *p, npy_intp stride, npy_intp n, int size);

//extern NpyArray_Descr * _array_small_type(NpyArray_Descr *chktype, NpyArray_Descr* mintype);

#endif

