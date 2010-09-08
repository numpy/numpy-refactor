#ifndef _NPY_API_H_
#define _NPY_API_H_

#include "assert.h"
#include "npy_defs.h"
#include "npy_descriptor.h"
#include "npy_iterators.h"
#include "npy_index.h"


#define NpyArray_UCS4 npy_ucs4

#define NpyDataType_FLAGCHK(dtype, flag)          \
        (((dtype)->flags & (flag)) == (flag))

#define NpyArray_DESCR_REPLACE(descr)                             \
    do {                                                          \
        NpyArray_Descr *_new_;                                    \
        _new_ = NpyArray_DescrNew(descr);                         \
        Npy_XDECREF(descr);                                      \
        descr = _new_;                                            \
    } while(0)

#define NpyArray_EquivByteorders(b1, b2)                          \
    (((b1) == (b2)) || (NpyArray_ISNBO(b1) == NpyArray_ISNBO(b2)))


#define NpyDataType_ISOBJECT(obj) NpyTypeNum_ISOBJECT(obj->type_num)
#define NpyDataType_ISSTRING(obj) NpyTypeNum_ISSTRING(obj->type_num)
#define NpyArray_Check(op) (&NpyArray_Type == (op)->nob_type)




/* Really internal to the core, but required for now by
   PyArray_TypeNumFromString */
/* TODO: Refactor and add an accessor for npy_userdescrs */
extern struct NpyArray_Descr **npy_userdescrs;





/*
 * Functions we need to convert.
 */

/* arraytypes.c.src */
/* TODO: Needs to call back to interface layer */

int NpyArray_dealloc(NpyArray *self);


/* common.c */
/* TODO: Npy_IsWriteable() need callback to interface for base of
   string, buffer */
/*
#define NpyString_Check(a) PyString_Check(a)
#define NpyObject_AsWriteBuffer(a, b, c) PyObject_AsWriteBuffer(a, b, c)
*/

int Npy_IsAligned(NpyArray *ap);
npy_bool Npy_IsWriteable(NpyArray *ap);
NpyArray_Descr *
NpyArray_SmallType(NpyArray_Descr *chktype, NpyArray_Descr *mintype);
char *
NpyArray_Index2Ptr(NpyArray *self, npy_intp i);


/* npy_convert.c */
NpyArray *
NpyArray_View(NpyArray *self, NpyArray_Descr *type, void *pytype);
int
NpyArray_SetDescr(NpyArray *self, NpyArray_Descr *newtype);
NpyArray *
NpyArray_NewCopy(NpyArray *m1, NPY_ORDER fortran);


/* ctors.c */
NDARRAY_API size_t npy_array_fill_strides(npy_intp *strides, npy_intp *dims, int nd,
                              size_t itemsize, int inflag, int *objflags);

NpyArray * NpyArray_FromTextFile(FILE *fp, NpyArray_Descr *dtype,
                                 npy_intp num, char *sep);
NpyArray * NpyArray_FromString(char *data, npy_intp slen, NpyArray_Descr *dtype,
                               npy_intp num, char *sep);

NpyArray_Descr *
NpyArray_DescrFromArray(NpyArray* array, NpyArray_Descr* mintype);

NDARRAY_API void
npy_byte_swap_vector(void *p, npy_intp n, int size);




/* flagsobject.c */
void NpyArray_UpdateFlags(NpyArray *ret, int flagmask);



/* methods.c */
NpyArray *NpyArray_GetField(NpyArray *self, NpyArray_Descr *typed, int offset);
int NpyArray_SetField(NpyArray *self, NpyArray_Descr *dtype, int offset,
                      NpyArray *val);
NpyArray *NpyArray_Byteswap(NpyArray *self, npy_bool inplace);
unsigned char NpyArray_EquivTypes(NpyArray_Descr *typ1, NpyArray_Descr *typ2);




/* mapping.c */
NpyArrayMapIterObject *NpyArray_MapIterNew(NpyIndex* indexes, int n);
int NpyArray_MapIterBind(NpyArrayMapIterObject *mit, NpyArray *arr,
                         NpyArray* true_array);
void NpyArray_MapIterNext(NpyArrayMapIterObject *mit);
void NpyArray_MapIterReset(NpyArrayMapIterObject *mit);
NpyArray * NpyArray_GetMap(NpyArrayMapIterObject *mit);
int NpyArray_SetMap(NpyArrayMapIterObject *mit, NpyArray *arr);
NpyArray * NpyArray_ArrayItem(NpyArray *self, npy_intp i);
NpyArray * NpyArray_IndexSimple(NpyArray* self, NpyIndex* indexes, int n);
int NpyArray_IndexFancyAssign(NpyArray *self, NpyIndex *indexes, int n,
                              NpyArray *value);
NpyArray * NpyArray_Subscript(NpyArray *self, NpyIndex *indexes, int n);
int NpyArray_SubscriptAssign(NpyArray *self, NpyIndex *indexes, int n,
                             NpyArray *value);


/* multiarraymodule.c */
int NpyArray_MultiplyIntList(int *l1, int n);
npy_intp NpyArray_OverflowMultiplyList(npy_intp *l1, int n);
void *NpyArray_GetPtr(NpyArray *obj, npy_intp *ind);
int NpyArray_CompareLists(npy_intp *l1, npy_intp *l2, int n);
int NpyArray_AsCArray(NpyArray **op, void *ptr, npy_intp *dims, int nd,
                      NpyArray_Descr* typedescr);
int NpyArray_Free(NpyArray *ap, void *ptr);
NPY_SCALARKIND NpyArray_ScalarKind(int typenum, NpyArray **arr);
int NpyArray_CanCoerceScalar(int thistype, int neededtype,
                             NPY_SCALARKIND scalar);
NpyArray *NpyArray_InnerProduct(NpyArray *ap1, NpyArray *ap2, int typenum);
NpyArray *NpyArray_MatrixProduct(NpyArray *ap1, NpyArray *ap2, int typenum);
NpyArray *NpyArray_CopyAndTranspose(NpyArray *arr);
NpyArray *NpyArray_Correlate2(NpyArray *ap1, NpyArray *ap2,
                              int typenum, int mode);
NpyArray *NpyArray_Correlate(NpyArray *ap1, NpyArray *ap2,
                             int typenum, int mode);
unsigned char NpyArray_EquivTypenums(int typenum1, int typenum2);
int NpyArray_GetEndianness(void);



/* number.c */
#define NpyArray_GenericReduceFunction(m1, op, axis, rtype, out) \
        PyArray_GenericReduceFunction(m1, op, axis, rtype, out)


/* refcount.c */
void
NpyArray_Item_INCREF(char *data, NpyArray_Descr *descr);
void
NpyArray_Item_XDECREF(char *data, NpyArray_Descr *descr);
int
NpyArray_INCREF(NpyArray *arr);
int
NpyArray_XDECREF(NpyArray *arr);


#define NpyArray_ContiguousFromArray(op, type)                  \
    NpyArray_FromArray(op, NpyArray_DescrFromType(type),        \
                       NPY_DEFAULT)

#define NpyArray_EquivArrTypes(a1, a2)                                   \
        NpyArray_EquivTypes(NpyArray_DESCR(a1), NpyArray_DESCR(a2))



/* getset.c */
int NpyArray_SetShape(NpyArray *self, NpyArray_Dims *newdims);
int NpyArray_SetStrides(NpyArray *self, NpyArray_Dims *newstrides);


/*
 * API functions.
 */
npy_intp NpyArray_Size(NpyArray *op);
NpyArray *NpyArray_CheckAxis(NpyArray *arr, int *axis, int flags);
int NpyArray_CompareUCS4(npy_ucs4 *s1, npy_ucs4 *s2, size_t len);
int NpyArray_CompareString(char *s1, char *s2, size_t len);
int NpyArray_ElementStrides(NpyArray *arr);
npy_bool NpyArray_CheckStrides(int elsize, int nd, npy_intp numbytes,
                               npy_intp offset,
                               npy_intp *dims, npy_intp *newstrides);
NpyArray *NpyArray_FromArray(NpyArray *arr, NpyArray_Descr *newtype, int flags);
NpyArray *NpyArray_FromBinaryFile(FILE *fp, NpyArray_Descr *dtype, npy_intp num);
NpyArray *NpyArray_FromBinaryString(char *data, npy_intp slen,
                                    NpyArray_Descr *dtype, npy_intp num);
NpyArray *NpyArray_CheckFromArray(NpyArray *arr, NpyArray_Descr *descr,
                                  int requires);
int NpyArray_ToBinaryFile(NpyArray *self, FILE *fp);

int NpyArray_MoveInto(NpyArray *dest, NpyArray *src);

NpyArray* NpyArray_Newshape(NpyArray *self, NpyArray_Dims *newdims,
                            NPY_ORDER fortran);
NpyArray* NpyArray_Squeeze(NpyArray *self);
NpyArray* NpyArray_SwapAxes(NpyArray *ap, int a1, int a2);
NpyArray* NpyArray_Transpose(NpyArray *ap, NpyArray_Dims *permute);
int NpyArray_TypestrConvert(int itemsize, int gentype);
NpyArray* NpyArray_Ravel(NpyArray *a, NPY_ORDER fortran);
NpyArray* NpyArray_Flatten(NpyArray *a, NPY_ORDER order);

NpyArray *NpyArray_CastToType(NpyArray *mp, NpyArray_Descr *at, int fortran);
NpyArray_VectorUnaryFunc *NpyArray_GetCastFunc(NpyArray_Descr *descr,
                                               int type_num);
int NpyArray_CastTo(NpyArray *out, NpyArray *mp);
int NpyArray_CastAnyTo(NpyArray *out, NpyArray *mp);
int NpyArray_CanCastSafely(int fromtype, int totype);
npy_bool NpyArray_CanCastTo(NpyArray_Descr *from, NpyArray_Descr *to);
npy_bool NpyArray_CanCastScalar(NpyTypeObject *from, NpyTypeObject *to);
int NpyArray_ValidType(int type);
struct NpyArray_Descr *NpyArray_DescrFromType(int type);

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
NpyArray * NpyArray_SearchSorted(NpyArray *op1, NpyArray *op2,
                                 NPY_SEARCHSIDE side);
int NpyArray_NonZero(NpyArray* self, NpyArray** index_arrays, void* obj);

void NpyArray_InitArrFuncs(NpyArray_ArrFuncs *f);
int NpyArray_RegisterDataType(NpyArray_Descr *descr);
int NpyArray_RegisterCastFunc(NpyArray_Descr *descr, int totype,
                              NpyArray_VectorUnaryFunc *castfunc);
int NpyArray_RegisterCanCast(NpyArray_Descr *descr, int totype,
                             NPY_SCALARKIND scalar);
int NpyArray_TypeNumFromName(char *str);
NpyArray_Descr* NpyArray_UserDescrFromTypeNum(int typenum);

NpyArray *NpyArray_NewFromDescr(NpyArray_Descr *descr, int nd,
                                npy_intp *dims, npy_intp *strides, void *data,
                                int flags, int ensureArray, void *subtype,
                                void *interfaceData);
NpyArray *NpyArray_New(void *subtype, int nd, npy_intp *dims, int type_num,
                       npy_intp *strides, void *data, int itemsize, int flags,
                       void *obj);
NpyArray *NpyArray_Alloc(NpyArray_Descr *descr, int nd, npy_intp* dims,
                         npy_bool is_fortran, void *interfaceData);
NpyArray * NpyArray_NewView(NpyArray_Descr *descr, int nd, npy_intp* dims,
                            npy_intp *strides,
                            NpyArray *array, npy_intp offset,
                            npy_bool ensure_array);
int NpyArray_CopyInto(NpyArray *dest, NpyArray *src);
int NpyArray_CopyAnyInto(NpyArray *dest, NpyArray *src);
int NpyArray_Resize(NpyArray *self, NpyArray_Dims *newshape, int refcheck,
                    NPY_ORDER fortran);

npy_datetime NpyArray_DatetimeStructToDatetime(NPY_DATETIMEUNIT fr,
                                               npy_datetimestruct *d);
npy_datetime NpyArray_TimedeltaStructToTimedelta(NPY_DATETIMEUNIT fr,
                                                 npy_timedeltastruct *d);
void NpyArray_DatetimeToDatetimeStruct(npy_datetime val, NPY_DATETIMEUNIT fr,
                                       npy_datetimestruct *result);
void NpyArray_TimedeltaToTimedeltaStruct(npy_timedelta val, NPY_DATETIMEUNIT fr,
                                         npy_timedeltastruct *result);

int
NpyArray_GetNumusertypes(void);



/*
 * Reference counting.
 */

/* TODO: This looks wrong. */
#define NpyArray_XDECREF_ERR(obj)                                         \
        if (obj && (NpyArray_FLAGS(obj) & NPY_UPDATEIFCOPY)) {            \
            NpyArray_FLAGS(NpyArray_BASE_ARRAY(obj)) |= NPY_WRITEABLE;    \
            NpyArray_FLAGS(obj) &= ~NPY_UPDATEIFCOPY;                     \
        }                                                                 \
        Npy_XDECREF(obj)

/*
 * Object model.
 */

/*
 * Memory
 */
#define NpyDataMem_NEW(sz) malloc(sz)
#define NpyDataMem_RENEW(p, sz) realloc(p, sz)
#define NpyDataMem_FREE(p) free(p)

#define NpyDimMem_NEW(size) ((npy_intp *)malloc(size*sizeof(npy_intp)))
#define NpyDimMem_RENEW(p, sz) ((npy_intp *)realloc(p, sz*sizeof(npy_intp)))
#define NpyDimMem_FREE(ptr) free(ptr)

#define NpyArray_malloc(size) malloc(size)
#define NpyArray_free(ptr) free(ptr)


/*
 * Exception handling
 */
enum npyexc_type {
    NpyExc_MemoryError,
    NpyExc_IOError,
    NpyExc_ValueError,
    NpyExc_TypeError,
    NpyExc_IndexError,
    NpyExc_RuntimeError,
    NpyExc_AttributeError,
    NpyExc_ComplexWarning,
};

typedef void (*npy_tp_error_set)(enum npyexc_type, const char *);
typedef int (*npy_tp_error_occurred)(void);
typedef void (*npy_tp_error_clear)(void);

/* these functions are set in npy_initlib */
extern npy_tp_error_set NpyErr_SetString;
extern npy_tp_error_occurred NpyErr_Occurred;
extern npy_tp_error_clear NpyErr_Clear;

#define NpyErr_MEMORY  NpyErr_SetString(NpyExc_MemoryError, "memory error")


typedef int (*npy_tp_cmp_priority)(void *, void *);

extern npy_tp_cmp_priority Npy_CmpPriority;


/*
 * Interface-provided reference management.  Note that even though these
 * mirror the Python routines they are slightly different because they also
 * work w/ garbage collected systems. Primarily, INCREF returns a possibly
 * different handle. This is the typical case and the second argument will
 * be NULL. When these are called from Npy_INCREF or Npy_DECREF and the
 * core object refcnt is going 0->1 or 1->0 the second argument is a pointer
 * to the nob_interface field.  This allows the interface routine to change
 * the interface pointer.  This is done instead of using the return value
 * to ensure that the switch is atomic.
 */
typedef void *(*npy_interface_incref)(void *, void **);
typedef void *(*npy_interface_decref)(void *, void **);

/* Do not call directly, use macros below because interface does not have
   to provide these. */
/* Not defined on Windows because __declspec(dllimport) is needed for any
   DLL's importing them. */
/* TODO: Clean up handling of exported/imported variables on windows. */
#if !defined(_WIN32)
extern npy_interface_incref _NpyInterface_Incref;
extern npy_interface_decref _NpyInterface_Decref;
#endif


#define NpyInterface_INCREF(ptr) (NULL != _NpyInterface_Incref ? \
                                  _NpyInterface_Incref(ptr, NULL) : NULL)
#define NpyInterface_DECREF(ptr) (NULL != _NpyInterface_Decref ? \
                                  _NpyInterface_Decref(ptr, NULL) : NULL)
#define NpyInterface_CLEAR(ptr) \
    do {                                    \
        void *tmp = (void *)(ptr);          \
        (ptr) = NULL;                       \
        NpyInterface_DECREF(tmp);           \
    } while(0);



/* Interface wrapper-generators.  These allows the interface the option of
   wrapping each core object with another structure, such as a PyObject
   derivative.
*/

typedef int (*npy_interface_array_new_wrapper)(
        NpyArray *newArray, int ensureArray,
        int customStrides, void *subtype,
        void *interfaceData, void **interfaceRet);
typedef int (*npy_interface_iter_new_wrapper)(
        NpyArrayIterObject *iter, void **interfaceRet);
typedef int (*npy_interface_multi_iter_new_wrapper)(
        NpyArrayMultiIterObject *iter,
        void **interfaceRet);
typedef int (*npy_interface_neighbor_iter_new_wrapper)(
        NpyArrayNeighborhoodIterObject *iter,
        void **interfaceRet);
typedef int (*npy_interface_descr_new_from_type)(
        int type, struct NpyArray_Descr *descr,
        void **interfaceRet);
typedef int (*npy_interface_descr_new_from_wrapper)(
        void *base, struct NpyArray_Descr *descr,
        void **interfaceRet);

struct NpyInterface_WrapperFuncs {
    npy_interface_array_new_wrapper array_new_wrapper;
    npy_interface_iter_new_wrapper iter_new_wrapper;
    npy_interface_multi_iter_new_wrapper multi_iter_new_wrapper;
    npy_interface_neighbor_iter_new_wrapper neighbor_iter_new_wrapper;
    npy_interface_descr_new_from_type descr_new_from_type;
    npy_interface_descr_new_from_wrapper descr_new_from_wrapper;
};



extern void npy_initlib(struct NpyArray_FunctionDefs *functionDefs,
                        struct NpyInterface_WrapperFuncs *wrapperFuncs,
                        npy_tp_error_set error_set,
                        npy_tp_error_occurred error_occurred,
                        npy_tp_error_clear error_clear,
                        npy_tp_cmp_priority cmp_priority,
                        npy_interface_incref incref,
                        npy_interface_decref decref);


/*
 * TMP
 */
NDARRAY_API extern int _flat_copyinto(NpyArray *dst, NpyArray *src, NPY_ORDER order);
extern void _unaligned_strided_byte_copy(char *dst, npy_intp outstrides,
                                         char *src, npy_intp instrides,
                                         npy_intp N, int elsize);
extern void _strided_byte_swap(void *p, npy_intp stride, npy_intp n, int size);


#endif
