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
#define NpyDataType_TYPE_NUM(obj) ((obj)->type_num)

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

NDARRAY_API int NpyArray_dealloc(NpyArray *self);


/* common.c */
/* TODO: Npy_IsWriteable() need callback to interface for base of
   string, buffer */
/*
#define NpyString_Check(a) PyString_Check(a)
#define NpyObject_AsWriteBuffer(a, b, c) PyObject_AsWriteBuffer(a, b, c)
*/

NDARRAY_API int Npy_IsAligned(NpyArray *ap);
NDARRAY_API npy_bool Npy_IsWriteable(NpyArray *ap);
NDARRAY_API char * NpyArray_Index2Ptr(NpyArray *self, npy_intp i);


/* npy_convert.c */
NDARRAY_API NpyArray *
NpyArray_View(NpyArray *self, NpyArray_Descr *type, void *pytype);
NDARRAY_API int
NpyArray_SetDescr(NpyArray *self, NpyArray_Descr *newtype);
NDARRAY_API NpyArray *
NpyArray_NewCopy(NpyArray *m1, NPY_ORDER fortran);


/* ctors.c */
NDARRAY_API size_t npy_array_fill_strides(npy_intp *strides, npy_intp *dims,
                                          int nd, size_t itemsize, int inflag,
                                          int *objflags);

NDARRAY_API NpyArray * NpyArray_FromTextFile(FILE *fp, NpyArray_Descr *dtype,
                                             npy_intp num, char *sep);
NDARRAY_API NpyArray * NpyArray_FromString(char *data, npy_intp slen,
                                           NpyArray_Descr *dtype,
                                           npy_intp num, char *sep);

NDARRAY_API void
npy_byte_swap_vector(void *p, npy_intp n, int size);




/* flagsobject.c */
NDARRAY_API void NpyArray_UpdateFlags(NpyArray *ret, int flagmask);



/* methods.c */
NDARRAY_API NpyArray *
NpyArray_GetField(NpyArray *self, NpyArray_Descr *typed, int offset);
NDARRAY_API int
NpyArray_SetField(NpyArray *self, NpyArray_Descr *dtype, int offset,
                  NpyArray *val);
NDARRAY_API NpyArray *
NpyArray_Byteswap(NpyArray *self, npy_bool inplace);
NDARRAY_API unsigned char
NpyArray_EquivTypes(NpyArray_Descr *typ1, NpyArray_Descr *typ2);




/* mapping.c */
NDARRAY_API NpyArrayMapIterObject *NpyArray_MapIterNew(NpyIndex* indexes, int n);
NDARRAY_API int NpyArray_MapIterBind(NpyArrayMapIterObject *mit, NpyArray *arr,
                                     NpyArray* true_array);
NDARRAY_API void NpyArray_MapIterNext(NpyArrayMapIterObject *mit);
NDARRAY_API void NpyArray_MapIterReset(NpyArrayMapIterObject *mit);
NDARRAY_API NpyArray * NpyArray_GetMap(NpyArrayMapIterObject *mit);
NDARRAY_API int NpyArray_SetMap(NpyArrayMapIterObject *mit, NpyArray *arr);
NDARRAY_API NpyArray * NpyArray_ArrayItem(NpyArray *self, npy_intp i);
NDARRAY_API NpyArray * NpyArray_IndexSimple(NpyArray* self, NpyIndex* indexes,
                                            int n);
NDARRAY_API int NpyArray_IndexFancyAssign(NpyArray *self, NpyIndex *indexes,
                                          int n, NpyArray *value);
NDARRAY_API NpyArray *
NpyArray_Subscript(NpyArray *self, NpyIndex *indexes, int n);
NDARRAY_API int
NpyArray_SubscriptAssign(NpyArray *self, NpyIndex *indexes, int n,
                         NpyArray *value);


/* multiarraymodule.c */
NDARRAY_API int NpyArray_MultiplyIntList(int *l1, int n);
NDARRAY_API npy_intp NpyArray_OverflowMultiplyList(npy_intp *l1, int n);
NDARRAY_API void *NpyArray_GetPtr(NpyArray *obj, npy_intp *ind);
NDARRAY_API int NpyArray_CompareLists(npy_intp *l1, npy_intp *l2, int n);
NDARRAY_API int NpyArray_AsCArray(NpyArray **op, void *ptr, npy_intp *dims,
                                  int nd, NpyArray_Descr* typedescr);
NDARRAY_API int NpyArray_Free(NpyArray *ap, void *ptr);
NDARRAY_API NPY_SCALARKIND NpyArray_ScalarKind(int typenum, NpyArray **arr);
NDARRAY_API int NpyArray_CanCoerceScalar(int thistype, int neededtype,
                                         NPY_SCALARKIND scalar);
NDARRAY_API NpyArray *NpyArray_InnerProduct(NpyArray *ap1, NpyArray *ap2,
                                            int typenum);
NDARRAY_API NpyArray *NpyArray_MatrixProduct(NpyArray *ap1, NpyArray *ap2,
                                             int typenum);
NDARRAY_API NpyArray *NpyArray_CopyAndTranspose(NpyArray *arr);
NDARRAY_API NpyArray *NpyArray_Correlate2(NpyArray *ap1, NpyArray *ap2,
                                          int typenum, int mode);
NDARRAY_API NpyArray *NpyArray_Correlate(NpyArray *ap1, NpyArray *ap2,
                                         int typenum, int mode);
NDARRAY_API unsigned char NpyArray_EquivTypenums(int typenum1, int typenum2);
NDARRAY_API int NpyArray_GetEndianness(void);



/* number.c */
#define NpyArray_GenericReduceFunction(m1, op, axis, rtype, out) \
        PyArray_GenericReduceFunction(m1, op, axis, rtype, out)
NDARRAY_API int NpyArray_Bool(NpyArray* arr);


/* refcount.c */
NDARRAY_API void
NpyArray_Item_INCREF(char *data, NpyArray_Descr *descr);
NDARRAY_API void
NpyArray_Item_XDECREF(char *data, NpyArray_Descr *descr);
NDARRAY_API int
NpyArray_INCREF(NpyArray *arr);
NDARRAY_API int
NpyArray_XDECREF(NpyArray *arr);


#define NpyArray_ContiguousFromArray(op, type)                  \
    NpyArray_FromArray(op, NpyArray_DescrFromType(type),        \
                       NPY_DEFAULT)

#define NpyArray_EquivArrTypes(a1, a2)                                   \
        NpyArray_EquivTypes(NpyArray_DESCR(a1), NpyArray_DESCR(a2))



/* getset.c */
NDARRAY_API int NpyArray_SetShape(NpyArray *self, NpyArray_Dims *newdims);
NDARRAY_API int NpyArray_SetStrides(NpyArray *self, NpyArray_Dims *newstrides);
NDARRAY_API NpyArray *NpyArray_GetReal(NpyArray *self);
NDARRAY_API NpyArray *NpyArray_GetImag(NpyArray *self);


/*
 * API functions.
 */
NDARRAY_API npy_intp NpyArray_Size(NpyArray *op);
NDARRAY_API NpyArray *NpyArray_CheckAxis(NpyArray *arr, int *axis, int flags);
NDARRAY_API int NpyArray_CompareUCS4(npy_ucs4 *s1, npy_ucs4 *s2, size_t len);
NDARRAY_API int NpyArray_CompareString(char *s1, char *s2, size_t len);
NDARRAY_API int NpyArray_ElementStrides(NpyArray *arr);
NDARRAY_API npy_bool NpyArray_CheckStrides(int elsize, int nd, npy_intp numbytes,
                                           npy_intp offset,
                                           npy_intp *dims, npy_intp *newstrides);
NDARRAY_API NpyArray *NpyArray_FromArray(NpyArray *arr,
                                         NpyArray_Descr *newtype, int flags);
NDARRAY_API NpyArray *NpyArray_FromBinaryFile(FILE *fp, NpyArray_Descr *dtype,
                                              npy_intp num);
NDARRAY_API NpyArray *NpyArray_FromBinaryString(char *data, npy_intp slen,
                                                NpyArray_Descr *dtype,
                                                npy_intp num);
NDARRAY_API NpyArray *NpyArray_CheckFromArray(NpyArray *arr,
                                              NpyArray_Descr *descr,
                                              int requires);
NDARRAY_API int NpyArray_ToBinaryFile(NpyArray *self, FILE *fp);
NDARRAY_API int NpyArray_FillWithObject(NpyArray* arr, void* object);
NDARRAY_API int NpyArray_FillWithScalar(NpyArray* arr, NpyArray* zero_d_array);


NDARRAY_API int NpyArray_MoveInto(NpyArray *dest, NpyArray *src);

NDARRAY_API NpyArray* NpyArray_Newshape(NpyArray *self, NpyArray_Dims *newdims,
                                        NPY_ORDER fortran);
NDARRAY_API NpyArray* NpyArray_Squeeze(NpyArray *self);
NDARRAY_API NpyArray* NpyArray_SwapAxes(NpyArray *ap, int a1, int a2);
NDARRAY_API NpyArray* NpyArray_Transpose(NpyArray *ap, NpyArray_Dims *permute);
NDARRAY_API int NpyArray_TypestrConvert(int itemsize, int gentype);
NDARRAY_API NpyArray* NpyArray_Ravel(NpyArray *a, NPY_ORDER fortran);
NDARRAY_API NpyArray* NpyArray_Flatten(NpyArray *a, NPY_ORDER order);
NDARRAY_API NpyArray* NpyArray_FlatView(NpyArray *a);

NDARRAY_API NpyArray *NpyArray_CastToType(NpyArray *mp, NpyArray_Descr *at, 
                                          int fortran);
NDARRAY_API NpyArray_VectorUnaryFunc *NpyArray_GetCastFunc(NpyArray_Descr *descr,
                                                           int type_num);
NDARRAY_API int NpyArray_CastTo(NpyArray *out, NpyArray *mp);
NDARRAY_API int NpyArray_CastAnyTo(NpyArray *out, NpyArray *mp);
NDARRAY_API int NpyArray_CanCastSafely(int fromtype, int totype);
NDARRAY_API npy_bool NpyArray_CanCastTo(NpyArray_Descr *from, NpyArray_Descr *to);
NDARRAY_API int NpyArray_ValidType(int type);
NDARRAY_API struct NpyArray_Descr *NpyArray_DescrFromType(int type);

NDARRAY_API NpyArray* NpyArray_TakeFrom(NpyArray *self0, NpyArray *indices0, int axis,
                                        NpyArray *ret, NPY_CLIPMODE clipmode);

NDARRAY_API int NpyArray_PutTo(NpyArray *self, NpyArray* values0, NpyArray *indices0,
                               NPY_CLIPMODE clipmode);
NDARRAY_API int NpyArray_PutMask(NpyArray *self, NpyArray* values0, NpyArray* mask0);
NDARRAY_API NpyArray * NpyArray_Repeat(NpyArray *aop, NpyArray *op, int axis);
NDARRAY_API NpyArray * NpyArray_Choose(NpyArray *ip, NpyArray** mps, int n, NpyArray *ret,
                                       NPY_CLIPMODE clipmode);
NDARRAY_API int NpyArray_Sort(NpyArray *op, int axis, NPY_SORTKIND which);
NDARRAY_API NpyArray * NpyArray_ArgSort(NpyArray *op, int axis, NPY_SORTKIND which);
NDARRAY_API NpyArray * NpyArray_LexSort(NpyArray** mps, int n, int axis);
NDARRAY_API NpyArray * NpyArray_SearchSorted(NpyArray *op1, NpyArray *op2,
                                             NPY_SEARCHSIDE side);
NDARRAY_API int NpyArray_NonZero(NpyArray* self, NpyArray** index_arrays, void* obj);

NDARRAY_API void NpyArray_InitArrFuncs(NpyArray_ArrFuncs *f);
NDARRAY_API int NpyArray_RegisterDataType(NpyArray_Descr *descr);
NDARRAY_API int NpyArray_RegisterCastFunc(NpyArray_Descr *descr, int totype,
                              NpyArray_VectorUnaryFunc *castfunc);
NDARRAY_API int NpyArray_RegisterCanCast(NpyArray_Descr *descr, int totype,
                                         NPY_SCALARKIND scalar);
NDARRAY_API NpyArray_Descr* NpyArray_UserDescrFromTypeNum(int typenum);

NDARRAY_API NpyArray *
NpyArray_NewFromDescr(NpyArray_Descr *descr, int nd,
                      npy_intp *dims, npy_intp *strides, void *data,
                      int flags, int ensureArray, void *subtype,
                      void *interfaceData);
NDARRAY_API NpyArray *
NpyArray_New(void *subtype, int nd, npy_intp *dims, int type_num,
             npy_intp *strides, void *data, int itemsize, int flags,
                       void *obj);
NDARRAY_API NpyArray *
NpyArray_Alloc(NpyArray_Descr *descr, int nd, npy_intp* dims,
               npy_bool is_fortran, void *interfaceData);
NDARRAY_API NpyArray *
NpyArray_NewView(NpyArray_Descr *descr, int nd, npy_intp* dims,
                 npy_intp *strides,
                 NpyArray *array, npy_intp offset,
                 npy_bool ensure_array);
NDARRAY_API int NpyArray_CopyInto(NpyArray *dest, NpyArray *src);
NDARRAY_API int NpyArray_CopyAnyInto(NpyArray *dest, NpyArray *src);
NDARRAY_API int
NpyArray_Resize(NpyArray *self, NpyArray_Dims *newshape, int refcheck,
                NPY_ORDER fortran);

NDARRAY_API npy_datetime
NpyArray_DatetimeStructToDatetime(NPY_DATETIMEUNIT fr,
                                  npy_datetimestruct *d);
NDARRAY_API npy_datetime
NpyArray_TimedeltaStructToTimedelta(NPY_DATETIMEUNIT fr,
                                    npy_timedeltastruct *d);
NDARRAY_API void
NpyArray_DatetimeToDatetimeStruct(npy_datetime val, NPY_DATETIMEUNIT fr,
                                  npy_datetimestruct *result);
NDARRAY_API void
NpyArray_TimedeltaToTimedeltaStruct(npy_timedelta val, NPY_DATETIMEUNIT fr,
                                    npy_timedeltastruct *result);

NDARRAY_API NpyArray_DateTimeInfo*
NpyArray_DateTimeInfoNew(const char* units, int num, int den, int events);

extern NDARRAY_API char *_datetime_strings[];

NDARRAY_API int
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
    NpyExc_MemoryError=0,
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
NDARRAY_API extern npy_tp_error_set NpyErr_SetString;
NDARRAY_API extern npy_tp_error_occurred NpyErr_Occurred;
NDARRAY_API extern npy_tp_error_clear NpyErr_Clear;

#define NpyErr_MEMORY  NpyErr_SetString(NpyExc_MemoryError, "memory error")


typedef int (*npy_tp_cmp_priority)(void *, void *);

NDARRAY_API extern npy_tp_cmp_priority Npy_CmpPriority;


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
NDARRAY_API extern npy_interface_incref _NpyInterface_Incref;
NDARRAY_API extern npy_interface_decref _NpyInterface_Decref;


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
typedef int (*npy_interface_ufunc_new_wrapper)(
        void *base, void **interfaceRet);


struct NpyInterface_WrapperFuncs {
    npy_interface_array_new_wrapper array_new_wrapper;
    npy_interface_iter_new_wrapper iter_new_wrapper;
    npy_interface_multi_iter_new_wrapper multi_iter_new_wrapper;
    npy_interface_neighbor_iter_new_wrapper neighbor_iter_new_wrapper;
    npy_interface_descr_new_from_type descr_new_from_type;
    npy_interface_descr_new_from_wrapper descr_new_from_wrapper;
    npy_interface_ufunc_new_wrapper ufunc_new_wrapper;
};



NDARRAY_API extern void 
npy_initlib(struct NpyArray_FunctionDefs *functionDefs,
            struct NpyInterface_WrapperFuncs *wrapperFuncs,
            npy_tp_error_set error_set,
            npy_tp_error_occurred error_occurred,
            npy_tp_error_clear error_clear,
            npy_tp_cmp_priority cmp_priority,
            npy_interface_incref incref,
            npy_interface_decref decref);

NDARRAY_API extern void 
npy_set_ufunc_wrapper_func(npy_interface_ufunc_new_wrapper wrapperFunc);


/*
 * TMP
 */
NDARRAY_API extern int _flat_copyinto(NpyArray *dst, NpyArray *src,
                                      NPY_ORDER order);
extern void _unaligned_strided_byte_copy(char *dst, npy_intp outstrides,
                                         char *src, npy_intp instrides,
                                         npy_intp N, int elsize);
extern void _strided_byte_swap(void *p, npy_intp stride, npy_intp n, int size);


#endif
