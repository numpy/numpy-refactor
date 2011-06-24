from libc.stdint cimport intptr_t, uintptr_t

ctypedef signed char      npy_byte
ctypedef signed short     npy_short
ctypedef signed int       npy_int
ctypedef signed long      npy_long
ctypedef signed long long npy_longlong
ctypedef unsigned char      npy_ubyte
ctypedef unsigned short     npy_ushort
ctypedef unsigned int       npy_uint
ctypedef unsigned long      npy_ulong
ctypedef unsigned long long npy_ulonglong
ctypedef float          npy_float
ctypedef double         npy_double
ctypedef long double    npy_longdouble
ctypedef double         double_t
ctypedef intptr_t       npy_intp
ctypedef uintptr_t      npy_uintp
ctypedef npy_byte       npy_int8
ctypedef npy_short      npy_int16
ctypedef npy_int        npy_int32
ctypedef npy_longlong   npy_int64
ctypedef npy_ubyte      npy_uint8
ctypedef npy_ushort     npy_uint16
ctypedef npy_uint       npy_uint32
ctypedef npy_ulonglong  npy_uint64
ctypedef npy_float      npy_float32
ctypedef npy_double     npy_float64
ctypedef npy_intp       intp_t
ctypedef npy_int8       int8_t
ctypedef npy_int16      int16_t
ctypedef npy_int32      int32_t
ctypedef npy_int64      int64_t
ctypedef npy_uint8      uint8_t
ctypedef npy_uint16     uint16_t
ctypedef npy_uint32     uint32_t
ctypedef npy_uint64     uint64_t
ctypedef npy_float32    float32_t
ctypedef npy_float64    float64_t

ctypedef void (*NpyUFuncGenericFunction) (char **, npy_intp *, npy_intp *, void *)
ctypedef NpyUFuncGenericFunction PyUFuncGenericFunction

cdef extern from "":
    ctypedef class numpy.ndarray [clr "NumpyDotNet::ndarray"]:
        pass

    ctypedef class numpy.dtype [clr "NumpyDotNet::dtype"]:
        pass

    ndarray ArrayReturn "NumpyDotNet::ndarray::ArrayReturn" (ndarray arr)

cdef extern from "npy_common.h":

    ctypedef struct npy_cfloat:
        double real
        double imag

    ctypedef struct npy_cdouble:
        double real
        double imag

    ctypedef struct npy_clongdouble:
        double real
        double imag

    ctypedef struct npy_complex64:
        double real
        double imag

    ctypedef struct npy_complex128:
        double real
        double imag

    ctypedef struct npy_complex160:
        double real
        double imag

    ctypedef struct npy_complex192:
        double real
        double imag
        
    ctypedef struct npy_complex256:
        double real
        double imag

cdef extern from "npy_defs.h":
    cdef enum NPY_TYPES:
        NPY_BOOL
        NPY_BYTE
        NPY_UBYTE
        NPY_SHORT
        NPY_USHORT 
        NPY_INT
        NPY_UINT 
        NPY_LONG
        NPY_ULONG
        NPY_LONGLONG
        NPY_ULONGLONG
        NPY_FLOAT
        NPY_DOUBLE 
        NPY_LONGDOUBLE
        NPY_CFLOAT
        NPY_CDOUBLE
        NPY_CLONGDOUBLE
        NPY_OBJECT
        NPY_STRING
        NPY_UNICODE
        NPY_VOID
        NPY_NTYPES
        NPY_NOTYPE

        NPY_INT8
        NPY_INT16
        NPY_INT32
        NPY_INT64
        NPY_INT128
        NPY_INT256
        NPY_UINT8
        NPY_UINT16
        NPY_UINT32
        NPY_UINT64
        NPY_UINT128
        NPY_UINT256
        NPY_FLOAT16
        NPY_FLOAT32
        NPY_FLOAT64
        NPY_FLOAT80
        NPY_FLOAT96
        NPY_FLOAT128
        NPY_FLOAT256
        NPY_COMPLEX32
        NPY_COMPLEX64
        NPY_COMPLEX128
        NPY_COMPLEX160
        NPY_COMPLEX192
        NPY_COMPLEX256
        NPY_COMPLEX512

        NPY_INTP
        NPY_UINTP

    enum NPY_ORDER:
        NPY_ANYORDER
        NPY_CORDER
        NPY_FORTRANORDER

    enum NPY_CLIPMODE:
        NPY_CLIP
        NPY_WRAP
        NPY_RAISE

    enum NPY_SCALARKIND:
        NPY_NOSCALAR,
        NPY_BOOL_SCALAR,
        NPY_INTPOS_SCALAR,
        NPY_INTNEG_SCALAR,
        NPY_FLOAT_SCALAR,
        NPY_COMPLEX_SCALAR,
        NPY_OBJECT_SCALAR


    enum NPY_SORTKIND:
        NPY_QUICKSORT
        NPY_HEAPSORT
        NPY_MERGESORT

    cdef enum requirements:
        NPY_C_CONTIGUOUS
        NPY_F_CONTIGUOUS
        NPY_CONTIGUOUS
        NPY_FORTRAN
        NPY_OWNDATA
        NPY_FORCECAST
        NPY_ENSURECOPY
        NPY_ENSUREARRAY
        NPY_ELEMENTSTRIDES
        NPY_ALIGNED
        NPY_NOTSWAPPED
        NPY_WRITEABLE
        NPY_UPDATEIFCOPY
        NPY_ARR_HAS_DESCR

        NPY_BEHAVED
        NPY_BEHAVED_NS
        NPY_CARRAY
        NPY_CARRAY_RO
        NPY_FARRAY
        NPY_FARRAY_RO
        NPY_DEFAULT

        NPY_IN_ARRAY
        NPY_OUT_ARRAY
        NPY_INOUT_ARRAY
        NPY_IN_FARRAY
        NPY_OUT_FARRAY
        NPY_INOUT_FARRAY

        NPY_UPDATE_ALL
        
    cdef enum:
        NPY_MAXDIMS

    ctypedef struct NpyArray:
        pass

    cdef void *Npy_INCREF(void *)
    cdef void Npy_DECREF(void *)

cdef extern from "npy_descriptor.h":
    ctypedef struct NpyArray_Descr:
        char type
        # ... many other fields

cdef extern from "npy_arrayobject.h":
    ctypedef struct NpyArray:
        pass

    bint NpyArray_CHKFLAGS(NpyArray* obj, int flags)
    void *NpyArray_DATA(NpyArray* obj)
    NpyArray_Descr *NpyArray_DESCR(NpyArray* obj)
    npy_intp *NpyArray_DIMS(NpyArray* obj)
    npy_intp NpyArray_DIM(NpyArray *obj, int i)
    int NpyArray_ITEMSIZE(NpyArray* obj)
    npy_intp NpyArray_SIZE(NpyArray* obj)
    npy_intp NpyArray_NBYTES(NpyArray *obj)
    npy_intp* NpyArray_STRIDES(NpyArray* obj)
    int NpyArray_TYPE(NpyArray* obj)


cdef extern from "npy_ufunc_object.h":
    ctypedef struct NpyUFuncObject:
        pass

    ctypedef void (*NpyUFuncGenericFunction) (char **, npy_intp *,
                                              npy_intp *, void *)

    NpyUFuncObject *NpyUFunc_FromFuncAndDataAndSignature(NpyUFuncGenericFunction *func, void **data,
                                     char *types, int ntypes,
                                     int nin, int nout, int identity,
                                     char *name, char *doc,
                                     int check_return, char *signature)

cdef extern from "npy_api.h":
    NpyArray_Descr *NpyArray_DescrFromType(int typenum)
    NpyArray *NpyArray_New(void *subtype, int nd, npy_intp *dims, int type_num,
                           npy_intp *strides, void *data, int itemsize,
                           int flags, void *obj)
    int NpyArray_INCREF(NpyArray *arr)
    int NpyDataType_TYPE_NUM(NpyArray_Descr *)
    int NpyArray_AsCArray(NpyArray **apIn, void *ptr, npy_intp *dims, int nd,
                          NpyArray_Descr* typedescr)
    void NpyDataMem_FREE(char *ptr)

cdef extern from "npy_iterators.h":
    cdef enum:
        NPY_NEIGHBORHOOD_ITER_ZERO_PADDING
        NPY_NEIGHBORHOOD_ITER_ONE_PADDING
        NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING
        NPY_NEIGHBORHOOD_ITER_CIRCULAR_PADDING
        NPY_NEIGHBORHOOD_ITER_MIRROR_PADDING
    
    ctypedef void (*npy_free_func)(void*)

    ctypedef struct NpyArrayIterObject:
        npy_intp size
        npy_intp strides[NPY_MAXDIMS]
        NpyArray *ao
        char *dataptr

    ctypedef struct NpyArrayNeighborhoodIterObject:
        npy_intp size
        NpyArray *ao
        char *dataptr

    NpyArrayIterObject *NpyArray_IterNew(NpyArray *obj)
    NpyArrayIterObject *NpyArray_IterAllButAxis(NpyArray* obj, int *inaxis)
    void NpyArray_ITER_NEXT(NpyArrayIterObject *obj)
    void NpyArray_ITER_RESET(NpyArrayIterObject *obj)
    void *NpyArray_ITER_DATA(NpyArrayIterObject *obj)

    NpyArrayNeighborhoodIterObject* NpyArray_NeighborhoodIterNew(NpyArrayIterObject *obj,
                                                                 npy_intp *bounds, int mode, 
                                                                 void *fill, npy_free_func fillfree)
    int NpyArrayNeighborhoodIter_Reset(NpyArrayNeighborhoodIterObject* iter)
    int NpyArrayNeighborhoodIter_Next(NpyArrayNeighborhoodIterObject* iter)
 
cdef extern from "": # NpyArray.cs
    # Cython doesn't handle overloading
    dtype NpyArray_FindArrayType_3args "NumpyDotNet::NpyArray::FindArrayType" (object src, dtype minitype, int max)

cdef inline dtype NpyArray_FindArrayType_2args(object src, dtype minitype):
    return NpyArray_FindArrayType_3args(src, minitype, NPY_MAXDIMS)


cdef extern from "npy_ironpython.h":
    object Npy_INTERFACE_ufunc "Npy_INTERFACE_OBJECT" (NpyUFuncObject*)
    dtype Npy_INTERFACE_descr "Npy_INTERFACE_OBJECT" (NpyArray_Descr*)
    ndarray Npy_INTERFACE_array "Npy_INTERFACE_OBJECT" (NpyArray*)

ctypedef float complex  complex64_t
ctypedef double complex complex128_t

ctypedef npy_cfloat      cfloat_t
ctypedef npy_cdouble     cdouble_t
ctypedef npy_clongdouble clongdouble_t

ctypedef npy_cdouble     complex_t

ctypedef void (*PyArray_CopySwapFunc)(void *, void *, int, NpyArray *)

cdef inline object PyUFunc_FromFuncAndData(PyUFuncGenericFunction* func, void** data,
        char* types, int ntypes, int nin, int nout,
        int identity, char* name, char* doc, int c):
   return Npy_INTERFACE_ufunc(NpyUFunc_FromFuncAndDataAndSignature(func, data, types, ntypes, nin, nout, identity, name, doc, c, NULL))

cdef inline object PyArray_DescrFromType(int typenum):
    return Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum))


cdef inline object PyArray_ZEROS(int ndim, npy_intp *shape, int typenum, int fortran):
    shape_list = []
    cdef int i
    for i in range(ndim):
        shape_list.append(shape[i])
    import numpy
    return numpy.zeros(shape_list, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), 'F' if fortran else 'C')

cdef inline object PyArray_EMPTY(int ndim, npy_intp *shape, int typenum, int fortran):
    shape_list = []
    cdef int i
    for i in range(ndim):
        shape_list.append(shape[i])
    import numpy
    return numpy.empty(shape_list, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), 'F' if fortran else 'C')

cdef inline object PyArray_Empty(int nd, npy_intp *dims, dtype descr, int fortran):
    shape_list = []
    cdef int i
    for i in range(nd):
        shape_list.append(dims[i])
    import numpy
    return numpy.empty(shape_list, descr, 'F' if fortran else 'C')
    

cdef inline object PyArray_New(void *subtype, int nd, npy_intp *dims, int type_num, npy_intp *strides, void *data, int itemsize, int flags, void *obj):
    assert subtype == NULL
    assert obj == NULL
    return Npy_INTERFACE_array(NpyArray_New(subtype, nd, dims, type_num, strides, data, itemsize, flags, obj))

cdef inline object PyArray_SimpleNew(int nd, npy_intp *dims, int type_num):
    return PyArray_New(NULL, nd, dims, type_num, NULL, NULL, 0, NPY_CARRAY, NULL)

cdef inline object PyArray_SimpleNewFromData(int nd, npy_intp *dims, int type_num, void *data):
    return PyArray_New(NULL, nd, dims, type_num, NULL, data, 0, NPY_CARRAY, NULL)

cdef inline bint PyArray_CHKFLAGS(ndarray n, int flags):
    return  NpyArray_CHKFLAGS(<NpyArray*> <npy_intp>n.Array, flags)

cdef inline void* PyArray_DATA(ndarray n) nogil:
    return NpyArray_DATA(<NpyArray*> <npy_intp>n.Array)

cdef inline intp_t* PyArray_DIMS(ndarray n) nogil:
    return NpyArray_DIMS(<NpyArray*> <npy_intp>n.Array)

cdef inline object PyArray_DESCR(ndarray n):
    return Npy_INTERFACE_descr(NpyArray_DESCR(<NpyArray*> <npy_intp>n.Array))

cdef inline int PyArray_ITEMSIZE(ndarray n):
    return NpyArray_ITEMSIZE(<NpyArray*> <npy_intp>n.Array)

cdef inline object PyArray_Return(arr):
    if arr is None:
        return None
    import clr
    import NumpyDotNet.ndarray
    return NumpyDotNet.ndarray.ArrayReturn(arr)

cdef inline intp_t PyArray_DIM(ndarray n, int dim):
    return NpyArray_DIM(<NpyArray*><long long>n.Array, dim)

cdef inline object PyArray_NDIM(ndarray obj):
    return obj.ndim

cdef inline intp_t PyArray_SIZE(ndarray n):
    return NpyArray_SIZE(<NpyArray*> <npy_intp>n.Array)

cdef inline npy_intp* PyArray_STRIDES(ndarray n):
    return NpyArray_STRIDES(<NpyArray*> <npy_intp>n.Array) 

cdef inline npy_intp PyArray_NBYTES(ndarray n):
    return NpyArray_NBYTES(<NpyArray *><long long>n.Array)

cdef inline NpyArray *PyArray_ARRAY(ndarray n):
    return <NpyArray*> <npy_intp>n.Array

cdef inline int PyArray_TYPE(ndarray n):
    return NpyArray_TYPE(<NpyArray*> <npy_intp>n.Array)

cdef inline void *PyArray_Zero(arr):
    import clr
    import NumpyDotNet.NpyArray
    return <void *><npy_intp>NumpyDotNet.NpyArray.Zero(arr)

cdef inline object NpyArray_Return(NpyArray *arr):
    ret = Npy_INTERFACE_array(arr)
    Npy_DECREF(arr)
    return ret

cdef inline int PyDataType_TYPE_NUM(dtype t):
    return NpyDataType_TYPE_NUM(<NpyArray_Descr *><long long>t.Dtype)

cdef inline object PyArray_FromAny(op, newtype, min_depth, max_depth, flags, context):
    import clr
    import NumpyDotNet.NpyArray
    return NumpyDotNet.NpyArray.FromAny(op, newtype, min_depth, max_depth, flags, context)


cdef inline object PyArray_CopyFromObject(op, descr, min_depth, max_depth):
    return PyArray_FromAny(op, descr, min_depth, max_depth,
                           NPY_ENSURECOPY | NPY_DEFAULT | NPY_ENSUREARRAY, NULL)


cdef inline object PyArray_FROMANY(m, type, min, max, flags):
    if flags & NPY_ENSURECOPY:
        flags |= NPY_DEFAULT
    return PyArray_FromAny(m, Npy_INTERFACE_descr(NpyArray_DescrFromType(type)), min, max, flags, None)

cdef inline object PyArray_ContiguousFromObject(op, type, minDepth, maxDepth):
    return PyArray_FromAny(op, Npy_INTERFACE_descr(NpyArray_DescrFromType(type)), minDepth, maxDepth,
                           NPY_DEFAULT | NPY_ENSUREARRAY, NULL)

cdef inline object PyArray_CheckFromAny(op, newtype, min_depth, max_depth, flags, context):
    import clr
    import NumpyDotNet.NpyArray
    return NumpyDotNet.NpyArray.CheckFromAny(op, newtype, min_depth, max_depth, flags, context)

cdef inline object PyArray_Check(obj):
    import numpy as np
    return isinstance(obj, np.ndarray)

cdef inline object PyArray_Cast(arr, typenum):
    import clr
    import NumpyDotNet.NpyCoreApi
    return NumpyDotNet.NpyCoreApi.CastToType(arr, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), False)

cdef inline void import_array():
    pass

cdef inline object PyArray_DescrConverter(obj):
    import clr
    import NumpyDotNet.NpyDescr
    return NumpyDotNet.NpyDescr.DescrConverter(obj)

cdef inline PyNumber_Check(o):
    import clr
    import NumpyDotNet.ScalarGeneric
    return isinstance(o, (int, long, float)) or isinstance(o, NumpyDotNet.ScalarGeneric)

cdef inline NpyArrayIterObject *PyArray_IterNew(ndarray n):
    return NpyArray_IterNew(<NpyArray*> <npy_intp>n.Array)

cdef inline NpyArrayIterObject *PyArray_IterAllButAxis(ndarray n, int *inaxis):
    return NpyArray_IterAllButAxis(<NpyArray*> <npy_intp>n.Array, inaxis)

cdef inline void PyArray_ITER_NEXT(NpyArrayIterObject *obj):
    NpyArray_ITER_NEXT(obj)

cdef inline void PyArray_ITER_RESET(NpyArrayIterObject *obj):
    NpyArray_ITER_RESET(obj)

cdef inline void * PyArray_ITER_DATA(NpyArrayIterObject *obj):
    return NpyArray_ITER_DATA(obj)

cdef inline NpyArrayNeighborhoodIterObject* PyArray_NeighborhoodIterNew(NpyArrayIterObject *obj,
                                                                        npy_intp *bounds,
                                                                        int mode, 
                                                                        void *fill,
                                                                        npy_free_func fillfree):
    return NpyArray_NeighborhoodIterNew(obj, bounds, mode, fill, fillfree)

cdef inline int PyArrayNeighborhoodIter_Reset(NpyArrayNeighborhoodIterObject* iter):
    return NpyArrayNeighborhoodIter_Reset(iter)

cdef inline int PyArrayNeighborhoodIter_Next(NpyArrayNeighborhoodIterObject* iter):
    return NpyArrayNeighborhoodIter_Next(iter)

cdef inline ndarray NpyIter_ARRAY(NpyArrayIterObject *iter):
    return Npy_INTERFACE_array(iter.ao)
