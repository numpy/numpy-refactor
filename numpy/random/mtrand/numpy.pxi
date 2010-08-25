# :Author:    Travis Oliphant


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

    cdef enum requirements:
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

    cdef enum defines:
        NPY_MAXDIMS

    ctypedef int npy_intp


cdef extern from "npy_arrayobject.h":

    ctypedef struct NpyArray:
        char *data
        int nd
        npy_intp *dimensions
        npy_intp *strides
        int flags


cdef extern from "npy_iterators.h":

    ctypedef struct NpyArrayIterObject:
        int nd_m1
        npy_intp index, size
        char *dataptr

    ctypedef struct NpyArrayMultiIterObject:
        int numiter
        npy_intp size, index
        int nd
        npy_intp *dimensions
        void **iters

    NpyArrayIterObject *NpyArray_IterNew(NpyArray *ao)
    NpyArrayMultiIterObject NpyArray_MultiIterNew()
    void NpyArray_ITER_NEXT(NpyArrayIterObject *it)


cdef extern from "npy_common.h":

    ctypedef struct npy_cdouble:
        double real
        double imag

    ctypedef struct npy_cfloat:
        double real
        double imag


cdef extern from "npy_descriptor.h":

    ctypedef struct NpyArray_Descr:
        int type_num, elsize, alignment
        char type, kind, byteorder, flags


cdef extern from "numpy/ndarraytypes.h":

    ctypedef extern class numpy.ndarray [object PyArrayObject]:
        cdef NpyArray *array

    ctypedef extern class numpy.flatiter [object PyArrayIterObject]:
        cdef NpyArrayIterObject *iter

    ctypedef extern class numpy.broadcast [object PyArrayMultiIterObject]:
        cdef NpyArrayMultiIterObject *iter

    object PyArray_ZEROS(int ndims, npy_intp* dims, NPY_TYPES type_num,
                         int fortran)
    object PyArray_EMPTY(int ndims, npy_intp* dims, NPY_TYPES type_num,
                         int fortran)
    NpyArray_Descr PyArray_DescrFromTypeNum(NPY_TYPES type_num)
    object PyArray_SimpleNew(int ndims, npy_intp* dims, NPY_TYPES type_num)
    int PyArray_Check(object obj)
    object PyArray_ContiguousFromAny(object obj, NPY_TYPES type,
                                     int mindim, int maxdim)
    object PyArray_ContiguousFromObject(object obj, NPY_TYPES type,
                                        int mindim, int maxdim)
    npy_intp PyArray_SIZE(ndarray arr)
    npy_intp PyArray_NBYTES(ndarray arr)
    void *PyArray_DATA(ndarray arr)
    object PyArray_FromAny(object obj, NpyArray_Descr newtype, int mindim,
                           int maxdim, int requirements, object context)
    object PyArray_FROMANY(object obj, NPY_TYPES type_num, int min,
                           int max, int requirements)
    object PyArray_NewFromDescr(object subtype, NpyArray_Descr newtype, int nd,
                                npy_intp* dims, npy_intp* strides, void* data,
                                int flags, object parent)

    object PyArray_FROM_OTF(object obj, NPY_TYPES type, int flags)
    object PyArray_EnsureArray(object)
    object PyArray_MultiIterNew(int n, ...)

    char *PyArray_MultiIter_DATA(broadcast multi, int i)
    void PyArray_MultiIter_NEXTi(broadcast multi, int i)
    void PyArray_MultiIter_NEXT(broadcast multi)

    object PyArray_IterNew(object arr)
    void PyArray_ITER_NEXT(flatiter it)

    void import_array()


cdef extern from "numpy/arrayobject.h":
    pass
