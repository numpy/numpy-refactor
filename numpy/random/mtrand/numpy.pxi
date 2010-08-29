# :Author:    Travis Oliphant


cdef extern from "npy_defs.h":

    cdef enum NPY_TYPES:
        NPY_LONG
        NPY_DOUBLE

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
    void *NpyArray_MultiIter_NEXT(NpyArrayMultiIterObject *it)
    void *NpyArray_MultiIter_NEXTi(NpyArrayMultiIterObject *it, npy_intp i)
    void *NpyArray_MultiIter_DATA(NpyArrayMultiIterObject *it, npy_intp i)


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

    ctypedef struct PyArrayObject:
        NpyArray *array

    ctypedef struct PyArrayMultiIterObject:
        NpyArrayMultiIterObject *iter

    void import_array()


cdef extern from "numpy/arrayobject.h":
    pass
