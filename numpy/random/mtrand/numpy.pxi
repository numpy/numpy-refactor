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

    ctypedef extern class numpy.broadcast [object PyArrayMultiIterObject]:
        cdef NpyArrayMultiIterObject *iter

    char *PyArray_MultiIter_DATA(broadcast multi, int i)
    void PyArray_MultiIter_NEXTi(broadcast multi, int i)
    void PyArray_MultiIter_NEXT(broadcast multi)

    void import_array()


cdef extern from "numpy/arrayobject.h":
    pass
