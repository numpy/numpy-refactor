

cdef extern from "npy_defs.h":

    cdef enum NPY_TYPES:
        NPY_LONG
        NPY_DOUBLE

    cdef enum requirements:
        NPY_ALIGNED

    ctypedef int npy_intp



cdef extern from "npy_arrayobject.h":

    ctypedef struct NpyArray:
        char *data
        int nd
        npy_intp *dimensions
        npy_intp *strides
        int flags



cdef extern from "npy_api.h":

    NpyArray *NpyArray_New(...)



cdef extern from "npy_object.h":

    object Npy_INTERFACE(...)


cdef extern from "npy_descriptor.h":

    ctypedef struct NpyArray_Descr:
        int type_num, elsize, alignment
        char type, kind, byteorder, flags


def create_new():
    cdef npy_intp dimensions[1]
    cdef NpyArray *a

    dimensions[0] = 10

    a = NpyArray_New(NULL, 1, dimensions, NPY_DOUBLE,
                     NULL, NULL, 0, 0, NULL)
    return Npy_INTERFACE(a)


cdef extern from "numpy/ndarraytypes.h":
    # This is the C-Cython version, eventually we need
    # something else for C#-Cython
    ctypedef struct PyArrayObject:
        NpyArray *array

cdef extern from "numpy/arrayobject.h":
    object PyArray_FROM_OTF(object obj, NPY_TYPES type, int flags)


def receive_array(object o):
    cdef PyArrayObject *v
    cdef NpyArray *a
    cdef double x
    cdef int success = 0

    try:
        x = <double>o
        success = 1
    except:
        pass

    if success:
        print "Object is a float:", x
    else:
        #PyArray_FROM_OTF(o, NPY_DOUBLE, NPY_ALIGNED)
        v = <PyArrayObject *> o
        a = v.array
        print "Object is array with nd =", a.nd


# this import is necessary to do the initiallization of the core library
import numpy.core.multiarray
