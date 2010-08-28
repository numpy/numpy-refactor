# this import is necessary to do the initiallization of the core library
import numpy.core.multiarray
from numpy import array
import numpy as np


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


cdef extern from "numpy/ndarraytypes.h":
    # This is the C-Cython version, eventually we need
    # something else for C#-Cython
    ctypedef struct PyArrayObject:
        NpyArray *array


def create_new():
    cdef npy_intp dimensions[1]
    cdef NpyArray *a

    dimensions[0] = 10

    a = NpyArray_New(NULL, 1, dimensions, NPY_DOUBLE,
                     NULL, NULL, 0, 0, NULL)
    return Npy_INTERFACE(a)


def create_new2():
    cdef long i
    cdef double *data

    o = np.empty(20, np.double)
    data = <double *>(<PyArrayObject *>o).array.data
    for i from 0 <= i < o.size:
        data[i] = <double> (i + 0.1)
    return o


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
        o = array(o)
        v = <PyArrayObject *> o
        a = v.array    
        print "Object is array with nd =", a.nd
