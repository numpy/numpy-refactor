""" Cythonized version of fftpack_litemodule.c
"""

cdef extern from "string.h":
    void *memcpy(void *s1, void *s2, size_t n)

cdef extern from "fftpack.h":
    ctypedef double Treal
    void fftpack_cfftf "GLOBALFUNC(cfftf)"(int N, Treal *data, Treal *wrk)
    void fftpack_cfftb "GLOBALFUNC(cfftb)"(int N, Treal *data, Treal *wrk)
    void fftpack_cffti "GLOBALFUNC(cffti)"(int N, Treal *wrk)
    void fftpack_rfftf "GLOBALFUNC(rfftf)"(int N, Treal *data, Treal *wrk)
    void fftpack_rfftb "GLOBALFUNC(rfftb)"(int N, Treal *data, Treal *wrk)
    void fftpack_rffti "GLOBALFUNC(rffti)"(int N, Treal *wrk)

cimport numpy as np
np.import_array()

class error(Exception):
    pass

cdef cfftf(np.ndarray op1, object op2):
    cdef double *wsave, *dptr
    cdef np.intp_t nsave
    cdef int npts, nrepeats, i
    cdef np.ndarray data
    
    data = np.PyArray_FROMANY(op1, np.PyArray_CDOUBLE, 1, 0, np.NPY_ENSURECOPY|np.NPY_C_CONTIGUOUS)
    
    #if np.PyArray_AsCArray(<void **>&(<object>op2), <void *>&wsave, &nsave, 1, np.PyArray_DOUBLE, 0) == -1:
    #    return None
    
    nsave = np.PyArray_DIMS(op2)[0]
    npts = np.PyArray_DIM(data, np.PyArray_NDIM(data) - 1)
    if nsave != npts*4 + 15:
        raise error("invalid work array for fft size")
    
    nrepeats = np.PyArray_SIZE(data) / npts
    dptr = <double *>np.PyArray_DATA(data)
    wsave = <double *>np.PyArray_DATA(op2)
    
    for i in range(nrepeats):
        fftpack_cfftf(npts, dptr, wsave)
        dptr += npts*2
    
    return data

cdef cfftb(np.ndarray op1, np.ndarray op2):
    cdef double *wsave, *dptr
    cdef np.intp_t nsave
    cdef int npts, nrepeats, i
    
    data = np.PyArray_FROMANY(op1, np.PyArray_CDOUBLE, 1, 0, np.NPY_ENSURECOPY|np.NPY_C_CONTIGUOUS)

    nsave = np.PyArray_DIMS(op2)[0] 
    npts = np.PyArray_DIM(data, np.PyArray_NDIM(data) - 1)
    if nsave != npts*4 + 15:
        raise error("invalid work array for fft size")

    nrepeats = np.PyArray_SIZE(data) / npts
    dptr = <double *>np.PyArray_DATA(data)
    wsave = <double *>np.PyArray_DATA(op2)
    
    for i in range(nrepeats):
        fftpack_cfftb(npts, dptr, wsave)
        dptr += npts*2
    
    return data

cdef cffti(long n):
    cdef np.intp_t dim
    cdef np.ndarray op

    # Magic size needed by cffti
    dim = 4*n + 15;
    # Create a 1 dimensional array of dimensions of type double
    op = np.PyArray_New(NULL, 1, &dim, np.PyArray_DOUBLE, NULL, NULL, 0, 0, NULL)

    fftpack_cffti(n, <double *>np.PyArray_DATA(op))

    return op

cdef rfftf(np.ndarray op1, np.ndarray op2):
    cdef double *wsave, *dptr, *rptr
    cdef np.intp_t nsave
    cdef int npts, nrepeats, rstep, i

    data = np.PyArray_FROMANY(op1, np.PyArray_DOUBLE, 1, 0, np.NPY_C_CONTIGUOUS)
    npts = np.PyArray_DIMS(data)[np.PyArray_NDIM(data)-1]
    
    np.PyArray_DIMS(data)[np.PyArray_NDIM(data) - 1] = npts/2 + 1
    ret = np.PyArray_ZEROS(np.PyArray_NDIM(data), np.PyArray_DIMS(data), np.PyArray_CDOUBLE, 0)
    np.PyArray_DIMS(data)[np.PyArray_NDIM(data) - 1] = npts
    
    rstep = np.PyArray_DIMS(ret)[np.PyArray_NDIM(ret) - 1]*2

    nsave = np.PyArray_DIMS(op2)[0]
    if nsave != npts*2+15:
        raise error("invalid work array for fft size")

    nrepeats = np.PyArray_SIZE(data) / npts
    rptr = <double *>np.PyArray_DATA(ret)
    dptr = <double *>np.PyArray_DATA(data)
    wsave = <double *>np.PyArray_DATA(op2)

    for i in range(nrepeats):
        memcpy(<char *>(rptr+1), dptr, npts*sizeof(double))
        fftpack_rfftf(npts, rptr+1, wsave)
        rptr[0] = rptr[1]
        rptr[1] = 0.0
        rptr += rstep
        dptr += npts

    return ret

cdef rfftb(np.ndarray op1, np.ndarray op2):
    cdef double *wsave, *dptr, *rptr
    cdef np.intp_t nsave
    cdef int npts, nrepeats, i

    data = np.PyArray_FROMANY(op1, np.PyArray_CDOUBLE, 1, 0, np.NPY_C_CONTIGUOUS)
    npts = np.PyArray_DIMS(data)[np.PyArray_NDIM(data)-1]
    
    ret = np.PyArray_ZEROS(np.PyArray_NDIM(data), np.PyArray_DIMS(data), np.PyArray_DOUBLE, 0)    
    
    nsave = np.PyArray_DIMS(op2)[0]
    if nsave != npts*2 + 15:
        raise error("invalid work array for fft size")

    nrepeats = np.PyArray_SIZE(ret) / npts
    rptr = <double *>np.PyArray_DATA(ret)
    dptr = <double *>np.PyArray_DATA(data)
    wsave = <double *>np.PyArray_DATA(data)
    
    for i in range(nrepeats):
        memcpy(<char *>(rptr + 1), (dptr + 2), (npts - 1)*sizeof(double))
        rptr[0] = dptr[0]
        fftpack_rfftb(npts, rptr, wsave)
        rptr += npts
        dptr += npts*2

    return ret

cdef rffti(long n):
    cdef np.intp_t dim
    cdef np.ndarray op
  
    # Magic size needed by rffti
    dim = 2*n + 15
    # Create a 1 dimensional array of dimensions of type double
    op = np.PyArray_New(NULL, 1, &dim, np.PyArray_DOUBLE, NULL, NULL, 0, 0, NULL)

    fftpack_rffti(n, <double *>np.PyArray_DATA(op))

    return op
