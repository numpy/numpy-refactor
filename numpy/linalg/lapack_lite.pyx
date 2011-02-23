""" Cythonized version of lapack_litemodule.c
"""

cdef extern from "lapack_lite.h":
    ctypedef struct f2c_doublecomplex:
        double r, i
    
    int lapack_dgeev "GLOBALFUNC(dgeev_)"(char *jobvl, char *jobvr, int *n, double *a, int *lda,double *wr,double *wi,double *vl, int *ldvl,double *vr, int *ldvr,double *work, int *lwork,int *info)
    int lapack_dsyevd "GLOBALFUNC(dsyevd_)"(char *jobz,char *uplo,int *n,double *a,int *lda,double *w,double *work,int *lwork,int *iwork,int *liwork,int *info)
    int lapack_zheevd "GLOBALFUNC(zheevd_)"(char *jobz,char *uplo,int *n,f2c_doublecomplex *a,int *lda,double *w,f2c_doublecomplex *work,int *lwork,double *rwork,int *lrwork,int *iwork,int *liwork,int *info)
    int lapack_dgelsd "GLOBALFUNC(dgelsd_)"(int *m,int *n,int *nrhs,double *a,int *lda,double *b,int *ldb,double *s,double *rcond,int *rank,double *work,int *lwork,int *iwork,int *info)
    int lapack_dgesv "GLOBALFUNC(dgesv_)"(int *n,int *nrhs,double *a,int *lda, int *ipiv, double *b,int *ldb, int *info)
    int lapack_dgesdd "GLOBALFUNC(dgesdd_)"(char *jobz,int *m,int *n, double *a,int *lda, double *s, double *u,int *ldu, double *vt,int *ldvt, double *work,int *lwork, int *iwork,int *info)
    int lapack_dgetrf "GLOBALFUNC(dgetrf_)"(int *m,int *n, double *a,int *lda, int *ipiv,int *info)
    int lapack_dpotrf "GLOBALFUNC(dpotrf_)"(char *uplo,int *n, double *a,int *lda, int *info)
    int lapack_dgeqrf "GLOBALFUNC(dgeqrf_)"(int *m, int *n,  double *a, int *lda, double *tau, double *work, int *lwork, int *info)
    int lapack_dorgqr "GLOBALFUNC(dorgqr_)"(int *m, int *n, int *k, double *a, int *lda, double *tau, double *work, int *lwork, int *info)
    int lapack_zgeev "GLOBALFUNC(zgeev_)"(char *jobvl,char *jobvr,int *n, f2c_doublecomplex *a,int *lda, f2c_doublecomplex *w, f2c_doublecomplex *vl,int *ldvl, f2c_doublecomplex *vr,int *ldvr, f2c_doublecomplex *work,int *lwork, double *rwork,int *info)
    int lapack_zgelsd "GLOBALFUNC(zgelsd_)"(int *m,int *n,int *nrhs, f2c_doublecomplex *a,int *lda, f2c_doublecomplex *b,int *ldb, double *s,double *rcond,int *rank, f2c_doublecomplex *work,int *lwork, double *rwork, int *iwork,int *info)
    int lapack_zgesv "GLOBALFUNC(zgesv_)"(int *n,int *nrhs, f2c_doublecomplex *a,int *lda, int *ipiv, f2c_doublecomplex *b,int *ldb, int *info)
    int lapack_zgesdd "GLOBALFUNC(zgesdd_)"(char *jobz,int *m,int *n, f2c_doublecomplex *a,int *lda, double *s, f2c_doublecomplex *u,int *ldu, f2c_doublecomplex *vt,int *ldvt, f2c_doublecomplex *work,int *lwork, double *rwork, int *iwork,int *info)
    int lapack_zgetrf "GLOBALFUNC(zgetrf_)"(int *m,int *n, f2c_doublecomplex *a,int *lda, int *ipiv,int *info)
    int lapack_zpotrf "GLOBALFUNC(zpotrf_)"(char *uplo,int *n, f2c_doublecomplex *a,int *lda, int *info)
    int lapack_zgeqrf "GLOBALFUNC(zgeqrf_)"(int *m, int *n, f2c_doublecomplex *a, int *lda, f2c_doublecomplex *tau, f2c_doublecomplex *work, int *lwork, int *info)
    int lapack_zungqr "GLOBALFUNC(zungqr_)"(int *m, int *n, int *k, f2c_doublecomplex *a, int *lda, f2c_doublecomplex *tau, f2c_doublecomplex *work,int *lwork,int *info)


cimport numpy as np
np.import_array()

class LapackError(Exception):
    pass


cdef int check_object(np.ndarray ob, int t, char *obname, char *tname, char *funname):
    if not np.PyArray_CHKFLAGS(ob, np.NPY_CONTIGUOUS):
        raise LapackError("Parameter %s is not contiguous in lapack_lite.%s" % (obname, funname))
    elif np.PyArray_TYPE(ob) != t:
        raise LapackError("Parameter %s is not of type %s in lapack_lite.%s" % (obname, tname, funname))
    elif np.PyArray_DESCR(ob).byteorder != '=' and np.PyArray_DESCR(ob).byteorder != '|':
        raise LapackError("Parameter %s has non-native byte order in lapack_lite.%s" % (obname, funname))
    
    return 1


cdef dgeev(jobvl, jobvr, int n, np.ndarray a, int lda,
           np.ndarray wr, np.ndarray wi, np.ndarray vl,
           int ldvl, np.ndarray vr, int ldvr, np.ndarray work, int lwork, int info):
    cdef int lapack_lite_status__
    cdef char jobvl_char = ord(jobvl[0])
    cdef char jobvr_char = ord(jobvr[0])

    if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgeev"): return None
    if not check_object(wr,np.NPY_DOUBLE,"wr","np.NPY_DOUBLE","dgeev"): return None
    if not check_object(wi,np.NPY_DOUBLE,"wi","np.NPY_DOUBLE","dgeev"): return None
    if not check_object(vl,np.NPY_DOUBLE,"vl","np.NPY_DOUBLE","dgeev"): return None
    if not check_object(vr,np.NPY_DOUBLE,"vr","np.NPY_DOUBLE","dgeev"): return None
    if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgeev"): return None

    lapack_lite_status__ = lapack_dgeev(&jobvl_char,&jobvr_char,&n,
                                        <double *>np.PyArray_DATA(a),&lda,
                                        <double *>np.PyArray_DATA(wr),
                                        <double *>np.PyArray_DATA(wi),
                                        <double *>np.PyArray_DATA(vl),&ldvl,
                                        <double *>np.PyArray_DATA(vr),&ldvr,
                                        <double *>np.PyArray_DATA(work),&lwork,
                                        &info)

    retval = {}
    retval["dgeev_"] = lapack_lite_status__
    retval["jobvl"] = jobvl_char
    retval["jobvr"] = jobvr_char
    retval["n"] = n
    retval["lda"] = lda
    retval["ldvl"] = ldvl
    retval["ldvr"] = ldvr
    retval["lwork"] = lwork
    retval["info"] = info
    
    return retval

            
cdef dsyevd(jobz, uplo, int n, np.ndarray a, int lda,
            np.ndarray w, np.ndarray work, int lwork, np.ndarray iwork, int liwork, int info):
    """ Arguments
        =========
    JOBZ    (input) CHARACTER*1
            = 'N':  Compute eigenvalues only;
            = 'V':  Compute eigenvalues and eigenvectors.
    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of A is stored.
            = 'L':  Lower triangle of A is stored.
    N       (input) INTEGER
            The order of the matrix A.  N >= 0.
    A       (input/output) DOUBLE PRECISION array, dimension (LDA, N)
            On entry, the symmetric matrix A.  If UPLO = 'U', the
            leading N-by-N upper triangular part of A contains the
            upper triangular part of the matrix A.  If UPLO = 'L',
            the leading N-by-N lower triangular part of A contains
            the lower triangular part of the matrix A.
            On exit, if JOBZ = 'V', then if INFO = 0, A contains the
            orthonormal eigenvectors of the matrix A.
            If JOBZ = 'N', then on exit the lower triangle (if UPLO='L')
            or the upper triangle (if UPLO='U') of A, including the
            diagonal, is destroyed.
    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).
    W       (output) DOUBLE PRECISION array, dimension (N)
            If INFO = 0, the eigenvalues in ascending order.
    WORK    (workspace/output) DOUBLE PRECISION array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
    LWORK   (input) INTEGER
            The length of the array WORK.  LWORK >= max(1,3*N-1).
            For optimal efficiency, LWORK >= (NB+2)*N,
            where NB is the blocksize for DSYTRD returned by ILAENV.
    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  if INFO = i, the algorithm failed to converge; i
                  off-diagonal elements of an intermediate tridiagonal
                  form did not converge to zero.
    """
    cdef int lapack_lite_status__
    cdef char jobz_char = ord(jobz[0])
    cdef char uplo_char = ord(uplo[0])

    if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dsyevd"): return None
    if not check_object(w,np.NPY_DOUBLE,"w","np.NPY_DOUBLE","dsyevd"): return None
    if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dsyevd"): return None
    if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","dsyevd"): return None

    lapack_lite_status__ = lapack_dsyevd(&jobz_char,&uplo_char,&n,
                                         <double *>np.PyArray_DATA(a),&lda,
                                         <double *>np.PyArray_DATA(w),
                                         <double *>np.PyArray_DATA(work),&lwork,
                                         <int *>np.PyArray_DATA(iwork),&liwork,&info)

    retval = {}
    retval["dsyevd_"] = lapack_lite_status__
    retval["jobz"] = jobz_char
    retval["uplo"] = uplo_char
    retval["n"] = n
    retval["lda"] = lda
    retval["lwork"] = lwork
    retval["liwork"] = liwork
    retval["info"] = info
    return retval


cdef zheevd(jobz, uplo, int n, np.ndarray a, int lda,
            np.ndarray w, np.ndarray work, int lwork,
            np.ndarray rwork, int lrwork,
            np.ndarray iwork, int liwork, int info):
    """ Arguments
        =========
        JOBZ    (input) CHARACTER*1
                = 'N':  Compute eigenvalues only;
                = 'V':  Compute eigenvalues and eigenvectors.
        UPLO    (input) CHARACTER*1
                = 'U':  Upper triangle of A is stored;
                = 'L':  Lower triangle of A is stored.
        N       (input) INTEGER
                The order of the matrix A.  N >= 0.
        A       (input/output) COMPLEX*16 array, dimension (LDA, N)
                On entry, the Hermitian matrix A.  If UPLO = 'U', the
                leading N-by-N upper triangular part of A contains the
                upper triangular part of the matrix A.  If UPLO = 'L',
                the leading N-by-N lower triangular part of A contains
                the lower triangular part of the matrix A.
                On exit, if JOBZ = 'V', then if INFO = 0, A contains the
                orthonormal eigenvectors of the matrix A.
                If JOBZ = 'N', then on exit the lower triangle (if UPLO='L')
                or the upper triangle (if UPLO='U') of A, including the
                diagonal, is destroyed.
        LDA     (input) INTEGER
                The leading dimension of the array A.  LDA >= max(1,N).
        W       (output) DOUBLE PRECISION array, dimension (N)
                If INFO = 0, the eigenvalues in ascending order.
        WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
                On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
        LWORK   (input) INTEGER
                The length of the array WORK.  LWORK >= max(1,3*N-1).
                For optimal efficiency, LWORK >= (NB+2)*N,
                where NB is the blocksize for DSYTRD returned by ILAENV.
        RWORK   (workspace) DOUBLE PRECISION array, dimension (max(1, 3*N-2))
        INFO    (output) INTEGER
                = 0:  successful exit
                < 0:  if INFO = -i, the i-th argument had an illegal value
                > 0:  if INFO = i, the algorithm failed to converge; i
                      off-diagonal elements of an intermediate tridiagonal
                      form did not converge to zero.
    """
    cdef int lapack_lite_status__
    cdef char jobz_char = ord(jobz[0])
    cdef char uplo_char = ord(uplo[0])

    if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zheevd"): return None
    if not check_object(w,np.NPY_DOUBLE,"w","np.NPY_DOUBLE","zheevd"): return None
    if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zheevd"): return None
    if not check_object(w,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zheevd"): return None
    if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","zheevd"): return None

    lapack_lite_status__ = lapack_zheevd(&jobz_char,&uplo_char,&n,
                                         <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
                                         <double *>np.PyArray_DATA(w),
                                         <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
                                         <double *>np.PyArray_DATA(rwork),&lrwork,
                                         <int *>np.PyArray_DATA(iwork),&liwork,&info)

    retval = {}
    retval["zheevd_"] = lapack_lite_status__
    retval["jobz"] = jobz_char
    retval["uplo"] = uplo_char
    retval["n"] = n
    retval["lda"] = lda
    retval["lwork"] = lwork
    retval["lrwork"] = lrwork
    retval["liwork"] = liwork
    retval["info"] = info
    return retval


cdef dgelsd(int m, int n, int nrhs, np.ndarray a, int lda, np.ndarray b, int ldb,
            np.ndarray s, double rcond, int rank,
            np.ndarray work, int lwork, np.ndarray iwork, int info):
    cdef int lapack_lite_status__

    if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgelsd"): return None
    if not check_object(b,np.NPY_DOUBLE,"b","np.NPY_DOUBLE","dgelsd"): return None
    if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","dgelsd"): return None
    if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgelsd"): return None
    if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","dgelsd"): return None

    lapack_lite_status__ = lapack_dgelsd(&m,&n,&nrhs,
                                         <double *>np.PyArray_DATA(a),&lda,
                                         <double *>np.PyArray_DATA(b),&ldb,
                                         <double *>np.PyArray_DATA(s),&rcond,&rank,
                                         <double *>np.PyArray_DATA(work),&lwork,
                                         <int *>np.PyArray_DATA(iwork),&info)

    retval = {}
    retval["dgelsd_"] = lapack_lite_status__
    retval["m"] = m
    retval["n"] = n
    retval["nrhs"] = nrhs
    retval["lda"] = lda
    retval["ldb"] = ldb
    retval["rcond"] = rcond
    retval["rank"] = rank
    retval["lwork"] = lwork
    retval["info"] = info
    return retval


cdef dgesv(int n, int nrhs, np.ndarray a, int lda, np.ndarray ipiv,
           np.ndarray b, int ldb, int info):
    cdef int lapack_lite_status__

    if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgesv"): return None
    if not check_object(ipiv,np.NPY_INT,"ipiv","np.NPY_INT","dgesv"): return None
    if not check_object(b,np.NPY_DOUBLE,"b","np.NPY_DOUBLE","dgesv"): return None

    lapack_lite_status__ = lapack_dgesv(&n,&nrhs,
                                        <double *>np.PyArray_DATA(a),&lda,
                                        <int *>np.PyArray_DATA(ipiv),
                                        <double *>np.PyArray_DATA(b),&ldb,
                                        &info)

    retval = {}
    retval["dgesv_"] = lapack_lite_status__
    retval["n"] = n
    retval["nrhs"] = nrhs
    retval["lda"] = lda
    retval["ldb"] = ldb
    retval["info"] = info
    return retval


cdef dgesdd(jobz, int m, int n, np.ndarray a, int lda,
            np.ndarray s, np.ndarray u, int ldu, np.ndarray vt, int ldvt,
            np.ndarray work, int lwork, np.ndarray iwork, int info):
    cdef int lapack_lite_status__
    cdef char jobz_char = ord(jobz[0])
    cdef long work0
    cdef int mn, mx

    if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgesdd"): return None
    if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","dgesdd"): return None
    if not check_object(u,np.NPY_DOUBLE,"u","np.NPY_DOUBLE","dgesdd"): return None
    if not check_object(vt,np.NPY_DOUBLE,"vt","np.NPY_DOUBLE","dgesdd"): return None
    if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgesdd"): return None
    if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","dgesdd"): return None

    lapack_lite_status__ = lapack_dgesdd(&jobz_char,&m,&n,
                                         <double *>np.PyArray_DATA(a),&lda,
                                         <double *>np.PyArray_DATA(s),
                                         <double *>np.PyArray_DATA(u),&ldu,
                                         <double *>np.PyArray_DATA(vt),&ldvt,
                                         <double *>np.PyArray_DATA(work),&lwork,
                                         <int *>np.PyArray_DATA(iwork),&info)

    if info == 0 and lwork == -1:
        # We need to check the result because
        # sometimes the "optimal" value is actually
        # too small.
        # Change it to the maximum of the minimum and the optimal.
        work0 = <long>(<double *>np.PyArray_DATA(work))[0]
        mn = min(m,n)
        mx = max(m,n)

        if jobz == 'N':
            work0 = max(work0,3*mn + max(mx,6*mn)+500)
        elif jobz == 'O':
            work0 = max(work0,3*mn*mn + max(mx,5*mn*mn+4*mn+500))
        elif jobz == 'S' or jobz == 'A':
            work0 = max(work0,3*mn*mn + max(mx,4*mn*(mn+1))+500)
        
        (<double *>np.PyArray_DATA(work))[0] = <double>work0
    
    retval = {}
    retval["dgesdd_"] = lapack_lite_status__
    retval["jobz"] = jobz_char
    retval["m"] = m
    retval["n"] = n
    retval["lda"] = lda
    retval["ldu"] = ldu
    retval["ldvt"] = ldvt
    retval["lwork"] = lwork
    retval["info"] = info
    return retval


cdef dgetrf(int m, int n, np.ndarray a, int lda, np.ndarray ipiv, int info):
    cdef int lapack_lite_status__

    if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgetrf"): return None
    if not check_object(ipiv,np.NPY_INT,"ipiv","np.NPY_INT","dgetrf"): return None

    lapack_lite_status__ = lapack_dgetrf(&m,&n,<double *>np.PyArray_DATA(a),&lda,
                                         <int *>np.PyArray_DATA(ipiv),&info)

    retval = {}
    retval["dgetrf_"] = lapack_lite_status__
    retval["m"] = m
    retval["n"] = n
    retval["lda"] = lda
    retval["info"] = info
    return retval


cdef dpotrf(uplo, int n, np.ndarray a, int lda, int info):
    cdef int lapack_lite_status__
    cdef char uplo_char = ord(uplo[0])

    if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dpotrf"): return None

    lapack_lite_status__ = lapack_dpotrf(&uplo_char,&n,
                                         <double *>np.PyArray_DATA(a),&lda,
                                         &info)

    retval = {}
    retval["dpotrf_"] = lapack_lite_status__
    retval["n"] = n
    retval["lda"] = lda
    retval["info"] = info
    return retval


cdef dgeqrf(int m, int n, np.ndarray a, int lda,
            np.ndarray tau, np.ndarray work, int lwork, int info):
    cdef int  lapack_lite_status__

    # check objects and convert to right storage order
    if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgeqrf"): return None
    if not check_object(tau,np.NPY_DOUBLE,"tau","np.NPY_DOUBLE","dgeqrf"): return None
    if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgeqrf"): return None

    lapack_lite_status__ = lapack_dgeqrf(&m, &n, 
                                         <double *>np.PyArray_DATA(a), &lda,
                                         <double *>np.PyArray_DATA(tau),
                                         <double *>np.PyArray_DATA(work), &lwork, 
                                         &info)

    retval = {}
    retval["dgeqrf_"] = lapack_lite_status__
    retval["m"] = m
    retval["n"] = n
    retval["lda"] = lda
    retval["lwork"] = lwork
    retval["info"] = info
    return retval


cdef dorgqr(int m, int n, int k, np.ndarray a, int lda,
            np.ndarray tau, np.ndarray work, int lwork, int info):
    cdef int  lapack_lite_status__

    if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dorgqr"): return None
    if not check_object(tau,np.NPY_DOUBLE,"tau","np.NPY_DOUBLE","dorgqr"): return None
    if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dorgqr"): return None
    
    lapack_lite_status__ = lapack_dorgqr(&m, &n, &k,
                                         <double *>np.PyArray_DATA(a), &lda,
                                         <double *>np.PyArray_DATA(tau),
                                         <double *>np.PyArray_DATA(work), &lwork,
                                         &info)

    retval = {}
    retval["dorgqr_"] = lapack_lite_status__
    retval["info"] = info
    return retval


cdef zgeev(jobvl, jobvr, int n, np.ndarray a, int lda,
           np.ndarray w, np.ndarray vl, int ldvl, np.ndarray vr, int ldvr,
           np.ndarray work, int lwork, np.ndarray rwork, int info):
    cdef int lapack_lite_status__
    cdef char jobvl_char = ord(jobvl[0])
    cdef char jobvr_char = ord(jobvr[0])

    if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgeev"): return None
    if not check_object(w,np.NPY_CDOUBLE,"w","np.NPY_CDOUBLE","zgeev"): return None
    if not check_object(vl,np.NPY_CDOUBLE,"vl","np.NPY_CDOUBLE","zgeev"): return None
    if not check_object(vr,np.NPY_CDOUBLE,"vr","np.NPY_CDOUBLE","zgeev"): return None
    if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgeev"): return None
    if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgeev"): return None

    lapack_lite_status__ = lapack_zgeev(&jobvl_char,&jobvr_char,&n,
                                        <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
                                        <f2c_doublecomplex *>np.PyArray_DATA(w),
                                        <f2c_doublecomplex *>np.PyArray_DATA(vl),&ldvl,
                                        <f2c_doublecomplex *>np.PyArray_DATA(vr),&ldvr,
                                        <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
                                        <double *>np.PyArray_DATA(rwork),&info)

    retval = {}
    retval["zgeev_"] = lapack_lite_status__
    retval["jobvl"] = jobvl_char
    retval["jobvr"] = jobvr_char
    retval["n"] = n
    retval["lda"] = lda
    retval["ldvl"] = ldvl
    retval["ldvr"] = ldvr
    retval["lwork"] = lwork
    retval["info"] = info
    return retval


cdef zgelsd(int m, int n, int nrhs, np.ndarray a, int lda,
            np.ndarray b, int ldb, np.ndarray s, double rcond,
            int rank, np.ndarray work, int lwork,
            np.ndarray rwork, np.ndarray iwork, int info):
    cdef int  lapack_lite_status__

    if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgelsd"): return None
    if not check_object(b,np.NPY_CDOUBLE,"b","np.NPY_CDOUBLE","zgelsd"): return None
    if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","zgelsd"): return None
    if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgelsd"): return None
    if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgelsd"): return None
    if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","zgelsd"): return None

    lapack_lite_status__ = lapack_zgelsd(&m,&n,&nrhs,
                                         <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
                                         <f2c_doublecomplex *>np.PyArray_DATA(b),&ldb,
                                         <double *>np.PyArray_DATA(s),&rcond,&rank,
                                         <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
                                         <double *>np.PyArray_DATA(rwork),
                                         <int *>np.PyArray_DATA(iwork),&info)

    retval = {}
    retval["zgelsd_"] = lapack_lite_status__
    retval["m"] = m
    retval["n"] = n
    retval["nrhs"] = nrhs
    retval["lda"] = lda
    retval["ldb"] = ldb
    retval["rank"] = rank
    retval["lwork"] = lwork
    retval["info"] = info
    return retval


cdef zgesv(int n, int nrhs, np.ndarray a, int lda,
           np.ndarray ipiv, np.ndarray b, int ldb, int info):
    cdef int lapack_lite_status__

    if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgesv"): return None
    if not check_object(ipiv,np.NPY_INT,"ipiv","np.NPY_INT","zgesv"): return None
    if not check_object(b,np.NPY_CDOUBLE,"b","np.NPY_CDOUBLE","zgesv"): return None

    lapack_lite_status__ = lapack_zgesv(&n,&nrhs,
                                        <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
                                        <int *>np.PyArray_DATA(ipiv),
                                        <f2c_doublecomplex *>np.PyArray_DATA(b),&ldb,
                                        &info)

    retval = {}
    retval["zgesv_"] = lapack_lite_status__
    retval["n"] = n
    retval["nrhs"] = nrhs
    retval["lda"] = lda
    retval["ldb"] = ldb
    retval["info"] = info
    return retval


cdef zgesdd(jobz, int m, int n, np.ndarray a, int lda,
            np.ndarray s, np.ndarray u, int ldu, np.ndarray vt, int ldvt,
            np.ndarray work, int lwork, np.ndarray rwork, np.ndarray iwork, int info):
    cdef int lapack_lite_status__
    cdef char jobz_char = ord(jobz[0])

    if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgesdd"): return None
    if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","zgesdd"): return None
    if not check_object(u,np.NPY_CDOUBLE,"u","np.NPY_CDOUBLE","zgesdd"): return None
    if not check_object(vt,np.NPY_CDOUBLE,"vt","np.NPY_CDOUBLE","zgesdd"): return None
    if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgesdd"): return None
    if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgesdd"): return None
    if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","zgesdd"): return None

    lapack_lite_status__ = lapack_zgesdd(&jobz_char,&m,&n,
                                         <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
                                         <double *>np.PyArray_DATA(s),
                                         <f2c_doublecomplex *>np.PyArray_DATA(u),&ldu,
                                         <f2c_doublecomplex *>np.PyArray_DATA(vt),&ldvt,
                                         <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
                                         <double *>np.PyArray_DATA(rwork),
                                         <int *>np.PyArray_DATA(iwork),&info)

    retval = {}
    retval["zgesdd_"] = lapack_lite_status__
    retval["jobz"] = jobz_char
    retval["m"] = m
    retval["n"] = n
    retval["lda"] = lda
    retval["ldu"] = ldu
    retval["ldvt"] = ldvt
    retval["lwork"] = lwork
    retval["info"] = info
    return retval


cdef zgetrf(int m, int n, np.ndarray a, int lda, np.ndarray ipiv, int info):
    cdef int lapack_lite_status__
    
    if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgetrf"): return None
    if not check_object(ipiv,np.NPY_INT,"ipiv","np.NPY_INT","zgetrf"): return None

    lapack_lite_status__ = lapack_zgetrf(&m,&n,
                                         <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
                                         <int *>np.PyArray_DATA(ipiv),&info)

    retval = {}
    retval["zgetrf_"] = lapack_lite_status__
    retval["m"] = m
    retval["n"] = n
    retval["lda"] = lda
    retval["info"] = info
    return retval


cdef zpotrf(uplo, int n, np.ndarray a, int lda, int info):
    cdef int  lapack_lite_status__
    cdef char uplo_char = ord(uplo[0])

    if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zpotrf"): return None

    lapack_lite_status__ = lapack_zpotrf(&uplo_char,&n,
                                         <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
                                         &info)

    retval = {}
    retval["zpotrf_"] = lapack_lite_status__
    retval["n"] = n
    retval["lda"] = lda
    retval["info"] = info
    return retval


cdef zgeqrf(int m, int n, np.ndarray a, int lda,
            np.ndarray tau, np.ndarray work, int lwork, int info):
    cdef int lapack_lite_status__

    # check objects and convert to right storage order
    if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgeqrf"): return None
    if not check_object(tau,np.NPY_CDOUBLE,"tau","np.NPY_CDOUBLE","zgeqrf"): return None
    if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgeqrf"): return None

    lapack_lite_status__ = lapack_zgeqrf(&m, &n,
                                         <f2c_doublecomplex *>np.PyArray_DATA(a), &lda,
                                         <f2c_doublecomplex *>np.PyArray_DATA(tau),
                                         <f2c_doublecomplex *>np.PyArray_DATA(work), &lwork,
                                         &info)

    retval = {}
    retval["zgeqrf_"] = lapack_lite_status__
    retval["m"] = m
    retval["n"] = n
    retval["lda"] = lda
    retval["lwork"] = lwork 
    retval["info"] = info
    return retval


cdef zungqr(int m, int n, int k, np.ndarray a, int lda,
            np.ndarray tau, np.ndarray work, int lwork, int info):
    cdef int  lapack_lite_status__

    if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zungqr"): return None
    if not check_object(tau,np.NPY_CDOUBLE,"tau","np.NPY_CDOUBLE","zungqr"): return None
    if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zungqr"): return None

    lapack_lite_status__ = lapack_zungqr(&m, &n, &k,
                                         <f2c_doublecomplex *>np.PyArray_DATA(a), &lda,
                                         <f2c_doublecomplex *>np.PyArray_DATA(tau), 
                                         <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,&info)

    retval = {}
    retval["zungqr_"] = lapack_lite_status__
    retval["info"] = info
    return retval
    
