
typedef struct { double r, i; } f2c_doublecomplex;

#ifdef __cplusplus
#define GLOBALFUNC(x) ::x
#else
#define GLOBALFUNC(x) x
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern int dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda,double *wr,double *wi,double *vl, int *ldvl,double *vr, int *ldvr,double *work, int *lwork,int *info);
extern int dsyevd_(char *jobz,char *uplo,int *n,double *a,int *lda,double *w,double *work,int *lwork,int *iwork,int *liwork,int *info);
extern int zheevd_(char *jobz,char *uplo,int *n,f2c_doublecomplex *a,int *lda,double *w,f2c_doublecomplex *work,int *lwork,double *rwork,int *lrwork,int *iwork,int *liwork,int *info);
extern int dgelsd_(int *m,int *n,int *nrhs,double *a,int *lda,double *b,int *ldb,double *s,double *rcond,int *rank,double *work,int *lwork,int *iwork,int *info);
extern int dgesv_(int *n,int *nrhs,double *a,int *lda, int *ipiv, double *b,int *ldb, int *info);
extern int dgesdd_(char *jobz,int *m,int *n, double *a,int *lda, double *s, double *u,int *ldu, double *vt,int *ldvt, double *work,int *lwork, int *iwork,int *info);
extern int dgetrf_(int *m,int *n, double *a,int *lda, int *ipiv,int *info);
extern int dpotrf_(char *uplo,int *n, double *a,int *lda, int *info);
extern int dgeqrf_(int *m, int *n,  double *a, int *lda, double *tau, double *work, int *lwork, int *info);
extern int dorgqr_(int *m, int *n, int *k, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
extern int zgeev_(char *jobvl,char *jobvr,int *n, f2c_doublecomplex *a,int *lda, f2c_doublecomplex *w, f2c_doublecomplex *vl,int *ldvl, f2c_doublecomplex *vr,int *ldvr, f2c_doublecomplex *work,int *lwork, double *rwork,int *info);
extern int zgelsd_(int *m,int *n,int *nrhs, f2c_doublecomplex *a,int *lda, f2c_doublecomplex *b,int *ldb, double *s,double *rcond,int *rank, f2c_doublecomplex *work,int *lwork, double *rwork, int *iwork,int *info);
extern int zgesv_(int *n,int *nrhs, f2c_doublecomplex *a,int *lda, int *ipiv, f2c_doublecomplex *b,int *ldb, int *info);
extern int zgesdd_(char *jobz,int *m,int *n, f2c_doublecomplex *a,int *lda, double *s, f2c_doublecomplex *u,int *ldu, f2c_doublecomplex *vt,int *ldvt, f2c_doublecomplex *work,int *lwork, double *rwork, int *iwork,int *info);
extern int zgetrf_(int *m,int *n, f2c_doublecomplex *a,int *lda, int *ipiv,int *info);
extern int zpotrf_(char *uplo,int *n, f2c_doublecomplex *a,int *lda, int *info);
extern int zgeqrf_(int *m, int *n, f2c_doublecomplex *a, int *lda, f2c_doublecomplex *tau, f2c_doublecomplex *work, int *lwork, int *info);
extern int zungqr_(int *m, int *n, int *k, f2c_doublecomplex *a, int *lda, f2c_doublecomplex *tau, f2c_doublecomplex *work,int *lwork,int *info);

#ifdef __cplusplus
}
#endif