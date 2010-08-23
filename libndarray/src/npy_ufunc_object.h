#ifndef _NPY_UFUNC_OBJECT_H_
#define _NPY_UFUNC_OBJECT_H_

#include <math.h>

#include "npy_object.h"
#include "npy_iterators.h"


typedef void (*NpyUFuncGenericFunction) (char **, npy_intp *,
                                         npy_intp *, void *);



#define NO_UFUNCLOOP        0
#define ZERO_EL_REDUCELOOP  0
#define ONE_UFUNCLOOP       1
#define ONE_EL_REDUCELOOP   1
#define NOBUFFER_UFUNCLOOP  2
#define NOBUFFER_REDUCELOOP 2
#define BUFFER_UFUNCLOOP    3
#define BUFFER_REDUCELOOP   3
#define SIGNATURE_NOBUFFER_UFUNCLOOP 4





typedef struct {
    int nin;
    int nout;
    void *callable;
} NpyUFunc_FuncData;


struct NpyDict_struct;

struct NpyUFuncObject {
    NpyObject_HEAD
    int magic_number;

    int nin, nout, nargs;
    int identity;
    NpyUFuncGenericFunction *functions;
    void **data;
    int ntypes;
    int check_return;
    char *name, *types;
    char *doc;
    void *ptr;
    struct NpyDict_struct *userloops;

    /* generalized ufunc */
    int core_enabled;      /* 0 for scalar ufunc; 1 for generalized ufunc */
    int core_num_dim_ix;   /* number of distinct dimension names in
                            signature */

    /* dimension indices of input/output argument k are stored in
     core_dim_ixs[core_offsets[k]..core_offsets[k]+core_num_dims[k]-1] */
    int *core_num_dims;    /* numbers of core dimensions of each argument */
    int *core_dim_ixs;     /* dimension indices in a flatted form; indices
                            are in the range of [0,core_num_dim_ix) */
    int *core_offsets;     /* positions of 1st core dimensions of each
                            argument in core_dim_ixs */
    char *core_signature;  /* signature string for printing purpose */
};

typedef struct NpyUFuncObject NpyUFuncObject;



extern struct NpyDict_struct *npy_create_userloops_table();


/* A linked-list of function information for
 user-defined 1-d loops.
 */
typedef struct _loop1d_info {
    NpyUFuncGenericFunction func;
    void *data;
    int *arg_types;
    struct _loop1d_info *next;
} NpyUFunc_Loop1d;



/* Only internal -- not exported, yet*/
typedef struct NpyUFuncLoopObject {
    /* Multi-iterator portion --- needs to be present in this order
     to work with PyArray_Broadcast */
    NpyObject_HEAD
    int magic_number;
    
    /* The iterators. */
    struct NpyArrayMultiIterObject *iter;
    
    /* The ufunc */
    struct NpyUFuncObject *ufunc;
    
    /* The error handling.  These fields are primary used by the interface
       layer to store info about what errors have occured. */
    int errormask;         /* Integer showing desired error handling */
    void *errobj;      /* currently a tuple with
                            (string, func or obj with write method or None) */
    int first;

    
    /* Specific function and data to use */
    NpyUFuncGenericFunction function;
    void *funcdata;
    
    /* Loop method */
    int meth;
    
    /* Whether we need to copy to a buffer or not.*/
    int needbuffer[NPY_MAXARGS];
    int leftover;
    int ninnerloops;
    int lastdim;
    
    /* Whether or not to swap */
    int swap[NPY_MAXARGS];
    
    /* Buffers for the loop */
    char *buffer[NPY_MAXARGS];
    int bufsize;
    npy_intp bufcnt;
    char *dptr[NPY_MAXARGS];
    
    /* For casting */
    char *castbuf[NPY_MAXARGS];
    NpyArray_VectorUnaryFunc *cast[NPY_MAXARGS];
    
    /* usually points to buffer but when a cast is to be
     done it switches for that argument to castbuf.
     */
    char *bufptr[NPY_MAXARGS];
    
    /* Steps filled in from iters or sizeof(item)
     depending on loop method.
     */
    npy_intp steps[NPY_MAXARGS];
    
    int obj;  /* This loop uses object arrays or needs the Python API */
    /* Flags: UFUNC_OBJ_ISOBJECT, UFUNC_OBJ_NEEDS_API */
    int notimplemented; /* The loop caused notimplemented */
    int objfunc; /* This loop calls object functions
                  (an inner-loop function with argument types */
    
    /* generalized ufunc */
    npy_intp *core_dim_sizes;   /* stores sizes of core dimensions;
                                 contains 1 + core_num_dim_ix elements */
    npy_intp *core_strides;     /* strides of loop and core dimensions */
} NpyUFuncLoopObject;




/* Could make this more clever someday */
#define NPY_UFUNC_MAXIDENTITY 32

typedef struct {
    NpyObject_HEAD
    int magic_number;
    
    NpyArrayIterObject *it;
    NpyArray *ret;
    NpyArrayIterObject *rit;   /* Needed for Accumulate */
    int  outsize;
    npy_intp  index;
    npy_intp  size;
    char idptr[NPY_UFUNC_MAXIDENTITY];
    
    /* The ufunc */
    NpyUFuncObject *ufunc;
    
    /* The error handling */
    int errormask;
    void *errobj;
    int first;
    
    NpyUFuncGenericFunction function;
    void *funcdata;
    int meth;
    int swap;
    
    char *buffer;
    int bufsize;
    
    char *castbuf;
    NpyArray_VectorUnaryFunc *cast;
    
    char *bufptr[3];
    npy_intp steps[3];
    
    npy_intp N;
    int  instrides;
    int  insize;
    char *inptr;
    
    /* For copying small arrays */
    NpyArray *decref_arr;
    
    int obj;
    int retbase;
    
} NpyUFuncReduceObject;



enum NpyArray_Ops {
    npy_op_add,
    npy_op_subtract,
    npy_op_multiply,
    npy_op_divide,
    npy_op_remainder,
    npy_op_power,
    npy_op_square,
    npy_op_reciprocal,
    npy_op_ones_like,
    npy_op_sqrt,
    npy_op_negative,
    npy_op_absolute,
    npy_op_invert,
    npy_op_left_shift,
    npy_op_right_shift,
    npy_op_bitwise_and,
    npy_op_bitwise_xor,
    npy_op_bitwise_or,
    npy_op_less,
    npy_op_less_equal,
    npy_op_equal,
    npy_op_not_equal,
    npy_op_greater,
    npy_op_greater_equal,
    npy_op_floor_divide,
    npy_op_true_divide,
    npy_op_logical_or,
    npy_op_logical_and,
    npy_op_floor,
    npy_op_ceil,
    npy_op_maximum,
    npy_op_minimum,
    npy_op_rint,
    npy_op_conjugate
};


typedef int (*npy_prepare_outputs_func)(NpyUFuncObject* self, NpyArray **mps,
                                        void* data);
NpyArray *
NpyArray_GenericBinaryFunction(NpyArray *m1, NpyArray *m2, NpyUFuncObject *op);

NpyArray *
NpyArray_GenericUnaryFunction(NpyArray *m1, NpyUFuncObject *op);


NpyUFuncObject *NpyArray_GetNumericOp(enum NpyArray_Ops);
int NpyArray_SetNumericOp(enum NpyArray_Ops, NpyUFuncObject *);
NpyUFuncObject *
    NpyUFunc_FromFuncAndData(NpyUFuncGenericFunction *func, void **data,
                             char *types, int ntypes,
                             int nin, int nout, int identity,
                             char *name, char *doc, int check_return);
NpyArray *
NpyUFunc_GenericReduction(NpyUFuncObject *self, NpyArray *arr, 
                          NpyArray *indicies, NpyArray *out, int axis, 
                          NpyArray_Descr *otype, int operation);

NpyArray *
NpyUFunc_Reduce(NpyUFuncObject *self, NpyArray *arr, NpyArray *out,
                int axis, int otype);
int NpyUFunc_GenericFunction(NpyUFuncObject *self, int nargs, NpyArray **mps,
                             int *rtypenums,
                             int originalArgWasObjArray,
                             npy_prepare_outputs_func prepare_output_func,
                             void *args);

NpyArray *
NpyUFunc_Accumulate(NpyUFuncObject *self, NpyArray *arr, NpyArray *out,
                    int axis, int otype);

NpyArray *
NpyUFunc_Reduceat(NpyUFuncObject *self, NpyArray *arr, NpyArray *ind,
                  NpyArray *out, int axis, int otype);

int
NpyUFunc_SetUsesArraysAsData(void **data, size_t i);
int
NpyUFunc_RegisterLoopForType(NpyUFuncObject *ufunc,
                             int usertype,
                             NpyUFuncGenericFunction function,
                             int *arg_types,
                             void *data);
NpyUFuncObject *
NpyUFunc_FromFuncAndData(NpyUFuncGenericFunction *func, void **data,
                         char *types, int ntypes,
                         int nin, int nout, int identity,
                         char *name, char *doc, int check_return);
NpyUFuncObject *
NpyUFunc_FromFuncAndDataAndSignature(NpyUFuncGenericFunction *func, void **data,
                                     char *types, int ntypes,
                                     int nin, int nout, int identity,
                                     char *name, char *doc,
                                     int check_return, const char *signature);
NpyUFuncObject *
npy_ufunc_frompyfunc(int nin, int nout, char *fname, size_t fname_len, 
                     NpyUFuncGenericFunction *gen_funcs, void *function);
void
npy_ufunc_dealloc(NpyUFuncObject *self);




/*
 * Platform-dependent floating point error handling
 */

void 
NpyUFunc_SetFpErrFuncs(void (*state)(char *, int *, int *, void **), 
                       void (*handler)(int, void *, int, int *));
int
NpyUFunc_getfperr(void);
int
NpyUFunc_checkfperr(int errmask, void *errobj, int *first);
void
NpyUFunc_clearfperr();


#define NpyUFunc_One 1
#define NpyUFunc_Zero 0
#define NpyUFunc_None -1

#define NPY_UFUNC_REDUCE 0
#define NPY_UFUNC_ACCUMULATE 1
#define NPY_UFUNC_REDUCEAT 2
#define NPY_UFUNC_OUTER 3



#define NPY_UFUNC_ERR_IGNORE 0
#define NPY_UFUNC_ERR_WARN   1
#define NPY_UFUNC_ERR_RAISE  2
#define NPY_UFUNC_ERR_CALL   3
#define NPY_UFUNC_ERR_PRINT  4
#define NPY_UFUNC_ERR_LOG    5

/* Python side integer mask */

#define NPY_UFUNC_MASK_DIVIDEBYZERO 0x07
#define NPY_UFUNC_MASK_OVERFLOW 0x3f
#define NPY_UFUNC_MASK_UNDERFLOW 0x1ff
#define NPY_UFUNC_MASK_INVALID 0xfff

#define NPY_UFUNC_SHIFT_DIVIDEBYZERO 0
#define NPY_UFUNC_SHIFT_OVERFLOW     3
#define NPY_UFUNC_SHIFT_UNDERFLOW    6
#define NPY_UFUNC_SHIFT_INVALID      9


/* platform-dependent code translates floating point
 status to an integer sum of these values
 */
#define NPY_UFUNC_FPE_DIVIDEBYZERO  1
#define NPY_UFUNC_FPE_OVERFLOW      2
#define NPY_UFUNC_FPE_UNDERFLOW     4
#define NPY_UFUNC_FPE_INVALID       8

#define NPY_UFUNC_ERR_DEFAULT 0 /* Error mode that avoids look-up (no checking) */

#define NPY_UFUNC_OBJ_ISOBJECT      1
#define NPY_UFUNC_OBJ_NEEDS_API     2

/* Default user error mode */
#define NPY_UFUNC_ERR_DEFAULT2                               \
        (NPY_UFUNC_ERR_PRINT << NPY_UFUNC_SHIFT_DIVIDEBYZERO) +  \
        (NPY_UFUNC_ERR_PRINT << NPY_UFUNC_SHIFT_OVERFLOW) +      \
        (NPY_UFUNC_ERR_PRINT << NPY_UFUNC_SHIFT_INVALID)




#define NPY_UFUNC_CHECK_ERROR(arg)                                           \
        do {if ((((arg)->obj & NPY_UFUNC_OBJ_NEEDS_API) && NpyErr_Occurred()) || \
            ((arg)->errormask &&                                             \
             NpyUFunc_checkfperr((arg)->errormask,                           \
                                (arg)->errobj,                               \
                                &(arg)->first)))                             \
                goto fail;} while (0)

/* This code checks the IEEE status flags in a platform-dependent way */
/* Adapted from Numarray  */

#if (defined(__unix__) || defined(unix)) && !defined(USG)
#include <sys/param.h>
#endif

/*  OSF/Alpha (Tru64)  ---------------------------------------------*/
#if defined(__osf__) && defined(__alpha)

#include <machine/fpu.h>

#define NPY_UFUNC_CHECK_STATUS(ret) {                                          \
        unsigned long fpstatus;                                                \
        fpstatus = ieee_get_fp_control();                                      \
        /* clear status bits as well as disable exception mode if on */        \
        ieee_set_fp_control( 0 );                                              \
        ret = ((IEEE_STATUS_DZE & fpstatus) ? NPY_UFUNC_FPE_DIVIDEBYZERO : 0)  \
                | ((IEEE_STATUS_OVF & fpstatus) ? NPY_UFUNC_FPE_OVERFLOW : 0)  \
                | ((IEEE_STATUS_UNF & fpstatus) ? NPY_UFUNC_FPE_UNDERFLOW : 0) \
                | ((IEEE_STATUS_INV & fpstatus) ? NPY_UFUNC_FPE_INVALID : 0);  \
        }

/* MS Windows -----------------------------------------------------*/
#elif defined(_MSC_VER)

#include <float.h>

  /* Clear the floating point exception default of Borland C++ */
#if defined(__BORLANDC__)
#define NPY_UFUNC_NOFPE _control87(MCW_EM, MCW_EM);
#endif

#define NPY_UFUNC_CHECK_STATUS(ret) {                                       \
        int fpstatus = (int) _clearfp();                                    \
        ret = ((SW_ZERODIVIDE & fpstatus) ? NPY_UFUNC_FPE_DIVIDEBYZERO : 0) \
                | ((SW_OVERFLOW & fpstatus) ? NPY_UFUNC_FPE_OVERFLOW : 0)   \
                | ((SW_UNDERFLOW & fpstatus) ? NPY_UFUNC_FPE_UNDERFLOW : 0) \
                | ((SW_INVALID & fpstatus) ? NPY_UFUNC_FPE_INVALID : 0);    \
        }

/* Solaris --------------------------------------------------------*/
/* --------ignoring SunOS ieee_flags approach, someone else can
**         deal with that! */
#elif defined(sun) || defined(__BSD__) || defined(__OpenBSD__) || \
  (defined(__FreeBSD__) && (__FreeBSD_version < 502114)) || defined(__NetBSD__)
#include <ieeefp.h>

#define NPY_UFUNC_CHECK_STATUS(ret) {                                       \
        int fpstatus;                                                   \
        fpstatus = (int) fpgetsticky();                                 \
        ret = ((FP_X_DZ  & fpstatus) ? NPY_UFUNC_FPE_DIVIDEBYZERO : 0)      \
                | ((FP_X_OFL & fpstatus) ? NPY_UFUNC_FPE_OVERFLOW : 0)      \
                | ((FP_X_UFL & fpstatus) ? NPY_UFUNC_FPE_UNDERFLOW : 0)     \
                | ((FP_X_INV & fpstatus) ? NPY_UFUNC_FPE_INVALID : 0);      \
        (void) fpsetsticky(0);                                          \
        }

#elif defined(__GLIBC__) || defined(__APPLE__) || defined(__CYGWIN__) || \
 defined(__MINGW32__) || (defined(__FreeBSD__) && (__FreeBSD_version >= 502114))

#if defined(__GLIBC__) || defined(__APPLE__) || defined(__MINGW32__) || \
  defined(__FreeBSD__)
#include <fenv.h>
#elif defined(__CYGWIN__)
#include "fenv/fenv.c"
#endif

#define NPY_UFUNC_CHECK_STATUS(ret) {                                        \
        int fpstatus = (int) fetestexcept(FE_DIVBYZERO | FE_OVERFLOW |   \
                                          FE_UNDERFLOW | FE_INVALID);    \
        ret = ((FE_DIVBYZERO  & fpstatus) ? NPY_UFUNC_FPE_DIVIDEBYZERO : 0)  \
                | ((FE_OVERFLOW   & fpstatus) ? NPY_UFUNC_FPE_OVERFLOW : 0)  \
                | ((FE_UNDERFLOW  & fpstatus) ? NPY_UFUNC_FPE_UNDERFLOW : 0) \
                | ((FE_INVALID    & fpstatus) ? NPY_UFUNC_FPE_INVALID : 0);  \
        (void) feclearexcept(FE_DIVBYZERO | FE_OVERFLOW |                \
                             FE_UNDERFLOW | FE_INVALID);                 \
}

#define generate_divbyzero_error() feraiseexcept(FE_DIVBYZERO)
#define generate_overflow_error() feraiseexcept(FE_OVERFLOW)

#elif defined(_AIX)

#include <float.h>
#include <fpxcp.h>

#define NPY_UFUNC_CHECK_STATUS(ret) {                                         \
        fpflag_t fpstatus;                                                \
                                                                          \
        fpstatus = fp_read_flag();                                        \
        ret = ((FP_DIV_BY_ZERO & fpstatus) ? NPY_UFUNC_FPE_DIVIDEBYZERO : 0)  \
                | ((FP_OVERFLOW & fpstatus) ? NPY_UFUNC_FPE_OVERFLOW : 0)     \
                | ((FP_UNDERFLOW & fpstatus) ? NPY_UFUNC_FPE_UNDERFLOW : 0)   \
                | ((FP_INVALID & fpstatus) ? NPY_UFUNC_FPE_INVALID : 0);      \
        fp_swap_flag(0);                                                  \
}

#define generate_divbyzero_error() fp_raise_xcp(FP_DIV_BY_ZERO)
#define generate_overflow_error() fp_raise_xcp(FP_OVERFLOW)

#else

#define NO_FLOATING_POINT_SUPPORT
#define NPY_UFUNC_CHECK_STATUS(ret) { ret = 0; }

#endif

/* These should really be altered to just set the corresponding bit
 in the floating point status flag.  Need to figure out how to do that
 on all the platforms...
 */

#if !defined(generate_divbyzero_error)
static int numeric_zero2 = 0;
static void generate_divbyzero_error(void)
{
    double dummy;

    dummy = 1./numeric_zero2;
    if (dummy) /* to prevent optimizer from eliminating expression */
        return;
    else /* should never be called */
        numeric_zero2 += 1;
    return;
}
#endif

#if !defined(generate_overflow_error)
static double numeric_two = 2.0;
static void generate_overflow_error(void)
{
    double dummy;

    dummy = pow(numeric_two, 1000.0);
    if (dummy)
        return;
    else
        numeric_two += 0.1;
    return;
}
#endif


#endif
