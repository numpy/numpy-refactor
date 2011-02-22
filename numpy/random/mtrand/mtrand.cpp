/* Cython code section 'h_code' */


#define PY_LONG_LONG long long

using namespace System::Collections;
using namespace System::Numerics;
using namespace System::Reflection;
using namespace System::Runtime::CompilerServices;
using namespace System::Runtime;
using namespace System::Security::Permissions;
using namespace System::Linq::Expressions;
using namespace Microsoft::Scripting::Actions;
using namespace Microsoft::Scripting::Runtime;
using namespace Microsoft::Scripting;
using namespace IronPython;
using namespace IronPython::Runtime;
using namespace IronPython::Runtime::Operations;

#define Py_None nullptr
typedef int Py_ssize_t; // IronPython uses "int" for sizes even on 64-bit platforms
#define PY_SSIZE_T_MAX 2147483647

enum class Markers { Default };

static CodeContext^ mk_empty_context(CodeContext^ ctx) {
  PythonDictionary^ dict = gcnew PythonDictionary;
  dict["__module__"] = "mtrand";
  return gcnew CodeContext(dict, ctx->ModuleContext);
}
  #define PyBUF_SIMPLE 0
  #define PyBUF_WRITABLE 0x0001
  #define PyBUF_FORMAT 0x0004
  #define PyBUF_ND 0x0008
  #define PyBUF_STRIDES (0x0010 | PyBUF_ND)
  #define PyBUF_C_CONTIGUOUS (0x0020 | PyBUF_STRIDES)
  #define PyBUF_F_CONTIGUOUS (0x0040 | PyBUF_STRIDES)
  #define PyBUF_ANY_CONTIGUOUS (0x0080 | PyBUF_STRIDES)
  #define PyBUF_INDIRECT (0x0100 | PyBUF_STRIDES)


/* inline attribute */
#ifndef CYTHON_INLINE
  #if defined(__GNUC__)
    #define CYTHON_INLINE __inline__
  #elif defined(_MSC_VER)
    #define CYTHON_INLINE __inline
  #elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define CYTHON_INLINE inline
  #else
    #define CYTHON_INLINE 
  #endif
#endif

/* unused attribute */
#ifndef CYTHON_UNUSED
# if defined(__GNUC__)
#   if !(defined(__cplusplus)) || (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#     define CYTHON_UNUSED __attribute__ ((__unused__)) 
#   else
#     define CYTHON_UNUSED
#   endif
# elif defined(__ICC) || defined(__INTEL_COMPILER)
#   define CYTHON_UNUSED __attribute__ ((__unused__)) 
# else
#   define CYTHON_UNUSED 
# endif
#endif
#ifdef __cplusplus
#define __PYX_EXTERN_C extern "C"
#else
#define __PYX_EXTERN_C extern
#endif

#if defined(WIN32) || defined(MS_WINDOWS)
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#define __PYX_HAVE_API__mtrand
#include "npy_defs.h"
#include "npy_arrayobject.h"
#include "npy_iterators.h"
#include "npy_common.h"
#include "npy_descriptor.h"
#include "stdio.h"
#include "string.h"
#include "memory.h"
#include "randomkit.h"
#include "distributions.h"
#include "initarray.h"

#ifdef __GNUC__
/* Test for GCC > 2.95 */
#if __GNUC__ > 2 || (__GNUC__ == 2 && (__GNUC_MINOR__ > 95)) 
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else /* __GNUC__ > 2 ... */
#define likely(x)   (x)
#define unlikely(x) (x)
#endif /* __GNUC__ > 2 ... */
#else /* __GNUC__ */
#define likely(x)   (x)
#define unlikely(x) (x)
#endif /* __GNUC__ */
    
static const char * __pyx_cfilenm= __FILE__;

/* Cython code section 'filename_table' */

static const char *__pyx_f[] = {
  0
};
/* Cython code section 'utility_code_proto_before_types' */
/* Cython code section 'numeric_typedefs' */
/* Cython code section 'complex_type_declarations' */
/* Cython code section 'type_declarations' */

/* Type declarations */

typedef double (*__pyx_t_6mtrand_rk_cont0)(rk_state *);

typedef double (*__pyx_t_6mtrand_rk_cont1)(rk_state *, double);

typedef double (*__pyx_t_6mtrand_rk_cont2)(rk_state *, double, double);

typedef double (*__pyx_t_6mtrand_rk_cont3)(rk_state *, double, double, double);

typedef long (*__pyx_t_6mtrand_rk_disc0)(rk_state *);

typedef long (*__pyx_t_6mtrand_rk_discnp)(rk_state *, long, double);

typedef long (*__pyx_t_6mtrand_rk_discdd)(rk_state *, double, double);

typedef long (*__pyx_t_6mtrand_rk_discnmN)(rk_state *, long, long, long);

typedef long (*__pyx_t_6mtrand_rk_discd)(rk_state *, double);
/* Cython code section 'utility_code_proto' */
/* Cython code section 'module_declarations' */
/* Module declarations from cpython.version */
/* Module declarations from cpython.ref */
/* Module declarations from cpython.exc */
/* Module declarations from cpython.module */
/* Module declarations from cpython.mem */
/* Module declarations from cpython.tuple */
/* Module declarations from cpython.list */
/* Module declarations from libc.stdio */
/* Module declarations from cpython.object */
/* Module declarations from cpython.sequence */
/* Module declarations from cpython.mapping */
/* Module declarations from cpython.iterator */
/* Module declarations from cpython.type */
/* Module declarations from cpython.number */
/* Module declarations from cpython.int */
/* Module declarations from cpython.bool */
/* Module declarations from cpython.long */
/* Module declarations from cpython.float */
/* Module declarations from cpython.complex */
/* Module declarations from cpython.string */
/* Module declarations from cpython.unicode */
/* Module declarations from cpython.dict */
/* Module declarations from cpython.instance */
/* Module declarations from cpython.function */
/* Module declarations from cpython.method */
/* Module declarations from cpython.weakref */
/* Module declarations from cpython.getargs */
/* Module declarations from cpython.pythread */
/* Module declarations from cpython.cobject */
/* Module declarations from cpython.oldbuffer */
/* Module declarations from cpython.set */
/* Module declarations from cpython.buffer */
/* Module declarations from cpython.bytes */
/* Module declarations from cpython.pycapsule */
/* Module declarations from cpython */
/* Module declarations from mtrand */
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate NpyArray *__pyx_delegate_t_6mtrand_npy_array_from_py_array(System::Object^);
static CYTHON_INLINE NpyArray *npy_array_from_py_array(System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate NpyArrayMultiIterObject *__pyx_delegate_t_6mtrand_npy_iter_from_py_iter(System::Object^);
static CYTHON_INLINE NpyArrayMultiIterObject *npy_iter_from_py_iter(System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate void *__pyx_delegate_t_6mtrand_dataptr(System::Object^);
static CYTHON_INLINE void *dataptr(System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate NpyArrayMultiIterObject *__pyx_delegate_t_6mtrand_getiter(System::Object^);
static CYTHON_INLINE NpyArrayMultiIterObject *getiter(System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_6mtrand_cont0_array(rk_state *, __pyx_t_6mtrand_rk_cont0, System::Object^);
static System::Object^ cont0_array(rk_state *, __pyx_t_6mtrand_rk_cont0, System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_6mtrand_cont1_array_sc(rk_state *, __pyx_t_6mtrand_rk_cont1, System::Object^, double);
static System::Object^ cont1_array_sc(rk_state *, __pyx_t_6mtrand_rk_cont1, System::Object^, double); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_6mtrand_cont1_array(rk_state *, __pyx_t_6mtrand_rk_cont1, System::Object^, System::Object^);
static System::Object^ cont1_array(rk_state *, __pyx_t_6mtrand_rk_cont1, System::Object^, System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_6mtrand_cont2_array_sc(rk_state *, __pyx_t_6mtrand_rk_cont2, System::Object^, double, double);
static System::Object^ cont2_array_sc(rk_state *, __pyx_t_6mtrand_rk_cont2, System::Object^, double, double); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_6mtrand_cont2_array(rk_state *, __pyx_t_6mtrand_rk_cont2, System::Object^, System::Object^, System::Object^);
static System::Object^ cont2_array(rk_state *, __pyx_t_6mtrand_rk_cont2, System::Object^, System::Object^, System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_6mtrand_cont3_array_sc(rk_state *, __pyx_t_6mtrand_rk_cont3, System::Object^, double, double, double);
static System::Object^ cont3_array_sc(rk_state *, __pyx_t_6mtrand_rk_cont3, System::Object^, double, double, double); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_6mtrand_cont3_array(rk_state *, __pyx_t_6mtrand_rk_cont3, System::Object^, System::Object^, System::Object^, System::Object^);
static System::Object^ cont3_array(rk_state *, __pyx_t_6mtrand_rk_cont3, System::Object^, System::Object^, System::Object^, System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_6mtrand_disc0_array(rk_state *, __pyx_t_6mtrand_rk_disc0, System::Object^);
static System::Object^ disc0_array(rk_state *, __pyx_t_6mtrand_rk_disc0, System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_6mtrand_discnp_array_sc(rk_state *, __pyx_t_6mtrand_rk_discnp, System::Object^, long, double);
static System::Object^ discnp_array_sc(rk_state *, __pyx_t_6mtrand_rk_discnp, System::Object^, long, double); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_6mtrand_discnp_array(rk_state *, __pyx_t_6mtrand_rk_discnp, System::Object^, System::Object^, System::Object^);
static System::Object^ discnp_array(rk_state *, __pyx_t_6mtrand_rk_discnp, System::Object^, System::Object^, System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_6mtrand_discdd_array_sc(rk_state *, __pyx_t_6mtrand_rk_discdd, System::Object^, double, double);
static System::Object^ discdd_array_sc(rk_state *, __pyx_t_6mtrand_rk_discdd, System::Object^, double, double); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_6mtrand_discdd_array(rk_state *, __pyx_t_6mtrand_rk_discdd, System::Object^, System::Object^, System::Object^);
static System::Object^ discdd_array(rk_state *, __pyx_t_6mtrand_rk_discdd, System::Object^, System::Object^, System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_6mtrand_discnmN_array_sc(rk_state *, __pyx_t_6mtrand_rk_discnmN, System::Object^, long, long, long);
static System::Object^ discnmN_array_sc(rk_state *, __pyx_t_6mtrand_rk_discnmN, System::Object^, long, long, long); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_6mtrand_discnmN_array(rk_state *, __pyx_t_6mtrand_rk_discnmN, System::Object^, System::Object^, System::Object^, System::Object^);
static System::Object^ discnmN_array(rk_state *, __pyx_t_6mtrand_rk_discnmN, System::Object^, System::Object^, System::Object^, System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_6mtrand_discd_array_sc(rk_state *, __pyx_t_6mtrand_rk_discd, System::Object^, double);
static System::Object^ discd_array_sc(rk_state *, __pyx_t_6mtrand_rk_discd, System::Object^, double); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_6mtrand_discd_array(rk_state *, __pyx_t_6mtrand_rk_discd, System::Object^, System::Object^);
static System::Object^ discd_array(rk_state *, __pyx_t_6mtrand_rk_discd, System::Object^, System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate double __pyx_delegate_t_6mtrand_kahan_sum(double *, long);
static double kahan_sum(double *, long); /*proto*/
/* Cython code section 'typeinfo' */
/* Cython code section 'before_global_var' */
#define __Pyx_MODULE_NAME "mtrand"

/* Implementation of mtrand */
namespace clr_mtrand {
  public ref class module_mtrand sealed abstract {
/* Cython code section 'global_var' */
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^ >^ >^ __site_call0_4297_19;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_seed_4298_12;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_get_state_4299_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_set_state_4300_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_random_sample_4301_21;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_randint_4302_15;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_bytes_4303_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_uniform_4304_15;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_rand_4305_12;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_randn_4306_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_random_integers_4307_23;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_standard_normal_4308_23;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_normal_4309_14;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_beta_4310_12;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_exponential_4311_19;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_standard_exponential_4312_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_standard_gamma_4313_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_gamma_4314_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_f_4315_9;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_noncentral_f_4316_20;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_chisquare_4317_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_noncentral_chisquare_4318_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_standard_cauchy_4319_23;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_standard_t_4320_18;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_vonmises_4321_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_pareto_4322_14;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_weibull_4323_15;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_power_4324_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_laplace_4325_15;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_gumbel_4326_14;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_logistic_4327_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_lognormal_4328_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_rayleigh_4329_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_wald_4330_12;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_triangular_4331_18;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_binomial_4333_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_negative_binomial_4334_25;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_poisson_4335_15;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_zipf_4336_12;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_geometric_4337_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_hypergeometric_4338_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_logseries_4339_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_multivariate_normal_4341_27;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_multinomial_4342_19;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_dirichlet_4343_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_shuffle_4345_15;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_permutation_4346_19;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Array_88_30;
static  CallSite< System::Func< CallSite^, System::Object^, PY_LONG_LONG >^ >^ __site_cvt_PY_LONG_LONG_88_30;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Iter_94_30;
static  CallSite< System::Func< CallSite^, System::Object^, PY_LONG_LONG >^ >^ __site_cvt_PY_LONG_LONG_94_30;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_141_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_141_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_141_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_142_20;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_142_20;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_157_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_157_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_157_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_158_20;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_158_20;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_172_11;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_172_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_172_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_like_174_16;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_174_27;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_175_20;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_175_20;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_182_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_182_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_182_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_broadcast_184_18;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_184_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_185_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_185_28;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ne_185_22;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_185_22;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_186_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_187_33;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_187_33;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_202_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_202_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_202_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_203_20;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_203_20;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_217_11;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_217_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_217_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_218_11;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_218_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_218_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_broadcast_220_18;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_220_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_221_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_shape_221_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_221_38;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_221_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_223_33;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_223_33;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_229_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_229_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_229_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_broadcast_231_18;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call3_231_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_232_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_232_28;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ne_232_22;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_232_22;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_233_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_234_33;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_234_33;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_251_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_251_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_251_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_252_20;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_252_20;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_267_11;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_267_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_267_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_268_11;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_268_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_268_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_269_11;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_269_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_269_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_broadcast_271_18;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call3_271_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_272_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_shape_272_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_272_38;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_272_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_274_33;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_274_33;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_281_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_281_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_281_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_broadcast_283_18;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call4_283_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_284_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_284_28;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ne_284_22;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_284_22;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_285_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_286_33;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_286_33;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_302_16;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_302_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_303_20;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_303_20;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_318_16;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_318_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_319_20;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_319_20;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_333_11;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_long_333_29;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_dtype_333_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_334_11;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_334_29;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_dtype_334_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_broadcast_336_18;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_336_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_337_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_shape_337_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_long_337_38;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_337_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_339_33;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_339_33;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_345_16;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_345_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_broadcast_347_18;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call3_347_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_348_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_348_28;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ne_348_22;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_348_22;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_349_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_350_33;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_350_33;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_367_16;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_367_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_368_20;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_368_20;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_382_11;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_382_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_382_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_383_11;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_383_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_383_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_broadcast_385_18;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_385_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_386_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_shape_386_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_long_386_38;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_386_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_388_33;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_388_33;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_394_16;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_394_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_broadcast_396_18;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call3_396_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_397_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_397_28;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ne_397_22;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_397_22;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_398_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_399_33;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_399_33;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_416_16;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_416_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_417_20;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_417_20;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_432_11;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_long_432_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_432_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_433_11;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_long_433_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_433_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_434_11;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_long_434_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_434_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_broadcast_436_18;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call3_436_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_437_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_shape_437_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_long_437_38;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_437_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_439_33;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_439_33;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_446_16;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_446_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_broadcast_448_18;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call4_448_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_449_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_449_28;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ne_449_22;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_449_22;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_450_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_451_33;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_451_33;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_468_16;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_468_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_469_20;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_469_20;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_483_11;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_483_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_483_17;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_485_18;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_shape_485_27;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_long_485_43;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_dtype_485_24;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_486_22;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_486_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_493_16;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_493_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_broadcast_495_18;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_495_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_496_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_496_28;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ne_496_22;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_496_22;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_497_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_498_33;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_498_33;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_520_10;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_dtype_520_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_shape_521_12;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_521_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_eq_521_20;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_521_20;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_flatten_524_16;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^ >^ >^ __site_call0_524_24;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_seed_565_12;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_565_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_595_17;
static  CallSite< System::Func< CallSite^, System::Object^, unsigned long >^ >^ __site_cvt_unsigned_long_596_24;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_integer_597_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_597_23;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_597_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_598_23;
static  CallSite< System::Func< CallSite^, System::Object^, unsigned long >^ >^ __site_cvt_unsigned_long_599_25;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_long_601_37;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_601_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_636_18;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_uint_636_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_636_24;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_asarray_640_18;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_uint32_640_36;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_640_26;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_693_30;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ne_694_26;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_694_26;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_695_28;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getslice_696_24;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_int_696_8;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_697_14;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_eq_697_22;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_697_22;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getslice_701_46;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_uint_703_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_703_24;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_shape_704_14;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_704_20;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ne_704_24;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_704_24;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_705_28;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_int_710_49;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_711_51;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_get_state_715_19;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^ >^ >^ __site_call0_715_29;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_set_state_718_12;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_718_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_random_721_18;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get___RandomState_ctor_721_25;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_get_state_721_54;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^ >^ >^ __site_call0_721_64;
static  CallSite< System::Func< CallSite^, System::Object^, long >^ >^ __site_cvt_long_851_20;
static  CallSite< System::Func< CallSite^, System::Object^, long >^ >^ __site_cvt_long_853_20;
static  CallSite< System::Func< CallSite^, System::Object^, long >^ >^ __site_cvt_long_854_21;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_858_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_863_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_863_26;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_864_24;
static  CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >^ __site_cvt_npy_intp_864_24;
static  CallSite< System::Func< CallSite^, System::Object^, unsigned int >^ >^ __site_cvt_870_4;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_mul_892_35;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_974_30;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_975_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_984_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_984_23;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_984_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_984_40;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_sub_984_30;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1026_14;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_eq_1026_21;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_1026_21;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_random_sample_1027_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^ >^ >^ __site_call0_1027_37;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_random_sample_1029_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call0_size_1029_37;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1082_14;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_eq_1082_21;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_1082_21;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_standard_normal_1083_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^ >^ >^ __site_call0_1083_39;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_standard_normal_1085_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1085_39;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_randint_1163_19;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_add_1163_38;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call3_1163_27;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_1284_30;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_1285_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1292_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_1296_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_1296_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_1296_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1296_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_1296_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1297_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_1341_26;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_1342_26;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1349_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1351_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_1354_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_1354_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_1354_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1354_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_1354_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1355_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_1356_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_1356_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_1356_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1356_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_1356_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1357_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_1402_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1409_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_1413_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_1413_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_1413_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1413_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_1413_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1414_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_1516_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1523_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_1527_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_1527_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_1527_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1527_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_1527_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1528_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_1605_34;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_1606_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1613_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1615_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_1619_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_1619_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_1619_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1619_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_1619_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1620_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_1621_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_1621_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_1621_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1621_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_1621_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1622_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_1710_34;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_1711_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1718_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1720_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_1724_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_1724_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_1724_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1724_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_1724_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1725_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_1726_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_1726_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_1726_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1726_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_1726_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1727_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_1798_34;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_1799_34;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_1800_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1807_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1809_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1811_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_1815_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_1815_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_1815_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1815_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_1815_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1816_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_1817_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_1817_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_1817_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1817_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_1817_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1818_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_1819_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_1819_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_1819_25;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1819_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_1819_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1820_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_1892_28;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1899_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_1902_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_1902_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_1902_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1902_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_1902_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1903_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_1979_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_1980_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1987_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1989_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_1993_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_1993_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_1993_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1993_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_1993_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1994_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_1995_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_1995_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_1995_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1995_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_1995_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_1996_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_2150_28;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2157_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_2160_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_2160_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_2160_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2160_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_2160_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2161_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_2246_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_2247_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2254_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_2258_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_2258_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_2258_25;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2258_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_2258_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2259_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_2339_26;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2346_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_2349_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_2349_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_2349_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2349_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_2349_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2350_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_2441_26;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2448_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_2451_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_2451_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_2451_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2451_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_2451_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2452_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_2552_26;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2559_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_2562_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_2562_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_2562_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2562_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_2562_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2563_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_2643_30;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_2644_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2651_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_2655_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_2655_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_2655_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2655_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_2655_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2656_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_2771_30;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_2772_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2779_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_2783_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_2783_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_2783_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2783_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_2783_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2784_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_2862_30;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_2863_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2870_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_2874_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_2874_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_2874_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2874_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_2874_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_2875_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_2994_32;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_2995_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3002_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3006_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_3006_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3006_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3006_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3006_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3007_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_3070_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3077_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3081_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_3081_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3081_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3081_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3081_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3082_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_3152_32;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_3153_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3160_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3162_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3166_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_3166_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3166_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3166_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3166_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3167_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3168_15;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_3168_22;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3168_33;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3168_19;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3168_19;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3169_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_3235_32;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_3236_34;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_3237_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3244_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3246_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3248_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3252_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_greater_3252_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3252_28;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3252_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3252_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3253_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3254_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_greater_3254_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3254_28;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3254_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3254_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3255_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3256_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_equal_3256_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3256_26;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3256_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3256_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3257_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_3348_26;
static  CallSite< System::Func< CallSite^, System::Object^, long >^ >^ __site_cvt_long_3349_24;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3356_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3358_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3360_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3364_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_3364_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3364_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3364_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3364_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3365_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3366_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_3366_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3366_25;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3366_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3366_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3367_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3368_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_greater_3368_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3368_28;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3368_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3368_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3369_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_3443_26;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_3444_26;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3451_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3453_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3455_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3459_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_3459_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3459_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3459_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3459_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3460_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3461_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_3461_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3461_25;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3461_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3461_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3462_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3463_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_greater_3463_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3463_28;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3463_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3463_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3464_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_3520_30;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_lt_3526_19;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3526_19;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3527_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3530_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_3530_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3530_25;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3530_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3530_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3531_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_3614_26;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3621_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3624_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_3624_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3624_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3624_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3624_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3625_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_3677_26;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3684_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3686_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3689_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_3689_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3689_25;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3689_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3689_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3690_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3691_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_greater_3691_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3691_28;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3691_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3691_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3692_28;
static  CallSite< System::Func< CallSite^, System::Object^, long >^ >^ __site_cvt_long_3783_32;
static  CallSite< System::Func< CallSite^, System::Object^, long >^ >^ __site_cvt_long_3784_30;
static  CallSite< System::Func< CallSite^, System::Object^, long >^ >^ __site_cvt_long_3785_36;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_lt_3791_21;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3791_21;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3792_32;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_lt_3793_20;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3793_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3794_32;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_lt_3795_23;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3795_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3796_32;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_add_3797_21;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_lt_3797_28;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3797_28;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3798_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3802_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_3802_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3802_25;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3802_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3802_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3803_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3804_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_3804_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3804_25;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3804_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3804_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3805_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3806_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_3806_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3806_25;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3806_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3806_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3807_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3808_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_3808_20;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_add_3808_28;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3808_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3808_25;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3808_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3808_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3809_28;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_double_3892_26;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3899_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3901_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3904_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_less_equal_3904_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3904_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3904_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3904_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3905_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_any_3906_13;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_greater_equal_3906_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_3906_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3906_17;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_3906_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_3907_28;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_4004_17;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4004_23;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_4005_16;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4005_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_shape_4010_19;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4010_14;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ne_4010_27;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_4010_27;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4011_31;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_shape_4012_19;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4012_15;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ne_4012_27;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_4012_27;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_shape_4012_40;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_4012_46;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_shape_4012_56;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_4012_62;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ne_4012_50;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_4012_50;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4013_31;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_shape_4014_15;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_4014_21;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_shape_4014_31;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_4014_37;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ne_4014_25;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_4014_25;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4015_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_4017_21;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_4017_21;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getslice_4019_32;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4019_26;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_append_4020_19;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_shape_4020_31;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_4020_37;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4020_26;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_standard_normal_4024_16;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_multiply_4024_35;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_reduce_4024_44;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4024_51;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4024_32;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_multiply_4025_21;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_reduce_4025_30;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4025_55;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_sub_4025_69;
static  CallSite< System::Func< CallSite^, System::Object^, Py_ssize_t >^ >^ __site_cvt_Py_ssize_t_4025_69;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getslice_4025_49;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4025_37;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_shape_4026_23;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_4026_29;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_set_shape_4025_9;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4036_21;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_dot_4037_14;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_sqrt_4037_25;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4037_30;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_mul_4037_21;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_4037_18;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_add_4040_10;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call3_4040_14;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4041_23;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_set_shape_4041_9;
static  CallSite< System::Func< CallSite^, System::Object^, long >^ >^ __site_cvt_4044_4;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4102_15;
static  CallSite< System::Func< CallSite^, System::Object^, long >^ >^ __site_cvt_long_4102_15;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_4103_35;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_4103_25;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4107_28;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4111_17;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_add_4114_25;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_zeros_4116_19;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_4116_25;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_4119_24;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_lt_4119_16;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_4119_16;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4196_15;
static  CallSite< System::Func< CallSite^, System::Object^, long >^ >^ __site_cvt_long_4196_15;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_4197_40;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call4_4197_30;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4202_17;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_add_4205_25;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_zeros_4207_18;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_double_4207_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_4207_24;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_size_4211_23;
static  CallSite< System::Func< CallSite^, System::Object^, long >^ >^ __site_cvt_long_4211_23;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4236_15;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_sub_4236_19;
static  CallSite< System::Func< CallSite^, System::Object^, long >^ >^ __site_cvt_long_4236_19;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_4238_21;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4238_19;
static  CallSite< System::Func< CallSite^, System::Object^, long >^ >^ __site_cvt_long_4238_19;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_4246_30;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_4246_36;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_4246_17;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_4246_23;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_4250_28;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_4250_26;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_int_4250_26;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_4254_34;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_copy_4254_37;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^ >^ >^ __site_call0_4254_42;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_4254_47;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_copy_4254_50;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^ >^ >^ __site_call0_4254_55;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_4254_21;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_4254_27;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_4259_34;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getslice_4259_37;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_4259_43;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getslice_4259_46;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_4259_21;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_4259_27;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_integer_4289_33;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_4289_21;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_4289_21;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_arange_4290_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4290_27;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_array_4292_20;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4292_26;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_shuffle_4293_12;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_4293_20;
static CodeContext^ __pyx_context;
/* Cython code section 'dotnet_globals' */




































static Types::PythonType^ __pyx_ptype_6mtrand_RandomState = nullptr;
/* Cython code section 'decls' */
static int^ __pyx_int_0;
static int^ __pyx_int_1;
static int^ __pyx_int_2;
static int^ __pyx_int_3;
static int^ __pyx_int_624;
static System::Object^ __pyx_k_1;
static System::Object^ __pyx_k_2;
static System::Object^ __pyx_k_3;
static System::Object^ __pyx_k_4;
static System::Object^ __pyx_k_5;
static System::Object^ __pyx_k_6;
static System::Object^ __pyx_k_7;
static System::Object^ __pyx_k_8;
static System::Object^ __pyx_k_9;
static System::Object^ __pyx_k_10;
static System::Object^ __pyx_k_11;
static System::Object^ __pyx_k_12;
static System::Object^ __pyx_k_13;
static System::Object^ __pyx_k_14;
static System::Object^ __pyx_k_15;
static System::Object^ __pyx_k_16;
/* Cython code section 'all_the_rest' */
public:
static System::String^ __module__ = __Pyx_MODULE_NAME;

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\numpy.pxi":85
 * 
 * IF CYTHON_BACKEND == "IronPython":
 *     cdef inline NpyArray *npy_array_from_py_array(x):             # <<<<<<<<<<<<<<
 *         # XXX This should have type-checking on 'x'
 *         # XXX "long long" is wrong type
 */

static CYTHON_INLINE NpyArray *npy_array_from_py_array(System::Object^ __pyx_v_x) {
  PY_LONG_LONG __pyx_v_ptr;
  NpyArray *__pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  PY_LONG_LONG __pyx_t_2;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\numpy.pxi":88
 *         # XXX This should have type-checking on 'x'
 *         # XXX "long long" is wrong type
 *         cdef long long ptr = x.Array             # <<<<<<<<<<<<<<
 *         return <NpyArray*>ptr
 * 
 */
  __pyx_t_1 = __site_get_Array_88_30->Target(__site_get_Array_88_30, __pyx_v_x, __pyx_context);
  __pyx_t_2 = __site_cvt_PY_LONG_LONG_88_30->Target(__site_cvt_PY_LONG_LONG_88_30, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_v_ptr = __pyx_t_2;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\numpy.pxi":89
 *         # XXX "long long" is wrong type
 *         cdef long long ptr = x.Array
 *         return <NpyArray*>ptr             # <<<<<<<<<<<<<<
 * 
 *     cdef inline NpyArrayMultiIterObject *npy_iter_from_py_iter(x):
 */
  __pyx_r = ((NpyArray *)__pyx_v_ptr);
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\numpy.pxi":91
 *         return <NpyArray*>ptr
 * 
 *     cdef inline NpyArrayMultiIterObject *npy_iter_from_py_iter(x):             # <<<<<<<<<<<<<<
 *         # XXX This should have type-checking on 'x'
 *         # XXX "long long" is wrong type
 */

static CYTHON_INLINE NpyArrayMultiIterObject *npy_iter_from_py_iter(System::Object^ __pyx_v_x) {
  PY_LONG_LONG __pyx_v_ptr;
  NpyArrayMultiIterObject *__pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  PY_LONG_LONG __pyx_t_2;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\numpy.pxi":94
 *         # XXX This should have type-checking on 'x'
 *         # XXX "long long" is wrong type
 *         cdef long long ptr = x.Iter             # <<<<<<<<<<<<<<
 *         return <NpyArrayMultiIterObject*>ptr
 */
  __pyx_t_1 = __site_get_Iter_94_30->Target(__site_get_Iter_94_30, __pyx_v_x, __pyx_context);
  __pyx_t_2 = __site_cvt_PY_LONG_LONG_94_30->Target(__site_cvt_PY_LONG_LONG_94_30, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_v_ptr = __pyx_t_2;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\numpy.pxi":95
 *         # XXX "long long" is wrong type
 *         cdef long long ptr = x.Iter
 *         return <NpyArrayMultiIterObject*>ptr             # <<<<<<<<<<<<<<
 */
  __pyx_r = ((NpyArrayMultiIterObject *)__pyx_v_ptr);
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":127
 * 
 * 
 * cdef inline void *dataptr(object arr):             # <<<<<<<<<<<<<<
 *     return npy_array_from_py_array(arr).data
 * 
 */

static CYTHON_INLINE void *dataptr(System::Object^ __pyx_v_arr) {
  void *__pyx_r;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":128
 * 
 * cdef inline void *dataptr(object arr):
 *     return npy_array_from_py_array(arr).data             # <<<<<<<<<<<<<<
 * 
 * cdef inline NpyArrayMultiIterObject *getiter(object multi):
 */
  __pyx_r = npy_array_from_py_array(__pyx_v_arr)->data;
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":130
 *     return npy_array_from_py_array(arr).data
 * 
 * cdef inline NpyArrayMultiIterObject *getiter(object multi):             # <<<<<<<<<<<<<<
 *     return npy_iter_from_py_iter(multi)
 * 
 */

static CYTHON_INLINE NpyArrayMultiIterObject *getiter(System::Object^ __pyx_v_multi) {
  NpyArrayMultiIterObject *__pyx_r;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":131
 * 
 * cdef inline NpyArrayMultiIterObject *getiter(object multi):
 *     return npy_iter_from_py_iter(multi)             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = npy_iter_from_py_iter(__pyx_v_multi);
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":134
 * 
 * 
 * cdef object cont0_array(rk_state *state, rk_cont0 func, object size):             # <<<<<<<<<<<<<<
 *     cdef double *data
 *     cdef npy_intp length, i
 */

static  System::Object^ cont0_array(rk_state *__pyx_v_state, __pyx_t_6mtrand_rk_cont0 __pyx_v_func, System::Object^ __pyx_v_size) {
  double *__pyx_v_data;
  npy_intp __pyx_v_length;
  npy_intp __pyx_v_i;
  System::Object^ __pyx_v_arr;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  System::Object^ __pyx_t_4 = nullptr;
  npy_intp __pyx_t_5;
  npy_intp __pyx_t_6;
  __pyx_v_arr = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":138
 *     cdef npy_intp length, i
 * 
 *     if size is None:             # <<<<<<<<<<<<<<
 *         return func(state)
 *     else:
 */
  __pyx_t_1 = (__pyx_v_size == nullptr);
  if (__pyx_t_1) {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":139
 * 
 *     if size is None:
 *         return func(state)             # <<<<<<<<<<<<<<
 *     else:
 *         arr = np.empty(size, np.double)
 */
    __pyx_t_2 = __pyx_v_func(__pyx_v_state);
    __pyx_r = __pyx_t_2;
    __pyx_t_2 = nullptr;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  /*else*/ {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":141
 *         return func(state)
 *     else:
 *         arr = np.empty(size, np.double)             # <<<<<<<<<<<<<<
 *         length = arr.size
 *         data = <double *>dataptr(arr)
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_empty_141_16->Target(__site_get_empty_141_16, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_4 = __site_get_double_141_31->Target(__site_get_double_141_31, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = __site_call2_141_22->Target(__site_call2_141_22, __pyx_context, __pyx_t_3, __pyx_v_size, __pyx_t_4);
    __pyx_t_3 = nullptr;
    __pyx_t_4 = nullptr;
    __pyx_v_arr = __pyx_t_2;
    __pyx_t_2 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":142
 *     else:
 *         arr = np.empty(size, np.double)
 *         length = arr.size             # <<<<<<<<<<<<<<
 *         data = <double *>dataptr(arr)
 *         for i from 0 <= i < length:
 */
    __pyx_t_2 = __site_get_size_142_20->Target(__site_get_size_142_20, __pyx_v_arr, __pyx_context);
    __pyx_t_5 = __site_cvt_npy_intp_142_20->Target(__site_cvt_npy_intp_142_20, __pyx_t_2);
    __pyx_t_2 = nullptr;
    __pyx_v_length = __pyx_t_5;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":143
 *         arr = np.empty(size, np.double)
 *         length = arr.size
 *         data = <double *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < length:
 *             data[i] = func(state)
 */
    __pyx_v_data = ((double *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":144
 *         length = arr.size
 *         data = <double *>dataptr(arr)
 *         for i from 0 <= i < length:             # <<<<<<<<<<<<<<
 *             data[i] = func(state)
 *         return arr
 */
    __pyx_t_6 = __pyx_v_length;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_6; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":145
 *         data = <double *>dataptr(arr)
 *         for i from 0 <= i < length:
 *             data[i] = func(state)             # <<<<<<<<<<<<<<
 *         return arr
 * 
 */
      (__pyx_v_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":146
 *         for i from 0 <= i < length:
 *             data[i] = func(state)
 *         return arr             # <<<<<<<<<<<<<<
 * 
 * 
 */
    __pyx_r = __pyx_v_arr;
    goto __pyx_L0;
  }
  __pyx_L3:;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":149
 * 
 * 
 * cdef object cont1_array_sc(rk_state *state, rk_cont1 func, object size,             # <<<<<<<<<<<<<<
 *                            double a):
 *     cdef double *data
 */

static  System::Object^ cont1_array_sc(rk_state *__pyx_v_state, __pyx_t_6mtrand_rk_cont1 __pyx_v_func, System::Object^ __pyx_v_size, double __pyx_v_a) {
  double *__pyx_v_data;
  npy_intp __pyx_v_length;
  npy_intp __pyx_v_i;
  System::Object^ __pyx_v_arr;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  System::Object^ __pyx_t_4 = nullptr;
  npy_intp __pyx_t_5;
  npy_intp __pyx_t_6;
  __pyx_v_arr = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":154
 *     cdef npy_intp length, i
 * 
 *     if size is None:             # <<<<<<<<<<<<<<
 *         return func(state, a)
 *     else:
 */
  __pyx_t_1 = (__pyx_v_size == nullptr);
  if (__pyx_t_1) {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":155
 * 
 *     if size is None:
 *         return func(state, a)             # <<<<<<<<<<<<<<
 *     else:
 *         arr = np.empty(size, np.double)
 */
    __pyx_t_2 = __pyx_v_func(__pyx_v_state, __pyx_v_a);
    __pyx_r = __pyx_t_2;
    __pyx_t_2 = nullptr;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  /*else*/ {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":157
 *         return func(state, a)
 *     else:
 *         arr = np.empty(size, np.double)             # <<<<<<<<<<<<<<
 *         length = arr.size
 *         data = <double *>dataptr(arr)
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_empty_157_16->Target(__site_get_empty_157_16, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_4 = __site_get_double_157_31->Target(__site_get_double_157_31, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = __site_call2_157_22->Target(__site_call2_157_22, __pyx_context, __pyx_t_3, __pyx_v_size, __pyx_t_4);
    __pyx_t_3 = nullptr;
    __pyx_t_4 = nullptr;
    __pyx_v_arr = __pyx_t_2;
    __pyx_t_2 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":158
 *     else:
 *         arr = np.empty(size, np.double)
 *         length = arr.size             # <<<<<<<<<<<<<<
 *         data = <double *>dataptr(arr)
 *         for i from 0 <= i < length:
 */
    __pyx_t_2 = __site_get_size_158_20->Target(__site_get_size_158_20, __pyx_v_arr, __pyx_context);
    __pyx_t_5 = __site_cvt_npy_intp_158_20->Target(__site_cvt_npy_intp_158_20, __pyx_t_2);
    __pyx_t_2 = nullptr;
    __pyx_v_length = __pyx_t_5;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":159
 *         arr = np.empty(size, np.double)
 *         length = arr.size
 *         data = <double *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < length:
 *             data[i] = func(state, a)
 */
    __pyx_v_data = ((double *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":160
 *         length = arr.size
 *         data = <double *>dataptr(arr)
 *         for i from 0 <= i < length:             # <<<<<<<<<<<<<<
 *             data[i] = func(state, a)
 *         return arr
 */
    __pyx_t_6 = __pyx_v_length;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_6; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":161
 *         data = <double *>dataptr(arr)
 *         for i from 0 <= i < length:
 *             data[i] = func(state, a)             # <<<<<<<<<<<<<<
 *         return arr
 * 
 */
      (__pyx_v_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, __pyx_v_a);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":162
 *         for i from 0 <= i < length:
 *             data[i] = func(state, a)
 *         return arr             # <<<<<<<<<<<<<<
 * 
 * 
 */
    __pyx_r = __pyx_v_arr;
    goto __pyx_L0;
  }
  __pyx_L3:;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":165
 * 
 * 
 * cdef object cont1_array(rk_state *state, rk_cont1 func, object size,             # <<<<<<<<<<<<<<
 *                         object a):
 *     cdef double *arr_data
 */

static  System::Object^ cont1_array(rk_state *__pyx_v_state, __pyx_t_6mtrand_rk_cont1 __pyx_v_func, System::Object^ __pyx_v_size, System::Object^ __pyx_v_a) {
  double *__pyx_v_arr_data;
  double *__pyx_v_oa_data;
  npy_intp __pyx_v_length;
  npy_intp __pyx_v_i;
  NpyArrayIterObject *__pyx_v_itera;
  System::Object^ __pyx_v_oa;
  System::Object^ __pyx_v_arr;
  System::Object^ __pyx_v_multi;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  int __pyx_t_4;
  npy_intp __pyx_t_5;
  npy_intp __pyx_t_6;
  __pyx_v_oa = nullptr;
  __pyx_v_arr = nullptr;
  __pyx_v_multi = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":172
 *     cdef NpyArrayIterObject *itera
 * 
 *     oa = np.array(a, np.double)             # <<<<<<<<<<<<<<
 *     if size is None:
 *         arr = np.empty_like(oa)
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_2 = __site_get_array_172_11->Target(__site_get_array_172_11, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_3 = __site_get_double_172_23->Target(__site_get_double_172_23, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call2_172_17->Target(__site_call2_172_17, __pyx_context, __pyx_t_2, __pyx_v_a, __pyx_t_3);
  __pyx_t_2 = nullptr;
  __pyx_t_3 = nullptr;
  __pyx_v_oa = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":173
 * 
 *     oa = np.array(a, np.double)
 *     if size is None:             # <<<<<<<<<<<<<<
 *         arr = np.empty_like(oa)
 *         length = arr.size
 */
  __pyx_t_4 = (__pyx_v_size == nullptr);
  if (__pyx_t_4) {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":174
 *     oa = np.array(a, np.double)
 *     if size is None:
 *         arr = np.empty_like(oa)             # <<<<<<<<<<<<<<
 *         length = arr.size
 *         arr_data = <double *>dataptr(arr)
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_empty_like_174_16->Target(__site_get_empty_like_174_16, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call1_174_27->Target(__site_call1_174_27, __pyx_context, __pyx_t_3, __pyx_v_oa);
    __pyx_t_3 = nullptr;
    __pyx_v_arr = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":175
 *     if size is None:
 *         arr = np.empty_like(oa)
 *         length = arr.size             # <<<<<<<<<<<<<<
 *         arr_data = <double *>dataptr(arr)
 *         itera = NpyArray_IterNew(npy_array_from_py_array(oa))
 */
    __pyx_t_1 = __site_get_size_175_20->Target(__site_get_size_175_20, __pyx_v_arr, __pyx_context);
    __pyx_t_5 = __site_cvt_npy_intp_175_20->Target(__site_cvt_npy_intp_175_20, __pyx_t_1);
    __pyx_t_1 = nullptr;
    __pyx_v_length = __pyx_t_5;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":176
 *         arr = np.empty_like(oa)
 *         length = arr.size
 *         arr_data = <double *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         itera = NpyArray_IterNew(npy_array_from_py_array(oa))
 *         for i from 0 <= i < length:
 */
    __pyx_v_arr_data = ((double *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":177
 *         length = arr.size
 *         arr_data = <double *>dataptr(arr)
 *         itera = NpyArray_IterNew(npy_array_from_py_array(oa))             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < length:
 *             arr_data[i] = func(state, (<double *>(itera.dataptr))[0])
 */
    __pyx_v_itera = NpyArray_IterNew(npy_array_from_py_array(__pyx_v_oa));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":178
 *         arr_data = <double *>dataptr(arr)
 *         itera = NpyArray_IterNew(npy_array_from_py_array(oa))
 *         for i from 0 <= i < length:             # <<<<<<<<<<<<<<
 *             arr_data[i] = func(state, (<double *>(itera.dataptr))[0])
 *             NpyArray_ITER_NEXT(itera)
 */
    __pyx_t_6 = __pyx_v_length;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_6; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":179
 *         itera = NpyArray_IterNew(npy_array_from_py_array(oa))
 *         for i from 0 <= i < length:
 *             arr_data[i] = func(state, (<double *>(itera.dataptr))[0])             # <<<<<<<<<<<<<<
 *             NpyArray_ITER_NEXT(itera)
 *     else:
 */
      (__pyx_v_arr_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, (((double *)__pyx_v_itera->dataptr)[0]));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":180
 *         for i from 0 <= i < length:
 *             arr_data[i] = func(state, (<double *>(itera.dataptr))[0])
 *             NpyArray_ITER_NEXT(itera)             # <<<<<<<<<<<<<<
 *     else:
 *         arr = np.empty(size, np.double)
 */
      NpyArray_ITER_NEXT(__pyx_v_itera);
    }
    goto __pyx_L3;
  }
  /*else*/ {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":182
 *             NpyArray_ITER_NEXT(itera)
 *     else:
 *         arr = np.empty(size, np.double)             # <<<<<<<<<<<<<<
 *         arr_data = <double *>dataptr(arr)
 *         multi = np.broadcast(arr, oa)
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_empty_182_16->Target(__site_get_empty_182_16, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_2 = __site_get_double_182_31->Target(__site_get_double_182_31, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call2_182_22->Target(__site_call2_182_22, __pyx_context, __pyx_t_3, __pyx_v_size, __pyx_t_2);
    __pyx_t_3 = nullptr;
    __pyx_t_2 = nullptr;
    __pyx_v_arr = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":183
 *     else:
 *         arr = np.empty(size, np.double)
 *         arr_data = <double *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         multi = np.broadcast(arr, oa)
 *         if multi.size != arr.size:
 */
    __pyx_v_arr_data = ((double *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":184
 *         arr = np.empty(size, np.double)
 *         arr_data = <double *>dataptr(arr)
 *         multi = np.broadcast(arr, oa)             # <<<<<<<<<<<<<<
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_2 = __site_get_broadcast_184_18->Target(__site_get_broadcast_184_18, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call2_184_28->Target(__site_call2_184_28, __pyx_context, __pyx_t_2, __pyx_v_arr, __pyx_v_oa);
    __pyx_t_2 = nullptr;
    __pyx_v_multi = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":185
 *         arr_data = <double *>dataptr(arr)
 *         multi = np.broadcast(arr, oa)
 *         if multi.size != arr.size:             # <<<<<<<<<<<<<<
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:
 */
    __pyx_t_1 = __site_get_size_185_16->Target(__site_get_size_185_16, __pyx_v_multi, __pyx_context);
    __pyx_t_2 = __site_get_size_185_28->Target(__site_get_size_185_28, __pyx_v_arr, __pyx_context);
    __pyx_t_3 = __site_op_ne_185_22->Target(__site_op_ne_185_22, __pyx_t_1, __pyx_t_2);
    __pyx_t_1 = nullptr;
    __pyx_t_2 = nullptr;
    __pyx_t_4 = __site_istrue_185_22->Target(__site_istrue_185_22, __pyx_t_3);
    __pyx_t_3 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":186
 *         multi = np.broadcast(arr, oa)
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < multi.size:
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(iter), 1)
 */
      __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_2 = __site_call1_186_28->Target(__site_call1_186_28, __pyx_context, __pyx_t_3, ((System::Object^)"size is not compatible with inputs"));
      __pyx_t_3 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_2, nullptr, nullptr);
      __pyx_t_2 = nullptr;
      goto __pyx_L6;
    }
    __pyx_L6:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":187
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:             # <<<<<<<<<<<<<<
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(iter), 1)
 *             arr_data[i] = func(state, oa_data[0])
 */
    __pyx_t_2 = __site_get_size_187_33->Target(__site_get_size_187_33, __pyx_v_multi, __pyx_context);
    __pyx_t_6 = __site_cvt_npy_intp_187_33->Target(__site_cvt_npy_intp_187_33, __pyx_t_2);
    __pyx_t_2 = nullptr;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_6; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":188
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(iter), 1)             # <<<<<<<<<<<<<<
 *             arr_data[i] = func(state, oa_data[0])
 *             NpyArray_MultiIter_NEXTi(getiter(iter), 1)
 */
      __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "iter");
      __pyx_v_oa_data = ((double *)NpyArray_MultiIter_DATA(getiter(__pyx_t_2), 1));
      __pyx_t_2 = nullptr;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":189
 *         for i from 0 <= i < multi.size:
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(iter), 1)
 *             arr_data[i] = func(state, oa_data[0])             # <<<<<<<<<<<<<<
 *             NpyArray_MultiIter_NEXTi(getiter(iter), 1)
 *     return arr
 */
      (__pyx_v_arr_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, (__pyx_v_oa_data[0]));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":190
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(iter), 1)
 *             arr_data[i] = func(state, oa_data[0])
 *             NpyArray_MultiIter_NEXTi(getiter(iter), 1)             # <<<<<<<<<<<<<<
 *     return arr
 * 
 */
      __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "iter");
      NpyArray_MultiIter_NEXTi(getiter(__pyx_t_2), 1);
      __pyx_t_2 = nullptr;
    }
  }
  __pyx_L3:;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":191
 *             arr_data[i] = func(state, oa_data[0])
 *             NpyArray_MultiIter_NEXTi(getiter(iter), 1)
 *     return arr             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = __pyx_v_arr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":194
 * 
 * 
 * cdef object cont2_array_sc(rk_state *state, rk_cont2 func, object size,             # <<<<<<<<<<<<<<
 *                            double a, double b):
 *     cdef double *data
 */

static  System::Object^ cont2_array_sc(rk_state *__pyx_v_state, __pyx_t_6mtrand_rk_cont2 __pyx_v_func, System::Object^ __pyx_v_size, double __pyx_v_a, double __pyx_v_b) {
  double *__pyx_v_data;
  npy_intp __pyx_v_length;
  npy_intp __pyx_v_i;
  System::Object^ __pyx_v_arr;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  System::Object^ __pyx_t_4 = nullptr;
  npy_intp __pyx_t_5;
  npy_intp __pyx_t_6;
  __pyx_v_arr = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":199
 *     cdef npy_intp length, i
 * 
 *     if size is None:             # <<<<<<<<<<<<<<
 *         return func(state, a, b)
 *     else:
 */
  __pyx_t_1 = (__pyx_v_size == nullptr);
  if (__pyx_t_1) {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":200
 * 
 *     if size is None:
 *         return func(state, a, b)             # <<<<<<<<<<<<<<
 *     else:
 *         arr = np.empty(size, np.double)
 */
    __pyx_t_2 = __pyx_v_func(__pyx_v_state, __pyx_v_a, __pyx_v_b);
    __pyx_r = __pyx_t_2;
    __pyx_t_2 = nullptr;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  /*else*/ {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":202
 *         return func(state, a, b)
 *     else:
 *         arr = np.empty(size, np.double)             # <<<<<<<<<<<<<<
 *         length = arr.size
 *         data = <double *>dataptr(arr)
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_empty_202_16->Target(__site_get_empty_202_16, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_4 = __site_get_double_202_31->Target(__site_get_double_202_31, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = __site_call2_202_22->Target(__site_call2_202_22, __pyx_context, __pyx_t_3, __pyx_v_size, __pyx_t_4);
    __pyx_t_3 = nullptr;
    __pyx_t_4 = nullptr;
    __pyx_v_arr = __pyx_t_2;
    __pyx_t_2 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":203
 *     else:
 *         arr = np.empty(size, np.double)
 *         length = arr.size             # <<<<<<<<<<<<<<
 *         data = <double *>dataptr(arr)
 *         for i from 0 <= i < length:
 */
    __pyx_t_2 = __site_get_size_203_20->Target(__site_get_size_203_20, __pyx_v_arr, __pyx_context);
    __pyx_t_5 = __site_cvt_npy_intp_203_20->Target(__site_cvt_npy_intp_203_20, __pyx_t_2);
    __pyx_t_2 = nullptr;
    __pyx_v_length = __pyx_t_5;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":204
 *         arr = np.empty(size, np.double)
 *         length = arr.size
 *         data = <double *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < length:
 *             data[i] = func(state, a, b)
 */
    __pyx_v_data = ((double *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":205
 *         length = arr.size
 *         data = <double *>dataptr(arr)
 *         for i from 0 <= i < length:             # <<<<<<<<<<<<<<
 *             data[i] = func(state, a, b)
 *         return arr
 */
    __pyx_t_6 = __pyx_v_length;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_6; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":206
 *         data = <double *>dataptr(arr)
 *         for i from 0 <= i < length:
 *             data[i] = func(state, a, b)             # <<<<<<<<<<<<<<
 *         return arr
 * 
 */
      (__pyx_v_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, __pyx_v_a, __pyx_v_b);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":207
 *         for i from 0 <= i < length:
 *             data[i] = func(state, a, b)
 *         return arr             # <<<<<<<<<<<<<<
 * 
 * 
 */
    __pyx_r = __pyx_v_arr;
    goto __pyx_L0;
  }
  __pyx_L3:;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":210
 * 
 * 
 * cdef object cont2_array(rk_state *state, rk_cont2 func, object size,             # <<<<<<<<<<<<<<
 *                         object a, object b):
 *     cdef double *arr_data
 */

static  System::Object^ cont2_array(rk_state *__pyx_v_state, __pyx_t_6mtrand_rk_cont2 __pyx_v_func, System::Object^ __pyx_v_size, System::Object^ __pyx_v_a, System::Object^ __pyx_v_b) {
  double *__pyx_v_arr_data;
  double *__pyx_v_oa_data;
  double *__pyx_v_ob_data;
  npy_intp __pyx_v_i;
  System::Object^ __pyx_v_oa;
  System::Object^ __pyx_v_ob;
  System::Object^ __pyx_v_multi;
  System::Object^ __pyx_v_arr;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  int __pyx_t_4;
  System::Object^ __pyx_t_5 = nullptr;
  npy_intp __pyx_t_6;
  npy_intp __pyx_t_7;
  __pyx_v_oa = nullptr;
  __pyx_v_ob = nullptr;
  __pyx_v_multi = nullptr;
  __pyx_v_arr = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":217
 *     cdef npy_intp length, i
 * 
 *     oa = np.array(a, np.double)             # <<<<<<<<<<<<<<
 *     ob = np.array(b, np.double)
 *     if size is None:
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_2 = __site_get_array_217_11->Target(__site_get_array_217_11, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_3 = __site_get_double_217_23->Target(__site_get_double_217_23, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call2_217_17->Target(__site_call2_217_17, __pyx_context, __pyx_t_2, __pyx_v_a, __pyx_t_3);
  __pyx_t_2 = nullptr;
  __pyx_t_3 = nullptr;
  __pyx_v_oa = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":218
 * 
 *     oa = np.array(a, np.double)
 *     ob = np.array(b, np.double)             # <<<<<<<<<<<<<<
 *     if size is None:
 *         multi = np.broadcast(oa, ob)
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_3 = __site_get_array_218_11->Target(__site_get_array_218_11, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_2 = __site_get_double_218_23->Target(__site_get_double_218_23, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call2_218_17->Target(__site_call2_218_17, __pyx_context, __pyx_t_3, __pyx_v_b, __pyx_t_2);
  __pyx_t_3 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_v_ob = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":219
 *     oa = np.array(a, np.double)
 *     ob = np.array(b, np.double)
 *     if size is None:             # <<<<<<<<<<<<<<
 *         multi = np.broadcast(oa, ob)
 *         arr = np.empty(multi.shape, np.double)
 */
  __pyx_t_4 = (__pyx_v_size == nullptr);
  if (__pyx_t_4) {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":220
 *     ob = np.array(b, np.double)
 *     if size is None:
 *         multi = np.broadcast(oa, ob)             # <<<<<<<<<<<<<<
 *         arr = np.empty(multi.shape, np.double)
 *         arr_data = <double *>dataptr(arr)
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_2 = __site_get_broadcast_220_18->Target(__site_get_broadcast_220_18, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call2_220_28->Target(__site_call2_220_28, __pyx_context, __pyx_t_2, __pyx_v_oa, __pyx_v_ob);
    __pyx_t_2 = nullptr;
    __pyx_v_multi = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":221
 *     if size is None:
 *         multi = np.broadcast(oa, ob)
 *         arr = np.empty(multi.shape, np.double)             # <<<<<<<<<<<<<<
 *         arr_data = <double *>dataptr(arr)
 *         for i from 0 <= i < multi.size:
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_2 = __site_get_empty_221_16->Target(__site_get_empty_221_16, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_get_shape_221_28->Target(__site_get_shape_221_28, __pyx_v_multi, __pyx_context);
    __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_double_221_38->Target(__site_get_double_221_38, __pyx_t_3, __pyx_context);
    __pyx_t_3 = nullptr;
    __pyx_t_3 = __site_call2_221_22->Target(__site_call2_221_22, __pyx_context, __pyx_t_2, __pyx_t_1, __pyx_t_5);
    __pyx_t_2 = nullptr;
    __pyx_t_1 = nullptr;
    __pyx_t_5 = nullptr;
    __pyx_v_arr = __pyx_t_3;
    __pyx_t_3 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":222
 *         multi = np.broadcast(oa, ob)
 *         arr = np.empty(multi.shape, np.double)
 *         arr_data = <double *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < multi.size:
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 */
    __pyx_v_arr_data = ((double *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":223
 *         arr = np.empty(multi.shape, np.double)
 *         arr_data = <double *>dataptr(arr)
 *         for i from 0 <= i < multi.size:             # <<<<<<<<<<<<<<
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 */
    __pyx_t_3 = __site_get_size_223_33->Target(__site_get_size_223_33, __pyx_v_multi, __pyx_context);
    __pyx_t_6 = __site_cvt_npy_intp_223_33->Target(__site_cvt_npy_intp_223_33, __pyx_t_3);
    __pyx_t_3 = nullptr;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_6; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":224
 *         arr_data = <double *>dataptr(arr)
 *         for i from 0 <= i < multi.size:
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 0)             # <<<<<<<<<<<<<<
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             arr_data[i] = func(state, oa_data[0], ob_data[0])
 */
      __pyx_v_oa_data = ((double *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 0));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":225
 *         for i from 0 <= i < multi.size:
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)             # <<<<<<<<<<<<<<
 *             arr_data[i] = func(state, oa_data[0], ob_data[0])
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 */
      __pyx_v_ob_data = ((double *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 1));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":226
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             arr_data[i] = func(state, oa_data[0], ob_data[0])             # <<<<<<<<<<<<<<
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 *     else:
 */
      (__pyx_v_arr_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, (__pyx_v_oa_data[0]), (__pyx_v_ob_data[0]));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":227
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             arr_data[i] = func(state, oa_data[0], ob_data[0])
 *             NpyArray_MultiIter_NEXT(getiter(multi))             # <<<<<<<<<<<<<<
 *     else:
 *         arr = np.empty(size, np.double)
 */
      NpyArray_MultiIter_NEXT(getiter(__pyx_v_multi));
    }
    goto __pyx_L3;
  }
  /*else*/ {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":229
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 *     else:
 *         arr = np.empty(size, np.double)             # <<<<<<<<<<<<<<
 *         arr_data = <double *>dataptr(arr)
 *         multi = np.broadcast(arr, oa, ob)
 */
    __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_empty_229_16->Target(__site_get_empty_229_16, __pyx_t_3, __pyx_context);
    __pyx_t_3 = nullptr;
    __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_1 = __site_get_double_229_31->Target(__site_get_double_229_31, __pyx_t_3, __pyx_context);
    __pyx_t_3 = nullptr;
    __pyx_t_3 = __site_call2_229_22->Target(__site_call2_229_22, __pyx_context, __pyx_t_5, __pyx_v_size, __pyx_t_1);
    __pyx_t_5 = nullptr;
    __pyx_t_1 = nullptr;
    __pyx_v_arr = __pyx_t_3;
    __pyx_t_3 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":230
 *     else:
 *         arr = np.empty(size, np.double)
 *         arr_data = <double *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         multi = np.broadcast(arr, oa, ob)
 *         if multi.size != arr.size:
 */
    __pyx_v_arr_data = ((double *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":231
 *         arr = np.empty(size, np.double)
 *         arr_data = <double *>dataptr(arr)
 *         multi = np.broadcast(arr, oa, ob)             # <<<<<<<<<<<<<<
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")
 */
    __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_1 = __site_get_broadcast_231_18->Target(__site_get_broadcast_231_18, __pyx_t_3, __pyx_context);
    __pyx_t_3 = nullptr;
    __pyx_t_3 = __site_call3_231_28->Target(__site_call3_231_28, __pyx_context, __pyx_t_1, __pyx_v_arr, __pyx_v_oa, __pyx_v_ob);
    __pyx_t_1 = nullptr;
    __pyx_v_multi = __pyx_t_3;
    __pyx_t_3 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":232
 *         arr_data = <double *>dataptr(arr)
 *         multi = np.broadcast(arr, oa, ob)
 *         if multi.size != arr.size:             # <<<<<<<<<<<<<<
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:
 */
    __pyx_t_3 = __site_get_size_232_16->Target(__site_get_size_232_16, __pyx_v_multi, __pyx_context);
    __pyx_t_1 = __site_get_size_232_28->Target(__site_get_size_232_28, __pyx_v_arr, __pyx_context);
    __pyx_t_5 = __site_op_ne_232_22->Target(__site_op_ne_232_22, __pyx_t_3, __pyx_t_1);
    __pyx_t_3 = nullptr;
    __pyx_t_1 = nullptr;
    __pyx_t_4 = __site_istrue_232_22->Target(__site_istrue_232_22, __pyx_t_5);
    __pyx_t_5 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":233
 *         multi = np.broadcast(arr, oa, ob)
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < multi.size:
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 */
      __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_1 = __site_call1_233_28->Target(__site_call1_233_28, __pyx_context, __pyx_t_5, ((System::Object^)"size is not compatible with inputs"));
      __pyx_t_5 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_1, nullptr, nullptr);
      __pyx_t_1 = nullptr;
      goto __pyx_L6;
    }
    __pyx_L6:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":234
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:             # <<<<<<<<<<<<<<
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 */
    __pyx_t_1 = __site_get_size_234_33->Target(__site_get_size_234_33, __pyx_v_multi, __pyx_context);
    __pyx_t_7 = __site_cvt_npy_intp_234_33->Target(__site_cvt_npy_intp_234_33, __pyx_t_1);
    __pyx_t_1 = nullptr;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_7; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":235
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)             # <<<<<<<<<<<<<<
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             arr_data[i] = func(state, oa_data[0], ob_data[0])
 */
      __pyx_v_oa_data = ((double *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 1));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":236
 *         for i from 0 <= i < multi.size:
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)             # <<<<<<<<<<<<<<
 *             arr_data[i] = func(state, oa_data[0], ob_data[0])
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)
 */
      __pyx_v_ob_data = ((double *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 2));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":237
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             arr_data[i] = func(state, oa_data[0], ob_data[0])             # <<<<<<<<<<<<<<
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 2)
 */
      (__pyx_v_arr_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, (__pyx_v_oa_data[0]), (__pyx_v_ob_data[0]));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":238
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             arr_data[i] = func(state, oa_data[0], ob_data[0])
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)             # <<<<<<<<<<<<<<
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 2)
 *     return arr
 */
      NpyArray_MultiIter_NEXTi(getiter(__pyx_v_multi), 1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":239
 *             arr_data[i] = func(state, oa_data[0], ob_data[0])
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 2)             # <<<<<<<<<<<<<<
 *     return arr
 * 
 */
      NpyArray_MultiIter_NEXTi(getiter(__pyx_v_multi), 2);
    }
  }
  __pyx_L3:;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":240
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 2)
 *     return arr             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = __pyx_v_arr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":243
 * 
 * 
 * cdef object cont3_array_sc(rk_state *state, rk_cont3 func, object size,             # <<<<<<<<<<<<<<
 *                            double a, double b, double c):
 *     cdef double *arr_data
 */

static  System::Object^ cont3_array_sc(rk_state *__pyx_v_state, __pyx_t_6mtrand_rk_cont3 __pyx_v_func, System::Object^ __pyx_v_size, double __pyx_v_a, double __pyx_v_b, double __pyx_v_c) {
  double *__pyx_v_arr_data;
  npy_intp __pyx_v_length;
  npy_intp __pyx_v_i;
  System::Object^ __pyx_v_arr;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  System::Object^ __pyx_t_4 = nullptr;
  npy_intp __pyx_t_5;
  npy_intp __pyx_t_6;
  __pyx_v_arr = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":248
 *     cdef npy_intp length, i
 * 
 *     if size is None:             # <<<<<<<<<<<<<<
 *         return func(state, a, b, c)
 *     else:
 */
  __pyx_t_1 = (__pyx_v_size == nullptr);
  if (__pyx_t_1) {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":249
 * 
 *     if size is None:
 *         return func(state, a, b, c)             # <<<<<<<<<<<<<<
 *     else:
 *         arr = np.empty(size, np.double)
 */
    __pyx_t_2 = __pyx_v_func(__pyx_v_state, __pyx_v_a, __pyx_v_b, __pyx_v_c);
    __pyx_r = __pyx_t_2;
    __pyx_t_2 = nullptr;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  /*else*/ {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":251
 *         return func(state, a, b, c)
 *     else:
 *         arr = np.empty(size, np.double)             # <<<<<<<<<<<<<<
 *         length = arr.size
 *         arr_data = <double *>dataptr(arr)
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_empty_251_16->Target(__site_get_empty_251_16, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_4 = __site_get_double_251_31->Target(__site_get_double_251_31, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = __site_call2_251_22->Target(__site_call2_251_22, __pyx_context, __pyx_t_3, __pyx_v_size, __pyx_t_4);
    __pyx_t_3 = nullptr;
    __pyx_t_4 = nullptr;
    __pyx_v_arr = __pyx_t_2;
    __pyx_t_2 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":252
 *     else:
 *         arr = np.empty(size, np.double)
 *         length = arr.size             # <<<<<<<<<<<<<<
 *         arr_data = <double *>dataptr(arr)
 *         for i from 0 <= i < length:
 */
    __pyx_t_2 = __site_get_size_252_20->Target(__site_get_size_252_20, __pyx_v_arr, __pyx_context);
    __pyx_t_5 = __site_cvt_npy_intp_252_20->Target(__site_cvt_npy_intp_252_20, __pyx_t_2);
    __pyx_t_2 = nullptr;
    __pyx_v_length = __pyx_t_5;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":253
 *         arr = np.empty(size, np.double)
 *         length = arr.size
 *         arr_data = <double *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < length:
 *             arr_data[i] = func(state, a, b, c)
 */
    __pyx_v_arr_data = ((double *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":254
 *         length = arr.size
 *         arr_data = <double *>dataptr(arr)
 *         for i from 0 <= i < length:             # <<<<<<<<<<<<<<
 *             arr_data[i] = func(state, a, b, c)
 *         return arr
 */
    __pyx_t_6 = __pyx_v_length;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_6; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":255
 *         arr_data = <double *>dataptr(arr)
 *         for i from 0 <= i < length:
 *             arr_data[i] = func(state, a, b, c)             # <<<<<<<<<<<<<<
 *         return arr
 * 
 */
      (__pyx_v_arr_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, __pyx_v_a, __pyx_v_b, __pyx_v_c);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":256
 *         for i from 0 <= i < length:
 *             arr_data[i] = func(state, a, b, c)
 *         return arr             # <<<<<<<<<<<<<<
 * 
 * 
 */
    __pyx_r = __pyx_v_arr;
    goto __pyx_L0;
  }
  __pyx_L3:;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":259
 * 
 * 
 * cdef object cont3_array(rk_state *state, rk_cont3 func, object size,             # <<<<<<<<<<<<<<
 *                         object a, object b, object c):
 *     cdef double *arr_data
 */

static  System::Object^ cont3_array(rk_state *__pyx_v_state, __pyx_t_6mtrand_rk_cont3 __pyx_v_func, System::Object^ __pyx_v_size, System::Object^ __pyx_v_a, System::Object^ __pyx_v_b, System::Object^ __pyx_v_c) {
  double *__pyx_v_arr_data;
  double *__pyx_v_oa_data;
  double *__pyx_v_ob_data;
  double *__pyx_v_oc_data;
  npy_intp __pyx_v_i;
  System::Object^ __pyx_v_oa;
  System::Object^ __pyx_v_ob;
  System::Object^ __pyx_v_oc;
  System::Object^ __pyx_v_multi;
  System::Object^ __pyx_v_arr;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  int __pyx_t_4;
  System::Object^ __pyx_t_5 = nullptr;
  npy_intp __pyx_t_6;
  npy_intp __pyx_t_7;
  __pyx_v_oa = nullptr;
  __pyx_v_ob = nullptr;
  __pyx_v_oc = nullptr;
  __pyx_v_multi = nullptr;
  __pyx_v_arr = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":267
 *     cdef npy_intp length, i
 * 
 *     oa = np.array(a, np.double)             # <<<<<<<<<<<<<<
 *     ob = np.array(b, np.double)
 *     oc = np.array(c, np.double)
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_2 = __site_get_array_267_11->Target(__site_get_array_267_11, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_3 = __site_get_double_267_23->Target(__site_get_double_267_23, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call2_267_17->Target(__site_call2_267_17, __pyx_context, __pyx_t_2, __pyx_v_a, __pyx_t_3);
  __pyx_t_2 = nullptr;
  __pyx_t_3 = nullptr;
  __pyx_v_oa = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":268
 * 
 *     oa = np.array(a, np.double)
 *     ob = np.array(b, np.double)             # <<<<<<<<<<<<<<
 *     oc = np.array(c, np.double)
 *     if size is None:
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_3 = __site_get_array_268_11->Target(__site_get_array_268_11, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_2 = __site_get_double_268_23->Target(__site_get_double_268_23, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call2_268_17->Target(__site_call2_268_17, __pyx_context, __pyx_t_3, __pyx_v_b, __pyx_t_2);
  __pyx_t_3 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_v_ob = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":269
 *     oa = np.array(a, np.double)
 *     ob = np.array(b, np.double)
 *     oc = np.array(c, np.double)             # <<<<<<<<<<<<<<
 *     if size is None:
 *         multi = np.broadcast(oa, ob, oc)
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_2 = __site_get_array_269_11->Target(__site_get_array_269_11, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_3 = __site_get_double_269_23->Target(__site_get_double_269_23, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call2_269_17->Target(__site_call2_269_17, __pyx_context, __pyx_t_2, __pyx_v_c, __pyx_t_3);
  __pyx_t_2 = nullptr;
  __pyx_t_3 = nullptr;
  __pyx_v_oc = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":270
 *     ob = np.array(b, np.double)
 *     oc = np.array(c, np.double)
 *     if size is None:             # <<<<<<<<<<<<<<
 *         multi = np.broadcast(oa, ob, oc)
 *         arr = np.empty(multi.shape, np.double)
 */
  __pyx_t_4 = (__pyx_v_size == nullptr);
  if (__pyx_t_4) {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":271
 *     oc = np.array(c, np.double)
 *     if size is None:
 *         multi = np.broadcast(oa, ob, oc)             # <<<<<<<<<<<<<<
 *         arr = np.empty(multi.shape, np.double)
 *         arr_data = <double *>dataptr(arr)
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_broadcast_271_18->Target(__site_get_broadcast_271_18, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call3_271_28->Target(__site_call3_271_28, __pyx_context, __pyx_t_3, __pyx_v_oa, __pyx_v_ob, __pyx_v_oc);
    __pyx_t_3 = nullptr;
    __pyx_v_multi = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":272
 *     if size is None:
 *         multi = np.broadcast(oa, ob, oc)
 *         arr = np.empty(multi.shape, np.double)             # <<<<<<<<<<<<<<
 *         arr_data = <double *>dataptr(arr)
 *         for i from 0 <= i < multi.size:
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_empty_272_16->Target(__site_get_empty_272_16, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_get_shape_272_28->Target(__site_get_shape_272_28, __pyx_v_multi, __pyx_context);
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_double_272_38->Target(__site_get_double_272_38, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = __site_call2_272_22->Target(__site_call2_272_22, __pyx_context, __pyx_t_3, __pyx_t_1, __pyx_t_5);
    __pyx_t_3 = nullptr;
    __pyx_t_1 = nullptr;
    __pyx_t_5 = nullptr;
    __pyx_v_arr = __pyx_t_2;
    __pyx_t_2 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":273
 *         multi = np.broadcast(oa, ob, oc)
 *         arr = np.empty(multi.shape, np.double)
 *         arr_data = <double *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < multi.size:
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 */
    __pyx_v_arr_data = ((double *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":274
 *         arr = np.empty(multi.shape, np.double)
 *         arr_data = <double *>dataptr(arr)
 *         for i from 0 <= i < multi.size:             # <<<<<<<<<<<<<<
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 */
    __pyx_t_2 = __site_get_size_274_33->Target(__site_get_size_274_33, __pyx_v_multi, __pyx_context);
    __pyx_t_6 = __site_cvt_npy_intp_274_33->Target(__site_cvt_npy_intp_274_33, __pyx_t_2);
    __pyx_t_2 = nullptr;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_6; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":275
 *         arr_data = <double *>dataptr(arr)
 *         for i from 0 <= i < multi.size:
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 0)             # <<<<<<<<<<<<<<
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             oc_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 */
      __pyx_v_oa_data = ((double *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 0));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":276
 *         for i from 0 <= i < multi.size:
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)             # <<<<<<<<<<<<<<
 *             oc_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             arr_data[i] = func(state, oa_data[0], ob_data[0], oc_data[0])
 */
      __pyx_v_ob_data = ((double *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 1));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":277
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             oc_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)             # <<<<<<<<<<<<<<
 *             arr_data[i] = func(state, oa_data[0], ob_data[0], oc_data[0])
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 */
      __pyx_v_oc_data = ((double *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 2));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":278
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             oc_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             arr_data[i] = func(state, oa_data[0], ob_data[0], oc_data[0])             # <<<<<<<<<<<<<<
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 *     else:
 */
      (__pyx_v_arr_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, (__pyx_v_oa_data[0]), (__pyx_v_ob_data[0]), (__pyx_v_oc_data[0]));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":279
 *             oc_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             arr_data[i] = func(state, oa_data[0], ob_data[0], oc_data[0])
 *             NpyArray_MultiIter_NEXT(getiter(multi))             # <<<<<<<<<<<<<<
 *     else:
 *         arr = np.empty(size, np.double)
 */
      NpyArray_MultiIter_NEXT(getiter(__pyx_v_multi));
    }
    goto __pyx_L3;
  }
  /*else*/ {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":281
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 *     else:
 *         arr = np.empty(size, np.double)             # <<<<<<<<<<<<<<
 *         arr_data = <double *>dataptr(arr)
 *         multi = np.broadcast(arr, oa, ob, oc)
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_empty_281_16->Target(__site_get_empty_281_16, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_1 = __site_get_double_281_31->Target(__site_get_double_281_31, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = __site_call2_281_22->Target(__site_call2_281_22, __pyx_context, __pyx_t_5, __pyx_v_size, __pyx_t_1);
    __pyx_t_5 = nullptr;
    __pyx_t_1 = nullptr;
    __pyx_v_arr = __pyx_t_2;
    __pyx_t_2 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":282
 *     else:
 *         arr = np.empty(size, np.double)
 *         arr_data = <double *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         multi = np.broadcast(arr, oa, ob, oc)
 *         if multi.size != arr.size:
 */
    __pyx_v_arr_data = ((double *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":283
 *         arr = np.empty(size, np.double)
 *         arr_data = <double *>dataptr(arr)
 *         multi = np.broadcast(arr, oa, ob, oc)             # <<<<<<<<<<<<<<
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_1 = __site_get_broadcast_283_18->Target(__site_get_broadcast_283_18, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = __site_call4_283_28->Target(__site_call4_283_28, __pyx_context, __pyx_t_1, __pyx_v_arr, __pyx_v_oa, __pyx_v_ob, __pyx_v_oc);
    __pyx_t_1 = nullptr;
    __pyx_v_multi = __pyx_t_2;
    __pyx_t_2 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":284
 *         arr_data = <double *>dataptr(arr)
 *         multi = np.broadcast(arr, oa, ob, oc)
 *         if multi.size != arr.size:             # <<<<<<<<<<<<<<
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:
 */
    __pyx_t_2 = __site_get_size_284_16->Target(__site_get_size_284_16, __pyx_v_multi, __pyx_context);
    __pyx_t_1 = __site_get_size_284_28->Target(__site_get_size_284_28, __pyx_v_arr, __pyx_context);
    __pyx_t_5 = __site_op_ne_284_22->Target(__site_op_ne_284_22, __pyx_t_2, __pyx_t_1);
    __pyx_t_2 = nullptr;
    __pyx_t_1 = nullptr;
    __pyx_t_4 = __site_istrue_284_22->Target(__site_istrue_284_22, __pyx_t_5);
    __pyx_t_5 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":285
 *         multi = np.broadcast(arr, oa, ob, oc)
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < multi.size:
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 */
      __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_1 = __site_call1_285_28->Target(__site_call1_285_28, __pyx_context, __pyx_t_5, ((System::Object^)"size is not compatible with inputs"));
      __pyx_t_5 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_1, nullptr, nullptr);
      __pyx_t_1 = nullptr;
      goto __pyx_L6;
    }
    __pyx_L6:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":286
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:             # <<<<<<<<<<<<<<
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 */
    __pyx_t_1 = __site_get_size_286_33->Target(__site_get_size_286_33, __pyx_v_multi, __pyx_context);
    __pyx_t_7 = __site_cvt_npy_intp_286_33->Target(__site_cvt_npy_intp_286_33, __pyx_t_1);
    __pyx_t_1 = nullptr;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_7; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":287
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)             # <<<<<<<<<<<<<<
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             oc_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 3)
 */
      __pyx_v_oa_data = ((double *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 1));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":288
 *         for i from 0 <= i < multi.size:
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)             # <<<<<<<<<<<<<<
 *             oc_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 3)
 *             arr_data[i] = func(state, oa_data[0], ob_data[0], oc_data[0])
 */
      __pyx_v_ob_data = ((double *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 2));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":289
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             oc_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 3)             # <<<<<<<<<<<<<<
 *             arr_data[i] = func(state, oa_data[0], ob_data[0], oc_data[0])
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 */
      __pyx_v_oc_data = ((double *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 3));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":290
 *             ob_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             oc_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 3)
 *             arr_data[i] = func(state, oa_data[0], ob_data[0], oc_data[0])             # <<<<<<<<<<<<<<
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 *     return arr
 */
      (__pyx_v_arr_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, (__pyx_v_oa_data[0]), (__pyx_v_ob_data[0]), (__pyx_v_oc_data[0]));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":291
 *             oc_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 3)
 *             arr_data[i] = func(state, oa_data[0], ob_data[0], oc_data[0])
 *             NpyArray_MultiIter_NEXT(getiter(multi))             # <<<<<<<<<<<<<<
 *     return arr
 * 
 */
      NpyArray_MultiIter_NEXT(getiter(__pyx_v_multi));
    }
  }
  __pyx_L3:;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":292
 *             arr_data[i] = func(state, oa_data[0], ob_data[0], oc_data[0])
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 *     return arr             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = __pyx_v_arr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":295
 * 
 * 
 * cdef object disc0_array(rk_state *state, rk_disc0 func, object size):             # <<<<<<<<<<<<<<
 *     cdef long *data
 *     cdef npy_intp length, i
 */

static  System::Object^ disc0_array(rk_state *__pyx_v_state, __pyx_t_6mtrand_rk_disc0 __pyx_v_func, System::Object^ __pyx_v_size) {
  long *__pyx_v_data;
  npy_intp __pyx_v_length;
  npy_intp __pyx_v_i;
  System::Object^ __pyx_v_arr;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  System::Object^ __pyx_t_4 = nullptr;
  npy_intp __pyx_t_5;
  npy_intp __pyx_t_6;
  __pyx_v_arr = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":299
 *     cdef npy_intp length, i
 * 
 *     if size is None:             # <<<<<<<<<<<<<<
 *         return func(state)
 *     else:
 */
  __pyx_t_1 = (__pyx_v_size == nullptr);
  if (__pyx_t_1) {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":300
 * 
 *     if size is None:
 *         return func(state)             # <<<<<<<<<<<<<<
 *     else:
 *         arr = np.empty(size, int)
 */
    __pyx_t_2 = __pyx_v_func(__pyx_v_state);
    __pyx_r = __pyx_t_2;
    __pyx_t_2 = nullptr;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  /*else*/ {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":302
 *         return func(state)
 *     else:
 *         arr = np.empty(size, int)             # <<<<<<<<<<<<<<
 *         length = arr.size
 *         data = <long *>dataptr(arr)
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_empty_302_16->Target(__site_get_empty_302_16, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "int");
    __pyx_t_4 = __site_call2_302_22->Target(__site_call2_302_22, __pyx_context, __pyx_t_3, __pyx_v_size, ((System::Object^)__pyx_t_2));
    __pyx_t_3 = nullptr;
    __pyx_t_2 = nullptr;
    __pyx_v_arr = __pyx_t_4;
    __pyx_t_4 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":303
 *     else:
 *         arr = np.empty(size, int)
 *         length = arr.size             # <<<<<<<<<<<<<<
 *         data = <long *>dataptr(arr)
 *         for i from 0 <= i < length:
 */
    __pyx_t_4 = __site_get_size_303_20->Target(__site_get_size_303_20, __pyx_v_arr, __pyx_context);
    __pyx_t_5 = __site_cvt_npy_intp_303_20->Target(__site_cvt_npy_intp_303_20, __pyx_t_4);
    __pyx_t_4 = nullptr;
    __pyx_v_length = __pyx_t_5;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":304
 *         arr = np.empty(size, int)
 *         length = arr.size
 *         data = <long *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < length:
 *             data[i] = func(state)
 */
    __pyx_v_data = ((long *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":305
 *         length = arr.size
 *         data = <long *>dataptr(arr)
 *         for i from 0 <= i < length:             # <<<<<<<<<<<<<<
 *             data[i] = func(state)
 *         return arr
 */
    __pyx_t_6 = __pyx_v_length;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_6; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":306
 *         data = <long *>dataptr(arr)
 *         for i from 0 <= i < length:
 *             data[i] = func(state)             # <<<<<<<<<<<<<<
 *         return arr
 * 
 */
      (__pyx_v_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":307
 *         for i from 0 <= i < length:
 *             data[i] = func(state)
 *         return arr             # <<<<<<<<<<<<<<
 * 
 * 
 */
    __pyx_r = __pyx_v_arr;
    goto __pyx_L0;
  }
  __pyx_L3:;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":310
 * 
 * 
 * cdef object discnp_array_sc(rk_state *state, rk_discnp func, object size,             # <<<<<<<<<<<<<<
 *                             long n, double p):
 *     cdef long *data
 */

static  System::Object^ discnp_array_sc(rk_state *__pyx_v_state, __pyx_t_6mtrand_rk_discnp __pyx_v_func, System::Object^ __pyx_v_size, long __pyx_v_n, double __pyx_v_p) {
  long *__pyx_v_data;
  npy_intp __pyx_v_length;
  npy_intp __pyx_v_i;
  System::Object^ __pyx_v_arr;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  System::Object^ __pyx_t_4 = nullptr;
  npy_intp __pyx_t_5;
  npy_intp __pyx_t_6;
  __pyx_v_arr = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":315
 *     cdef npy_intp length, i
 * 
 *     if size is None:             # <<<<<<<<<<<<<<
 *         return func(state, n, p)
 *     else:
 */
  __pyx_t_1 = (__pyx_v_size == nullptr);
  if (__pyx_t_1) {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":316
 * 
 *     if size is None:
 *         return func(state, n, p)             # <<<<<<<<<<<<<<
 *     else:
 *         arr = np.empty(size, int)
 */
    __pyx_t_2 = __pyx_v_func(__pyx_v_state, __pyx_v_n, __pyx_v_p);
    __pyx_r = __pyx_t_2;
    __pyx_t_2 = nullptr;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  /*else*/ {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":318
 *         return func(state, n, p)
 *     else:
 *         arr = np.empty(size, int)             # <<<<<<<<<<<<<<
 *         length = arr.size
 *         data = <long *>dataptr(arr)
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_empty_318_16->Target(__site_get_empty_318_16, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "int");
    __pyx_t_4 = __site_call2_318_22->Target(__site_call2_318_22, __pyx_context, __pyx_t_3, __pyx_v_size, ((System::Object^)__pyx_t_2));
    __pyx_t_3 = nullptr;
    __pyx_t_2 = nullptr;
    __pyx_v_arr = __pyx_t_4;
    __pyx_t_4 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":319
 *     else:
 *         arr = np.empty(size, int)
 *         length = arr.size             # <<<<<<<<<<<<<<
 *         data = <long *>dataptr(arr)
 *         for i from 0 <= i < length:
 */
    __pyx_t_4 = __site_get_size_319_20->Target(__site_get_size_319_20, __pyx_v_arr, __pyx_context);
    __pyx_t_5 = __site_cvt_npy_intp_319_20->Target(__site_cvt_npy_intp_319_20, __pyx_t_4);
    __pyx_t_4 = nullptr;
    __pyx_v_length = __pyx_t_5;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":320
 *         arr = np.empty(size, int)
 *         length = arr.size
 *         data = <long *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < length:
 *             data[i] = func(state, n, p)
 */
    __pyx_v_data = ((long *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":321
 *         length = arr.size
 *         data = <long *>dataptr(arr)
 *         for i from 0 <= i < length:             # <<<<<<<<<<<<<<
 *             data[i] = func(state, n, p)
 *         return arr
 */
    __pyx_t_6 = __pyx_v_length;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_6; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":322
 *         data = <long *>dataptr(arr)
 *         for i from 0 <= i < length:
 *             data[i] = func(state, n, p)             # <<<<<<<<<<<<<<
 *         return arr
 * 
 */
      (__pyx_v_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, __pyx_v_n, __pyx_v_p);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":323
 *         for i from 0 <= i < length:
 *             data[i] = func(state, n, p)
 *         return arr             # <<<<<<<<<<<<<<
 * 
 * 
 */
    __pyx_r = __pyx_v_arr;
    goto __pyx_L0;
  }
  __pyx_L3:;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":326
 * 
 * 
 * cdef object discnp_array(rk_state *state, rk_discnp func, object size,             # <<<<<<<<<<<<<<
 *                          object n, object p):
 *     cdef long *arr_data
 */

static  System::Object^ discnp_array(rk_state *__pyx_v_state, __pyx_t_6mtrand_rk_discnp __pyx_v_func, System::Object^ __pyx_v_size, System::Object^ __pyx_v_n, System::Object^ __pyx_v_p) {
  long *__pyx_v_arr_data;
  double *__pyx_v_op_data;
  long *__pyx_v_on_data;
  npy_intp __pyx_v_i;
  System::Object^ __pyx_v_on;
  System::Object^ __pyx_v_op;
  System::Object^ __pyx_v_multi;
  System::Object^ __pyx_v_arr;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  int __pyx_t_4;
  System::Object^ __pyx_t_5 = nullptr;
  npy_intp __pyx_t_6;
  npy_intp __pyx_t_7;
  __pyx_v_on = nullptr;
  __pyx_v_op = nullptr;
  __pyx_v_multi = nullptr;
  __pyx_v_arr = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":333
 *     cdef npy_intp length, i
 * 
 *     on = np.array(n, dtype=np.long)             # <<<<<<<<<<<<<<
 *     op = np.array(p, dtype=np.double)
 *     if size is None:
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_2 = __site_get_array_333_11->Target(__site_get_array_333_11, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_3 = __site_get_long_333_29->Target(__site_get_long_333_29, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call1_dtype_333_17->Target(__site_call1_dtype_333_17, __pyx_context, __pyx_t_2, __pyx_v_n, __pyx_t_3);
  __pyx_t_2 = nullptr;
  __pyx_t_3 = nullptr;
  __pyx_v_on = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":334
 * 
 *     on = np.array(n, dtype=np.long)
 *     op = np.array(p, dtype=np.double)             # <<<<<<<<<<<<<<
 *     if size is None:
 *         multi = np.broadcast(on, op)
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_3 = __site_get_array_334_11->Target(__site_get_array_334_11, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_2 = __site_get_double_334_29->Target(__site_get_double_334_29, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call1_dtype_334_17->Target(__site_call1_dtype_334_17, __pyx_context, __pyx_t_3, __pyx_v_p, __pyx_t_2);
  __pyx_t_3 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_v_op = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":335
 *     on = np.array(n, dtype=np.long)
 *     op = np.array(p, dtype=np.double)
 *     if size is None:             # <<<<<<<<<<<<<<
 *         multi = np.broadcast(on, op)
 *         arr = np.empty(multi.shape, np.long)
 */
  __pyx_t_4 = (__pyx_v_size == nullptr);
  if (__pyx_t_4) {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":336
 *     op = np.array(p, dtype=np.double)
 *     if size is None:
 *         multi = np.broadcast(on, op)             # <<<<<<<<<<<<<<
 *         arr = np.empty(multi.shape, np.long)
 *         arr_data = <long *>dataptr(arr)
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_2 = __site_get_broadcast_336_18->Target(__site_get_broadcast_336_18, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call2_336_28->Target(__site_call2_336_28, __pyx_context, __pyx_t_2, __pyx_v_on, __pyx_v_op);
    __pyx_t_2 = nullptr;
    __pyx_v_multi = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":337
 *     if size is None:
 *         multi = np.broadcast(on, op)
 *         arr = np.empty(multi.shape, np.long)             # <<<<<<<<<<<<<<
 *         arr_data = <long *>dataptr(arr)
 *         for i from 0 <= i < multi.size:
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_2 = __site_get_empty_337_16->Target(__site_get_empty_337_16, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_get_shape_337_28->Target(__site_get_shape_337_28, __pyx_v_multi, __pyx_context);
    __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_long_337_38->Target(__site_get_long_337_38, __pyx_t_3, __pyx_context);
    __pyx_t_3 = nullptr;
    __pyx_t_3 = __site_call2_337_22->Target(__site_call2_337_22, __pyx_context, __pyx_t_2, __pyx_t_1, __pyx_t_5);
    __pyx_t_2 = nullptr;
    __pyx_t_1 = nullptr;
    __pyx_t_5 = nullptr;
    __pyx_v_arr = __pyx_t_3;
    __pyx_t_3 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":338
 *         multi = np.broadcast(on, op)
 *         arr = np.empty(multi.shape, np.long)
 *         arr_data = <long *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < multi.size:
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 */
    __pyx_v_arr_data = ((long *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":339
 *         arr = np.empty(multi.shape, np.long)
 *         arr_data = <long *>dataptr(arr)
 *         for i from 0 <= i < multi.size:             # <<<<<<<<<<<<<<
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 */
    __pyx_t_3 = __site_get_size_339_33->Target(__site_get_size_339_33, __pyx_v_multi, __pyx_context);
    __pyx_t_6 = __site_cvt_npy_intp_339_33->Target(__site_cvt_npy_intp_339_33, __pyx_t_3);
    __pyx_t_3 = nullptr;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_6; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":340
 *         arr_data = <long *>dataptr(arr)
 *         for i from 0 <= i < multi.size:
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 0)             # <<<<<<<<<<<<<<
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             arr_data[i] = func(state, on_data[0], op_data[0])
 */
      __pyx_v_on_data = ((long *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 0));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":341
 *         for i from 0 <= i < multi.size:
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)             # <<<<<<<<<<<<<<
 *             arr_data[i] = func(state, on_data[0], op_data[0])
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 */
      __pyx_v_op_data = ((double *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 1));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":342
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             arr_data[i] = func(state, on_data[0], op_data[0])             # <<<<<<<<<<<<<<
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 *     else:
 */
      (__pyx_v_arr_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, (__pyx_v_on_data[0]), (__pyx_v_op_data[0]));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":343
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             arr_data[i] = func(state, on_data[0], op_data[0])
 *             NpyArray_MultiIter_NEXT(getiter(multi))             # <<<<<<<<<<<<<<
 *     else:
 *         arr = np.empty(size, int)
 */
      NpyArray_MultiIter_NEXT(getiter(__pyx_v_multi));
    }
    goto __pyx_L3;
  }
  /*else*/ {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":345
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 *     else:
 *         arr = np.empty(size, int)             # <<<<<<<<<<<<<<
 *         arr_data = <long *>dataptr(arr)
 *         multi = np.broadcast(arr, on, op)
 */
    __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_empty_345_16->Target(__site_get_empty_345_16, __pyx_t_3, __pyx_context);
    __pyx_t_3 = nullptr;
    __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "int");
    __pyx_t_1 = __site_call2_345_22->Target(__site_call2_345_22, __pyx_context, __pyx_t_5, __pyx_v_size, ((System::Object^)__pyx_t_3));
    __pyx_t_5 = nullptr;
    __pyx_t_3 = nullptr;
    __pyx_v_arr = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":346
 *     else:
 *         arr = np.empty(size, int)
 *         arr_data = <long *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         multi = np.broadcast(arr, on, op)
 *         if multi.size != arr.size:
 */
    __pyx_v_arr_data = ((long *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":347
 *         arr = np.empty(size, int)
 *         arr_data = <long *>dataptr(arr)
 *         multi = np.broadcast(arr, on, op)             # <<<<<<<<<<<<<<
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_broadcast_347_18->Target(__site_get_broadcast_347_18, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call3_347_28->Target(__site_call3_347_28, __pyx_context, __pyx_t_3, __pyx_v_arr, __pyx_v_on, __pyx_v_op);
    __pyx_t_3 = nullptr;
    __pyx_v_multi = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":348
 *         arr_data = <long *>dataptr(arr)
 *         multi = np.broadcast(arr, on, op)
 *         if multi.size != arr.size:             # <<<<<<<<<<<<<<
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:
 */
    __pyx_t_1 = __site_get_size_348_16->Target(__site_get_size_348_16, __pyx_v_multi, __pyx_context);
    __pyx_t_3 = __site_get_size_348_28->Target(__site_get_size_348_28, __pyx_v_arr, __pyx_context);
    __pyx_t_5 = __site_op_ne_348_22->Target(__site_op_ne_348_22, __pyx_t_1, __pyx_t_3);
    __pyx_t_1 = nullptr;
    __pyx_t_3 = nullptr;
    __pyx_t_4 = __site_istrue_348_22->Target(__site_istrue_348_22, __pyx_t_5);
    __pyx_t_5 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":349
 *         multi = np.broadcast(arr, on, op)
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < multi.size:
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 */
      __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_3 = __site_call1_349_28->Target(__site_call1_349_28, __pyx_context, __pyx_t_5, ((System::Object^)"size is not compatible with inputs"));
      __pyx_t_5 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_3, nullptr, nullptr);
      __pyx_t_3 = nullptr;
      goto __pyx_L6;
    }
    __pyx_L6:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":350
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:             # <<<<<<<<<<<<<<
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 */
    __pyx_t_3 = __site_get_size_350_33->Target(__site_get_size_350_33, __pyx_v_multi, __pyx_context);
    __pyx_t_7 = __site_cvt_npy_intp_350_33->Target(__site_cvt_npy_intp_350_33, __pyx_t_3);
    __pyx_t_3 = nullptr;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_7; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":351
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 1)             # <<<<<<<<<<<<<<
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             arr_data[i] = func(state, on_data[0], op_data[0])
 */
      __pyx_v_on_data = ((long *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 1));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":352
 *         for i from 0 <= i < multi.size:
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)             # <<<<<<<<<<<<<<
 *             arr_data[i] = func(state, on_data[0], op_data[0])
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)
 */
      __pyx_v_op_data = ((double *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 2));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":353
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             arr_data[i] = func(state, on_data[0], op_data[0])             # <<<<<<<<<<<<<<
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 2)
 */
      (__pyx_v_arr_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, (__pyx_v_on_data[0]), (__pyx_v_op_data[0]));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":354
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             arr_data[i] = func(state, on_data[0], op_data[0])
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)             # <<<<<<<<<<<<<<
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 2)
 *     return arr
 */
      NpyArray_MultiIter_NEXTi(getiter(__pyx_v_multi), 1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":355
 *             arr_data[i] = func(state, on_data[0], op_data[0])
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 2)             # <<<<<<<<<<<<<<
 *     return arr
 * 
 */
      NpyArray_MultiIter_NEXTi(getiter(__pyx_v_multi), 2);
    }
  }
  __pyx_L3:;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":356
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 2)
 *     return arr             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = __pyx_v_arr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":359
 * 
 * 
 * cdef object discdd_array_sc(rk_state *state, rk_discdd func, object size,             # <<<<<<<<<<<<<<
 *                             double n, double p):
 *     cdef long *data
 */

static  System::Object^ discdd_array_sc(rk_state *__pyx_v_state, __pyx_t_6mtrand_rk_discdd __pyx_v_func, System::Object^ __pyx_v_size, double __pyx_v_n, double __pyx_v_p) {
  long *__pyx_v_data;
  npy_intp __pyx_v_length;
  npy_intp __pyx_v_i;
  System::Object^ __pyx_v_arr;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  System::Object^ __pyx_t_4 = nullptr;
  npy_intp __pyx_t_5;
  npy_intp __pyx_t_6;
  __pyx_v_arr = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":364
 *     cdef npy_intp length, i
 * 
 *     if size is None:             # <<<<<<<<<<<<<<
 *         return func(state, n, p)
 *     else:
 */
  __pyx_t_1 = (__pyx_v_size == nullptr);
  if (__pyx_t_1) {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":365
 * 
 *     if size is None:
 *         return func(state, n, p)             # <<<<<<<<<<<<<<
 *     else:
 *         arr = np.empty(size, int)
 */
    __pyx_t_2 = __pyx_v_func(__pyx_v_state, __pyx_v_n, __pyx_v_p);
    __pyx_r = __pyx_t_2;
    __pyx_t_2 = nullptr;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  /*else*/ {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":367
 *         return func(state, n, p)
 *     else:
 *         arr = np.empty(size, int)             # <<<<<<<<<<<<<<
 *         length = arr.size
 *         data = <long *>dataptr(arr)
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_empty_367_16->Target(__site_get_empty_367_16, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "int");
    __pyx_t_4 = __site_call2_367_22->Target(__site_call2_367_22, __pyx_context, __pyx_t_3, __pyx_v_size, ((System::Object^)__pyx_t_2));
    __pyx_t_3 = nullptr;
    __pyx_t_2 = nullptr;
    __pyx_v_arr = __pyx_t_4;
    __pyx_t_4 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":368
 *     else:
 *         arr = np.empty(size, int)
 *         length = arr.size             # <<<<<<<<<<<<<<
 *         data = <long *>dataptr(arr)
 *         for i from 0 <= i < length:
 */
    __pyx_t_4 = __site_get_size_368_20->Target(__site_get_size_368_20, __pyx_v_arr, __pyx_context);
    __pyx_t_5 = __site_cvt_npy_intp_368_20->Target(__site_cvt_npy_intp_368_20, __pyx_t_4);
    __pyx_t_4 = nullptr;
    __pyx_v_length = __pyx_t_5;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":369
 *         arr = np.empty(size, int)
 *         length = arr.size
 *         data = <long *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < length:
 *             data[i] = func(state, n, p)
 */
    __pyx_v_data = ((long *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":370
 *         length = arr.size
 *         data = <long *>dataptr(arr)
 *         for i from 0 <= i < length:             # <<<<<<<<<<<<<<
 *             data[i] = func(state, n, p)
 *         return arr
 */
    __pyx_t_6 = __pyx_v_length;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_6; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":371
 *         data = <long *>dataptr(arr)
 *         for i from 0 <= i < length:
 *             data[i] = func(state, n, p)             # <<<<<<<<<<<<<<
 *         return arr
 * 
 */
      (__pyx_v_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, __pyx_v_n, __pyx_v_p);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":372
 *         for i from 0 <= i < length:
 *             data[i] = func(state, n, p)
 *         return arr             # <<<<<<<<<<<<<<
 * 
 * 
 */
    __pyx_r = __pyx_v_arr;
    goto __pyx_L0;
  }
  __pyx_L3:;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":375
 * 
 * 
 * cdef object discdd_array(rk_state *state, rk_discdd func, object size,             # <<<<<<<<<<<<<<
 *                          object n, object p):
 *     cdef long *arr_data
 */

static  System::Object^ discdd_array(rk_state *__pyx_v_state, __pyx_t_6mtrand_rk_discdd __pyx_v_func, System::Object^ __pyx_v_size, System::Object^ __pyx_v_n, System::Object^ __pyx_v_p) {
  long *__pyx_v_arr_data;
  double *__pyx_v_op_data;
  double *__pyx_v_on_data;
  npy_intp __pyx_v_i;
  System::Object^ __pyx_v_on;
  System::Object^ __pyx_v_op;
  System::Object^ __pyx_v_multi;
  System::Object^ __pyx_v_arr;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  int __pyx_t_4;
  System::Object^ __pyx_t_5 = nullptr;
  npy_intp __pyx_t_6;
  npy_intp __pyx_t_7;
  __pyx_v_on = nullptr;
  __pyx_v_op = nullptr;
  __pyx_v_multi = nullptr;
  __pyx_v_arr = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":382
 *     cdef npy_intp length, i
 * 
 *     on = np.array(n, np.double)             # <<<<<<<<<<<<<<
 *     op = np.array(p, np.double)
 *     if size is None:
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_2 = __site_get_array_382_11->Target(__site_get_array_382_11, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_3 = __site_get_double_382_23->Target(__site_get_double_382_23, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call2_382_17->Target(__site_call2_382_17, __pyx_context, __pyx_t_2, __pyx_v_n, __pyx_t_3);
  __pyx_t_2 = nullptr;
  __pyx_t_3 = nullptr;
  __pyx_v_on = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":383
 * 
 *     on = np.array(n, np.double)
 *     op = np.array(p, np.double)             # <<<<<<<<<<<<<<
 *     if size is None:
 *         multi = np.broadcast(on, op)
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_3 = __site_get_array_383_11->Target(__site_get_array_383_11, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_2 = __site_get_double_383_23->Target(__site_get_double_383_23, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call2_383_17->Target(__site_call2_383_17, __pyx_context, __pyx_t_3, __pyx_v_p, __pyx_t_2);
  __pyx_t_3 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_v_op = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":384
 *     on = np.array(n, np.double)
 *     op = np.array(p, np.double)
 *     if size is None:             # <<<<<<<<<<<<<<
 *         multi = np.broadcast(on, op)
 *         arr = np.empty(multi.shape, np.long)
 */
  __pyx_t_4 = (__pyx_v_size == nullptr);
  if (__pyx_t_4) {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":385
 *     op = np.array(p, np.double)
 *     if size is None:
 *         multi = np.broadcast(on, op)             # <<<<<<<<<<<<<<
 *         arr = np.empty(multi.shape, np.long)
 *         arr_data = <long *>dataptr(arr)
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_2 = __site_get_broadcast_385_18->Target(__site_get_broadcast_385_18, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call2_385_28->Target(__site_call2_385_28, __pyx_context, __pyx_t_2, __pyx_v_on, __pyx_v_op);
    __pyx_t_2 = nullptr;
    __pyx_v_multi = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":386
 *     if size is None:
 *         multi = np.broadcast(on, op)
 *         arr = np.empty(multi.shape, np.long)             # <<<<<<<<<<<<<<
 *         arr_data = <long *>dataptr(arr)
 *         for i from 0 <= i < multi.size:
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_2 = __site_get_empty_386_16->Target(__site_get_empty_386_16, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_get_shape_386_28->Target(__site_get_shape_386_28, __pyx_v_multi, __pyx_context);
    __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_long_386_38->Target(__site_get_long_386_38, __pyx_t_3, __pyx_context);
    __pyx_t_3 = nullptr;
    __pyx_t_3 = __site_call2_386_22->Target(__site_call2_386_22, __pyx_context, __pyx_t_2, __pyx_t_1, __pyx_t_5);
    __pyx_t_2 = nullptr;
    __pyx_t_1 = nullptr;
    __pyx_t_5 = nullptr;
    __pyx_v_arr = __pyx_t_3;
    __pyx_t_3 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":387
 *         multi = np.broadcast(on, op)
 *         arr = np.empty(multi.shape, np.long)
 *         arr_data = <long *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < multi.size:
 *             on_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 */
    __pyx_v_arr_data = ((long *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":388
 *         arr = np.empty(multi.shape, np.long)
 *         arr_data = <long *>dataptr(arr)
 *         for i from 0 <= i < multi.size:             # <<<<<<<<<<<<<<
 *             on_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 */
    __pyx_t_3 = __site_get_size_388_33->Target(__site_get_size_388_33, __pyx_v_multi, __pyx_context);
    __pyx_t_6 = __site_cvt_npy_intp_388_33->Target(__site_cvt_npy_intp_388_33, __pyx_t_3);
    __pyx_t_3 = nullptr;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_6; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":389
 *         arr_data = <long *>dataptr(arr)
 *         for i from 0 <= i < multi.size:
 *             on_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 0)             # <<<<<<<<<<<<<<
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             arr_data[i] = func(state, on_data[0], op_data[0])
 */
      __pyx_v_on_data = ((double *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 0));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":390
 *         for i from 0 <= i < multi.size:
 *             on_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)             # <<<<<<<<<<<<<<
 *             arr_data[i] = func(state, on_data[0], op_data[0])
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 */
      __pyx_v_op_data = ((double *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 1));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":391
 *             on_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             arr_data[i] = func(state, on_data[0], op_data[0])             # <<<<<<<<<<<<<<
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 *     else:
 */
      (__pyx_v_arr_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, (__pyx_v_on_data[0]), (__pyx_v_op_data[0]));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":392
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             arr_data[i] = func(state, on_data[0], op_data[0])
 *             NpyArray_MultiIter_NEXT(getiter(multi))             # <<<<<<<<<<<<<<
 *     else:
 *         arr = np.empty(size, int)
 */
      NpyArray_MultiIter_NEXT(getiter(__pyx_v_multi));
    }
    goto __pyx_L3;
  }
  /*else*/ {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":394
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 *     else:
 *         arr = np.empty(size, int)             # <<<<<<<<<<<<<<
 *         arr_data = <long *>dataptr(arr)
 *         multi = np.broadcast(arr, on, op)
 */
    __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_empty_394_16->Target(__site_get_empty_394_16, __pyx_t_3, __pyx_context);
    __pyx_t_3 = nullptr;
    __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "int");
    __pyx_t_1 = __site_call2_394_22->Target(__site_call2_394_22, __pyx_context, __pyx_t_5, __pyx_v_size, ((System::Object^)__pyx_t_3));
    __pyx_t_5 = nullptr;
    __pyx_t_3 = nullptr;
    __pyx_v_arr = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":395
 *     else:
 *         arr = np.empty(size, int)
 *         arr_data = <long *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         multi = np.broadcast(arr, on, op)
 *         if multi.size != arr.size:
 */
    __pyx_v_arr_data = ((long *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":396
 *         arr = np.empty(size, int)
 *         arr_data = <long *>dataptr(arr)
 *         multi = np.broadcast(arr, on, op)             # <<<<<<<<<<<<<<
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_broadcast_396_18->Target(__site_get_broadcast_396_18, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call3_396_28->Target(__site_call3_396_28, __pyx_context, __pyx_t_3, __pyx_v_arr, __pyx_v_on, __pyx_v_op);
    __pyx_t_3 = nullptr;
    __pyx_v_multi = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":397
 *         arr_data = <long *>dataptr(arr)
 *         multi = np.broadcast(arr, on, op)
 *         if multi.size != arr.size:             # <<<<<<<<<<<<<<
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:
 */
    __pyx_t_1 = __site_get_size_397_16->Target(__site_get_size_397_16, __pyx_v_multi, __pyx_context);
    __pyx_t_3 = __site_get_size_397_28->Target(__site_get_size_397_28, __pyx_v_arr, __pyx_context);
    __pyx_t_5 = __site_op_ne_397_22->Target(__site_op_ne_397_22, __pyx_t_1, __pyx_t_3);
    __pyx_t_1 = nullptr;
    __pyx_t_3 = nullptr;
    __pyx_t_4 = __site_istrue_397_22->Target(__site_istrue_397_22, __pyx_t_5);
    __pyx_t_5 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":398
 *         multi = np.broadcast(arr, on, op)
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < multi.size:
 *             on_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 */
      __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_3 = __site_call1_398_28->Target(__site_call1_398_28, __pyx_context, __pyx_t_5, ((System::Object^)"size is not compatible with inputs"));
      __pyx_t_5 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_3, nullptr, nullptr);
      __pyx_t_3 = nullptr;
      goto __pyx_L6;
    }
    __pyx_L6:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":399
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:             # <<<<<<<<<<<<<<
 *             on_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 */
    __pyx_t_3 = __site_get_size_399_33->Target(__site_get_size_399_33, __pyx_v_multi, __pyx_context);
    __pyx_t_7 = __site_cvt_npy_intp_399_33->Target(__site_cvt_npy_intp_399_33, __pyx_t_3);
    __pyx_t_3 = nullptr;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_7; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":400
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:
 *             on_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)             # <<<<<<<<<<<<<<
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             arr_data[i] = func(state, on_data[0], op_data[0])
 */
      __pyx_v_on_data = ((double *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 1));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":401
 *         for i from 0 <= i < multi.size:
 *             on_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)             # <<<<<<<<<<<<<<
 *             arr_data[i] = func(state, on_data[0], op_data[0])
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)
 */
      __pyx_v_op_data = ((double *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 2));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":402
 *             on_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             arr_data[i] = func(state, on_data[0], op_data[0])             # <<<<<<<<<<<<<<
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 2)
 */
      (__pyx_v_arr_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, (__pyx_v_on_data[0]), (__pyx_v_op_data[0]));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":403
 *             op_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             arr_data[i] = func(state, on_data[0], op_data[0])
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)             # <<<<<<<<<<<<<<
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 2)
 *     return arr
 */
      NpyArray_MultiIter_NEXTi(getiter(__pyx_v_multi), 1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":404
 *             arr_data[i] = func(state, on_data[0], op_data[0])
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 2)             # <<<<<<<<<<<<<<
 *     return arr
 * 
 */
      NpyArray_MultiIter_NEXTi(getiter(__pyx_v_multi), 2);
    }
  }
  __pyx_L3:;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":405
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 2)
 *     return arr             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = __pyx_v_arr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":408
 * 
 * 
 * cdef object discnmN_array_sc(rk_state *state, rk_discnmN func, object size,             # <<<<<<<<<<<<<<
 *                              long n, long m, long N):
 *     cdef long *data
 */

static  System::Object^ discnmN_array_sc(rk_state *__pyx_v_state, __pyx_t_6mtrand_rk_discnmN __pyx_v_func, System::Object^ __pyx_v_size, long __pyx_v_n, long __pyx_v_m, long __pyx_v_N) {
  long *__pyx_v_data;
  npy_intp __pyx_v_length;
  npy_intp __pyx_v_i;
  System::Object^ __pyx_v_arr;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  System::Object^ __pyx_t_4 = nullptr;
  npy_intp __pyx_t_5;
  npy_intp __pyx_t_6;
  __pyx_v_arr = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":413
 *     cdef npy_intp length, i
 * 
 *     if size is None:             # <<<<<<<<<<<<<<
 *         return func(state, n, m, N)
 *     else:
 */
  __pyx_t_1 = (__pyx_v_size == nullptr);
  if (__pyx_t_1) {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":414
 * 
 *     if size is None:
 *         return func(state, n, m, N)             # <<<<<<<<<<<<<<
 *     else:
 *         arr = np.empty(size, int)
 */
    __pyx_t_2 = __pyx_v_func(__pyx_v_state, __pyx_v_n, __pyx_v_m, __pyx_v_N);
    __pyx_r = __pyx_t_2;
    __pyx_t_2 = nullptr;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  /*else*/ {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":416
 *         return func(state, n, m, N)
 *     else:
 *         arr = np.empty(size, int)             # <<<<<<<<<<<<<<
 *         length = arr.size
 *         data = <long *>dataptr(arr)
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_empty_416_16->Target(__site_get_empty_416_16, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "int");
    __pyx_t_4 = __site_call2_416_22->Target(__site_call2_416_22, __pyx_context, __pyx_t_3, __pyx_v_size, ((System::Object^)__pyx_t_2));
    __pyx_t_3 = nullptr;
    __pyx_t_2 = nullptr;
    __pyx_v_arr = __pyx_t_4;
    __pyx_t_4 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":417
 *     else:
 *         arr = np.empty(size, int)
 *         length = arr.size             # <<<<<<<<<<<<<<
 *         data = <long *>dataptr(arr)
 *         for i from 0 <= i < length:
 */
    __pyx_t_4 = __site_get_size_417_20->Target(__site_get_size_417_20, __pyx_v_arr, __pyx_context);
    __pyx_t_5 = __site_cvt_npy_intp_417_20->Target(__site_cvt_npy_intp_417_20, __pyx_t_4);
    __pyx_t_4 = nullptr;
    __pyx_v_length = __pyx_t_5;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":418
 *         arr = np.empty(size, int)
 *         length = arr.size
 *         data = <long *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < length:
 *             data[i] = func(state, n, m, N)
 */
    __pyx_v_data = ((long *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":419
 *         length = arr.size
 *         data = <long *>dataptr(arr)
 *         for i from 0 <= i < length:             # <<<<<<<<<<<<<<
 *             data[i] = func(state, n, m, N)
 *         return arr
 */
    __pyx_t_6 = __pyx_v_length;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_6; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":420
 *         data = <long *>dataptr(arr)
 *         for i from 0 <= i < length:
 *             data[i] = func(state, n, m, N)             # <<<<<<<<<<<<<<
 *         return arr
 * 
 */
      (__pyx_v_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, __pyx_v_n, __pyx_v_m, __pyx_v_N);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":421
 *         for i from 0 <= i < length:
 *             data[i] = func(state, n, m, N)
 *         return arr             # <<<<<<<<<<<<<<
 * 
 * 
 */
    __pyx_r = __pyx_v_arr;
    goto __pyx_L0;
  }
  __pyx_L3:;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":424
 * 
 * 
 * cdef object discnmN_array(rk_state *state, rk_discnmN func, object size,             # <<<<<<<<<<<<<<
 *                           object n, object m, object N):
 *     cdef long *arr_data
 */

static  System::Object^ discnmN_array(rk_state *__pyx_v_state, __pyx_t_6mtrand_rk_discnmN __pyx_v_func, System::Object^ __pyx_v_size, System::Object^ __pyx_v_n, System::Object^ __pyx_v_m, System::Object^ __pyx_v_N) {
  long *__pyx_v_arr_data;
  long *__pyx_v_on_data;
  long *__pyx_v_om_data;
  long *__pyx_v_oN_data;
  npy_intp __pyx_v_i;
  System::Object^ __pyx_v_on;
  System::Object^ __pyx_v_om;
  System::Object^ __pyx_v_oN;
  System::Object^ __pyx_v_multi;
  System::Object^ __pyx_v_arr;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  int __pyx_t_4;
  System::Object^ __pyx_t_5 = nullptr;
  npy_intp __pyx_t_6;
  npy_intp __pyx_t_7;
  __pyx_v_on = nullptr;
  __pyx_v_om = nullptr;
  __pyx_v_oN = nullptr;
  __pyx_v_multi = nullptr;
  __pyx_v_arr = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":432
 *     cdef npy_intp length, i
 * 
 *     on = np.array(n, np.long)             # <<<<<<<<<<<<<<
 *     om = np.array(m, np.long)
 *     oN = np.array(N, np.long)
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_2 = __site_get_array_432_11->Target(__site_get_array_432_11, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_3 = __site_get_long_432_23->Target(__site_get_long_432_23, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call2_432_17->Target(__site_call2_432_17, __pyx_context, __pyx_t_2, __pyx_v_n, __pyx_t_3);
  __pyx_t_2 = nullptr;
  __pyx_t_3 = nullptr;
  __pyx_v_on = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":433
 * 
 *     on = np.array(n, np.long)
 *     om = np.array(m, np.long)             # <<<<<<<<<<<<<<
 *     oN = np.array(N, np.long)
 *     if size is None:
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_3 = __site_get_array_433_11->Target(__site_get_array_433_11, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_2 = __site_get_long_433_23->Target(__site_get_long_433_23, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call2_433_17->Target(__site_call2_433_17, __pyx_context, __pyx_t_3, __pyx_v_m, __pyx_t_2);
  __pyx_t_3 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_v_om = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":434
 *     on = np.array(n, np.long)
 *     om = np.array(m, np.long)
 *     oN = np.array(N, np.long)             # <<<<<<<<<<<<<<
 *     if size is None:
 *         multi = np.broadcast(on, om, oN)
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_2 = __site_get_array_434_11->Target(__site_get_array_434_11, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_3 = __site_get_long_434_23->Target(__site_get_long_434_23, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call2_434_17->Target(__site_call2_434_17, __pyx_context, __pyx_t_2, __pyx_v_N, __pyx_t_3);
  __pyx_t_2 = nullptr;
  __pyx_t_3 = nullptr;
  __pyx_v_oN = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":435
 *     om = np.array(m, np.long)
 *     oN = np.array(N, np.long)
 *     if size is None:             # <<<<<<<<<<<<<<
 *         multi = np.broadcast(on, om, oN)
 *         arr = np.empty(multi.shape, np.long)
 */
  __pyx_t_4 = (__pyx_v_size == nullptr);
  if (__pyx_t_4) {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":436
 *     oN = np.array(N, np.long)
 *     if size is None:
 *         multi = np.broadcast(on, om, oN)             # <<<<<<<<<<<<<<
 *         arr = np.empty(multi.shape, np.long)
 *         arr_data = <long *>dataptr(arr)
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_broadcast_436_18->Target(__site_get_broadcast_436_18, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call3_436_28->Target(__site_call3_436_28, __pyx_context, __pyx_t_3, __pyx_v_on, __pyx_v_om, __pyx_v_oN);
    __pyx_t_3 = nullptr;
    __pyx_v_multi = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":437
 *     if size is None:
 *         multi = np.broadcast(on, om, oN)
 *         arr = np.empty(multi.shape, np.long)             # <<<<<<<<<<<<<<
 *         arr_data = <long *>dataptr(arr)
 *         for i from 0 <= i < multi.size:
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_empty_437_16->Target(__site_get_empty_437_16, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_get_shape_437_28->Target(__site_get_shape_437_28, __pyx_v_multi, __pyx_context);
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_long_437_38->Target(__site_get_long_437_38, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = __site_call2_437_22->Target(__site_call2_437_22, __pyx_context, __pyx_t_3, __pyx_t_1, __pyx_t_5);
    __pyx_t_3 = nullptr;
    __pyx_t_1 = nullptr;
    __pyx_t_5 = nullptr;
    __pyx_v_arr = __pyx_t_2;
    __pyx_t_2 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":438
 *         multi = np.broadcast(on, om, oN)
 *         arr = np.empty(multi.shape, np.long)
 *         arr_data = <long *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < multi.size:
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 */
    __pyx_v_arr_data = ((long *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":439
 *         arr = np.empty(multi.shape, np.long)
 *         arr_data = <long *>dataptr(arr)
 *         for i from 0 <= i < multi.size:             # <<<<<<<<<<<<<<
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 *             om_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 */
    __pyx_t_2 = __site_get_size_439_33->Target(__site_get_size_439_33, __pyx_v_multi, __pyx_context);
    __pyx_t_6 = __site_cvt_npy_intp_439_33->Target(__site_cvt_npy_intp_439_33, __pyx_t_2);
    __pyx_t_2 = nullptr;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_6; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":440
 *         arr_data = <long *>dataptr(arr)
 *         for i from 0 <= i < multi.size:
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 0)             # <<<<<<<<<<<<<<
 *             om_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             oN_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 */
      __pyx_v_on_data = ((long *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 0));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":441
 *         for i from 0 <= i < multi.size:
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 *             om_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 1)             # <<<<<<<<<<<<<<
 *             oN_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             arr_data[i] = func(state, on_data[0], om_data[0], oN_data[0])
 */
      __pyx_v_om_data = ((long *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 1));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":442
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 0)
 *             om_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             oN_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 2)             # <<<<<<<<<<<<<<
 *             arr_data[i] = func(state, on_data[0], om_data[0], oN_data[0])
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 */
      __pyx_v_oN_data = ((long *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 2));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":443
 *             om_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             oN_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             arr_data[i] = func(state, on_data[0], om_data[0], oN_data[0])             # <<<<<<<<<<<<<<
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 *     else:
 */
      (__pyx_v_arr_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, (__pyx_v_on_data[0]), (__pyx_v_om_data[0]), (__pyx_v_oN_data[0]));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":444
 *             oN_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             arr_data[i] = func(state, on_data[0], om_data[0], oN_data[0])
 *             NpyArray_MultiIter_NEXT(getiter(multi))             # <<<<<<<<<<<<<<
 *     else:
 *         arr = np.empty(size, int)
 */
      NpyArray_MultiIter_NEXT(getiter(__pyx_v_multi));
    }
    goto __pyx_L3;
  }
  /*else*/ {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":446
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 *     else:
 *         arr = np.empty(size, int)             # <<<<<<<<<<<<<<
 *         arr_data = <long *>dataptr(arr)
 *         multi = np.broadcast(arr, on, om, oN)
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_empty_446_16->Target(__site_get_empty_446_16, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "int");
    __pyx_t_1 = __site_call2_446_22->Target(__site_call2_446_22, __pyx_context, __pyx_t_5, __pyx_v_size, ((System::Object^)__pyx_t_2));
    __pyx_t_5 = nullptr;
    __pyx_t_2 = nullptr;
    __pyx_v_arr = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":447
 *     else:
 *         arr = np.empty(size, int)
 *         arr_data = <long *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         multi = np.broadcast(arr, on, om, oN)
 *         if multi.size != arr.size:
 */
    __pyx_v_arr_data = ((long *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":448
 *         arr = np.empty(size, int)
 *         arr_data = <long *>dataptr(arr)
 *         multi = np.broadcast(arr, on, om, oN)             # <<<<<<<<<<<<<<
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_2 = __site_get_broadcast_448_18->Target(__site_get_broadcast_448_18, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call4_448_28->Target(__site_call4_448_28, __pyx_context, __pyx_t_2, __pyx_v_arr, __pyx_v_on, __pyx_v_om, __pyx_v_oN);
    __pyx_t_2 = nullptr;
    __pyx_v_multi = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":449
 *         arr_data = <long *>dataptr(arr)
 *         multi = np.broadcast(arr, on, om, oN)
 *         if multi.size != arr.size:             # <<<<<<<<<<<<<<
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:
 */
    __pyx_t_1 = __site_get_size_449_16->Target(__site_get_size_449_16, __pyx_v_multi, __pyx_context);
    __pyx_t_2 = __site_get_size_449_28->Target(__site_get_size_449_28, __pyx_v_arr, __pyx_context);
    __pyx_t_5 = __site_op_ne_449_22->Target(__site_op_ne_449_22, __pyx_t_1, __pyx_t_2);
    __pyx_t_1 = nullptr;
    __pyx_t_2 = nullptr;
    __pyx_t_4 = __site_istrue_449_22->Target(__site_istrue_449_22, __pyx_t_5);
    __pyx_t_5 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":450
 *         multi = np.broadcast(arr, on, om, oN)
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < multi.size:
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 */
      __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_2 = __site_call1_450_28->Target(__site_call1_450_28, __pyx_context, __pyx_t_5, ((System::Object^)"size is not compatible with inputs"));
      __pyx_t_5 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_2, nullptr, nullptr);
      __pyx_t_2 = nullptr;
      goto __pyx_L6;
    }
    __pyx_L6:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":451
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:             # <<<<<<<<<<<<<<
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             om_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 */
    __pyx_t_2 = __site_get_size_451_33->Target(__site_get_size_451_33, __pyx_v_multi, __pyx_context);
    __pyx_t_7 = __site_cvt_npy_intp_451_33->Target(__site_cvt_npy_intp_451_33, __pyx_t_2);
    __pyx_t_2 = nullptr;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_7; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":452
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 1)             # <<<<<<<<<<<<<<
 *             om_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             oN_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 3)
 */
      __pyx_v_on_data = ((long *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 1));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":453
 *         for i from 0 <= i < multi.size:
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             om_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 2)             # <<<<<<<<<<<<<<
 *             oN_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 3)
 *             arr_data[i] = func(state, on_data[0], om_data[0], oN_data[0])
 */
      __pyx_v_om_data = ((long *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 2));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":454
 *             on_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             om_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             oN_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 3)             # <<<<<<<<<<<<<<
 *             arr_data[i] = func(state, on_data[0], om_data[0], oN_data[0])
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 */
      __pyx_v_oN_data = ((long *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 3));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":455
 *             om_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 2)
 *             oN_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 3)
 *             arr_data[i] = func(state, on_data[0], om_data[0], oN_data[0])             # <<<<<<<<<<<<<<
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 *     return arr
 */
      (__pyx_v_arr_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, (__pyx_v_on_data[0]), (__pyx_v_om_data[0]), (__pyx_v_oN_data[0]));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":456
 *             oN_data = <long *>NpyArray_MultiIter_DATA(getiter(multi), 3)
 *             arr_data[i] = func(state, on_data[0], om_data[0], oN_data[0])
 *             NpyArray_MultiIter_NEXT(getiter(multi))             # <<<<<<<<<<<<<<
 *     return arr
 * 
 */
      NpyArray_MultiIter_NEXT(getiter(__pyx_v_multi));
    }
  }
  __pyx_L3:;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":457
 *             arr_data[i] = func(state, on_data[0], om_data[0], oN_data[0])
 *             NpyArray_MultiIter_NEXT(getiter(multi))
 *     return arr             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = __pyx_v_arr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":460
 * 
 * 
 * cdef object discd_array_sc(rk_state *state, rk_discd func, object size,             # <<<<<<<<<<<<<<
 *                            double a):
 *     cdef long *arr_data
 */

static  System::Object^ discd_array_sc(rk_state *__pyx_v_state, __pyx_t_6mtrand_rk_discd __pyx_v_func, System::Object^ __pyx_v_size, double __pyx_v_a) {
  long *__pyx_v_arr_data;
  npy_intp __pyx_v_length;
  npy_intp __pyx_v_i;
  System::Object^ __pyx_v_arr;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  System::Object^ __pyx_t_4 = nullptr;
  npy_intp __pyx_t_5;
  npy_intp __pyx_t_6;
  __pyx_v_arr = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":465
 *     cdef npy_intp length, i
 * 
 *     if size is None:             # <<<<<<<<<<<<<<
 *         return func(state, a)
 *     else:
 */
  __pyx_t_1 = (__pyx_v_size == nullptr);
  if (__pyx_t_1) {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":466
 * 
 *     if size is None:
 *         return func(state, a)             # <<<<<<<<<<<<<<
 *     else:
 *         arr = np.empty(size, int)
 */
    __pyx_t_2 = __pyx_v_func(__pyx_v_state, __pyx_v_a);
    __pyx_r = __pyx_t_2;
    __pyx_t_2 = nullptr;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  /*else*/ {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":468
 *         return func(state, a)
 *     else:
 *         arr = np.empty(size, int)             # <<<<<<<<<<<<<<
 *         length = arr.size
 *         arr_data = <long *>dataptr(arr)
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_empty_468_16->Target(__site_get_empty_468_16, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "int");
    __pyx_t_4 = __site_call2_468_22->Target(__site_call2_468_22, __pyx_context, __pyx_t_3, __pyx_v_size, ((System::Object^)__pyx_t_2));
    __pyx_t_3 = nullptr;
    __pyx_t_2 = nullptr;
    __pyx_v_arr = __pyx_t_4;
    __pyx_t_4 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":469
 *     else:
 *         arr = np.empty(size, int)
 *         length = arr.size             # <<<<<<<<<<<<<<
 *         arr_data = <long *>dataptr(arr)
 *         for i from 0 <= i < length:
 */
    __pyx_t_4 = __site_get_size_469_20->Target(__site_get_size_469_20, __pyx_v_arr, __pyx_context);
    __pyx_t_5 = __site_cvt_npy_intp_469_20->Target(__site_cvt_npy_intp_469_20, __pyx_t_4);
    __pyx_t_4 = nullptr;
    __pyx_v_length = __pyx_t_5;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":470
 *         arr = np.empty(size, int)
 *         length = arr.size
 *         arr_data = <long *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < length:
 *             arr_data[i] = func(state, a)
 */
    __pyx_v_arr_data = ((long *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":471
 *         length = arr.size
 *         arr_data = <long *>dataptr(arr)
 *         for i from 0 <= i < length:             # <<<<<<<<<<<<<<
 *             arr_data[i] = func(state, a)
 *         return arr
 */
    __pyx_t_6 = __pyx_v_length;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_6; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":472
 *         arr_data = <long *>dataptr(arr)
 *         for i from 0 <= i < length:
 *             arr_data[i] = func(state, a)             # <<<<<<<<<<<<<<
 *         return arr
 * 
 */
      (__pyx_v_arr_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, __pyx_v_a);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":473
 *         for i from 0 <= i < length:
 *             arr_data[i] = func(state, a)
 *         return arr             # <<<<<<<<<<<<<<
 * 
 * 
 */
    __pyx_r = __pyx_v_arr;
    goto __pyx_L0;
  }
  __pyx_L3:;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":476
 * 
 * 
 * cdef object discd_array(rk_state *state, rk_discd func, object size,             # <<<<<<<<<<<<<<
 *                         object a):
 *     cdef long *arr_data
 */

static  System::Object^ discd_array(rk_state *__pyx_v_state, __pyx_t_6mtrand_rk_discd __pyx_v_func, System::Object^ __pyx_v_size, System::Object^ __pyx_v_a) {
  long *__pyx_v_arr_data;
  double *__pyx_v_oa_data;
  NpyArrayIterObject *__pyx_v_itera;
  npy_intp __pyx_v_length;
  npy_intp __pyx_v_i;
  System::Object^ __pyx_v_oa;
  System::Object^ __pyx_v_array;
  System::Object^ __pyx_v_arr;
  System::Object^ __pyx_v_multi;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  int __pyx_t_4;
  System::Object^ __pyx_t_5 = nullptr;
  npy_intp __pyx_t_6;
  npy_intp __pyx_t_7;
  __pyx_v_oa = nullptr;
  __pyx_v_array = nullptr;
  __pyx_v_arr = nullptr;
  __pyx_v_multi = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":483
 *     cdef npy_intp length, i
 * 
 *     oa = np.array(a, np.double)             # <<<<<<<<<<<<<<
 *     if size is None:
 *         array = np.empty(oa.shape, dtype=np.long)
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_2 = __site_get_array_483_11->Target(__site_get_array_483_11, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_3 = __site_get_double_483_23->Target(__site_get_double_483_23, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call2_483_17->Target(__site_call2_483_17, __pyx_context, __pyx_t_2, __pyx_v_a, __pyx_t_3);
  __pyx_t_2 = nullptr;
  __pyx_t_3 = nullptr;
  __pyx_v_oa = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":484
 * 
 *     oa = np.array(a, np.double)
 *     if size is None:             # <<<<<<<<<<<<<<
 *         array = np.empty(oa.shape, dtype=np.long)
 *         length = array.size
 */
  __pyx_t_4 = (__pyx_v_size == nullptr);
  if (__pyx_t_4) {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":485
 *     oa = np.array(a, np.double)
 *     if size is None:
 *         array = np.empty(oa.shape, dtype=np.long)             # <<<<<<<<<<<<<<
 *         length = array.size
 *         arr_data = <long *>dataptr(arr)
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_empty_485_18->Target(__site_get_empty_485_18, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_get_shape_485_27->Target(__site_get_shape_485_27, __pyx_v_oa, __pyx_context);
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_long_485_43->Target(__site_get_long_485_43, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = __site_call1_dtype_485_24->Target(__site_call1_dtype_485_24, __pyx_context, __pyx_t_3, __pyx_t_1, __pyx_t_5);
    __pyx_t_3 = nullptr;
    __pyx_t_1 = nullptr;
    __pyx_t_5 = nullptr;
    __pyx_v_array = __pyx_t_2;
    __pyx_t_2 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":486
 *     if size is None:
 *         array = np.empty(oa.shape, dtype=np.long)
 *         length = array.size             # <<<<<<<<<<<<<<
 *         arr_data = <long *>dataptr(arr)
 *         itera = NpyArray_IterNew(npy_array_from_py_array(oa))
 */
    __pyx_t_2 = __site_get_size_486_22->Target(__site_get_size_486_22, __pyx_v_array, __pyx_context);
    __pyx_t_6 = __site_cvt_npy_intp_486_22->Target(__site_cvt_npy_intp_486_22, __pyx_t_2);
    __pyx_t_2 = nullptr;
    __pyx_v_length = __pyx_t_6;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":487
 *         array = np.empty(oa.shape, dtype=np.long)
 *         length = array.size
 *         arr_data = <long *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         itera = NpyArray_IterNew(npy_array_from_py_array(oa))
 *         for i from 0 <= i < length:
 */
    __pyx_v_arr_data = ((long *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":488
 *         length = array.size
 *         arr_data = <long *>dataptr(arr)
 *         itera = NpyArray_IterNew(npy_array_from_py_array(oa))             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < length:
 *             arr_data[i] = func(state, (<double *>(itera.dataptr))[0])
 */
    __pyx_v_itera = NpyArray_IterNew(npy_array_from_py_array(__pyx_v_oa));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":489
 *         arr_data = <long *>dataptr(arr)
 *         itera = NpyArray_IterNew(npy_array_from_py_array(oa))
 *         for i from 0 <= i < length:             # <<<<<<<<<<<<<<
 *             arr_data[i] = func(state, (<double *>(itera.dataptr))[0])
 *             NpyArray_ITER_NEXT(itera)
 */
    __pyx_t_7 = __pyx_v_length;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_7; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":490
 *         itera = NpyArray_IterNew(npy_array_from_py_array(oa))
 *         for i from 0 <= i < length:
 *             arr_data[i] = func(state, (<double *>(itera.dataptr))[0])             # <<<<<<<<<<<<<<
 *             NpyArray_ITER_NEXT(itera)
 *     else:
 */
      (__pyx_v_arr_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, (((double *)__pyx_v_itera->dataptr)[0]));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":491
 *         for i from 0 <= i < length:
 *             arr_data[i] = func(state, (<double *>(itera.dataptr))[0])
 *             NpyArray_ITER_NEXT(itera)             # <<<<<<<<<<<<<<
 *     else:
 *         arr = np.empty(size, int)
 */
      NpyArray_ITER_NEXT(__pyx_v_itera);
    }
    goto __pyx_L3;
  }
  /*else*/ {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":493
 *             NpyArray_ITER_NEXT(itera)
 *     else:
 *         arr = np.empty(size, int)             # <<<<<<<<<<<<<<
 *         arr_data = <long *>dataptr(arr)
 *         multi = np.broadcast(arr, oa)
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_empty_493_16->Target(__site_get_empty_493_16, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "int");
    __pyx_t_1 = __site_call2_493_22->Target(__site_call2_493_22, __pyx_context, __pyx_t_5, __pyx_v_size, ((System::Object^)__pyx_t_2));
    __pyx_t_5 = nullptr;
    __pyx_t_2 = nullptr;
    __pyx_v_arr = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":494
 *     else:
 *         arr = np.empty(size, int)
 *         arr_data = <long *>dataptr(arr)             # <<<<<<<<<<<<<<
 *         multi = np.broadcast(arr, oa)
 *         if multi.size != arr.size:
 */
    __pyx_v_arr_data = ((long *)dataptr(__pyx_v_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":495
 *         arr = np.empty(size, int)
 *         arr_data = <long *>dataptr(arr)
 *         multi = np.broadcast(arr, oa)             # <<<<<<<<<<<<<<
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_2 = __site_get_broadcast_495_18->Target(__site_get_broadcast_495_18, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call2_495_28->Target(__site_call2_495_28, __pyx_context, __pyx_t_2, __pyx_v_arr, __pyx_v_oa);
    __pyx_t_2 = nullptr;
    __pyx_v_multi = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":496
 *         arr_data = <long *>dataptr(arr)
 *         multi = np.broadcast(arr, oa)
 *         if multi.size != arr.size:             # <<<<<<<<<<<<<<
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:
 */
    __pyx_t_1 = __site_get_size_496_16->Target(__site_get_size_496_16, __pyx_v_multi, __pyx_context);
    __pyx_t_2 = __site_get_size_496_28->Target(__site_get_size_496_28, __pyx_v_arr, __pyx_context);
    __pyx_t_5 = __site_op_ne_496_22->Target(__site_op_ne_496_22, __pyx_t_1, __pyx_t_2);
    __pyx_t_1 = nullptr;
    __pyx_t_2 = nullptr;
    __pyx_t_4 = __site_istrue_496_22->Target(__site_istrue_496_22, __pyx_t_5);
    __pyx_t_5 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":497
 *         multi = np.broadcast(arr, oa)
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")             # <<<<<<<<<<<<<<
 *         for i from 0 <= i < multi.size:
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 */
      __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_2 = __site_call1_497_28->Target(__site_call1_497_28, __pyx_context, __pyx_t_5, ((System::Object^)"size is not compatible with inputs"));
      __pyx_t_5 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_2, nullptr, nullptr);
      __pyx_t_2 = nullptr;
      goto __pyx_L6;
    }
    __pyx_L6:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":498
 *         if multi.size != arr.size:
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:             # <<<<<<<<<<<<<<
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             arr_data[i] = func(state, oa_data[0])
 */
    __pyx_t_2 = __site_get_size_498_33->Target(__site_get_size_498_33, __pyx_v_multi, __pyx_context);
    __pyx_t_7 = __site_cvt_npy_intp_498_33->Target(__site_cvt_npy_intp_498_33, __pyx_t_2);
    __pyx_t_2 = nullptr;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_7; __pyx_v_i++) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":499
 *             raise ValueError("size is not compatible with inputs")
 *         for i from 0 <= i < multi.size:
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)             # <<<<<<<<<<<<<<
 *             arr_data[i] = func(state, oa_data[0])
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)
 */
      __pyx_v_oa_data = ((double *)NpyArray_MultiIter_DATA(getiter(__pyx_v_multi), 1));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":500
 *         for i from 0 <= i < multi.size:
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             arr_data[i] = func(state, oa_data[0])             # <<<<<<<<<<<<<<
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)
 *     return arr
 */
      (__pyx_v_arr_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state, (__pyx_v_oa_data[0]));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":501
 *             oa_data = <double *>NpyArray_MultiIter_DATA(getiter(multi), 1)
 *             arr_data[i] = func(state, oa_data[0])
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)             # <<<<<<<<<<<<<<
 *     return arr
 * 
 */
      NpyArray_MultiIter_NEXTi(getiter(__pyx_v_multi), 1);
    }
  }
  __pyx_L3:;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":502
 *             arr_data[i] = func(state, oa_data[0])
 *             NpyArray_MultiIter_NEXTi(getiter(multi), 1)
 *     return arr             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = __pyx_v_arr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":505
 * 
 * 
 * cdef double kahan_sum(double *darr, long n):             # <<<<<<<<<<<<<<
 *     cdef double c, y, t, sum
 *     cdef long i
 */

static  double kahan_sum(double *__pyx_v_darr, long __pyx_v_n) {
  double __pyx_v_c;
  double __pyx_v_y;
  double __pyx_v_t;
  double __pyx_v_sum;
  long __pyx_v_i;
  double __pyx_r;
  long __pyx_t_1;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":509
 *     cdef long i
 * 
 *     sum = darr[0]             # <<<<<<<<<<<<<<
 *     c = 0.0
 *     for i from 1 <= i < n:
 */
  __pyx_v_sum = (__pyx_v_darr[0]);

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":510
 * 
 *     sum = darr[0]
 *     c = 0.0             # <<<<<<<<<<<<<<
 *     for i from 1 <= i < n:
 *         y = darr[i] - c
 */
  __pyx_v_c = 0.0;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":511
 *     sum = darr[0]
 *     c = 0.0
 *     for i from 1 <= i < n:             # <<<<<<<<<<<<<<
 *         y = darr[i] - c
 *         t = sum + y
 */
  __pyx_t_1 = __pyx_v_n;
  for (__pyx_v_i = 1; __pyx_v_i < __pyx_t_1; __pyx_v_i++) {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":512
 *     c = 0.0
 *     for i from 1 <= i < n:
 *         y = darr[i] - c             # <<<<<<<<<<<<<<
 *         t = sum + y
 *         c = (t - sum) - y
 */
    __pyx_v_y = ((__pyx_v_darr[__pyx_v_i]) - __pyx_v_c);

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":513
 *     for i from 1 <= i < n:
 *         y = darr[i] - c
 *         t = sum + y             # <<<<<<<<<<<<<<
 *         c = (t - sum) - y
 *         sum = t
 */
    __pyx_v_t = (__pyx_v_sum + __pyx_v_y);

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":514
 *         y = darr[i] - c
 *         t = sum + y
 *         c = (t - sum) - y             # <<<<<<<<<<<<<<
 *         sum = t
 *     return sum
 */
    __pyx_v_c = ((__pyx_v_t - __pyx_v_sum) - __pyx_v_y);

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":515
 *         t = sum + y
 *         c = (t - sum) - y
 *         sum = t             # <<<<<<<<<<<<<<
 *     return sum
 * 
 */
    __pyx_v_sum = __pyx_v_t;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":516
 *         c = (t - sum) - y
 *         sum = t
 *     return sum             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = __pyx_v_sum;
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":519
 * 
 * 
 * def flat_array(o, dtype):             # <<<<<<<<<<<<<<
 *     a = np.array(o, dtype=dtype)
 *     if len(a.shape) == 1:
 */

static System::Object^ flat_array(System::Object^ o, System::Object^ dtype) {
  System::Object^ __pyx_v_o = nullptr;
  System::Object^ __pyx_v_dtype = nullptr;
  System::Object^ __pyx_v_a;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  int __pyx_t_4;
  __pyx_v_o = o;
  __pyx_v_dtype = dtype;
  __pyx_v_a = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":520
 * 
 * def flat_array(o, dtype):
 *     a = np.array(o, dtype=dtype)             # <<<<<<<<<<<<<<
 *     if len(a.shape) == 1:
 *         return a
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_2 = __site_get_array_520_10->Target(__site_get_array_520_10, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call1_dtype_520_16->Target(__site_call1_dtype_520_16, __pyx_context, __pyx_t_2, __pyx_v_o, __pyx_v_dtype);
  __pyx_t_2 = nullptr;
  __pyx_v_a = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":521
 * def flat_array(o, dtype):
 *     a = np.array(o, dtype=dtype)
 *     if len(a.shape) == 1:             # <<<<<<<<<<<<<<
 *         return a
 *     else:
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "len");
  __pyx_t_2 = __site_get_shape_521_12->Target(__site_get_shape_521_12, __pyx_v_a, __pyx_context);
  __pyx_t_3 = __site_call1_521_10->Target(__site_call1_521_10, __pyx_context, __pyx_t_1, __pyx_t_2);
  __pyx_t_1 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_t_2 = __site_op_eq_521_20->Target(__site_op_eq_521_20, __pyx_t_3, __pyx_int_1);
  __pyx_t_3 = nullptr;
  __pyx_t_4 = __site_istrue_521_20->Target(__site_istrue_521_20, __pyx_t_2);
  __pyx_t_2 = nullptr;
  if (__pyx_t_4) {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":522
 *     a = np.array(o, dtype=dtype)
 *     if len(a.shape) == 1:
 *         return a             # <<<<<<<<<<<<<<
 *     else:
 *         return a.flatten()
 */
    __pyx_r = __pyx_v_a;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  /*else*/ {

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":524
 *         return a
 *     else:
 *         return a.flatten()             # <<<<<<<<<<<<<<
 * 
 * 
 */
    __pyx_t_2 = __site_get_flatten_524_16->Target(__site_get_flatten_524_16, __pyx_v_a, __pyx_context);
    __pyx_t_3 = __site_call0_524_24->Target(__site_call0_524_24, __pyx_context, __pyx_t_2);
    __pyx_t_2 = nullptr;
    __pyx_r = __pyx_t_3;
    __pyx_t_3 = nullptr;
    goto __pyx_L0;
  }
  __pyx_L5:;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":527
 * 
 * 
 * cdef class RandomState:             # <<<<<<<<<<<<<<
 *     """
 *     RandomState(seed=None)
 */
[PythonType]
ref struct RandomState {
  rk_state *internal_state;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":562
 *     cdef rk_state *internal_state
 * 
 *     def __cinit__(self, seed=None):             # <<<<<<<<<<<<<<
 *         self.internal_state = <rk_state *>malloc(sizeof(rk_state))
 * 
 */

  RandomState([InteropServices::Optional]System::Object^ seed) {
    System::Object^ __pyx_v_seed = nullptr;
    int __pyx_r;
    System::Object^ __pyx_t_1 = nullptr;
    System::Object^ __pyx_t_2 = nullptr;
    System::Object^ __pyx_v_self = this;
    if (dynamic_cast<System::Reflection::Missing^>(seed) == nullptr) {
      __pyx_v_seed = seed;
    } else {
      __pyx_v_seed = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":563
 * 
 *     def __cinit__(self, seed=None):
 *         self.internal_state = <rk_state *>malloc(sizeof(rk_state))             # <<<<<<<<<<<<<<
 * 
 *         self.seed(seed)
 */
    ((RandomState^)__pyx_v_self)->internal_state = ((rk_state *)malloc((sizeof(rk_state))));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":565
 *         self.internal_state = <rk_state *>malloc(sizeof(rk_state))
 * 
 *         self.seed(seed)             # <<<<<<<<<<<<<<
 * 
 *     def __dealloc__(self):
 */
    __pyx_t_1 = __site_get_seed_565_12->Target(__site_get_seed_565_12, __pyx_v_self, __pyx_context);
    __pyx_t_2 = __site_call1_565_17->Target(__site_call1_565_17, __pyx_context, __pyx_t_1, __pyx_v_seed);
    __pyx_t_1 = nullptr;
    __pyx_t_2 = nullptr;

    __pyx_r = 0;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":567
 *         self.seed(seed)
 * 
 *     def __dealloc__(self):             # <<<<<<<<<<<<<<
 *         if self.internal_state != NULL:
 *             free(self.internal_state)
 */

  !RandomState() {
    int __pyx_t_1;
    System::Object^ __pyx_v_self = this;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":568
 * 
 *     def __dealloc__(self):
 *         if self.internal_state != NULL:             # <<<<<<<<<<<<<<
 *             free(self.internal_state)
 *             self.internal_state = NULL
 */
    __pyx_t_1 = (((RandomState^)__pyx_v_self)->internal_state != NULL);
    if (__pyx_t_1) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":569
 *     def __dealloc__(self):
 *         if self.internal_state != NULL:
 *             free(self.internal_state)             # <<<<<<<<<<<<<<
 *             self.internal_state = NULL
 * 
 */
      free(((RandomState^)__pyx_v_self)->internal_state);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":570
 *         if self.internal_state != NULL:
 *             free(self.internal_state)
 *             self.internal_state = NULL             # <<<<<<<<<<<<<<
 * 
 *     def seed(self, seed=None):
 */
      ((RandomState^)__pyx_v_self)->internal_state = NULL;
      goto __pyx_L5;
    }
    __pyx_L5:;

  }
  ~RandomState() { this->!RandomState(); }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":572
 *             self.internal_state = NULL
 * 
 *     def seed(self, seed=None):             # <<<<<<<<<<<<<<
 *         """
 *         seed(seed=None)
 */

  virtual System::Object^ seed([InteropServices::Optional]System::Object^ seed) {
    System::Object^ __pyx_v_seed = nullptr;
    rk_error __pyx_v_errcode;
    System::Object^ __pyx_v_iseed;
    System::Object^ __pyx_v_obj;
    System::Object^ __pyx_r = nullptr;
    int __pyx_t_1;
    System::Object^ __pyx_t_2 = nullptr;
    System::Object^ __pyx_t_3 = nullptr;
    unsigned long __pyx_t_4;
    System::Object^ __pyx_t_5 = nullptr;
    unsigned long __pyx_t_6;
    System::Object^ __pyx_v_self = this;
    if (dynamic_cast<System::Reflection::Missing^>(seed) == nullptr) {
      __pyx_v_seed = seed;
    } else {
      __pyx_v_seed = ((System::Object^)nullptr);
    }
    __pyx_v_iseed = nullptr;
    __pyx_v_obj = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":593
 *         cdef rk_error errcode
 * 
 *         if seed is None:             # <<<<<<<<<<<<<<
 *             errcode = rk_randomseed(self.internal_state)
 *         elif type(seed) is int:
 */
    __pyx_t_1 = (__pyx_v_seed == nullptr);
    if (__pyx_t_1) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":594
 * 
 *         if seed is None:
 *             errcode = rk_randomseed(self.internal_state)             # <<<<<<<<<<<<<<
 *         elif type(seed) is int:
 *             rk_seed(seed, self.internal_state)
 */
      __pyx_v_errcode = rk_randomseed(((RandomState^)__pyx_v_self)->internal_state);
      goto __pyx_L5;
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":595
 *         if seed is None:
 *             errcode = rk_randomseed(self.internal_state)
 *         elif type(seed) is int:             # <<<<<<<<<<<<<<
 *             rk_seed(seed, self.internal_state)
 *         elif isinstance(seed, np.integer):
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "type");
    __pyx_t_3 = __site_call1_595_17->Target(__site_call1_595_17, __pyx_context, ((System::Object^)__pyx_t_2), __pyx_v_seed);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "int");
    __pyx_t_1 = (((System::Object^)__pyx_t_3) == __pyx_t_2);
    __pyx_t_3 = nullptr;
    __pyx_t_2 = nullptr;
    if (__pyx_t_1) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":596
 *             errcode = rk_randomseed(self.internal_state)
 *         elif type(seed) is int:
 *             rk_seed(seed, self.internal_state)             # <<<<<<<<<<<<<<
 *         elif isinstance(seed, np.integer):
 *             iseed = int(seed)
 */
      __pyx_t_4 = __site_cvt_unsigned_long_596_24->Target(__site_cvt_unsigned_long_596_24, __pyx_v_seed);
      rk_seed(__pyx_t_4, ((RandomState^)__pyx_v_self)->internal_state);
      goto __pyx_L5;
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":597
 *         elif type(seed) is int:
 *             rk_seed(seed, self.internal_state)
 *         elif isinstance(seed, np.integer):             # <<<<<<<<<<<<<<
 *             iseed = int(seed)
 *             rk_seed(iseed, self.internal_state)
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "isinstance");
    __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_integer_597_32->Target(__site_get_integer_597_32, __pyx_t_3, __pyx_context);
    __pyx_t_3 = nullptr;
    __pyx_t_3 = __site_call2_597_23->Target(__site_call2_597_23, __pyx_context, __pyx_t_2, __pyx_v_seed, __pyx_t_5);
    __pyx_t_2 = nullptr;
    __pyx_t_5 = nullptr;
    __pyx_t_1 = __site_istrue_597_23->Target(__site_istrue_597_23, __pyx_t_3);
    __pyx_t_3 = nullptr;
    if (__pyx_t_1) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":598
 *             rk_seed(seed, self.internal_state)
 *         elif isinstance(seed, np.integer):
 *             iseed = int(seed)             # <<<<<<<<<<<<<<
 *             rk_seed(iseed, self.internal_state)
 *         else:
 */
      __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "int");
      __pyx_t_5 = __site_call1_598_23->Target(__site_call1_598_23, __pyx_context, ((System::Object^)__pyx_t_3), __pyx_v_seed);
      __pyx_t_3 = nullptr;
      __pyx_v_iseed = __pyx_t_5;
      __pyx_t_5 = nullptr;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":599
 *         elif isinstance(seed, np.integer):
 *             iseed = int(seed)
 *             rk_seed(iseed, self.internal_state)             # <<<<<<<<<<<<<<
 *         else:
 *             obj = flat_array(seed, np.long)
 */
      __pyx_t_6 = __site_cvt_unsigned_long_599_25->Target(__site_cvt_unsigned_long_599_25, __pyx_v_iseed);
      rk_seed(__pyx_t_6, ((RandomState^)__pyx_v_self)->internal_state);
      goto __pyx_L5;
    }
    /*else*/ {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":601
 *             rk_seed(iseed, self.internal_state)
 *         else:
 *             obj = flat_array(seed, np.long)             # <<<<<<<<<<<<<<
 *             init_by_array(self.internal_state,
 *                           <unsigned long *>dataptr(obj),
 */
      __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "flat_array");
      __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "np");
      __pyx_t_2 = __site_get_long_601_37->Target(__site_get_long_601_37, __pyx_t_3, __pyx_context);
      __pyx_t_3 = nullptr;
      __pyx_t_3 = __site_call2_601_28->Target(__site_call2_601_28, __pyx_context, __pyx_t_5, __pyx_v_seed, __pyx_t_2);
      __pyx_t_5 = nullptr;
      __pyx_t_2 = nullptr;
      __pyx_v_obj = __pyx_t_3;
      __pyx_t_3 = nullptr;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":604
 *             init_by_array(self.internal_state,
 *                           <unsigned long *>dataptr(obj),
 *                           npy_array_from_py_array(obj).dimensions[0])             # <<<<<<<<<<<<<<
 * 
 *     def get_state(self):
 */
      init_by_array(((RandomState^)__pyx_v_self)->internal_state, ((unsigned long *)dataptr(__pyx_v_obj)), (npy_array_from_py_array(__pyx_v_obj)->dimensions[0]));
    }
    __pyx_L5:;

    __pyx_r = nullptr;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":606
 *                           npy_array_from_py_array(obj).dimensions[0])
 * 
 *     def get_state(self):             # <<<<<<<<<<<<<<
 *         """
 *         get_state()
 */

  virtual System::Object^ get_state() {
    System::Object^ __pyx_v_state;
    System::Object^ __pyx_r = nullptr;
    System::Object^ __pyx_t_1 = nullptr;
    System::Object^ __pyx_t_2 = nullptr;
    System::Object^ __pyx_t_3 = nullptr;
    System::Object^ __pyx_t_4 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_state = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":636
 * 
 *         """
 *         state = np.empty(624, np.uint)             # <<<<<<<<<<<<<<
 *         memcpy(<void *>dataptr(state),
 *                <void *>(self.internal_state.key),
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_2 = __site_get_empty_636_18->Target(__site_get_empty_636_18, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_uint_636_32->Target(__site_get_uint_636_32, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call2_636_24->Target(__site_call2_636_24, __pyx_context, __pyx_t_2, __pyx_int_624, __pyx_t_3);
    __pyx_t_2 = nullptr;
    __pyx_t_3 = nullptr;
    __pyx_v_state = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":639
 *         memcpy(<void *>dataptr(state),
 *                <void *>(self.internal_state.key),
 *                624 * sizeof(long))             # <<<<<<<<<<<<<<
 *         state = np.asarray(state, np.uint32)
 *         return ('MT19937', state, self.internal_state.pos,
 */
    memcpy(((void *)dataptr(__pyx_v_state)), ((void *)((RandomState^)__pyx_v_self)->internal_state->key), (624 * (sizeof(long))));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":640
 *                <void *>(self.internal_state.key),
 *                624 * sizeof(long))
 *         state = np.asarray(state, np.uint32)             # <<<<<<<<<<<<<<
 *         return ('MT19937', state, self.internal_state.pos,
 *             self.internal_state.has_gauss, self.internal_state.gauss)
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_3 = __site_get_asarray_640_18->Target(__site_get_asarray_640_18, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_2 = __site_get_uint32_640_36->Target(__site_get_uint32_640_36, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call2_640_26->Target(__site_call2_640_26, __pyx_context, __pyx_t_3, __pyx_v_state, __pyx_t_2);
    __pyx_t_3 = nullptr;
    __pyx_t_2 = nullptr;
    __pyx_v_state = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":641
 *                624 * sizeof(long))
 *         state = np.asarray(state, np.uint32)
 *         return ('MT19937', state, self.internal_state.pos,             # <<<<<<<<<<<<<<
 *             self.internal_state.has_gauss, self.internal_state.gauss)
 * 
 */
    __pyx_t_1 = ((RandomState^)__pyx_v_self)->internal_state->pos;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":642
 *         state = np.asarray(state, np.uint32)
 *         return ('MT19937', state, self.internal_state.pos,
 *             self.internal_state.has_gauss, self.internal_state.gauss)             # <<<<<<<<<<<<<<
 * 
 *     def set_state(self, state):
 */
    __pyx_t_2 = ((RandomState^)__pyx_v_self)->internal_state->has_gauss;
    __pyx_t_3 = ((RandomState^)__pyx_v_self)->internal_state->gauss;
    __pyx_t_4 = PythonOps::MakeTuple(gcnew array<System::Object^>{((System::Object^)"MT19937"), __pyx_v_state, __pyx_t_1, __pyx_t_2, __pyx_t_3});
    __pyx_t_1 = nullptr;
    __pyx_t_2 = nullptr;
    __pyx_t_3 = nullptr;
    __pyx_r = __pyx_t_4;
    __pyx_t_4 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":644
 *             self.internal_state.has_gauss, self.internal_state.gauss)
 * 
 *     def set_state(self, state):             # <<<<<<<<<<<<<<
 *         """
 *         set_state(state)
 */

  virtual System::Object^ set_state(System::Object^ state) {
    System::Object^ __pyx_v_state = nullptr;
    int __pyx_v_pos;
    System::Object^ __pyx_v_algorithm_name;
    System::Object^ __pyx_v_key;
    System::Object^ __pyx_v_has_gauss;
    System::Object^ __pyx_v_cached_gaussian;
    System::Object^ __pyx_v_obj;
    System::Object^ __pyx_r = nullptr;
    System::Object^ __pyx_t_1 = nullptr;
    int __pyx_t_2;
    System::Object^ __pyx_t_3 = nullptr;
    array<System::Object^>^ __pyx_t_4;
    System::Object^ __pyx_t_5 = nullptr;
    int __pyx_t_6;
    int __pyx_t_7;
    double __pyx_t_8;
    System::Object^ __pyx_v_self = this;
    __pyx_v_state = state;
    __pyx_v_algorithm_name = nullptr;
    __pyx_v_key = nullptr;
    __pyx_v_has_gauss = nullptr;
    __pyx_v_cached_gaussian = nullptr;
    __pyx_v_obj = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":693
 *         cdef int pos
 * 
 *         algorithm_name = state[0]             # <<<<<<<<<<<<<<
 *         if algorithm_name != 'MT19937':
 *             raise ValueError("algorithm must be 'MT19937'")
 */
    __pyx_t_1 = __site_getindex_693_30->Target(__site_getindex_693_30, __pyx_v_state, ((System::Object^)0));
    __pyx_v_algorithm_name = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":694
 * 
 *         algorithm_name = state[0]
 *         if algorithm_name != 'MT19937':             # <<<<<<<<<<<<<<
 *             raise ValueError("algorithm must be 'MT19937'")
 *         key, pos = state[1:3]
 */
    __pyx_t_1 = __site_op_ne_694_26->Target(__site_op_ne_694_26, __pyx_v_algorithm_name, ((System::Object^)"MT19937"));
    __pyx_t_2 = __site_istrue_694_26->Target(__site_istrue_694_26, __pyx_t_1);
    __pyx_t_1 = nullptr;
    if (__pyx_t_2) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":695
 *         algorithm_name = state[0]
 *         if algorithm_name != 'MT19937':
 *             raise ValueError("algorithm must be 'MT19937'")             # <<<<<<<<<<<<<<
 *         key, pos = state[1:3]
 *         if len(state) == 3:
 */
      __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_3 = __site_call1_695_28->Target(__site_call1_695_28, __pyx_context, __pyx_t_1, ((System::Object^)"algorithm must be 'MT19937'"));
      __pyx_t_1 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_3, nullptr, nullptr);
      __pyx_t_3 = nullptr;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":696
 *         if algorithm_name != 'MT19937':
 *             raise ValueError("algorithm must be 'MT19937'")
 *         key, pos = state[1:3]             # <<<<<<<<<<<<<<
 *         if len(state) == 3:
 *             has_gauss = 0
 */
    __pyx_t_3 = __site_getslice_696_24->Target(__site_getslice_696_24, __pyx_v_state, 1, 3);
    __pyx_t_4 = safe_cast< array<System::Object^>^ >(LightExceptions::CheckAndThrow(PythonOps::GetEnumeratorValuesNoComplexSets(__pyx_context, __pyx_t_3, 2)));
    __pyx_t_1 = __pyx_t_4[0];
    __pyx_t_5 = __pyx_t_4[1];
    __pyx_t_6 = __site_cvt_int_696_8->Target(__site_cvt_int_696_8, __pyx_t_5);
    __pyx_t_5 = nullptr;
    __pyx_t_3 = nullptr;
    __pyx_t_4 = nullptr;
    __pyx_v_key = __pyx_t_1;
    __pyx_t_1 = nullptr;
    __pyx_v_pos = __pyx_t_6;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":697
 *             raise ValueError("algorithm must be 'MT19937'")
 *         key, pos = state[1:3]
 *         if len(state) == 3:             # <<<<<<<<<<<<<<
 *             has_gauss = 0
 *             cached_gaussian = 0.0
 */
    __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "len");
    __pyx_t_5 = __site_call1_697_14->Target(__site_call1_697_14, __pyx_context, __pyx_t_3, __pyx_v_state);
    __pyx_t_3 = nullptr;
    __pyx_t_3 = __site_op_eq_697_22->Target(__site_op_eq_697_22, __pyx_t_5, __pyx_int_3);
    __pyx_t_5 = nullptr;
    __pyx_t_2 = __site_istrue_697_22->Target(__site_istrue_697_22, __pyx_t_3);
    __pyx_t_3 = nullptr;
    if (__pyx_t_2) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":698
 *         key, pos = state[1:3]
 *         if len(state) == 3:
 *             has_gauss = 0             # <<<<<<<<<<<<<<
 *             cached_gaussian = 0.0
 *         else:
 */
      __pyx_v_has_gauss = __pyx_int_0;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":699
 *         if len(state) == 3:
 *             has_gauss = 0
 *             cached_gaussian = 0.0             # <<<<<<<<<<<<<<
 *         else:
 *             has_gauss, cached_gaussian = state[3:5]
 */
      __pyx_t_3 = 0.0;
      __pyx_v_cached_gaussian = __pyx_t_3;
      __pyx_t_3 = nullptr;
      goto __pyx_L6;
    }
    /*else*/ {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":701
 *             cached_gaussian = 0.0
 *         else:
 *             has_gauss, cached_gaussian = state[3:5]             # <<<<<<<<<<<<<<
 * 
 *         obj = flat_array(key, np.uint)
 */
      __pyx_t_3 = __site_getslice_701_46->Target(__site_getslice_701_46, __pyx_v_state, 3, 5);
      __pyx_t_4 = safe_cast< array<System::Object^>^ >(LightExceptions::CheckAndThrow(PythonOps::GetEnumeratorValuesNoComplexSets(__pyx_context, __pyx_t_3, 2)));
      __pyx_t_5 = __pyx_t_4[0];
      __pyx_t_1 = __pyx_t_4[1];
      __pyx_t_3 = nullptr;
      __pyx_t_4 = nullptr;
      __pyx_v_has_gauss = __pyx_t_5;
      __pyx_t_5 = nullptr;
      __pyx_v_cached_gaussian = __pyx_t_1;
      __pyx_t_1 = nullptr;
    }
    __pyx_L6:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":703
 *             has_gauss, cached_gaussian = state[3:5]
 * 
 *         obj = flat_array(key, np.uint)             # <<<<<<<<<<<<<<
 *         if obj.shape[0] != 624:
 *             raise ValueError("state must be 624 longs")
 */
    __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "flat_array");
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_uint_703_32->Target(__site_get_uint_703_32, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call2_703_24->Target(__site_call2_703_24, __pyx_context, __pyx_t_3, __pyx_v_key, __pyx_t_5);
    __pyx_t_3 = nullptr;
    __pyx_t_5 = nullptr;
    __pyx_v_obj = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":704
 * 
 *         obj = flat_array(key, np.uint)
 *         if obj.shape[0] != 624:             # <<<<<<<<<<<<<<
 *             raise ValueError("state must be 624 longs")
 *         memcpy(<void *>(self.internal_state.key),
 */
    __pyx_t_1 = __site_get_shape_704_14->Target(__site_get_shape_704_14, __pyx_v_obj, __pyx_context);
    __pyx_t_5 = __site_getindex_704_20->Target(__site_getindex_704_20, __pyx_t_1, ((System::Object^)0));
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_op_ne_704_24->Target(__site_op_ne_704_24, __pyx_t_5, __pyx_int_624);
    __pyx_t_5 = nullptr;
    __pyx_t_2 = __site_istrue_704_24->Target(__site_istrue_704_24, __pyx_t_1);
    __pyx_t_1 = nullptr;
    if (__pyx_t_2) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":705
 *         obj = flat_array(key, np.uint)
 *         if obj.shape[0] != 624:
 *             raise ValueError("state must be 624 longs")             # <<<<<<<<<<<<<<
 *         memcpy(<void *>(self.internal_state.key),
 *                <void *>dataptr(obj),
 */
      __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_5 = __site_call1_705_28->Target(__site_call1_705_28, __pyx_context, __pyx_t_1, ((System::Object^)"state must be 624 longs"));
      __pyx_t_1 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_5, nullptr, nullptr);
      __pyx_t_5 = nullptr;
      goto __pyx_L7;
    }
    __pyx_L7:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":708
 *         memcpy(<void *>(self.internal_state.key),
 *                <void *>dataptr(obj),
 *                624 * sizeof(long))             # <<<<<<<<<<<<<<
 *         self.internal_state.pos = pos
 *         self.internal_state.has_gauss = has_gauss
 */
    memcpy(((void *)((RandomState^)__pyx_v_self)->internal_state->key), ((void *)dataptr(__pyx_v_obj)), (624 * (sizeof(long))));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":709
 *                <void *>dataptr(obj),
 *                624 * sizeof(long))
 *         self.internal_state.pos = pos             # <<<<<<<<<<<<<<
 *         self.internal_state.has_gauss = has_gauss
 *         self.internal_state.gauss = cached_gaussian
 */
    ((RandomState^)__pyx_v_self)->internal_state->pos = __pyx_v_pos;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":710
 *                624 * sizeof(long))
 *         self.internal_state.pos = pos
 *         self.internal_state.has_gauss = has_gauss             # <<<<<<<<<<<<<<
 *         self.internal_state.gauss = cached_gaussian
 * 
 */
    __pyx_t_7 = __site_cvt_int_710_49->Target(__site_cvt_int_710_49, __pyx_v_has_gauss);
    ((RandomState^)__pyx_v_self)->internal_state->has_gauss = __pyx_t_7;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":711
 *         self.internal_state.pos = pos
 *         self.internal_state.has_gauss = has_gauss
 *         self.internal_state.gauss = cached_gaussian             # <<<<<<<<<<<<<<
 * 
 *     # Pickling support:
 */
    __pyx_t_8 = __site_cvt_double_711_51->Target(__site_cvt_double_711_51, __pyx_v_cached_gaussian);
    ((RandomState^)__pyx_v_self)->internal_state->gauss = __pyx_t_8;

    __pyx_r = nullptr;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":714
 * 
 *     # Pickling support:
 *     def __getstate__(self):             # <<<<<<<<<<<<<<
 *         return self.get_state()
 * 
 */

  virtual System::Object^ __getstate__() {
    System::Object^ __pyx_r = nullptr;
    System::Object^ __pyx_t_1 = nullptr;
    System::Object^ __pyx_t_2 = nullptr;
    System::Object^ __pyx_v_self = this;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":715
 *     # Pickling support:
 *     def __getstate__(self):
 *         return self.get_state()             # <<<<<<<<<<<<<<
 * 
 *     def __setstate__(self, state):
 */
    __pyx_t_1 = __site_get_get_state_715_19->Target(__site_get_get_state_715_19, __pyx_v_self, __pyx_context);
    __pyx_t_2 = __site_call0_715_29->Target(__site_call0_715_29, __pyx_context, __pyx_t_1);
    __pyx_t_1 = nullptr;
    __pyx_r = __pyx_t_2;
    __pyx_t_2 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":717
 *         return self.get_state()
 * 
 *     def __setstate__(self, state):             # <<<<<<<<<<<<<<
 *         self.set_state(state)
 * 
 */

  virtual System::Object^ __setstate__(System::Object^ state) {
    System::Object^ __pyx_v_state = nullptr;
    System::Object^ __pyx_r = nullptr;
    System::Object^ __pyx_t_1 = nullptr;
    System::Object^ __pyx_t_2 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_state = state;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":718
 * 
 *     def __setstate__(self, state):
 *         self.set_state(state)             # <<<<<<<<<<<<<<
 * 
 *     def __reduce__(self):
 */
    __pyx_t_1 = __site_get_set_state_718_12->Target(__site_get_set_state_718_12, __pyx_v_self, __pyx_context);
    __pyx_t_2 = __site_call1_718_22->Target(__site_call1_718_22, __pyx_context, __pyx_t_1, __pyx_v_state);
    __pyx_t_1 = nullptr;
    __pyx_t_2 = nullptr;

    __pyx_r = nullptr;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":720
 *         self.set_state(state)
 * 
 *     def __reduce__(self):             # <<<<<<<<<<<<<<
 *         return (np.random.__RandomState_ctor, (), self.get_state())
 * 
 */

  virtual System::Object^ __reduce__() {
    System::Object^ __pyx_r = nullptr;
    System::Object^ __pyx_t_1 = nullptr;
    System::Object^ __pyx_t_2 = nullptr;
    System::Object^ __pyx_t_3 = nullptr;
    System::Object^ __pyx_v_self = this;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":721
 * 
 *     def __reduce__(self):
 *         return (np.random.__RandomState_ctor, (), self.get_state())             # <<<<<<<<<<<<<<
 * 
 *     # Basic distributions:
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_2 = __site_get_random_721_18->Target(__site_get_random_721_18, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_get___RandomState_ctor_721_25->Target(__site_get___RandomState_ctor_721_25, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = __site_get_get_state_721_54->Target(__site_get_get_state_721_54, __pyx_v_self, __pyx_context);
    __pyx_t_3 = __site_call0_721_64->Target(__site_call0_721_64, __pyx_context, __pyx_t_2);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = PythonOps::MakeTuple(gcnew array<System::Object^>{__pyx_t_1, ((System::Object^)PythonOps::EmptyTuple), __pyx_t_3});
    __pyx_t_1 = nullptr;
    __pyx_t_3 = nullptr;
    __pyx_r = __pyx_t_2;
    __pyx_t_2 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":724
 * 
 *     # Basic distributions:
 *     def random_sample(self, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         random_sample(size=None)
 */

  virtual System::Object^ random_sample([InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_size = nullptr;
    System::Object^ __pyx_r = nullptr;
    System::Object^ __pyx_t_1 = nullptr;
    System::Object^ __pyx_v_self = this;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":765
 * 
 *         """
 *         return cont0_array(self.internal_state, rk_double, size)             # <<<<<<<<<<<<<<
 * 
 *     def tomaxint(self, size=None):
 */
    __pyx_t_1 = cont0_array(((RandomState^)__pyx_v_self)->internal_state, rk_double, __pyx_v_size); 
    __pyx_r = __pyx_t_1;
    __pyx_t_1 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":767
 *         return cont0_array(self.internal_state, rk_double, size)
 * 
 *     def tomaxint(self, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         tomaxint(size=None)
 */

  virtual System::Object^ tomaxint([InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_size = nullptr;
    System::Object^ __pyx_r = nullptr;
    System::Object^ __pyx_t_1 = nullptr;
    System::Object^ __pyx_v_self = this;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":793
 * 
 *         """
 *         return disc0_array(self.internal_state, rk_long, size)             # <<<<<<<<<<<<<<
 * 
 *     def randint(self, low, high=None, size=None):
 */
    __pyx_t_1 = disc0_array(((RandomState^)__pyx_v_self)->internal_state, rk_long, __pyx_v_size); 
    __pyx_r = __pyx_t_1;
    __pyx_t_1 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":795
 *         return disc0_array(self.internal_state, rk_long, size)
 * 
 *     def randint(self, low, high=None, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         randint(low, high=None, size=None)
 */

  virtual System::Object^ randint(System::Object^ low, [InteropServices::Optional]System::Object^ high, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_low = nullptr;
    System::Object^ __pyx_v_high = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    long __pyx_v_lo;
    long __pyx_v_hi;
    long __pyx_v_diff;
    long *__pyx_v_arr_data;
    npy_intp __pyx_v_length;
    npy_intp __pyx_v_i;
    System::Object^ __pyx_v_arr;
    System::Object^ __pyx_r = nullptr;
    int __pyx_t_1;
    long __pyx_t_2;
    long __pyx_t_3;
    long __pyx_t_4;
    System::Object^ __pyx_t_5 = nullptr;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    npy_intp __pyx_t_8;
    npy_intp __pyx_t_9;
    System::Object^ __pyx_v_self = this;
    __pyx_v_low = low;
    if (dynamic_cast<System::Reflection::Missing^>(high) == nullptr) {
      __pyx_v_high = high;
    } else {
      __pyx_v_high = ((System::Object^)nullptr);
    }
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }
    __pyx_v_arr = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":849
 *         cdef npy_intp length, i
 * 
 *         if high is None:             # <<<<<<<<<<<<<<
 *             lo = 0
 *             hi = low
 */
    __pyx_t_1 = (__pyx_v_high == nullptr);
    if (__pyx_t_1) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":850
 * 
 *         if high is None:
 *             lo = 0             # <<<<<<<<<<<<<<
 *             hi = low
 *         else:
 */
      __pyx_v_lo = 0;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":851
 *         if high is None:
 *             lo = 0
 *             hi = low             # <<<<<<<<<<<<<<
 *         else:
 *             lo = low
 */
      __pyx_t_2 = __site_cvt_long_851_20->Target(__site_cvt_long_851_20, __pyx_v_low);
      __pyx_v_hi = __pyx_t_2;
      goto __pyx_L5;
    }
    /*else*/ {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":853
 *             hi = low
 *         else:
 *             lo = low             # <<<<<<<<<<<<<<
 *             hi = high
 * 
 */
      __pyx_t_3 = __site_cvt_long_853_20->Target(__site_cvt_long_853_20, __pyx_v_low);
      __pyx_v_lo = __pyx_t_3;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":854
 *         else:
 *             lo = low
 *             hi = high             # <<<<<<<<<<<<<<
 * 
 *         diff = hi - lo - 1
 */
      __pyx_t_4 = __site_cvt_long_854_21->Target(__site_cvt_long_854_21, __pyx_v_high);
      __pyx_v_hi = __pyx_t_4;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":856
 *             hi = high
 * 
 *         diff = hi - lo - 1             # <<<<<<<<<<<<<<
 *         if diff < 0:
 *             raise ValueError("low >= high")
 */
    __pyx_v_diff = ((__pyx_v_hi - __pyx_v_lo) - 1);

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":857
 * 
 *         diff = hi - lo - 1
 *         if diff < 0:             # <<<<<<<<<<<<<<
 *             raise ValueError("low >= high")
 * 
 */
    __pyx_t_1 = (__pyx_v_diff < 0);
    if (__pyx_t_1) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":858
 *         diff = hi - lo - 1
 *         if diff < 0:
 *             raise ValueError("low >= high")             # <<<<<<<<<<<<<<
 * 
 *         if size is None:
 */
      __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_6 = __site_call1_858_28->Target(__site_call1_858_28, __pyx_context, __pyx_t_5, ((System::Object^)"low >= high"));
      __pyx_t_5 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
      __pyx_t_6 = nullptr;
      goto __pyx_L6;
    }
    __pyx_L6:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":860
 *             raise ValueError("low >= high")
 * 
 *         if size is None:             # <<<<<<<<<<<<<<
 *             return <long>rk_interval(diff, self.internal_state) + lo
 *         else:
 */
    __pyx_t_1 = (__pyx_v_size == nullptr);
    if (__pyx_t_1) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":861
 * 
 *         if size is None:
 *             return <long>rk_interval(diff, self.internal_state) + lo             # <<<<<<<<<<<<<<
 *         else:
 *             arr = np.empty(size, int)
 */
      __pyx_t_6 = (((long)rk_interval(__pyx_v_diff, ((RandomState^)__pyx_v_self)->internal_state)) + __pyx_v_lo);
      __pyx_r = __pyx_t_6;
      __pyx_t_6 = nullptr;
      goto __pyx_L0;
      goto __pyx_L7;
    }
    /*else*/ {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":863
 *             return <long>rk_interval(diff, self.internal_state) + lo
 *         else:
 *             arr = np.empty(size, int)             # <<<<<<<<<<<<<<
 *             length = arr.size
 *             arr_data = <long *>dataptr(arr)
 */
      __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
      __pyx_t_5 = __site_get_empty_863_20->Target(__site_get_empty_863_20, __pyx_t_6, __pyx_context);
      __pyx_t_6 = nullptr;
      __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "int");
      __pyx_t_7 = __site_call2_863_26->Target(__site_call2_863_26, __pyx_context, __pyx_t_5, __pyx_v_size, ((System::Object^)__pyx_t_6));
      __pyx_t_5 = nullptr;
      __pyx_t_6 = nullptr;
      __pyx_v_arr = __pyx_t_7;
      __pyx_t_7 = nullptr;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":864
 *         else:
 *             arr = np.empty(size, int)
 *             length = arr.size             # <<<<<<<<<<<<<<
 *             arr_data = <long *>dataptr(arr)
 *             for i from 0 <= i < length:
 */
      __pyx_t_7 = __site_get_size_864_24->Target(__site_get_size_864_24, __pyx_v_arr, __pyx_context);
      __pyx_t_8 = __site_cvt_npy_intp_864_24->Target(__site_cvt_npy_intp_864_24, __pyx_t_7);
      __pyx_t_7 = nullptr;
      __pyx_v_length = __pyx_t_8;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":865
 *             arr = np.empty(size, int)
 *             length = arr.size
 *             arr_data = <long *>dataptr(arr)             # <<<<<<<<<<<<<<
 *             for i from 0 <= i < length:
 *                 arr_data[i] = lo + <long>rk_interval(diff, self.internal_state)
 */
      __pyx_v_arr_data = ((long *)dataptr(__pyx_v_arr));

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":866
 *             length = arr.size
 *             arr_data = <long *>dataptr(arr)
 *             for i from 0 <= i < length:             # <<<<<<<<<<<<<<
 *                 arr_data[i] = lo + <long>rk_interval(diff, self.internal_state)
 *             return arr
 */
      __pyx_t_9 = __pyx_v_length;
      for (__pyx_v_i = 0; __pyx_v_i < __pyx_t_9; __pyx_v_i++) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":867
 *             arr_data = <long *>dataptr(arr)
 *             for i from 0 <= i < length:
 *                 arr_data[i] = lo + <long>rk_interval(diff, self.internal_state)             # <<<<<<<<<<<<<<
 *             return arr
 * 
 */
        (__pyx_v_arr_data[__pyx_v_i]) = (__pyx_v_lo + ((long)rk_interval(__pyx_v_diff, ((RandomState^)__pyx_v_self)->internal_state)));
      }

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":868
 *             for i from 0 <= i < length:
 *                 arr_data[i] = lo + <long>rk_interval(diff, self.internal_state)
 *             return arr             # <<<<<<<<<<<<<<
 * 
 *     def bytes(self, unsigned int length):
 */
      __pyx_r = __pyx_v_arr;
      goto __pyx_L0;
    }
    __pyx_L7:;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":870
 *             return arr
 * 
 *     def bytes(self, unsigned int length):             # <<<<<<<<<<<<<<
 *         """
 *         bytes(length)
 */

  virtual System::Object^ bytes(System::Object^ length) {
    unsigned int __pyx_v_length;
    System::Object^ __pyx_v_res = nullptr;
    char *__pyx_v_bytes_ptr;
    System::Object^ __pyx_r = nullptr;
    System::Object^ __pyx_t_1 = nullptr;
    System::Object^ __pyx_t_2 = nullptr;
    char *__pyx_t_3;
    System::IntPtr __pyx_t_4;
    System::Object^ __pyx_v_self = this;
    __pyx_v_length = __site_cvt_870_4->Target(__site_cvt_870_4, length);

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":892
 * 
 *         """
 *         cdef bytes_type res = b'x' * length             # <<<<<<<<<<<<<<
 *         cdef char *bytes_ptr = res
 * 
 */
    __pyx_t_1 = __pyx_v_length;
    __pyx_t_2 = __site_op_mul_892_35->Target(__site_op_mul_892_35, ((System::Object^)PythonOps::MakeBytes(gcnew array<unsigned char>{120})), __pyx_t_1);
    __pyx_t_1 = nullptr;
    __pyx_v_res = __pyx_t_2;
    __pyx_t_2 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":893
 *         """
 *         cdef bytes_type res = b'x' * length
 *         cdef char *bytes_ptr = res             # <<<<<<<<<<<<<<
 * 
 *         rk_fill(bytes_ptr, length, self.internal_state)
 */
    __pyx_t_4 = InteropServices::Marshal::StringToHGlobalAnsi(dynamic_cast<System::String^>(((System::Object^)__pyx_v_res)));
    __pyx_t_3 = static_cast<char *>(__pyx_t_4.ToPointer());
    __pyx_v_bytes_ptr = __pyx_t_3;
    InteropServices::Marshal::FreeHGlobal(__pyx_t_4);

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":895
 *         cdef char *bytes_ptr = res
 * 
 *         rk_fill(bytes_ptr, length, self.internal_state)             # <<<<<<<<<<<<<<
 *         return res
 * 
 */
    rk_fill(__pyx_v_bytes_ptr, __pyx_v_length, ((RandomState^)__pyx_v_self)->internal_state);

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":896
 * 
 *         rk_fill(bytes_ptr, length, self.internal_state)
 *         return res             # <<<<<<<<<<<<<<
 * 
 *     def uniform(self, low=0.0, high=1.0, size=None):
 */
    __pyx_r = ((System::Object^)__pyx_v_res);
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":898
 *         return res
 * 
 *     def uniform(self, low=0.0, high=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         uniform(low=0.0, high=1.0, size=1)
 */

  virtual System::Object^ uniform([InteropServices::Optional]System::Object^ low, [InteropServices::Optional]System::Object^ high, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_low = nullptr;
    System::Object^ __pyx_v_high = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_flow;
    double __pyx_v_fhigh;
    int __pyx_v_sc;
    System::Object^ __pyx_v_diff;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    double __pyx_t_4;
    System::Object^ __pyx_t_5 = nullptr;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_v_self = this;
    if (dynamic_cast<System::Reflection::Missing^>(low) == nullptr) {
      __pyx_v_low = low;
    } else {
      __pyx_v_low = __pyx_k_1;
    }
    if (dynamic_cast<System::Reflection::Missing^>(high) == nullptr) {
      __pyx_v_high = high;
    } else {
      __pyx_v_high = __pyx_k_2;
    }
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }
    __pyx_v_diff = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":971
 *         cdef double flow, fhigh
 *         cdef object temp
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":973
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             flow = <double>low
 *             fhigh = <double>high
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":974
 * 
 *         try:
 *             flow = <double>low             # <<<<<<<<<<<<<<
 *             fhigh = <double>high
 *             sc = 1
 */
      __pyx_t_3 = __site_cvt_double_974_30->Target(__site_cvt_double_974_30, __pyx_v_low);
      __pyx_v_flow = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":975
 *         try:
 *             flow = <double>low
 *             fhigh = <double>high             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_4 = __site_cvt_double_975_32->Target(__site_cvt_double_975_32, __pyx_v_high);
      __pyx_v_fhigh = ((double)__pyx_t_4);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":976
 *             flow = <double>low
 *             fhigh = <double>high
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":977
 *             fhigh = <double>high
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.uniform");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":980
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             return cont2_array_sc(self.internal_state, rk_uniform, size,
 *                                   flow, fhigh - flow)
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":982
 *         if sc:
 *             return cont2_array_sc(self.internal_state, rk_uniform, size,
 *                                   flow, fhigh - flow)             # <<<<<<<<<<<<<<
 * 
 *         diff = np.array(high) - np.array(low)
 */
      __pyx_t_5 = cont2_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_uniform, __pyx_v_size, __pyx_v_flow, (__pyx_v_fhigh - __pyx_v_flow)); 
      __pyx_r = __pyx_t_5;
      __pyx_t_5 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":984
 *                                   flow, fhigh - flow)
 * 
 *         diff = np.array(high) - np.array(low)             # <<<<<<<<<<<<<<
 *         return cont2_array(self.internal_state, rk_uniform, size,
 *                            low, diff)
 */
    __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_array_984_17->Target(__site_get_array_984_17, __pyx_t_5, __pyx_context);
    __pyx_t_5 = nullptr;
    __pyx_t_5 = __site_call1_984_23->Target(__site_call1_984_23, __pyx_context, __pyx_t_6, __pyx_v_high);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_array_984_34->Target(__site_get_array_984_34, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_984_40->Target(__site_call1_984_40, __pyx_context, __pyx_t_7, __pyx_v_low);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_op_sub_984_30->Target(__site_op_sub_984_30, __pyx_t_5, __pyx_t_6);
    __pyx_t_5 = nullptr;
    __pyx_t_6 = nullptr;
    __pyx_v_diff = __pyx_t_7;
    __pyx_t_7 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":986
 *         diff = np.array(high) - np.array(low)
 *         return cont2_array(self.internal_state, rk_uniform, size,
 *                            low, diff)             # <<<<<<<<<<<<<<
 * 
 *     def rand(self, *args):
 */
    __pyx_t_7 = cont2_array(((RandomState^)__pyx_v_self)->internal_state, rk_uniform, __pyx_v_size, __pyx_v_low, __pyx_v_diff); 
    __pyx_r = __pyx_t_7;
    __pyx_t_7 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":988
 *                            low, diff)
 * 
 *     def rand(self, *args):             # <<<<<<<<<<<<<<
 *         """
 *         rand(d0, d1, ..., dn)
 */

  virtual System::Object^ rand(... array<System::Object^>^ args) {
    System::Object^ __pyx_r = nullptr;
    System::Object^ __pyx_t_1 = nullptr;
    System::Object^ __pyx_t_2 = nullptr;
    int __pyx_t_3;
    System::Object^ __pyx_v_self = this;
    System::Object^ __pyx_v_args = PythonOps::MakeTuple(args);

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1026
 * 
 *         """
 *         if len(args) == 0:             # <<<<<<<<<<<<<<
 *             return self.random_sample()
 *         else:
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "len");
    __pyx_t_2 = __site_call1_1026_14->Target(__site_call1_1026_14, __pyx_context, __pyx_t_1, ((System::Object^)__pyx_v_args));
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_op_eq_1026_21->Target(__site_op_eq_1026_21, __pyx_t_2, __pyx_int_0);
    __pyx_t_2 = nullptr;
    __pyx_t_3 = __site_istrue_1026_21->Target(__site_istrue_1026_21, __pyx_t_1);
    __pyx_t_1 = nullptr;
    if (__pyx_t_3) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1027
 *         """
 *         if len(args) == 0:
 *             return self.random_sample()             # <<<<<<<<<<<<<<
 *         else:
 *             return self.random_sample(size=args)
 */
      __pyx_t_1 = __site_get_random_sample_1027_23->Target(__site_get_random_sample_1027_23, __pyx_v_self, __pyx_context);
      __pyx_t_2 = __site_call0_1027_37->Target(__site_call0_1027_37, __pyx_context, __pyx_t_1);
      __pyx_t_1 = nullptr;
      __pyx_r = __pyx_t_2;
      __pyx_t_2 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    /*else*/ {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1029
 *             return self.random_sample()
 *         else:
 *             return self.random_sample(size=args)             # <<<<<<<<<<<<<<
 * 
 *     def randn(self, *args):
 */
      __pyx_t_2 = __site_get_random_sample_1029_23->Target(__site_get_random_sample_1029_23, __pyx_v_self, __pyx_context);
      __pyx_t_1 = __site_call0_size_1029_37->Target(__site_call0_size_1029_37, __pyx_context, __pyx_t_2, ((System::Object^)__pyx_v_args));
      __pyx_t_2 = nullptr;
      __pyx_r = __pyx_t_1;
      __pyx_t_1 = nullptr;
      goto __pyx_L0;
    }
    __pyx_L5:;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1031
 *             return self.random_sample(size=args)
 * 
 *     def randn(self, *args):             # <<<<<<<<<<<<<<
 *         """
 *         randn([d1, ..., dn])
 */

  virtual System::Object^ randn(... array<System::Object^>^ args) {
    System::Object^ __pyx_r = nullptr;
    System::Object^ __pyx_t_1 = nullptr;
    System::Object^ __pyx_t_2 = nullptr;
    int __pyx_t_3;
    System::Object^ __pyx_v_self = this;
    System::Object^ __pyx_v_args = PythonOps::MakeTuple(args);

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1082
 * 
 *         """
 *         if len(args) == 0:             # <<<<<<<<<<<<<<
 *             return self.standard_normal()
 *         else:
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "len");
    __pyx_t_2 = __site_call1_1082_14->Target(__site_call1_1082_14, __pyx_context, __pyx_t_1, ((System::Object^)__pyx_v_args));
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_op_eq_1082_21->Target(__site_op_eq_1082_21, __pyx_t_2, __pyx_int_0);
    __pyx_t_2 = nullptr;
    __pyx_t_3 = __site_istrue_1082_21->Target(__site_istrue_1082_21, __pyx_t_1);
    __pyx_t_1 = nullptr;
    if (__pyx_t_3) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1083
 *         """
 *         if len(args) == 0:
 *             return self.standard_normal()             # <<<<<<<<<<<<<<
 *         else:
 *             return self.standard_normal(args)
 */
      __pyx_t_1 = __site_get_standard_normal_1083_23->Target(__site_get_standard_normal_1083_23, __pyx_v_self, __pyx_context);
      __pyx_t_2 = __site_call0_1083_39->Target(__site_call0_1083_39, __pyx_context, __pyx_t_1);
      __pyx_t_1 = nullptr;
      __pyx_r = __pyx_t_2;
      __pyx_t_2 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    /*else*/ {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1085
 *             return self.standard_normal()
 *         else:
 *             return self.standard_normal(args)             # <<<<<<<<<<<<<<
 * 
 *     def random_integers(self, low, high=None, size=None):
 */
      __pyx_t_2 = __site_get_standard_normal_1085_23->Target(__site_get_standard_normal_1085_23, __pyx_v_self, __pyx_context);
      __pyx_t_1 = __site_call1_1085_39->Target(__site_call1_1085_39, __pyx_context, __pyx_t_2, ((System::Object^)__pyx_v_args));
      __pyx_t_2 = nullptr;
      __pyx_r = __pyx_t_1;
      __pyx_t_1 = nullptr;
      goto __pyx_L0;
    }
    __pyx_L5:;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1087
 *             return self.standard_normal(args)
 * 
 *     def random_integers(self, low, high=None, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         random_integers(low, high=None, size=None)
 */

  virtual System::Object^ random_integers(System::Object^ low, [InteropServices::Optional]System::Object^ high, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_low = nullptr;
    System::Object^ __pyx_v_high = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    System::Object^ __pyx_r = nullptr;
    int __pyx_t_1;
    System::Object^ __pyx_t_2 = nullptr;
    System::Object^ __pyx_t_3 = nullptr;
    System::Object^ __pyx_t_4 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_low = low;
    if (dynamic_cast<System::Reflection::Missing^>(high) == nullptr) {
      __pyx_v_high = high;
    } else {
      __pyx_v_high = ((System::Object^)nullptr);
    }
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1160
 * 
 *         """
 *         if high is None:             # <<<<<<<<<<<<<<
 *             high = low
 *             low = 1
 */
    __pyx_t_1 = (__pyx_v_high == nullptr);
    if (__pyx_t_1) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1161
 *         """
 *         if high is None:
 *             high = low             # <<<<<<<<<<<<<<
 *             low = 1
 *         return self.randint(low, high + 1, size)
 */
      __pyx_v_high = __pyx_v_low;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1162
 *         if high is None:
 *             high = low
 *             low = 1             # <<<<<<<<<<<<<<
 *         return self.randint(low, high + 1, size)
 * 
 */
      __pyx_v_low = __pyx_int_1;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1163
 *             high = low
 *             low = 1
 *         return self.randint(low, high + 1, size)             # <<<<<<<<<<<<<<
 * 
 *     # Complicated, continuous distributions:
 */
    __pyx_t_2 = __site_get_randint_1163_19->Target(__site_get_randint_1163_19, __pyx_v_self, __pyx_context);
    __pyx_t_3 = __site_op_add_1163_38->Target(__site_op_add_1163_38, __pyx_v_high, __pyx_int_1);
    __pyx_t_4 = __site_call3_1163_27->Target(__site_call3_1163_27, __pyx_context, __pyx_t_2, __pyx_v_low, __pyx_t_3, __pyx_v_size);
    __pyx_t_2 = nullptr;
    __pyx_t_3 = nullptr;
    __pyx_r = __pyx_t_4;
    __pyx_t_4 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1166
 * 
 *     # Complicated, continuous distributions:
 *     def standard_normal(self, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         standard_normal(size=None)
 */

  virtual System::Object^ standard_normal([InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_size = nullptr;
    System::Object^ __pyx_r = nullptr;
    System::Object^ __pyx_t_1 = nullptr;
    System::Object^ __pyx_v_self = this;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1196
 * 
 *         """
 *         return cont0_array(self.internal_state, rk_gauss, size)             # <<<<<<<<<<<<<<
 * 
 *     def normal(self, loc=0.0, scale=1.0, size=None):
 */
    __pyx_t_1 = cont0_array(((RandomState^)__pyx_v_self)->internal_state, rk_gauss, __pyx_v_size); 
    __pyx_r = __pyx_t_1;
    __pyx_t_1 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1198
 *         return cont0_array(self.internal_state, rk_gauss, size)
 * 
 *     def normal(self, loc=0.0, scale=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         normal(loc=0.0, scale=1.0, size=None)
 */

  virtual System::Object^ normal([InteropServices::Optional]System::Object^ loc, [InteropServices::Optional]System::Object^ scale, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_loc = nullptr;
    System::Object^ __pyx_v_scale = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_floc;
    double __pyx_v_fscale;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    double __pyx_t_4;
    int __pyx_t_5;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_v_self = this;
    if (dynamic_cast<System::Reflection::Missing^>(loc) == nullptr) {
      __pyx_v_loc = loc;
    } else {
      __pyx_v_loc = __pyx_k_3;
    }
    if (dynamic_cast<System::Reflection::Missing^>(scale) == nullptr) {
      __pyx_v_scale = scale;
    } else {
      __pyx_v_scale = __pyx_k_4;
    }
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1281
 *         """
 *         cdef double floc, fscale
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1283
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             floc = <double>loc
 *             fscale = <double>scale
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1284
 * 
 *         try:
 *             floc = <double>loc             # <<<<<<<<<<<<<<
 *             fscale = <double>scale
 *             sc = 1
 */
      __pyx_t_3 = __site_cvt_double_1284_30->Target(__site_cvt_double_1284_30, __pyx_v_loc);
      __pyx_v_floc = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1285
 *         try:
 *             floc = <double>loc
 *             fscale = <double>scale             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_4 = __site_cvt_double_1285_34->Target(__site_cvt_double_1285_34, __pyx_v_scale);
      __pyx_v_fscale = ((double)__pyx_t_4);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1286
 *             floc = <double>loc
 *             fscale = <double>scale
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1287
 *             fscale = <double>scale
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.normal");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1290
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fscale <= 0:
 *                 raise ValueError("scale <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1291
 * 
 *         if sc:
 *             if fscale <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("scale <= 0")
 *             return cont2_array_sc(self.internal_state, rk_normal, size, floc,
 */
      __pyx_t_5 = (__pyx_v_fscale <= 0.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1292
 *         if sc:
 *             if fscale <= 0:
 *                 raise ValueError("scale <= 0")             # <<<<<<<<<<<<<<
 *             return cont2_array_sc(self.internal_state, rk_normal, size, floc,
 *                                   fscale)
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_7 = __site_call1_1292_32->Target(__site_call1_1292_32, __pyx_context, __pyx_t_6, ((System::Object^)"scale <= 0"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
        __pyx_t_7 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1294
 *                 raise ValueError("scale <= 0")
 *             return cont2_array_sc(self.internal_state, rk_normal, size, floc,
 *                                   fscale)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(scale, 0)):
 */
      __pyx_t_7 = cont2_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_normal, __pyx_v_size, __pyx_v_floc, __pyx_v_fscale); 
      __pyx_r = __pyx_t_7;
      __pyx_t_7 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1296
 *                                   fscale)
 * 
 *         if np.any(np.less_equal(scale, 0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("scale <= 0")
 *         return cont2_array(self.internal_state, rk_normal, size, loc, scale)
 */
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_any_1296_13->Target(__site_get_any_1296_13, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_less_equal_1296_20->Target(__site_get_less_equal_1296_20, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_call2_1296_31->Target(__site_call2_1296_31, __pyx_context, __pyx_t_8, __pyx_v_scale, __pyx_int_0);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = __site_call1_1296_17->Target(__site_call1_1296_17, __pyx_context, __pyx_t_6, __pyx_t_7);
    __pyx_t_6 = nullptr;
    __pyx_t_7 = nullptr;
    __pyx_t_5 = __site_istrue_1296_17->Target(__site_istrue_1296_17, __pyx_t_8);
    __pyx_t_8 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1297
 * 
 *         if np.any(np.less_equal(scale, 0)):
 *             raise ValueError("scale <= 0")             # <<<<<<<<<<<<<<
 *         return cont2_array(self.internal_state, rk_normal, size, loc, scale)
 * 
 */
      __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_7 = __site_call1_1297_28->Target(__site_call1_1297_28, __pyx_context, __pyx_t_8, ((System::Object^)"scale <= 0"));
      __pyx_t_8 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
      __pyx_t_7 = nullptr;
      goto __pyx_L7;
    }
    __pyx_L7:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1298
 *         if np.any(np.less_equal(scale, 0)):
 *             raise ValueError("scale <= 0")
 *         return cont2_array(self.internal_state, rk_normal, size, loc, scale)             # <<<<<<<<<<<<<<
 * 
 *     def beta(self, a, b, size=None):
 */
    __pyx_t_7 = cont2_array(((RandomState^)__pyx_v_self)->internal_state, rk_normal, __pyx_v_size, __pyx_v_loc, __pyx_v_scale); 
    __pyx_r = __pyx_t_7;
    __pyx_t_7 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1300
 *         return cont2_array(self.internal_state, rk_normal, size, loc, scale)
 * 
 *     def beta(self, a, b, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         beta(a, b, size=None)
 */

  virtual System::Object^ beta(System::Object^ a, System::Object^ b, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_a = nullptr;
    System::Object^ __pyx_v_b = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fa;
    double __pyx_v_fb;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    double __pyx_t_4;
    int __pyx_t_5;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_a = a;
    __pyx_v_b = b;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1338
 *         """
 *         cdef double fa, fb
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1340
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fa = <double>a
 *             fb = <double>b
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1341
 * 
 *         try:
 *             fa = <double>a             # <<<<<<<<<<<<<<
 *             fb = <double>b
 *             sc = 1
 */
      __pyx_t_3 = __site_cvt_double_1341_26->Target(__site_cvt_double_1341_26, __pyx_v_a);
      __pyx_v_fa = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1342
 *         try:
 *             fa = <double>a
 *             fb = <double>b             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_4 = __site_cvt_double_1342_26->Target(__site_cvt_double_1342_26, __pyx_v_b);
      __pyx_v_fb = ((double)__pyx_t_4);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1343
 *             fa = <double>a
 *             fb = <double>b
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1344
 *             fb = <double>b
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.beta");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1347
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fa <= 0:
 *                 raise ValueError("a <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1348
 * 
 *         if sc:
 *             if fa <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("a <= 0")
 *             if fb <= 0:
 */
      __pyx_t_5 = (__pyx_v_fa <= 0.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1349
 *         if sc:
 *             if fa <= 0:
 *                 raise ValueError("a <= 0")             # <<<<<<<<<<<<<<
 *             if fb <= 0:
 *                 raise ValueError("b <= 0")
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_7 = __site_call1_1349_32->Target(__site_call1_1349_32, __pyx_context, __pyx_t_6, ((System::Object^)"a <= 0"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
        __pyx_t_7 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1350
 *             if fa <= 0:
 *                 raise ValueError("a <= 0")
 *             if fb <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("b <= 0")
 *             return cont2_array_sc(self.internal_state, rk_beta, size, fa, fb)
 */
      __pyx_t_5 = (__pyx_v_fb <= 0.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1351
 *                 raise ValueError("a <= 0")
 *             if fb <= 0:
 *                 raise ValueError("b <= 0")             # <<<<<<<<<<<<<<
 *             return cont2_array_sc(self.internal_state, rk_beta, size, fa, fb)
 * 
 */
        __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_1351_32->Target(__site_call1_1351_32, __pyx_context, __pyx_t_7, ((System::Object^)"b <= 0"));
        __pyx_t_7 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L7;
      }
      __pyx_L7:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1352
 *             if fb <= 0:
 *                 raise ValueError("b <= 0")
 *             return cont2_array_sc(self.internal_state, rk_beta, size, fa, fb)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(a, 0)):
 */
      __pyx_t_6 = cont2_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_beta, __pyx_v_size, __pyx_v_fa, __pyx_v_fb); 
      __pyx_r = __pyx_t_6;
      __pyx_t_6 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1354
 *             return cont2_array_sc(self.internal_state, rk_beta, size, fa, fb)
 * 
 *         if np.any(np.less_equal(a, 0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("a <= 0")
 *         if np.any(np.less_equal(b, 0)):
 */
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_any_1354_13->Target(__site_get_any_1354_13, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_less_equal_1354_20->Target(__site_get_less_equal_1354_20, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call2_1354_31->Target(__site_call2_1354_31, __pyx_context, __pyx_t_8, __pyx_v_a, __pyx_int_0);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = __site_call1_1354_17->Target(__site_call1_1354_17, __pyx_context, __pyx_t_7, __pyx_t_6);
    __pyx_t_7 = nullptr;
    __pyx_t_6 = nullptr;
    __pyx_t_5 = __site_istrue_1354_17->Target(__site_istrue_1354_17, __pyx_t_8);
    __pyx_t_8 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1355
 * 
 *         if np.any(np.less_equal(a, 0)):
 *             raise ValueError("a <= 0")             # <<<<<<<<<<<<<<
 *         if np.any(np.less_equal(b, 0)):
 *             raise ValueError("b <= 0")
 */
      __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_6 = __site_call1_1355_28->Target(__site_call1_1355_28, __pyx_context, __pyx_t_8, ((System::Object^)"a <= 0"));
      __pyx_t_8 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
      __pyx_t_6 = nullptr;
      goto __pyx_L8;
    }
    __pyx_L8:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1356
 *         if np.any(np.less_equal(a, 0)):
 *             raise ValueError("a <= 0")
 *         if np.any(np.less_equal(b, 0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("b <= 0")
 *         return cont2_array(self.internal_state, rk_beta, size, a, b)
 */
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_any_1356_13->Target(__site_get_any_1356_13, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_less_equal_1356_20->Target(__site_get_less_equal_1356_20, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call2_1356_31->Target(__site_call2_1356_31, __pyx_context, __pyx_t_7, __pyx_v_b, __pyx_int_0);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_call1_1356_17->Target(__site_call1_1356_17, __pyx_context, __pyx_t_8, __pyx_t_6);
    __pyx_t_8 = nullptr;
    __pyx_t_6 = nullptr;
    __pyx_t_5 = __site_istrue_1356_17->Target(__site_istrue_1356_17, __pyx_t_7);
    __pyx_t_7 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1357
 *             raise ValueError("a <= 0")
 *         if np.any(np.less_equal(b, 0)):
 *             raise ValueError("b <= 0")             # <<<<<<<<<<<<<<
 *         return cont2_array(self.internal_state, rk_beta, size, a, b)
 * 
 */
      __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_6 = __site_call1_1357_28->Target(__site_call1_1357_28, __pyx_context, __pyx_t_7, ((System::Object^)"b <= 0"));
      __pyx_t_7 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
      __pyx_t_6 = nullptr;
      goto __pyx_L9;
    }
    __pyx_L9:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1358
 *         if np.any(np.less_equal(b, 0)):
 *             raise ValueError("b <= 0")
 *         return cont2_array(self.internal_state, rk_beta, size, a, b)             # <<<<<<<<<<<<<<
 * 
 *     def exponential(self, scale=1.0, size=None):
 */
    __pyx_t_6 = cont2_array(((RandomState^)__pyx_v_self)->internal_state, rk_beta, __pyx_v_size, __pyx_v_a, __pyx_v_b); 
    __pyx_r = __pyx_t_6;
    __pyx_t_6 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1360
 *         return cont2_array(self.internal_state, rk_beta, size, a, b)
 * 
 *     def exponential(self, scale=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         exponential(scale=1.0, size=None)
 */

  virtual System::Object^ exponential([InteropServices::Optional]System::Object^ scale, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_scale = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fscale;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    int __pyx_t_4;
    System::Object^ __pyx_t_5 = nullptr;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_v_self = this;
    if (dynamic_cast<System::Reflection::Missing^>(scale) == nullptr) {
      __pyx_v_scale = scale;
    } else {
      __pyx_v_scale = __pyx_k_5;
    }
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1399
 *         """
 *         cdef double fscale
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1401
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fscale = <double>scale
 *             sc = 1
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1402
 * 
 *         try:
 *             fscale = <double>scale             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_3 = __site_cvt_double_1402_34->Target(__site_cvt_double_1402_34, __pyx_v_scale);
      __pyx_v_fscale = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1403
 *         try:
 *             fscale = <double>scale
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1404
 *             fscale = <double>scale
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.exponential");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1407
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fscale <= 0:
 *                 raise ValueError("scale <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1408
 * 
 *         if sc:
 *             if fscale <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("scale <= 0")
 *             return cont1_array_sc(self.internal_state, rk_exponential, size,
 */
      __pyx_t_4 = (__pyx_v_fscale <= 0.0);
      if (__pyx_t_4) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1409
 *         if sc:
 *             if fscale <= 0:
 *                 raise ValueError("scale <= 0")             # <<<<<<<<<<<<<<
 *             return cont1_array_sc(self.internal_state, rk_exponential, size,
 *                                   fscale)
 */
        __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_1409_32->Target(__site_call1_1409_32, __pyx_context, __pyx_t_5, ((System::Object^)"scale <= 0"));
        __pyx_t_5 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1411
 *                 raise ValueError("scale <= 0")
 *             return cont1_array_sc(self.internal_state, rk_exponential, size,
 *                                   fscale)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(scale, 0.0)):
 */
      __pyx_t_6 = cont1_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_exponential, __pyx_v_size, __pyx_v_fscale); 
      __pyx_r = __pyx_t_6;
      __pyx_t_6 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1413
 *                                   fscale)
 * 
 *         if np.any(np.less_equal(scale, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("scale <= 0")
 *         return cont1_array(self.internal_state, rk_exponential, size, scale)
 */
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_any_1413_13->Target(__site_get_any_1413_13, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_less_equal_1413_20->Target(__site_get_less_equal_1413_20, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = 0.0;
    __pyx_t_8 = __site_call2_1413_31->Target(__site_call2_1413_31, __pyx_context, __pyx_t_7, __pyx_v_scale, __pyx_t_6);
    __pyx_t_7 = nullptr;
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_1413_17->Target(__site_call1_1413_17, __pyx_context, __pyx_t_5, __pyx_t_8);
    __pyx_t_5 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_4 = __site_istrue_1413_17->Target(__site_istrue_1413_17, __pyx_t_6);
    __pyx_t_6 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1414
 * 
 *         if np.any(np.less_equal(scale, 0.0)):
 *             raise ValueError("scale <= 0")             # <<<<<<<<<<<<<<
 *         return cont1_array(self.internal_state, rk_exponential, size, scale)
 * 
 */
      __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_1414_28->Target(__site_call1_1414_28, __pyx_context, __pyx_t_6, ((System::Object^)"scale <= 0"));
      __pyx_t_6 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L7;
    }
    __pyx_L7:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1415
 *         if np.any(np.less_equal(scale, 0.0)):
 *             raise ValueError("scale <= 0")
 *         return cont1_array(self.internal_state, rk_exponential, size, scale)             # <<<<<<<<<<<<<<
 * 
 *     def standard_exponential(self, size=None):
 */
    __pyx_t_8 = cont1_array(((RandomState^)__pyx_v_self)->internal_state, rk_exponential, __pyx_v_size, __pyx_v_scale); 
    __pyx_r = __pyx_t_8;
    __pyx_t_8 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1417
 *         return cont1_array(self.internal_state, rk_exponential, size, scale)
 * 
 *     def standard_exponential(self, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         standard_exponential(size=None)
 */

  virtual System::Object^ standard_exponential([InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_size = nullptr;
    System::Object^ __pyx_r = nullptr;
    System::Object^ __pyx_t_1 = nullptr;
    System::Object^ __pyx_v_self = this;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1443
 * 
 *         """
 *         return cont0_array(self.internal_state, rk_standard_exponential, size)             # <<<<<<<<<<<<<<
 * 
 *     def standard_gamma(self, shape, size=None):
 */
    __pyx_t_1 = cont0_array(((RandomState^)__pyx_v_self)->internal_state, rk_standard_exponential, __pyx_v_size); 
    __pyx_r = __pyx_t_1;
    __pyx_t_1 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1445
 *         return cont0_array(self.internal_state, rk_standard_exponential, size)
 * 
 *     def standard_gamma(self, shape, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         standard_gamma(shape, size=None)
 */

  virtual System::Object^ standard_gamma(System::Object^ shape, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_shape = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fshape;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    int __pyx_t_4;
    System::Object^ __pyx_t_5 = nullptr;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_shape = shape;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1513
 *         """
 *         cdef double fshape
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1515
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fshape = <double>shape
 *             sc = 1
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1516
 * 
 *         try:
 *             fshape = <double>shape             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_3 = __site_cvt_double_1516_34->Target(__site_cvt_double_1516_34, __pyx_v_shape);
      __pyx_v_fshape = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1517
 *         try:
 *             fshape = <double>shape
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1518
 *             fshape = <double>shape
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.standard_gamma");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1521
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fshape <= 0:
 *                 raise ValueError("shape <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1522
 * 
 *         if sc:
 *             if fshape <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("shape <= 0")
 *             return cont1_array_sc(self.internal_state, rk_standard_gamma, size,
 */
      __pyx_t_4 = (__pyx_v_fshape <= 0.0);
      if (__pyx_t_4) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1523
 *         if sc:
 *             if fshape <= 0:
 *                 raise ValueError("shape <= 0")             # <<<<<<<<<<<<<<
 *             return cont1_array_sc(self.internal_state, rk_standard_gamma, size,
 *                                   fshape)
 */
        __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_1523_32->Target(__site_call1_1523_32, __pyx_context, __pyx_t_5, ((System::Object^)"shape <= 0"));
        __pyx_t_5 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1525
 *                 raise ValueError("shape <= 0")
 *             return cont1_array_sc(self.internal_state, rk_standard_gamma, size,
 *                                   fshape)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(shape, 0.0)):
 */
      __pyx_t_6 = cont1_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_standard_gamma, __pyx_v_size, __pyx_v_fshape); 
      __pyx_r = __pyx_t_6;
      __pyx_t_6 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1527
 *                                   fshape)
 * 
 *         if np.any(np.less_equal(shape, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("shape <= 0")
 *         return cont1_array(self.internal_state, rk_standard_gamma, size, shape)
 */
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_any_1527_13->Target(__site_get_any_1527_13, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_less_equal_1527_20->Target(__site_get_less_equal_1527_20, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = 0.0;
    __pyx_t_8 = __site_call2_1527_31->Target(__site_call2_1527_31, __pyx_context, __pyx_t_7, __pyx_v_shape, __pyx_t_6);
    __pyx_t_7 = nullptr;
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_1527_17->Target(__site_call1_1527_17, __pyx_context, __pyx_t_5, __pyx_t_8);
    __pyx_t_5 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_4 = __site_istrue_1527_17->Target(__site_istrue_1527_17, __pyx_t_6);
    __pyx_t_6 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1528
 * 
 *         if np.any(np.less_equal(shape, 0.0)):
 *             raise ValueError("shape <= 0")             # <<<<<<<<<<<<<<
 *         return cont1_array(self.internal_state, rk_standard_gamma, size, shape)
 * 
 */
      __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_1528_28->Target(__site_call1_1528_28, __pyx_context, __pyx_t_6, ((System::Object^)"shape <= 0"));
      __pyx_t_6 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L7;
    }
    __pyx_L7:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1529
 *         if np.any(np.less_equal(shape, 0.0)):
 *             raise ValueError("shape <= 0")
 *         return cont1_array(self.internal_state, rk_standard_gamma, size, shape)             # <<<<<<<<<<<<<<
 * 
 *     def gamma(self, shape, scale=1.0, size=None):
 */
    __pyx_t_8 = cont1_array(((RandomState^)__pyx_v_self)->internal_state, rk_standard_gamma, __pyx_v_size, __pyx_v_shape); 
    __pyx_r = __pyx_t_8;
    __pyx_t_8 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1531
 *         return cont1_array(self.internal_state, rk_standard_gamma, size, shape)
 * 
 *     def gamma(self, shape, scale=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         gamma(shape, scale=1.0, size=None)
 */

  virtual System::Object^ gamma(System::Object^ shape, [InteropServices::Optional]System::Object^ scale, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_shape = nullptr;
    System::Object^ __pyx_v_scale = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fshape;
    double __pyx_v_fscale;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    double __pyx_t_4;
    int __pyx_t_5;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_t_9 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_shape = shape;
    if (dynamic_cast<System::Reflection::Missing^>(scale) == nullptr) {
      __pyx_v_scale = scale;
    } else {
      __pyx_v_scale = __pyx_k_6;
    }
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1602
 *         """
 *         cdef double fshape, fscale
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1604
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fshape = <double>shape
 *             fscale = <double>scale
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1605
 * 
 *         try:
 *             fshape = <double>shape             # <<<<<<<<<<<<<<
 *             fscale = <double>scale
 *             sc = 1
 */
      __pyx_t_3 = __site_cvt_double_1605_34->Target(__site_cvt_double_1605_34, __pyx_v_shape);
      __pyx_v_fshape = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1606
 *         try:
 *             fshape = <double>shape
 *             fscale = <double>scale             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_4 = __site_cvt_double_1606_34->Target(__site_cvt_double_1606_34, __pyx_v_scale);
      __pyx_v_fscale = ((double)__pyx_t_4);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1607
 *             fshape = <double>shape
 *             fscale = <double>scale
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1608
 *             fscale = <double>scale
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.gamma");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1611
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fshape <= 0:
 *                 raise ValueError("shape <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1612
 * 
 *         if sc:
 *             if fshape <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("shape <= 0")
 *             if fscale <= 0:
 */
      __pyx_t_5 = (__pyx_v_fshape <= 0.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1613
 *         if sc:
 *             if fshape <= 0:
 *                 raise ValueError("shape <= 0")             # <<<<<<<<<<<<<<
 *             if fscale <= 0:
 *                 raise ValueError("scale <= 0")
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_7 = __site_call1_1613_32->Target(__site_call1_1613_32, __pyx_context, __pyx_t_6, ((System::Object^)"shape <= 0"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
        __pyx_t_7 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1614
 *             if fshape <= 0:
 *                 raise ValueError("shape <= 0")
 *             if fscale <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("scale <= 0")
 *             return cont2_array_sc(self.internal_state, rk_gamma, size,
 */
      __pyx_t_5 = (__pyx_v_fscale <= 0.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1615
 *                 raise ValueError("shape <= 0")
 *             if fscale <= 0:
 *                 raise ValueError("scale <= 0")             # <<<<<<<<<<<<<<
 *             return cont2_array_sc(self.internal_state, rk_gamma, size,
 *                                   fshape, fscale)
 */
        __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_1615_32->Target(__site_call1_1615_32, __pyx_context, __pyx_t_7, ((System::Object^)"scale <= 0"));
        __pyx_t_7 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L7;
      }
      __pyx_L7:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1617
 *                 raise ValueError("scale <= 0")
 *             return cont2_array_sc(self.internal_state, rk_gamma, size,
 *                                   fshape, fscale)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(shape, 0.0)):
 */
      __pyx_t_6 = cont2_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_gamma, __pyx_v_size, __pyx_v_fshape, __pyx_v_fscale); 
      __pyx_r = __pyx_t_6;
      __pyx_t_6 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1619
 *                                   fshape, fscale)
 * 
 *         if np.any(np.less_equal(shape, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("shape <= 0")
 *         if np.any(np.less_equal(scale, 0.0)):
 */
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_any_1619_13->Target(__site_get_any_1619_13, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_less_equal_1619_20->Target(__site_get_less_equal_1619_20, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = 0.0;
    __pyx_t_9 = __site_call2_1619_31->Target(__site_call2_1619_31, __pyx_context, __pyx_t_8, __pyx_v_shape, __pyx_t_6);
    __pyx_t_8 = nullptr;
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_1619_17->Target(__site_call1_1619_17, __pyx_context, __pyx_t_7, __pyx_t_9);
    __pyx_t_7 = nullptr;
    __pyx_t_9 = nullptr;
    __pyx_t_5 = __site_istrue_1619_17->Target(__site_istrue_1619_17, __pyx_t_6);
    __pyx_t_6 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1620
 * 
 *         if np.any(np.less_equal(shape, 0.0)):
 *             raise ValueError("shape <= 0")             # <<<<<<<<<<<<<<
 *         if np.any(np.less_equal(scale, 0.0)):
 *             raise ValueError("scale <= 0")
 */
      __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_9 = __site_call1_1620_28->Target(__site_call1_1620_28, __pyx_context, __pyx_t_6, ((System::Object^)"shape <= 0"));
      __pyx_t_6 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_9, nullptr, nullptr);
      __pyx_t_9 = nullptr;
      goto __pyx_L8;
    }
    __pyx_L8:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1621
 *         if np.any(np.less_equal(shape, 0.0)):
 *             raise ValueError("shape <= 0")
 *         if np.any(np.less_equal(scale, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("scale <= 0")
 *         return cont2_array(self.internal_state, rk_gamma, size, shape, scale)
 */
    __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_any_1621_13->Target(__site_get_any_1621_13, __pyx_t_9, __pyx_context);
    __pyx_t_9 = nullptr;
    __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_less_equal_1621_20->Target(__site_get_less_equal_1621_20, __pyx_t_9, __pyx_context);
    __pyx_t_9 = nullptr;
    __pyx_t_9 = 0.0;
    __pyx_t_8 = __site_call2_1621_31->Target(__site_call2_1621_31, __pyx_context, __pyx_t_7, __pyx_v_scale, __pyx_t_9);
    __pyx_t_7 = nullptr;
    __pyx_t_9 = nullptr;
    __pyx_t_9 = __site_call1_1621_17->Target(__site_call1_1621_17, __pyx_context, __pyx_t_6, __pyx_t_8);
    __pyx_t_6 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_5 = __site_istrue_1621_17->Target(__site_istrue_1621_17, __pyx_t_9);
    __pyx_t_9 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1622
 *             raise ValueError("shape <= 0")
 *         if np.any(np.less_equal(scale, 0.0)):
 *             raise ValueError("scale <= 0")             # <<<<<<<<<<<<<<
 *         return cont2_array(self.internal_state, rk_gamma, size, shape, scale)
 * 
 */
      __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_1622_28->Target(__site_call1_1622_28, __pyx_context, __pyx_t_9, ((System::Object^)"scale <= 0"));
      __pyx_t_9 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L9;
    }
    __pyx_L9:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1623
 *         if np.any(np.less_equal(scale, 0.0)):
 *             raise ValueError("scale <= 0")
 *         return cont2_array(self.internal_state, rk_gamma, size, shape, scale)             # <<<<<<<<<<<<<<
 * 
 *     def f(self, dfnum, dfden, size=None):
 */
    __pyx_t_8 = cont2_array(((RandomState^)__pyx_v_self)->internal_state, rk_gamma, __pyx_v_size, __pyx_v_shape, __pyx_v_scale); 
    __pyx_r = __pyx_t_8;
    __pyx_t_8 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1625
 *         return cont2_array(self.internal_state, rk_gamma, size, shape, scale)
 * 
 *     def f(self, dfnum, dfden, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         f(dfnum, dfden, size=None)
 */

  virtual System::Object^ f(System::Object^ dfnum, System::Object^ dfden, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_dfnum = nullptr;
    System::Object^ __pyx_v_dfden = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fdfnum;
    double __pyx_v_fdfden;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    double __pyx_t_4;
    int __pyx_t_5;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_t_9 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_dfnum = dfnum;
    __pyx_v_dfden = dfden;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1707
 *         """
 *         cdef double fdfnum, fdfden
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1709
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fdfnum = <double>dfnum
 *             fdfden = <double>dfden
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1710
 * 
 *         try:
 *             fdfnum = <double>dfnum             # <<<<<<<<<<<<<<
 *             fdfden = <double>dfden
 *             sc = 1
 */
      __pyx_t_3 = __site_cvt_double_1710_34->Target(__site_cvt_double_1710_34, __pyx_v_dfnum);
      __pyx_v_fdfnum = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1711
 *         try:
 *             fdfnum = <double>dfnum
 *             fdfden = <double>dfden             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_4 = __site_cvt_double_1711_34->Target(__site_cvt_double_1711_34, __pyx_v_dfden);
      __pyx_v_fdfden = ((double)__pyx_t_4);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1712
 *             fdfnum = <double>dfnum
 *             fdfden = <double>dfden
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1713
 *             fdfden = <double>dfden
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.f");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1716
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fdfnum <= 0:
 *                 raise ValueError("shape <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1717
 * 
 *         if sc:
 *             if fdfnum <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("shape <= 0")
 *             if fdfden <= 0:
 */
      __pyx_t_5 = (__pyx_v_fdfnum <= 0.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1718
 *         if sc:
 *             if fdfnum <= 0:
 *                 raise ValueError("shape <= 0")             # <<<<<<<<<<<<<<
 *             if fdfden <= 0:
 *                 raise ValueError("scale <= 0")
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_7 = __site_call1_1718_32->Target(__site_call1_1718_32, __pyx_context, __pyx_t_6, ((System::Object^)"shape <= 0"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
        __pyx_t_7 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1719
 *             if fdfnum <= 0:
 *                 raise ValueError("shape <= 0")
 *             if fdfden <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("scale <= 0")
 *             return cont2_array_sc(self.internal_state, rk_f, size, fdfnum,
 */
      __pyx_t_5 = (__pyx_v_fdfden <= 0.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1720
 *                 raise ValueError("shape <= 0")
 *             if fdfden <= 0:
 *                 raise ValueError("scale <= 0")             # <<<<<<<<<<<<<<
 *             return cont2_array_sc(self.internal_state, rk_f, size, fdfnum,
 *                                   fdfden)
 */
        __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_1720_32->Target(__site_call1_1720_32, __pyx_context, __pyx_t_7, ((System::Object^)"scale <= 0"));
        __pyx_t_7 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L7;
      }
      __pyx_L7:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1722
 *                 raise ValueError("scale <= 0")
 *             return cont2_array_sc(self.internal_state, rk_f, size, fdfnum,
 *                                   fdfden)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(dfnum, 0.0)):
 */
      __pyx_t_6 = cont2_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_f, __pyx_v_size, __pyx_v_fdfnum, __pyx_v_fdfden); 
      __pyx_r = __pyx_t_6;
      __pyx_t_6 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1724
 *                                   fdfden)
 * 
 *         if np.any(np.less_equal(dfnum, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("dfnum <= 0")
 *         if np.any(np.less_equal(dfden, 0.0)):
 */
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_any_1724_13->Target(__site_get_any_1724_13, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_less_equal_1724_20->Target(__site_get_less_equal_1724_20, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = 0.0;
    __pyx_t_9 = __site_call2_1724_31->Target(__site_call2_1724_31, __pyx_context, __pyx_t_8, __pyx_v_dfnum, __pyx_t_6);
    __pyx_t_8 = nullptr;
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_1724_17->Target(__site_call1_1724_17, __pyx_context, __pyx_t_7, __pyx_t_9);
    __pyx_t_7 = nullptr;
    __pyx_t_9 = nullptr;
    __pyx_t_5 = __site_istrue_1724_17->Target(__site_istrue_1724_17, __pyx_t_6);
    __pyx_t_6 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1725
 * 
 *         if np.any(np.less_equal(dfnum, 0.0)):
 *             raise ValueError("dfnum <= 0")             # <<<<<<<<<<<<<<
 *         if np.any(np.less_equal(dfden, 0.0)):
 *             raise ValueError("dfden <= 0")
 */
      __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_9 = __site_call1_1725_28->Target(__site_call1_1725_28, __pyx_context, __pyx_t_6, ((System::Object^)"dfnum <= 0"));
      __pyx_t_6 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_9, nullptr, nullptr);
      __pyx_t_9 = nullptr;
      goto __pyx_L8;
    }
    __pyx_L8:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1726
 *         if np.any(np.less_equal(dfnum, 0.0)):
 *             raise ValueError("dfnum <= 0")
 *         if np.any(np.less_equal(dfden, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("dfden <= 0")
 *         return cont2_array(self.internal_state, rk_f, size, dfnum, dfden)
 */
    __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_any_1726_13->Target(__site_get_any_1726_13, __pyx_t_9, __pyx_context);
    __pyx_t_9 = nullptr;
    __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_less_equal_1726_20->Target(__site_get_less_equal_1726_20, __pyx_t_9, __pyx_context);
    __pyx_t_9 = nullptr;
    __pyx_t_9 = 0.0;
    __pyx_t_8 = __site_call2_1726_31->Target(__site_call2_1726_31, __pyx_context, __pyx_t_7, __pyx_v_dfden, __pyx_t_9);
    __pyx_t_7 = nullptr;
    __pyx_t_9 = nullptr;
    __pyx_t_9 = __site_call1_1726_17->Target(__site_call1_1726_17, __pyx_context, __pyx_t_6, __pyx_t_8);
    __pyx_t_6 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_5 = __site_istrue_1726_17->Target(__site_istrue_1726_17, __pyx_t_9);
    __pyx_t_9 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1727
 *             raise ValueError("dfnum <= 0")
 *         if np.any(np.less_equal(dfden, 0.0)):
 *             raise ValueError("dfden <= 0")             # <<<<<<<<<<<<<<
 *         return cont2_array(self.internal_state, rk_f, size, dfnum, dfden)
 * 
 */
      __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_1727_28->Target(__site_call1_1727_28, __pyx_context, __pyx_t_9, ((System::Object^)"dfden <= 0"));
      __pyx_t_9 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L9;
    }
    __pyx_L9:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1728
 *         if np.any(np.less_equal(dfden, 0.0)):
 *             raise ValueError("dfden <= 0")
 *         return cont2_array(self.internal_state, rk_f, size, dfnum, dfden)             # <<<<<<<<<<<<<<
 * 
 *     def noncentral_f(self, dfnum, dfden, nonc, size=None):
 */
    __pyx_t_8 = cont2_array(((RandomState^)__pyx_v_self)->internal_state, rk_f, __pyx_v_size, __pyx_v_dfnum, __pyx_v_dfden); 
    __pyx_r = __pyx_t_8;
    __pyx_t_8 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1730
 *         return cont2_array(self.internal_state, rk_f, size, dfnum, dfden)
 * 
 *     def noncentral_f(self, dfnum, dfden, nonc, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         noncentral_f(dfnum, dfden, nonc, size=None)
 */

  virtual System::Object^ noncentral_f(System::Object^ dfnum, System::Object^ dfden, System::Object^ nonc, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_dfnum = nullptr;
    System::Object^ __pyx_v_dfden = nullptr;
    System::Object^ __pyx_v_nonc = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fdfnum;
    double __pyx_v_fdfden;
    double __pyx_v_fnonc;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    double __pyx_t_4;
    double __pyx_t_5;
    int __pyx_t_6;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_t_9 = nullptr;
    System::Object^ __pyx_t_10 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_dfnum = dfnum;
    __pyx_v_dfden = dfden;
    __pyx_v_nonc = nonc;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1795
 *         """
 *         cdef double fdfnum, fdfden, fnonc
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1797
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fdfnum = <double>dfnum
 *             fdfden = <double>dfden
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1798
 * 
 *         try:
 *             fdfnum = <double>dfnum             # <<<<<<<<<<<<<<
 *             fdfden = <double>dfden
 *             fnonc = <double>nonc
 */
      __pyx_t_3 = __site_cvt_double_1798_34->Target(__site_cvt_double_1798_34, __pyx_v_dfnum);
      __pyx_v_fdfnum = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1799
 *         try:
 *             fdfnum = <double>dfnum
 *             fdfden = <double>dfden             # <<<<<<<<<<<<<<
 *             fnonc = <double>nonc
 *             sc = 1
 */
      __pyx_t_4 = __site_cvt_double_1799_34->Target(__site_cvt_double_1799_34, __pyx_v_dfden);
      __pyx_v_fdfden = ((double)__pyx_t_4);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1800
 *             fdfnum = <double>dfnum
 *             fdfden = <double>dfden
 *             fnonc = <double>nonc             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_5 = __site_cvt_double_1800_32->Target(__site_cvt_double_1800_32, __pyx_v_nonc);
      __pyx_v_fnonc = ((double)__pyx_t_5);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1801
 *             fdfden = <double>dfden
 *             fnonc = <double>nonc
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1802
 *             fnonc = <double>nonc
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.noncentral_f");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1805
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fdfnum <= 1:
 *                 raise ValueError("dfnum <= 1")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1806
 * 
 *         if sc:
 *             if fdfnum <= 1:             # <<<<<<<<<<<<<<
 *                 raise ValueError("dfnum <= 1")
 *             if fdfden <= 0:
 */
      __pyx_t_6 = (__pyx_v_fdfnum <= 1.0);
      if (__pyx_t_6) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1807
 *         if sc:
 *             if fdfnum <= 1:
 *                 raise ValueError("dfnum <= 1")             # <<<<<<<<<<<<<<
 *             if fdfden <= 0:
 *                 raise ValueError("dfden <= 0")
 */
        __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_8 = __site_call1_1807_32->Target(__site_call1_1807_32, __pyx_context, __pyx_t_7, ((System::Object^)"dfnum <= 1"));
        __pyx_t_7 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
        __pyx_t_8 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1808
 *             if fdfnum <= 1:
 *                 raise ValueError("dfnum <= 1")
 *             if fdfden <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("dfden <= 0")
 *             if fnonc < 0:
 */
      __pyx_t_6 = (__pyx_v_fdfden <= 0.0);
      if (__pyx_t_6) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1809
 *                 raise ValueError("dfnum <= 1")
 *             if fdfden <= 0:
 *                 raise ValueError("dfden <= 0")             # <<<<<<<<<<<<<<
 *             if fnonc < 0:
 *                 raise ValueError("nonc < 0")
 */
        __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_7 = __site_call1_1809_32->Target(__site_call1_1809_32, __pyx_context, __pyx_t_8, ((System::Object^)"dfden <= 0"));
        __pyx_t_8 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
        __pyx_t_7 = nullptr;
        goto __pyx_L7;
      }
      __pyx_L7:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1810
 *             if fdfden <= 0:
 *                 raise ValueError("dfden <= 0")
 *             if fnonc < 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("nonc < 0")
 *             return cont3_array_sc(self.internal_state, rk_noncentral_f, size,
 */
      __pyx_t_6 = (__pyx_v_fnonc < 0.0);
      if (__pyx_t_6) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1811
 *                 raise ValueError("dfden <= 0")
 *             if fnonc < 0:
 *                 raise ValueError("nonc < 0")             # <<<<<<<<<<<<<<
 *             return cont3_array_sc(self.internal_state, rk_noncentral_f, size,
 *                                   fdfnum, fdfden, fnonc)
 */
        __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_8 = __site_call1_1811_32->Target(__site_call1_1811_32, __pyx_context, __pyx_t_7, ((System::Object^)"nonc < 0"));
        __pyx_t_7 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
        __pyx_t_8 = nullptr;
        goto __pyx_L8;
      }
      __pyx_L8:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1813
 *                 raise ValueError("nonc < 0")
 *             return cont3_array_sc(self.internal_state, rk_noncentral_f, size,
 *                                   fdfnum, fdfden, fnonc)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(dfnum, 1.0)):
 */
      __pyx_t_8 = cont3_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_noncentral_f, __pyx_v_size, __pyx_v_fdfnum, __pyx_v_fdfden, __pyx_v_fnonc); 
      __pyx_r = __pyx_t_8;
      __pyx_t_8 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1815
 *                                   fdfnum, fdfden, fnonc)
 * 
 *         if np.any(np.less_equal(dfnum, 1.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("dfnum <= 1")
 *         if np.any(np.less_equal(dfden, 0.0)):
 */
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_any_1815_13->Target(__site_get_any_1815_13, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_9 = __site_get_less_equal_1815_20->Target(__site_get_less_equal_1815_20, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = 1.0;
    __pyx_t_10 = __site_call2_1815_31->Target(__site_call2_1815_31, __pyx_context, __pyx_t_9, __pyx_v_dfnum, __pyx_t_8);
    __pyx_t_9 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_8 = __site_call1_1815_17->Target(__site_call1_1815_17, __pyx_context, __pyx_t_7, __pyx_t_10);
    __pyx_t_7 = nullptr;
    __pyx_t_10 = nullptr;
    __pyx_t_6 = __site_istrue_1815_17->Target(__site_istrue_1815_17, __pyx_t_8);
    __pyx_t_8 = nullptr;
    if (__pyx_t_6) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1816
 * 
 *         if np.any(np.less_equal(dfnum, 1.0)):
 *             raise ValueError("dfnum <= 1")             # <<<<<<<<<<<<<<
 *         if np.any(np.less_equal(dfden, 0.0)):
 *             raise ValueError("dfden <= 0")
 */
      __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_10 = __site_call1_1816_28->Target(__site_call1_1816_28, __pyx_context, __pyx_t_8, ((System::Object^)"dfnum <= 1"));
      __pyx_t_8 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_10, nullptr, nullptr);
      __pyx_t_10 = nullptr;
      goto __pyx_L9;
    }
    __pyx_L9:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1817
 *         if np.any(np.less_equal(dfnum, 1.0)):
 *             raise ValueError("dfnum <= 1")
 *         if np.any(np.less_equal(dfden, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("dfden <= 0")
 *         if np.any(np.less(nonc, 0.0)):
 */
    __pyx_t_10 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_any_1817_13->Target(__site_get_any_1817_13, __pyx_t_10, __pyx_context);
    __pyx_t_10 = nullptr;
    __pyx_t_10 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_less_equal_1817_20->Target(__site_get_less_equal_1817_20, __pyx_t_10, __pyx_context);
    __pyx_t_10 = nullptr;
    __pyx_t_10 = 0.0;
    __pyx_t_9 = __site_call2_1817_31->Target(__site_call2_1817_31, __pyx_context, __pyx_t_7, __pyx_v_dfden, __pyx_t_10);
    __pyx_t_7 = nullptr;
    __pyx_t_10 = nullptr;
    __pyx_t_10 = __site_call1_1817_17->Target(__site_call1_1817_17, __pyx_context, __pyx_t_8, __pyx_t_9);
    __pyx_t_8 = nullptr;
    __pyx_t_9 = nullptr;
    __pyx_t_6 = __site_istrue_1817_17->Target(__site_istrue_1817_17, __pyx_t_10);
    __pyx_t_10 = nullptr;
    if (__pyx_t_6) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1818
 *             raise ValueError("dfnum <= 1")
 *         if np.any(np.less_equal(dfden, 0.0)):
 *             raise ValueError("dfden <= 0")             # <<<<<<<<<<<<<<
 *         if np.any(np.less(nonc, 0.0)):
 *             raise ValueError("nonc < 0")
 */
      __pyx_t_10 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_9 = __site_call1_1818_28->Target(__site_call1_1818_28, __pyx_context, __pyx_t_10, ((System::Object^)"dfden <= 0"));
      __pyx_t_10 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_9, nullptr, nullptr);
      __pyx_t_9 = nullptr;
      goto __pyx_L10;
    }
    __pyx_L10:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1819
 *         if np.any(np.less_equal(dfden, 0.0)):
 *             raise ValueError("dfden <= 0")
 *         if np.any(np.less(nonc, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("nonc < 0")
 *         return cont3_array(self.internal_state, rk_noncentral_f, size,
 */
    __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_10 = __site_get_any_1819_13->Target(__site_get_any_1819_13, __pyx_t_9, __pyx_context);
    __pyx_t_9 = nullptr;
    __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_less_1819_20->Target(__site_get_less_1819_20, __pyx_t_9, __pyx_context);
    __pyx_t_9 = nullptr;
    __pyx_t_9 = 0.0;
    __pyx_t_7 = __site_call2_1819_25->Target(__site_call2_1819_25, __pyx_context, __pyx_t_8, __pyx_v_nonc, __pyx_t_9);
    __pyx_t_8 = nullptr;
    __pyx_t_9 = nullptr;
    __pyx_t_9 = __site_call1_1819_17->Target(__site_call1_1819_17, __pyx_context, __pyx_t_10, __pyx_t_7);
    __pyx_t_10 = nullptr;
    __pyx_t_7 = nullptr;
    __pyx_t_6 = __site_istrue_1819_17->Target(__site_istrue_1819_17, __pyx_t_9);
    __pyx_t_9 = nullptr;
    if (__pyx_t_6) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1820
 *             raise ValueError("dfden <= 0")
 *         if np.any(np.less(nonc, 0.0)):
 *             raise ValueError("nonc < 0")             # <<<<<<<<<<<<<<
 *         return cont3_array(self.internal_state, rk_noncentral_f, size,
 *                            dfnum, dfden, nonc)
 */
      __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_7 = __site_call1_1820_28->Target(__site_call1_1820_28, __pyx_context, __pyx_t_9, ((System::Object^)"nonc < 0"));
      __pyx_t_9 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
      __pyx_t_7 = nullptr;
      goto __pyx_L11;
    }
    __pyx_L11:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1822
 *             raise ValueError("nonc < 0")
 *         return cont3_array(self.internal_state, rk_noncentral_f, size,
 *                            dfnum, dfden, nonc)             # <<<<<<<<<<<<<<
 * 
 *     def chisquare(self, df, size=None):
 */
    __pyx_t_7 = cont3_array(((RandomState^)__pyx_v_self)->internal_state, rk_noncentral_f, __pyx_v_size, __pyx_v_dfnum, __pyx_v_dfden, __pyx_v_nonc); 
    __pyx_r = __pyx_t_7;
    __pyx_t_7 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1824
 *                            dfnum, dfden, nonc)
 * 
 *     def chisquare(self, df, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         chisquare(df, size=None)
 */

  virtual System::Object^ chisquare(System::Object^ df, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_df = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fdf;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    int __pyx_t_4;
    System::Object^ __pyx_t_5 = nullptr;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_df = df;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1889
 *         """
 *         cdef double fdf
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1891
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fdf = <double>df
 *             sc = 1
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1892
 * 
 *         try:
 *             fdf = <double>df             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_3 = __site_cvt_double_1892_28->Target(__site_cvt_double_1892_28, __pyx_v_df);
      __pyx_v_fdf = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1893
 *         try:
 *             fdf = <double>df
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1894
 *             fdf = <double>df
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.chisquare");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1897
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fdf <= 0:
 *                 raise ValueError("df <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1898
 * 
 *         if sc:
 *             if fdf <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("df <= 0")
 *             return cont1_array_sc(self.internal_state, rk_chisquare, size, fdf)
 */
      __pyx_t_4 = (__pyx_v_fdf <= 0.0);
      if (__pyx_t_4) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1899
 *         if sc:
 *             if fdf <= 0:
 *                 raise ValueError("df <= 0")             # <<<<<<<<<<<<<<
 *             return cont1_array_sc(self.internal_state, rk_chisquare, size, fdf)
 * 
 */
        __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_1899_32->Target(__site_call1_1899_32, __pyx_context, __pyx_t_5, ((System::Object^)"df <= 0"));
        __pyx_t_5 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1900
 *             if fdf <= 0:
 *                 raise ValueError("df <= 0")
 *             return cont1_array_sc(self.internal_state, rk_chisquare, size, fdf)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(df, 0.0)):
 */
      __pyx_t_6 = cont1_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_chisquare, __pyx_v_size, __pyx_v_fdf); 
      __pyx_r = __pyx_t_6;
      __pyx_t_6 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1902
 *             return cont1_array_sc(self.internal_state, rk_chisquare, size, fdf)
 * 
 *         if np.any(np.less_equal(df, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("df <= 0")
 *         return cont1_array(self.internal_state, rk_chisquare, size, df)
 */
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_any_1902_13->Target(__site_get_any_1902_13, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_less_equal_1902_20->Target(__site_get_less_equal_1902_20, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = 0.0;
    __pyx_t_8 = __site_call2_1902_31->Target(__site_call2_1902_31, __pyx_context, __pyx_t_7, __pyx_v_df, __pyx_t_6);
    __pyx_t_7 = nullptr;
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_1902_17->Target(__site_call1_1902_17, __pyx_context, __pyx_t_5, __pyx_t_8);
    __pyx_t_5 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_4 = __site_istrue_1902_17->Target(__site_istrue_1902_17, __pyx_t_6);
    __pyx_t_6 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1903
 * 
 *         if np.any(np.less_equal(df, 0.0)):
 *             raise ValueError("df <= 0")             # <<<<<<<<<<<<<<
 *         return cont1_array(self.internal_state, rk_chisquare, size, df)
 * 
 */
      __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_1903_28->Target(__site_call1_1903_28, __pyx_context, __pyx_t_6, ((System::Object^)"df <= 0"));
      __pyx_t_6 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L7;
    }
    __pyx_L7:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1904
 *         if np.any(np.less_equal(df, 0.0)):
 *             raise ValueError("df <= 0")
 *         return cont1_array(self.internal_state, rk_chisquare, size, df)             # <<<<<<<<<<<<<<
 * 
 *     def noncentral_chisquare(self, df, nonc, size=None):
 */
    __pyx_t_8 = cont1_array(((RandomState^)__pyx_v_self)->internal_state, rk_chisquare, __pyx_v_size, __pyx_v_df); 
    __pyx_r = __pyx_t_8;
    __pyx_t_8 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1906
 *         return cont1_array(self.internal_state, rk_chisquare, size, df)
 * 
 *     def noncentral_chisquare(self, df, nonc, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         noncentral_chisquare(df, nonc, size=None)
 */

  virtual System::Object^ noncentral_chisquare(System::Object^ df, System::Object^ nonc, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_df = nullptr;
    System::Object^ __pyx_v_nonc = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fdf;
    double __pyx_v_fnonc;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    double __pyx_t_4;
    int __pyx_t_5;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_t_9 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_df = df;
    __pyx_v_nonc = nonc;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1976
 *         """
 *         cdef double fdf, fnonc
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1978
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fdf = <double>df
 *             fnonc = <double>nonc
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1979
 * 
 *         try:
 *             fdf = <double>df             # <<<<<<<<<<<<<<
 *             fnonc = <double>nonc
 *             sc = 1
 */
      __pyx_t_3 = __site_cvt_double_1979_28->Target(__site_cvt_double_1979_28, __pyx_v_df);
      __pyx_v_fdf = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1980
 *         try:
 *             fdf = <double>df
 *             fnonc = <double>nonc             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_4 = __site_cvt_double_1980_32->Target(__site_cvt_double_1980_32, __pyx_v_nonc);
      __pyx_v_fnonc = ((double)__pyx_t_4);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1981
 *             fdf = <double>df
 *             fnonc = <double>nonc
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1982
 *             fnonc = <double>nonc
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.noncentral_chisquare");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1985
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fdf <= 1:
 *                 raise ValueError("df <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1986
 * 
 *         if sc:
 *             if fdf <= 1:             # <<<<<<<<<<<<<<
 *                 raise ValueError("df <= 0")
 *             if fnonc <= 0:
 */
      __pyx_t_5 = (__pyx_v_fdf <= 1.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1987
 *         if sc:
 *             if fdf <= 1:
 *                 raise ValueError("df <= 0")             # <<<<<<<<<<<<<<
 *             if fnonc <= 0:
 *                 raise ValueError("nonc <= 0")
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_7 = __site_call1_1987_32->Target(__site_call1_1987_32, __pyx_context, __pyx_t_6, ((System::Object^)"df <= 0"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
        __pyx_t_7 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1988
 *             if fdf <= 1:
 *                 raise ValueError("df <= 0")
 *             if fnonc <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("nonc <= 0")
 *             return cont2_array_sc(self.internal_state, rk_noncentral_chisquare,
 */
      __pyx_t_5 = (__pyx_v_fnonc <= 0.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1989
 *                 raise ValueError("df <= 0")
 *             if fnonc <= 0:
 *                 raise ValueError("nonc <= 0")             # <<<<<<<<<<<<<<
 *             return cont2_array_sc(self.internal_state, rk_noncentral_chisquare,
 *                                   size, fdf, fnonc)
 */
        __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_1989_32->Target(__site_call1_1989_32, __pyx_context, __pyx_t_7, ((System::Object^)"nonc <= 0"));
        __pyx_t_7 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L7;
      }
      __pyx_L7:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1991
 *                 raise ValueError("nonc <= 0")
 *             return cont2_array_sc(self.internal_state, rk_noncentral_chisquare,
 *                                   size, fdf, fnonc)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(df, 0.0)):
 */
      __pyx_t_6 = cont2_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_noncentral_chisquare, __pyx_v_size, __pyx_v_fdf, __pyx_v_fnonc); 
      __pyx_r = __pyx_t_6;
      __pyx_t_6 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1993
 *                                   size, fdf, fnonc)
 * 
 *         if np.any(np.less_equal(df, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("df <= 1")
 *         if np.any(np.less_equal(nonc, 0.0)):
 */
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_any_1993_13->Target(__site_get_any_1993_13, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_less_equal_1993_20->Target(__site_get_less_equal_1993_20, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = 0.0;
    __pyx_t_9 = __site_call2_1993_31->Target(__site_call2_1993_31, __pyx_context, __pyx_t_8, __pyx_v_df, __pyx_t_6);
    __pyx_t_8 = nullptr;
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_1993_17->Target(__site_call1_1993_17, __pyx_context, __pyx_t_7, __pyx_t_9);
    __pyx_t_7 = nullptr;
    __pyx_t_9 = nullptr;
    __pyx_t_5 = __site_istrue_1993_17->Target(__site_istrue_1993_17, __pyx_t_6);
    __pyx_t_6 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1994
 * 
 *         if np.any(np.less_equal(df, 0.0)):
 *             raise ValueError("df <= 1")             # <<<<<<<<<<<<<<
 *         if np.any(np.less_equal(nonc, 0.0)):
 *             raise ValueError("nonc < 0")
 */
      __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_9 = __site_call1_1994_28->Target(__site_call1_1994_28, __pyx_context, __pyx_t_6, ((System::Object^)"df <= 1"));
      __pyx_t_6 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_9, nullptr, nullptr);
      __pyx_t_9 = nullptr;
      goto __pyx_L8;
    }
    __pyx_L8:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1995
 *         if np.any(np.less_equal(df, 0.0)):
 *             raise ValueError("df <= 1")
 *         if np.any(np.less_equal(nonc, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("nonc < 0")
 *         return cont2_array(self.internal_state, rk_noncentral_chisquare, size,
 */
    __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_any_1995_13->Target(__site_get_any_1995_13, __pyx_t_9, __pyx_context);
    __pyx_t_9 = nullptr;
    __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_less_equal_1995_20->Target(__site_get_less_equal_1995_20, __pyx_t_9, __pyx_context);
    __pyx_t_9 = nullptr;
    __pyx_t_9 = 0.0;
    __pyx_t_8 = __site_call2_1995_31->Target(__site_call2_1995_31, __pyx_context, __pyx_t_7, __pyx_v_nonc, __pyx_t_9);
    __pyx_t_7 = nullptr;
    __pyx_t_9 = nullptr;
    __pyx_t_9 = __site_call1_1995_17->Target(__site_call1_1995_17, __pyx_context, __pyx_t_6, __pyx_t_8);
    __pyx_t_6 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_5 = __site_istrue_1995_17->Target(__site_istrue_1995_17, __pyx_t_9);
    __pyx_t_9 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1996
 *             raise ValueError("df <= 1")
 *         if np.any(np.less_equal(nonc, 0.0)):
 *             raise ValueError("nonc < 0")             # <<<<<<<<<<<<<<
 *         return cont2_array(self.internal_state, rk_noncentral_chisquare, size,
 *                            df, nonc)
 */
      __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_1996_28->Target(__site_call1_1996_28, __pyx_context, __pyx_t_9, ((System::Object^)"nonc < 0"));
      __pyx_t_9 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L9;
    }
    __pyx_L9:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1998
 *             raise ValueError("nonc < 0")
 *         return cont2_array(self.internal_state, rk_noncentral_chisquare, size,
 *                            df, nonc)             # <<<<<<<<<<<<<<
 * 
 *     def standard_cauchy(self, size=None):
 */
    __pyx_t_8 = cont2_array(((RandomState^)__pyx_v_self)->internal_state, rk_noncentral_chisquare, __pyx_v_size, __pyx_v_df, __pyx_v_nonc); 
    __pyx_r = __pyx_t_8;
    __pyx_t_8 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2000
 *                            df, nonc)
 * 
 *     def standard_cauchy(self, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         standard_cauchy(size=None)
 */

  virtual System::Object^ standard_cauchy([InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_size = nullptr;
    System::Object^ __pyx_r = nullptr;
    System::Object^ __pyx_t_1 = nullptr;
    System::Object^ __pyx_v_self = this;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2059
 * 
 *         """
 *         return cont0_array(self.internal_state, rk_standard_cauchy, size)             # <<<<<<<<<<<<<<
 * 
 *     def standard_t(self, df, size=None):
 */
    __pyx_t_1 = cont0_array(((RandomState^)__pyx_v_self)->internal_state, rk_standard_cauchy, __pyx_v_size); 
    __pyx_r = __pyx_t_1;
    __pyx_t_1 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2061
 *         return cont0_array(self.internal_state, rk_standard_cauchy, size)
 * 
 *     def standard_t(self, df, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         standard_t(df, size=None)
 */

  virtual System::Object^ standard_t(System::Object^ df, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_df = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fdf;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    int __pyx_t_4;
    System::Object^ __pyx_t_5 = nullptr;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_df = df;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2147
 *         """
 *         cdef double fdf
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2149
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fdf = <double>df
 *             sc = 1
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2150
 * 
 *         try:
 *             fdf = <double>df             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_3 = __site_cvt_double_2150_28->Target(__site_cvt_double_2150_28, __pyx_v_df);
      __pyx_v_fdf = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2151
 *         try:
 *             fdf = <double>df
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2152
 *             fdf = <double>df
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.standard_t");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2155
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fdf <= 0:
 *                 raise ValueError("df <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2156
 * 
 *         if sc:
 *             if fdf <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("df <= 0")
 *             return cont1_array_sc(self.internal_state, rk_standard_t, size, fdf)
 */
      __pyx_t_4 = (__pyx_v_fdf <= 0.0);
      if (__pyx_t_4) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2157
 *         if sc:
 *             if fdf <= 0:
 *                 raise ValueError("df <= 0")             # <<<<<<<<<<<<<<
 *             return cont1_array_sc(self.internal_state, rk_standard_t, size, fdf)
 * 
 */
        __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_2157_32->Target(__site_call1_2157_32, __pyx_context, __pyx_t_5, ((System::Object^)"df <= 0"));
        __pyx_t_5 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2158
 *             if fdf <= 0:
 *                 raise ValueError("df <= 0")
 *             return cont1_array_sc(self.internal_state, rk_standard_t, size, fdf)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(df, 0.0)):
 */
      __pyx_t_6 = cont1_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_standard_t, __pyx_v_size, __pyx_v_fdf); 
      __pyx_r = __pyx_t_6;
      __pyx_t_6 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2160
 *             return cont1_array_sc(self.internal_state, rk_standard_t, size, fdf)
 * 
 *         if np.any(np.less_equal(df, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("df <= 0")
 *         return cont1_array(self.internal_state, rk_standard_t, size, df)
 */
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_any_2160_13->Target(__site_get_any_2160_13, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_less_equal_2160_20->Target(__site_get_less_equal_2160_20, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = 0.0;
    __pyx_t_8 = __site_call2_2160_31->Target(__site_call2_2160_31, __pyx_context, __pyx_t_7, __pyx_v_df, __pyx_t_6);
    __pyx_t_7 = nullptr;
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_2160_17->Target(__site_call1_2160_17, __pyx_context, __pyx_t_5, __pyx_t_8);
    __pyx_t_5 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_4 = __site_istrue_2160_17->Target(__site_istrue_2160_17, __pyx_t_6);
    __pyx_t_6 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2161
 * 
 *         if np.any(np.less_equal(df, 0.0)):
 *             raise ValueError("df <= 0")             # <<<<<<<<<<<<<<
 *         return cont1_array(self.internal_state, rk_standard_t, size, df)
 * 
 */
      __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_2161_28->Target(__site_call1_2161_28, __pyx_context, __pyx_t_6, ((System::Object^)"df <= 0"));
      __pyx_t_6 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L7;
    }
    __pyx_L7:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2162
 *         if np.any(np.less_equal(df, 0.0)):
 *             raise ValueError("df <= 0")
 *         return cont1_array(self.internal_state, rk_standard_t, size, df)             # <<<<<<<<<<<<<<
 * 
 * 
 */
    __pyx_t_8 = cont1_array(((RandomState^)__pyx_v_self)->internal_state, rk_standard_t, __pyx_v_size, __pyx_v_df); 
    __pyx_r = __pyx_t_8;
    __pyx_t_8 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2165
 * 
 * 
 *     def vonmises(self, mu, kappa, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         vonmises(mu=0.0, kappa=1.0, size=None)
 */

  virtual System::Object^ vonmises(System::Object^ mu, System::Object^ kappa, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_mu = nullptr;
    System::Object^ __pyx_v_kappa = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fmu;
    double __pyx_v_fkappa;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    double __pyx_t_4;
    int __pyx_t_5;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_t_9 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_mu = mu;
    __pyx_v_kappa = kappa;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2243
 *         """
 *         cdef double fmu, fkappa
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2245
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fmu = <double>mu
 *             fkappa = <double>kappa
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2246
 * 
 *         try:
 *             fmu = <double>mu             # <<<<<<<<<<<<<<
 *             fkappa = <double>kappa
 *             sc = 1
 */
      __pyx_t_3 = __site_cvt_double_2246_28->Target(__site_cvt_double_2246_28, __pyx_v_mu);
      __pyx_v_fmu = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2247
 *         try:
 *             fmu = <double>mu
 *             fkappa = <double>kappa             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_4 = __site_cvt_double_2247_34->Target(__site_cvt_double_2247_34, __pyx_v_kappa);
      __pyx_v_fkappa = ((double)__pyx_t_4);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2248
 *             fmu = <double>mu
 *             fkappa = <double>kappa
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2249
 *             fkappa = <double>kappa
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.vonmises");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2252
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fkappa < 0:
 *                 raise ValueError("kappa < 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2253
 * 
 *         if sc:
 *             if fkappa < 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("kappa < 0")
 *             return cont2_array_sc(self.internal_state, rk_vonmises, size, fmu,
 */
      __pyx_t_5 = (__pyx_v_fkappa < 0.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2254
 *         if sc:
 *             if fkappa < 0:
 *                 raise ValueError("kappa < 0")             # <<<<<<<<<<<<<<
 *             return cont2_array_sc(self.internal_state, rk_vonmises, size, fmu,
 *                                   fkappa)
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_7 = __site_call1_2254_32->Target(__site_call1_2254_32, __pyx_context, __pyx_t_6, ((System::Object^)"kappa < 0"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
        __pyx_t_7 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2256
 *                 raise ValueError("kappa < 0")
 *             return cont2_array_sc(self.internal_state, rk_vonmises, size, fmu,
 *                                   fkappa)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less(kappa, 0.0)):
 */
      __pyx_t_7 = cont2_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_vonmises, __pyx_v_size, __pyx_v_fmu, __pyx_v_fkappa); 
      __pyx_r = __pyx_t_7;
      __pyx_t_7 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2258
 *                                   fkappa)
 * 
 *         if np.any(np.less(kappa, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("kappa < 0")
 *         return cont2_array(self.internal_state, rk_vonmises, size, mu, kappa)
 */
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_any_2258_13->Target(__site_get_any_2258_13, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_less_2258_20->Target(__site_get_less_2258_20, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = 0.0;
    __pyx_t_9 = __site_call2_2258_25->Target(__site_call2_2258_25, __pyx_context, __pyx_t_8, __pyx_v_kappa, __pyx_t_7);
    __pyx_t_8 = nullptr;
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_call1_2258_17->Target(__site_call1_2258_17, __pyx_context, __pyx_t_6, __pyx_t_9);
    __pyx_t_6 = nullptr;
    __pyx_t_9 = nullptr;
    __pyx_t_5 = __site_istrue_2258_17->Target(__site_istrue_2258_17, __pyx_t_7);
    __pyx_t_7 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2259
 * 
 *         if np.any(np.less(kappa, 0.0)):
 *             raise ValueError("kappa < 0")             # <<<<<<<<<<<<<<
 *         return cont2_array(self.internal_state, rk_vonmises, size, mu, kappa)
 * 
 */
      __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_9 = __site_call1_2259_28->Target(__site_call1_2259_28, __pyx_context, __pyx_t_7, ((System::Object^)"kappa < 0"));
      __pyx_t_7 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_9, nullptr, nullptr);
      __pyx_t_9 = nullptr;
      goto __pyx_L7;
    }
    __pyx_L7:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2260
 *         if np.any(np.less(kappa, 0.0)):
 *             raise ValueError("kappa < 0")
 *         return cont2_array(self.internal_state, rk_vonmises, size, mu, kappa)             # <<<<<<<<<<<<<<
 * 
 *     def pareto(self, a, size=None):
 */
    __pyx_t_9 = cont2_array(((RandomState^)__pyx_v_self)->internal_state, rk_vonmises, __pyx_v_size, __pyx_v_mu, __pyx_v_kappa); 
    __pyx_r = __pyx_t_9;
    __pyx_t_9 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2262
 *         return cont2_array(self.internal_state, rk_vonmises, size, mu, kappa)
 * 
 *     def pareto(self, a, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         pareto(a, size=None)
 */

  virtual System::Object^ pareto(System::Object^ a, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_a = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fa;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    int __pyx_t_4;
    System::Object^ __pyx_t_5 = nullptr;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_a = a;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2336
 *         """
 *         cdef double fa
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2338
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fa = <double>a
 *             sc = 1
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2339
 * 
 *         try:
 *             fa = <double>a             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_3 = __site_cvt_double_2339_26->Target(__site_cvt_double_2339_26, __pyx_v_a);
      __pyx_v_fa = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2340
 *         try:
 *             fa = <double>a
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2341
 *             fa = <double>a
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.pareto");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2344
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fa <= 0:
 *                 raise ValueError("a <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2345
 * 
 *         if sc:
 *             if fa <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("a <= 0")
 *             return cont1_array_sc(self.internal_state, rk_pareto, size, fa)
 */
      __pyx_t_4 = (__pyx_v_fa <= 0.0);
      if (__pyx_t_4) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2346
 *         if sc:
 *             if fa <= 0:
 *                 raise ValueError("a <= 0")             # <<<<<<<<<<<<<<
 *             return cont1_array_sc(self.internal_state, rk_pareto, size, fa)
 * 
 */
        __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_2346_32->Target(__site_call1_2346_32, __pyx_context, __pyx_t_5, ((System::Object^)"a <= 0"));
        __pyx_t_5 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2347
 *             if fa <= 0:
 *                 raise ValueError("a <= 0")
 *             return cont1_array_sc(self.internal_state, rk_pareto, size, fa)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(a, 0.0)):
 */
      __pyx_t_6 = cont1_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_pareto, __pyx_v_size, __pyx_v_fa); 
      __pyx_r = __pyx_t_6;
      __pyx_t_6 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2349
 *             return cont1_array_sc(self.internal_state, rk_pareto, size, fa)
 * 
 *         if np.any(np.less_equal(a, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("a <= 0")
 *         return cont1_array(self.internal_state, rk_pareto, size, a)
 */
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_any_2349_13->Target(__site_get_any_2349_13, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_less_equal_2349_20->Target(__site_get_less_equal_2349_20, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = 0.0;
    __pyx_t_8 = __site_call2_2349_31->Target(__site_call2_2349_31, __pyx_context, __pyx_t_7, __pyx_v_a, __pyx_t_6);
    __pyx_t_7 = nullptr;
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_2349_17->Target(__site_call1_2349_17, __pyx_context, __pyx_t_5, __pyx_t_8);
    __pyx_t_5 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_4 = __site_istrue_2349_17->Target(__site_istrue_2349_17, __pyx_t_6);
    __pyx_t_6 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2350
 * 
 *         if np.any(np.less_equal(a, 0.0)):
 *             raise ValueError("a <= 0")             # <<<<<<<<<<<<<<
 *         return cont1_array(self.internal_state, rk_pareto, size, a)
 * 
 */
      __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_2350_28->Target(__site_call1_2350_28, __pyx_context, __pyx_t_6, ((System::Object^)"a <= 0"));
      __pyx_t_6 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L7;
    }
    __pyx_L7:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2351
 *         if np.any(np.less_equal(a, 0.0)):
 *             raise ValueError("a <= 0")
 *         return cont1_array(self.internal_state, rk_pareto, size, a)             # <<<<<<<<<<<<<<
 * 
 *     def weibull(self, a, size=None):
 */
    __pyx_t_8 = cont1_array(((RandomState^)__pyx_v_self)->internal_state, rk_pareto, __pyx_v_size, __pyx_v_a); 
    __pyx_r = __pyx_t_8;
    __pyx_t_8 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2353
 *         return cont1_array(self.internal_state, rk_pareto, size, a)
 * 
 *     def weibull(self, a, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         weibull(a, size=None)
 */

  virtual System::Object^ weibull(System::Object^ a, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_a = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fa;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    int __pyx_t_4;
    System::Object^ __pyx_t_5 = nullptr;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_a = a;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2438
 *         """
 *         cdef double fa
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2440
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fa = <double>a
 *             sc = 1
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2441
 * 
 *         try:
 *             fa = <double>a             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_3 = __site_cvt_double_2441_26->Target(__site_cvt_double_2441_26, __pyx_v_a);
      __pyx_v_fa = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2442
 *         try:
 *             fa = <double>a
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2443
 *             fa = <double>a
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.weibull");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2446
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fa <= 0:
 *                 raise ValueError("a <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2447
 * 
 *         if sc:
 *             if fa <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("a <= 0")
 *             return cont1_array_sc(self.internal_state, rk_weibull, size, fa)
 */
      __pyx_t_4 = (__pyx_v_fa <= 0.0);
      if (__pyx_t_4) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2448
 *         if sc:
 *             if fa <= 0:
 *                 raise ValueError("a <= 0")             # <<<<<<<<<<<<<<
 *             return cont1_array_sc(self.internal_state, rk_weibull, size, fa)
 * 
 */
        __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_2448_32->Target(__site_call1_2448_32, __pyx_context, __pyx_t_5, ((System::Object^)"a <= 0"));
        __pyx_t_5 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2449
 *             if fa <= 0:
 *                 raise ValueError("a <= 0")
 *             return cont1_array_sc(self.internal_state, rk_weibull, size, fa)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(a, 0.0)):
 */
      __pyx_t_6 = cont1_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_weibull, __pyx_v_size, __pyx_v_fa); 
      __pyx_r = __pyx_t_6;
      __pyx_t_6 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2451
 *             return cont1_array_sc(self.internal_state, rk_weibull, size, fa)
 * 
 *         if np.any(np.less_equal(a, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("a <= 0")
 *         return cont1_array(self.internal_state, rk_weibull, size, a)
 */
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_any_2451_13->Target(__site_get_any_2451_13, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_less_equal_2451_20->Target(__site_get_less_equal_2451_20, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = 0.0;
    __pyx_t_8 = __site_call2_2451_31->Target(__site_call2_2451_31, __pyx_context, __pyx_t_7, __pyx_v_a, __pyx_t_6);
    __pyx_t_7 = nullptr;
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_2451_17->Target(__site_call1_2451_17, __pyx_context, __pyx_t_5, __pyx_t_8);
    __pyx_t_5 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_4 = __site_istrue_2451_17->Target(__site_istrue_2451_17, __pyx_t_6);
    __pyx_t_6 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2452
 * 
 *         if np.any(np.less_equal(a, 0.0)):
 *             raise ValueError("a <= 0")             # <<<<<<<<<<<<<<
 *         return cont1_array(self.internal_state, rk_weibull, size, a)
 * 
 */
      __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_2452_28->Target(__site_call1_2452_28, __pyx_context, __pyx_t_6, ((System::Object^)"a <= 0"));
      __pyx_t_6 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L7;
    }
    __pyx_L7:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2453
 *         if np.any(np.less_equal(a, 0.0)):
 *             raise ValueError("a <= 0")
 *         return cont1_array(self.internal_state, rk_weibull, size, a)             # <<<<<<<<<<<<<<
 * 
 *     def power(self, a, size=None):
 */
    __pyx_t_8 = cont1_array(((RandomState^)__pyx_v_self)->internal_state, rk_weibull, __pyx_v_size, __pyx_v_a); 
    __pyx_r = __pyx_t_8;
    __pyx_t_8 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2455
 *         return cont1_array(self.internal_state, rk_weibull, size, a)
 * 
 *     def power(self, a, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         power(a, size=None)
 */

  virtual System::Object^ power(System::Object^ a, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_a = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fa;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    int __pyx_t_4;
    System::Object^ __pyx_t_5 = nullptr;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_a = a;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2549
 *         """
 *         cdef double fa
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2551
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fa = <double>a
 *             sc = 1
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2552
 * 
 *         try:
 *             fa = <double>a             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_3 = __site_cvt_double_2552_26->Target(__site_cvt_double_2552_26, __pyx_v_a);
      __pyx_v_fa = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2553
 *         try:
 *             fa = <double>a
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2554
 *             fa = <double>a
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.power");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2557
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fa <= 0:
 *                 raise ValueError("a <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2558
 * 
 *         if sc:
 *             if fa <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("a <= 0")
 *             return cont1_array_sc(self.internal_state, rk_power, size, fa)
 */
      __pyx_t_4 = (__pyx_v_fa <= 0.0);
      if (__pyx_t_4) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2559
 *         if sc:
 *             if fa <= 0:
 *                 raise ValueError("a <= 0")             # <<<<<<<<<<<<<<
 *             return cont1_array_sc(self.internal_state, rk_power, size, fa)
 * 
 */
        __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_2559_32->Target(__site_call1_2559_32, __pyx_context, __pyx_t_5, ((System::Object^)"a <= 0"));
        __pyx_t_5 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2560
 *             if fa <= 0:
 *                 raise ValueError("a <= 0")
 *             return cont1_array_sc(self.internal_state, rk_power, size, fa)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(a, 0.0)):
 */
      __pyx_t_6 = cont1_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_power, __pyx_v_size, __pyx_v_fa); 
      __pyx_r = __pyx_t_6;
      __pyx_t_6 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2562
 *             return cont1_array_sc(self.internal_state, rk_power, size, fa)
 * 
 *         if np.any(np.less_equal(a, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("a <= 0")
 *         return cont1_array(self.internal_state, rk_power, size, a)
 */
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_any_2562_13->Target(__site_get_any_2562_13, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_less_equal_2562_20->Target(__site_get_less_equal_2562_20, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = 0.0;
    __pyx_t_8 = __site_call2_2562_31->Target(__site_call2_2562_31, __pyx_context, __pyx_t_7, __pyx_v_a, __pyx_t_6);
    __pyx_t_7 = nullptr;
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_2562_17->Target(__site_call1_2562_17, __pyx_context, __pyx_t_5, __pyx_t_8);
    __pyx_t_5 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_4 = __site_istrue_2562_17->Target(__site_istrue_2562_17, __pyx_t_6);
    __pyx_t_6 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2563
 * 
 *         if np.any(np.less_equal(a, 0.0)):
 *             raise ValueError("a <= 0")             # <<<<<<<<<<<<<<
 *         return cont1_array(self.internal_state, rk_power, size, a)
 * 
 */
      __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_2563_28->Target(__site_call1_2563_28, __pyx_context, __pyx_t_6, ((System::Object^)"a <= 0"));
      __pyx_t_6 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L7;
    }
    __pyx_L7:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2564
 *         if np.any(np.less_equal(a, 0.0)):
 *             raise ValueError("a <= 0")
 *         return cont1_array(self.internal_state, rk_power, size, a)             # <<<<<<<<<<<<<<
 * 
 *     def laplace(self, loc=0.0, scale=1.0, size=None):
 */
    __pyx_t_8 = cont1_array(((RandomState^)__pyx_v_self)->internal_state, rk_power, __pyx_v_size, __pyx_v_a); 
    __pyx_r = __pyx_t_8;
    __pyx_t_8 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2566
 *         return cont1_array(self.internal_state, rk_power, size, a)
 * 
 *     def laplace(self, loc=0.0, scale=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         laplace(loc=0.0, scale=1.0, size=None)
 */

  virtual System::Object^ laplace([InteropServices::Optional]System::Object^ loc, [InteropServices::Optional]System::Object^ scale, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_loc = nullptr;
    System::Object^ __pyx_v_scale = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_floc;
    double __pyx_v_fscale;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    double __pyx_t_4;
    int __pyx_t_5;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_t_9 = nullptr;
    System::Object^ __pyx_v_self = this;
    if (dynamic_cast<System::Reflection::Missing^>(loc) == nullptr) {
      __pyx_v_loc = loc;
    } else {
      __pyx_v_loc = __pyx_k_7;
    }
    if (dynamic_cast<System::Reflection::Missing^>(scale) == nullptr) {
      __pyx_v_scale = scale;
    } else {
      __pyx_v_scale = __pyx_k_8;
    }
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2640
 *         """
 *         cdef double floc, fscale
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2642
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             floc = <double>loc
 *             fscale = <double>scale
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2643
 * 
 *         try:
 *             floc = <double>loc             # <<<<<<<<<<<<<<
 *             fscale = <double>scale
 *             sc = 1
 */
      __pyx_t_3 = __site_cvt_double_2643_30->Target(__site_cvt_double_2643_30, __pyx_v_loc);
      __pyx_v_floc = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2644
 *         try:
 *             floc = <double>loc
 *             fscale = <double>scale             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_4 = __site_cvt_double_2644_34->Target(__site_cvt_double_2644_34, __pyx_v_scale);
      __pyx_v_fscale = ((double)__pyx_t_4);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2645
 *             floc = <double>loc
 *             fscale = <double>scale
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2646
 *             fscale = <double>scale
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.laplace");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2649
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fscale <= 0:
 *                 raise ValueError("scale <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2650
 * 
 *         if sc:
 *             if fscale <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("scale <= 0")
 *             return cont2_array_sc(self.internal_state, rk_laplace, size,
 */
      __pyx_t_5 = (__pyx_v_fscale <= 0.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2651
 *         if sc:
 *             if fscale <= 0:
 *                 raise ValueError("scale <= 0")             # <<<<<<<<<<<<<<
 *             return cont2_array_sc(self.internal_state, rk_laplace, size,
 *                                   floc, fscale)
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_7 = __site_call1_2651_32->Target(__site_call1_2651_32, __pyx_context, __pyx_t_6, ((System::Object^)"scale <= 0"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
        __pyx_t_7 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2653
 *                 raise ValueError("scale <= 0")
 *             return cont2_array_sc(self.internal_state, rk_laplace, size,
 *                                   floc, fscale)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(scale, 0.0)):
 */
      __pyx_t_7 = cont2_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_laplace, __pyx_v_size, __pyx_v_floc, __pyx_v_fscale); 
      __pyx_r = __pyx_t_7;
      __pyx_t_7 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2655
 *                                   floc, fscale)
 * 
 *         if np.any(np.less_equal(scale, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("scale <= 0")
 *         return cont2_array(self.internal_state, rk_laplace, size, loc, scale)
 */
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_any_2655_13->Target(__site_get_any_2655_13, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_less_equal_2655_20->Target(__site_get_less_equal_2655_20, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = 0.0;
    __pyx_t_9 = __site_call2_2655_31->Target(__site_call2_2655_31, __pyx_context, __pyx_t_8, __pyx_v_scale, __pyx_t_7);
    __pyx_t_8 = nullptr;
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_call1_2655_17->Target(__site_call1_2655_17, __pyx_context, __pyx_t_6, __pyx_t_9);
    __pyx_t_6 = nullptr;
    __pyx_t_9 = nullptr;
    __pyx_t_5 = __site_istrue_2655_17->Target(__site_istrue_2655_17, __pyx_t_7);
    __pyx_t_7 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2656
 * 
 *         if np.any(np.less_equal(scale, 0.0)):
 *             raise ValueError("scale <= 0")             # <<<<<<<<<<<<<<
 *         return cont2_array(self.internal_state, rk_laplace, size, loc, scale)
 * 
 */
      __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_9 = __site_call1_2656_28->Target(__site_call1_2656_28, __pyx_context, __pyx_t_7, ((System::Object^)"scale <= 0"));
      __pyx_t_7 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_9, nullptr, nullptr);
      __pyx_t_9 = nullptr;
      goto __pyx_L7;
    }
    __pyx_L7:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2657
 *         if np.any(np.less_equal(scale, 0.0)):
 *             raise ValueError("scale <= 0")
 *         return cont2_array(self.internal_state, rk_laplace, size, loc, scale)             # <<<<<<<<<<<<<<
 * 
 * 
 */
    __pyx_t_9 = cont2_array(((RandomState^)__pyx_v_self)->internal_state, rk_laplace, __pyx_v_size, __pyx_v_loc, __pyx_v_scale); 
    __pyx_r = __pyx_t_9;
    __pyx_t_9 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2660
 * 
 * 
 *     def gumbel(self, loc=0.0, scale=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         gumbel(loc=0.0, scale=1.0, size=None)
 */

  virtual System::Object^ gumbel([InteropServices::Optional]System::Object^ loc, [InteropServices::Optional]System::Object^ scale, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_loc = nullptr;
    System::Object^ __pyx_v_scale = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_floc;
    double __pyx_v_fscale;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    double __pyx_t_4;
    int __pyx_t_5;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_t_9 = nullptr;
    System::Object^ __pyx_v_self = this;
    if (dynamic_cast<System::Reflection::Missing^>(loc) == nullptr) {
      __pyx_v_loc = loc;
    } else {
      __pyx_v_loc = __pyx_k_9;
    }
    if (dynamic_cast<System::Reflection::Missing^>(scale) == nullptr) {
      __pyx_v_scale = scale;
    } else {
      __pyx_v_scale = __pyx_k_10;
    }
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2768
 *         """
 *         cdef double floc, fscale
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2770
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             floc = <double>loc
 *             fscale = <double>scale
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2771
 * 
 *         try:
 *             floc = <double>loc             # <<<<<<<<<<<<<<
 *             fscale = <double>scale
 *             sc = 1
 */
      __pyx_t_3 = __site_cvt_double_2771_30->Target(__site_cvt_double_2771_30, __pyx_v_loc);
      __pyx_v_floc = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2772
 *         try:
 *             floc = <double>loc
 *             fscale = <double>scale             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_4 = __site_cvt_double_2772_34->Target(__site_cvt_double_2772_34, __pyx_v_scale);
      __pyx_v_fscale = ((double)__pyx_t_4);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2773
 *             floc = <double>loc
 *             fscale = <double>scale
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2774
 *             fscale = <double>scale
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.gumbel");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2777
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fscale <= 0:
 *                 raise ValueError("scale <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2778
 * 
 *         if sc:
 *             if fscale <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("scale <= 0")
 *             return cont2_array_sc(self.internal_state, rk_gumbel, size,
 */
      __pyx_t_5 = (__pyx_v_fscale <= 0.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2779
 *         if sc:
 *             if fscale <= 0:
 *                 raise ValueError("scale <= 0")             # <<<<<<<<<<<<<<
 *             return cont2_array_sc(self.internal_state, rk_gumbel, size,
 *                                   floc, fscale)
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_7 = __site_call1_2779_32->Target(__site_call1_2779_32, __pyx_context, __pyx_t_6, ((System::Object^)"scale <= 0"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
        __pyx_t_7 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2781
 *                 raise ValueError("scale <= 0")
 *             return cont2_array_sc(self.internal_state, rk_gumbel, size,
 *                                   floc, fscale)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(scale, 0.0)):
 */
      __pyx_t_7 = cont2_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_gumbel, __pyx_v_size, __pyx_v_floc, __pyx_v_fscale); 
      __pyx_r = __pyx_t_7;
      __pyx_t_7 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2783
 *                                   floc, fscale)
 * 
 *         if np.any(np.less_equal(scale, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("scale <= 0")
 *         return cont2_array(self.internal_state, rk_gumbel, size, loc, scale)
 */
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_any_2783_13->Target(__site_get_any_2783_13, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_less_equal_2783_20->Target(__site_get_less_equal_2783_20, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = 0.0;
    __pyx_t_9 = __site_call2_2783_31->Target(__site_call2_2783_31, __pyx_context, __pyx_t_8, __pyx_v_scale, __pyx_t_7);
    __pyx_t_8 = nullptr;
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_call1_2783_17->Target(__site_call1_2783_17, __pyx_context, __pyx_t_6, __pyx_t_9);
    __pyx_t_6 = nullptr;
    __pyx_t_9 = nullptr;
    __pyx_t_5 = __site_istrue_2783_17->Target(__site_istrue_2783_17, __pyx_t_7);
    __pyx_t_7 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2784
 * 
 *         if np.any(np.less_equal(scale, 0.0)):
 *             raise ValueError("scale <= 0")             # <<<<<<<<<<<<<<
 *         return cont2_array(self.internal_state, rk_gumbel, size, loc, scale)
 * 
 */
      __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_9 = __site_call1_2784_28->Target(__site_call1_2784_28, __pyx_context, __pyx_t_7, ((System::Object^)"scale <= 0"));
      __pyx_t_7 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_9, nullptr, nullptr);
      __pyx_t_9 = nullptr;
      goto __pyx_L7;
    }
    __pyx_L7:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2785
 *         if np.any(np.less_equal(scale, 0.0)):
 *             raise ValueError("scale <= 0")
 *         return cont2_array(self.internal_state, rk_gumbel, size, loc, scale)             # <<<<<<<<<<<<<<
 * 
 *     def logistic(self, loc=0.0, scale=1.0, size=None):
 */
    __pyx_t_9 = cont2_array(((RandomState^)__pyx_v_self)->internal_state, rk_gumbel, __pyx_v_size, __pyx_v_loc, __pyx_v_scale); 
    __pyx_r = __pyx_t_9;
    __pyx_t_9 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2787
 *         return cont2_array(self.internal_state, rk_gumbel, size, loc, scale)
 * 
 *     def logistic(self, loc=0.0, scale=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         logistic(loc=0.0, scale=1.0, size=None)
 */

  virtual System::Object^ logistic([InteropServices::Optional]System::Object^ loc, [InteropServices::Optional]System::Object^ scale, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_loc = nullptr;
    System::Object^ __pyx_v_scale = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_floc;
    double __pyx_v_fscale;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    double __pyx_t_4;
    int __pyx_t_5;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_t_9 = nullptr;
    System::Object^ __pyx_v_self = this;
    if (dynamic_cast<System::Reflection::Missing^>(loc) == nullptr) {
      __pyx_v_loc = loc;
    } else {
      __pyx_v_loc = __pyx_k_11;
    }
    if (dynamic_cast<System::Reflection::Missing^>(scale) == nullptr) {
      __pyx_v_scale = scale;
    } else {
      __pyx_v_scale = __pyx_k_12;
    }
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2859
 *         """
 *         cdef double floc, fscale
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2861
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             floc = <double>loc
 *             fscale = <double>scale
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2862
 * 
 *         try:
 *             floc = <double>loc             # <<<<<<<<<<<<<<
 *             fscale = <double>scale
 *             sc = 1
 */
      __pyx_t_3 = __site_cvt_double_2862_30->Target(__site_cvt_double_2862_30, __pyx_v_loc);
      __pyx_v_floc = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2863
 *         try:
 *             floc = <double>loc
 *             fscale = <double>scale             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_4 = __site_cvt_double_2863_34->Target(__site_cvt_double_2863_34, __pyx_v_scale);
      __pyx_v_fscale = ((double)__pyx_t_4);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2864
 *             floc = <double>loc
 *             fscale = <double>scale
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2865
 *             fscale = <double>scale
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.logistic");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2868
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fscale <= 0:
 *                 raise ValueError("scale <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2869
 * 
 *         if sc:
 *             if fscale <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("scale <= 0")
 *             return cont2_array_sc(self.internal_state, rk_logistic, size,
 */
      __pyx_t_5 = (__pyx_v_fscale <= 0.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2870
 *         if sc:
 *             if fscale <= 0:
 *                 raise ValueError("scale <= 0")             # <<<<<<<<<<<<<<
 *             return cont2_array_sc(self.internal_state, rk_logistic, size,
 *                                   floc, fscale)
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_7 = __site_call1_2870_32->Target(__site_call1_2870_32, __pyx_context, __pyx_t_6, ((System::Object^)"scale <= 0"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
        __pyx_t_7 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2872
 *                 raise ValueError("scale <= 0")
 *             return cont2_array_sc(self.internal_state, rk_logistic, size,
 *                                   floc, fscale)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(scale, 0.0)):
 */
      __pyx_t_7 = cont2_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_logistic, __pyx_v_size, __pyx_v_floc, __pyx_v_fscale); 
      __pyx_r = __pyx_t_7;
      __pyx_t_7 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2874
 *                                   floc, fscale)
 * 
 *         if np.any(np.less_equal(scale, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("scale <= 0")
 *         return cont2_array(self.internal_state, rk_logistic, size, loc, scale)
 */
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_any_2874_13->Target(__site_get_any_2874_13, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_less_equal_2874_20->Target(__site_get_less_equal_2874_20, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = 0.0;
    __pyx_t_9 = __site_call2_2874_31->Target(__site_call2_2874_31, __pyx_context, __pyx_t_8, __pyx_v_scale, __pyx_t_7);
    __pyx_t_8 = nullptr;
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_call1_2874_17->Target(__site_call1_2874_17, __pyx_context, __pyx_t_6, __pyx_t_9);
    __pyx_t_6 = nullptr;
    __pyx_t_9 = nullptr;
    __pyx_t_5 = __site_istrue_2874_17->Target(__site_istrue_2874_17, __pyx_t_7);
    __pyx_t_7 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2875
 * 
 *         if np.any(np.less_equal(scale, 0.0)):
 *             raise ValueError("scale <= 0")             # <<<<<<<<<<<<<<
 *         return cont2_array(self.internal_state, rk_logistic, size, loc, scale)
 * 
 */
      __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_9 = __site_call1_2875_28->Target(__site_call1_2875_28, __pyx_context, __pyx_t_7, ((System::Object^)"scale <= 0"));
      __pyx_t_7 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_9, nullptr, nullptr);
      __pyx_t_9 = nullptr;
      goto __pyx_L7;
    }
    __pyx_L7:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2876
 *         if np.any(np.less_equal(scale, 0.0)):
 *             raise ValueError("scale <= 0")
 *         return cont2_array(self.internal_state, rk_logistic, size, loc, scale)             # <<<<<<<<<<<<<<
 * 
 *     def lognormal(self, mean=0.0, sigma=1.0, size=None):
 */
    __pyx_t_9 = cont2_array(((RandomState^)__pyx_v_self)->internal_state, rk_logistic, __pyx_v_size, __pyx_v_loc, __pyx_v_scale); 
    __pyx_r = __pyx_t_9;
    __pyx_t_9 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2878
 *         return cont2_array(self.internal_state, rk_logistic, size, loc, scale)
 * 
 *     def lognormal(self, mean=0.0, sigma=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         lognormal(mean=0.0, sigma=1.0, size=None)
 */

  virtual System::Object^ lognormal([InteropServices::Optional]System::Object^ mean, [InteropServices::Optional]System::Object^ sigma, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_mean = nullptr;
    System::Object^ __pyx_v_sigma = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fmean;
    double __pyx_v_fsigma;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    double __pyx_t_4;
    int __pyx_t_5;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_t_9 = nullptr;
    System::Object^ __pyx_v_self = this;
    if (dynamic_cast<System::Reflection::Missing^>(mean) == nullptr) {
      __pyx_v_mean = mean;
    } else {
      __pyx_v_mean = __pyx_k_13;
    }
    if (dynamic_cast<System::Reflection::Missing^>(sigma) == nullptr) {
      __pyx_v_sigma = sigma;
    } else {
      __pyx_v_sigma = __pyx_k_14;
    }
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2991
 *         """
 *         cdef double fmean, fsigma
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2993
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fmean = <double>mean
 *             fsigma = <double>sigma
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2994
 * 
 *         try:
 *             fmean = <double>mean             # <<<<<<<<<<<<<<
 *             fsigma = <double>sigma
 *             sc = 1
 */
      __pyx_t_3 = __site_cvt_double_2994_32->Target(__site_cvt_double_2994_32, __pyx_v_mean);
      __pyx_v_fmean = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2995
 *         try:
 *             fmean = <double>mean
 *             fsigma = <double>sigma             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_4 = __site_cvt_double_2995_34->Target(__site_cvt_double_2995_34, __pyx_v_sigma);
      __pyx_v_fsigma = ((double)__pyx_t_4);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2996
 *             fmean = <double>mean
 *             fsigma = <double>sigma
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2997
 *             fsigma = <double>sigma
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.lognormal");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3000
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fsigma <= 0:
 *                 raise ValueError("sigma <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3001
 * 
 *         if sc:
 *             if fsigma <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("sigma <= 0")
 *             return cont2_array_sc(self.internal_state, rk_lognormal, size,
 */
      __pyx_t_5 = (__pyx_v_fsigma <= 0.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3002
 *         if sc:
 *             if fsigma <= 0:
 *                 raise ValueError("sigma <= 0")             # <<<<<<<<<<<<<<
 *             return cont2_array_sc(self.internal_state, rk_lognormal, size,
 *                                   fmean, fsigma)
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_7 = __site_call1_3002_32->Target(__site_call1_3002_32, __pyx_context, __pyx_t_6, ((System::Object^)"sigma <= 0"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
        __pyx_t_7 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3004
 *                 raise ValueError("sigma <= 0")
 *             return cont2_array_sc(self.internal_state, rk_lognormal, size,
 *                                   fmean, fsigma)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(sigma, 0.0)):
 */
      __pyx_t_7 = cont2_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_lognormal, __pyx_v_size, __pyx_v_fmean, __pyx_v_fsigma); 
      __pyx_r = __pyx_t_7;
      __pyx_t_7 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3006
 *                                   fmean, fsigma)
 * 
 *         if np.any(np.less_equal(sigma, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("sigma <= 0.0")
 *         return cont2_array(self.internal_state, rk_lognormal, size,
 */
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_any_3006_13->Target(__site_get_any_3006_13, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_less_equal_3006_20->Target(__site_get_less_equal_3006_20, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = 0.0;
    __pyx_t_9 = __site_call2_3006_31->Target(__site_call2_3006_31, __pyx_context, __pyx_t_8, __pyx_v_sigma, __pyx_t_7);
    __pyx_t_8 = nullptr;
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_call1_3006_17->Target(__site_call1_3006_17, __pyx_context, __pyx_t_6, __pyx_t_9);
    __pyx_t_6 = nullptr;
    __pyx_t_9 = nullptr;
    __pyx_t_5 = __site_istrue_3006_17->Target(__site_istrue_3006_17, __pyx_t_7);
    __pyx_t_7 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3007
 * 
 *         if np.any(np.less_equal(sigma, 0.0)):
 *             raise ValueError("sigma <= 0.0")             # <<<<<<<<<<<<<<
 *         return cont2_array(self.internal_state, rk_lognormal, size,
 *                            mean, sigma)
 */
      __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_9 = __site_call1_3007_28->Target(__site_call1_3007_28, __pyx_context, __pyx_t_7, ((System::Object^)"sigma <= 0.0"));
      __pyx_t_7 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_9, nullptr, nullptr);
      __pyx_t_9 = nullptr;
      goto __pyx_L7;
    }
    __pyx_L7:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3009
 *             raise ValueError("sigma <= 0.0")
 *         return cont2_array(self.internal_state, rk_lognormal, size,
 *                            mean, sigma)             # <<<<<<<<<<<<<<
 * 
 *     def rayleigh(self, scale=1.0, size=None):
 */
    __pyx_t_9 = cont2_array(((RandomState^)__pyx_v_self)->internal_state, rk_lognormal, __pyx_v_size, __pyx_v_mean, __pyx_v_sigma); 
    __pyx_r = __pyx_t_9;
    __pyx_t_9 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3011
 *                            mean, sigma)
 * 
 *     def rayleigh(self, scale=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         rayleigh(scale=1.0, size=None)
 */

  virtual System::Object^ rayleigh([InteropServices::Optional]System::Object^ scale, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_scale = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fscale;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    int __pyx_t_4;
    System::Object^ __pyx_t_5 = nullptr;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_v_self = this;
    if (dynamic_cast<System::Reflection::Missing^>(scale) == nullptr) {
      __pyx_v_scale = scale;
    } else {
      __pyx_v_scale = __pyx_k_15;
    }
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3067
 *         """
 *         cdef double fscale
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3069
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fscale = <double>scale
 *             sc = 1
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3070
 * 
 *         try:
 *             fscale = <double>scale             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_3 = __site_cvt_double_3070_34->Target(__site_cvt_double_3070_34, __pyx_v_scale);
      __pyx_v_fscale = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3071
 *         try:
 *             fscale = <double>scale
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3072
 *             fscale = <double>scale
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.rayleigh");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3075
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fscale <= 0:
 *                 raise ValueError("scale <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3076
 * 
 *         if sc:
 *             if fscale <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("scale <= 0")
 *             return cont1_array_sc(self.internal_state, rk_rayleigh, size,
 */
      __pyx_t_4 = (__pyx_v_fscale <= 0.0);
      if (__pyx_t_4) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3077
 *         if sc:
 *             if fscale <= 0:
 *                 raise ValueError("scale <= 0")             # <<<<<<<<<<<<<<
 *             return cont1_array_sc(self.internal_state, rk_rayleigh, size,
 *                                   fscale)
 */
        __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_3077_32->Target(__site_call1_3077_32, __pyx_context, __pyx_t_5, ((System::Object^)"scale <= 0"));
        __pyx_t_5 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3079
 *                 raise ValueError("scale <= 0")
 *             return cont1_array_sc(self.internal_state, rk_rayleigh, size,
 *                                   fscale)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(scale, 0.0)):
 */
      __pyx_t_6 = cont1_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_rayleigh, __pyx_v_size, __pyx_v_fscale); 
      __pyx_r = __pyx_t_6;
      __pyx_t_6 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3081
 *                                   fscale)
 * 
 *         if np.any(np.less_equal(scale, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("scale <= 0.0")
 *         return cont1_array(self.internal_state, rk_rayleigh, size, scale)
 */
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_any_3081_13->Target(__site_get_any_3081_13, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_less_equal_3081_20->Target(__site_get_less_equal_3081_20, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = 0.0;
    __pyx_t_8 = __site_call2_3081_31->Target(__site_call2_3081_31, __pyx_context, __pyx_t_7, __pyx_v_scale, __pyx_t_6);
    __pyx_t_7 = nullptr;
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_3081_17->Target(__site_call1_3081_17, __pyx_context, __pyx_t_5, __pyx_t_8);
    __pyx_t_5 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_4 = __site_istrue_3081_17->Target(__site_istrue_3081_17, __pyx_t_6);
    __pyx_t_6 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3082
 * 
 *         if np.any(np.less_equal(scale, 0.0)):
 *             raise ValueError("scale <= 0.0")             # <<<<<<<<<<<<<<
 *         return cont1_array(self.internal_state, rk_rayleigh, size, scale)
 * 
 */
      __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_3082_28->Target(__site_call1_3082_28, __pyx_context, __pyx_t_6, ((System::Object^)"scale <= 0.0"));
      __pyx_t_6 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L7;
    }
    __pyx_L7:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3083
 *         if np.any(np.less_equal(scale, 0.0)):
 *             raise ValueError("scale <= 0.0")
 *         return cont1_array(self.internal_state, rk_rayleigh, size, scale)             # <<<<<<<<<<<<<<
 * 
 *     def wald(self, mean, scale, size=None):
 */
    __pyx_t_8 = cont1_array(((RandomState^)__pyx_v_self)->internal_state, rk_rayleigh, __pyx_v_size, __pyx_v_scale); 
    __pyx_r = __pyx_t_8;
    __pyx_t_8 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3085
 *         return cont1_array(self.internal_state, rk_rayleigh, size, scale)
 * 
 *     def wald(self, mean, scale, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         wald(mean, scale, size=None)
 */

  virtual System::Object^ wald(System::Object^ mean, System::Object^ scale, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_mean = nullptr;
    System::Object^ __pyx_v_scale = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fmean;
    double __pyx_v_fscale;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    double __pyx_t_4;
    int __pyx_t_5;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_t_9 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_mean = mean;
    __pyx_v_scale = scale;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3149
 *         """
 *         cdef double fmean, fscale
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3151
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fmean = <double>mean
 *             fscale = <double>scale
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3152
 * 
 *         try:
 *             fmean = <double>mean             # <<<<<<<<<<<<<<
 *             fscale = <double>scale
 *             sc = 1
 */
      __pyx_t_3 = __site_cvt_double_3152_32->Target(__site_cvt_double_3152_32, __pyx_v_mean);
      __pyx_v_fmean = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3153
 *         try:
 *             fmean = <double>mean
 *             fscale = <double>scale             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_4 = __site_cvt_double_3153_34->Target(__site_cvt_double_3153_34, __pyx_v_scale);
      __pyx_v_fscale = ((double)__pyx_t_4);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3154
 *             fmean = <double>mean
 *             fscale = <double>scale
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3155
 *             fscale = <double>scale
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.wald");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3158
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fmean <= 0:
 *                 raise ValueError("mean <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3159
 * 
 *         if sc:
 *             if fmean <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("mean <= 0")
 *             if fscale <= 0:
 */
      __pyx_t_5 = (__pyx_v_fmean <= 0.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3160
 *         if sc:
 *             if fmean <= 0:
 *                 raise ValueError("mean <= 0")             # <<<<<<<<<<<<<<
 *             if fscale <= 0:
 *                 raise ValueError("scale <= 0")
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_7 = __site_call1_3160_32->Target(__site_call1_3160_32, __pyx_context, __pyx_t_6, ((System::Object^)"mean <= 0"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
        __pyx_t_7 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3161
 *             if fmean <= 0:
 *                 raise ValueError("mean <= 0")
 *             if fscale <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("scale <= 0")
 *             return cont2_array_sc(self.internal_state, rk_wald, size,
 */
      __pyx_t_5 = (__pyx_v_fscale <= 0.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3162
 *                 raise ValueError("mean <= 0")
 *             if fscale <= 0:
 *                 raise ValueError("scale <= 0")             # <<<<<<<<<<<<<<
 *             return cont2_array_sc(self.internal_state, rk_wald, size,
 *                                   fmean, fscale)
 */
        __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_3162_32->Target(__site_call1_3162_32, __pyx_context, __pyx_t_7, ((System::Object^)"scale <= 0"));
        __pyx_t_7 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L7;
      }
      __pyx_L7:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3164
 *                 raise ValueError("scale <= 0")
 *             return cont2_array_sc(self.internal_state, rk_wald, size,
 *                                   fmean, fscale)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(mean,0.0)):
 */
      __pyx_t_6 = cont2_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_wald, __pyx_v_size, __pyx_v_fmean, __pyx_v_fscale); 
      __pyx_r = __pyx_t_6;
      __pyx_t_6 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3166
 *                                   fmean, fscale)
 * 
 *         if np.any(np.less_equal(mean,0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("mean <= 0.0")
 *         elif np.any(np.less_equal(scale,0.0)):
 */
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_any_3166_13->Target(__site_get_any_3166_13, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_less_equal_3166_20->Target(__site_get_less_equal_3166_20, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = 0.0;
    __pyx_t_9 = __site_call2_3166_31->Target(__site_call2_3166_31, __pyx_context, __pyx_t_8, __pyx_v_mean, __pyx_t_6);
    __pyx_t_8 = nullptr;
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_3166_17->Target(__site_call1_3166_17, __pyx_context, __pyx_t_7, __pyx_t_9);
    __pyx_t_7 = nullptr;
    __pyx_t_9 = nullptr;
    __pyx_t_5 = __site_istrue_3166_17->Target(__site_istrue_3166_17, __pyx_t_6);
    __pyx_t_6 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3167
 * 
 *         if np.any(np.less_equal(mean,0.0)):
 *             raise ValueError("mean <= 0.0")             # <<<<<<<<<<<<<<
 *         elif np.any(np.less_equal(scale,0.0)):
 *             raise ValueError("scale <= 0.0")
 */
      __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_9 = __site_call1_3167_28->Target(__site_call1_3167_28, __pyx_context, __pyx_t_6, ((System::Object^)"mean <= 0.0"));
      __pyx_t_6 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_9, nullptr, nullptr);
      __pyx_t_9 = nullptr;
      goto __pyx_L8;
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3168
 *         if np.any(np.less_equal(mean,0.0)):
 *             raise ValueError("mean <= 0.0")
 *         elif np.any(np.less_equal(scale,0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("scale <= 0.0")
 *         return cont2_array(self.internal_state, rk_wald, size, mean, scale)
 */
    __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_any_3168_15->Target(__site_get_any_3168_15, __pyx_t_9, __pyx_context);
    __pyx_t_9 = nullptr;
    __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_less_equal_3168_22->Target(__site_get_less_equal_3168_22, __pyx_t_9, __pyx_context);
    __pyx_t_9 = nullptr;
    __pyx_t_9 = 0.0;
    __pyx_t_8 = __site_call2_3168_33->Target(__site_call2_3168_33, __pyx_context, __pyx_t_7, __pyx_v_scale, __pyx_t_9);
    __pyx_t_7 = nullptr;
    __pyx_t_9 = nullptr;
    __pyx_t_9 = __site_call1_3168_19->Target(__site_call1_3168_19, __pyx_context, __pyx_t_6, __pyx_t_8);
    __pyx_t_6 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_5 = __site_istrue_3168_19->Target(__site_istrue_3168_19, __pyx_t_9);
    __pyx_t_9 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3169
 *             raise ValueError("mean <= 0.0")
 *         elif np.any(np.less_equal(scale,0.0)):
 *             raise ValueError("scale <= 0.0")             # <<<<<<<<<<<<<<
 *         return cont2_array(self.internal_state, rk_wald, size, mean, scale)
 * 
 */
      __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_3169_28->Target(__site_call1_3169_28, __pyx_context, __pyx_t_9, ((System::Object^)"scale <= 0.0"));
      __pyx_t_9 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L8;
    }
    __pyx_L8:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3170
 *         elif np.any(np.less_equal(scale,0.0)):
 *             raise ValueError("scale <= 0.0")
 *         return cont2_array(self.internal_state, rk_wald, size, mean, scale)             # <<<<<<<<<<<<<<
 * 
 * 
 */
    __pyx_t_8 = cont2_array(((RandomState^)__pyx_v_self)->internal_state, rk_wald, __pyx_v_size, __pyx_v_mean, __pyx_v_scale); 
    __pyx_r = __pyx_t_8;
    __pyx_t_8 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3174
 * 
 * 
 *     def triangular(self, left, mode, right, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         triangular(left, mode, right, size=None)
 */

  virtual System::Object^ triangular(System::Object^ left, System::Object^ mode, System::Object^ right, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_left = nullptr;
    System::Object^ __pyx_v_mode = nullptr;
    System::Object^ __pyx_v_right = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fleft;
    double __pyx_v_fmode;
    double __pyx_v_fright;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    double __pyx_t_4;
    double __pyx_t_5;
    int __pyx_t_6;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_t_9 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_left = left;
    __pyx_v_mode = mode;
    __pyx_v_right = right;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3232
 *         """
 *         cdef double fleft, fmode, fright
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3234
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fleft = <double>left
 *             fright = <double>right
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3235
 * 
 *         try:
 *             fleft = <double>left             # <<<<<<<<<<<<<<
 *             fright = <double>right
 *             fmode = <double>mode
 */
      __pyx_t_3 = __site_cvt_double_3235_32->Target(__site_cvt_double_3235_32, __pyx_v_left);
      __pyx_v_fleft = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3236
 *         try:
 *             fleft = <double>left
 *             fright = <double>right             # <<<<<<<<<<<<<<
 *             fmode = <double>mode
 *             sc = 1
 */
      __pyx_t_4 = __site_cvt_double_3236_34->Target(__site_cvt_double_3236_34, __pyx_v_right);
      __pyx_v_fright = ((double)__pyx_t_4);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3237
 *             fleft = <double>left
 *             fright = <double>right
 *             fmode = <double>mode             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_5 = __site_cvt_double_3237_32->Target(__site_cvt_double_3237_32, __pyx_v_mode);
      __pyx_v_fmode = ((double)__pyx_t_5);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3238
 *             fright = <double>right
 *             fmode = <double>mode
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3239
 *             fmode = <double>mode
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.triangular");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3242
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fleft > fmode:
 *                 raise ValueError("left > mode")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3243
 * 
 *         if sc:
 *             if fleft > fmode:             # <<<<<<<<<<<<<<
 *                 raise ValueError("left > mode")
 *             if fmode > fright:
 */
      __pyx_t_6 = (__pyx_v_fleft > __pyx_v_fmode);
      if (__pyx_t_6) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3244
 *         if sc:
 *             if fleft > fmode:
 *                 raise ValueError("left > mode")             # <<<<<<<<<<<<<<
 *             if fmode > fright:
 *                 raise ValueError("mode > right")
 */
        __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_8 = __site_call1_3244_32->Target(__site_call1_3244_32, __pyx_context, __pyx_t_7, ((System::Object^)"left > mode"));
        __pyx_t_7 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
        __pyx_t_8 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3245
 *             if fleft > fmode:
 *                 raise ValueError("left > mode")
 *             if fmode > fright:             # <<<<<<<<<<<<<<
 *                 raise ValueError("mode > right")
 *             if fleft == fright:
 */
      __pyx_t_6 = (__pyx_v_fmode > __pyx_v_fright);
      if (__pyx_t_6) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3246
 *                 raise ValueError("left > mode")
 *             if fmode > fright:
 *                 raise ValueError("mode > right")             # <<<<<<<<<<<<<<
 *             if fleft == fright:
 *                 raise ValueError("left == right")
 */
        __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_7 = __site_call1_3246_32->Target(__site_call1_3246_32, __pyx_context, __pyx_t_8, ((System::Object^)"mode > right"));
        __pyx_t_8 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
        __pyx_t_7 = nullptr;
        goto __pyx_L7;
      }
      __pyx_L7:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3247
 *             if fmode > fright:
 *                 raise ValueError("mode > right")
 *             if fleft == fright:             # <<<<<<<<<<<<<<
 *                 raise ValueError("left == right")
 *             return cont3_array_sc(self.internal_state, rk_triangular, size,
 */
      __pyx_t_6 = (__pyx_v_fleft == __pyx_v_fright);
      if (__pyx_t_6) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3248
 *                 raise ValueError("mode > right")
 *             if fleft == fright:
 *                 raise ValueError("left == right")             # <<<<<<<<<<<<<<
 *             return cont3_array_sc(self.internal_state, rk_triangular, size,
 *                                   fleft, fmode, fright)
 */
        __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_8 = __site_call1_3248_32->Target(__site_call1_3248_32, __pyx_context, __pyx_t_7, ((System::Object^)"left == right"));
        __pyx_t_7 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
        __pyx_t_8 = nullptr;
        goto __pyx_L8;
      }
      __pyx_L8:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3250
 *                 raise ValueError("left == right")
 *             return cont3_array_sc(self.internal_state, rk_triangular, size,
 *                                   fleft, fmode, fright)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.greater(left, mode)):
 */
      __pyx_t_8 = cont3_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_triangular, __pyx_v_size, __pyx_v_fleft, __pyx_v_fmode, __pyx_v_fright); 
      __pyx_r = __pyx_t_8;
      __pyx_t_8 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3252
 *                                   fleft, fmode, fright)
 * 
 *         if np.any(np.greater(left, mode)):             # <<<<<<<<<<<<<<
 *             raise ValueError("left > mode")
 *         if np.any(np.greater(mode, right)):
 */
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_any_3252_13->Target(__site_get_any_3252_13, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_9 = __site_get_greater_3252_20->Target(__site_get_greater_3252_20, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = __site_call2_3252_28->Target(__site_call2_3252_28, __pyx_context, __pyx_t_9, __pyx_v_left, __pyx_v_mode);
    __pyx_t_9 = nullptr;
    __pyx_t_9 = __site_call1_3252_17->Target(__site_call1_3252_17, __pyx_context, __pyx_t_7, __pyx_t_8);
    __pyx_t_7 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_6 = __site_istrue_3252_17->Target(__site_istrue_3252_17, __pyx_t_9);
    __pyx_t_9 = nullptr;
    if (__pyx_t_6) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3253
 * 
 *         if np.any(np.greater(left, mode)):
 *             raise ValueError("left > mode")             # <<<<<<<<<<<<<<
 *         if np.any(np.greater(mode, right)):
 *             raise ValueError("mode > right")
 */
      __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_3253_28->Target(__site_call1_3253_28, __pyx_context, __pyx_t_9, ((System::Object^)"left > mode"));
      __pyx_t_9 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L9;
    }
    __pyx_L9:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3254
 *         if np.any(np.greater(left, mode)):
 *             raise ValueError("left > mode")
 *         if np.any(np.greater(mode, right)):             # <<<<<<<<<<<<<<
 *             raise ValueError("mode > right")
 *         if np.any(np.equal(left, right)):
 */
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_9 = __site_get_any_3254_13->Target(__site_get_any_3254_13, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_greater_3254_20->Target(__site_get_greater_3254_20, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = __site_call2_3254_28->Target(__site_call2_3254_28, __pyx_context, __pyx_t_7, __pyx_v_mode, __pyx_v_right);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_call1_3254_17->Target(__site_call1_3254_17, __pyx_context, __pyx_t_9, __pyx_t_8);
    __pyx_t_9 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_6 = __site_istrue_3254_17->Target(__site_istrue_3254_17, __pyx_t_7);
    __pyx_t_7 = nullptr;
    if (__pyx_t_6) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3255
 *             raise ValueError("left > mode")
 *         if np.any(np.greater(mode, right)):
 *             raise ValueError("mode > right")             # <<<<<<<<<<<<<<
 *         if np.any(np.equal(left, right)):
 *             raise ValueError("left == right")
 */
      __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_3255_28->Target(__site_call1_3255_28, __pyx_context, __pyx_t_7, ((System::Object^)"mode > right"));
      __pyx_t_7 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L10;
    }
    __pyx_L10:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3256
 *         if np.any(np.greater(mode, right)):
 *             raise ValueError("mode > right")
 *         if np.any(np.equal(left, right)):             # <<<<<<<<<<<<<<
 *             raise ValueError("left == right")
 *         return cont3_array(self.internal_state, rk_triangular, size,
 */
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_any_3256_13->Target(__site_get_any_3256_13, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_9 = __site_get_equal_3256_20->Target(__site_get_equal_3256_20, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = __site_call2_3256_26->Target(__site_call2_3256_26, __pyx_context, __pyx_t_9, __pyx_v_left, __pyx_v_right);
    __pyx_t_9 = nullptr;
    __pyx_t_9 = __site_call1_3256_17->Target(__site_call1_3256_17, __pyx_context, __pyx_t_7, __pyx_t_8);
    __pyx_t_7 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_6 = __site_istrue_3256_17->Target(__site_istrue_3256_17, __pyx_t_9);
    __pyx_t_9 = nullptr;
    if (__pyx_t_6) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3257
 *             raise ValueError("mode > right")
 *         if np.any(np.equal(left, right)):
 *             raise ValueError("left == right")             # <<<<<<<<<<<<<<
 *         return cont3_array(self.internal_state, rk_triangular, size,
 *                            left, mode, right)
 */
      __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_3257_28->Target(__site_call1_3257_28, __pyx_context, __pyx_t_9, ((System::Object^)"left == right"));
      __pyx_t_9 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L11;
    }
    __pyx_L11:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3259
 *             raise ValueError("left == right")
 *         return cont3_array(self.internal_state, rk_triangular, size,
 *                            left, mode, right)             # <<<<<<<<<<<<<<
 * 
 *     # Complicated, discrete distributions:
 */
    __pyx_t_8 = cont3_array(((RandomState^)__pyx_v_self)->internal_state, rk_triangular, __pyx_v_size, __pyx_v_left, __pyx_v_mode, __pyx_v_right); 
    __pyx_r = __pyx_t_8;
    __pyx_t_8 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3262
 * 
 *     # Complicated, discrete distributions:
 *     def binomial(self, n, p, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         binomial(n, p, size=None)
 */

  virtual System::Object^ binomial(System::Object^ n, System::Object^ p, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_n = nullptr;
    System::Object^ __pyx_v_p = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    long __pyx_v_ln;
    double __pyx_v_fp;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    long __pyx_t_4;
    int __pyx_t_5;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_n = n;
    __pyx_v_p = p;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3345
 *         cdef long ln
 *         cdef double fp
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3347
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fp = <double>p
 *             ln = <long>n
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3348
 * 
 *         try:
 *             fp = <double>p             # <<<<<<<<<<<<<<
 *             ln = <long>n
 *             sc = 1
 */
      __pyx_t_3 = __site_cvt_double_3348_26->Target(__site_cvt_double_3348_26, __pyx_v_p);
      __pyx_v_fp = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3349
 *         try:
 *             fp = <double>p
 *             ln = <long>n             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_4 = __site_cvt_long_3349_24->Target(__site_cvt_long_3349_24, __pyx_v_n);
      __pyx_v_ln = ((long)__pyx_t_4);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3350
 *             fp = <double>p
 *             ln = <long>n
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3351
 *             ln = <long>n
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.binomial");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3354
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if ln <= 0:
 *                 raise ValueError("n <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3355
 * 
 *         if sc:
 *             if ln <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("n <= 0")
 *             if fp < 0:
 */
      __pyx_t_5 = (__pyx_v_ln <= 0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3356
 *         if sc:
 *             if ln <= 0:
 *                 raise ValueError("n <= 0")             # <<<<<<<<<<<<<<
 *             if fp < 0:
 *                 raise ValueError("p < 0")
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_7 = __site_call1_3356_32->Target(__site_call1_3356_32, __pyx_context, __pyx_t_6, ((System::Object^)"n <= 0"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
        __pyx_t_7 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3357
 *             if ln <= 0:
 *                 raise ValueError("n <= 0")
 *             if fp < 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("p < 0")
 *             elif fp > 1:
 */
      __pyx_t_5 = (__pyx_v_fp < 0.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3358
 *                 raise ValueError("n <= 0")
 *             if fp < 0:
 *                 raise ValueError("p < 0")             # <<<<<<<<<<<<<<
 *             elif fp > 1:
 *                 raise ValueError("p > 1")
 */
        __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_3358_32->Target(__site_call1_3358_32, __pyx_context, __pyx_t_7, ((System::Object^)"p < 0"));
        __pyx_t_7 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L7;
      }

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3359
 *             if fp < 0:
 *                 raise ValueError("p < 0")
 *             elif fp > 1:             # <<<<<<<<<<<<<<
 *                 raise ValueError("p > 1")
 *             return discnp_array_sc(self.internal_state, rk_binomial, size,
 */
      __pyx_t_5 = (__pyx_v_fp > 1.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3360
 *                 raise ValueError("p < 0")
 *             elif fp > 1:
 *                 raise ValueError("p > 1")             # <<<<<<<<<<<<<<
 *             return discnp_array_sc(self.internal_state, rk_binomial, size,
 *                                    ln, fp)
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_7 = __site_call1_3360_32->Target(__site_call1_3360_32, __pyx_context, __pyx_t_6, ((System::Object^)"p > 1"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
        __pyx_t_7 = nullptr;
        goto __pyx_L7;
      }
      __pyx_L7:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3362
 *                 raise ValueError("p > 1")
 *             return discnp_array_sc(self.internal_state, rk_binomial, size,
 *                                    ln, fp)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(n, 0)):
 */
      __pyx_t_7 = discnp_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_binomial, __pyx_v_size, __pyx_v_ln, __pyx_v_fp); 
      __pyx_r = __pyx_t_7;
      __pyx_t_7 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3364
 *                                    ln, fp)
 * 
 *         if np.any(np.less_equal(n, 0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("n <= 0")
 *         if np.any(np.less(p, 0)):
 */
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_any_3364_13->Target(__site_get_any_3364_13, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_less_equal_3364_20->Target(__site_get_less_equal_3364_20, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_call2_3364_31->Target(__site_call2_3364_31, __pyx_context, __pyx_t_8, __pyx_v_n, __pyx_int_0);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = __site_call1_3364_17->Target(__site_call1_3364_17, __pyx_context, __pyx_t_6, __pyx_t_7);
    __pyx_t_6 = nullptr;
    __pyx_t_7 = nullptr;
    __pyx_t_5 = __site_istrue_3364_17->Target(__site_istrue_3364_17, __pyx_t_8);
    __pyx_t_8 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3365
 * 
 *         if np.any(np.less_equal(n, 0)):
 *             raise ValueError("n <= 0")             # <<<<<<<<<<<<<<
 *         if np.any(np.less(p, 0)):
 *             raise ValueError("p < 0")
 */
      __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_7 = __site_call1_3365_28->Target(__site_call1_3365_28, __pyx_context, __pyx_t_8, ((System::Object^)"n <= 0"));
      __pyx_t_8 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
      __pyx_t_7 = nullptr;
      goto __pyx_L8;
    }
    __pyx_L8:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3366
 *         if np.any(np.less_equal(n, 0)):
 *             raise ValueError("n <= 0")
 *         if np.any(np.less(p, 0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("p < 0")
 *         if np.any(np.greater(p, 1)):
 */
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_any_3366_13->Target(__site_get_any_3366_13, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_less_3366_20->Target(__site_get_less_3366_20, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_call2_3366_25->Target(__site_call2_3366_25, __pyx_context, __pyx_t_6, __pyx_v_p, __pyx_int_0);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_3366_17->Target(__site_call1_3366_17, __pyx_context, __pyx_t_8, __pyx_t_7);
    __pyx_t_8 = nullptr;
    __pyx_t_7 = nullptr;
    __pyx_t_5 = __site_istrue_3366_17->Target(__site_istrue_3366_17, __pyx_t_6);
    __pyx_t_6 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3367
 *             raise ValueError("n <= 0")
 *         if np.any(np.less(p, 0)):
 *             raise ValueError("p < 0")             # <<<<<<<<<<<<<<
 *         if np.any(np.greater(p, 1)):
 *             raise ValueError("p > 1")
 */
      __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_7 = __site_call1_3367_28->Target(__site_call1_3367_28, __pyx_context, __pyx_t_6, ((System::Object^)"p < 0"));
      __pyx_t_6 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
      __pyx_t_7 = nullptr;
      goto __pyx_L9;
    }
    __pyx_L9:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3368
 *         if np.any(np.less(p, 0)):
 *             raise ValueError("p < 0")
 *         if np.any(np.greater(p, 1)):             # <<<<<<<<<<<<<<
 *             raise ValueError("p > 1")
 *         return discnp_array(self.internal_state, rk_binomial, size, n, p)
 */
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_any_3368_13->Target(__site_get_any_3368_13, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_greater_3368_20->Target(__site_get_greater_3368_20, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_call2_3368_28->Target(__site_call2_3368_28, __pyx_context, __pyx_t_8, __pyx_v_p, __pyx_int_1);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = __site_call1_3368_17->Target(__site_call1_3368_17, __pyx_context, __pyx_t_6, __pyx_t_7);
    __pyx_t_6 = nullptr;
    __pyx_t_7 = nullptr;
    __pyx_t_5 = __site_istrue_3368_17->Target(__site_istrue_3368_17, __pyx_t_8);
    __pyx_t_8 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3369
 *             raise ValueError("p < 0")
 *         if np.any(np.greater(p, 1)):
 *             raise ValueError("p > 1")             # <<<<<<<<<<<<<<
 *         return discnp_array(self.internal_state, rk_binomial, size, n, p)
 * 
 */
      __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_7 = __site_call1_3369_28->Target(__site_call1_3369_28, __pyx_context, __pyx_t_8, ((System::Object^)"p > 1"));
      __pyx_t_8 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
      __pyx_t_7 = nullptr;
      goto __pyx_L10;
    }
    __pyx_L10:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3370
 *         if np.any(np.greater(p, 1)):
 *             raise ValueError("p > 1")
 *         return discnp_array(self.internal_state, rk_binomial, size, n, p)             # <<<<<<<<<<<<<<
 * 
 * 
 */
    __pyx_t_7 = discnp_array(((RandomState^)__pyx_v_self)->internal_state, rk_binomial, __pyx_v_size, __pyx_v_n, __pyx_v_p); 
    __pyx_r = __pyx_t_7;
    __pyx_t_7 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3373
 * 
 * 
 *     def negative_binomial(self, n, p, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         negative_binomial(n, p, size=None)
 */

  virtual System::Object^ negative_binomial(System::Object^ n, System::Object^ p, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_n = nullptr;
    System::Object^ __pyx_v_p = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fn;
    double __pyx_v_fp;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    double __pyx_t_4;
    int __pyx_t_5;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_n = n;
    __pyx_v_p = p;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3440
 *         cdef double fn
 *         cdef double fp
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3442
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fp = <double>p
 *             fn = <double>n
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3443
 * 
 *         try:
 *             fp = <double>p             # <<<<<<<<<<<<<<
 *             fn = <double>n
 *             sc = 1
 */
      __pyx_t_3 = __site_cvt_double_3443_26->Target(__site_cvt_double_3443_26, __pyx_v_p);
      __pyx_v_fp = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3444
 *         try:
 *             fp = <double>p
 *             fn = <double>n             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_4 = __site_cvt_double_3444_26->Target(__site_cvt_double_3444_26, __pyx_v_n);
      __pyx_v_fn = ((double)__pyx_t_4);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3445
 *             fp = <double>p
 *             fn = <double>n
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3446
 *             fn = <double>n
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.negative_binomial");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3449
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fn <= 0:
 *                 raise ValueError("n <= 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3450
 * 
 *         if sc:
 *             if fn <= 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("n <= 0")
 *             if fp < 0:
 */
      __pyx_t_5 = (__pyx_v_fn <= 0.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3451
 *         if sc:
 *             if fn <= 0:
 *                 raise ValueError("n <= 0")             # <<<<<<<<<<<<<<
 *             if fp < 0:
 *                 raise ValueError("p < 0")
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_7 = __site_call1_3451_32->Target(__site_call1_3451_32, __pyx_context, __pyx_t_6, ((System::Object^)"n <= 0"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
        __pyx_t_7 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3452
 *             if fn <= 0:
 *                 raise ValueError("n <= 0")
 *             if fp < 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("p < 0")
 *             elif fp > 1:
 */
      __pyx_t_5 = (__pyx_v_fp < 0.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3453
 *                 raise ValueError("n <= 0")
 *             if fp < 0:
 *                 raise ValueError("p < 0")             # <<<<<<<<<<<<<<
 *             elif fp > 1:
 *                 raise ValueError("p > 1")
 */
        __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_3453_32->Target(__site_call1_3453_32, __pyx_context, __pyx_t_7, ((System::Object^)"p < 0"));
        __pyx_t_7 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L7;
      }

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3454
 *             if fp < 0:
 *                 raise ValueError("p < 0")
 *             elif fp > 1:             # <<<<<<<<<<<<<<
 *                 raise ValueError("p > 1")
 *             return discdd_array_sc(self.internal_state, rk_negative_binomial,
 */
      __pyx_t_5 = (__pyx_v_fp > 1.0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3455
 *                 raise ValueError("p < 0")
 *             elif fp > 1:
 *                 raise ValueError("p > 1")             # <<<<<<<<<<<<<<
 *             return discdd_array_sc(self.internal_state, rk_negative_binomial,
 *                                    size, fn, fp)
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_7 = __site_call1_3455_32->Target(__site_call1_3455_32, __pyx_context, __pyx_t_6, ((System::Object^)"p > 1"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
        __pyx_t_7 = nullptr;
        goto __pyx_L7;
      }
      __pyx_L7:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3457
 *                 raise ValueError("p > 1")
 *             return discdd_array_sc(self.internal_state, rk_negative_binomial,
 *                                    size, fn, fp)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(n, 0)):
 */
      __pyx_t_7 = discdd_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_negative_binomial, __pyx_v_size, __pyx_v_fn, __pyx_v_fp); 
      __pyx_r = __pyx_t_7;
      __pyx_t_7 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3459
 *                                    size, fn, fp)
 * 
 *         if np.any(np.less_equal(n, 0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("n <= 0")
 *         if np.any(np.less(p, 0)):
 */
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_any_3459_13->Target(__site_get_any_3459_13, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_less_equal_3459_20->Target(__site_get_less_equal_3459_20, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_call2_3459_31->Target(__site_call2_3459_31, __pyx_context, __pyx_t_8, __pyx_v_n, __pyx_int_0);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = __site_call1_3459_17->Target(__site_call1_3459_17, __pyx_context, __pyx_t_6, __pyx_t_7);
    __pyx_t_6 = nullptr;
    __pyx_t_7 = nullptr;
    __pyx_t_5 = __site_istrue_3459_17->Target(__site_istrue_3459_17, __pyx_t_8);
    __pyx_t_8 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3460
 * 
 *         if np.any(np.less_equal(n, 0)):
 *             raise ValueError("n <= 0")             # <<<<<<<<<<<<<<
 *         if np.any(np.less(p, 0)):
 *             raise ValueError("p < 0")
 */
      __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_7 = __site_call1_3460_28->Target(__site_call1_3460_28, __pyx_context, __pyx_t_8, ((System::Object^)"n <= 0"));
      __pyx_t_8 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
      __pyx_t_7 = nullptr;
      goto __pyx_L8;
    }
    __pyx_L8:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3461
 *         if np.any(np.less_equal(n, 0)):
 *             raise ValueError("n <= 0")
 *         if np.any(np.less(p, 0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("p < 0")
 *         if np.any(np.greater(p, 1)):
 */
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_any_3461_13->Target(__site_get_any_3461_13, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_less_3461_20->Target(__site_get_less_3461_20, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_call2_3461_25->Target(__site_call2_3461_25, __pyx_context, __pyx_t_6, __pyx_v_p, __pyx_int_0);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_3461_17->Target(__site_call1_3461_17, __pyx_context, __pyx_t_8, __pyx_t_7);
    __pyx_t_8 = nullptr;
    __pyx_t_7 = nullptr;
    __pyx_t_5 = __site_istrue_3461_17->Target(__site_istrue_3461_17, __pyx_t_6);
    __pyx_t_6 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3462
 *             raise ValueError("n <= 0")
 *         if np.any(np.less(p, 0)):
 *             raise ValueError("p < 0")             # <<<<<<<<<<<<<<
 *         if np.any(np.greater(p, 1)):
 *             raise ValueError("p > 1")
 */
      __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_7 = __site_call1_3462_28->Target(__site_call1_3462_28, __pyx_context, __pyx_t_6, ((System::Object^)"p < 0"));
      __pyx_t_6 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
      __pyx_t_7 = nullptr;
      goto __pyx_L9;
    }
    __pyx_L9:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3463
 *         if np.any(np.less(p, 0)):
 *             raise ValueError("p < 0")
 *         if np.any(np.greater(p, 1)):             # <<<<<<<<<<<<<<
 *             raise ValueError("p > 1")
 *         return discdd_array(self.internal_state, rk_negative_binomial,
 */
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_any_3463_13->Target(__site_get_any_3463_13, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_8 = __site_get_greater_3463_20->Target(__site_get_greater_3463_20, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_call2_3463_28->Target(__site_call2_3463_28, __pyx_context, __pyx_t_8, __pyx_v_p, __pyx_int_1);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = __site_call1_3463_17->Target(__site_call1_3463_17, __pyx_context, __pyx_t_6, __pyx_t_7);
    __pyx_t_6 = nullptr;
    __pyx_t_7 = nullptr;
    __pyx_t_5 = __site_istrue_3463_17->Target(__site_istrue_3463_17, __pyx_t_8);
    __pyx_t_8 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3464
 *             raise ValueError("p < 0")
 *         if np.any(np.greater(p, 1)):
 *             raise ValueError("p > 1")             # <<<<<<<<<<<<<<
 *         return discdd_array(self.internal_state, rk_negative_binomial,
 *                             size, n, p)
 */
      __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_7 = __site_call1_3464_28->Target(__site_call1_3464_28, __pyx_context, __pyx_t_8, ((System::Object^)"p > 1"));
      __pyx_t_8 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
      __pyx_t_7 = nullptr;
      goto __pyx_L10;
    }
    __pyx_L10:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3466
 *             raise ValueError("p > 1")
 *         return discdd_array(self.internal_state, rk_negative_binomial,
 *                             size, n, p)             # <<<<<<<<<<<<<<
 * 
 *     def poisson(self, lam=1.0, size=None):
 */
    __pyx_t_7 = discdd_array(((RandomState^)__pyx_v_self)->internal_state, rk_negative_binomial, __pyx_v_size, __pyx_v_n, __pyx_v_p); 
    __pyx_r = __pyx_t_7;
    __pyx_t_7 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3468
 *                             size, n, p)
 * 
 *     def poisson(self, lam=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         poisson(lam=1.0, size=None)
 */

  virtual System::Object^ poisson([InteropServices::Optional]System::Object^ lam, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_lam = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_flam;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    System::Object^ __pyx_t_4 = nullptr;
    int __pyx_t_5;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_v_self = this;
    if (dynamic_cast<System::Reflection::Missing^>(lam) == nullptr) {
      __pyx_v_lam = lam;
    } else {
      __pyx_v_lam = __pyx_k_16;
    }
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3517
 *         """
 *         cdef double flam
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3519
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             flam = <double>lam
 *             sc = 1
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3520
 * 
 *         try:
 *             flam = <double>lam             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_3 = __site_cvt_double_3520_30->Target(__site_cvt_double_3520_30, __pyx_v_lam);
      __pyx_v_flam = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3521
 *         try:
 *             flam = <double>lam
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3522
 *             flam = <double>lam
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.poisson");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3525
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if lam < 0:
 *                 raise ValueError("lam < 0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3526
 * 
 *         if sc:
 *             if lam < 0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("lam < 0")
 *             return discd_array_sc(self.internal_state, rk_poisson, size, flam)
 */
      __pyx_t_4 = __site_op_lt_3526_19->Target(__site_op_lt_3526_19, __pyx_v_lam, __pyx_int_0);
      __pyx_t_5 = __site_istrue_3526_19->Target(__site_istrue_3526_19, __pyx_t_4);
      __pyx_t_4 = nullptr;
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3527
 *         if sc:
 *             if lam < 0:
 *                 raise ValueError("lam < 0")             # <<<<<<<<<<<<<<
 *             return discd_array_sc(self.internal_state, rk_poisson, size, flam)
 * 
 */
        __pyx_t_4 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_3527_32->Target(__site_call1_3527_32, __pyx_context, __pyx_t_4, ((System::Object^)"lam < 0"));
        __pyx_t_4 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3528
 *             if lam < 0:
 *                 raise ValueError("lam < 0")
 *             return discd_array_sc(self.internal_state, rk_poisson, size, flam)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less(lam, 0)):
 */
      __pyx_t_6 = discd_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_poisson, __pyx_v_size, __pyx_v_flam); 
      __pyx_r = __pyx_t_6;
      __pyx_t_6 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3530
 *             return discd_array_sc(self.internal_state, rk_poisson, size, flam)
 * 
 *         if np.any(np.less(lam, 0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("lam < 0")
 *         return discd_array(self.internal_state, rk_poisson, size, lam)
 */
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_4 = __site_get_any_3530_13->Target(__site_get_any_3530_13, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_less_3530_20->Target(__site_get_less_3530_20, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call2_3530_25->Target(__site_call2_3530_25, __pyx_context, __pyx_t_7, __pyx_v_lam, __pyx_int_0);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_call1_3530_17->Target(__site_call1_3530_17, __pyx_context, __pyx_t_4, __pyx_t_6);
    __pyx_t_4 = nullptr;
    __pyx_t_6 = nullptr;
    __pyx_t_5 = __site_istrue_3530_17->Target(__site_istrue_3530_17, __pyx_t_7);
    __pyx_t_7 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3531
 * 
 *         if np.any(np.less(lam, 0)):
 *             raise ValueError("lam < 0")             # <<<<<<<<<<<<<<
 *         return discd_array(self.internal_state, rk_poisson, size, lam)
 * 
 */
      __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_6 = __site_call1_3531_28->Target(__site_call1_3531_28, __pyx_context, __pyx_t_7, ((System::Object^)"lam < 0"));
      __pyx_t_7 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
      __pyx_t_6 = nullptr;
      goto __pyx_L7;
    }
    __pyx_L7:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3532
 *         if np.any(np.less(lam, 0)):
 *             raise ValueError("lam < 0")
 *         return discd_array(self.internal_state, rk_poisson, size, lam)             # <<<<<<<<<<<<<<
 * 
 *     def zipf(self, a, size=None):
 */
    __pyx_t_6 = discd_array(((RandomState^)__pyx_v_self)->internal_state, rk_poisson, __pyx_v_size, __pyx_v_lam); 
    __pyx_r = __pyx_t_6;
    __pyx_t_6 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3534
 *         return discd_array(self.internal_state, rk_poisson, size, lam)
 * 
 *     def zipf(self, a, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         zipf(a, size=None)
 */

  virtual System::Object^ zipf(System::Object^ a, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_a = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fa;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    int __pyx_t_4;
    System::Object^ __pyx_t_5 = nullptr;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_a = a;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3611
 *         """
 *         cdef double fa
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3613
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fa = <double>a
 *             sc = 1
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3614
 * 
 *         try:
 *             fa = <double>a             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_3 = __site_cvt_double_3614_26->Target(__site_cvt_double_3614_26, __pyx_v_a);
      __pyx_v_fa = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3615
 *         try:
 *             fa = <double>a
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3616
 *             fa = <double>a
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.zipf");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3619
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fa <= 1.0:
 *                 raise ValueError("a <= 1.0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3620
 * 
 *         if sc:
 *             if fa <= 1.0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("a <= 1.0")
 *             return discd_array_sc(self.internal_state, rk_zipf, size, fa)
 */
      __pyx_t_4 = (__pyx_v_fa <= 1.0);
      if (__pyx_t_4) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3621
 *         if sc:
 *             if fa <= 1.0:
 *                 raise ValueError("a <= 1.0")             # <<<<<<<<<<<<<<
 *             return discd_array_sc(self.internal_state, rk_zipf, size, fa)
 * 
 */
        __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_3621_32->Target(__site_call1_3621_32, __pyx_context, __pyx_t_5, ((System::Object^)"a <= 1.0"));
        __pyx_t_5 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3622
 *             if fa <= 1.0:
 *                 raise ValueError("a <= 1.0")
 *             return discd_array_sc(self.internal_state, rk_zipf, size, fa)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(a, 1.0)):
 */
      __pyx_t_6 = discd_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_zipf, __pyx_v_size, __pyx_v_fa); 
      __pyx_r = __pyx_t_6;
      __pyx_t_6 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3624
 *             return discd_array_sc(self.internal_state, rk_zipf, size, fa)
 * 
 *         if np.any(np.less_equal(a, 1.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("a <= 1.0")
 *         return discd_array(self.internal_state, rk_zipf, size, a)
 */
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_any_3624_13->Target(__site_get_any_3624_13, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_less_equal_3624_20->Target(__site_get_less_equal_3624_20, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = 1.0;
    __pyx_t_8 = __site_call2_3624_31->Target(__site_call2_3624_31, __pyx_context, __pyx_t_7, __pyx_v_a, __pyx_t_6);
    __pyx_t_7 = nullptr;
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_3624_17->Target(__site_call1_3624_17, __pyx_context, __pyx_t_5, __pyx_t_8);
    __pyx_t_5 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_4 = __site_istrue_3624_17->Target(__site_istrue_3624_17, __pyx_t_6);
    __pyx_t_6 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3625
 * 
 *         if np.any(np.less_equal(a, 1.0)):
 *             raise ValueError("a <= 1.0")             # <<<<<<<<<<<<<<
 *         return discd_array(self.internal_state, rk_zipf, size, a)
 * 
 */
      __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_3625_28->Target(__site_call1_3625_28, __pyx_context, __pyx_t_6, ((System::Object^)"a <= 1.0"));
      __pyx_t_6 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L7;
    }
    __pyx_L7:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3626
 *         if np.any(np.less_equal(a, 1.0)):
 *             raise ValueError("a <= 1.0")
 *         return discd_array(self.internal_state, rk_zipf, size, a)             # <<<<<<<<<<<<<<
 * 
 *     def geometric(self, p, size=None):
 */
    __pyx_t_8 = discd_array(((RandomState^)__pyx_v_self)->internal_state, rk_zipf, __pyx_v_size, __pyx_v_a); 
    __pyx_r = __pyx_t_8;
    __pyx_t_8 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3628
 *         return discd_array(self.internal_state, rk_zipf, size, a)
 * 
 *     def geometric(self, p, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         geometric(p, size=None)
 */

  virtual System::Object^ geometric(System::Object^ p, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_p = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fp;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    int __pyx_t_4;
    System::Object^ __pyx_t_5 = nullptr;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_p = p;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3674
 *         """
 *         cdef double fp
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3676
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fp = <double>p
 *             sc = 1
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3677
 * 
 *         try:
 *             fp = <double>p             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_3 = __site_cvt_double_3677_26->Target(__site_cvt_double_3677_26, __pyx_v_p);
      __pyx_v_fp = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3678
 *         try:
 *             fp = <double>p
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3679
 *             fp = <double>p
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.geometric");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3682
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fp < 0.0:
 *                 raise ValueError("p < 0.0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3683
 * 
 *         if sc:
 *             if fp < 0.0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("p < 0.0")
 *             if fp > 1.0:
 */
      __pyx_t_4 = (__pyx_v_fp < 0.0);
      if (__pyx_t_4) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3684
 *         if sc:
 *             if fp < 0.0:
 *                 raise ValueError("p < 0.0")             # <<<<<<<<<<<<<<
 *             if fp > 1.0:
 *                 raise ValueError("p > 1.0")
 */
        __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_3684_32->Target(__site_call1_3684_32, __pyx_context, __pyx_t_5, ((System::Object^)"p < 0.0"));
        __pyx_t_5 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3685
 *             if fp < 0.0:
 *                 raise ValueError("p < 0.0")
 *             if fp > 1.0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("p > 1.0")
 *             return discd_array_sc(self.internal_state, rk_geometric, size, fp)
 */
      __pyx_t_4 = (__pyx_v_fp > 1.0);
      if (__pyx_t_4) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3686
 *                 raise ValueError("p < 0.0")
 *             if fp > 1.0:
 *                 raise ValueError("p > 1.0")             # <<<<<<<<<<<<<<
 *             return discd_array_sc(self.internal_state, rk_geometric, size, fp)
 * 
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_5 = __site_call1_3686_32->Target(__site_call1_3686_32, __pyx_context, __pyx_t_6, ((System::Object^)"p > 1.0"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_5, nullptr, nullptr);
        __pyx_t_5 = nullptr;
        goto __pyx_L7;
      }
      __pyx_L7:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3687
 *             if fp > 1.0:
 *                 raise ValueError("p > 1.0")
 *             return discd_array_sc(self.internal_state, rk_geometric, size, fp)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less(p, 0.0)):
 */
      __pyx_t_5 = discd_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_geometric, __pyx_v_size, __pyx_v_fp); 
      __pyx_r = __pyx_t_5;
      __pyx_t_5 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3689
 *             return discd_array_sc(self.internal_state, rk_geometric, size, fp)
 * 
 *         if np.any(np.less(p, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("p < 0.0")
 *         if np.any(np.greater(p, 1.0)):
 */
    __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_any_3689_13->Target(__site_get_any_3689_13, __pyx_t_5, __pyx_context);
    __pyx_t_5 = nullptr;
    __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_less_3689_20->Target(__site_get_less_3689_20, __pyx_t_5, __pyx_context);
    __pyx_t_5 = nullptr;
    __pyx_t_5 = 0.0;
    __pyx_t_8 = __site_call2_3689_25->Target(__site_call2_3689_25, __pyx_context, __pyx_t_7, __pyx_v_p, __pyx_t_5);
    __pyx_t_7 = nullptr;
    __pyx_t_5 = nullptr;
    __pyx_t_5 = __site_call1_3689_17->Target(__site_call1_3689_17, __pyx_context, __pyx_t_6, __pyx_t_8);
    __pyx_t_6 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_4 = __site_istrue_3689_17->Target(__site_istrue_3689_17, __pyx_t_5);
    __pyx_t_5 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3690
 * 
 *         if np.any(np.less(p, 0.0)):
 *             raise ValueError("p < 0.0")             # <<<<<<<<<<<<<<
 *         if np.any(np.greater(p, 1.0)):
 *             raise ValueError("p > 1.0")
 */
      __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_3690_28->Target(__site_call1_3690_28, __pyx_context, __pyx_t_5, ((System::Object^)"p < 0.0"));
      __pyx_t_5 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L8;
    }
    __pyx_L8:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3691
 *         if np.any(np.less(p, 0.0)):
 *             raise ValueError("p < 0.0")
 *         if np.any(np.greater(p, 1.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("p > 1.0")
 *         return discd_array(self.internal_state, rk_geometric, size, p)
 */
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_any_3691_13->Target(__site_get_any_3691_13, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_greater_3691_20->Target(__site_get_greater_3691_20, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = 1.0;
    __pyx_t_7 = __site_call2_3691_28->Target(__site_call2_3691_28, __pyx_context, __pyx_t_6, __pyx_v_p, __pyx_t_8);
    __pyx_t_6 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_8 = __site_call1_3691_17->Target(__site_call1_3691_17, __pyx_context, __pyx_t_5, __pyx_t_7);
    __pyx_t_5 = nullptr;
    __pyx_t_7 = nullptr;
    __pyx_t_4 = __site_istrue_3691_17->Target(__site_istrue_3691_17, __pyx_t_8);
    __pyx_t_8 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3692
 *             raise ValueError("p < 0.0")
 *         if np.any(np.greater(p, 1.0)):
 *             raise ValueError("p > 1.0")             # <<<<<<<<<<<<<<
 *         return discd_array(self.internal_state, rk_geometric, size, p)
 * 
 */
      __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_7 = __site_call1_3692_28->Target(__site_call1_3692_28, __pyx_context, __pyx_t_8, ((System::Object^)"p > 1.0"));
      __pyx_t_8 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
      __pyx_t_7 = nullptr;
      goto __pyx_L9;
    }
    __pyx_L9:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3693
 *         if np.any(np.greater(p, 1.0)):
 *             raise ValueError("p > 1.0")
 *         return discd_array(self.internal_state, rk_geometric, size, p)             # <<<<<<<<<<<<<<
 * 
 *     def hypergeometric(self, ngood, nbad, nsample, size=None):
 */
    __pyx_t_7 = discd_array(((RandomState^)__pyx_v_self)->internal_state, rk_geometric, __pyx_v_size, __pyx_v_p); 
    __pyx_r = __pyx_t_7;
    __pyx_t_7 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3695
 *         return discd_array(self.internal_state, rk_geometric, size, p)
 * 
 *     def hypergeometric(self, ngood, nbad, nsample, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         hypergeometric(ngood, nbad, nsample, size=None)
 */

  virtual System::Object^ hypergeometric(System::Object^ ngood, System::Object^ nbad, System::Object^ nsample, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_ngood = nullptr;
    System::Object^ __pyx_v_nbad = nullptr;
    System::Object^ __pyx_v_nsample = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    long __pyx_v_lngood;
    long __pyx_v_lnbad;
    long __pyx_v_lnsample;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    long __pyx_t_3;
    long __pyx_t_4;
    long __pyx_t_5;
    System::Object^ __pyx_t_6 = nullptr;
    int __pyx_t_7;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_t_9 = nullptr;
    System::Object^ __pyx_t_10 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_ngood = ngood;
    __pyx_v_nbad = nbad;
    __pyx_v_nsample = nsample;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3780
 *         """
 *         cdef long lngood, lnbad, lnsample
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3782
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             lngood = <long>ngood
 *             lnbad = <long>nbad
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3783
 * 
 *         try:
 *             lngood = <long>ngood             # <<<<<<<<<<<<<<
 *             lnbad = <long>nbad
 *             lnsample = <long>nsample
 */
      __pyx_t_3 = __site_cvt_long_3783_32->Target(__site_cvt_long_3783_32, __pyx_v_ngood);
      __pyx_v_lngood = ((long)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3784
 *         try:
 *             lngood = <long>ngood
 *             lnbad = <long>nbad             # <<<<<<<<<<<<<<
 *             lnsample = <long>nsample
 *             sc = 1
 */
      __pyx_t_4 = __site_cvt_long_3784_30->Target(__site_cvt_long_3784_30, __pyx_v_nbad);
      __pyx_v_lnbad = ((long)__pyx_t_4);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3785
 *             lngood = <long>ngood
 *             lnbad = <long>nbad
 *             lnsample = <long>nsample             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_5 = __site_cvt_long_3785_36->Target(__site_cvt_long_3785_36, __pyx_v_nsample);
      __pyx_v_lnsample = ((long)__pyx_t_5);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3786
 *             lnbad = <long>nbad
 *             lnsample = <long>nsample
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3787
 *             lnsample = <long>nsample
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.hypergeometric");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3790
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if ngood < 1:
 *                 raise ValueError("ngood < 1")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3791
 * 
 *         if sc:
 *             if ngood < 1:             # <<<<<<<<<<<<<<
 *                 raise ValueError("ngood < 1")
 *             if nbad < 1:
 */
      __pyx_t_6 = __site_op_lt_3791_21->Target(__site_op_lt_3791_21, __pyx_v_ngood, __pyx_int_1);
      __pyx_t_7 = __site_istrue_3791_21->Target(__site_istrue_3791_21, __pyx_t_6);
      __pyx_t_6 = nullptr;
      if (__pyx_t_7) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3792
 *         if sc:
 *             if ngood < 1:
 *                 raise ValueError("ngood < 1")             # <<<<<<<<<<<<<<
 *             if nbad < 1:
 *                 raise ValueError("nbad < 1")
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_8 = __site_call1_3792_32->Target(__site_call1_3792_32, __pyx_context, __pyx_t_6, ((System::Object^)"ngood < 1"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
        __pyx_t_8 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3793
 *             if ngood < 1:
 *                 raise ValueError("ngood < 1")
 *             if nbad < 1:             # <<<<<<<<<<<<<<
 *                 raise ValueError("nbad < 1")
 *             if nsample < 1:
 */
      __pyx_t_8 = __site_op_lt_3793_20->Target(__site_op_lt_3793_20, __pyx_v_nbad, __pyx_int_1);
      __pyx_t_7 = __site_istrue_3793_20->Target(__site_istrue_3793_20, __pyx_t_8);
      __pyx_t_8 = nullptr;
      if (__pyx_t_7) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3794
 *                 raise ValueError("ngood < 1")
 *             if nbad < 1:
 *                 raise ValueError("nbad < 1")             # <<<<<<<<<<<<<<
 *             if nsample < 1:
 *                 raise ValueError("nsample < 1")
 */
        __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_3794_32->Target(__site_call1_3794_32, __pyx_context, __pyx_t_8, ((System::Object^)"nbad < 1"));
        __pyx_t_8 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L7;
      }
      __pyx_L7:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3795
 *             if nbad < 1:
 *                 raise ValueError("nbad < 1")
 *             if nsample < 1:             # <<<<<<<<<<<<<<
 *                 raise ValueError("nsample < 1")
 *             if ngood + nbad < nsample:
 */
      __pyx_t_6 = __site_op_lt_3795_23->Target(__site_op_lt_3795_23, __pyx_v_nsample, __pyx_int_1);
      __pyx_t_7 = __site_istrue_3795_23->Target(__site_istrue_3795_23, __pyx_t_6);
      __pyx_t_6 = nullptr;
      if (__pyx_t_7) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3796
 *                 raise ValueError("nbad < 1")
 *             if nsample < 1:
 *                 raise ValueError("nsample < 1")             # <<<<<<<<<<<<<<
 *             if ngood + nbad < nsample:
 *                 raise ValueError("ngood + nbad < nsample")
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_8 = __site_call1_3796_32->Target(__site_call1_3796_32, __pyx_context, __pyx_t_6, ((System::Object^)"nsample < 1"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
        __pyx_t_8 = nullptr;
        goto __pyx_L8;
      }
      __pyx_L8:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3797
 *             if nsample < 1:
 *                 raise ValueError("nsample < 1")
 *             if ngood + nbad < nsample:             # <<<<<<<<<<<<<<
 *                 raise ValueError("ngood + nbad < nsample")
 *             return discnmN_array_sc(self.internal_state, rk_hypergeometric, size,
 */
      __pyx_t_8 = __site_op_add_3797_21->Target(__site_op_add_3797_21, __pyx_v_ngood, __pyx_v_nbad);
      __pyx_t_6 = __site_op_lt_3797_28->Target(__site_op_lt_3797_28, __pyx_t_8, __pyx_v_nsample);
      __pyx_t_8 = nullptr;
      __pyx_t_7 = __site_istrue_3797_28->Target(__site_istrue_3797_28, __pyx_t_6);
      __pyx_t_6 = nullptr;
      if (__pyx_t_7) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3798
 *                 raise ValueError("nsample < 1")
 *             if ngood + nbad < nsample:
 *                 raise ValueError("ngood + nbad < nsample")             # <<<<<<<<<<<<<<
 *             return discnmN_array_sc(self.internal_state, rk_hypergeometric, size,
 *                                     lngood, lnbad, lnsample)
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_8 = __site_call1_3798_32->Target(__site_call1_3798_32, __pyx_context, __pyx_t_6, ((System::Object^)"ngood + nbad < nsample"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
        __pyx_t_8 = nullptr;
        goto __pyx_L9;
      }
      __pyx_L9:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3800
 *                 raise ValueError("ngood + nbad < nsample")
 *             return discnmN_array_sc(self.internal_state, rk_hypergeometric, size,
 *                                     lngood, lnbad, lnsample)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less(ngood, 1)):
 */
      __pyx_t_8 = discnmN_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_hypergeometric, __pyx_v_size, __pyx_v_lngood, __pyx_v_lnbad, __pyx_v_lnsample); 
      __pyx_r = __pyx_t_8;
      __pyx_t_8 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3802
 *                                     lngood, lnbad, lnsample)
 * 
 *         if np.any(np.less(ngood, 1)):             # <<<<<<<<<<<<<<
 *             raise ValueError("ngood < 1")
 *         if np.any(np.less(nbad, 1)):
 */
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_any_3802_13->Target(__site_get_any_3802_13, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_9 = __site_get_less_3802_20->Target(__site_get_less_3802_20, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = __site_call2_3802_25->Target(__site_call2_3802_25, __pyx_context, __pyx_t_9, __pyx_v_ngood, __pyx_int_1);
    __pyx_t_9 = nullptr;
    __pyx_t_9 = __site_call1_3802_17->Target(__site_call1_3802_17, __pyx_context, __pyx_t_6, __pyx_t_8);
    __pyx_t_6 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_7 = __site_istrue_3802_17->Target(__site_istrue_3802_17, __pyx_t_9);
    __pyx_t_9 = nullptr;
    if (__pyx_t_7) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3803
 * 
 *         if np.any(np.less(ngood, 1)):
 *             raise ValueError("ngood < 1")             # <<<<<<<<<<<<<<
 *         if np.any(np.less(nbad, 1)):
 *             raise ValueError("nbad < 1")
 */
      __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_3803_28->Target(__site_call1_3803_28, __pyx_context, __pyx_t_9, ((System::Object^)"ngood < 1"));
      __pyx_t_9 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L10;
    }
    __pyx_L10:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3804
 *         if np.any(np.less(ngood, 1)):
 *             raise ValueError("ngood < 1")
 *         if np.any(np.less(nbad, 1)):             # <<<<<<<<<<<<<<
 *             raise ValueError("nbad < 1")
 *         if np.any(np.less(nsample, 1)):
 */
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_9 = __site_get_any_3804_13->Target(__site_get_any_3804_13, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_less_3804_20->Target(__site_get_less_3804_20, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = __site_call2_3804_25->Target(__site_call2_3804_25, __pyx_context, __pyx_t_6, __pyx_v_nbad, __pyx_int_1);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_3804_17->Target(__site_call1_3804_17, __pyx_context, __pyx_t_9, __pyx_t_8);
    __pyx_t_9 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_7 = __site_istrue_3804_17->Target(__site_istrue_3804_17, __pyx_t_6);
    __pyx_t_6 = nullptr;
    if (__pyx_t_7) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3805
 *             raise ValueError("ngood < 1")
 *         if np.any(np.less(nbad, 1)):
 *             raise ValueError("nbad < 1")             # <<<<<<<<<<<<<<
 *         if np.any(np.less(nsample, 1)):
 *             raise ValueError("nsample < 1")
 */
      __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_3805_28->Target(__site_call1_3805_28, __pyx_context, __pyx_t_6, ((System::Object^)"nbad < 1"));
      __pyx_t_6 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L11;
    }
    __pyx_L11:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3806
 *         if np.any(np.less(nbad, 1)):
 *             raise ValueError("nbad < 1")
 *         if np.any(np.less(nsample, 1)):             # <<<<<<<<<<<<<<
 *             raise ValueError("nsample < 1")
 *         if np.any(np.less(np.add(ngood, nbad), nsample)):
 */
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_any_3806_13->Target(__site_get_any_3806_13, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_9 = __site_get_less_3806_20->Target(__site_get_less_3806_20, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = __site_call2_3806_25->Target(__site_call2_3806_25, __pyx_context, __pyx_t_9, __pyx_v_nsample, __pyx_int_1);
    __pyx_t_9 = nullptr;
    __pyx_t_9 = __site_call1_3806_17->Target(__site_call1_3806_17, __pyx_context, __pyx_t_6, __pyx_t_8);
    __pyx_t_6 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_7 = __site_istrue_3806_17->Target(__site_istrue_3806_17, __pyx_t_9);
    __pyx_t_9 = nullptr;
    if (__pyx_t_7) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3807
 *             raise ValueError("nbad < 1")
 *         if np.any(np.less(nsample, 1)):
 *             raise ValueError("nsample < 1")             # <<<<<<<<<<<<<<
 *         if np.any(np.less(np.add(ngood, nbad), nsample)):
 *             raise ValueError("ngood + nbad < nsample")
 */
      __pyx_t_9 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_3807_28->Target(__site_call1_3807_28, __pyx_context, __pyx_t_9, ((System::Object^)"nsample < 1"));
      __pyx_t_9 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L12;
    }
    __pyx_L12:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3808
 *         if np.any(np.less(nsample, 1)):
 *             raise ValueError("nsample < 1")
 *         if np.any(np.less(np.add(ngood, nbad), nsample)):             # <<<<<<<<<<<<<<
 *             raise ValueError("ngood + nbad < nsample")
 *         return discnmN_array(self.internal_state, rk_hypergeometric, size,
 */
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_9 = __site_get_any_3808_13->Target(__site_get_any_3808_13, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_less_3808_20->Target(__site_get_less_3808_20, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_10 = __site_get_add_3808_28->Target(__site_get_add_3808_28, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = __site_call2_3808_32->Target(__site_call2_3808_32, __pyx_context, __pyx_t_10, __pyx_v_ngood, __pyx_v_nbad);
    __pyx_t_10 = nullptr;
    __pyx_t_10 = __site_call2_3808_25->Target(__site_call2_3808_25, __pyx_context, __pyx_t_6, __pyx_t_8, __pyx_v_nsample);
    __pyx_t_6 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_8 = __site_call1_3808_17->Target(__site_call1_3808_17, __pyx_context, __pyx_t_9, __pyx_t_10);
    __pyx_t_9 = nullptr;
    __pyx_t_10 = nullptr;
    __pyx_t_7 = __site_istrue_3808_17->Target(__site_istrue_3808_17, __pyx_t_8);
    __pyx_t_8 = nullptr;
    if (__pyx_t_7) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3809
 *             raise ValueError("nsample < 1")
 *         if np.any(np.less(np.add(ngood, nbad), nsample)):
 *             raise ValueError("ngood + nbad < nsample")             # <<<<<<<<<<<<<<
 *         return discnmN_array(self.internal_state, rk_hypergeometric, size,
 *                              ngood, nbad, nsample)
 */
      __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_10 = __site_call1_3809_28->Target(__site_call1_3809_28, __pyx_context, __pyx_t_8, ((System::Object^)"ngood + nbad < nsample"));
      __pyx_t_8 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_10, nullptr, nullptr);
      __pyx_t_10 = nullptr;
      goto __pyx_L13;
    }
    __pyx_L13:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3811
 *             raise ValueError("ngood + nbad < nsample")
 *         return discnmN_array(self.internal_state, rk_hypergeometric, size,
 *                              ngood, nbad, nsample)             # <<<<<<<<<<<<<<
 * 
 * 
 */
    __pyx_t_10 = discnmN_array(((RandomState^)__pyx_v_self)->internal_state, rk_hypergeometric, __pyx_v_size, __pyx_v_ngood, __pyx_v_nbad, __pyx_v_nsample); 
    __pyx_r = __pyx_t_10;
    __pyx_t_10 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3814
 * 
 * 
 *     def logseries(self, p, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         logseries(p, size=None)
 */

  virtual System::Object^ logseries(System::Object^ p, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_p = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    double __pyx_v_fp;
    int __pyx_v_sc;
    System::Object^ __pyx_r = nullptr;
    double __pyx_t_3;
    int __pyx_t_4;
    System::Object^ __pyx_t_5 = nullptr;
    System::Object^ __pyx_t_6 = nullptr;
    System::Object^ __pyx_t_7 = nullptr;
    System::Object^ __pyx_t_8 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_p = p;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3889
 *         """
 *         cdef double fp
 *         cdef int sc = 0             # <<<<<<<<<<<<<<
 * 
 *         try:
 */
    __pyx_v_sc = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3891
 *         cdef int sc = 0
 * 
 *         try:             # <<<<<<<<<<<<<<
 *             fp = <double>p
 *             sc = 1
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3892
 * 
 *         try:
 *             fp = <double>p             # <<<<<<<<<<<<<<
 *             sc = 1
 *         except:
 */
      __pyx_t_3 = __site_cvt_double_3892_26->Target(__site_cvt_double_3892_26, __pyx_v_p);
      __pyx_v_fp = ((double)__pyx_t_3);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3893
 *         try:
 *             fp = <double>p
 *             sc = 1             # <<<<<<<<<<<<<<
 *         except:
 *             pass
 */
      __pyx_v_sc = 1;
    } catch (System::Exception^ __pyx_lt_1) {
      System::Object^ __pyx_lt_2 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_1);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3894
 *             fp = <double>p
 *             sc = 1
 *         except:             # <<<<<<<<<<<<<<
 *             pass
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.logseries");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_1);
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3897
 *             pass
 * 
 *         if sc:             # <<<<<<<<<<<<<<
 *             if fp <= 0.0:
 *                 raise ValueError("p <= 0.0")
 */
    if (__pyx_v_sc) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3898
 * 
 *         if sc:
 *             if fp <= 0.0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("p <= 0.0")
 *             if fp >= 1.0:
 */
      __pyx_t_4 = (__pyx_v_fp <= 0.0);
      if (__pyx_t_4) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3899
 *         if sc:
 *             if fp <= 0.0:
 *                 raise ValueError("p <= 0.0")             # <<<<<<<<<<<<<<
 *             if fp >= 1.0:
 *                 raise ValueError("p >= 1.0")
 */
        __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_6 = __site_call1_3899_32->Target(__site_call1_3899_32, __pyx_context, __pyx_t_5, ((System::Object^)"p <= 0.0"));
        __pyx_t_5 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
        __pyx_t_6 = nullptr;
        goto __pyx_L6;
      }
      __pyx_L6:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3900
 *             if fp <= 0.0:
 *                 raise ValueError("p <= 0.0")
 *             if fp >= 1.0:             # <<<<<<<<<<<<<<
 *                 raise ValueError("p >= 1.0")
 *             return discd_array_sc(self.internal_state, rk_logseries, size, fp)
 */
      __pyx_t_4 = (__pyx_v_fp >= 1.0);
      if (__pyx_t_4) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3901
 *                 raise ValueError("p <= 0.0")
 *             if fp >= 1.0:
 *                 raise ValueError("p >= 1.0")             # <<<<<<<<<<<<<<
 *             return discd_array_sc(self.internal_state, rk_logseries, size, fp)
 * 
 */
        __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "ValueError");
        __pyx_t_5 = __site_call1_3901_32->Target(__site_call1_3901_32, __pyx_context, __pyx_t_6, ((System::Object^)"p >= 1.0"));
        __pyx_t_6 = nullptr;
        throw PythonOps::MakeException(__pyx_context, __pyx_t_5, nullptr, nullptr);
        __pyx_t_5 = nullptr;
        goto __pyx_L7;
      }
      __pyx_L7:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3902
 *             if fp >= 1.0:
 *                 raise ValueError("p >= 1.0")
 *             return discd_array_sc(self.internal_state, rk_logseries, size, fp)             # <<<<<<<<<<<<<<
 * 
 *         if np.any(np.less_equal(p, 0.0)):
 */
      __pyx_t_5 = discd_array_sc(((RandomState^)__pyx_v_self)->internal_state, rk_logseries, __pyx_v_size, __pyx_v_fp); 
      __pyx_r = __pyx_t_5;
      __pyx_t_5 = nullptr;
      goto __pyx_L0;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3904
 *             return discd_array_sc(self.internal_state, rk_logseries, size, fp)
 * 
 *         if np.any(np.less_equal(p, 0.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("p <= 0.0")
 *         if np.any(np.greater_equal(p, 1.0)):
 */
    __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_any_3904_13->Target(__site_get_any_3904_13, __pyx_t_5, __pyx_context);
    __pyx_t_5 = nullptr;
    __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_7 = __site_get_less_equal_3904_20->Target(__site_get_less_equal_3904_20, __pyx_t_5, __pyx_context);
    __pyx_t_5 = nullptr;
    __pyx_t_5 = 0.0;
    __pyx_t_8 = __site_call2_3904_31->Target(__site_call2_3904_31, __pyx_context, __pyx_t_7, __pyx_v_p, __pyx_t_5);
    __pyx_t_7 = nullptr;
    __pyx_t_5 = nullptr;
    __pyx_t_5 = __site_call1_3904_17->Target(__site_call1_3904_17, __pyx_context, __pyx_t_6, __pyx_t_8);
    __pyx_t_6 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_4 = __site_istrue_3904_17->Target(__site_istrue_3904_17, __pyx_t_5);
    __pyx_t_5 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3905
 * 
 *         if np.any(np.less_equal(p, 0.0)):
 *             raise ValueError("p <= 0.0")             # <<<<<<<<<<<<<<
 *         if np.any(np.greater_equal(p, 1.0)):
 *             raise ValueError("p >= 1.0")
 */
      __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_8 = __site_call1_3905_28->Target(__site_call1_3905_28, __pyx_context, __pyx_t_5, ((System::Object^)"p <= 0.0"));
      __pyx_t_5 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_8, nullptr, nullptr);
      __pyx_t_8 = nullptr;
      goto __pyx_L8;
    }
    __pyx_L8:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3906
 *         if np.any(np.less_equal(p, 0.0)):
 *             raise ValueError("p <= 0.0")
 *         if np.any(np.greater_equal(p, 1.0)):             # <<<<<<<<<<<<<<
 *             raise ValueError("p >= 1.0")
 *         return discd_array(self.internal_state, rk_logseries, size, p)
 */
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_any_3906_13->Target(__site_get_any_3906_13, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_6 = __site_get_greater_equal_3906_20->Target(__site_get_greater_equal_3906_20, __pyx_t_8, __pyx_context);
    __pyx_t_8 = nullptr;
    __pyx_t_8 = 1.0;
    __pyx_t_7 = __site_call2_3906_34->Target(__site_call2_3906_34, __pyx_context, __pyx_t_6, __pyx_v_p, __pyx_t_8);
    __pyx_t_6 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_t_8 = __site_call1_3906_17->Target(__site_call1_3906_17, __pyx_context, __pyx_t_5, __pyx_t_7);
    __pyx_t_5 = nullptr;
    __pyx_t_7 = nullptr;
    __pyx_t_4 = __site_istrue_3906_17->Target(__site_istrue_3906_17, __pyx_t_8);
    __pyx_t_8 = nullptr;
    if (__pyx_t_4) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3907
 *             raise ValueError("p <= 0.0")
 *         if np.any(np.greater_equal(p, 1.0)):
 *             raise ValueError("p >= 1.0")             # <<<<<<<<<<<<<<
 *         return discd_array(self.internal_state, rk_logseries, size, p)
 * 
 */
      __pyx_t_8 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_7 = __site_call1_3907_28->Target(__site_call1_3907_28, __pyx_context, __pyx_t_8, ((System::Object^)"p >= 1.0"));
      __pyx_t_8 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
      __pyx_t_7 = nullptr;
      goto __pyx_L9;
    }
    __pyx_L9:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3908
 *         if np.any(np.greater_equal(p, 1.0)):
 *             raise ValueError("p >= 1.0")
 *         return discd_array(self.internal_state, rk_logseries, size, p)             # <<<<<<<<<<<<<<
 * 
 *     # Multivariate distributions:
 */
    __pyx_t_7 = discd_array(((RandomState^)__pyx_v_self)->internal_state, rk_logseries, __pyx_v_size, __pyx_v_p); 
    __pyx_r = __pyx_t_7;
    __pyx_t_7 = nullptr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3911
 * 
 *     # Multivariate distributions:
 *     def multivariate_normal(self, mean, cov, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         multivariate_normal(mean, cov[, size])
 */

  virtual System::Object^ multivariate_normal(System::Object^ mean, System::Object^ cov, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_mean = nullptr;
    System::Object^ __pyx_v_cov = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    System::Object^ __pyx_v_shape;
    System::Object^ __pyx_v_final_shape;
    System::Object^ __pyx_v_x;
    System::Object^ __pyx_v_svd;
    System::Object^ __pyx_v_u;
    System::Object^ __pyx_v_s;
    System::Object^ __pyx_v_v;
    System::Object^ __pyx_r = nullptr;
    System::Object^ __pyx_t_1 = nullptr;
    System::Object^ __pyx_t_2 = nullptr;
    int __pyx_t_3;
    System::Object^ __pyx_t_4 = nullptr;
    int __pyx_t_5;
    int __pyx_t_6;
    Py_ssize_t __pyx_t_7;
    array<System::Object^>^ __pyx_t_8;
    System::Object^ __pyx_t_9 = nullptr;
    System::Object^ __pyx_v_self = this;
    __pyx_v_mean = mean;
    __pyx_v_cov = cov;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }
    __pyx_v_shape = nullptr;
    __pyx_v_final_shape = nullptr;
    __pyx_v_x = nullptr;
    __pyx_v_svd = nullptr;
    __pyx_v_u = nullptr;
    __pyx_v_s = nullptr;
    __pyx_v_v = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4004
 *         """
 *         # Check preconditions on arguments
 *         mean = np.array(mean)             # <<<<<<<<<<<<<<
 *         cov = np.array(cov)
 *         if size is None:
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_2 = __site_get_array_4004_17->Target(__site_get_array_4004_17, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call1_4004_23->Target(__site_call1_4004_23, __pyx_context, __pyx_t_2, __pyx_v_mean);
    __pyx_t_2 = nullptr;
    __pyx_v_mean = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4005
 *         # Check preconditions on arguments
 *         mean = np.array(mean)
 *         cov = np.array(cov)             # <<<<<<<<<<<<<<
 *         if size is None:
 *             shape = []
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_2 = __site_get_array_4005_16->Target(__site_get_array_4005_16, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call1_4005_22->Target(__site_call1_4005_22, __pyx_context, __pyx_t_2, __pyx_v_cov);
    __pyx_t_2 = nullptr;
    __pyx_v_cov = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4006
 *         mean = np.array(mean)
 *         cov = np.array(cov)
 *         if size is None:             # <<<<<<<<<<<<<<
 *             shape = []
 *         else:
 */
    __pyx_t_3 = (__pyx_v_size == nullptr);
    if (__pyx_t_3) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4007
 *         cov = np.array(cov)
 *         if size is None:
 *             shape = []             # <<<<<<<<<<<<<<
 *         else:
 *             shape = size
 */
      __pyx_t_1 = PythonOps::MakeListNoCopy(gcnew array<System::Object^>{});
      __pyx_v_shape = ((System::Object^)__pyx_t_1);
      __pyx_t_1 = nullptr;
      goto __pyx_L5;
    }
    /*else*/ {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4009
 *             shape = []
 *         else:
 *             shape = size             # <<<<<<<<<<<<<<
 *         if len(mean.shape) != 1:
 *                raise ValueError("mean must be 1 dimensional")
 */
      __pyx_v_shape = __pyx_v_size;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4010
 *         else:
 *             shape = size
 *         if len(mean.shape) != 1:             # <<<<<<<<<<<<<<
 *                raise ValueError("mean must be 1 dimensional")
 *         if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "len");
    __pyx_t_2 = __site_get_shape_4010_19->Target(__site_get_shape_4010_19, __pyx_v_mean, __pyx_context);
    __pyx_t_4 = __site_call1_4010_14->Target(__site_call1_4010_14, __pyx_context, __pyx_t_1, __pyx_t_2);
    __pyx_t_1 = nullptr;
    __pyx_t_2 = nullptr;
    __pyx_t_2 = __site_op_ne_4010_27->Target(__site_op_ne_4010_27, __pyx_t_4, __pyx_int_1);
    __pyx_t_4 = nullptr;
    __pyx_t_3 = __site_istrue_4010_27->Target(__site_istrue_4010_27, __pyx_t_2);
    __pyx_t_2 = nullptr;
    if (__pyx_t_3) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4011
 *             shape = size
 *         if len(mean.shape) != 1:
 *                raise ValueError("mean must be 1 dimensional")             # <<<<<<<<<<<<<<
 *         if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):
 *                raise ValueError("cov must be 2 dimensional and square")
 */
      __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_4 = __site_call1_4011_31->Target(__site_call1_4011_31, __pyx_context, __pyx_t_2, ((System::Object^)"mean must be 1 dimensional"));
      __pyx_t_2 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_4, nullptr, nullptr);
      __pyx_t_4 = nullptr;
      goto __pyx_L6;
    }
    __pyx_L6:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4012
 *         if len(mean.shape) != 1:
 *                raise ValueError("mean must be 1 dimensional")
 *         if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):             # <<<<<<<<<<<<<<
 *                raise ValueError("cov must be 2 dimensional and square")
 *         if mean.shape[0] != cov.shape[0]:
 */
    __pyx_t_4 = PythonOps::GetGlobal(__pyx_context, "len");
    __pyx_t_2 = __site_get_shape_4012_19->Target(__site_get_shape_4012_19, __pyx_v_cov, __pyx_context);
    __pyx_t_1 = __site_call1_4012_15->Target(__site_call1_4012_15, __pyx_context, __pyx_t_4, __pyx_t_2);
    __pyx_t_4 = nullptr;
    __pyx_t_2 = nullptr;
    __pyx_t_2 = __site_op_ne_4012_27->Target(__site_op_ne_4012_27, __pyx_t_1, __pyx_int_2);
    __pyx_t_1 = nullptr;
    __pyx_t_3 = __site_istrue_4012_27->Target(__site_istrue_4012_27, __pyx_t_2);
    __pyx_t_2 = nullptr;
    if (!__pyx_t_3) {
      __pyx_t_2 = __site_get_shape_4012_40->Target(__site_get_shape_4012_40, __pyx_v_cov, __pyx_context);
      __pyx_t_1 = __site_getindex_4012_46->Target(__site_getindex_4012_46, __pyx_t_2, ((System::Object^)0));
      __pyx_t_2 = nullptr;
      __pyx_t_2 = __site_get_shape_4012_56->Target(__site_get_shape_4012_56, __pyx_v_cov, __pyx_context);
      __pyx_t_4 = __site_getindex_4012_62->Target(__site_getindex_4012_62, __pyx_t_2, ((System::Object^)1));
      __pyx_t_2 = nullptr;
      __pyx_t_2 = __site_op_ne_4012_50->Target(__site_op_ne_4012_50, __pyx_t_1, __pyx_t_4);
      __pyx_t_1 = nullptr;
      __pyx_t_4 = nullptr;
      __pyx_t_5 = __site_istrue_4012_50->Target(__site_istrue_4012_50, __pyx_t_2);
      __pyx_t_2 = nullptr;
      __pyx_t_6 = __pyx_t_5;
    } else {
      __pyx_t_6 = __pyx_t_3;
    }
    if (__pyx_t_6) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4013
 *                raise ValueError("mean must be 1 dimensional")
 *         if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):
 *                raise ValueError("cov must be 2 dimensional and square")             # <<<<<<<<<<<<<<
 *         if mean.shape[0] != cov.shape[0]:
 *                raise ValueError("mean and cov must have same length")
 */
      __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_4 = __site_call1_4013_31->Target(__site_call1_4013_31, __pyx_context, __pyx_t_2, ((System::Object^)"cov must be 2 dimensional and square"));
      __pyx_t_2 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_4, nullptr, nullptr);
      __pyx_t_4 = nullptr;
      goto __pyx_L7;
    }
    __pyx_L7:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4014
 *         if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):
 *                raise ValueError("cov must be 2 dimensional and square")
 *         if mean.shape[0] != cov.shape[0]:             # <<<<<<<<<<<<<<
 *                raise ValueError("mean and cov must have same length")
 *         # Compute shape of output
 */
    __pyx_t_4 = __site_get_shape_4014_15->Target(__site_get_shape_4014_15, __pyx_v_mean, __pyx_context);
    __pyx_t_2 = __site_getindex_4014_21->Target(__site_getindex_4014_21, __pyx_t_4, ((System::Object^)0));
    __pyx_t_4 = nullptr;
    __pyx_t_4 = __site_get_shape_4014_31->Target(__site_get_shape_4014_31, __pyx_v_cov, __pyx_context);
    __pyx_t_1 = __site_getindex_4014_37->Target(__site_getindex_4014_37, __pyx_t_4, ((System::Object^)0));
    __pyx_t_4 = nullptr;
    __pyx_t_4 = __site_op_ne_4014_25->Target(__site_op_ne_4014_25, __pyx_t_2, __pyx_t_1);
    __pyx_t_2 = nullptr;
    __pyx_t_1 = nullptr;
    __pyx_t_6 = __site_istrue_4014_25->Target(__site_istrue_4014_25, __pyx_t_4);
    __pyx_t_4 = nullptr;
    if (__pyx_t_6) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4015
 *                raise ValueError("cov must be 2 dimensional and square")
 *         if mean.shape[0] != cov.shape[0]:
 *                raise ValueError("mean and cov must have same length")             # <<<<<<<<<<<<<<
 *         # Compute shape of output
 *         if isinstance(shape, int):
 */
      __pyx_t_4 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_1 = __site_call1_4015_31->Target(__site_call1_4015_31, __pyx_context, __pyx_t_4, ((System::Object^)"mean and cov must have same length"));
      __pyx_t_4 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_1, nullptr, nullptr);
      __pyx_t_1 = nullptr;
      goto __pyx_L8;
    }
    __pyx_L8:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4017
 *                raise ValueError("mean and cov must have same length")
 *         # Compute shape of output
 *         if isinstance(shape, int):             # <<<<<<<<<<<<<<
 *             shape = [shape]
 *         final_shape = list(shape[:])
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "isinstance");
    __pyx_t_4 = PythonOps::GetGlobal(__pyx_context, "int");
    __pyx_t_2 = __site_call2_4017_21->Target(__site_call2_4017_21, __pyx_context, __pyx_t_1, __pyx_v_shape, ((System::Object^)__pyx_t_4));
    __pyx_t_1 = nullptr;
    __pyx_t_4 = nullptr;
    __pyx_t_6 = __site_istrue_4017_21->Target(__site_istrue_4017_21, __pyx_t_2);
    __pyx_t_2 = nullptr;
    if (__pyx_t_6) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4018
 *         # Compute shape of output
 *         if isinstance(shape, int):
 *             shape = [shape]             # <<<<<<<<<<<<<<
 *         final_shape = list(shape[:])
 *         final_shape.append(mean.shape[0])
 */
      __pyx_t_2 = PythonOps::MakeListNoCopy(gcnew array<System::Object^>{__pyx_v_shape});
      __pyx_v_shape = ((System::Object^)__pyx_t_2);
      __pyx_t_2 = nullptr;
      goto __pyx_L9;
    }
    __pyx_L9:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4019
 *         if isinstance(shape, int):
 *             shape = [shape]
 *         final_shape = list(shape[:])             # <<<<<<<<<<<<<<
 *         final_shape.append(mean.shape[0])
 *         # Create a matrix of independent standard normally distributed random
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "list");
    __pyx_t_4 = __site_getslice_4019_32->Target(__site_getslice_4019_32, __pyx_v_shape, 0, PY_SSIZE_T_MAX);
    __pyx_t_1 = __site_call1_4019_26->Target(__site_call1_4019_26, __pyx_context, ((System::Object^)__pyx_t_2), __pyx_t_4);
    __pyx_t_2 = nullptr;
    __pyx_t_4 = nullptr;
    __pyx_v_final_shape = ((System::Object^)__pyx_t_1);
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4020
 *             shape = [shape]
 *         final_shape = list(shape[:])
 *         final_shape.append(mean.shape[0])             # <<<<<<<<<<<<<<
 *         # Create a matrix of independent standard normally distributed random
 *         # numbers. The matrix has rows with the same length as mean and as
 */
    __pyx_t_1 = __site_get_append_4020_19->Target(__site_get_append_4020_19, ((System::Object^)__pyx_v_final_shape), __pyx_context);
    __pyx_t_4 = __site_get_shape_4020_31->Target(__site_get_shape_4020_31, __pyx_v_mean, __pyx_context);
    __pyx_t_2 = __site_getindex_4020_37->Target(__site_getindex_4020_37, __pyx_t_4, ((System::Object^)0));
    __pyx_t_4 = nullptr;
    __pyx_t_4 = __site_call1_4020_26->Target(__site_call1_4020_26, __pyx_context, __pyx_t_1, __pyx_t_2);
    __pyx_t_1 = nullptr;
    __pyx_t_2 = nullptr;
    __pyx_t_4 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4024
 *         # numbers. The matrix has rows with the same length as mean and as
 *         # many rows are necessary to form a matrix of shape final_shape.
 *         x = self.standard_normal(np.multiply.reduce(final_shape))             # <<<<<<<<<<<<<<
 *         x.shape = (np.multiply.reduce(final_shape[0:len(final_shape) - 1]),
 *                    mean.shape[0])
 */
    __pyx_t_4 = __site_get_standard_normal_4024_16->Target(__site_get_standard_normal_4024_16, __pyx_v_self, __pyx_context);
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_1 = __site_get_multiply_4024_35->Target(__site_get_multiply_4024_35, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = __site_get_reduce_4024_44->Target(__site_get_reduce_4024_44, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call1_4024_51->Target(__site_call1_4024_51, __pyx_context, __pyx_t_2, ((System::Object^)__pyx_v_final_shape));
    __pyx_t_2 = nullptr;
    __pyx_t_2 = __site_call1_4024_32->Target(__site_call1_4024_32, __pyx_context, __pyx_t_4, __pyx_t_1);
    __pyx_t_4 = nullptr;
    __pyx_t_1 = nullptr;
    __pyx_v_x = __pyx_t_2;
    __pyx_t_2 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4025
 *         # many rows are necessary to form a matrix of shape final_shape.
 *         x = self.standard_normal(np.multiply.reduce(final_shape))
 *         x.shape = (np.multiply.reduce(final_shape[0:len(final_shape) - 1]),             # <<<<<<<<<<<<<<
 *                    mean.shape[0])
 *         # Transform matrix of standard normals into matrix where each row
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_1 = __site_get_multiply_4025_21->Target(__site_get_multiply_4025_21, __pyx_t_2, __pyx_context);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = __site_get_reduce_4025_30->Target(__site_get_reduce_4025_30, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "len");
    __pyx_t_4 = __site_call1_4025_55->Target(__site_call1_4025_55, __pyx_context, __pyx_t_1, ((System::Object^)__pyx_v_final_shape));
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_op_sub_4025_69->Target(__site_op_sub_4025_69, __pyx_t_4, __pyx_int_1);
    __pyx_t_4 = nullptr;
    __pyx_t_7 = __site_cvt_Py_ssize_t_4025_69->Target(__site_cvt_Py_ssize_t_4025_69, __pyx_t_1);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_getslice_4025_49->Target(__site_getslice_4025_49, ((System::Object^)__pyx_v_final_shape), 0, __pyx_t_7);
    __pyx_t_4 = __site_call1_4025_37->Target(__site_call1_4025_37, __pyx_context, __pyx_t_2, ((System::Object^)__pyx_t_1));
    __pyx_t_2 = nullptr;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4026
 *         x = self.standard_normal(np.multiply.reduce(final_shape))
 *         x.shape = (np.multiply.reduce(final_shape[0:len(final_shape) - 1]),
 *                    mean.shape[0])             # <<<<<<<<<<<<<<
 *         # Transform matrix of standard normals into matrix where each row
 *         # contains multivariate normals with the desired covariance.
 */
    __pyx_t_1 = __site_get_shape_4026_23->Target(__site_get_shape_4026_23, __pyx_v_mean, __pyx_context);
    __pyx_t_2 = __site_getindex_4026_29->Target(__site_getindex_4026_29, __pyx_t_1, ((System::Object^)0));
    __pyx_t_1 = nullptr;
    __pyx_t_1 = PythonOps::MakeTuple(gcnew array<System::Object^>{__pyx_t_4, __pyx_t_2});
    __pyx_t_4 = nullptr;
    __pyx_t_2 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4025
 *         # many rows are necessary to form a matrix of shape final_shape.
 *         x = self.standard_normal(np.multiply.reduce(final_shape))
 *         x.shape = (np.multiply.reduce(final_shape[0:len(final_shape) - 1]),             # <<<<<<<<<<<<<<
 *                    mean.shape[0])
 *         # Transform matrix of standard normals into matrix where each row
 */
    __site_set_shape_4025_9->Target(__site_set_shape_4025_9, __pyx_v_x, __pyx_t_1);
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4034
 *         # decomposition of cov is such an A.
 * 
 *         from numpy.dual import svd             # <<<<<<<<<<<<<<
 *         # XXX: we really should be doing this by Cholesky decomposition
 *         u, s, v = svd(cov)
 */
    __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportWithNames(__pyx_context, "numpy.dual", gcnew array<System::String^>{"svd"}, -1));
    __pyx_t_2 = PythonOps::ImportFrom(__pyx_context, __pyx_t_1, "svd");
    __pyx_v_svd = __pyx_t_2;
    __pyx_t_2 = nullptr;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4036
 *         from numpy.dual import svd
 *         # XXX: we really should be doing this by Cholesky decomposition
 *         u, s, v = svd(cov)             # <<<<<<<<<<<<<<
 *         x = np.dot(x * np.sqrt(s), v)
 *         # The rows of x now have the correct covariance but mean 0. Add
 */
    __pyx_t_1 = __site_call1_4036_21->Target(__site_call1_4036_21, __pyx_context, __pyx_v_svd, __pyx_v_cov);
    __pyx_t_8 = safe_cast< array<System::Object^>^ >(LightExceptions::CheckAndThrow(PythonOps::GetEnumeratorValuesNoComplexSets(__pyx_context, __pyx_t_1, 3)));
    __pyx_t_2 = __pyx_t_8[0];
    __pyx_t_4 = __pyx_t_8[1];
    __pyx_t_9 = __pyx_t_8[2];
    __pyx_t_1 = nullptr;
    __pyx_t_8 = nullptr;
    __pyx_v_u = __pyx_t_2;
    __pyx_t_2 = nullptr;
    __pyx_v_s = __pyx_t_4;
    __pyx_t_4 = nullptr;
    __pyx_v_v = __pyx_t_9;
    __pyx_t_9 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4037
 *         # XXX: we really should be doing this by Cholesky decomposition
 *         u, s, v = svd(cov)
 *         x = np.dot(x * np.sqrt(s), v)             # <<<<<<<<<<<<<<
 *         # The rows of x now have the correct covariance but mean 0. Add
 *         # mean to each row. Then each row will have mean mean.
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_9 = __site_get_dot_4037_14->Target(__site_get_dot_4037_14, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_4 = __site_get_sqrt_4037_25->Target(__site_get_sqrt_4037_25, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call1_4037_30->Target(__site_call1_4037_30, __pyx_context, __pyx_t_4, __pyx_v_s);
    __pyx_t_4 = nullptr;
    __pyx_t_4 = __site_op_mul_4037_21->Target(__site_op_mul_4037_21, __pyx_v_x, __pyx_t_1);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call2_4037_18->Target(__site_call2_4037_18, __pyx_context, __pyx_t_9, __pyx_t_4, __pyx_v_v);
    __pyx_t_9 = nullptr;
    __pyx_t_4 = nullptr;
    __pyx_v_x = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4040
 *         # The rows of x now have the correct covariance but mean 0. Add
 *         # mean to each row. Then each row will have mean mean.
 *         np.add(mean, x, x)             # <<<<<<<<<<<<<<
 *         x.shape = tuple(final_shape)
 *         return x
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_4 = __site_get_add_4040_10->Target(__site_get_add_4040_10, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call3_4040_14->Target(__site_call3_4040_14, __pyx_context, __pyx_t_4, __pyx_v_mean, __pyx_v_x, __pyx_v_x);
    __pyx_t_4 = nullptr;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4041
 *         # mean to each row. Then each row will have mean mean.
 *         np.add(mean, x, x)
 *         x.shape = tuple(final_shape)             # <<<<<<<<<<<<<<
 *         return x
 * 
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "tuple");
    __pyx_t_4 = __site_call1_4041_23->Target(__site_call1_4041_23, __pyx_context, ((System::Object^)__pyx_t_1), ((System::Object^)__pyx_v_final_shape));
    __pyx_t_1 = nullptr;
    __site_set_shape_4041_9->Target(__site_set_shape_4041_9, __pyx_v_x, __pyx_t_4);
    __pyx_t_4 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4042
 *         np.add(mean, x, x)
 *         x.shape = tuple(final_shape)
 *         return x             # <<<<<<<<<<<<<<
 * 
 *     def multinomial(self, long n, object pvals, size=None):
 */
    __pyx_r = __pyx_v_x;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4044
 *         return x
 * 
 *     def multinomial(self, long n, object pvals, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         multinomial(n, pvals, size=None)
 */

  virtual System::Object^ multinomial(System::Object^ n, System::Object^ pvals, [InteropServices::Optional]System::Object^ size) {
    long __pyx_v_n;
    System::Object^ __pyx_v_pvals = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    long __pyx_v_d;
    double *__pyx_v_pix;
    long *__pyx_v_mnix;
    long __pyx_v_i;
    long __pyx_v_j;
    long __pyx_v_dn;
    double __pyx_v_Sum;
    System::Object^ __pyx_v_parr;
    System::Object^ __pyx_v_shape;
    System::Object^ __pyx_v_multin;
    System::Object^ __pyx_r = nullptr;
    System::Object^ __pyx_t_1 = nullptr;
    System::Object^ __pyx_t_2 = nullptr;
    long __pyx_t_3;
    System::Object^ __pyx_t_4 = nullptr;
    int __pyx_t_5;
    long __pyx_t_6;
    double __pyx_t_7;
    System::Object^ __pyx_v_self = this;
    __pyx_v_n = __site_cvt_4044_4->Target(__site_cvt_4044_4, n);
    __pyx_v_pvals = pvals;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }
    __pyx_v_parr = nullptr;
    __pyx_v_shape = nullptr;
    __pyx_v_multin = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4102
 *         cdef double Sum
 * 
 *         d = len(pvals)             # <<<<<<<<<<<<<<
 *         parr = flat_array(pvals, np.double)
 *         pix = <double *>dataptr(parr)
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "len");
    __pyx_t_2 = __site_call1_4102_15->Target(__site_call1_4102_15, __pyx_context, __pyx_t_1, __pyx_v_pvals);
    __pyx_t_1 = nullptr;
    __pyx_t_3 = __site_cvt_long_4102_15->Target(__site_cvt_long_4102_15, __pyx_t_2);
    __pyx_t_2 = nullptr;
    __pyx_v_d = __pyx_t_3;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4103
 * 
 *         d = len(pvals)
 *         parr = flat_array(pvals, np.double)             # <<<<<<<<<<<<<<
 *         pix = <double *>dataptr(parr)
 * 
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "flat_array");
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_4 = __site_get_double_4103_35->Target(__site_get_double_4103_35, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call2_4103_25->Target(__site_call2_4103_25, __pyx_context, __pyx_t_2, __pyx_v_pvals, __pyx_t_4);
    __pyx_t_2 = nullptr;
    __pyx_t_4 = nullptr;
    __pyx_v_parr = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4104
 *         d = len(pvals)
 *         parr = flat_array(pvals, np.double)
 *         pix = <double *>dataptr(parr)             # <<<<<<<<<<<<<<
 * 
 *         if kahan_sum(pix, d-1) > (1.0 + 1e-12):
 */
    __pyx_v_pix = ((double *)dataptr(__pyx_v_parr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4106
 *         pix = <double *>dataptr(parr)
 * 
 *         if kahan_sum(pix, d-1) > (1.0 + 1e-12):             # <<<<<<<<<<<<<<
 *             raise ValueError("sum(pvals[:-1]) > 1.0")
 * 
 */
    __pyx_t_5 = (kahan_sum(__pyx_v_pix, (__pyx_v_d - 1)) > (1.0 + 1e-12));
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4107
 * 
 *         if kahan_sum(pix, d-1) > (1.0 + 1e-12):
 *             raise ValueError("sum(pvals[:-1]) > 1.0")             # <<<<<<<<<<<<<<
 * 
 *         if size is None:
 */
      __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "ValueError");
      __pyx_t_4 = __site_call1_4107_28->Target(__site_call1_4107_28, __pyx_context, __pyx_t_1, ((System::Object^)"sum(pvals[:-1]) > 1.0"));
      __pyx_t_1 = nullptr;
      throw PythonOps::MakeException(__pyx_context, __pyx_t_4, nullptr, nullptr);
      __pyx_t_4 = nullptr;
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4109
 *             raise ValueError("sum(pvals[:-1]) > 1.0")
 * 
 *         if size is None:             # <<<<<<<<<<<<<<
 *             shape = (d,)
 *         elif type(size) is int:
 */
    __pyx_t_5 = (__pyx_v_size == nullptr);
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4110
 * 
 *         if size is None:
 *             shape = (d,)             # <<<<<<<<<<<<<<
 *         elif type(size) is int:
 *             shape = (size, d)
 */
      __pyx_t_4 = __pyx_v_d;
      __pyx_t_1 = PythonOps::MakeTuple(gcnew array<System::Object^>{__pyx_t_4});
      __pyx_t_4 = nullptr;
      __pyx_v_shape = __pyx_t_1;
      __pyx_t_1 = nullptr;
      goto __pyx_L6;
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4111
 *         if size is None:
 *             shape = (d,)
 *         elif type(size) is int:             # <<<<<<<<<<<<<<
 *             shape = (size, d)
 *         else:
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "type");
    __pyx_t_4 = __site_call1_4111_17->Target(__site_call1_4111_17, __pyx_context, ((System::Object^)__pyx_t_1), __pyx_v_size);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "int");
    __pyx_t_5 = (((System::Object^)__pyx_t_4) == __pyx_t_1);
    __pyx_t_4 = nullptr;
    __pyx_t_1 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4112
 *             shape = (d,)
 *         elif type(size) is int:
 *             shape = (size, d)             # <<<<<<<<<<<<<<
 *         else:
 *             shape = size + (d,)
 */
      __pyx_t_1 = __pyx_v_d;
      __pyx_t_4 = PythonOps::MakeTuple(gcnew array<System::Object^>{__pyx_v_size, __pyx_t_1});
      __pyx_t_1 = nullptr;
      __pyx_v_shape = __pyx_t_4;
      __pyx_t_4 = nullptr;
      goto __pyx_L6;
    }
    /*else*/ {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4114
 *             shape = (size, d)
 *         else:
 *             shape = size + (d,)             # <<<<<<<<<<<<<<
 * 
 *         multin = np.zeros(shape, int)
 */
      __pyx_t_4 = __pyx_v_d;
      __pyx_t_1 = PythonOps::MakeTuple(gcnew array<System::Object^>{__pyx_t_4});
      __pyx_t_4 = nullptr;
      __pyx_t_4 = __site_op_add_4114_25->Target(__site_op_add_4114_25, __pyx_v_size, __pyx_t_1);
      __pyx_t_1 = nullptr;
      __pyx_v_shape = __pyx_t_4;
      __pyx_t_4 = nullptr;
    }
    __pyx_L6:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4116
 *             shape = size + (d,)
 * 
 *         multin = np.zeros(shape, int)             # <<<<<<<<<<<<<<
 *         mnix = <long *>dataptr(multin)
 *         i = 0
 */
    __pyx_t_4 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_1 = __site_get_zeros_4116_19->Target(__site_get_zeros_4116_19, __pyx_t_4, __pyx_context);
    __pyx_t_4 = nullptr;
    __pyx_t_4 = PythonOps::GetGlobal(__pyx_context, "int");
    __pyx_t_2 = __site_call2_4116_25->Target(__site_call2_4116_25, __pyx_context, __pyx_t_1, __pyx_v_shape, ((System::Object^)__pyx_t_4));
    __pyx_t_1 = nullptr;
    __pyx_t_4 = nullptr;
    __pyx_v_multin = __pyx_t_2;
    __pyx_t_2 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4117
 * 
 *         multin = np.zeros(shape, int)
 *         mnix = <long *>dataptr(multin)             # <<<<<<<<<<<<<<
 *         i = 0
 *         while i < multin.size:
 */
    __pyx_v_mnix = ((long *)dataptr(__pyx_v_multin));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4118
 *         multin = np.zeros(shape, int)
 *         mnix = <long *>dataptr(multin)
 *         i = 0             # <<<<<<<<<<<<<<
 *         while i < multin.size:
 *             Sum = 1.0
 */
    __pyx_v_i = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4119
 *         mnix = <long *>dataptr(multin)
 *         i = 0
 *         while i < multin.size:             # <<<<<<<<<<<<<<
 *             Sum = 1.0
 *             dn = n
 */
    while (1) {
      __pyx_t_2 = __pyx_v_i;
      __pyx_t_4 = __site_get_size_4119_24->Target(__site_get_size_4119_24, __pyx_v_multin, __pyx_context);
      __pyx_t_1 = __site_op_lt_4119_16->Target(__site_op_lt_4119_16, __pyx_t_2, __pyx_t_4);
      __pyx_t_2 = nullptr;
      __pyx_t_4 = nullptr;
      __pyx_t_5 = __site_istrue_4119_16->Target(__site_istrue_4119_16, __pyx_t_1);
      __pyx_t_1 = nullptr;
      if (!__pyx_t_5) break;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4120
 *         i = 0
 *         while i < multin.size:
 *             Sum = 1.0             # <<<<<<<<<<<<<<
 *             dn = n
 *             for j from 0 <= j < d-1:
 */
      __pyx_v_Sum = 1.0;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4121
 *         while i < multin.size:
 *             Sum = 1.0
 *             dn = n             # <<<<<<<<<<<<<<
 *             for j from 0 <= j < d-1:
 *                 mnix[i+j] = rk_binomial(self.internal_state, dn, pix[j]/Sum)
 */
      __pyx_v_dn = __pyx_v_n;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4122
 *             Sum = 1.0
 *             dn = n
 *             for j from 0 <= j < d-1:             # <<<<<<<<<<<<<<
 *                 mnix[i+j] = rk_binomial(self.internal_state, dn, pix[j]/Sum)
 *                 dn = dn - mnix[i+j]
 */
      __pyx_t_6 = (__pyx_v_d - 1);
      for (__pyx_v_j = 0; __pyx_v_j < __pyx_t_6; __pyx_v_j++) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4123
 *             dn = n
 *             for j from 0 <= j < d-1:
 *                 mnix[i+j] = rk_binomial(self.internal_state, dn, pix[j]/Sum)             # <<<<<<<<<<<<<<
 *                 dn = dn - mnix[i+j]
 *                 if dn <= 0:
 */
        __pyx_t_7 = (__pyx_v_pix[__pyx_v_j]);
        if (unlikely(__pyx_v_Sum == 0)) {
          throw PythonOps::ZeroDivisionError("float division");
        }
        (__pyx_v_mnix[(__pyx_v_i + __pyx_v_j)]) = rk_binomial(((RandomState^)__pyx_v_self)->internal_state, __pyx_v_dn, (__pyx_t_7 / __pyx_v_Sum));

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4124
 *             for j from 0 <= j < d-1:
 *                 mnix[i+j] = rk_binomial(self.internal_state, dn, pix[j]/Sum)
 *                 dn = dn - mnix[i+j]             # <<<<<<<<<<<<<<
 *                 if dn <= 0:
 *                     break
 */
        __pyx_v_dn = (__pyx_v_dn - (__pyx_v_mnix[(__pyx_v_i + __pyx_v_j)]));

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4125
 *                 mnix[i+j] = rk_binomial(self.internal_state, dn, pix[j]/Sum)
 *                 dn = dn - mnix[i+j]
 *                 if dn <= 0:             # <<<<<<<<<<<<<<
 *                     break
 *                 Sum = Sum - pix[j]
 */
        __pyx_t_5 = (__pyx_v_dn <= 0);
        if (__pyx_t_5) {

          /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4126
 *                 dn = dn - mnix[i+j]
 *                 if dn <= 0:
 *                     break             # <<<<<<<<<<<<<<
 *                 Sum = Sum - pix[j]
 *             if dn > 0:
 */
          goto __pyx_L10_break;
          goto __pyx_L11;
        }
        __pyx_L11:;

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4127
 *                 if dn <= 0:
 *                     break
 *                 Sum = Sum - pix[j]             # <<<<<<<<<<<<<<
 *             if dn > 0:
 *                 mnix[i+d-1] = dn
 */
        __pyx_v_Sum = (__pyx_v_Sum - (__pyx_v_pix[__pyx_v_j]));
      }
      __pyx_L10_break:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4128
 *                     break
 *                 Sum = Sum - pix[j]
 *             if dn > 0:             # <<<<<<<<<<<<<<
 *                 mnix[i+d-1] = dn
 * 
 */
      __pyx_t_5 = (__pyx_v_dn > 0);
      if (__pyx_t_5) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4129
 *                 Sum = Sum - pix[j]
 *             if dn > 0:
 *                 mnix[i+d-1] = dn             # <<<<<<<<<<<<<<
 * 
 *             i = i + d
 */
        (__pyx_v_mnix[((__pyx_v_i + __pyx_v_d) - 1)]) = __pyx_v_dn;
        goto __pyx_L12;
      }
      __pyx_L12:;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4131
 *                 mnix[i+d-1] = dn
 * 
 *             i = i + d             # <<<<<<<<<<<<<<
 * 
 *         return multin
 */
      __pyx_v_i = (__pyx_v_i + __pyx_v_d);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4133
 *             i = i + d
 * 
 *         return multin             # <<<<<<<<<<<<<<
 * 
 *     def dirichlet(self, object alpha, size=None):
 */
    __pyx_r = __pyx_v_multin;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4135
 *         return multin
 * 
 *     def dirichlet(self, object alpha, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         dirichlet(alpha, size=None)
 */

  virtual System::Object^ dirichlet(System::Object^ alpha, [InteropServices::Optional]System::Object^ size) {
    System::Object^ __pyx_v_alpha = nullptr;
    System::Object^ __pyx_v_size = nullptr;
    long __pyx_v_k;
    long __pyx_v_i;
    long __pyx_v_j;
    long __pyx_v_totsize;
    double *__pyx_v_alpha_data;
    double *__pyx_v_val_data;
    double __pyx_v_acc;
    double __pyx_v_invacc;
    System::Object^ __pyx_v_alpha_arr;
    System::Object^ __pyx_v_shape;
    System::Object^ __pyx_v_diric;
    System::Object^ __pyx_r = nullptr;
    System::Object^ __pyx_t_1 = nullptr;
    System::Object^ __pyx_t_2 = nullptr;
    long __pyx_t_3;
    System::Object^ __pyx_t_4 = nullptr;
    int __pyx_t_5;
    long __pyx_t_6;
    long __pyx_t_7;
    System::Object^ __pyx_v_self = this;
    __pyx_v_alpha = alpha;
    if (dynamic_cast<System::Reflection::Missing^>(size) == nullptr) {
      __pyx_v_size = size;
    } else {
      __pyx_v_size = ((System::Object^)nullptr);
    }
    __pyx_v_alpha_arr = nullptr;
    __pyx_v_shape = nullptr;
    __pyx_v_diric = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4196
 *         cdef double acc, invacc
 * 
 *         k = len(alpha)             # <<<<<<<<<<<<<<
 *         alpha_arr = flat_array(alpha, np.double, 1, 1)
 *         alpha_data = <double *>dataptr(alpha_arr)
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "len");
    __pyx_t_2 = __site_call1_4196_15->Target(__site_call1_4196_15, __pyx_context, __pyx_t_1, __pyx_v_alpha);
    __pyx_t_1 = nullptr;
    __pyx_t_3 = __site_cvt_long_4196_15->Target(__site_cvt_long_4196_15, __pyx_t_2);
    __pyx_t_2 = nullptr;
    __pyx_v_k = __pyx_t_3;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4197
 * 
 *         k = len(alpha)
 *         alpha_arr = flat_array(alpha, np.double, 1, 1)             # <<<<<<<<<<<<<<
 *         alpha_data = <double *>dataptr(alpha_arr)
 * 
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "flat_array");
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_4 = __site_get_double_4197_40->Target(__site_get_double_4197_40, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call4_4197_30->Target(__site_call4_4197_30, __pyx_context, __pyx_t_2, __pyx_v_alpha, __pyx_t_4, __pyx_int_1, __pyx_int_1);
    __pyx_t_2 = nullptr;
    __pyx_t_4 = nullptr;
    __pyx_v_alpha_arr = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4198
 *         k = len(alpha)
 *         alpha_arr = flat_array(alpha, np.double, 1, 1)
 *         alpha_data = <double *>dataptr(alpha_arr)             # <<<<<<<<<<<<<<
 * 
 *         if size is None:
 */
    __pyx_v_alpha_data = ((double *)dataptr(__pyx_v_alpha_arr));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4200
 *         alpha_data = <double *>dataptr(alpha_arr)
 * 
 *         if size is None:             # <<<<<<<<<<<<<<
 *             shape = (k,)
 *         elif type(size) is int:
 */
    __pyx_t_5 = (__pyx_v_size == nullptr);
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4201
 * 
 *         if size is None:
 *             shape = (k,)             # <<<<<<<<<<<<<<
 *         elif type(size) is int:
 *             shape = (size, k)
 */
      __pyx_t_1 = __pyx_v_k;
      __pyx_t_4 = PythonOps::MakeTuple(gcnew array<System::Object^>{__pyx_t_1});
      __pyx_t_1 = nullptr;
      __pyx_v_shape = __pyx_t_4;
      __pyx_t_4 = nullptr;
      goto __pyx_L5;
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4202
 *         if size is None:
 *             shape = (k,)
 *         elif type(size) is int:             # <<<<<<<<<<<<<<
 *             shape = (size, k)
 *         else:
 */
    __pyx_t_4 = PythonOps::GetGlobal(__pyx_context, "type");
    __pyx_t_1 = __site_call1_4202_17->Target(__site_call1_4202_17, __pyx_context, ((System::Object^)__pyx_t_4), __pyx_v_size);
    __pyx_t_4 = nullptr;
    __pyx_t_4 = PythonOps::GetGlobal(__pyx_context, "int");
    __pyx_t_5 = (((System::Object^)__pyx_t_1) == __pyx_t_4);
    __pyx_t_1 = nullptr;
    __pyx_t_4 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4203
 *             shape = (k,)
 *         elif type(size) is int:
 *             shape = (size, k)             # <<<<<<<<<<<<<<
 *         else:
 *             shape = size + (k,)
 */
      __pyx_t_4 = __pyx_v_k;
      __pyx_t_1 = PythonOps::MakeTuple(gcnew array<System::Object^>{__pyx_v_size, __pyx_t_4});
      __pyx_t_4 = nullptr;
      __pyx_v_shape = __pyx_t_1;
      __pyx_t_1 = nullptr;
      goto __pyx_L5;
    }
    /*else*/ {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4205
 *             shape = (size, k)
 *         else:
 *             shape = size + (k,)             # <<<<<<<<<<<<<<
 * 
 *         diric = np.zeros(shape, np.double)
 */
      __pyx_t_1 = __pyx_v_k;
      __pyx_t_4 = PythonOps::MakeTuple(gcnew array<System::Object^>{__pyx_t_1});
      __pyx_t_1 = nullptr;
      __pyx_t_1 = __site_op_add_4205_25->Target(__site_op_add_4205_25, __pyx_v_size, __pyx_t_4);
      __pyx_t_4 = nullptr;
      __pyx_v_shape = __pyx_t_1;
      __pyx_t_1 = nullptr;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4207
 *             shape = size + (k,)
 * 
 *         diric = np.zeros(shape, np.double)             # <<<<<<<<<<<<<<
 *         val_data = <double *>dataptr(diric)
 * 
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_4 = __site_get_zeros_4207_18->Target(__site_get_zeros_4207_18, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_2 = __site_get_double_4207_34->Target(__site_get_double_4207_34, __pyx_t_1, __pyx_context);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_call2_4207_24->Target(__site_call2_4207_24, __pyx_context, __pyx_t_4, __pyx_v_shape, __pyx_t_2);
    __pyx_t_4 = nullptr;
    __pyx_t_2 = nullptr;
    __pyx_v_diric = __pyx_t_1;
    __pyx_t_1 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4208
 * 
 *         diric = np.zeros(shape, np.double)
 *         val_data = <double *>dataptr(diric)             # <<<<<<<<<<<<<<
 * 
 *         i = 0
 */
    __pyx_v_val_data = ((double *)dataptr(__pyx_v_diric));

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4210
 *         val_data = <double *>dataptr(diric)
 * 
 *         i = 0             # <<<<<<<<<<<<<<
 *         totsize = diric.size
 *         while i < totsize:
 */
    __pyx_v_i = 0;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4211
 * 
 *         i = 0
 *         totsize = diric.size             # <<<<<<<<<<<<<<
 *         while i < totsize:
 *             acc = 0.0
 */
    __pyx_t_1 = __site_get_size_4211_23->Target(__site_get_size_4211_23, __pyx_v_diric, __pyx_context);
    __pyx_t_6 = __site_cvt_long_4211_23->Target(__site_cvt_long_4211_23, __pyx_t_1);
    __pyx_t_1 = nullptr;
    __pyx_v_totsize = __pyx_t_6;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4212
 *         i = 0
 *         totsize = diric.size
 *         while i < totsize:             # <<<<<<<<<<<<<<
 *             acc = 0.0
 *             for j from 0 <= j < k:
 */
    while (1) {
      __pyx_t_5 = (__pyx_v_i < __pyx_v_totsize);
      if (!__pyx_t_5) break;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4213
 *         totsize = diric.size
 *         while i < totsize:
 *             acc = 0.0             # <<<<<<<<<<<<<<
 *             for j from 0 <= j < k:
 *                 val_data[i+j] = rk_standard_gamma(self.internal_state,
 */
      __pyx_v_acc = 0.0;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4214
 *         while i < totsize:
 *             acc = 0.0
 *             for j from 0 <= j < k:             # <<<<<<<<<<<<<<
 *                 val_data[i+j] = rk_standard_gamma(self.internal_state,
 *                                                   alpha_data[j])
 */
      __pyx_t_7 = __pyx_v_k;
      for (__pyx_v_j = 0; __pyx_v_j < __pyx_t_7; __pyx_v_j++) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4215
 *             acc = 0.0
 *             for j from 0 <= j < k:
 *                 val_data[i+j] = rk_standard_gamma(self.internal_state,             # <<<<<<<<<<<<<<
 *                                                   alpha_data[j])
 *                 acc = acc + val_data[i+j]
 */
        (__pyx_v_val_data[(__pyx_v_i + __pyx_v_j)]) = rk_standard_gamma(((RandomState^)__pyx_v_self)->internal_state, (__pyx_v_alpha_data[__pyx_v_j]));

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4217
 *                 val_data[i+j] = rk_standard_gamma(self.internal_state,
 *                                                   alpha_data[j])
 *                 acc = acc + val_data[i+j]             # <<<<<<<<<<<<<<
 *             invacc  = 1/acc
 *             for j from 0 <= j < k:
 */
        __pyx_v_acc = (__pyx_v_acc + (__pyx_v_val_data[(__pyx_v_i + __pyx_v_j)]));
      }

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4218
 *                                                   alpha_data[j])
 *                 acc = acc + val_data[i+j]
 *             invacc  = 1/acc             # <<<<<<<<<<<<<<
 *             for j from 0 <= j < k:
 *                 val_data[i+j]   = val_data[i+j] * invacc
 */
      if (unlikely(__pyx_v_acc == 0)) {
        throw PythonOps::ZeroDivisionError("float division");
      }
      __pyx_v_invacc = (1.0 / __pyx_v_acc);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4219
 *                 acc = acc + val_data[i+j]
 *             invacc  = 1/acc
 *             for j from 0 <= j < k:             # <<<<<<<<<<<<<<
 *                 val_data[i+j]   = val_data[i+j] * invacc
 *             i = i + k
 */
      __pyx_t_7 = __pyx_v_k;
      for (__pyx_v_j = 0; __pyx_v_j < __pyx_t_7; __pyx_v_j++) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4220
 *             invacc  = 1/acc
 *             for j from 0 <= j < k:
 *                 val_data[i+j]   = val_data[i+j] * invacc             # <<<<<<<<<<<<<<
 *             i = i + k
 * 
 */
        (__pyx_v_val_data[(__pyx_v_i + __pyx_v_j)]) = ((__pyx_v_val_data[(__pyx_v_i + __pyx_v_j)]) * __pyx_v_invacc);
      }

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4221
 *             for j from 0 <= j < k:
 *                 val_data[i+j]   = val_data[i+j] * invacc
 *             i = i + k             # <<<<<<<<<<<<<<
 * 
 *         return diric
 */
      __pyx_v_i = (__pyx_v_i + __pyx_v_k);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4223
 *             i = i + k
 * 
 *         return diric             # <<<<<<<<<<<<<<
 * 
 *     # Shuffling and permutations:
 */
    __pyx_r = __pyx_v_diric;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4226
 * 
 *     # Shuffling and permutations:
 *     def shuffle(self, object x):             # <<<<<<<<<<<<<<
 *         """
 *         shuffle(x)
 */

  virtual System::Object^ shuffle(System::Object^ x) {
    System::Object^ __pyx_v_x = nullptr;
    long __pyx_v_i;
    long __pyx_v_j;
    int __pyx_v_copy;
    System::Object^ __pyx_r = nullptr;
    System::Object^ __pyx_t_1 = nullptr;
    System::Object^ __pyx_t_2 = nullptr;
    long __pyx_t_3;
    System::Object^ __pyx_t_6 = nullptr;
    long __pyx_t_7;
    int __pyx_t_8;
    int __pyx_t_9;
    System::Object^ __pyx_v_self = this;
    __pyx_v_x = x;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4236
 *         cdef int copy
 * 
 *         i = len(x) - 1             # <<<<<<<<<<<<<<
 *         try:
 *             j = len(x[0])
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "len");
    __pyx_t_2 = __site_call1_4236_15->Target(__site_call1_4236_15, __pyx_context, __pyx_t_1, __pyx_v_x);
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_op_sub_4236_19->Target(__site_op_sub_4236_19, __pyx_t_2, __pyx_int_1);
    __pyx_t_2 = nullptr;
    __pyx_t_3 = __site_cvt_long_4236_19->Target(__site_cvt_long_4236_19, __pyx_t_1);
    __pyx_t_1 = nullptr;
    __pyx_v_i = __pyx_t_3;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4237
 * 
 *         i = len(x) - 1
 *         try:             # <<<<<<<<<<<<<<
 *             j = len(x[0])
 *         except:
 */
    try {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4238
 *         i = len(x) - 1
 *         try:
 *             j = len(x[0])             # <<<<<<<<<<<<<<
 *         except:
 *             j = 0
 */
      __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "len");
      __pyx_t_2 = __site_getindex_4238_21->Target(__site_getindex_4238_21, __pyx_v_x, ((System::Object^)0));
      __pyx_t_6 = __site_call1_4238_19->Target(__site_call1_4238_19, __pyx_context, __pyx_t_1, __pyx_t_2);
      __pyx_t_1 = nullptr;
      __pyx_t_2 = nullptr;
      __pyx_t_7 = __site_cvt_long_4238_19->Target(__site_cvt_long_4238_19, __pyx_t_6);
      __pyx_t_6 = nullptr;
      __pyx_v_j = __pyx_t_7;
    } catch (System::Exception^ __pyx_lt_4) {
      System::Object^ __pyx_lt_5 = PythonOps::SetCurrentException(__pyx_context, __pyx_lt_4);

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4239
 *         try:
 *             j = len(x[0])
 *         except:             # <<<<<<<<<<<<<<
 *             j = 0
 * 
 */
      /*except:*/ {
        // XXX should update traceback here __Pyx_AddTraceback("mtrand.RandomState.shuffle");
        PythonOps::BuildExceptionInfo(__pyx_context, __pyx_lt_4);

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4240
 *             j = len(x[0])
 *         except:
 *             j = 0             # <<<<<<<<<<<<<<
 * 
 *         if j == 0:
 */
        __pyx_v_j = 0;
      }
      PythonOps::ExceptionHandled(__pyx_context);
    }

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4242
 *             j = 0
 * 
 *         if j == 0:             # <<<<<<<<<<<<<<
 *             # adaptation of random.shuffle()
 *             while i > 0:
 */
    __pyx_t_8 = (__pyx_v_j == 0);
    if (__pyx_t_8) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4244
 *         if j == 0:
 *             # adaptation of random.shuffle()
 *             while i > 0:             # <<<<<<<<<<<<<<
 *                 j = rk_interval(i, self.internal_state)
 *                 x[i], x[j] = x[j], x[i]
 */
      while (1) {
        __pyx_t_8 = (__pyx_v_i > 0);
        if (!__pyx_t_8) break;

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4245
 *             # adaptation of random.shuffle()
 *             while i > 0:
 *                 j = rk_interval(i, self.internal_state)             # <<<<<<<<<<<<<<
 *                 x[i], x[j] = x[j], x[i]
 *                 i = i - 1
 */
        __pyx_v_j = rk_interval(__pyx_v_i, ((RandomState^)__pyx_v_self)->internal_state);

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4246
 *             while i > 0:
 *                 j = rk_interval(i, self.internal_state)
 *                 x[i], x[j] = x[j], x[i]             # <<<<<<<<<<<<<<
 *                 i = i - 1
 *         else:
 */
        __pyx_t_6 = __site_getindex_4246_30->Target(__site_getindex_4246_30, __pyx_v_x, ((System::Object^)__pyx_v_j));
        __pyx_t_2 = __site_getindex_4246_36->Target(__site_getindex_4246_36, __pyx_v_x, ((System::Object^)__pyx_v_i));
        __site_setindex_4246_17->Target(__site_setindex_4246_17, __pyx_v_x, ((System::Object^)__pyx_v_i), __pyx_t_6);
        __pyx_t_6 = nullptr;
        __site_setindex_4246_23->Target(__site_setindex_4246_23, __pyx_v_x, ((System::Object^)__pyx_v_j), __pyx_t_2);
        __pyx_t_2 = nullptr;

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4247
 *                 j = rk_interval(i, self.internal_state)
 *                 x[i], x[j] = x[j], x[i]
 *                 i = i - 1             # <<<<<<<<<<<<<<
 *         else:
 *             # make copies
 */
        __pyx_v_i = (__pyx_v_i - 1);
      }
      goto __pyx_L5;
    }
    /*else*/ {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4250
 *         else:
 *             # make copies
 *             copy = hasattr(x[0], 'copy')             # <<<<<<<<<<<<<<
 *             if copy:
 *                 while(i > 0):
 */
      __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "hasattr");
      __pyx_t_6 = __site_getindex_4250_28->Target(__site_getindex_4250_28, __pyx_v_x, ((System::Object^)0));
      __pyx_t_1 = __site_call2_4250_26->Target(__site_call2_4250_26, __pyx_context, __pyx_t_2, __pyx_t_6, ((System::Object^)"copy"));
      __pyx_t_2 = nullptr;
      __pyx_t_6 = nullptr;
      __pyx_t_9 = __site_cvt_int_4250_26->Target(__site_cvt_int_4250_26, __pyx_t_1);
      __pyx_t_1 = nullptr;
      __pyx_v_copy = __pyx_t_9;

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4251
 *             # make copies
 *             copy = hasattr(x[0], 'copy')
 *             if copy:             # <<<<<<<<<<<<<<
 *                 while(i > 0):
 *                     j = rk_interval(i, self.internal_state)
 */
      if (__pyx_v_copy) {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4252
 *             copy = hasattr(x[0], 'copy')
 *             if copy:
 *                 while(i > 0):             # <<<<<<<<<<<<<<
 *                     j = rk_interval(i, self.internal_state)
 *                     x[i], x[j] = x[j].copy(), x[i].copy()
 */
        while (1) {
          __pyx_t_8 = (__pyx_v_i > 0);
          if (!__pyx_t_8) break;

          /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4253
 *             if copy:
 *                 while(i > 0):
 *                     j = rk_interval(i, self.internal_state)             # <<<<<<<<<<<<<<
 *                     x[i], x[j] = x[j].copy(), x[i].copy()
 *                     i = i - 1
 */
          __pyx_v_j = rk_interval(__pyx_v_i, ((RandomState^)__pyx_v_self)->internal_state);

          /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4254
 *                 while(i > 0):
 *                     j = rk_interval(i, self.internal_state)
 *                     x[i], x[j] = x[j].copy(), x[i].copy()             # <<<<<<<<<<<<<<
 *                     i = i - 1
 *             else:
 */
          __pyx_t_1 = __site_getindex_4254_34->Target(__site_getindex_4254_34, __pyx_v_x, ((System::Object^)__pyx_v_j));
          __pyx_t_6 = __site_get_copy_4254_37->Target(__site_get_copy_4254_37, __pyx_t_1, __pyx_context);
          __pyx_t_1 = nullptr;
          __pyx_t_1 = __site_call0_4254_42->Target(__site_call0_4254_42, __pyx_context, __pyx_t_6);
          __pyx_t_6 = nullptr;
          __pyx_t_6 = __site_getindex_4254_47->Target(__site_getindex_4254_47, __pyx_v_x, ((System::Object^)__pyx_v_i));
          __pyx_t_2 = __site_get_copy_4254_50->Target(__site_get_copy_4254_50, __pyx_t_6, __pyx_context);
          __pyx_t_6 = nullptr;
          __pyx_t_6 = __site_call0_4254_55->Target(__site_call0_4254_55, __pyx_context, __pyx_t_2);
          __pyx_t_2 = nullptr;
          __site_setindex_4254_21->Target(__site_setindex_4254_21, __pyx_v_x, ((System::Object^)__pyx_v_i), __pyx_t_1);
          __pyx_t_1 = nullptr;
          __site_setindex_4254_27->Target(__site_setindex_4254_27, __pyx_v_x, ((System::Object^)__pyx_v_j), __pyx_t_6);
          __pyx_t_6 = nullptr;

          /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4255
 *                     j = rk_interval(i, self.internal_state)
 *                     x[i], x[j] = x[j].copy(), x[i].copy()
 *                     i = i - 1             # <<<<<<<<<<<<<<
 *             else:
 *                 while(i > 0):
 */
          __pyx_v_i = (__pyx_v_i - 1);
        }
        goto __pyx_L8;
      }
      /*else*/ {

        /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4257
 *                     i = i - 1
 *             else:
 *                 while(i > 0):             # <<<<<<<<<<<<<<
 *                     j = rk_interval(i, self.internal_state)
 *                     x[i], x[j] = x[j][:], x[i][:]
 */
        while (1) {
          __pyx_t_8 = (__pyx_v_i > 0);
          if (!__pyx_t_8) break;

          /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4258
 *             else:
 *                 while(i > 0):
 *                     j = rk_interval(i, self.internal_state)             # <<<<<<<<<<<<<<
 *                     x[i], x[j] = x[j][:], x[i][:]
 *                     i = i - 1
 */
          __pyx_v_j = rk_interval(__pyx_v_i, ((RandomState^)__pyx_v_self)->internal_state);

          /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4259
 *                 while(i > 0):
 *                     j = rk_interval(i, self.internal_state)
 *                     x[i], x[j] = x[j][:], x[i][:]             # <<<<<<<<<<<<<<
 *                     i = i - 1
 * 
 */
          __pyx_t_6 = __site_getindex_4259_34->Target(__site_getindex_4259_34, __pyx_v_x, ((System::Object^)__pyx_v_j));
          __pyx_t_1 = __site_getslice_4259_37->Target(__site_getslice_4259_37, __pyx_t_6, 0, PY_SSIZE_T_MAX);
          __pyx_t_6 = nullptr;
          __pyx_t_6 = __site_getindex_4259_43->Target(__site_getindex_4259_43, __pyx_v_x, ((System::Object^)__pyx_v_i));
          __pyx_t_2 = __site_getslice_4259_46->Target(__site_getslice_4259_46, __pyx_t_6, 0, PY_SSIZE_T_MAX);
          __pyx_t_6 = nullptr;
          __site_setindex_4259_21->Target(__site_setindex_4259_21, __pyx_v_x, ((System::Object^)__pyx_v_i), __pyx_t_1);
          __pyx_t_1 = nullptr;
          __site_setindex_4259_27->Target(__site_setindex_4259_27, __pyx_v_x, ((System::Object^)__pyx_v_j), __pyx_t_2);
          __pyx_t_2 = nullptr;

          /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4260
 *                     j = rk_interval(i, self.internal_state)
 *                     x[i], x[j] = x[j][:], x[i][:]
 *                     i = i - 1             # <<<<<<<<<<<<<<
 * 
 *     def permutation(self, object x):
 */
          __pyx_v_i = (__pyx_v_i - 1);
        }
      }
      __pyx_L8:;
    }
    __pyx_L5:;

    __pyx_r = nullptr;
    return __pyx_r;
  }

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4262
 *                     i = i - 1
 * 
 *     def permutation(self, object x):             # <<<<<<<<<<<<<<
 *         """
 *         permutation(x)
 */

  virtual System::Object^ permutation(System::Object^ x) {
    System::Object^ __pyx_v_x = nullptr;
    System::Object^ __pyx_v_arr;
    System::Object^ __pyx_r = nullptr;
    System::Object^ __pyx_t_1 = nullptr;
    System::Object^ __pyx_t_2 = nullptr;
    System::Object^ __pyx_t_3 = nullptr;
    System::Object^ __pyx_t_4 = nullptr;
    int __pyx_t_5;
    System::Object^ __pyx_v_self = this;
    __pyx_v_x = x;
    __pyx_v_arr = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4289
 * 
 *         """
 *         if isinstance(x, (int, np.integer)):             # <<<<<<<<<<<<<<
 *             arr = np.arange(x)
 *         else:
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "isinstance");
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "int");
    __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_4 = __site_get_integer_4289_33->Target(__site_get_integer_4289_33, __pyx_t_3, __pyx_context);
    __pyx_t_3 = nullptr;
    __pyx_t_3 = PythonOps::MakeTuple(gcnew array<System::Object^>{((System::Object^)__pyx_t_2), __pyx_t_4});
    __pyx_t_2 = nullptr;
    __pyx_t_4 = nullptr;
    __pyx_t_4 = __site_call2_4289_21->Target(__site_call2_4289_21, __pyx_context, __pyx_t_1, __pyx_v_x, __pyx_t_3);
    __pyx_t_1 = nullptr;
    __pyx_t_3 = nullptr;
    __pyx_t_5 = __site_istrue_4289_21->Target(__site_istrue_4289_21, __pyx_t_4);
    __pyx_t_4 = nullptr;
    if (__pyx_t_5) {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4290
 *         """
 *         if isinstance(x, (int, np.integer)):
 *             arr = np.arange(x)             # <<<<<<<<<<<<<<
 *         else:
 *             arr = np.array(x)
 */
      __pyx_t_4 = PythonOps::GetGlobal(__pyx_context, "np");
      __pyx_t_3 = __site_get_arange_4290_20->Target(__site_get_arange_4290_20, __pyx_t_4, __pyx_context);
      __pyx_t_4 = nullptr;
      __pyx_t_4 = __site_call1_4290_27->Target(__site_call1_4290_27, __pyx_context, __pyx_t_3, __pyx_v_x);
      __pyx_t_3 = nullptr;
      __pyx_v_arr = __pyx_t_4;
      __pyx_t_4 = nullptr;
      goto __pyx_L5;
    }
    /*else*/ {

      /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4292
 *             arr = np.arange(x)
 *         else:
 *             arr = np.array(x)             # <<<<<<<<<<<<<<
 *         self.shuffle(arr)
 *         return arr
 */
      __pyx_t_4 = PythonOps::GetGlobal(__pyx_context, "np");
      __pyx_t_3 = __site_get_array_4292_20->Target(__site_get_array_4292_20, __pyx_t_4, __pyx_context);
      __pyx_t_4 = nullptr;
      __pyx_t_4 = __site_call1_4292_26->Target(__site_call1_4292_26, __pyx_context, __pyx_t_3, __pyx_v_x);
      __pyx_t_3 = nullptr;
      __pyx_v_arr = __pyx_t_4;
      __pyx_t_4 = nullptr;
    }
    __pyx_L5:;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4293
 *         else:
 *             arr = np.array(x)
 *         self.shuffle(arr)             # <<<<<<<<<<<<<<
 *         return arr
 * 
 */
    __pyx_t_4 = __site_get_shuffle_4293_12->Target(__site_get_shuffle_4293_12, __pyx_v_self, __pyx_context);
    __pyx_t_3 = __site_call1_4293_20->Target(__site_call1_4293_20, __pyx_context, __pyx_t_4, __pyx_v_arr);
    __pyx_t_4 = nullptr;
    __pyx_t_3 = nullptr;

    /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4294
 *             arr = np.array(x)
 *         self.shuffle(arr)
 *         return arr             # <<<<<<<<<<<<<<
 * 
 * 
 */
    __pyx_r = __pyx_v_arr;
    goto __pyx_L0;

    __pyx_r = nullptr;
    __pyx_L0:;
    return __pyx_r;
  }
};
// XXX skipping all typeobj definitions
/* Cython code section 'pystring_table' */
/* Cython code section 'cached_builtins' */
/* Cython code section 'init_globals' */

static int __Pyx_InitGlobals(void) {
  __pyx_int_0 = 0;
  __pyx_int_1 = 1;
  __pyx_int_2 = 2;
  __pyx_int_3 = 3;
  __pyx_int_624 = 624;

  return 0;
}
/* Cython code section 'init_module' */
static void __Pyx_InitSites(CodeContext^ __pyx_context) {
  const int PythonOperationKind_Contains = 5;
  const int PythonOperationKind_GetEnumeratorForIteration = 18;
  const int PythonOperationKind_FloorDivide = 23;
  const int PythonOperationKind_TrueDivide = 25;
  const int PythonOperationKind_InPlaceFloorDivide = 0x20000000 | 23;
  const int PythonOperationKind_InPlaceTrueDivide = 0x20000000 | 25;
  __site_call0_4297_19 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(0)));
  __site_get_seed_4298_12 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "seed", false));
  __site_get_get_state_4299_17 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "get_state", false));
  __site_get_set_state_4300_17 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "set_state", false));
  __site_get_random_sample_4301_21 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "random_sample", false));
  __site_get_randint_4302_15 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "randint", false));
  __site_get_bytes_4303_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "bytes", false));
  __site_get_uniform_4304_15 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "uniform", false));
  __site_get_rand_4305_12 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "rand", false));
  __site_get_randn_4306_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "randn", false));
  __site_get_random_integers_4307_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "random_integers", false));
  __site_get_standard_normal_4308_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "standard_normal", false));
  __site_get_normal_4309_14 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "normal", false));
  __site_get_beta_4310_12 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "beta", false));
  __site_get_exponential_4311_19 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "exponential", false));
  __site_get_standard_exponential_4312_28 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "standard_exponential", false));
  __site_get_standard_gamma_4313_22 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "standard_gamma", false));
  __site_get_gamma_4314_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "gamma", false));
  __site_get_f_4315_9 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "f", false));
  __site_get_noncentral_f_4316_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "noncentral_f", false));
  __site_get_chisquare_4317_17 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "chisquare", false));
  __site_get_noncentral_chisquare_4318_28 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "noncentral_chisquare", false));
  __site_get_standard_cauchy_4319_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "standard_cauchy", false));
  __site_get_standard_t_4320_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "standard_t", false));
  __site_get_vonmises_4321_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "vonmises", false));
  __site_get_pareto_4322_14 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "pareto", false));
  __site_get_weibull_4323_15 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "weibull", false));
  __site_get_power_4324_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "power", false));
  __site_get_laplace_4325_15 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "laplace", false));
  __site_get_gumbel_4326_14 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "gumbel", false));
  __site_get_logistic_4327_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "logistic", false));
  __site_get_lognormal_4328_17 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "lognormal", false));
  __site_get_rayleigh_4329_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "rayleigh", false));
  __site_get_wald_4330_12 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "wald", false));
  __site_get_triangular_4331_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "triangular", false));
  __site_get_binomial_4333_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "binomial", false));
  __site_get_negative_binomial_4334_25 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "negative_binomial", false));
  __site_get_poisson_4335_15 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "poisson", false));
  __site_get_zipf_4336_12 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "zipf", false));
  __site_get_geometric_4337_17 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "geometric", false));
  __site_get_hypergeometric_4338_22 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "hypergeometric", false));
  __site_get_logseries_4339_17 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "logseries", false));
  __site_get_multivariate_normal_4341_27 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "multivariate_normal", false));
  __site_get_multinomial_4342_19 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "multinomial", false));
  __site_get_dirichlet_4343_17 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "dirichlet", false));
  __site_get_shuffle_4345_15 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "shuffle", false));
  __site_get_permutation_4346_19 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "permutation", false));
  __site_get_Array_88_30 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Array", false));
  __site_cvt_PY_LONG_LONG_88_30 = CallSite< System::Func< CallSite^, System::Object^, PY_LONG_LONG >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, PY_LONG_LONG::typeid, ConversionResultKind::ExplicitCast));
  __site_get_Iter_94_30 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Iter", false));
  __site_cvt_PY_LONG_LONG_94_30 = CallSite< System::Func< CallSite^, System::Object^, PY_LONG_LONG >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, PY_LONG_LONG::typeid, ConversionResultKind::ExplicitCast));
  __site_get_empty_141_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_get_double_141_31 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_141_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_142_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_142_20 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_empty_157_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_get_double_157_31 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_157_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_158_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_158_20 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_array_172_11 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_get_double_172_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_172_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_empty_like_174_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty_like", false));
  __site_call1_174_27 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_size_175_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_175_20 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_empty_182_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_get_double_182_31 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_182_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_broadcast_184_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "broadcast", false));
  __site_call2_184_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_185_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_get_size_185_28 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_op_ne_185_22 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::NotEqual));
  __site_istrue_185_22 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_186_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_size_187_33 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_187_33 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_empty_202_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_get_double_202_31 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_202_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_203_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_203_20 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_array_217_11 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_get_double_217_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_217_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_array_218_11 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_get_double_218_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_218_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_broadcast_220_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "broadcast", false));
  __site_call2_220_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_empty_221_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_get_shape_221_28 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "shape", false));
  __site_get_double_221_38 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_221_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_223_33 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_223_33 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_empty_229_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_get_double_229_31 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_229_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_broadcast_231_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "broadcast", false));
  __site_call3_231_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(3)));
  __site_get_size_232_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_get_size_232_28 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_op_ne_232_22 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::NotEqual));
  __site_istrue_232_22 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_233_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_size_234_33 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_234_33 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_empty_251_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_get_double_251_31 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_251_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_252_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_252_20 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_array_267_11 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_get_double_267_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_267_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_array_268_11 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_get_double_268_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_268_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_array_269_11 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_get_double_269_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_269_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_broadcast_271_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "broadcast", false));
  __site_call3_271_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(3)));
  __site_get_empty_272_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_get_shape_272_28 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "shape", false));
  __site_get_double_272_38 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_272_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_274_33 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_274_33 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_empty_281_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_get_double_281_31 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_281_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_broadcast_283_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "broadcast", false));
  __site_call4_283_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(4)));
  __site_get_size_284_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_get_size_284_28 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_op_ne_284_22 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::NotEqual));
  __site_istrue_284_22 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_285_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_size_286_33 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_286_33 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_empty_302_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_call2_302_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_303_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_303_20 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_empty_318_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_call2_318_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_319_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_319_20 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_array_333_11 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_get_long_333_29 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "long", false));
  __site_call1_dtype_333_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(gcnew array<Argument>{Argument::Simple, Argument("dtype")})));
  __site_get_array_334_11 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_get_double_334_29 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call1_dtype_334_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(gcnew array<Argument>{Argument::Simple, Argument("dtype")})));
  __site_get_broadcast_336_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "broadcast", false));
  __site_call2_336_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_empty_337_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_get_shape_337_28 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "shape", false));
  __site_get_long_337_38 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "long", false));
  __site_call2_337_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_339_33 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_339_33 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_empty_345_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_call2_345_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_broadcast_347_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "broadcast", false));
  __site_call3_347_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(3)));
  __site_get_size_348_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_get_size_348_28 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_op_ne_348_22 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::NotEqual));
  __site_istrue_348_22 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_349_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_size_350_33 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_350_33 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_empty_367_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_call2_367_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_368_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_368_20 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_array_382_11 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_get_double_382_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_382_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_array_383_11 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_get_double_383_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_383_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_broadcast_385_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "broadcast", false));
  __site_call2_385_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_empty_386_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_get_shape_386_28 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "shape", false));
  __site_get_long_386_38 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "long", false));
  __site_call2_386_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_388_33 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_388_33 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_empty_394_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_call2_394_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_broadcast_396_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "broadcast", false));
  __site_call3_396_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(3)));
  __site_get_size_397_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_get_size_397_28 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_op_ne_397_22 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::NotEqual));
  __site_istrue_397_22 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_398_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_size_399_33 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_399_33 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_empty_416_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_call2_416_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_417_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_417_20 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_array_432_11 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_get_long_432_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "long", false));
  __site_call2_432_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_array_433_11 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_get_long_433_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "long", false));
  __site_call2_433_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_array_434_11 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_get_long_434_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "long", false));
  __site_call2_434_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_broadcast_436_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "broadcast", false));
  __site_call3_436_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(3)));
  __site_get_empty_437_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_get_shape_437_28 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "shape", false));
  __site_get_long_437_38 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "long", false));
  __site_call2_437_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_439_33 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_439_33 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_empty_446_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_call2_446_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_broadcast_448_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "broadcast", false));
  __site_call4_448_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(4)));
  __site_get_size_449_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_get_size_449_28 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_op_ne_449_22 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::NotEqual));
  __site_istrue_449_22 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_450_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_size_451_33 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_451_33 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_empty_468_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_call2_468_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_469_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_469_20 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_array_483_11 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_get_double_483_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_483_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_empty_485_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_get_shape_485_27 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "shape", false));
  __site_get_long_485_43 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "long", false));
  __site_call1_dtype_485_24 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(gcnew array<Argument>{Argument::Simple, Argument("dtype")})));
  __site_get_size_486_22 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_486_22 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_empty_493_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_call2_493_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_broadcast_495_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "broadcast", false));
  __site_call2_495_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_496_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_get_size_496_28 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_op_ne_496_22 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::NotEqual));
  __site_istrue_496_22 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_497_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_size_498_33 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_498_33 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_array_520_10 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_call1_dtype_520_16 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(gcnew array<Argument>{Argument::Simple, Argument("dtype")})));
  __site_get_shape_521_12 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "shape", false));
  __site_call1_521_10 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_eq_521_20 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Equal));
  __site_istrue_521_20 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_get_flatten_524_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "flatten", false));
  __site_call0_524_24 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(0)));
  __site_get_seed_565_12 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "seed", false));
  __site_call1_565_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call1_595_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_unsigned_long_596_24 = CallSite< System::Func< CallSite^, System::Object^, unsigned long >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, unsigned long::typeid, ConversionResultKind::ExplicitCast));
  __site_get_integer_597_32 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "integer", false));
  __site_call2_597_23 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_istrue_597_23 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_598_23 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_unsigned_long_599_25 = CallSite< System::Func< CallSite^, System::Object^, unsigned long >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, unsigned long::typeid, ConversionResultKind::ExplicitCast));
  __site_get_long_601_37 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "long", false));
  __site_call2_601_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_empty_636_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_get_uint_636_32 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "uint", false));
  __site_call2_636_24 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_asarray_640_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "asarray", false));
  __site_get_uint32_640_36 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "uint32", false));
  __site_call2_640_26 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_getindex_693_30 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_op_ne_694_26 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::NotEqual));
  __site_istrue_694_26 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_695_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_getslice_696_24 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetSliceBinder(__pyx_context));
  __site_cvt_int_696_8 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_697_14 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_eq_697_22 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Equal));
  __site_istrue_697_22 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_getslice_701_46 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetSliceBinder(__pyx_context));
  __site_get_uint_703_32 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "uint", false));
  __site_call2_703_24 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_shape_704_14 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "shape", false));
  __site_getindex_704_20 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_op_ne_704_24 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::NotEqual));
  __site_istrue_704_24 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_705_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_int_710_49 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_double_711_51 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_get_get_state_715_19 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "get_state", false));
  __site_call0_715_29 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(0)));
  __site_get_set_state_718_12 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "set_state", false));
  __site_call1_718_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_random_721_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "random", false));
  __site_get___RandomState_ctor_721_25 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "__RandomState_ctor", false));
  __site_get_get_state_721_54 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "get_state", false));
  __site_call0_721_64 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(0)));
  __site_cvt_long_851_20 = CallSite< System::Func< CallSite^, System::Object^, long >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, long::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_long_853_20 = CallSite< System::Func< CallSite^, System::Object^, long >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, long::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_long_854_21 = CallSite< System::Func< CallSite^, System::Object^, long >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, long::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_858_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_empty_863_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_call2_863_26 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_864_24 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_npy_intp_864_24 = CallSite< System::Func< CallSite^, System::Object^, npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_870_4 = CallSite< System::Func< CallSite^, System::Object^, unsigned int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, unsigned int::typeid, ConversionResultKind::ExplicitCast));
  __site_op_mul_892_35 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Multiply));
  __site_cvt_double_974_30 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_double_975_32 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_get_array_984_17 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_call1_984_23 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_array_984_34 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_call1_984_40 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_sub_984_30 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Subtract));
  __site_call1_1026_14 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_eq_1026_21 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Equal));
  __site_istrue_1026_21 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_get_random_sample_1027_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "random_sample", false));
  __site_call0_1027_37 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(0)));
  __site_get_random_sample_1029_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "random_sample", false));
  __site_call0_size_1029_37 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(gcnew array<Argument>{Argument("size")})));
  __site_call1_1082_14 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_eq_1082_21 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Equal));
  __site_istrue_1082_21 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_get_standard_normal_1083_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "standard_normal", false));
  __site_call0_1083_39 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(0)));
  __site_get_standard_normal_1085_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "standard_normal", false));
  __site_call1_1085_39 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_randint_1163_19 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "randint", false));
  __site_op_add_1163_38 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Add));
  __site_call3_1163_27 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(3)));
  __site_cvt_double_1284_30 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_double_1285_34 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1292_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_1296_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_1296_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_1296_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_1296_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_1296_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1297_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_1341_26 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_double_1342_26 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1349_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call1_1351_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_1354_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_1354_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_1354_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_1354_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_1354_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1355_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_1356_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_1356_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_1356_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_1356_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_1356_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1357_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_1402_34 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1409_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_1413_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_1413_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_1413_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_1413_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_1413_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1414_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_1516_34 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1523_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_1527_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_1527_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_1527_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_1527_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_1527_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1528_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_1605_34 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_double_1606_34 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1613_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call1_1615_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_1619_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_1619_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_1619_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_1619_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_1619_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1620_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_1621_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_1621_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_1621_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_1621_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_1621_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1622_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_1710_34 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_double_1711_34 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1718_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call1_1720_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_1724_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_1724_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_1724_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_1724_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_1724_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1725_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_1726_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_1726_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_1726_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_1726_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_1726_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1727_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_1798_34 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_double_1799_34 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_double_1800_32 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1807_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call1_1809_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call1_1811_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_1815_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_1815_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_1815_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_1815_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_1815_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1816_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_1817_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_1817_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_1817_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_1817_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_1817_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1818_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_1819_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_1819_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less", false));
  __site_call2_1819_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_1819_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_1819_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1820_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_1892_28 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1899_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_1902_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_1902_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_1902_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_1902_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_1902_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1903_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_1979_28 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_double_1980_32 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1987_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call1_1989_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_1993_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_1993_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_1993_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_1993_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_1993_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1994_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_1995_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_1995_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_1995_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_1995_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_1995_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_1996_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_2150_28 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_2157_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_2160_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_2160_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_2160_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_2160_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_2160_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_2161_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_2246_28 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_double_2247_34 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_2254_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_2258_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_2258_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less", false));
  __site_call2_2258_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_2258_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_2258_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_2259_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_2339_26 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_2346_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_2349_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_2349_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_2349_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_2349_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_2349_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_2350_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_2441_26 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_2448_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_2451_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_2451_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_2451_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_2451_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_2451_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_2452_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_2552_26 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_2559_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_2562_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_2562_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_2562_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_2562_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_2562_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_2563_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_2643_30 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_double_2644_34 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_2651_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_2655_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_2655_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_2655_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_2655_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_2655_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_2656_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_2771_30 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_double_2772_34 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_2779_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_2783_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_2783_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_2783_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_2783_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_2783_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_2784_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_2862_30 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_double_2863_34 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_2870_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_2874_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_2874_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_2874_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_2874_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_2874_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_2875_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_2994_32 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_double_2995_34 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3002_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3006_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_3006_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_3006_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3006_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3006_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3007_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_3070_34 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3077_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3081_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_3081_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_3081_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3081_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3081_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3082_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_3152_32 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_double_3153_34 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3160_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call1_3162_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3166_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_3166_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_3166_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3166_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3166_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3167_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3168_15 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_3168_22 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_3168_33 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3168_19 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3168_19 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3169_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_3235_32 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_double_3236_34 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_double_3237_32 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3244_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call1_3246_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call1_3248_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3252_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_greater_3252_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "greater", false));
  __site_call2_3252_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3252_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3252_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3253_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3254_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_greater_3254_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "greater", false));
  __site_call2_3254_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3254_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3254_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3255_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3256_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_equal_3256_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "equal", false));
  __site_call2_3256_26 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3256_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3256_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3257_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_3348_26 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_long_3349_24 = CallSite< System::Func< CallSite^, System::Object^, long >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, long::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3356_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call1_3358_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call1_3360_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3364_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_3364_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_3364_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3364_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3364_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3365_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3366_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_3366_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less", false));
  __site_call2_3366_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3366_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3366_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3367_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3368_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_greater_3368_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "greater", false));
  __site_call2_3368_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3368_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3368_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3369_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_3443_26 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_double_3444_26 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3451_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call1_3453_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call1_3455_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3459_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_3459_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_3459_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3459_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3459_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3460_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3461_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_3461_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less", false));
  __site_call2_3461_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3461_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3461_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3462_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3463_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_greater_3463_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "greater", false));
  __site_call2_3463_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3463_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3463_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3464_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_3520_30 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_op_lt_3526_19 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::LessThan));
  __site_istrue_3526_19 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3527_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3530_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_3530_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less", false));
  __site_call2_3530_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3530_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3530_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3531_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_3614_26 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3621_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3624_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_3624_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_3624_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3624_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3624_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3625_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_3677_26 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3684_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call1_3686_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3689_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_3689_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less", false));
  __site_call2_3689_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3689_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3689_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3690_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3691_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_greater_3691_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "greater", false));
  __site_call2_3691_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3691_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3691_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3692_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_long_3783_32 = CallSite< System::Func< CallSite^, System::Object^, long >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, long::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_long_3784_30 = CallSite< System::Func< CallSite^, System::Object^, long >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, long::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_long_3785_36 = CallSite< System::Func< CallSite^, System::Object^, long >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, long::typeid, ConversionResultKind::ExplicitCast));
  __site_op_lt_3791_21 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::LessThan));
  __site_istrue_3791_21 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3792_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_lt_3793_20 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::LessThan));
  __site_istrue_3793_20 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3794_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_lt_3795_23 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::LessThan));
  __site_istrue_3795_23 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3796_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_add_3797_21 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Add));
  __site_op_lt_3797_28 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::LessThan));
  __site_istrue_3797_28 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3798_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3802_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_3802_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less", false));
  __site_call2_3802_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3802_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3802_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3803_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3804_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_3804_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less", false));
  __site_call2_3804_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3804_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3804_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3805_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3806_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_3806_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less", false));
  __site_call2_3806_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3806_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3806_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3807_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3808_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_3808_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less", false));
  __site_get_add_3808_28 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "add", false));
  __site_call2_3808_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call2_3808_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3808_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3808_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3809_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_double_3892_26 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3899_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call1_3901_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3904_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_less_equal_3904_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "less_equal", false));
  __site_call2_3904_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3904_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3904_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3905_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_any_3906_13 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "any", false));
  __site_get_greater_equal_3906_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "greater_equal", false));
  __site_call2_3906_34 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_3906_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_istrue_3906_17 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_3907_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_array_4004_17 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_call1_4004_23 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_array_4005_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_call1_4005_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_shape_4010_19 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "shape", false));
  __site_call1_4010_14 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_ne_4010_27 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::NotEqual));
  __site_istrue_4010_27 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_4011_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_shape_4012_19 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "shape", false));
  __site_call1_4012_15 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_ne_4012_27 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::NotEqual));
  __site_istrue_4012_27 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_get_shape_4012_40 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "shape", false));
  __site_getindex_4012_46 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_get_shape_4012_56 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "shape", false));
  __site_getindex_4012_62 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_op_ne_4012_50 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::NotEqual));
  __site_istrue_4012_50 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_4013_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_shape_4014_15 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "shape", false));
  __site_getindex_4014_21 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_get_shape_4014_31 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "shape", false));
  __site_getindex_4014_37 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_op_ne_4014_25 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::NotEqual));
  __site_istrue_4014_25 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_4015_31 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call2_4017_21 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_istrue_4017_21 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_getslice_4019_32 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetSliceBinder(__pyx_context));
  __site_call1_4019_26 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_append_4020_19 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "append", false));
  __site_get_shape_4020_31 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "shape", false));
  __site_getindex_4020_37 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_call1_4020_26 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_standard_normal_4024_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "standard_normal", false));
  __site_get_multiply_4024_35 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "multiply", false));
  __site_get_reduce_4024_44 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "reduce", false));
  __site_call1_4024_51 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call1_4024_32 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_multiply_4025_21 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "multiply", false));
  __site_get_reduce_4025_30 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "reduce", false));
  __site_call1_4025_55 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_sub_4025_69 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Subtract));
  __site_cvt_Py_ssize_t_4025_69 = CallSite< System::Func< CallSite^, System::Object^, Py_ssize_t >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, Py_ssize_t::typeid, ConversionResultKind::ExplicitCast));
  __site_getslice_4025_49 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetSliceBinder(__pyx_context));
  __site_call1_4025_37 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_shape_4026_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "shape", false));
  __site_getindex_4026_29 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_set_shape_4025_9 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetAction(__pyx_context, "shape"));
  __site_call1_4036_21 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_dot_4037_14 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "dot", false));
  __site_get_sqrt_4037_25 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "sqrt", false));
  __site_call1_4037_30 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_mul_4037_21 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Multiply));
  __site_call2_4037_18 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_add_4040_10 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "add", false));
  __site_call3_4040_14 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(3)));
  __site_call1_4041_23 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_set_shape_4041_9 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetAction(__pyx_context, "shape"));
  __site_cvt_4044_4 = CallSite< System::Func< CallSite^, System::Object^, long >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, long::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_4102_15 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_long_4102_15 = CallSite< System::Func< CallSite^, System::Object^, long >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, long::typeid, ConversionResultKind::ExplicitCast));
  __site_get_double_4103_35 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_4103_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_call1_4107_28 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call1_4111_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_add_4114_25 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Add));
  __site_get_zeros_4116_19 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "zeros", false));
  __site_call2_4116_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_4119_24 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_op_lt_4119_16 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::LessThan));
  __site_istrue_4119_16 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_4196_15 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_long_4196_15 = CallSite< System::Func< CallSite^, System::Object^, long >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, long::typeid, ConversionResultKind::ExplicitCast));
  __site_get_double_4197_40 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call4_4197_30 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(4)));
  __site_call1_4202_17 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_add_4205_25 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Add));
  __site_get_zeros_4207_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "zeros", false));
  __site_get_double_4207_34 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "double", false));
  __site_call2_4207_24 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_size_4211_23 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "size", false));
  __site_cvt_long_4211_23 = CallSite< System::Func< CallSite^, System::Object^, long >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, long::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_4236_15 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_sub_4236_19 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Subtract));
  __site_cvt_long_4236_19 = CallSite< System::Func< CallSite^, System::Object^, long >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, long::typeid, ConversionResultKind::ExplicitCast));
  __site_getindex_4238_21 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_call1_4238_19 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_long_4238_19 = CallSite< System::Func< CallSite^, System::Object^, long >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, long::typeid, ConversionResultKind::ExplicitCast));
  __site_getindex_4246_30 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_getindex_4246_36 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_setindex_4246_17 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_4246_23 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_getindex_4250_28 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_call2_4250_26 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_cvt_int_4250_26 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_getindex_4254_34 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_get_copy_4254_37 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "copy", false));
  __site_call0_4254_42 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(0)));
  __site_getindex_4254_47 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_get_copy_4254_50 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "copy", false));
  __site_call0_4254_55 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(0)));
  __site_setindex_4254_21 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_4254_27 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_getindex_4259_34 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_getslice_4259_37 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetSliceBinder(__pyx_context));
  __site_getindex_4259_43 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_getslice_4259_46 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetSliceBinder(__pyx_context));
  __site_setindex_4259_21 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_4259_27 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_get_integer_4289_33 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "integer", false));
  __site_call2_4289_21 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_istrue_4289_21 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_get_arange_4290_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "arange", false));
  __site_call1_4290_27 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_array_4292_20 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "array", false));
  __site_call1_4292_26 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_shuffle_4293_12 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "shuffle", false));
  __site_call1_4293_20 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
}
[SpecialName]
static void PerformModuleReload(PythonContext^ context, PythonDictionary^ dict) {
  dict["__builtins__"] = context->BuiltinModuleInstance;
  __pyx_context = (gcnew ModuleContext(dict, context))->GlobalContext;
  __Pyx_InitSites(__pyx_context);
  __Pyx_InitGlobals();
  /*--- Type init code ---*/
  __pyx_ptype_6mtrand_RandomState = safe_cast<Types::PythonType^>(dict["RandomState"]);
  /*--- Create function pointers ---*/
  /*--- Execution code ---*/
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":124
 * 
 * # Initialize numpy
 * import numpy as np             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "numpy", -1));
  PythonOps::SetGlobal(__pyx_context, "np", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":898
 *         return res
 * 
 *     def uniform(self, low=0.0, high=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         uniform(low=0.0, high=1.0, size=1)
 */
  __pyx_t_1 = 0.0;
  __pyx_k_1 = __pyx_t_1;
  __pyx_t_1 = nullptr;
  __pyx_t_1 = 1.0;
  __pyx_k_2 = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1198
 *         return cont0_array(self.internal_state, rk_gauss, size)
 * 
 *     def normal(self, loc=0.0, scale=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         normal(loc=0.0, scale=1.0, size=None)
 */
  __pyx_t_1 = 0.0;
  __pyx_k_3 = __pyx_t_1;
  __pyx_t_1 = nullptr;
  __pyx_t_1 = 1.0;
  __pyx_k_4 = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1360
 *         return cont2_array(self.internal_state, rk_beta, size, a, b)
 * 
 *     def exponential(self, scale=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         exponential(scale=1.0, size=None)
 */
  __pyx_t_1 = 1.0;
  __pyx_k_5 = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1531
 *         return cont1_array(self.internal_state, rk_standard_gamma, size, shape)
 * 
 *     def gamma(self, shape, scale=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         gamma(shape, scale=1.0, size=None)
 */
  __pyx_t_1 = 1.0;
  __pyx_k_6 = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2566
 *         return cont1_array(self.internal_state, rk_power, size, a)
 * 
 *     def laplace(self, loc=0.0, scale=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         laplace(loc=0.0, scale=1.0, size=None)
 */
  __pyx_t_1 = 0.0;
  __pyx_k_7 = __pyx_t_1;
  __pyx_t_1 = nullptr;
  __pyx_t_1 = 1.0;
  __pyx_k_8 = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2660
 * 
 * 
 *     def gumbel(self, loc=0.0, scale=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         gumbel(loc=0.0, scale=1.0, size=None)
 */
  __pyx_t_1 = 0.0;
  __pyx_k_9 = __pyx_t_1;
  __pyx_t_1 = nullptr;
  __pyx_t_1 = 1.0;
  __pyx_k_10 = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2787
 *         return cont2_array(self.internal_state, rk_gumbel, size, loc, scale)
 * 
 *     def logistic(self, loc=0.0, scale=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         logistic(loc=0.0, scale=1.0, size=None)
 */
  __pyx_t_1 = 0.0;
  __pyx_k_11 = __pyx_t_1;
  __pyx_t_1 = nullptr;
  __pyx_t_1 = 1.0;
  __pyx_k_12 = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":2878
 *         return cont2_array(self.internal_state, rk_logistic, size, loc, scale)
 * 
 *     def lognormal(self, mean=0.0, sigma=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         lognormal(mean=0.0, sigma=1.0, size=None)
 */
  __pyx_t_1 = 0.0;
  __pyx_k_13 = __pyx_t_1;
  __pyx_t_1 = nullptr;
  __pyx_t_1 = 1.0;
  __pyx_k_14 = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3011
 *                            mean, sigma)
 * 
 *     def rayleigh(self, scale=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         rayleigh(scale=1.0, size=None)
 */
  __pyx_t_1 = 1.0;
  __pyx_k_15 = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":3468
 *                             size, n, p)
 * 
 *     def poisson(self, lam=1.0, size=None):             # <<<<<<<<<<<<<<
 *         """
 *         poisson(lam=1.0, size=None)
 */
  __pyx_t_1 = 1.0;
  __pyx_k_16 = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4297
 * 
 * 
 * _rand = RandomState()             # <<<<<<<<<<<<<<
 * seed = _rand.seed
 * get_state = _rand.get_state
 */
  __pyx_t_1 = __site_call0_4297_19->Target(__site_call0_4297_19, __pyx_context, ((System::Object^)((System::Object^)__pyx_ptype_6mtrand_RandomState)));
  PythonOps::SetGlobal(__pyx_context, "_rand", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4298
 * 
 * _rand = RandomState()
 * seed = _rand.seed             # <<<<<<<<<<<<<<
 * get_state = _rand.get_state
 * set_state = _rand.set_state
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_seed_4298_12->Target(__site_get_seed_4298_12, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "seed", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4299
 * _rand = RandomState()
 * seed = _rand.seed
 * get_state = _rand.get_state             # <<<<<<<<<<<<<<
 * set_state = _rand.set_state
 * random_sample = _rand.random_sample
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_get_state_4299_17->Target(__site_get_get_state_4299_17, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "get_state", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4300
 * seed = _rand.seed
 * get_state = _rand.get_state
 * set_state = _rand.set_state             # <<<<<<<<<<<<<<
 * random_sample = _rand.random_sample
 * randint = _rand.randint
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_set_state_4300_17->Target(__site_get_set_state_4300_17, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "set_state", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4301
 * get_state = _rand.get_state
 * set_state = _rand.set_state
 * random_sample = _rand.random_sample             # <<<<<<<<<<<<<<
 * randint = _rand.randint
 * bytes = _rand.bytes
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_random_sample_4301_21->Target(__site_get_random_sample_4301_21, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "random_sample", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4302
 * set_state = _rand.set_state
 * random_sample = _rand.random_sample
 * randint = _rand.randint             # <<<<<<<<<<<<<<
 * bytes = _rand.bytes
 * uniform = _rand.uniform
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_randint_4302_15->Target(__site_get_randint_4302_15, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "randint", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4303
 * random_sample = _rand.random_sample
 * randint = _rand.randint
 * bytes = _rand.bytes             # <<<<<<<<<<<<<<
 * uniform = _rand.uniform
 * rand = _rand.rand
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_bytes_4303_13->Target(__site_get_bytes_4303_13, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "bytes", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4304
 * randint = _rand.randint
 * bytes = _rand.bytes
 * uniform = _rand.uniform             # <<<<<<<<<<<<<<
 * rand = _rand.rand
 * randn = _rand.randn
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_uniform_4304_15->Target(__site_get_uniform_4304_15, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "uniform", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4305
 * bytes = _rand.bytes
 * uniform = _rand.uniform
 * rand = _rand.rand             # <<<<<<<<<<<<<<
 * randn = _rand.randn
 * random_integers = _rand.random_integers
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_rand_4305_12->Target(__site_get_rand_4305_12, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "rand", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4306
 * uniform = _rand.uniform
 * rand = _rand.rand
 * randn = _rand.randn             # <<<<<<<<<<<<<<
 * random_integers = _rand.random_integers
 * standard_normal = _rand.standard_normal
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_randn_4306_13->Target(__site_get_randn_4306_13, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "randn", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4307
 * rand = _rand.rand
 * randn = _rand.randn
 * random_integers = _rand.random_integers             # <<<<<<<<<<<<<<
 * standard_normal = _rand.standard_normal
 * normal = _rand.normal
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_random_integers_4307_23->Target(__site_get_random_integers_4307_23, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "random_integers", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4308
 * randn = _rand.randn
 * random_integers = _rand.random_integers
 * standard_normal = _rand.standard_normal             # <<<<<<<<<<<<<<
 * normal = _rand.normal
 * beta = _rand.beta
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_standard_normal_4308_23->Target(__site_get_standard_normal_4308_23, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "standard_normal", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4309
 * random_integers = _rand.random_integers
 * standard_normal = _rand.standard_normal
 * normal = _rand.normal             # <<<<<<<<<<<<<<
 * beta = _rand.beta
 * exponential = _rand.exponential
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_normal_4309_14->Target(__site_get_normal_4309_14, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "normal", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4310
 * standard_normal = _rand.standard_normal
 * normal = _rand.normal
 * beta = _rand.beta             # <<<<<<<<<<<<<<
 * exponential = _rand.exponential
 * standard_exponential = _rand.standard_exponential
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_beta_4310_12->Target(__site_get_beta_4310_12, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "beta", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4311
 * normal = _rand.normal
 * beta = _rand.beta
 * exponential = _rand.exponential             # <<<<<<<<<<<<<<
 * standard_exponential = _rand.standard_exponential
 * standard_gamma = _rand.standard_gamma
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_exponential_4311_19->Target(__site_get_exponential_4311_19, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "exponential", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4312
 * beta = _rand.beta
 * exponential = _rand.exponential
 * standard_exponential = _rand.standard_exponential             # <<<<<<<<<<<<<<
 * standard_gamma = _rand.standard_gamma
 * gamma = _rand.gamma
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_standard_exponential_4312_28->Target(__site_get_standard_exponential_4312_28, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "standard_exponential", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4313
 * exponential = _rand.exponential
 * standard_exponential = _rand.standard_exponential
 * standard_gamma = _rand.standard_gamma             # <<<<<<<<<<<<<<
 * gamma = _rand.gamma
 * f = _rand.f
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_standard_gamma_4313_22->Target(__site_get_standard_gamma_4313_22, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "standard_gamma", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4314
 * standard_exponential = _rand.standard_exponential
 * standard_gamma = _rand.standard_gamma
 * gamma = _rand.gamma             # <<<<<<<<<<<<<<
 * f = _rand.f
 * noncentral_f = _rand.noncentral_f
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_gamma_4314_13->Target(__site_get_gamma_4314_13, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "gamma", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4315
 * standard_gamma = _rand.standard_gamma
 * gamma = _rand.gamma
 * f = _rand.f             # <<<<<<<<<<<<<<
 * noncentral_f = _rand.noncentral_f
 * chisquare = _rand.chisquare
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_f_4315_9->Target(__site_get_f_4315_9, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "f", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4316
 * gamma = _rand.gamma
 * f = _rand.f
 * noncentral_f = _rand.noncentral_f             # <<<<<<<<<<<<<<
 * chisquare = _rand.chisquare
 * noncentral_chisquare = _rand.noncentral_chisquare
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_noncentral_f_4316_20->Target(__site_get_noncentral_f_4316_20, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "noncentral_f", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4317
 * f = _rand.f
 * noncentral_f = _rand.noncentral_f
 * chisquare = _rand.chisquare             # <<<<<<<<<<<<<<
 * noncentral_chisquare = _rand.noncentral_chisquare
 * standard_cauchy = _rand.standard_cauchy
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_chisquare_4317_17->Target(__site_get_chisquare_4317_17, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "chisquare", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4318
 * noncentral_f = _rand.noncentral_f
 * chisquare = _rand.chisquare
 * noncentral_chisquare = _rand.noncentral_chisquare             # <<<<<<<<<<<<<<
 * standard_cauchy = _rand.standard_cauchy
 * standard_t = _rand.standard_t
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_noncentral_chisquare_4318_28->Target(__site_get_noncentral_chisquare_4318_28, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "noncentral_chisquare", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4319
 * chisquare = _rand.chisquare
 * noncentral_chisquare = _rand.noncentral_chisquare
 * standard_cauchy = _rand.standard_cauchy             # <<<<<<<<<<<<<<
 * standard_t = _rand.standard_t
 * vonmises = _rand.vonmises
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_standard_cauchy_4319_23->Target(__site_get_standard_cauchy_4319_23, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "standard_cauchy", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4320
 * noncentral_chisquare = _rand.noncentral_chisquare
 * standard_cauchy = _rand.standard_cauchy
 * standard_t = _rand.standard_t             # <<<<<<<<<<<<<<
 * vonmises = _rand.vonmises
 * pareto = _rand.pareto
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_standard_t_4320_18->Target(__site_get_standard_t_4320_18, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "standard_t", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4321
 * standard_cauchy = _rand.standard_cauchy
 * standard_t = _rand.standard_t
 * vonmises = _rand.vonmises             # <<<<<<<<<<<<<<
 * pareto = _rand.pareto
 * weibull = _rand.weibull
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_vonmises_4321_16->Target(__site_get_vonmises_4321_16, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "vonmises", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4322
 * standard_t = _rand.standard_t
 * vonmises = _rand.vonmises
 * pareto = _rand.pareto             # <<<<<<<<<<<<<<
 * weibull = _rand.weibull
 * power = _rand.power
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_pareto_4322_14->Target(__site_get_pareto_4322_14, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "pareto", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4323
 * vonmises = _rand.vonmises
 * pareto = _rand.pareto
 * weibull = _rand.weibull             # <<<<<<<<<<<<<<
 * power = _rand.power
 * laplace = _rand.laplace
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_weibull_4323_15->Target(__site_get_weibull_4323_15, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "weibull", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4324
 * pareto = _rand.pareto
 * weibull = _rand.weibull
 * power = _rand.power             # <<<<<<<<<<<<<<
 * laplace = _rand.laplace
 * gumbel = _rand.gumbel
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_power_4324_13->Target(__site_get_power_4324_13, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "power", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4325
 * weibull = _rand.weibull
 * power = _rand.power
 * laplace = _rand.laplace             # <<<<<<<<<<<<<<
 * gumbel = _rand.gumbel
 * logistic = _rand.logistic
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_laplace_4325_15->Target(__site_get_laplace_4325_15, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "laplace", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4326
 * power = _rand.power
 * laplace = _rand.laplace
 * gumbel = _rand.gumbel             # <<<<<<<<<<<<<<
 * logistic = _rand.logistic
 * lognormal = _rand.lognormal
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_gumbel_4326_14->Target(__site_get_gumbel_4326_14, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "gumbel", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4327
 * laplace = _rand.laplace
 * gumbel = _rand.gumbel
 * logistic = _rand.logistic             # <<<<<<<<<<<<<<
 * lognormal = _rand.lognormal
 * rayleigh = _rand.rayleigh
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_logistic_4327_16->Target(__site_get_logistic_4327_16, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "logistic", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4328
 * gumbel = _rand.gumbel
 * logistic = _rand.logistic
 * lognormal = _rand.lognormal             # <<<<<<<<<<<<<<
 * rayleigh = _rand.rayleigh
 * wald = _rand.wald
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_lognormal_4328_17->Target(__site_get_lognormal_4328_17, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "lognormal", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4329
 * logistic = _rand.logistic
 * lognormal = _rand.lognormal
 * rayleigh = _rand.rayleigh             # <<<<<<<<<<<<<<
 * wald = _rand.wald
 * triangular = _rand.triangular
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_rayleigh_4329_16->Target(__site_get_rayleigh_4329_16, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "rayleigh", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4330
 * lognormal = _rand.lognormal
 * rayleigh = _rand.rayleigh
 * wald = _rand.wald             # <<<<<<<<<<<<<<
 * triangular = _rand.triangular
 * 
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_wald_4330_12->Target(__site_get_wald_4330_12, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "wald", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4331
 * rayleigh = _rand.rayleigh
 * wald = _rand.wald
 * triangular = _rand.triangular             # <<<<<<<<<<<<<<
 * 
 * binomial = _rand.binomial
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_triangular_4331_18->Target(__site_get_triangular_4331_18, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "triangular", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4333
 * triangular = _rand.triangular
 * 
 * binomial = _rand.binomial             # <<<<<<<<<<<<<<
 * negative_binomial = _rand.negative_binomial
 * poisson = _rand.poisson
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_binomial_4333_16->Target(__site_get_binomial_4333_16, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "binomial", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4334
 * 
 * binomial = _rand.binomial
 * negative_binomial = _rand.negative_binomial             # <<<<<<<<<<<<<<
 * poisson = _rand.poisson
 * zipf = _rand.zipf
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_negative_binomial_4334_25->Target(__site_get_negative_binomial_4334_25, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "negative_binomial", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4335
 * binomial = _rand.binomial
 * negative_binomial = _rand.negative_binomial
 * poisson = _rand.poisson             # <<<<<<<<<<<<<<
 * zipf = _rand.zipf
 * geometric = _rand.geometric
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_poisson_4335_15->Target(__site_get_poisson_4335_15, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "poisson", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4336
 * negative_binomial = _rand.negative_binomial
 * poisson = _rand.poisson
 * zipf = _rand.zipf             # <<<<<<<<<<<<<<
 * geometric = _rand.geometric
 * hypergeometric = _rand.hypergeometric
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_zipf_4336_12->Target(__site_get_zipf_4336_12, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "zipf", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4337
 * poisson = _rand.poisson
 * zipf = _rand.zipf
 * geometric = _rand.geometric             # <<<<<<<<<<<<<<
 * hypergeometric = _rand.hypergeometric
 * logseries = _rand.logseries
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_geometric_4337_17->Target(__site_get_geometric_4337_17, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "geometric", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4338
 * zipf = _rand.zipf
 * geometric = _rand.geometric
 * hypergeometric = _rand.hypergeometric             # <<<<<<<<<<<<<<
 * logseries = _rand.logseries
 * 
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_hypergeometric_4338_22->Target(__site_get_hypergeometric_4338_22, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "hypergeometric", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4339
 * geometric = _rand.geometric
 * hypergeometric = _rand.hypergeometric
 * logseries = _rand.logseries             # <<<<<<<<<<<<<<
 * 
 * multivariate_normal = _rand.multivariate_normal
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_logseries_4339_17->Target(__site_get_logseries_4339_17, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "logseries", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4341
 * logseries = _rand.logseries
 * 
 * multivariate_normal = _rand.multivariate_normal             # <<<<<<<<<<<<<<
 * multinomial = _rand.multinomial
 * dirichlet = _rand.dirichlet
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_multivariate_normal_4341_27->Target(__site_get_multivariate_normal_4341_27, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "multivariate_normal", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4342
 * 
 * multivariate_normal = _rand.multivariate_normal
 * multinomial = _rand.multinomial             # <<<<<<<<<<<<<<
 * dirichlet = _rand.dirichlet
 * 
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_multinomial_4342_19->Target(__site_get_multinomial_4342_19, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "multinomial", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4343
 * multivariate_normal = _rand.multivariate_normal
 * multinomial = _rand.multinomial
 * dirichlet = _rand.dirichlet             # <<<<<<<<<<<<<<
 * 
 * shuffle = _rand.shuffle
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_dirichlet_4343_17->Target(__site_get_dirichlet_4343_17, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "dirichlet", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4345
 * dirichlet = _rand.dirichlet
 * 
 * shuffle = _rand.shuffle             # <<<<<<<<<<<<<<
 * permutation = _rand.permutation
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_2 = __site_get_shuffle_4345_15->Target(__site_get_shuffle_4345_15, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "shuffle", __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":4346
 * 
 * shuffle = _rand.shuffle
 * permutation = _rand.permutation             # <<<<<<<<<<<<<<
 */
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "_rand");
  __pyx_t_1 = __site_get_permutation_4346_19->Target(__site_get_permutation_4346_19, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "permutation", __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Documents and Settings\Jason\Documents\Visual Studio 2010\Projects\numpy-refactor\numpy\random\mtrand\mtrand.pyx":1
 * # mtrad.pyx -- A Pyrex wrapper of Jean-Sebastien Roy's RandomKit             # <<<<<<<<<<<<<<
 * #
 * # Copyright 2005 Robert Kern (robert.kern@gmail.com)
 */
  __pyx_t_1 = PythonOps::MakeEmptyDict();
  PythonOps::SetGlobal(__pyx_context, "__test__", ((System::Object^)__pyx_t_1));
  __pyx_t_1 = nullptr;

  /* "C:\Python26\lib\site-packages\Cython\Includes\cpython\type.pxd":2
 * 
 * cdef extern from "Python.h":             # <<<<<<<<<<<<<<
 *     # The C structure of the objects used to describe built-in types.
 * 
 */
}
/* Cython code section 'cleanup_globals' */
/* Cython code section 'cleanup_module' */
/* Cython code section 'main_method' */
/* Cython code section 'utility_code_def' */

/* Runtime support code */
/* Cython code section 'end' */
};
[assembly: PythonModule("mtrand", module_mtrand::typeid)];
};
