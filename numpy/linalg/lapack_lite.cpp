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
  dict["__module__"] = "numpy.linalg.lapack_lite";
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
#define __PYX_HAVE_API__numpy__linalg__lapack_lite
#include "lapack_lite.h"
#include "stdint.h"
#include "npy_common.h"
#include "npy_defs.h"
#include "npy_descriptor.h"
#include "npy_arrayobject.h"
#include "npy_ufunc_object.h"
#include "npy_api.h"
#include "npy_iterators.h"
#include "npy_ironpython.h"

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


#if !defined(CYTHON_CCOMPLEX)
  #if defined(__cplusplus)
    #define CYTHON_CCOMPLEX 1
  #elif defined(_Complex_I)
    #define CYTHON_CCOMPLEX 1
  #else
    #define CYTHON_CCOMPLEX 0
  #endif
#endif

#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    #include <complex>
  #else
    #include <complex.h>
  #endif
#endif

#if CYTHON_CCOMPLEX && !defined(__cplusplus) && defined(__sun__) && defined(__GNUC__)
  #undef _Complex_I
  #define _Complex_I 1.0fj
#endif
/* Cython code section 'filename_table' */

static const char *__pyx_f[] = {
  0
};
/* Cython code section 'utility_code_proto_before_types' */
/* Cython code section 'numeric_typedefs' */

typedef signed char __pyx_t_5numpy_6linalg_5numpy_npy_byte;

typedef signed short __pyx_t_5numpy_6linalg_5numpy_npy_short;

typedef signed int __pyx_t_5numpy_6linalg_5numpy_npy_int;

typedef signed long __pyx_t_5numpy_6linalg_5numpy_npy_long;

typedef signed PY_LONG_LONG __pyx_t_5numpy_6linalg_5numpy_npy_longlong;

typedef unsigned char __pyx_t_5numpy_6linalg_5numpy_npy_ubyte;

typedef unsigned short __pyx_t_5numpy_6linalg_5numpy_npy_ushort;

typedef unsigned int __pyx_t_5numpy_6linalg_5numpy_npy_uint;

typedef unsigned long __pyx_t_5numpy_6linalg_5numpy_npy_ulong;

typedef unsigned PY_LONG_LONG __pyx_t_5numpy_6linalg_5numpy_npy_ulonglong;

typedef float __pyx_t_5numpy_6linalg_5numpy_npy_float;

typedef double __pyx_t_5numpy_6linalg_5numpy_npy_double;

typedef long double __pyx_t_5numpy_6linalg_5numpy_npy_longdouble;

typedef double __pyx_t_5numpy_6linalg_5numpy_double_t;

typedef intptr_t __pyx_t_5numpy_6linalg_5numpy_npy_intp;

typedef uintptr_t __pyx_t_5numpy_6linalg_5numpy_npy_uintp;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_byte __pyx_t_5numpy_6linalg_5numpy_npy_int8;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_short __pyx_t_5numpy_6linalg_5numpy_npy_int16;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_int __pyx_t_5numpy_6linalg_5numpy_npy_int32;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_longlong __pyx_t_5numpy_6linalg_5numpy_npy_int64;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_ubyte __pyx_t_5numpy_6linalg_5numpy_npy_uint8;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_ushort __pyx_t_5numpy_6linalg_5numpy_npy_uint16;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_uint __pyx_t_5numpy_6linalg_5numpy_npy_uint32;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_ulonglong __pyx_t_5numpy_6linalg_5numpy_npy_uint64;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_float __pyx_t_5numpy_6linalg_5numpy_npy_float32;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_double __pyx_t_5numpy_6linalg_5numpy_npy_float64;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_intp __pyx_t_5numpy_6linalg_5numpy_intp_t;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_int8 __pyx_t_5numpy_6linalg_5numpy_int8_t;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_int16 __pyx_t_5numpy_6linalg_5numpy_int16_t;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_int32 __pyx_t_5numpy_6linalg_5numpy_int32_t;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_int64 __pyx_t_5numpy_6linalg_5numpy_int64_t;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_uint8 __pyx_t_5numpy_6linalg_5numpy_uint8_t;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_uint16 __pyx_t_5numpy_6linalg_5numpy_uint16_t;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_uint32 __pyx_t_5numpy_6linalg_5numpy_uint32_t;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_uint64 __pyx_t_5numpy_6linalg_5numpy_uint64_t;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_float32 __pyx_t_5numpy_6linalg_5numpy_float32_t;

typedef __pyx_t_5numpy_6linalg_5numpy_npy_float64 __pyx_t_5numpy_6linalg_5numpy_float64_t;
/* Cython code section 'complex_type_declarations' */

#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    typedef ::std::complex< float > __pyx_t_float_complex;
  #else
    typedef float _Complex __pyx_t_float_complex;
  #endif
#else
    typedef struct { float real, imag; } __pyx_t_float_complex;
#endif

#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    typedef ::std::complex< double > __pyx_t_double_complex;
  #else
    typedef double _Complex __pyx_t_double_complex;
  #endif
#else
    typedef struct { double real, imag; } __pyx_t_double_complex;
#endif
/* Cython code section 'type_declarations' */

/* Type declarations */

typedef void (*__pyx_t_5numpy_6linalg_5numpy_NpyUFuncGenericFunction)(char **, __pyx_t_5numpy_6linalg_5numpy_npy_intp *, __pyx_t_5numpy_6linalg_5numpy_npy_intp *, void *);

typedef __pyx_t_5numpy_6linalg_5numpy_NpyUFuncGenericFunction __pyx_t_5numpy_6linalg_5numpy_PyUFuncGenericFunction;

typedef npy_cfloat __pyx_t_5numpy_6linalg_5numpy_cfloat_t;

typedef npy_cdouble __pyx_t_5numpy_6linalg_5numpy_cdouble_t;

typedef npy_clongdouble __pyx_t_5numpy_6linalg_5numpy_clongdouble_t;

typedef npy_cdouble __pyx_t_5numpy_6linalg_5numpy_complex_t;

typedef void (*__pyx_t_5numpy_6linalg_5numpy_PyArray_CopySwapFunc)(void *, void *, int, NpyArray *);
/* Cython code section 'utility_code_proto' */

#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    #define __Pyx_CREAL(z) ((z).real())
    #define __Pyx_CIMAG(z) ((z).imag())
  #else
    #define __Pyx_CREAL(z) (__real__(z))
    #define __Pyx_CIMAG(z) (__imag__(z))
  #endif
#else
    #define __Pyx_CREAL(z) ((z).real)
    #define __Pyx_CIMAG(z) ((z).imag)
#endif

#if defined(_WIN32) && defined(__cplusplus) && CYTHON_CCOMPLEX
    #define __Pyx_SET_CREAL(z,x) ((z).real(x))
    #define __Pyx_SET_CIMAG(z,y) ((z).imag(y))
#else
    #define __Pyx_SET_CREAL(z,x) __Pyx_CREAL(z) = (x)
    #define __Pyx_SET_CIMAG(z,y) __Pyx_CIMAG(z) = (y)
#endif

static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float, float);

#if CYTHON_CCOMPLEX
    #define __Pyx_c_eqf(a, b)   ((a)==(b))
    #define __Pyx_c_sumf(a, b)  ((a)+(b))
    #define __Pyx_c_difff(a, b) ((a)-(b))
    #define __Pyx_c_prodf(a, b) ((a)*(b))
    #define __Pyx_c_quotf(a, b) ((a)/(b))
    #define __Pyx_c_negf(a)     (-(a))
  #ifdef __cplusplus
    #define __Pyx_c_is_zerof(z) ((z)==(float)0)
    #define __Pyx_c_conjf(z)    (::std::conj(z))
    /*#define __Pyx_c_absf(z)     (::std::abs(z))*/
  #else
    #define __Pyx_c_is_zerof(z) ((z)==0)
    #define __Pyx_c_conjf(z)    (conjf(z))
    /*#define __Pyx_c_absf(z)     (cabsf(z))*/
 #endif
#else
    static CYTHON_INLINE int __Pyx_c_eqf(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_sumf(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_difff(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_prodf(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_quotf(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_negf(__pyx_t_float_complex);
    static CYTHON_INLINE int __Pyx_c_is_zerof(__pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_conjf(__pyx_t_float_complex);
    /*static CYTHON_INLINE float __Pyx_c_absf(__pyx_t_float_complex);*/
#endif

static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double, double);

#if CYTHON_CCOMPLEX
    #define __Pyx_c_eq(a, b)   ((a)==(b))
    #define __Pyx_c_sum(a, b)  ((a)+(b))
    #define __Pyx_c_diff(a, b) ((a)-(b))
    #define __Pyx_c_prod(a, b) ((a)*(b))
    #define __Pyx_c_quot(a, b) ((a)/(b))
    #define __Pyx_c_neg(a)     (-(a))
  #ifdef __cplusplus
    #define __Pyx_c_is_zero(z) ((z)==(double)0)
    #define __Pyx_c_conj(z)    (::std::conj(z))
    /*#define __Pyx_c_abs(z)     (::std::abs(z))*/
  #else
    #define __Pyx_c_is_zero(z) ((z)==0)
    #define __Pyx_c_conj(z)    (conj(z))
    /*#define __Pyx_c_abs(z)     (cabs(z))*/
 #endif
#else
    static CYTHON_INLINE int __Pyx_c_eq(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_sum(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_diff(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_prod(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_quot(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_neg(__pyx_t_double_complex);
    static CYTHON_INLINE int __Pyx_c_is_zero(__pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_conj(__pyx_t_double_complex);
    /*static CYTHON_INLINE double __Pyx_c_abs(__pyx_t_double_complex);*/
#endif
/* Cython code section 'module_declarations' */
/* Module declarations from libc.stdint */
/* Module declarations from numpy */
/* Module declarations from numpy.linalg.numpy */
static CYTHON_INLINE NumpyDotNet::dtype^ NpyArray_FindArrayType_2args(System::Object^, NumpyDotNet::dtype^); /*proto*/
static CYTHON_INLINE System::Object^ PyUFunc_FromFuncAndData(__pyx_t_5numpy_6linalg_5numpy_PyUFuncGenericFunction *, void **, char *, int, int, int, int, char *, char *, int); /*proto*/
static CYTHON_INLINE System::Object^ PyArray_DescrFromType(int); /*proto*/
static CYTHON_INLINE System::Object^ PyArray_ZEROS(int, __pyx_t_5numpy_6linalg_5numpy_npy_intp *, int, int); /*proto*/
static CYTHON_INLINE System::Object^ PyArray_EMPTY(int, __pyx_t_5numpy_6linalg_5numpy_npy_intp *, int, int); /*proto*/
static CYTHON_INLINE System::Object^ PyArray_Empty(int, __pyx_t_5numpy_6linalg_5numpy_npy_intp *, NumpyDotNet::dtype^, int); /*proto*/
static CYTHON_INLINE System::Object^ PyArray_New(void *, int, __pyx_t_5numpy_6linalg_5numpy_npy_intp *, int, __pyx_t_5numpy_6linalg_5numpy_npy_intp *, void *, int, int, void *); /*proto*/
static CYTHON_INLINE System::Object^ PyArray_SimpleNew(int, __pyx_t_5numpy_6linalg_5numpy_npy_intp *, int); /*proto*/
static CYTHON_INLINE System::Object^ PyArray_SimpleNewFromData(int, __pyx_t_5numpy_6linalg_5numpy_npy_intp *, int, void *); /*proto*/
static CYTHON_INLINE int PyArray_CHKFLAGS(NumpyDotNet::ndarray^, int); /*proto*/
static CYTHON_INLINE void *PyArray_DATA(NumpyDotNet::ndarray^); /*proto*/
static CYTHON_INLINE __pyx_t_5numpy_6linalg_5numpy_intp_t *PyArray_DIMS(NumpyDotNet::ndarray^); /*proto*/
static CYTHON_INLINE System::Object^ PyArray_DESCR(NumpyDotNet::ndarray^); /*proto*/
static CYTHON_INLINE int PyArray_ITEMSIZE(NumpyDotNet::ndarray^); /*proto*/
static CYTHON_INLINE System::Object^ PyArray_Return(System::Object^); /*proto*/
static CYTHON_INLINE __pyx_t_5numpy_6linalg_5numpy_intp_t PyArray_DIM(NumpyDotNet::ndarray^, int); /*proto*/
static CYTHON_INLINE System::Object^ PyArray_NDIM(NumpyDotNet::ndarray^); /*proto*/
static CYTHON_INLINE __pyx_t_5numpy_6linalg_5numpy_intp_t PyArray_SIZE(NumpyDotNet::ndarray^); /*proto*/
static CYTHON_INLINE __pyx_t_5numpy_6linalg_5numpy_npy_intp *PyArray_STRIDES(NumpyDotNet::ndarray^); /*proto*/
static CYTHON_INLINE __pyx_t_5numpy_6linalg_5numpy_npy_intp PyArray_NBYTES(NumpyDotNet::ndarray^); /*proto*/
static CYTHON_INLINE NpyArray *PyArray_ARRAY(NumpyDotNet::ndarray^); /*proto*/
static CYTHON_INLINE int PyArray_TYPE(NumpyDotNet::ndarray^); /*proto*/
static CYTHON_INLINE void *PyArray_Zero(System::Object^); /*proto*/
static CYTHON_INLINE System::Object^ NpyArray_Return(NpyArray *); /*proto*/
static CYTHON_INLINE int PyDataType_TYPE_NUM(NumpyDotNet::dtype^); /*proto*/
static CYTHON_INLINE System::Object^ PyArray_FromAny(System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^); /*proto*/
static CYTHON_INLINE System::Object^ PyArray_CopyFromObject(System::Object^, System::Object^, System::Object^, System::Object^); /*proto*/
static CYTHON_INLINE System::Object^ PyArray_FROMANY(System::Object^, System::Object^, System::Object^, System::Object^, System::Object^); /*proto*/
static CYTHON_INLINE System::Object^ PyArray_ContiguousFromObject(System::Object^, System::Object^, System::Object^, System::Object^); /*proto*/
static CYTHON_INLINE System::Object^ PyArray_CheckFromAny(System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^); /*proto*/
static CYTHON_INLINE System::Object^ PyArray_Check(System::Object^); /*proto*/
static CYTHON_INLINE System::Object^ PyArray_Cast(System::Object^, System::Object^); /*proto*/
static CYTHON_INLINE void import_array(void); /*proto*/
static CYTHON_INLINE System::Object^ PyArray_DescrConverter(System::Object^); /*proto*/
static CYTHON_INLINE System::Object^ PyNumber_Check(System::Object^); /*proto*/
static CYTHON_INLINE NpyArrayIterObject *PyArray_IterNew(NumpyDotNet::ndarray^); /*proto*/
static CYTHON_INLINE NpyArrayIterObject *PyArray_IterAllButAxis(NumpyDotNet::ndarray^, int *); /*proto*/
static CYTHON_INLINE void PyArray_ITER_NEXT(NpyArrayIterObject *); /*proto*/
static CYTHON_INLINE void PyArray_ITER_RESET(NpyArrayIterObject *); /*proto*/
static CYTHON_INLINE void *PyArray_ITER_DATA(NpyArrayIterObject *); /*proto*/
static CYTHON_INLINE NpyArrayNeighborhoodIterObject *PyArray_NeighborhoodIterNew(NpyArrayIterObject *, __pyx_t_5numpy_6linalg_5numpy_npy_intp *, int, void *, npy_free_func); /*proto*/
static CYTHON_INLINE int PyArrayNeighborhoodIter_Reset(NpyArrayNeighborhoodIterObject *); /*proto*/
static CYTHON_INLINE int PyArrayNeighborhoodIter_Next(NpyArrayNeighborhoodIterObject *); /*proto*/
static CYTHON_INLINE NumpyDotNet::ndarray^ NpyIter_ARRAY(NpyArrayIterObject *); /*proto*/
/* Module declarations from numpy.linalg.lapack_lite */
static int check_object(NumpyDotNet::ndarray^, int, char *, char *, char *); /*proto*/
/* Cython code section 'typeinfo' */
/* Cython code section 'before_global_var' */
#define __Pyx_MODULE_NAME "numpy.linalg.lapack_lite"

/* Implementation of numpy.linalg.lapack_lite */
namespace clr_lapack_lite {
  public ref class module_lapack_lite sealed abstract {
/* Cython code section 'global_var' */
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_mod_37_77;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_37_25;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_mod_39_77;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_39_25;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_byteorder_40_29;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ne_40_40;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_40_40;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_byteorder_40_71;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ne_40_82;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_40_82;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_mod_41_85;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_41_25;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_46_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_46_0_1;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_46_0_2;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_46_0_3;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_46_0_4;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_46_0_5;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_50_36;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_50_30;
static  CallSite< System::Func< CallSite^, System::Object^, char >^ >^ __site_cvt_cvt_char_50_30;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_51_36;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_51_30;
static  CallSite< System::Func< CallSite^, System::Object^, char >^ >^ __site_cvt_cvt_char_51_30;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_70_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_71_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_72_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_73_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_74_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_75_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_76_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_77_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_78_10;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_83_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_83_0_1;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_83_0_2;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_83_0_3;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_83_0_4;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_124_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_124_29;
static  CallSite< System::Func< CallSite^, System::Object^, char >^ >^ __site_cvt_cvt_char_124_29;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_125_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_125_29;
static  CallSite< System::Func< CallSite^, System::Object^, char >^ >^ __site_cvt_cvt_char_125_29;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_139_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_140_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_141_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_142_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_143_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_144_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_145_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_146_10;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_150_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_150_0_1;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_150_0_2;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_150_0_3;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_150_0_4;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_150_0_5;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_194_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_194_29;
static  CallSite< System::Func< CallSite^, System::Object^, char >^ >^ __site_cvt_cvt_char_194_29;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_195_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_195_29;
static  CallSite< System::Func< CallSite^, System::Object^, char >^ >^ __site_cvt_cvt_char_195_29;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_211_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_212_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_213_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_214_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_215_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_216_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_217_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_218_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_219_10;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_223_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_223_0_1;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_223_0_2;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_223_0_3;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_223_0_4;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_cvt_double_223_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_223_0_5;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_223_0_6;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_223_0_7;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_242_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_243_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_244_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_245_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_246_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_247_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_248_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_249_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_250_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_251_10;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_255_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_255_0_1;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_255_0_2;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_255_0_3;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_255_0_4;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_270_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_271_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_272_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_273_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_274_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_275_10;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_279_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_279_0_1;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_279_0_2;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_279_0_3;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_279_0_4;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_279_0_5;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_279_0_6;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_283_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_283_29;
static  CallSite< System::Func< CallSite^, System::Object^, char >^ >^ __site_cvt_cvt_char_283_29;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_eq_311_16;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_311_16;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_eq_313_18;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_313_18;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_eq_315_18;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_315_18;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_eq_315_33;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_315_33;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_321_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_322_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_323_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_324_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_325_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_326_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_327_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_328_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_329_10;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_333_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_333_0_1;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_333_0_2;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_333_0_3;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_343_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_344_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_345_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_346_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_347_10;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_351_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_351_0_1;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_351_0_2;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_353_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_353_29;
static  CallSite< System::Func< CallSite^, System::Object^, char >^ >^ __site_cvt_cvt_char_353_29;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_362_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_363_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_364_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_365_10;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_369_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_369_0_1;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_369_0_2;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_369_0_3;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_369_0_4;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_385_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_386_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_387_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_388_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_389_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_390_10;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_394_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_394_0_1;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_394_0_2;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_394_0_3;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_394_0_4;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_394_0_5;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_409_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_410_10;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_414_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_414_0_1;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_414_0_2;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_414_0_3;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_414_0_4;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_414_0_5;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_418_36;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_418_30;
static  CallSite< System::Func< CallSite^, System::Object^, char >^ >^ __site_cvt_cvt_char_418_30;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_419_36;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_419_30;
static  CallSite< System::Func< CallSite^, System::Object^, char >^ >^ __site_cvt_cvt_char_419_30;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_437_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_438_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_439_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_440_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_441_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_442_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_443_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_444_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_445_10;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_449_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_449_0_1;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_449_0_2;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_449_0_3;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_449_0_4;
static  CallSite< System::Func< CallSite^, System::Object^, double >^ >^ __site_cvt_cvt_double_449_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_449_0_5;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_449_0_6;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_449_0_7;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_471_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_472_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_473_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_474_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_475_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_476_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_477_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_478_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_479_10;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_483_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_483_0_1;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_483_0_2;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_483_0_3;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_483_0_4;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_498_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_499_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_500_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_501_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_502_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_503_10;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_507_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_507_0_1;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_507_0_2;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_507_0_3;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_507_0_4;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_507_0_5;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_507_0_6;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_511_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_511_29;
static  CallSite< System::Func< CallSite^, System::Object^, char >^ >^ __site_cvt_cvt_char_511_29;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_531_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_532_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_533_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_534_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_535_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_536_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_537_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_538_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_539_10;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_543_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_543_0_1;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_543_0_2;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_543_0_3;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_554_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_555_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_556_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_557_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_558_10;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_562_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_562_0_1;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_562_0_2;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_getindex_564_34;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_564_29;
static  CallSite< System::Func< CallSite^, System::Object^, char >^ >^ __site_cvt_cvt_char_564_29;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_573_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_574_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_575_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_576_10;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_580_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_580_0_1;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_580_0_2;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_580_0_3;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_580_0_4;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_596_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_597_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_598_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_599_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_600_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_601_10;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_605_0;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_605_0_1;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_605_0_2;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_605_0_3;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_605_0_4;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_605_0_5;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_619_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_620_10;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_append_325_18;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_325_25;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_zeros_327_16;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call3_327_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_append_333_18;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_333_25;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_335_16;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call3_335_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_append_341_18;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_341_25;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_empty_343_16;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call3_343_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Array_358_53;
static  CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >^ __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_358_53;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Array_361_48;
static  CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >^ __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_361_48;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Array_364_48;
static  CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >^ __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_364_48;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Array_367_69;
static  CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >^ __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_367_69;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Array_370_52;
static  CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >^ __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_370_52;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_ndarray_377_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_ArrayReturn_377_30;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_377_42;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Array_380_47;
static  CallSite< System::Func< CallSite^, System::Object^, PY_LONG_LONG >^ >^ __site_cvt_cvt_PY_LONG_LONG_380_47;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_ndim_383_14;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Array_386_48;
static  CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >^ __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_386_48;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Array_389_51;
static  CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >^ __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_389_51;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Array_392_51;
static  CallSite< System::Func< CallSite^, System::Object^, PY_LONG_LONG >^ >^ __site_cvt_cvt_PY_LONG_LONG_392_51;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Array_395_34;
static  CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >^ __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_395_34;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Array_398_48;
static  CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >^ __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_398_48;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_NpyArray_403_40;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Zero_403_49;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_403_54;
static  CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >^ __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_403_54;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Dtype_411_62;
static  CallSite< System::Func< CallSite^, System::Object^, PY_LONG_LONG >^ >^ __site_cvt_cvt_PY_LONG_LONG_411_62;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_NpyArray_416_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_FromAny_416_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call6_416_39;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_and_425_13;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_425_13;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ior_426_14;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_427_77;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_430_78;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_NpyArray_436_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_CheckFromAny_436_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call6_436_44;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_ndarray_440_29;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_440_21;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_NpyCoreApi_445_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_CastToType_445_33;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_cvt_int_445_100;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call3_445_44;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_NpyDescr_453_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_DescrConverter_453_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_453_46;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_458_21;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_cvt_bool_458_45;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_ScalarGeneric_458_73;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_458_58;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Array_461_51;
static  CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >^ __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_461_51;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Array_464_58;
static  CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >^ __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_464_58;
static CodeContext^ __pyx_context;
/* Cython code section 'decls' */
static char *__pyx_k_1 = "np.NPY_DOUBLE";
static char *__pyx_k_2 = "np.NPY_INT";
static char *__pyx_k_3 = "np.NPY_CDOUBLE";
static char *__pyx_k__a = "a";
static char *__pyx_k__b = "b";
static char *__pyx_k__s = "s";
static char *__pyx_k__u = "u";
static char *__pyx_k__w = "w";
static char *__pyx_k__vl = "vl";
static char *__pyx_k__vr = "vr";
static char *__pyx_k__vt = "vt";
static char *__pyx_k__wi = "wi";
static char *__pyx_k__wr = "wr";
static char *__pyx_k__tau = "tau";
static char *__pyx_k__ipiv = "ipiv";
static char *__pyx_k__work = "work";
static char *__pyx_k__dgeev = "dgeev";
static char *__pyx_k__dgesv = "dgesv";
static char *__pyx_k__iwork = "iwork";
static char *__pyx_k__rwork = "rwork";
static char *__pyx_k__zgeev = "zgeev";
static char *__pyx_k__zgesv = "zgesv";
static char *__pyx_k__dgelsd = "dgelsd";
static char *__pyx_k__dgeqrf = "dgeqrf";
static char *__pyx_k__dgesdd = "dgesdd";
static char *__pyx_k__dgetrf = "dgetrf";
static char *__pyx_k__dorgqr = "dorgqr";
static char *__pyx_k__dpotrf = "dpotrf";
static char *__pyx_k__dsyevd = "dsyevd";
static char *__pyx_k__zgelsd = "zgelsd";
static char *__pyx_k__zgeqrf = "zgeqrf";
static char *__pyx_k__zgesdd = "zgesdd";
static char *__pyx_k__zgetrf = "zgetrf";
static char *__pyx_k__zheevd = "zheevd";
static char *__pyx_k__zpotrf = "zpotrf";
static char *__pyx_k__zungqr = "zungqr";
/* Cython code section 'all_the_rest' */
public:
static System::String^ __module__ = __Pyx_MODULE_NAME;

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":35
 * 
 * 
 * cdef int check_object(np.ndarray ob, int t, char *obname, char *tname, char *funname):             # <<<<<<<<<<<<<<
 *     if not np.PyArray_CHKFLAGS(ob, np.NPY_CONTIGUOUS):
 *         raise LapackError("Parameter %s is not contiguous in lapack_lite.%s" % (obname, funname))
 */

static  int check_object(NumpyDotNet::ndarray^ __pyx_v_ob, int __pyx_v_t, char *__pyx_v_obname, char *__pyx_v_tname, char *__pyx_v_funname) {
  int __pyx_r;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  System::Object^ __pyx_t_4 = nullptr;
  System::Object^ __pyx_t_5 = nullptr;
  System::Object^ __pyx_t_6 = nullptr;
  int __pyx_t_7;
  int __pyx_t_8;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":36
 * 
 * cdef int check_object(np.ndarray ob, int t, char *obname, char *tname, char *funname):
 *     if not np.PyArray_CHKFLAGS(ob, np.NPY_CONTIGUOUS):             # <<<<<<<<<<<<<<
 *         raise LapackError("Parameter %s is not contiguous in lapack_lite.%s" % (obname, funname))
 *     elif np.PyArray_TYPE(ob) != t:
 */
  __pyx_t_1 = (!PyArray_CHKFLAGS(__pyx_v_ob, NPY_CONTIGUOUS));
  if (__pyx_t_1) {

    /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":37
 * cdef int check_object(np.ndarray ob, int t, char *obname, char *tname, char *funname):
 *     if not np.PyArray_CHKFLAGS(ob, np.NPY_CONTIGUOUS):
 *         raise LapackError("Parameter %s is not contiguous in lapack_lite.%s" % (obname, funname))             # <<<<<<<<<<<<<<
 *     elif np.PyArray_TYPE(ob) != t:
 *         raise LapackError("Parameter %s is not of type %s in lapack_lite.%s" % (obname, tname, funname))
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "LapackError");
    __pyx_t_3 = gcnew System::String(__pyx_v_obname);
    __pyx_t_4 = gcnew System::String(__pyx_v_funname);
    __pyx_t_5 = PythonOps::MakeTuple(gcnew array<System::Object^>{((System::Object^)__pyx_t_3), ((System::Object^)__pyx_t_4)});
    __pyx_t_3 = nullptr;
    __pyx_t_4 = nullptr;
    __pyx_t_4 = __site_op_mod_37_77->Target(__site_op_mod_37_77, ((System::Object^)"Parameter %s is not contiguous in lapack_lite.%s"), __pyx_t_5);
    __pyx_t_5 = nullptr;
    __pyx_t_5 = __site_call1_37_25->Target(__site_call1_37_25, __pyx_context, __pyx_t_2, ((System::Object^)__pyx_t_4));
    __pyx_t_2 = nullptr;
    __pyx_t_4 = nullptr;
    throw PythonOps::MakeException(__pyx_context, __pyx_t_5, nullptr, nullptr);
    __pyx_t_5 = nullptr;
    goto __pyx_L3;
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":38
 *     if not np.PyArray_CHKFLAGS(ob, np.NPY_CONTIGUOUS):
 *         raise LapackError("Parameter %s is not contiguous in lapack_lite.%s" % (obname, funname))
 *     elif np.PyArray_TYPE(ob) != t:             # <<<<<<<<<<<<<<
 *         raise LapackError("Parameter %s is not of type %s in lapack_lite.%s" % (obname, tname, funname))
 *     elif np.PyArray_DESCR(ob).byteorder != '=' and np.PyArray_DESCR(ob).byteorder != '|':
 */
  __pyx_t_1 = (PyArray_TYPE(__pyx_v_ob) != __pyx_v_t);
  if (__pyx_t_1) {

    /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":39
 *         raise LapackError("Parameter %s is not contiguous in lapack_lite.%s" % (obname, funname))
 *     elif np.PyArray_TYPE(ob) != t:
 *         raise LapackError("Parameter %s is not of type %s in lapack_lite.%s" % (obname, tname, funname))             # <<<<<<<<<<<<<<
 *     elif np.PyArray_DESCR(ob).byteorder != '=' and np.PyArray_DESCR(ob).byteorder != '|':
 *         raise LapackError("Parameter %s has non-native byte order in lapack_lite.%s" % (obname, funname))
 */
    __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "LapackError");
    __pyx_t_4 = gcnew System::String(__pyx_v_obname);
    __pyx_t_2 = gcnew System::String(__pyx_v_tname);
    __pyx_t_3 = gcnew System::String(__pyx_v_funname);
    __pyx_t_6 = PythonOps::MakeTuple(gcnew array<System::Object^>{((System::Object^)__pyx_t_4), ((System::Object^)__pyx_t_2), ((System::Object^)__pyx_t_3)});
    __pyx_t_4 = nullptr;
    __pyx_t_2 = nullptr;
    __pyx_t_3 = nullptr;
    __pyx_t_3 = __site_op_mod_39_77->Target(__site_op_mod_39_77, ((System::Object^)"Parameter %s is not of type %s in lapack_lite.%s"), __pyx_t_6);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_39_25->Target(__site_call1_39_25, __pyx_context, __pyx_t_5, ((System::Object^)__pyx_t_3));
    __pyx_t_5 = nullptr;
    __pyx_t_3 = nullptr;
    throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
    __pyx_t_6 = nullptr;
    goto __pyx_L3;
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":40
 *     elif np.PyArray_TYPE(ob) != t:
 *         raise LapackError("Parameter %s is not of type %s in lapack_lite.%s" % (obname, tname, funname))
 *     elif np.PyArray_DESCR(ob).byteorder != '=' and np.PyArray_DESCR(ob).byteorder != '|':             # <<<<<<<<<<<<<<
 *         raise LapackError("Parameter %s has non-native byte order in lapack_lite.%s" % (obname, funname))
 * 
 */
  __pyx_t_6 = PyArray_DESCR(__pyx_v_ob); 
  __pyx_t_3 = __site_get_byteorder_40_29->Target(__site_get_byteorder_40_29, __pyx_t_6, __pyx_context);
  __pyx_t_6 = nullptr;
  __pyx_t_6 = __site_op_ne_40_40->Target(__site_op_ne_40_40, __pyx_t_3, ((System::Object^)"="));
  __pyx_t_3 = nullptr;
  __pyx_t_1 = __site_istrue_40_40->Target(__site_istrue_40_40, __pyx_t_6);
  __pyx_t_6 = nullptr;
  if (__pyx_t_1) {
    __pyx_t_6 = PyArray_DESCR(__pyx_v_ob); 
    __pyx_t_3 = __site_get_byteorder_40_71->Target(__site_get_byteorder_40_71, __pyx_t_6, __pyx_context);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_op_ne_40_82->Target(__site_op_ne_40_82, __pyx_t_3, ((System::Object^)"|"));
    __pyx_t_3 = nullptr;
    __pyx_t_7 = __site_istrue_40_82->Target(__site_istrue_40_82, __pyx_t_6);
    __pyx_t_6 = nullptr;
    __pyx_t_8 = __pyx_t_7;
  } else {
    __pyx_t_8 = __pyx_t_1;
  }
  if (__pyx_t_8) {

    /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":41
 *         raise LapackError("Parameter %s is not of type %s in lapack_lite.%s" % (obname, tname, funname))
 *     elif np.PyArray_DESCR(ob).byteorder != '=' and np.PyArray_DESCR(ob).byteorder != '|':
 *         raise LapackError("Parameter %s has non-native byte order in lapack_lite.%s" % (obname, funname))             # <<<<<<<<<<<<<<
 * 
 *     return 1
 */
    __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "LapackError");
    __pyx_t_3 = gcnew System::String(__pyx_v_obname);
    __pyx_t_5 = gcnew System::String(__pyx_v_funname);
    __pyx_t_2 = PythonOps::MakeTuple(gcnew array<System::Object^>{((System::Object^)__pyx_t_3), ((System::Object^)__pyx_t_5)});
    __pyx_t_3 = nullptr;
    __pyx_t_5 = nullptr;
    __pyx_t_5 = __site_op_mod_41_85->Target(__site_op_mod_41_85, ((System::Object^)"Parameter %s has non-native byte order in lapack_lite.%s"), __pyx_t_2);
    __pyx_t_2 = nullptr;
    __pyx_t_2 = __site_call1_41_25->Target(__site_call1_41_25, __pyx_context, __pyx_t_6, ((System::Object^)__pyx_t_5));
    __pyx_t_6 = nullptr;
    __pyx_t_5 = nullptr;
    throw PythonOps::MakeException(__pyx_context, __pyx_t_2, nullptr, nullptr);
    __pyx_t_2 = nullptr;
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":43
 *         raise LapackError("Parameter %s has non-native byte order in lapack_lite.%s" % (obname, funname))
 * 
 *     return 1             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = 1;
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":46
 * 
 * 
 * def dgeev(jobvl, jobvr, int n, np.ndarray a, int lda,             # <<<<<<<<<<<<<<
 *            np.ndarray wr, np.ndarray wi, np.ndarray vl,
 *            int ldvl, np.ndarray vr, int ldvr, np.ndarray work, int lwork, int info):
 */

static System::Object^ dgeev(System::Object^ jobvl, System::Object^ jobvr, System::Object^ n, System::Object^ a, System::Object^ lda, System::Object^ wr, System::Object^ wi, System::Object^ vl, System::Object^ ldvl, System::Object^ vr, System::Object^ ldvr, System::Object^ work, System::Object^ lwork, System::Object^ info) {
  System::Object^ __pyx_v_jobvl = nullptr;
  System::Object^ __pyx_v_jobvr = nullptr;
  int __pyx_v_n;
  NumpyDotNet::ndarray^ __pyx_v_a = nullptr;
  int __pyx_v_lda;
  NumpyDotNet::ndarray^ __pyx_v_wr = nullptr;
  NumpyDotNet::ndarray^ __pyx_v_wi = nullptr;
  NumpyDotNet::ndarray^ __pyx_v_vl = nullptr;
  int __pyx_v_ldvl;
  NumpyDotNet::ndarray^ __pyx_v_vr = nullptr;
  int __pyx_v_ldvr;
  NumpyDotNet::ndarray^ __pyx_v_work = nullptr;
  int __pyx_v_lwork;
  int __pyx_v_info;
  int __pyx_v_lapack_lite_status__;
  char __pyx_v_jobvl_char;
  char __pyx_v_jobvr_char;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  char __pyx_t_4;
  char __pyx_t_5;
  int __pyx_t_6;
  PythonDictionary^ __pyx_t_7;
  __pyx_v_jobvl = jobvl;
  __pyx_v_jobvr = jobvr;
  __pyx_v_n = __site_cvt_cvt_int_46_0->Target(__site_cvt_cvt_int_46_0, n);
  __pyx_v_a = ((NumpyDotNet::ndarray^)a);
  __pyx_v_lda = __site_cvt_cvt_int_46_0_1->Target(__site_cvt_cvt_int_46_0_1, lda);
  __pyx_v_wr = ((NumpyDotNet::ndarray^)wr);
  __pyx_v_wi = ((NumpyDotNet::ndarray^)wi);
  __pyx_v_vl = ((NumpyDotNet::ndarray^)vl);
  __pyx_v_ldvl = __site_cvt_cvt_int_46_0_2->Target(__site_cvt_cvt_int_46_0_2, ldvl);
  __pyx_v_vr = ((NumpyDotNet::ndarray^)vr);
  __pyx_v_ldvr = __site_cvt_cvt_int_46_0_3->Target(__site_cvt_cvt_int_46_0_3, ldvr);
  __pyx_v_work = ((NumpyDotNet::ndarray^)work);
  __pyx_v_lwork = __site_cvt_cvt_int_46_0_4->Target(__site_cvt_cvt_int_46_0_4, lwork);
  __pyx_v_info = __site_cvt_cvt_int_46_0_5->Target(__site_cvt_cvt_int_46_0_5, info);
  __pyx_v_retval = nullptr;
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr)) {
    throw PythonOps::TypeError("Argument 'a' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_wr) == nullptr)) {
    throw PythonOps::TypeError("Argument 'wr' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_wi) == nullptr)) {
    throw PythonOps::TypeError("Argument 'wi' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_vl) == nullptr)) {
    throw PythonOps::TypeError("Argument 'vl' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_vr) == nullptr)) {
    throw PythonOps::TypeError("Argument 'vr' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr)) {
    throw PythonOps::TypeError("Argument 'work' has incorrect type");
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":50
 *            int ldvl, np.ndarray vr, int ldvr, np.ndarray work, int lwork, int info):
 *     cdef int lapack_lite_status__
 *     cdef char jobvl_char = ord(jobvl[0])             # <<<<<<<<<<<<<<
 *     cdef char jobvr_char = ord(jobvr[0])
 * 
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "ord");
  __pyx_t_2 = __site_getindex_50_36->Target(__site_getindex_50_36, __pyx_v_jobvl, ((System::Object^)0));
  __pyx_t_3 = __site_call1_50_30->Target(__site_call1_50_30, __pyx_context, __pyx_t_1, __pyx_t_2);
  __pyx_t_1 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_t_4 = __site_cvt_cvt_char_50_30->Target(__site_cvt_cvt_char_50_30, __pyx_t_3);
  __pyx_t_3 = nullptr;
  __pyx_v_jobvl_char = __pyx_t_4;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":51
 *     cdef int lapack_lite_status__
 *     cdef char jobvl_char = ord(jobvl[0])
 *     cdef char jobvr_char = ord(jobvr[0])             # <<<<<<<<<<<<<<
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgeev"): return None
 */
  __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "ord");
  __pyx_t_2 = __site_getindex_51_36->Target(__site_getindex_51_36, __pyx_v_jobvr, ((System::Object^)0));
  __pyx_t_1 = __site_call1_51_30->Target(__site_call1_51_30, __pyx_context, __pyx_t_3, __pyx_t_2);
  __pyx_t_3 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_t_5 = __site_cvt_cvt_char_51_30->Target(__site_cvt_cvt_char_51_30, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_v_jobvr_char = __pyx_t_5;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":53
 *     cdef char jobvr_char = ord(jobvr[0])
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(wr,np.NPY_DOUBLE,"wr","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(wi,np.NPY_DOUBLE,"wi","np.NPY_DOUBLE","dgeev"): return None
 */
  __pyx_t_6 = (!check_object(__pyx_v_a, NPY_DOUBLE, __pyx_k__a, __pyx_k_1, __pyx_k__dgeev));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":54
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(wr,np.NPY_DOUBLE,"wr","np.NPY_DOUBLE","dgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(wi,np.NPY_DOUBLE,"wi","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(vl,np.NPY_DOUBLE,"vl","np.NPY_DOUBLE","dgeev"): return None
 */
  __pyx_t_6 = (!check_object(__pyx_v_wr, NPY_DOUBLE, __pyx_k__wr, __pyx_k_1, __pyx_k__dgeev));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":55
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(wr,np.NPY_DOUBLE,"wr","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(wi,np.NPY_DOUBLE,"wi","np.NPY_DOUBLE","dgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(vl,np.NPY_DOUBLE,"vl","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(vr,np.NPY_DOUBLE,"vr","np.NPY_DOUBLE","dgeev"): return None
 */
  __pyx_t_6 = (!check_object(__pyx_v_wi, NPY_DOUBLE, __pyx_k__wi, __pyx_k_1, __pyx_k__dgeev));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":56
 *     if not check_object(wr,np.NPY_DOUBLE,"wr","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(wi,np.NPY_DOUBLE,"wi","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(vl,np.NPY_DOUBLE,"vl","np.NPY_DOUBLE","dgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(vr,np.NPY_DOUBLE,"vr","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgeev"): return None
 */
  __pyx_t_6 = (!check_object(__pyx_v_vl, NPY_DOUBLE, __pyx_k__vl, __pyx_k_1, __pyx_k__dgeev));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L8;
  }
  __pyx_L8:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":57
 *     if not check_object(wi,np.NPY_DOUBLE,"wi","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(vl,np.NPY_DOUBLE,"vl","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(vr,np.NPY_DOUBLE,"vr","np.NPY_DOUBLE","dgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgeev"): return None
 * 
 */
  __pyx_t_6 = (!check_object(__pyx_v_vr, NPY_DOUBLE, __pyx_k__vr, __pyx_k_1, __pyx_k__dgeev));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L9;
  }
  __pyx_L9:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":58
 *     if not check_object(vl,np.NPY_DOUBLE,"vl","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(vr,np.NPY_DOUBLE,"vr","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgeev"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_dgeev(&jobvl_char,&jobvr_char,&n,
 */
  __pyx_t_6 = (!check_object(__pyx_v_work, NPY_DOUBLE, __pyx_k__work, __pyx_k_1, __pyx_k__dgeev));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L10;
  }
  __pyx_L10:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":67
 *                                         <double *>np.PyArray_DATA(vr),&ldvr,
 *                                         <double *>np.PyArray_DATA(work),&lwork,
 *                                         &info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(dgeev_)((&__pyx_v_jobvl_char), (&__pyx_v_jobvr_char), (&__pyx_v_n), ((double *)PyArray_DATA(__pyx_v_a)), (&__pyx_v_lda), ((double *)PyArray_DATA(__pyx_v_wr)), ((double *)PyArray_DATA(__pyx_v_wi)), ((double *)PyArray_DATA(__pyx_v_vl)), (&__pyx_v_ldvl), ((double *)PyArray_DATA(__pyx_v_vr)), (&__pyx_v_ldvr), ((double *)PyArray_DATA(__pyx_v_work)), (&__pyx_v_lwork), (&__pyx_v_info));

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":69
 *                                         &info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["dgeev_"] = lapack_lite_status__
 *     retval["jobvl"] = jobvl_char
 */
  __pyx_t_7 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_7;
  __pyx_t_7 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":70
 * 
 *     retval = {}
 *     retval["dgeev_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["jobvl"] = jobvl_char
 *     retval["jobvr"] = jobvr_char
 */
  __pyx_t_1 = __pyx_v_lapack_lite_status__;
  __site_setindex_70_10->Target(__site_setindex_70_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"dgeev_"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":71
 *     retval = {}
 *     retval["dgeev_"] = lapack_lite_status__
 *     retval["jobvl"] = jobvl_char             # <<<<<<<<<<<<<<
 *     retval["jobvr"] = jobvr_char
 *     retval["n"] = n
 */
  __pyx_t_1 = __pyx_v_jobvl_char;
  __site_setindex_71_10->Target(__site_setindex_71_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"jobvl"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":72
 *     retval["dgeev_"] = lapack_lite_status__
 *     retval["jobvl"] = jobvl_char
 *     retval["jobvr"] = jobvr_char             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_1 = __pyx_v_jobvr_char;
  __site_setindex_72_10->Target(__site_setindex_72_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"jobvr"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":73
 *     retval["jobvl"] = jobvl_char
 *     retval["jobvr"] = jobvr_char
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["ldvl"] = ldvl
 */
  __pyx_t_1 = __pyx_v_n;
  __site_setindex_73_10->Target(__site_setindex_73_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":74
 *     retval["jobvr"] = jobvr_char
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["ldvl"] = ldvl
 *     retval["ldvr"] = ldvr
 */
  __pyx_t_1 = __pyx_v_lda;
  __site_setindex_74_10->Target(__site_setindex_74_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":75
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["ldvl"] = ldvl             # <<<<<<<<<<<<<<
 *     retval["ldvr"] = ldvr
 *     retval["lwork"] = lwork
 */
  __pyx_t_1 = __pyx_v_ldvl;
  __site_setindex_75_10->Target(__site_setindex_75_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldvl"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":76
 *     retval["lda"] = lda
 *     retval["ldvl"] = ldvl
 *     retval["ldvr"] = ldvr             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 */
  __pyx_t_1 = __pyx_v_ldvr;
  __site_setindex_76_10->Target(__site_setindex_76_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldvr"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":77
 *     retval["ldvl"] = ldvl
 *     retval["ldvr"] = ldvr
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 * 
 */
  __pyx_t_1 = __pyx_v_lwork;
  __site_setindex_77_10->Target(__site_setindex_77_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":78
 *     retval["ldvr"] = ldvr
 *     retval["lwork"] = lwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 * 
 *     return retval
 */
  __pyx_t_1 = __pyx_v_info;
  __site_setindex_78_10->Target(__site_setindex_78_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":80
 *     retval["info"] = info
 * 
 *     return retval             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = ((System::Object^)__pyx_v_retval);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":83
 * 
 * 
 * def dsyevd(jobz, uplo, int n, np.ndarray a, int lda,             # <<<<<<<<<<<<<<
 *             np.ndarray w, np.ndarray work, int lwork, np.ndarray iwork, int liwork, int info):
 *     """ Arguments
 */

static System::Object^ dsyevd(System::Object^ jobz, System::Object^ uplo, System::Object^ n, System::Object^ a, System::Object^ lda, System::Object^ w, System::Object^ work, System::Object^ lwork, System::Object^ iwork, System::Object^ liwork, System::Object^ info) {
  System::Object^ __pyx_v_jobz = nullptr;
  System::Object^ __pyx_v_uplo = nullptr;
  int __pyx_v_n;
  NumpyDotNet::ndarray^ __pyx_v_a = nullptr;
  int __pyx_v_lda;
  NumpyDotNet::ndarray^ __pyx_v_w = nullptr;
  NumpyDotNet::ndarray^ __pyx_v_work = nullptr;
  int __pyx_v_lwork;
  NumpyDotNet::ndarray^ __pyx_v_iwork = nullptr;
  int __pyx_v_liwork;
  int __pyx_v_info;
  int __pyx_v_lapack_lite_status__;
  char __pyx_v_jobz_char;
  char __pyx_v_uplo_char;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  char __pyx_t_4;
  char __pyx_t_5;
  int __pyx_t_6;
  PythonDictionary^ __pyx_t_7;
  __pyx_v_jobz = jobz;
  __pyx_v_uplo = uplo;
  __pyx_v_n = __site_cvt_cvt_int_83_0->Target(__site_cvt_cvt_int_83_0, n);
  __pyx_v_a = ((NumpyDotNet::ndarray^)a);
  __pyx_v_lda = __site_cvt_cvt_int_83_0_1->Target(__site_cvt_cvt_int_83_0_1, lda);
  __pyx_v_w = ((NumpyDotNet::ndarray^)w);
  __pyx_v_work = ((NumpyDotNet::ndarray^)work);
  __pyx_v_lwork = __site_cvt_cvt_int_83_0_2->Target(__site_cvt_cvt_int_83_0_2, lwork);
  __pyx_v_iwork = ((NumpyDotNet::ndarray^)iwork);
  __pyx_v_liwork = __site_cvt_cvt_int_83_0_3->Target(__site_cvt_cvt_int_83_0_3, liwork);
  __pyx_v_info = __site_cvt_cvt_int_83_0_4->Target(__site_cvt_cvt_int_83_0_4, info);
  __pyx_v_retval = nullptr;
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr)) {
    throw PythonOps::TypeError("Argument 'a' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_w) == nullptr)) {
    throw PythonOps::TypeError("Argument 'w' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr)) {
    throw PythonOps::TypeError("Argument 'work' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_iwork) == nullptr)) {
    throw PythonOps::TypeError("Argument 'iwork' has incorrect type");
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":124
 *     """
 *     cdef int lapack_lite_status__
 *     cdef char jobz_char = ord(jobz[0])             # <<<<<<<<<<<<<<
 *     cdef char uplo_char = ord(uplo[0])
 * 
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "ord");
  __pyx_t_2 = __site_getindex_124_34->Target(__site_getindex_124_34, __pyx_v_jobz, ((System::Object^)0));
  __pyx_t_3 = __site_call1_124_29->Target(__site_call1_124_29, __pyx_context, __pyx_t_1, __pyx_t_2);
  __pyx_t_1 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_t_4 = __site_cvt_cvt_char_124_29->Target(__site_cvt_cvt_char_124_29, __pyx_t_3);
  __pyx_t_3 = nullptr;
  __pyx_v_jobz_char = __pyx_t_4;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":125
 *     cdef int lapack_lite_status__
 *     cdef char jobz_char = ord(jobz[0])
 *     cdef char uplo_char = ord(uplo[0])             # <<<<<<<<<<<<<<
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dsyevd"): return None
 */
  __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "ord");
  __pyx_t_2 = __site_getindex_125_34->Target(__site_getindex_125_34, __pyx_v_uplo, ((System::Object^)0));
  __pyx_t_1 = __site_call1_125_29->Target(__site_call1_125_29, __pyx_context, __pyx_t_3, __pyx_t_2);
  __pyx_t_3 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_t_5 = __site_cvt_cvt_char_125_29->Target(__site_cvt_cvt_char_125_29, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_v_uplo_char = __pyx_t_5;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":127
 *     cdef char uplo_char = ord(uplo[0])
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dsyevd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(w,np.NPY_DOUBLE,"w","np.NPY_DOUBLE","dsyevd"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dsyevd"): return None
 */
  __pyx_t_6 = (!check_object(__pyx_v_a, NPY_DOUBLE, __pyx_k__a, __pyx_k_1, __pyx_k__dsyevd));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":128
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dsyevd"): return None
 *     if not check_object(w,np.NPY_DOUBLE,"w","np.NPY_DOUBLE","dsyevd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dsyevd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","dsyevd"): return None
 */
  __pyx_t_6 = (!check_object(__pyx_v_w, NPY_DOUBLE, __pyx_k__w, __pyx_k_1, __pyx_k__dsyevd));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":129
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dsyevd"): return None
 *     if not check_object(w,np.NPY_DOUBLE,"w","np.NPY_DOUBLE","dsyevd"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dsyevd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","dsyevd"): return None
 * 
 */
  __pyx_t_6 = (!check_object(__pyx_v_work, NPY_DOUBLE, __pyx_k__work, __pyx_k_1, __pyx_k__dsyevd));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":130
 *     if not check_object(w,np.NPY_DOUBLE,"w","np.NPY_DOUBLE","dsyevd"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dsyevd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","dsyevd"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_dsyevd(&jobz_char,&uplo_char,&n,
 */
  __pyx_t_6 = (!check_object(__pyx_v_iwork, NPY_INT, __pyx_k__iwork, __pyx_k_2, __pyx_k__dsyevd));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L8;
  }
  __pyx_L8:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":136
 *                                          <double *>np.PyArray_DATA(w),
 *                                          <double *>np.PyArray_DATA(work),&lwork,
 *                                          <int *>np.PyArray_DATA(iwork),&liwork,&info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(dsyevd_)((&__pyx_v_jobz_char), (&__pyx_v_uplo_char), (&__pyx_v_n), ((double *)PyArray_DATA(__pyx_v_a)), (&__pyx_v_lda), ((double *)PyArray_DATA(__pyx_v_w)), ((double *)PyArray_DATA(__pyx_v_work)), (&__pyx_v_lwork), ((int *)PyArray_DATA(__pyx_v_iwork)), (&__pyx_v_liwork), (&__pyx_v_info));

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":138
 *                                          <int *>np.PyArray_DATA(iwork),&liwork,&info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["dsyevd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz_char
 */
  __pyx_t_7 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_7;
  __pyx_t_7 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":139
 * 
 *     retval = {}
 *     retval["dsyevd_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["jobz"] = jobz_char
 *     retval["uplo"] = uplo_char
 */
  __pyx_t_1 = __pyx_v_lapack_lite_status__;
  __site_setindex_139_10->Target(__site_setindex_139_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"dsyevd_"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":140
 *     retval = {}
 *     retval["dsyevd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz_char             # <<<<<<<<<<<<<<
 *     retval["uplo"] = uplo_char
 *     retval["n"] = n
 */
  __pyx_t_1 = __pyx_v_jobz_char;
  __site_setindex_140_10->Target(__site_setindex_140_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"jobz"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":141
 *     retval["dsyevd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz_char
 *     retval["uplo"] = uplo_char             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_1 = __pyx_v_uplo_char;
  __site_setindex_141_10->Target(__site_setindex_141_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"uplo"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":142
 *     retval["jobz"] = jobz_char
 *     retval["uplo"] = uplo_char
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork
 */
  __pyx_t_1 = __pyx_v_n;
  __site_setindex_142_10->Target(__site_setindex_142_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":143
 *     retval["uplo"] = uplo_char
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["liwork"] = liwork
 */
  __pyx_t_1 = __pyx_v_lda;
  __site_setindex_143_10->Target(__site_setindex_143_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":144
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["liwork"] = liwork
 *     retval["info"] = info
 */
  __pyx_t_1 = __pyx_v_lwork;
  __site_setindex_144_10->Target(__site_setindex_144_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":145
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork
 *     retval["liwork"] = liwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_1 = __pyx_v_liwork;
  __site_setindex_145_10->Target(__site_setindex_145_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"liwork"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":146
 *     retval["lwork"] = lwork
 *     retval["liwork"] = liwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_1 = __pyx_v_info;
  __site_setindex_146_10->Target(__site_setindex_146_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":147
 *     retval["liwork"] = liwork
 *     retval["info"] = info
 *     return retval             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = ((System::Object^)__pyx_v_retval);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":150
 * 
 * 
 * def zheevd(jobz, uplo, int n, np.ndarray a, int lda,             # <<<<<<<<<<<<<<
 *             np.ndarray w, np.ndarray work, int lwork,
 *             np.ndarray rwork, int lrwork,
 */

static System::Object^ zheevd(System::Object^ jobz, System::Object^ uplo, System::Object^ n, System::Object^ a, System::Object^ lda, System::Object^ w, System::Object^ work, System::Object^ lwork, System::Object^ rwork, System::Object^ lrwork, System::Object^ iwork, System::Object^ liwork, System::Object^ info) {
  System::Object^ __pyx_v_jobz = nullptr;
  System::Object^ __pyx_v_uplo = nullptr;
  int __pyx_v_n;
  NumpyDotNet::ndarray^ __pyx_v_a = nullptr;
  int __pyx_v_lda;
  NumpyDotNet::ndarray^ __pyx_v_w = nullptr;
  NumpyDotNet::ndarray^ __pyx_v_work = nullptr;
  int __pyx_v_lwork;
  NumpyDotNet::ndarray^ __pyx_v_rwork = nullptr;
  int __pyx_v_lrwork;
  NumpyDotNet::ndarray^ __pyx_v_iwork = nullptr;
  int __pyx_v_liwork;
  int __pyx_v_info;
  int __pyx_v_lapack_lite_status__;
  char __pyx_v_jobz_char;
  char __pyx_v_uplo_char;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  char __pyx_t_4;
  char __pyx_t_5;
  int __pyx_t_6;
  PythonDictionary^ __pyx_t_7;
  __pyx_v_jobz = jobz;
  __pyx_v_uplo = uplo;
  __pyx_v_n = __site_cvt_cvt_int_150_0->Target(__site_cvt_cvt_int_150_0, n);
  __pyx_v_a = ((NumpyDotNet::ndarray^)a);
  __pyx_v_lda = __site_cvt_cvt_int_150_0_1->Target(__site_cvt_cvt_int_150_0_1, lda);
  __pyx_v_w = ((NumpyDotNet::ndarray^)w);
  __pyx_v_work = ((NumpyDotNet::ndarray^)work);
  __pyx_v_lwork = __site_cvt_cvt_int_150_0_2->Target(__site_cvt_cvt_int_150_0_2, lwork);
  __pyx_v_rwork = ((NumpyDotNet::ndarray^)rwork);
  __pyx_v_lrwork = __site_cvt_cvt_int_150_0_3->Target(__site_cvt_cvt_int_150_0_3, lrwork);
  __pyx_v_iwork = ((NumpyDotNet::ndarray^)iwork);
  __pyx_v_liwork = __site_cvt_cvt_int_150_0_4->Target(__site_cvt_cvt_int_150_0_4, liwork);
  __pyx_v_info = __site_cvt_cvt_int_150_0_5->Target(__site_cvt_cvt_int_150_0_5, info);
  __pyx_v_retval = nullptr;
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr)) {
    throw PythonOps::TypeError("Argument 'a' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_w) == nullptr)) {
    throw PythonOps::TypeError("Argument 'w' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr)) {
    throw PythonOps::TypeError("Argument 'work' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_rwork) == nullptr)) {
    throw PythonOps::TypeError("Argument 'rwork' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_iwork) == nullptr)) {
    throw PythonOps::TypeError("Argument 'iwork' has incorrect type");
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":194
 *     """
 *     cdef int lapack_lite_status__
 *     cdef char jobz_char = ord(jobz[0])             # <<<<<<<<<<<<<<
 *     cdef char uplo_char = ord(uplo[0])
 * 
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "ord");
  __pyx_t_2 = __site_getindex_194_34->Target(__site_getindex_194_34, __pyx_v_jobz, ((System::Object^)0));
  __pyx_t_3 = __site_call1_194_29->Target(__site_call1_194_29, __pyx_context, __pyx_t_1, __pyx_t_2);
  __pyx_t_1 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_t_4 = __site_cvt_cvt_char_194_29->Target(__site_cvt_cvt_char_194_29, __pyx_t_3);
  __pyx_t_3 = nullptr;
  __pyx_v_jobz_char = __pyx_t_4;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":195
 *     cdef int lapack_lite_status__
 *     cdef char jobz_char = ord(jobz[0])
 *     cdef char uplo_char = ord(uplo[0])             # <<<<<<<<<<<<<<
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zheevd"): return None
 */
  __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "ord");
  __pyx_t_2 = __site_getindex_195_34->Target(__site_getindex_195_34, __pyx_v_uplo, ((System::Object^)0));
  __pyx_t_1 = __site_call1_195_29->Target(__site_call1_195_29, __pyx_context, __pyx_t_3, __pyx_t_2);
  __pyx_t_3 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_t_5 = __site_cvt_cvt_char_195_29->Target(__site_cvt_cvt_char_195_29, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_v_uplo_char = __pyx_t_5;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":197
 *     cdef char uplo_char = ord(uplo[0])
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zheevd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(w,np.NPY_DOUBLE,"w","np.NPY_DOUBLE","zheevd"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zheevd"): return None
 */
  __pyx_t_6 = (!check_object(__pyx_v_a, NPY_CDOUBLE, __pyx_k__a, __pyx_k_3, __pyx_k__zheevd));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":198
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zheevd"): return None
 *     if not check_object(w,np.NPY_DOUBLE,"w","np.NPY_DOUBLE","zheevd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zheevd"): return None
 *     if not check_object(w,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zheevd"): return None
 */
  __pyx_t_6 = (!check_object(__pyx_v_w, NPY_DOUBLE, __pyx_k__w, __pyx_k_1, __pyx_k__zheevd));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":199
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zheevd"): return None
 *     if not check_object(w,np.NPY_DOUBLE,"w","np.NPY_DOUBLE","zheevd"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zheevd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(w,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zheevd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","zheevd"): return None
 */
  __pyx_t_6 = (!check_object(__pyx_v_work, NPY_CDOUBLE, __pyx_k__work, __pyx_k_3, __pyx_k__zheevd));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":200
 *     if not check_object(w,np.NPY_DOUBLE,"w","np.NPY_DOUBLE","zheevd"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zheevd"): return None
 *     if not check_object(w,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zheevd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","zheevd"): return None
 * 
 */
  __pyx_t_6 = (!check_object(__pyx_v_w, NPY_DOUBLE, __pyx_k__rwork, __pyx_k_1, __pyx_k__zheevd));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L8;
  }
  __pyx_L8:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":201
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zheevd"): return None
 *     if not check_object(w,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zheevd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","zheevd"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_zheevd(&jobz_char,&uplo_char,&n,
 */
  __pyx_t_6 = (!check_object(__pyx_v_iwork, NPY_INT, __pyx_k__iwork, __pyx_k_2, __pyx_k__zheevd));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L9;
  }
  __pyx_L9:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":208
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 *                                          <double *>np.PyArray_DATA(rwork),&lrwork,
 *                                          <int *>np.PyArray_DATA(iwork),&liwork,&info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(zheevd_)((&__pyx_v_jobz_char), (&__pyx_v_uplo_char), (&__pyx_v_n), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_a)), (&__pyx_v_lda), ((double *)PyArray_DATA(__pyx_v_w)), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_work)), (&__pyx_v_lwork), ((double *)PyArray_DATA(__pyx_v_rwork)), (&__pyx_v_lrwork), ((int *)PyArray_DATA(__pyx_v_iwork)), (&__pyx_v_liwork), (&__pyx_v_info));

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":210
 *                                          <int *>np.PyArray_DATA(iwork),&liwork,&info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["zheevd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz_char
 */
  __pyx_t_7 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_7;
  __pyx_t_7 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":211
 * 
 *     retval = {}
 *     retval["zheevd_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["jobz"] = jobz_char
 *     retval["uplo"] = uplo_char
 */
  __pyx_t_1 = __pyx_v_lapack_lite_status__;
  __site_setindex_211_10->Target(__site_setindex_211_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"zheevd_"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":212
 *     retval = {}
 *     retval["zheevd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz_char             # <<<<<<<<<<<<<<
 *     retval["uplo"] = uplo_char
 *     retval["n"] = n
 */
  __pyx_t_1 = __pyx_v_jobz_char;
  __site_setindex_212_10->Target(__site_setindex_212_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"jobz"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":213
 *     retval["zheevd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz_char
 *     retval["uplo"] = uplo_char             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_1 = __pyx_v_uplo_char;
  __site_setindex_213_10->Target(__site_setindex_213_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"uplo"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":214
 *     retval["jobz"] = jobz_char
 *     retval["uplo"] = uplo_char
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork
 */
  __pyx_t_1 = __pyx_v_n;
  __site_setindex_214_10->Target(__site_setindex_214_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":215
 *     retval["uplo"] = uplo_char
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["lrwork"] = lrwork
 */
  __pyx_t_1 = __pyx_v_lda;
  __site_setindex_215_10->Target(__site_setindex_215_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":216
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["lrwork"] = lrwork
 *     retval["liwork"] = liwork
 */
  __pyx_t_1 = __pyx_v_lwork;
  __site_setindex_216_10->Target(__site_setindex_216_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":217
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork
 *     retval["lrwork"] = lrwork             # <<<<<<<<<<<<<<
 *     retval["liwork"] = liwork
 *     retval["info"] = info
 */
  __pyx_t_1 = __pyx_v_lrwork;
  __site_setindex_217_10->Target(__site_setindex_217_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lrwork"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":218
 *     retval["lwork"] = lwork
 *     retval["lrwork"] = lrwork
 *     retval["liwork"] = liwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_1 = __pyx_v_liwork;
  __site_setindex_218_10->Target(__site_setindex_218_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"liwork"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":219
 *     retval["lrwork"] = lrwork
 *     retval["liwork"] = liwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_1 = __pyx_v_info;
  __site_setindex_219_10->Target(__site_setindex_219_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":220
 *     retval["liwork"] = liwork
 *     retval["info"] = info
 *     return retval             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = ((System::Object^)__pyx_v_retval);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":223
 * 
 * 
 * def dgelsd(int m, int n, int nrhs, np.ndarray a, int lda, np.ndarray b, int ldb,             # <<<<<<<<<<<<<<
 *             np.ndarray s, double rcond, int rank,
 *             np.ndarray work, int lwork, np.ndarray iwork, int info):
 */

static System::Object^ dgelsd(System::Object^ m, System::Object^ n, System::Object^ nrhs, System::Object^ a, System::Object^ lda, System::Object^ b, System::Object^ ldb, System::Object^ s, System::Object^ rcond, System::Object^ rank, System::Object^ work, System::Object^ lwork, System::Object^ iwork, System::Object^ info) {
  int __pyx_v_m;
  int __pyx_v_n;
  int __pyx_v_nrhs;
  NumpyDotNet::ndarray^ __pyx_v_a = nullptr;
  int __pyx_v_lda;
  NumpyDotNet::ndarray^ __pyx_v_b = nullptr;
  int __pyx_v_ldb;
  NumpyDotNet::ndarray^ __pyx_v_s = nullptr;
  double __pyx_v_rcond;
  int __pyx_v_rank;
  NumpyDotNet::ndarray^ __pyx_v_work = nullptr;
  int __pyx_v_lwork;
  NumpyDotNet::ndarray^ __pyx_v_iwork = nullptr;
  int __pyx_v_info;
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  PythonDictionary^ __pyx_t_2;
  System::Object^ __pyx_t_3 = nullptr;
  __pyx_v_m = __site_cvt_cvt_int_223_0->Target(__site_cvt_cvt_int_223_0, m);
  __pyx_v_n = __site_cvt_cvt_int_223_0_1->Target(__site_cvt_cvt_int_223_0_1, n);
  __pyx_v_nrhs = __site_cvt_cvt_int_223_0_2->Target(__site_cvt_cvt_int_223_0_2, nrhs);
  __pyx_v_a = ((NumpyDotNet::ndarray^)a);
  __pyx_v_lda = __site_cvt_cvt_int_223_0_3->Target(__site_cvt_cvt_int_223_0_3, lda);
  __pyx_v_b = ((NumpyDotNet::ndarray^)b);
  __pyx_v_ldb = __site_cvt_cvt_int_223_0_4->Target(__site_cvt_cvt_int_223_0_4, ldb);
  __pyx_v_s = ((NumpyDotNet::ndarray^)s);
  __pyx_v_rcond = __site_cvt_cvt_double_223_0->Target(__site_cvt_cvt_double_223_0, rcond);
  __pyx_v_rank = __site_cvt_cvt_int_223_0_5->Target(__site_cvt_cvt_int_223_0_5, rank);
  __pyx_v_work = ((NumpyDotNet::ndarray^)work);
  __pyx_v_lwork = __site_cvt_cvt_int_223_0_6->Target(__site_cvt_cvt_int_223_0_6, lwork);
  __pyx_v_iwork = ((NumpyDotNet::ndarray^)iwork);
  __pyx_v_info = __site_cvt_cvt_int_223_0_7->Target(__site_cvt_cvt_int_223_0_7, info);
  __pyx_v_retval = nullptr;
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr)) {
    throw PythonOps::TypeError("Argument 'a' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_b) == nullptr)) {
    throw PythonOps::TypeError("Argument 'b' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_s) == nullptr)) {
    throw PythonOps::TypeError("Argument 's' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr)) {
    throw PythonOps::TypeError("Argument 'work' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_iwork) == nullptr)) {
    throw PythonOps::TypeError("Argument 'iwork' has incorrect type");
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":228
 *     cdef int lapack_lite_status__
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgelsd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(b,np.NPY_DOUBLE,"b","np.NPY_DOUBLE","dgelsd"): return None
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","dgelsd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_a, NPY_DOUBLE, __pyx_k__a, __pyx_k_1, __pyx_k__dgelsd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":229
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgelsd"): return None
 *     if not check_object(b,np.NPY_DOUBLE,"b","np.NPY_DOUBLE","dgelsd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","dgelsd"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgelsd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_b, NPY_DOUBLE, __pyx_k__b, __pyx_k_1, __pyx_k__dgelsd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":230
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgelsd"): return None
 *     if not check_object(b,np.NPY_DOUBLE,"b","np.NPY_DOUBLE","dgelsd"): return None
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","dgelsd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgelsd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","dgelsd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_s, NPY_DOUBLE, __pyx_k__s, __pyx_k_1, __pyx_k__dgelsd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":231
 *     if not check_object(b,np.NPY_DOUBLE,"b","np.NPY_DOUBLE","dgelsd"): return None
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","dgelsd"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgelsd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","dgelsd"): return None
 * 
 */
  __pyx_t_1 = (!check_object(__pyx_v_work, NPY_DOUBLE, __pyx_k__work, __pyx_k_1, __pyx_k__dgelsd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L8;
  }
  __pyx_L8:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":232
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","dgelsd"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgelsd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","dgelsd"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_dgelsd(&m,&n,&nrhs,
 */
  __pyx_t_1 = (!check_object(__pyx_v_iwork, NPY_INT, __pyx_k__iwork, __pyx_k_2, __pyx_k__dgelsd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L9;
  }
  __pyx_L9:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":239
 *                                          <double *>np.PyArray_DATA(s),&rcond,&rank,
 *                                          <double *>np.PyArray_DATA(work),&lwork,
 *                                          <int *>np.PyArray_DATA(iwork),&info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(dgelsd_)((&__pyx_v_m), (&__pyx_v_n), (&__pyx_v_nrhs), ((double *)PyArray_DATA(__pyx_v_a)), (&__pyx_v_lda), ((double *)PyArray_DATA(__pyx_v_b)), (&__pyx_v_ldb), ((double *)PyArray_DATA(__pyx_v_s)), (&__pyx_v_rcond), (&__pyx_v_rank), ((double *)PyArray_DATA(__pyx_v_work)), (&__pyx_v_lwork), ((int *)PyArray_DATA(__pyx_v_iwork)), (&__pyx_v_info));

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":241
 *                                          <int *>np.PyArray_DATA(iwork),&info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["dgelsd_"] = lapack_lite_status__
 *     retval["m"] = m
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":242
 * 
 *     retval = {}
 *     retval["dgelsd_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["m"] = m
 *     retval["n"] = n
 */
  __pyx_t_3 = __pyx_v_lapack_lite_status__;
  __site_setindex_242_10->Target(__site_setindex_242_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"dgelsd_"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":243
 *     retval = {}
 *     retval["dgelsd_"] = lapack_lite_status__
 *     retval["m"] = m             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs
 */
  __pyx_t_3 = __pyx_v_m;
  __site_setindex_243_10->Target(__site_setindex_243_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"m"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":244
 *     retval["dgelsd_"] = lapack_lite_status__
 *     retval["m"] = m
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda
 */
  __pyx_t_3 = __pyx_v_n;
  __site_setindex_244_10->Target(__site_setindex_244_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":245
 *     retval["m"] = m
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb
 */
  __pyx_t_3 = __pyx_v_nrhs;
  __site_setindex_245_10->Target(__site_setindex_245_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"nrhs"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":246
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["ldb"] = ldb
 *     retval["rcond"] = rcond
 */
  __pyx_t_3 = __pyx_v_lda;
  __site_setindex_246_10->Target(__site_setindex_246_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":247
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb             # <<<<<<<<<<<<<<
 *     retval["rcond"] = rcond
 *     retval["rank"] = rank
 */
  __pyx_t_3 = __pyx_v_ldb;
  __site_setindex_247_10->Target(__site_setindex_247_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldb"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":248
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb
 *     retval["rcond"] = rcond             # <<<<<<<<<<<<<<
 *     retval["rank"] = rank
 *     retval["lwork"] = lwork
 */
  __pyx_t_3 = __pyx_v_rcond;
  __site_setindex_248_10->Target(__site_setindex_248_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"rcond"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":249
 *     retval["ldb"] = ldb
 *     retval["rcond"] = rcond
 *     retval["rank"] = rank             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 */
  __pyx_t_3 = __pyx_v_rank;
  __site_setindex_249_10->Target(__site_setindex_249_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"rank"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":250
 *     retval["rcond"] = rcond
 *     retval["rank"] = rank
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_3 = __pyx_v_lwork;
  __site_setindex_250_10->Target(__site_setindex_250_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":251
 *     retval["rank"] = rank
 *     retval["lwork"] = lwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_3 = __pyx_v_info;
  __site_setindex_251_10->Target(__site_setindex_251_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":252
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 *     return retval             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = ((System::Object^)__pyx_v_retval);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":255
 * 
 * 
 * def dgesv(int n, int nrhs, np.ndarray a, int lda, np.ndarray ipiv,             # <<<<<<<<<<<<<<
 *            np.ndarray b, int ldb, int info):
 *     cdef int lapack_lite_status__
 */

static System::Object^ dgesv(System::Object^ n, System::Object^ nrhs, System::Object^ a, System::Object^ lda, System::Object^ ipiv, System::Object^ b, System::Object^ ldb, System::Object^ info) {
  int __pyx_v_n;
  int __pyx_v_nrhs;
  NumpyDotNet::ndarray^ __pyx_v_a = nullptr;
  int __pyx_v_lda;
  NumpyDotNet::ndarray^ __pyx_v_ipiv = nullptr;
  NumpyDotNet::ndarray^ __pyx_v_b = nullptr;
  int __pyx_v_ldb;
  int __pyx_v_info;
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  PythonDictionary^ __pyx_t_2;
  System::Object^ __pyx_t_3 = nullptr;
  __pyx_v_n = __site_cvt_cvt_int_255_0->Target(__site_cvt_cvt_int_255_0, n);
  __pyx_v_nrhs = __site_cvt_cvt_int_255_0_1->Target(__site_cvt_cvt_int_255_0_1, nrhs);
  __pyx_v_a = ((NumpyDotNet::ndarray^)a);
  __pyx_v_lda = __site_cvt_cvt_int_255_0_2->Target(__site_cvt_cvt_int_255_0_2, lda);
  __pyx_v_ipiv = ((NumpyDotNet::ndarray^)ipiv);
  __pyx_v_b = ((NumpyDotNet::ndarray^)b);
  __pyx_v_ldb = __site_cvt_cvt_int_255_0_3->Target(__site_cvt_cvt_int_255_0_3, ldb);
  __pyx_v_info = __site_cvt_cvt_int_255_0_4->Target(__site_cvt_cvt_int_255_0_4, info);
  __pyx_v_retval = nullptr;
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr)) {
    throw PythonOps::TypeError("Argument 'a' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_ipiv) == nullptr)) {
    throw PythonOps::TypeError("Argument 'ipiv' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_b) == nullptr)) {
    throw PythonOps::TypeError("Argument 'b' has incorrect type");
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":259
 *     cdef int lapack_lite_status__
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgesv"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(ipiv,np.NPY_INT,"ipiv","np.NPY_INT","dgesv"): return None
 *     if not check_object(b,np.NPY_DOUBLE,"b","np.NPY_DOUBLE","dgesv"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_a, NPY_DOUBLE, __pyx_k__a, __pyx_k_1, __pyx_k__dgesv));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":260
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgesv"): return None
 *     if not check_object(ipiv,np.NPY_INT,"ipiv","np.NPY_INT","dgesv"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(b,np.NPY_DOUBLE,"b","np.NPY_DOUBLE","dgesv"): return None
 * 
 */
  __pyx_t_1 = (!check_object(__pyx_v_ipiv, NPY_INT, __pyx_k__ipiv, __pyx_k_2, __pyx_k__dgesv));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":261
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgesv"): return None
 *     if not check_object(ipiv,np.NPY_INT,"ipiv","np.NPY_INT","dgesv"): return None
 *     if not check_object(b,np.NPY_DOUBLE,"b","np.NPY_DOUBLE","dgesv"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_dgesv(&n,&nrhs,
 */
  __pyx_t_1 = (!check_object(__pyx_v_b, NPY_DOUBLE, __pyx_k__b, __pyx_k_1, __pyx_k__dgesv));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":267
 *                                         <int *>np.PyArray_DATA(ipiv),
 *                                         <double *>np.PyArray_DATA(b),&ldb,
 *                                         &info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(dgesv_)((&__pyx_v_n), (&__pyx_v_nrhs), ((double *)PyArray_DATA(__pyx_v_a)), (&__pyx_v_lda), ((int *)PyArray_DATA(__pyx_v_ipiv)), ((double *)PyArray_DATA(__pyx_v_b)), (&__pyx_v_ldb), (&__pyx_v_info));

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":269
 *                                         &info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["dgesv_"] = lapack_lite_status__
 *     retval["n"] = n
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":270
 * 
 *     retval = {}
 *     retval["dgesv_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs
 */
  __pyx_t_3 = __pyx_v_lapack_lite_status__;
  __site_setindex_270_10->Target(__site_setindex_270_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"dgesv_"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":271
 *     retval = {}
 *     retval["dgesv_"] = lapack_lite_status__
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda
 */
  __pyx_t_3 = __pyx_v_n;
  __site_setindex_271_10->Target(__site_setindex_271_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":272
 *     retval["dgesv_"] = lapack_lite_status__
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb
 */
  __pyx_t_3 = __pyx_v_nrhs;
  __site_setindex_272_10->Target(__site_setindex_272_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"nrhs"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":273
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["ldb"] = ldb
 *     retval["info"] = info
 */
  __pyx_t_3 = __pyx_v_lda;
  __site_setindex_273_10->Target(__site_setindex_273_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":274
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_3 = __pyx_v_ldb;
  __site_setindex_274_10->Target(__site_setindex_274_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldb"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":275
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_3 = __pyx_v_info;
  __site_setindex_275_10->Target(__site_setindex_275_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":276
 *     retval["ldb"] = ldb
 *     retval["info"] = info
 *     return retval             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = ((System::Object^)__pyx_v_retval);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":279
 * 
 * 
 * def dgesdd(jobz, int m, int n, np.ndarray a, int lda,             # <<<<<<<<<<<<<<
 *             np.ndarray s, np.ndarray u, int ldu, np.ndarray vt, int ldvt,
 *             np.ndarray work, int lwork, np.ndarray iwork, int info):
 */

static System::Object^ dgesdd(System::Object^ jobz, System::Object^ m, System::Object^ n, System::Object^ a, System::Object^ lda, System::Object^ s, System::Object^ u, System::Object^ ldu, System::Object^ vt, System::Object^ ldvt, System::Object^ work, System::Object^ lwork, System::Object^ iwork, System::Object^ info) {
  System::Object^ __pyx_v_jobz = nullptr;
  int __pyx_v_m;
  int __pyx_v_n;
  NumpyDotNet::ndarray^ __pyx_v_a = nullptr;
  int __pyx_v_lda;
  NumpyDotNet::ndarray^ __pyx_v_s = nullptr;
  NumpyDotNet::ndarray^ __pyx_v_u = nullptr;
  int __pyx_v_ldu;
  NumpyDotNet::ndarray^ __pyx_v_vt = nullptr;
  int __pyx_v_ldvt;
  NumpyDotNet::ndarray^ __pyx_v_work = nullptr;
  int __pyx_v_lwork;
  NumpyDotNet::ndarray^ __pyx_v_iwork = nullptr;
  int __pyx_v_info;
  int __pyx_v_lapack_lite_status__;
  char __pyx_v_jobz_char;
  long __pyx_v_work0;
  int __pyx_v_mn;
  int __pyx_v_mx;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  char __pyx_t_4;
  int __pyx_t_5;
  int __pyx_t_6;
  int __pyx_t_7;
  int __pyx_t_8;
  int __pyx_t_9;
  int __pyx_t_10;
  long __pyx_t_11;
  long __pyx_t_12;
  long __pyx_t_13;
  long __pyx_t_14;
  PythonDictionary^ __pyx_t_15;
  __pyx_v_jobz = jobz;
  __pyx_v_m = __site_cvt_cvt_int_279_0->Target(__site_cvt_cvt_int_279_0, m);
  __pyx_v_n = __site_cvt_cvt_int_279_0_1->Target(__site_cvt_cvt_int_279_0_1, n);
  __pyx_v_a = ((NumpyDotNet::ndarray^)a);
  __pyx_v_lda = __site_cvt_cvt_int_279_0_2->Target(__site_cvt_cvt_int_279_0_2, lda);
  __pyx_v_s = ((NumpyDotNet::ndarray^)s);
  __pyx_v_u = ((NumpyDotNet::ndarray^)u);
  __pyx_v_ldu = __site_cvt_cvt_int_279_0_3->Target(__site_cvt_cvt_int_279_0_3, ldu);
  __pyx_v_vt = ((NumpyDotNet::ndarray^)vt);
  __pyx_v_ldvt = __site_cvt_cvt_int_279_0_4->Target(__site_cvt_cvt_int_279_0_4, ldvt);
  __pyx_v_work = ((NumpyDotNet::ndarray^)work);
  __pyx_v_lwork = __site_cvt_cvt_int_279_0_5->Target(__site_cvt_cvt_int_279_0_5, lwork);
  __pyx_v_iwork = ((NumpyDotNet::ndarray^)iwork);
  __pyx_v_info = __site_cvt_cvt_int_279_0_6->Target(__site_cvt_cvt_int_279_0_6, info);
  __pyx_v_retval = nullptr;
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr)) {
    throw PythonOps::TypeError("Argument 'a' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_s) == nullptr)) {
    throw PythonOps::TypeError("Argument 's' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_u) == nullptr)) {
    throw PythonOps::TypeError("Argument 'u' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_vt) == nullptr)) {
    throw PythonOps::TypeError("Argument 'vt' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr)) {
    throw PythonOps::TypeError("Argument 'work' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_iwork) == nullptr)) {
    throw PythonOps::TypeError("Argument 'iwork' has incorrect type");
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":283
 *             np.ndarray work, int lwork, np.ndarray iwork, int info):
 *     cdef int lapack_lite_status__
 *     cdef char jobz_char = ord(jobz[0])             # <<<<<<<<<<<<<<
 *     cdef long work0
 *     cdef int mn, mx
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "ord");
  __pyx_t_2 = __site_getindex_283_34->Target(__site_getindex_283_34, __pyx_v_jobz, ((System::Object^)0));
  __pyx_t_3 = __site_call1_283_29->Target(__site_call1_283_29, __pyx_context, __pyx_t_1, __pyx_t_2);
  __pyx_t_1 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_t_4 = __site_cvt_cvt_char_283_29->Target(__site_cvt_cvt_char_283_29, __pyx_t_3);
  __pyx_t_3 = nullptr;
  __pyx_v_jobz_char = __pyx_t_4;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":287
 *     cdef int mn, mx
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(u,np.NPY_DOUBLE,"u","np.NPY_DOUBLE","dgesdd"): return None
 */
  __pyx_t_5 = (!check_object(__pyx_v_a, NPY_DOUBLE, __pyx_k__a, __pyx_k_1, __pyx_k__dgesdd));
  if (__pyx_t_5) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":288
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","dgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(u,np.NPY_DOUBLE,"u","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(vt,np.NPY_DOUBLE,"vt","np.NPY_DOUBLE","dgesdd"): return None
 */
  __pyx_t_5 = (!check_object(__pyx_v_s, NPY_DOUBLE, __pyx_k__s, __pyx_k_1, __pyx_k__dgesdd));
  if (__pyx_t_5) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":289
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(u,np.NPY_DOUBLE,"u","np.NPY_DOUBLE","dgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(vt,np.NPY_DOUBLE,"vt","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgesdd"): return None
 */
  __pyx_t_5 = (!check_object(__pyx_v_u, NPY_DOUBLE, __pyx_k__u, __pyx_k_1, __pyx_k__dgesdd));
  if (__pyx_t_5) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":290
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(u,np.NPY_DOUBLE,"u","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(vt,np.NPY_DOUBLE,"vt","np.NPY_DOUBLE","dgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","dgesdd"): return None
 */
  __pyx_t_5 = (!check_object(__pyx_v_vt, NPY_DOUBLE, __pyx_k__vt, __pyx_k_1, __pyx_k__dgesdd));
  if (__pyx_t_5) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L8;
  }
  __pyx_L8:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":291
 *     if not check_object(u,np.NPY_DOUBLE,"u","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(vt,np.NPY_DOUBLE,"vt","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","dgesdd"): return None
 * 
 */
  __pyx_t_5 = (!check_object(__pyx_v_work, NPY_DOUBLE, __pyx_k__work, __pyx_k_1, __pyx_k__dgesdd));
  if (__pyx_t_5) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L9;
  }
  __pyx_L9:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":292
 *     if not check_object(vt,np.NPY_DOUBLE,"vt","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","dgesdd"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_dgesdd(&jobz_char,&m,&n,
 */
  __pyx_t_5 = (!check_object(__pyx_v_iwork, NPY_INT, __pyx_k__iwork, __pyx_k_2, __pyx_k__dgesdd));
  if (__pyx_t_5) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L10;
  }
  __pyx_L10:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":300
 *                                          <double *>np.PyArray_DATA(vt),&ldvt,
 *                                          <double *>np.PyArray_DATA(work),&lwork,
 *                                          <int *>np.PyArray_DATA(iwork),&info)             # <<<<<<<<<<<<<<
 * 
 *     if info == 0 and lwork == -1:
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(dgesdd_)((&__pyx_v_jobz_char), (&__pyx_v_m), (&__pyx_v_n), ((double *)PyArray_DATA(__pyx_v_a)), (&__pyx_v_lda), ((double *)PyArray_DATA(__pyx_v_s)), ((double *)PyArray_DATA(__pyx_v_u)), (&__pyx_v_ldu), ((double *)PyArray_DATA(__pyx_v_vt)), (&__pyx_v_ldvt), ((double *)PyArray_DATA(__pyx_v_work)), (&__pyx_v_lwork), ((int *)PyArray_DATA(__pyx_v_iwork)), (&__pyx_v_info));

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":302
 *                                          <int *>np.PyArray_DATA(iwork),&info)
 * 
 *     if info == 0 and lwork == -1:             # <<<<<<<<<<<<<<
 *         # We need to check the result because
 *         # sometimes the "optimal" value is actually
 */
  __pyx_t_5 = (__pyx_v_info == 0);
  if (__pyx_t_5) {
    __pyx_t_6 = (__pyx_v_lwork == -1);
    __pyx_t_7 = __pyx_t_6;
  } else {
    __pyx_t_7 = __pyx_t_5;
  }
  if (__pyx_t_7) {

    /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":307
 *         # too small.
 *         # Change it to the maximum of the minimum and the optimal.
 *         work0 = <long>(<double *>np.PyArray_DATA(work))[0]             # <<<<<<<<<<<<<<
 *         mn = min(m,n)
 *         mx = max(m,n)
 */
    __pyx_v_work0 = ((long)(((double *)PyArray_DATA(__pyx_v_work))[0]));

    /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":308
 *         # Change it to the maximum of the minimum and the optimal.
 *         work0 = <long>(<double *>np.PyArray_DATA(work))[0]
 *         mn = min(m,n)             # <<<<<<<<<<<<<<
 *         mx = max(m,n)
 * 
 */
    __pyx_t_8 = __pyx_v_n;
    __pyx_t_9 = __pyx_v_m;
    if ((__pyx_t_8 < __pyx_t_9)) {
      __pyx_t_10 = __pyx_t_8;
    } else {
      __pyx_t_10 = __pyx_t_9;
    }
    __pyx_v_mn = __pyx_t_10;

    /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":309
 *         work0 = <long>(<double *>np.PyArray_DATA(work))[0]
 *         mn = min(m,n)
 *         mx = max(m,n)             # <<<<<<<<<<<<<<
 * 
 *         if jobz == 'N':
 */
    __pyx_t_10 = __pyx_v_n;
    __pyx_t_8 = __pyx_v_m;
    if ((__pyx_t_10 > __pyx_t_8)) {
      __pyx_t_9 = __pyx_t_10;
    } else {
      __pyx_t_9 = __pyx_t_8;
    }
    __pyx_v_mx = __pyx_t_9;

    /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":311
 *         mx = max(m,n)
 * 
 *         if jobz == 'N':             # <<<<<<<<<<<<<<
 *             work0 = max(work0,3*mn + max(mx,6*mn)+500)
 *         elif jobz == 'O':
 */
    __pyx_t_3 = __site_op_eq_311_16->Target(__site_op_eq_311_16, __pyx_v_jobz, ((System::Object^)"N"));
    __pyx_t_7 = __site_istrue_311_16->Target(__site_istrue_311_16, __pyx_t_3);
    __pyx_t_3 = nullptr;
    if (__pyx_t_7) {

      /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":312
 * 
 *         if jobz == 'N':
 *             work0 = max(work0,3*mn + max(mx,6*mn)+500)             # <<<<<<<<<<<<<<
 *         elif jobz == 'O':
 *             work0 = max(work0,3*mn*mn + max(mx,5*mn*mn+4*mn+500))
 */
      __pyx_t_11 = (6 * __pyx_v_mn);
      __pyx_t_9 = __pyx_v_mx;
      if ((__pyx_t_11 > __pyx_t_9)) {
        __pyx_t_12 = __pyx_t_11;
      } else {
        __pyx_t_12 = __pyx_t_9;
      }
      __pyx_t_11 = (((3 * __pyx_v_mn) + __pyx_t_12) + 500);
      __pyx_t_13 = __pyx_v_work0;
      if ((__pyx_t_11 > __pyx_t_13)) {
        __pyx_t_14 = __pyx_t_11;
      } else {
        __pyx_t_14 = __pyx_t_13;
      }
      __pyx_v_work0 = __pyx_t_14;
      goto __pyx_L12;
    }

    /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":313
 *         if jobz == 'N':
 *             work0 = max(work0,3*mn + max(mx,6*mn)+500)
 *         elif jobz == 'O':             # <<<<<<<<<<<<<<
 *             work0 = max(work0,3*mn*mn + max(mx,5*mn*mn+4*mn+500))
 *         elif jobz == 'S' or jobz == 'A':
 */
    __pyx_t_3 = __site_op_eq_313_18->Target(__site_op_eq_313_18, __pyx_v_jobz, ((System::Object^)"O"));
    __pyx_t_7 = __site_istrue_313_18->Target(__site_istrue_313_18, __pyx_t_3);
    __pyx_t_3 = nullptr;
    if (__pyx_t_7) {

      /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":314
 *             work0 = max(work0,3*mn + max(mx,6*mn)+500)
 *         elif jobz == 'O':
 *             work0 = max(work0,3*mn*mn + max(mx,5*mn*mn+4*mn+500))             # <<<<<<<<<<<<<<
 *         elif jobz == 'S' or jobz == 'A':
 *             work0 = max(work0,3*mn*mn + max(mx,4*mn*(mn+1))+500)
 */
      __pyx_t_14 = ((((5 * __pyx_v_mn) * __pyx_v_mn) + (4 * __pyx_v_mn)) + 500);
      __pyx_t_9 = __pyx_v_mx;
      if ((__pyx_t_14 > __pyx_t_9)) {
        __pyx_t_12 = __pyx_t_14;
      } else {
        __pyx_t_12 = __pyx_t_9;
      }
      __pyx_t_14 = (((3 * __pyx_v_mn) * __pyx_v_mn) + __pyx_t_12);
      __pyx_t_11 = __pyx_v_work0;
      if ((__pyx_t_14 > __pyx_t_11)) {
        __pyx_t_13 = __pyx_t_14;
      } else {
        __pyx_t_13 = __pyx_t_11;
      }
      __pyx_v_work0 = __pyx_t_13;
      goto __pyx_L12;
    }

    /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":315
 *         elif jobz == 'O':
 *             work0 = max(work0,3*mn*mn + max(mx,5*mn*mn+4*mn+500))
 *         elif jobz == 'S' or jobz == 'A':             # <<<<<<<<<<<<<<
 *             work0 = max(work0,3*mn*mn + max(mx,4*mn*(mn+1))+500)
 * 
 */
    __pyx_t_3 = __site_op_eq_315_18->Target(__site_op_eq_315_18, __pyx_v_jobz, ((System::Object^)"S"));
    __pyx_t_7 = __site_istrue_315_18->Target(__site_istrue_315_18, __pyx_t_3);
    __pyx_t_3 = nullptr;
    if (!__pyx_t_7) {
      __pyx_t_3 = __site_op_eq_315_33->Target(__site_op_eq_315_33, __pyx_v_jobz, ((System::Object^)"A"));
      __pyx_t_5 = __site_istrue_315_33->Target(__site_istrue_315_33, __pyx_t_3);
      __pyx_t_3 = nullptr;
      __pyx_t_6 = __pyx_t_5;
    } else {
      __pyx_t_6 = __pyx_t_7;
    }
    if (__pyx_t_6) {

      /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":316
 *             work0 = max(work0,3*mn*mn + max(mx,5*mn*mn+4*mn+500))
 *         elif jobz == 'S' or jobz == 'A':
 *             work0 = max(work0,3*mn*mn + max(mx,4*mn*(mn+1))+500)             # <<<<<<<<<<<<<<
 * 
 *         (<double *>np.PyArray_DATA(work))[0] = <double>work0
 */
      __pyx_t_13 = ((4 * __pyx_v_mn) * (__pyx_v_mn + 1));
      __pyx_t_9 = __pyx_v_mx;
      if ((__pyx_t_13 > __pyx_t_9)) {
        __pyx_t_12 = __pyx_t_13;
      } else {
        __pyx_t_12 = __pyx_t_9;
      }
      __pyx_t_13 = ((((3 * __pyx_v_mn) * __pyx_v_mn) + __pyx_t_12) + 500);
      __pyx_t_14 = __pyx_v_work0;
      if ((__pyx_t_13 > __pyx_t_14)) {
        __pyx_t_11 = __pyx_t_13;
      } else {
        __pyx_t_11 = __pyx_t_14;
      }
      __pyx_v_work0 = __pyx_t_11;
      goto __pyx_L12;
    }
    __pyx_L12:;

    /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":318
 *             work0 = max(work0,3*mn*mn + max(mx,4*mn*(mn+1))+500)
 * 
 *         (<double *>np.PyArray_DATA(work))[0] = <double>work0             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
    (((double *)PyArray_DATA(__pyx_v_work))[0]) = ((double)__pyx_v_work0);
    goto __pyx_L11;
  }
  __pyx_L11:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":320
 *         (<double *>np.PyArray_DATA(work))[0] = <double>work0
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["dgesdd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz_char
 */
  __pyx_t_15 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_15;
  __pyx_t_15 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":321
 * 
 *     retval = {}
 *     retval["dgesdd_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["jobz"] = jobz_char
 *     retval["m"] = m
 */
  __pyx_t_3 = __pyx_v_lapack_lite_status__;
  __site_setindex_321_10->Target(__site_setindex_321_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"dgesdd_"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":322
 *     retval = {}
 *     retval["dgesdd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz_char             # <<<<<<<<<<<<<<
 *     retval["m"] = m
 *     retval["n"] = n
 */
  __pyx_t_3 = __pyx_v_jobz_char;
  __site_setindex_322_10->Target(__site_setindex_322_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"jobz"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":323
 *     retval["dgesdd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz_char
 *     retval["m"] = m             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_3 = __pyx_v_m;
  __site_setindex_323_10->Target(__site_setindex_323_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"m"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":324
 *     retval["jobz"] = jobz_char
 *     retval["m"] = m
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["ldu"] = ldu
 */
  __pyx_t_3 = __pyx_v_n;
  __site_setindex_324_10->Target(__site_setindex_324_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":325
 *     retval["m"] = m
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["ldu"] = ldu
 *     retval["ldvt"] = ldvt
 */
  __pyx_t_3 = __pyx_v_lda;
  __site_setindex_325_10->Target(__site_setindex_325_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":326
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["ldu"] = ldu             # <<<<<<<<<<<<<<
 *     retval["ldvt"] = ldvt
 *     retval["lwork"] = lwork
 */
  __pyx_t_3 = __pyx_v_ldu;
  __site_setindex_326_10->Target(__site_setindex_326_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldu"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":327
 *     retval["lda"] = lda
 *     retval["ldu"] = ldu
 *     retval["ldvt"] = ldvt             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 */
  __pyx_t_3 = __pyx_v_ldvt;
  __site_setindex_327_10->Target(__site_setindex_327_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldvt"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":328
 *     retval["ldu"] = ldu
 *     retval["ldvt"] = ldvt
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_3 = __pyx_v_lwork;
  __site_setindex_328_10->Target(__site_setindex_328_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":329
 *     retval["ldvt"] = ldvt
 *     retval["lwork"] = lwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_3 = __pyx_v_info;
  __site_setindex_329_10->Target(__site_setindex_329_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":330
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 *     return retval             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = ((System::Object^)__pyx_v_retval);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":333
 * 
 * 
 * def dgetrf(int m, int n, np.ndarray a, int lda, np.ndarray ipiv, int info):             # <<<<<<<<<<<<<<
 *     cdef int lapack_lite_status__
 * 
 */

static System::Object^ dgetrf(System::Object^ m, System::Object^ n, System::Object^ a, System::Object^ lda, System::Object^ ipiv, System::Object^ info) {
  int __pyx_v_m;
  int __pyx_v_n;
  NumpyDotNet::ndarray^ __pyx_v_a = nullptr;
  int __pyx_v_lda;
  NumpyDotNet::ndarray^ __pyx_v_ipiv = nullptr;
  int __pyx_v_info;
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  PythonDictionary^ __pyx_t_2;
  System::Object^ __pyx_t_3 = nullptr;
  __pyx_v_m = __site_cvt_cvt_int_333_0->Target(__site_cvt_cvt_int_333_0, m);
  __pyx_v_n = __site_cvt_cvt_int_333_0_1->Target(__site_cvt_cvt_int_333_0_1, n);
  __pyx_v_a = ((NumpyDotNet::ndarray^)a);
  __pyx_v_lda = __site_cvt_cvt_int_333_0_2->Target(__site_cvt_cvt_int_333_0_2, lda);
  __pyx_v_ipiv = ((NumpyDotNet::ndarray^)ipiv);
  __pyx_v_info = __site_cvt_cvt_int_333_0_3->Target(__site_cvt_cvt_int_333_0_3, info);
  __pyx_v_retval = nullptr;
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr)) {
    throw PythonOps::TypeError("Argument 'a' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_ipiv) == nullptr)) {
    throw PythonOps::TypeError("Argument 'ipiv' has incorrect type");
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":336
 *     cdef int lapack_lite_status__
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgetrf"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(ipiv,np.NPY_INT,"ipiv","np.NPY_INT","dgetrf"): return None
 * 
 */
  __pyx_t_1 = (!check_object(__pyx_v_a, NPY_DOUBLE, __pyx_k__a, __pyx_k_1, __pyx_k__dgetrf));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":337
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgetrf"): return None
 *     if not check_object(ipiv,np.NPY_INT,"ipiv","np.NPY_INT","dgetrf"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_dgetrf(&m,&n,<double *>np.PyArray_DATA(a),&lda,
 */
  __pyx_t_1 = (!check_object(__pyx_v_ipiv, NPY_INT, __pyx_k__ipiv, __pyx_k_2, __pyx_k__dgetrf));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":340
 * 
 *     lapack_lite_status__ = lapack_dgetrf(&m,&n,<double *>np.PyArray_DATA(a),&lda,
 *                                          <int *>np.PyArray_DATA(ipiv),&info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(dgetrf_)((&__pyx_v_m), (&__pyx_v_n), ((double *)PyArray_DATA(__pyx_v_a)), (&__pyx_v_lda), ((int *)PyArray_DATA(__pyx_v_ipiv)), (&__pyx_v_info));

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":342
 *                                          <int *>np.PyArray_DATA(ipiv),&info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["dgetrf_"] = lapack_lite_status__
 *     retval["m"] = m
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":343
 * 
 *     retval = {}
 *     retval["dgetrf_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["m"] = m
 *     retval["n"] = n
 */
  __pyx_t_3 = __pyx_v_lapack_lite_status__;
  __site_setindex_343_10->Target(__site_setindex_343_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"dgetrf_"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":344
 *     retval = {}
 *     retval["dgetrf_"] = lapack_lite_status__
 *     retval["m"] = m             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_3 = __pyx_v_m;
  __site_setindex_344_10->Target(__site_setindex_344_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"m"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":345
 *     retval["dgetrf_"] = lapack_lite_status__
 *     retval["m"] = m
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["info"] = info
 */
  __pyx_t_3 = __pyx_v_n;
  __site_setindex_345_10->Target(__site_setindex_345_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":346
 *     retval["m"] = m
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_3 = __pyx_v_lda;
  __site_setindex_346_10->Target(__site_setindex_346_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":347
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_3 = __pyx_v_info;
  __site_setindex_347_10->Target(__site_setindex_347_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":348
 *     retval["lda"] = lda
 *     retval["info"] = info
 *     return retval             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = ((System::Object^)__pyx_v_retval);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":351
 * 
 * 
 * def dpotrf(uplo, int n, np.ndarray a, int lda, int info):             # <<<<<<<<<<<<<<
 *     cdef int lapack_lite_status__
 *     cdef char uplo_char = ord(uplo[0])
 */

static System::Object^ dpotrf(System::Object^ uplo, System::Object^ n, System::Object^ a, System::Object^ lda, System::Object^ info) {
  System::Object^ __pyx_v_uplo = nullptr;
  int __pyx_v_n;
  NumpyDotNet::ndarray^ __pyx_v_a = nullptr;
  int __pyx_v_lda;
  int __pyx_v_info;
  int __pyx_v_lapack_lite_status__;
  char __pyx_v_uplo_char;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  char __pyx_t_4;
  int __pyx_t_5;
  PythonDictionary^ __pyx_t_6;
  __pyx_v_uplo = uplo;
  __pyx_v_n = __site_cvt_cvt_int_351_0->Target(__site_cvt_cvt_int_351_0, n);
  __pyx_v_a = ((NumpyDotNet::ndarray^)a);
  __pyx_v_lda = __site_cvt_cvt_int_351_0_1->Target(__site_cvt_cvt_int_351_0_1, lda);
  __pyx_v_info = __site_cvt_cvt_int_351_0_2->Target(__site_cvt_cvt_int_351_0_2, info);
  __pyx_v_retval = nullptr;
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr)) {
    throw PythonOps::TypeError("Argument 'a' has incorrect type");
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":353
 * def dpotrf(uplo, int n, np.ndarray a, int lda, int info):
 *     cdef int lapack_lite_status__
 *     cdef char uplo_char = ord(uplo[0])             # <<<<<<<<<<<<<<
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dpotrf"): return None
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "ord");
  __pyx_t_2 = __site_getindex_353_34->Target(__site_getindex_353_34, __pyx_v_uplo, ((System::Object^)0));
  __pyx_t_3 = __site_call1_353_29->Target(__site_call1_353_29, __pyx_context, __pyx_t_1, __pyx_t_2);
  __pyx_t_1 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_t_4 = __site_cvt_cvt_char_353_29->Target(__site_cvt_cvt_char_353_29, __pyx_t_3);
  __pyx_t_3 = nullptr;
  __pyx_v_uplo_char = __pyx_t_4;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":355
 *     cdef char uplo_char = ord(uplo[0])
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dpotrf"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_dpotrf(&uplo_char,&n,
 */
  __pyx_t_5 = (!check_object(__pyx_v_a, NPY_DOUBLE, __pyx_k__a, __pyx_k_1, __pyx_k__dpotrf));
  if (__pyx_t_5) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":359
 *     lapack_lite_status__ = lapack_dpotrf(&uplo_char,&n,
 *                                          <double *>np.PyArray_DATA(a),&lda,
 *                                          &info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(dpotrf_)((&__pyx_v_uplo_char), (&__pyx_v_n), ((double *)PyArray_DATA(__pyx_v_a)), (&__pyx_v_lda), (&__pyx_v_info));

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":361
 *                                          &info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["dpotrf_"] = lapack_lite_status__
 *     retval["n"] = n
 */
  __pyx_t_6 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_6;
  __pyx_t_6 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":362
 * 
 *     retval = {}
 *     retval["dpotrf_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_3 = __pyx_v_lapack_lite_status__;
  __site_setindex_362_10->Target(__site_setindex_362_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"dpotrf_"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":363
 *     retval = {}
 *     retval["dpotrf_"] = lapack_lite_status__
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["info"] = info
 */
  __pyx_t_3 = __pyx_v_n;
  __site_setindex_363_10->Target(__site_setindex_363_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":364
 *     retval["dpotrf_"] = lapack_lite_status__
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_3 = __pyx_v_lda;
  __site_setindex_364_10->Target(__site_setindex_364_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":365
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_3 = __pyx_v_info;
  __site_setindex_365_10->Target(__site_setindex_365_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":366
 *     retval["lda"] = lda
 *     retval["info"] = info
 *     return retval             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = ((System::Object^)__pyx_v_retval);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":369
 * 
 * 
 * def dgeqrf(int m, int n, np.ndarray a, int lda,             # <<<<<<<<<<<<<<
 *             np.ndarray tau, np.ndarray work, int lwork, int info):
 *     cdef int  lapack_lite_status__
 */

static System::Object^ dgeqrf(System::Object^ m, System::Object^ n, System::Object^ a, System::Object^ lda, System::Object^ tau, System::Object^ work, System::Object^ lwork, System::Object^ info) {
  int __pyx_v_m;
  int __pyx_v_n;
  NumpyDotNet::ndarray^ __pyx_v_a = nullptr;
  int __pyx_v_lda;
  NumpyDotNet::ndarray^ __pyx_v_tau = nullptr;
  NumpyDotNet::ndarray^ __pyx_v_work = nullptr;
  int __pyx_v_lwork;
  int __pyx_v_info;
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  PythonDictionary^ __pyx_t_2;
  System::Object^ __pyx_t_3 = nullptr;
  __pyx_v_m = __site_cvt_cvt_int_369_0->Target(__site_cvt_cvt_int_369_0, m);
  __pyx_v_n = __site_cvt_cvt_int_369_0_1->Target(__site_cvt_cvt_int_369_0_1, n);
  __pyx_v_a = ((NumpyDotNet::ndarray^)a);
  __pyx_v_lda = __site_cvt_cvt_int_369_0_2->Target(__site_cvt_cvt_int_369_0_2, lda);
  __pyx_v_tau = ((NumpyDotNet::ndarray^)tau);
  __pyx_v_work = ((NumpyDotNet::ndarray^)work);
  __pyx_v_lwork = __site_cvt_cvt_int_369_0_3->Target(__site_cvt_cvt_int_369_0_3, lwork);
  __pyx_v_info = __site_cvt_cvt_int_369_0_4->Target(__site_cvt_cvt_int_369_0_4, info);
  __pyx_v_retval = nullptr;
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr)) {
    throw PythonOps::TypeError("Argument 'a' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_tau) == nullptr)) {
    throw PythonOps::TypeError("Argument 'tau' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr)) {
    throw PythonOps::TypeError("Argument 'work' has incorrect type");
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":374
 * 
 *     # check objects and convert to right storage order
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgeqrf"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(tau,np.NPY_DOUBLE,"tau","np.NPY_DOUBLE","dgeqrf"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgeqrf"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_a, NPY_DOUBLE, __pyx_k__a, __pyx_k_1, __pyx_k__dgeqrf));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":375
 *     # check objects and convert to right storage order
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgeqrf"): return None
 *     if not check_object(tau,np.NPY_DOUBLE,"tau","np.NPY_DOUBLE","dgeqrf"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgeqrf"): return None
 * 
 */
  __pyx_t_1 = (!check_object(__pyx_v_tau, NPY_DOUBLE, __pyx_k__tau, __pyx_k_1, __pyx_k__dgeqrf));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":376
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgeqrf"): return None
 *     if not check_object(tau,np.NPY_DOUBLE,"tau","np.NPY_DOUBLE","dgeqrf"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgeqrf"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_dgeqrf(&m, &n,
 */
  __pyx_t_1 = (!check_object(__pyx_v_work, NPY_DOUBLE, __pyx_k__work, __pyx_k_1, __pyx_k__dgeqrf));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":382
 *                                          <double *>np.PyArray_DATA(tau),
 *                                          <double *>np.PyArray_DATA(work), &lwork,
 *                                          &info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(dgeqrf_)((&__pyx_v_m), (&__pyx_v_n), ((double *)PyArray_DATA(__pyx_v_a)), (&__pyx_v_lda), ((double *)PyArray_DATA(__pyx_v_tau)), ((double *)PyArray_DATA(__pyx_v_work)), (&__pyx_v_lwork), (&__pyx_v_info));

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":384
 *                                          &info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["dgeqrf_"] = lapack_lite_status__
 *     retval["m"] = m
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":385
 * 
 *     retval = {}
 *     retval["dgeqrf_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["m"] = m
 *     retval["n"] = n
 */
  __pyx_t_3 = __pyx_v_lapack_lite_status__;
  __site_setindex_385_10->Target(__site_setindex_385_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"dgeqrf_"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":386
 *     retval = {}
 *     retval["dgeqrf_"] = lapack_lite_status__
 *     retval["m"] = m             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_3 = __pyx_v_m;
  __site_setindex_386_10->Target(__site_setindex_386_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"m"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":387
 *     retval["dgeqrf_"] = lapack_lite_status__
 *     retval["m"] = m
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork
 */
  __pyx_t_3 = __pyx_v_n;
  __site_setindex_387_10->Target(__site_setindex_387_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":388
 *     retval["m"] = m
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 */
  __pyx_t_3 = __pyx_v_lda;
  __site_setindex_388_10->Target(__site_setindex_388_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":389
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_3 = __pyx_v_lwork;
  __site_setindex_389_10->Target(__site_setindex_389_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":390
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_3 = __pyx_v_info;
  __site_setindex_390_10->Target(__site_setindex_390_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":391
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 *     return retval             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = ((System::Object^)__pyx_v_retval);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":394
 * 
 * 
 * def dorgqr(int m, int n, int k, np.ndarray a, int lda,             # <<<<<<<<<<<<<<
 *             np.ndarray tau, np.ndarray work, int lwork, int info):
 *     cdef int  lapack_lite_status__
 */

static System::Object^ dorgqr(System::Object^ m, System::Object^ n, System::Object^ k, System::Object^ a, System::Object^ lda, System::Object^ tau, System::Object^ work, System::Object^ lwork, System::Object^ info) {
  int __pyx_v_m;
  int __pyx_v_n;
  int __pyx_v_k;
  NumpyDotNet::ndarray^ __pyx_v_a = nullptr;
  int __pyx_v_lda;
  NumpyDotNet::ndarray^ __pyx_v_tau = nullptr;
  NumpyDotNet::ndarray^ __pyx_v_work = nullptr;
  int __pyx_v_lwork;
  int __pyx_v_info;
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  PythonDictionary^ __pyx_t_2;
  System::Object^ __pyx_t_3 = nullptr;
  __pyx_v_m = __site_cvt_cvt_int_394_0->Target(__site_cvt_cvt_int_394_0, m);
  __pyx_v_n = __site_cvt_cvt_int_394_0_1->Target(__site_cvt_cvt_int_394_0_1, n);
  __pyx_v_k = __site_cvt_cvt_int_394_0_2->Target(__site_cvt_cvt_int_394_0_2, k);
  __pyx_v_a = ((NumpyDotNet::ndarray^)a);
  __pyx_v_lda = __site_cvt_cvt_int_394_0_3->Target(__site_cvt_cvt_int_394_0_3, lda);
  __pyx_v_tau = ((NumpyDotNet::ndarray^)tau);
  __pyx_v_work = ((NumpyDotNet::ndarray^)work);
  __pyx_v_lwork = __site_cvt_cvt_int_394_0_4->Target(__site_cvt_cvt_int_394_0_4, lwork);
  __pyx_v_info = __site_cvt_cvt_int_394_0_5->Target(__site_cvt_cvt_int_394_0_5, info);
  __pyx_v_retval = nullptr;
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr)) {
    throw PythonOps::TypeError("Argument 'a' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_tau) == nullptr)) {
    throw PythonOps::TypeError("Argument 'tau' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr)) {
    throw PythonOps::TypeError("Argument 'work' has incorrect type");
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":398
 *     cdef int  lapack_lite_status__
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dorgqr"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(tau,np.NPY_DOUBLE,"tau","np.NPY_DOUBLE","dorgqr"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dorgqr"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_a, NPY_DOUBLE, __pyx_k__a, __pyx_k_1, __pyx_k__dorgqr));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":399
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dorgqr"): return None
 *     if not check_object(tau,np.NPY_DOUBLE,"tau","np.NPY_DOUBLE","dorgqr"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dorgqr"): return None
 * 
 */
  __pyx_t_1 = (!check_object(__pyx_v_tau, NPY_DOUBLE, __pyx_k__tau, __pyx_k_1, __pyx_k__dorgqr));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":400
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dorgqr"): return None
 *     if not check_object(tau,np.NPY_DOUBLE,"tau","np.NPY_DOUBLE","dorgqr"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dorgqr"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_dorgqr(&m, &n, &k,
 */
  __pyx_t_1 = (!check_object(__pyx_v_work, NPY_DOUBLE, __pyx_k__work, __pyx_k_1, __pyx_k__dorgqr));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":406
 *                                          <double *>np.PyArray_DATA(tau),
 *                                          <double *>np.PyArray_DATA(work), &lwork,
 *                                          &info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(dorgqr_)((&__pyx_v_m), (&__pyx_v_n), (&__pyx_v_k), ((double *)PyArray_DATA(__pyx_v_a)), (&__pyx_v_lda), ((double *)PyArray_DATA(__pyx_v_tau)), ((double *)PyArray_DATA(__pyx_v_work)), (&__pyx_v_lwork), (&__pyx_v_info));

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":408
 *                                          &info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["dorgqr_"] = lapack_lite_status__
 *     retval["info"] = info
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":409
 * 
 *     retval = {}
 *     retval["dorgqr_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_3 = __pyx_v_lapack_lite_status__;
  __site_setindex_409_10->Target(__site_setindex_409_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"dorgqr_"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":410
 *     retval = {}
 *     retval["dorgqr_"] = lapack_lite_status__
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_3 = __pyx_v_info;
  __site_setindex_410_10->Target(__site_setindex_410_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":411
 *     retval["dorgqr_"] = lapack_lite_status__
 *     retval["info"] = info
 *     return retval             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = ((System::Object^)__pyx_v_retval);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":414
 * 
 * 
 * def zgeev(jobvl, jobvr, int n, np.ndarray a, int lda,             # <<<<<<<<<<<<<<
 *            np.ndarray w, np.ndarray vl, int ldvl, np.ndarray vr, int ldvr,
 *            np.ndarray work, int lwork, np.ndarray rwork, int info):
 */

static System::Object^ zgeev(System::Object^ jobvl, System::Object^ jobvr, System::Object^ n, System::Object^ a, System::Object^ lda, System::Object^ w, System::Object^ vl, System::Object^ ldvl, System::Object^ vr, System::Object^ ldvr, System::Object^ work, System::Object^ lwork, System::Object^ rwork, System::Object^ info) {
  System::Object^ __pyx_v_jobvl = nullptr;
  System::Object^ __pyx_v_jobvr = nullptr;
  int __pyx_v_n;
  NumpyDotNet::ndarray^ __pyx_v_a = nullptr;
  int __pyx_v_lda;
  NumpyDotNet::ndarray^ __pyx_v_w = nullptr;
  NumpyDotNet::ndarray^ __pyx_v_vl = nullptr;
  int __pyx_v_ldvl;
  NumpyDotNet::ndarray^ __pyx_v_vr = nullptr;
  int __pyx_v_ldvr;
  NumpyDotNet::ndarray^ __pyx_v_work = nullptr;
  int __pyx_v_lwork;
  NumpyDotNet::ndarray^ __pyx_v_rwork = nullptr;
  int __pyx_v_info;
  int __pyx_v_lapack_lite_status__;
  char __pyx_v_jobvl_char;
  char __pyx_v_jobvr_char;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  char __pyx_t_4;
  char __pyx_t_5;
  int __pyx_t_6;
  PythonDictionary^ __pyx_t_7;
  __pyx_v_jobvl = jobvl;
  __pyx_v_jobvr = jobvr;
  __pyx_v_n = __site_cvt_cvt_int_414_0->Target(__site_cvt_cvt_int_414_0, n);
  __pyx_v_a = ((NumpyDotNet::ndarray^)a);
  __pyx_v_lda = __site_cvt_cvt_int_414_0_1->Target(__site_cvt_cvt_int_414_0_1, lda);
  __pyx_v_w = ((NumpyDotNet::ndarray^)w);
  __pyx_v_vl = ((NumpyDotNet::ndarray^)vl);
  __pyx_v_ldvl = __site_cvt_cvt_int_414_0_2->Target(__site_cvt_cvt_int_414_0_2, ldvl);
  __pyx_v_vr = ((NumpyDotNet::ndarray^)vr);
  __pyx_v_ldvr = __site_cvt_cvt_int_414_0_3->Target(__site_cvt_cvt_int_414_0_3, ldvr);
  __pyx_v_work = ((NumpyDotNet::ndarray^)work);
  __pyx_v_lwork = __site_cvt_cvt_int_414_0_4->Target(__site_cvt_cvt_int_414_0_4, lwork);
  __pyx_v_rwork = ((NumpyDotNet::ndarray^)rwork);
  __pyx_v_info = __site_cvt_cvt_int_414_0_5->Target(__site_cvt_cvt_int_414_0_5, info);
  __pyx_v_retval = nullptr;
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr)) {
    throw PythonOps::TypeError("Argument 'a' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_w) == nullptr)) {
    throw PythonOps::TypeError("Argument 'w' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_vl) == nullptr)) {
    throw PythonOps::TypeError("Argument 'vl' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_vr) == nullptr)) {
    throw PythonOps::TypeError("Argument 'vr' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr)) {
    throw PythonOps::TypeError("Argument 'work' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_rwork) == nullptr)) {
    throw PythonOps::TypeError("Argument 'rwork' has incorrect type");
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":418
 *            np.ndarray work, int lwork, np.ndarray rwork, int info):
 *     cdef int lapack_lite_status__
 *     cdef char jobvl_char = ord(jobvl[0])             # <<<<<<<<<<<<<<
 *     cdef char jobvr_char = ord(jobvr[0])
 * 
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "ord");
  __pyx_t_2 = __site_getindex_418_36->Target(__site_getindex_418_36, __pyx_v_jobvl, ((System::Object^)0));
  __pyx_t_3 = __site_call1_418_30->Target(__site_call1_418_30, __pyx_context, __pyx_t_1, __pyx_t_2);
  __pyx_t_1 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_t_4 = __site_cvt_cvt_char_418_30->Target(__site_cvt_cvt_char_418_30, __pyx_t_3);
  __pyx_t_3 = nullptr;
  __pyx_v_jobvl_char = __pyx_t_4;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":419
 *     cdef int lapack_lite_status__
 *     cdef char jobvl_char = ord(jobvl[0])
 *     cdef char jobvr_char = ord(jobvr[0])             # <<<<<<<<<<<<<<
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgeev"): return None
 */
  __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "ord");
  __pyx_t_2 = __site_getindex_419_36->Target(__site_getindex_419_36, __pyx_v_jobvr, ((System::Object^)0));
  __pyx_t_1 = __site_call1_419_30->Target(__site_call1_419_30, __pyx_context, __pyx_t_3, __pyx_t_2);
  __pyx_t_3 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_t_5 = __site_cvt_cvt_char_419_30->Target(__site_cvt_cvt_char_419_30, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_v_jobvr_char = __pyx_t_5;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":421
 *     cdef char jobvr_char = ord(jobvr[0])
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(w,np.NPY_CDOUBLE,"w","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(vl,np.NPY_CDOUBLE,"vl","np.NPY_CDOUBLE","zgeev"): return None
 */
  __pyx_t_6 = (!check_object(__pyx_v_a, NPY_CDOUBLE, __pyx_k__a, __pyx_k_3, __pyx_k__zgeev));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":422
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(w,np.NPY_CDOUBLE,"w","np.NPY_CDOUBLE","zgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(vl,np.NPY_CDOUBLE,"vl","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(vr,np.NPY_CDOUBLE,"vr","np.NPY_CDOUBLE","zgeev"): return None
 */
  __pyx_t_6 = (!check_object(__pyx_v_w, NPY_CDOUBLE, __pyx_k__w, __pyx_k_3, __pyx_k__zgeev));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":423
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(w,np.NPY_CDOUBLE,"w","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(vl,np.NPY_CDOUBLE,"vl","np.NPY_CDOUBLE","zgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(vr,np.NPY_CDOUBLE,"vr","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgeev"): return None
 */
  __pyx_t_6 = (!check_object(__pyx_v_vl, NPY_CDOUBLE, __pyx_k__vl, __pyx_k_3, __pyx_k__zgeev));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":424
 *     if not check_object(w,np.NPY_CDOUBLE,"w","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(vl,np.NPY_CDOUBLE,"vl","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(vr,np.NPY_CDOUBLE,"vr","np.NPY_CDOUBLE","zgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgeev"): return None
 */
  __pyx_t_6 = (!check_object(__pyx_v_vr, NPY_CDOUBLE, __pyx_k__vr, __pyx_k_3, __pyx_k__zgeev));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L8;
  }
  __pyx_L8:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":425
 *     if not check_object(vl,np.NPY_CDOUBLE,"vl","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(vr,np.NPY_CDOUBLE,"vr","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgeev"): return None
 * 
 */
  __pyx_t_6 = (!check_object(__pyx_v_work, NPY_CDOUBLE, __pyx_k__work, __pyx_k_3, __pyx_k__zgeev));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L9;
  }
  __pyx_L9:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":426
 *     if not check_object(vr,np.NPY_CDOUBLE,"vr","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgeev"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_zgeev(&jobvl_char,&jobvr_char,&n,
 */
  __pyx_t_6 = (!check_object(__pyx_v_rwork, NPY_DOUBLE, __pyx_k__rwork, __pyx_k_1, __pyx_k__zgeev));
  if (__pyx_t_6) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L10;
  }
  __pyx_L10:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":434
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(vr),&ldvr,
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 *                                         <double *>np.PyArray_DATA(rwork),&info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(zgeev_)((&__pyx_v_jobvl_char), (&__pyx_v_jobvr_char), (&__pyx_v_n), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_a)), (&__pyx_v_lda), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_w)), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_vl)), (&__pyx_v_ldvl), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_vr)), (&__pyx_v_ldvr), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_work)), (&__pyx_v_lwork), ((double *)PyArray_DATA(__pyx_v_rwork)), (&__pyx_v_info));

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":436
 *                                         <double *>np.PyArray_DATA(rwork),&info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["zgeev_"] = lapack_lite_status__
 *     retval["jobvl"] = jobvl_char
 */
  __pyx_t_7 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_7;
  __pyx_t_7 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":437
 * 
 *     retval = {}
 *     retval["zgeev_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["jobvl"] = jobvl_char
 *     retval["jobvr"] = jobvr_char
 */
  __pyx_t_1 = __pyx_v_lapack_lite_status__;
  __site_setindex_437_10->Target(__site_setindex_437_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"zgeev_"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":438
 *     retval = {}
 *     retval["zgeev_"] = lapack_lite_status__
 *     retval["jobvl"] = jobvl_char             # <<<<<<<<<<<<<<
 *     retval["jobvr"] = jobvr_char
 *     retval["n"] = n
 */
  __pyx_t_1 = __pyx_v_jobvl_char;
  __site_setindex_438_10->Target(__site_setindex_438_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"jobvl"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":439
 *     retval["zgeev_"] = lapack_lite_status__
 *     retval["jobvl"] = jobvl_char
 *     retval["jobvr"] = jobvr_char             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_1 = __pyx_v_jobvr_char;
  __site_setindex_439_10->Target(__site_setindex_439_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"jobvr"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":440
 *     retval["jobvl"] = jobvl_char
 *     retval["jobvr"] = jobvr_char
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["ldvl"] = ldvl
 */
  __pyx_t_1 = __pyx_v_n;
  __site_setindex_440_10->Target(__site_setindex_440_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":441
 *     retval["jobvr"] = jobvr_char
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["ldvl"] = ldvl
 *     retval["ldvr"] = ldvr
 */
  __pyx_t_1 = __pyx_v_lda;
  __site_setindex_441_10->Target(__site_setindex_441_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":442
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["ldvl"] = ldvl             # <<<<<<<<<<<<<<
 *     retval["ldvr"] = ldvr
 *     retval["lwork"] = lwork
 */
  __pyx_t_1 = __pyx_v_ldvl;
  __site_setindex_442_10->Target(__site_setindex_442_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldvl"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":443
 *     retval["lda"] = lda
 *     retval["ldvl"] = ldvl
 *     retval["ldvr"] = ldvr             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 */
  __pyx_t_1 = __pyx_v_ldvr;
  __site_setindex_443_10->Target(__site_setindex_443_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldvr"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":444
 *     retval["ldvl"] = ldvl
 *     retval["ldvr"] = ldvr
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_1 = __pyx_v_lwork;
  __site_setindex_444_10->Target(__site_setindex_444_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":445
 *     retval["ldvr"] = ldvr
 *     retval["lwork"] = lwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_1 = __pyx_v_info;
  __site_setindex_445_10->Target(__site_setindex_445_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":446
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 *     return retval             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = ((System::Object^)__pyx_v_retval);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":449
 * 
 * 
 * def zgelsd(int m, int n, int nrhs, np.ndarray a, int lda,             # <<<<<<<<<<<<<<
 *             np.ndarray b, int ldb, np.ndarray s, double rcond,
 *             int rank, np.ndarray work, int lwork,
 */

static System::Object^ zgelsd(System::Object^ m, System::Object^ n, System::Object^ nrhs, System::Object^ a, System::Object^ lda, System::Object^ b, System::Object^ ldb, System::Object^ s, System::Object^ rcond, System::Object^ rank, System::Object^ work, System::Object^ lwork, System::Object^ rwork, System::Object^ iwork, System::Object^ info) {
  int __pyx_v_m;
  int __pyx_v_n;
  int __pyx_v_nrhs;
  NumpyDotNet::ndarray^ __pyx_v_a = nullptr;
  int __pyx_v_lda;
  NumpyDotNet::ndarray^ __pyx_v_b = nullptr;
  int __pyx_v_ldb;
  NumpyDotNet::ndarray^ __pyx_v_s = nullptr;
  double __pyx_v_rcond;
  int __pyx_v_rank;
  NumpyDotNet::ndarray^ __pyx_v_work = nullptr;
  int __pyx_v_lwork;
  NumpyDotNet::ndarray^ __pyx_v_rwork = nullptr;
  NumpyDotNet::ndarray^ __pyx_v_iwork = nullptr;
  int __pyx_v_info;
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  PythonDictionary^ __pyx_t_2;
  System::Object^ __pyx_t_3 = nullptr;
  __pyx_v_m = __site_cvt_cvt_int_449_0->Target(__site_cvt_cvt_int_449_0, m);
  __pyx_v_n = __site_cvt_cvt_int_449_0_1->Target(__site_cvt_cvt_int_449_0_1, n);
  __pyx_v_nrhs = __site_cvt_cvt_int_449_0_2->Target(__site_cvt_cvt_int_449_0_2, nrhs);
  __pyx_v_a = ((NumpyDotNet::ndarray^)a);
  __pyx_v_lda = __site_cvt_cvt_int_449_0_3->Target(__site_cvt_cvt_int_449_0_3, lda);
  __pyx_v_b = ((NumpyDotNet::ndarray^)b);
  __pyx_v_ldb = __site_cvt_cvt_int_449_0_4->Target(__site_cvt_cvt_int_449_0_4, ldb);
  __pyx_v_s = ((NumpyDotNet::ndarray^)s);
  __pyx_v_rcond = __site_cvt_cvt_double_449_0->Target(__site_cvt_cvt_double_449_0, rcond);
  __pyx_v_rank = __site_cvt_cvt_int_449_0_5->Target(__site_cvt_cvt_int_449_0_5, rank);
  __pyx_v_work = ((NumpyDotNet::ndarray^)work);
  __pyx_v_lwork = __site_cvt_cvt_int_449_0_6->Target(__site_cvt_cvt_int_449_0_6, lwork);
  __pyx_v_rwork = ((NumpyDotNet::ndarray^)rwork);
  __pyx_v_iwork = ((NumpyDotNet::ndarray^)iwork);
  __pyx_v_info = __site_cvt_cvt_int_449_0_7->Target(__site_cvt_cvt_int_449_0_7, info);
  __pyx_v_retval = nullptr;
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr)) {
    throw PythonOps::TypeError("Argument 'a' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_b) == nullptr)) {
    throw PythonOps::TypeError("Argument 'b' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_s) == nullptr)) {
    throw PythonOps::TypeError("Argument 's' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr)) {
    throw PythonOps::TypeError("Argument 'work' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_rwork) == nullptr)) {
    throw PythonOps::TypeError("Argument 'rwork' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_iwork) == nullptr)) {
    throw PythonOps::TypeError("Argument 'iwork' has incorrect type");
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":455
 *     cdef int  lapack_lite_status__
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgelsd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(b,np.NPY_CDOUBLE,"b","np.NPY_CDOUBLE","zgelsd"): return None
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","zgelsd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_a, NPY_CDOUBLE, __pyx_k__a, __pyx_k_3, __pyx_k__zgelsd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":456
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgelsd"): return None
 *     if not check_object(b,np.NPY_CDOUBLE,"b","np.NPY_CDOUBLE","zgelsd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","zgelsd"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgelsd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_b, NPY_CDOUBLE, __pyx_k__b, __pyx_k_3, __pyx_k__zgelsd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":457
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgelsd"): return None
 *     if not check_object(b,np.NPY_CDOUBLE,"b","np.NPY_CDOUBLE","zgelsd"): return None
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","zgelsd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgelsd"): return None
 *     if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgelsd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_s, NPY_DOUBLE, __pyx_k__s, __pyx_k_1, __pyx_k__zgelsd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":458
 *     if not check_object(b,np.NPY_CDOUBLE,"b","np.NPY_CDOUBLE","zgelsd"): return None
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","zgelsd"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgelsd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgelsd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","zgelsd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_work, NPY_CDOUBLE, __pyx_k__work, __pyx_k_3, __pyx_k__zgelsd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L8;
  }
  __pyx_L8:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":459
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","zgelsd"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgelsd"): return None
 *     if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgelsd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","zgelsd"): return None
 * 
 */
  __pyx_t_1 = (!check_object(__pyx_v_rwork, NPY_DOUBLE, __pyx_k__rwork, __pyx_k_1, __pyx_k__zgelsd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L9;
  }
  __pyx_L9:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":460
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgelsd"): return None
 *     if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgelsd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","zgelsd"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_zgelsd(&m,&n,&nrhs,
 */
  __pyx_t_1 = (!check_object(__pyx_v_iwork, NPY_INT, __pyx_k__iwork, __pyx_k_2, __pyx_k__zgelsd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L10;
  }
  __pyx_L10:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":468
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 *                                          <double *>np.PyArray_DATA(rwork),
 *                                          <int *>np.PyArray_DATA(iwork),&info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(zgelsd_)((&__pyx_v_m), (&__pyx_v_n), (&__pyx_v_nrhs), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_a)), (&__pyx_v_lda), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_b)), (&__pyx_v_ldb), ((double *)PyArray_DATA(__pyx_v_s)), (&__pyx_v_rcond), (&__pyx_v_rank), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_work)), (&__pyx_v_lwork), ((double *)PyArray_DATA(__pyx_v_rwork)), ((int *)PyArray_DATA(__pyx_v_iwork)), (&__pyx_v_info));

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":470
 *                                          <int *>np.PyArray_DATA(iwork),&info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["zgelsd_"] = lapack_lite_status__
 *     retval["m"] = m
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":471
 * 
 *     retval = {}
 *     retval["zgelsd_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["m"] = m
 *     retval["n"] = n
 */
  __pyx_t_3 = __pyx_v_lapack_lite_status__;
  __site_setindex_471_10->Target(__site_setindex_471_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"zgelsd_"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":472
 *     retval = {}
 *     retval["zgelsd_"] = lapack_lite_status__
 *     retval["m"] = m             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs
 */
  __pyx_t_3 = __pyx_v_m;
  __site_setindex_472_10->Target(__site_setindex_472_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"m"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":473
 *     retval["zgelsd_"] = lapack_lite_status__
 *     retval["m"] = m
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda
 */
  __pyx_t_3 = __pyx_v_n;
  __site_setindex_473_10->Target(__site_setindex_473_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":474
 *     retval["m"] = m
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb
 */
  __pyx_t_3 = __pyx_v_nrhs;
  __site_setindex_474_10->Target(__site_setindex_474_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"nrhs"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":475
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["ldb"] = ldb
 *     retval["rank"] = rank
 */
  __pyx_t_3 = __pyx_v_lda;
  __site_setindex_475_10->Target(__site_setindex_475_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":476
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb             # <<<<<<<<<<<<<<
 *     retval["rank"] = rank
 *     retval["lwork"] = lwork
 */
  __pyx_t_3 = __pyx_v_ldb;
  __site_setindex_476_10->Target(__site_setindex_476_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldb"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":477
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb
 *     retval["rank"] = rank             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 */
  __pyx_t_3 = __pyx_v_rank;
  __site_setindex_477_10->Target(__site_setindex_477_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"rank"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":478
 *     retval["ldb"] = ldb
 *     retval["rank"] = rank
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_3 = __pyx_v_lwork;
  __site_setindex_478_10->Target(__site_setindex_478_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":479
 *     retval["rank"] = rank
 *     retval["lwork"] = lwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_3 = __pyx_v_info;
  __site_setindex_479_10->Target(__site_setindex_479_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":480
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 *     return retval             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = ((System::Object^)__pyx_v_retval);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":483
 * 
 * 
 * def zgesv(int n, int nrhs, np.ndarray a, int lda,             # <<<<<<<<<<<<<<
 *            np.ndarray ipiv, np.ndarray b, int ldb, int info):
 *     cdef int lapack_lite_status__
 */

static System::Object^ zgesv(System::Object^ n, System::Object^ nrhs, System::Object^ a, System::Object^ lda, System::Object^ ipiv, System::Object^ b, System::Object^ ldb, System::Object^ info) {
  int __pyx_v_n;
  int __pyx_v_nrhs;
  NumpyDotNet::ndarray^ __pyx_v_a = nullptr;
  int __pyx_v_lda;
  NumpyDotNet::ndarray^ __pyx_v_ipiv = nullptr;
  NumpyDotNet::ndarray^ __pyx_v_b = nullptr;
  int __pyx_v_ldb;
  int __pyx_v_info;
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  PythonDictionary^ __pyx_t_2;
  System::Object^ __pyx_t_3 = nullptr;
  __pyx_v_n = __site_cvt_cvt_int_483_0->Target(__site_cvt_cvt_int_483_0, n);
  __pyx_v_nrhs = __site_cvt_cvt_int_483_0_1->Target(__site_cvt_cvt_int_483_0_1, nrhs);
  __pyx_v_a = ((NumpyDotNet::ndarray^)a);
  __pyx_v_lda = __site_cvt_cvt_int_483_0_2->Target(__site_cvt_cvt_int_483_0_2, lda);
  __pyx_v_ipiv = ((NumpyDotNet::ndarray^)ipiv);
  __pyx_v_b = ((NumpyDotNet::ndarray^)b);
  __pyx_v_ldb = __site_cvt_cvt_int_483_0_3->Target(__site_cvt_cvt_int_483_0_3, ldb);
  __pyx_v_info = __site_cvt_cvt_int_483_0_4->Target(__site_cvt_cvt_int_483_0_4, info);
  __pyx_v_retval = nullptr;
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr)) {
    throw PythonOps::TypeError("Argument 'a' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_ipiv) == nullptr)) {
    throw PythonOps::TypeError("Argument 'ipiv' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_b) == nullptr)) {
    throw PythonOps::TypeError("Argument 'b' has incorrect type");
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":487
 *     cdef int lapack_lite_status__
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgesv"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(ipiv,np.NPY_INT,"ipiv","np.NPY_INT","zgesv"): return None
 *     if not check_object(b,np.NPY_CDOUBLE,"b","np.NPY_CDOUBLE","zgesv"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_a, NPY_CDOUBLE, __pyx_k__a, __pyx_k_3, __pyx_k__zgesv));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":488
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgesv"): return None
 *     if not check_object(ipiv,np.NPY_INT,"ipiv","np.NPY_INT","zgesv"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(b,np.NPY_CDOUBLE,"b","np.NPY_CDOUBLE","zgesv"): return None
 * 
 */
  __pyx_t_1 = (!check_object(__pyx_v_ipiv, NPY_INT, __pyx_k__ipiv, __pyx_k_2, __pyx_k__zgesv));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":489
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgesv"): return None
 *     if not check_object(ipiv,np.NPY_INT,"ipiv","np.NPY_INT","zgesv"): return None
 *     if not check_object(b,np.NPY_CDOUBLE,"b","np.NPY_CDOUBLE","zgesv"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_zgesv(&n,&nrhs,
 */
  __pyx_t_1 = (!check_object(__pyx_v_b, NPY_CDOUBLE, __pyx_k__b, __pyx_k_3, __pyx_k__zgesv));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":495
 *                                         <int *>np.PyArray_DATA(ipiv),
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(b),&ldb,
 *                                         &info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(zgesv_)((&__pyx_v_n), (&__pyx_v_nrhs), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_a)), (&__pyx_v_lda), ((int *)PyArray_DATA(__pyx_v_ipiv)), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_b)), (&__pyx_v_ldb), (&__pyx_v_info));

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":497
 *                                         &info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["zgesv_"] = lapack_lite_status__
 *     retval["n"] = n
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":498
 * 
 *     retval = {}
 *     retval["zgesv_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs
 */
  __pyx_t_3 = __pyx_v_lapack_lite_status__;
  __site_setindex_498_10->Target(__site_setindex_498_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"zgesv_"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":499
 *     retval = {}
 *     retval["zgesv_"] = lapack_lite_status__
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda
 */
  __pyx_t_3 = __pyx_v_n;
  __site_setindex_499_10->Target(__site_setindex_499_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":500
 *     retval["zgesv_"] = lapack_lite_status__
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb
 */
  __pyx_t_3 = __pyx_v_nrhs;
  __site_setindex_500_10->Target(__site_setindex_500_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"nrhs"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":501
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["ldb"] = ldb
 *     retval["info"] = info
 */
  __pyx_t_3 = __pyx_v_lda;
  __site_setindex_501_10->Target(__site_setindex_501_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":502
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_3 = __pyx_v_ldb;
  __site_setindex_502_10->Target(__site_setindex_502_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldb"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":503
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_3 = __pyx_v_info;
  __site_setindex_503_10->Target(__site_setindex_503_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":504
 *     retval["ldb"] = ldb
 *     retval["info"] = info
 *     return retval             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = ((System::Object^)__pyx_v_retval);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":507
 * 
 * 
 * def zgesdd(jobz, int m, int n, np.ndarray a, int lda,             # <<<<<<<<<<<<<<
 *             np.ndarray s, np.ndarray u, int ldu, np.ndarray vt, int ldvt,
 *             np.ndarray work, int lwork, np.ndarray rwork, np.ndarray iwork, int info):
 */

static System::Object^ zgesdd(System::Object^ jobz, System::Object^ m, System::Object^ n, System::Object^ a, System::Object^ lda, System::Object^ s, System::Object^ u, System::Object^ ldu, System::Object^ vt, System::Object^ ldvt, System::Object^ work, System::Object^ lwork, System::Object^ rwork, System::Object^ iwork, System::Object^ info) {
  System::Object^ __pyx_v_jobz = nullptr;
  int __pyx_v_m;
  int __pyx_v_n;
  NumpyDotNet::ndarray^ __pyx_v_a = nullptr;
  int __pyx_v_lda;
  NumpyDotNet::ndarray^ __pyx_v_s = nullptr;
  NumpyDotNet::ndarray^ __pyx_v_u = nullptr;
  int __pyx_v_ldu;
  NumpyDotNet::ndarray^ __pyx_v_vt = nullptr;
  int __pyx_v_ldvt;
  NumpyDotNet::ndarray^ __pyx_v_work = nullptr;
  int __pyx_v_lwork;
  NumpyDotNet::ndarray^ __pyx_v_rwork = nullptr;
  NumpyDotNet::ndarray^ __pyx_v_iwork = nullptr;
  int __pyx_v_info;
  int __pyx_v_lapack_lite_status__;
  char __pyx_v_jobz_char;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  char __pyx_t_4;
  int __pyx_t_5;
  PythonDictionary^ __pyx_t_6;
  __pyx_v_jobz = jobz;
  __pyx_v_m = __site_cvt_cvt_int_507_0->Target(__site_cvt_cvt_int_507_0, m);
  __pyx_v_n = __site_cvt_cvt_int_507_0_1->Target(__site_cvt_cvt_int_507_0_1, n);
  __pyx_v_a = ((NumpyDotNet::ndarray^)a);
  __pyx_v_lda = __site_cvt_cvt_int_507_0_2->Target(__site_cvt_cvt_int_507_0_2, lda);
  __pyx_v_s = ((NumpyDotNet::ndarray^)s);
  __pyx_v_u = ((NumpyDotNet::ndarray^)u);
  __pyx_v_ldu = __site_cvt_cvt_int_507_0_3->Target(__site_cvt_cvt_int_507_0_3, ldu);
  __pyx_v_vt = ((NumpyDotNet::ndarray^)vt);
  __pyx_v_ldvt = __site_cvt_cvt_int_507_0_4->Target(__site_cvt_cvt_int_507_0_4, ldvt);
  __pyx_v_work = ((NumpyDotNet::ndarray^)work);
  __pyx_v_lwork = __site_cvt_cvt_int_507_0_5->Target(__site_cvt_cvt_int_507_0_5, lwork);
  __pyx_v_rwork = ((NumpyDotNet::ndarray^)rwork);
  __pyx_v_iwork = ((NumpyDotNet::ndarray^)iwork);
  __pyx_v_info = __site_cvt_cvt_int_507_0_6->Target(__site_cvt_cvt_int_507_0_6, info);
  __pyx_v_retval = nullptr;
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr)) {
    throw PythonOps::TypeError("Argument 'a' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_s) == nullptr)) {
    throw PythonOps::TypeError("Argument 's' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_u) == nullptr)) {
    throw PythonOps::TypeError("Argument 'u' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_vt) == nullptr)) {
    throw PythonOps::TypeError("Argument 'vt' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr)) {
    throw PythonOps::TypeError("Argument 'work' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_rwork) == nullptr)) {
    throw PythonOps::TypeError("Argument 'rwork' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_iwork) == nullptr)) {
    throw PythonOps::TypeError("Argument 'iwork' has incorrect type");
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":511
 *             np.ndarray work, int lwork, np.ndarray rwork, np.ndarray iwork, int info):
 *     cdef int lapack_lite_status__
 *     cdef char jobz_char = ord(jobz[0])             # <<<<<<<<<<<<<<
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgesdd"): return None
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "ord");
  __pyx_t_2 = __site_getindex_511_34->Target(__site_getindex_511_34, __pyx_v_jobz, ((System::Object^)0));
  __pyx_t_3 = __site_call1_511_29->Target(__site_call1_511_29, __pyx_context, __pyx_t_1, __pyx_t_2);
  __pyx_t_1 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_t_4 = __site_cvt_cvt_char_511_29->Target(__site_cvt_cvt_char_511_29, __pyx_t_3);
  __pyx_t_3 = nullptr;
  __pyx_v_jobz_char = __pyx_t_4;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":513
 *     cdef char jobz_char = ord(jobz[0])
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","zgesdd"): return None
 *     if not check_object(u,np.NPY_CDOUBLE,"u","np.NPY_CDOUBLE","zgesdd"): return None
 */
  __pyx_t_5 = (!check_object(__pyx_v_a, NPY_CDOUBLE, __pyx_k__a, __pyx_k_3, __pyx_k__zgesdd));
  if (__pyx_t_5) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":514
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","zgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(u,np.NPY_CDOUBLE,"u","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(vt,np.NPY_CDOUBLE,"vt","np.NPY_CDOUBLE","zgesdd"): return None
 */
  __pyx_t_5 = (!check_object(__pyx_v_s, NPY_DOUBLE, __pyx_k__s, __pyx_k_1, __pyx_k__zgesdd));
  if (__pyx_t_5) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":515
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","zgesdd"): return None
 *     if not check_object(u,np.NPY_CDOUBLE,"u","np.NPY_CDOUBLE","zgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(vt,np.NPY_CDOUBLE,"vt","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgesdd"): return None
 */
  __pyx_t_5 = (!check_object(__pyx_v_u, NPY_CDOUBLE, __pyx_k__u, __pyx_k_3, __pyx_k__zgesdd));
  if (__pyx_t_5) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":516
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","zgesdd"): return None
 *     if not check_object(u,np.NPY_CDOUBLE,"u","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(vt,np.NPY_CDOUBLE,"vt","np.NPY_CDOUBLE","zgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgesdd"): return None
 */
  __pyx_t_5 = (!check_object(__pyx_v_vt, NPY_CDOUBLE, __pyx_k__vt, __pyx_k_3, __pyx_k__zgesdd));
  if (__pyx_t_5) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L8;
  }
  __pyx_L8:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":517
 *     if not check_object(u,np.NPY_CDOUBLE,"u","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(vt,np.NPY_CDOUBLE,"vt","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgesdd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","zgesdd"): return None
 */
  __pyx_t_5 = (!check_object(__pyx_v_work, NPY_CDOUBLE, __pyx_k__work, __pyx_k_3, __pyx_k__zgesdd));
  if (__pyx_t_5) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L9;
  }
  __pyx_L9:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":518
 *     if not check_object(vt,np.NPY_CDOUBLE,"vt","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","zgesdd"): return None
 * 
 */
  __pyx_t_5 = (!check_object(__pyx_v_rwork, NPY_DOUBLE, __pyx_k__rwork, __pyx_k_1, __pyx_k__zgesdd));
  if (__pyx_t_5) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L10;
  }
  __pyx_L10:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":519
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgesdd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","zgesdd"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_zgesdd(&jobz_char,&m,&n,
 */
  __pyx_t_5 = (!check_object(__pyx_v_iwork, NPY_INT, __pyx_k__iwork, __pyx_k_2, __pyx_k__zgesdd));
  if (__pyx_t_5) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L11;
  }
  __pyx_L11:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":528
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 *                                          <double *>np.PyArray_DATA(rwork),
 *                                          <int *>np.PyArray_DATA(iwork),&info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(zgesdd_)((&__pyx_v_jobz_char), (&__pyx_v_m), (&__pyx_v_n), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_a)), (&__pyx_v_lda), ((double *)PyArray_DATA(__pyx_v_s)), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_u)), (&__pyx_v_ldu), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_vt)), (&__pyx_v_ldvt), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_work)), (&__pyx_v_lwork), ((double *)PyArray_DATA(__pyx_v_rwork)), ((int *)PyArray_DATA(__pyx_v_iwork)), (&__pyx_v_info));

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":530
 *                                          <int *>np.PyArray_DATA(iwork),&info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["zgesdd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz_char
 */
  __pyx_t_6 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_6;
  __pyx_t_6 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":531
 * 
 *     retval = {}
 *     retval["zgesdd_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["jobz"] = jobz_char
 *     retval["m"] = m
 */
  __pyx_t_3 = __pyx_v_lapack_lite_status__;
  __site_setindex_531_10->Target(__site_setindex_531_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"zgesdd_"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":532
 *     retval = {}
 *     retval["zgesdd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz_char             # <<<<<<<<<<<<<<
 *     retval["m"] = m
 *     retval["n"] = n
 */
  __pyx_t_3 = __pyx_v_jobz_char;
  __site_setindex_532_10->Target(__site_setindex_532_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"jobz"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":533
 *     retval["zgesdd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz_char
 *     retval["m"] = m             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_3 = __pyx_v_m;
  __site_setindex_533_10->Target(__site_setindex_533_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"m"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":534
 *     retval["jobz"] = jobz_char
 *     retval["m"] = m
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["ldu"] = ldu
 */
  __pyx_t_3 = __pyx_v_n;
  __site_setindex_534_10->Target(__site_setindex_534_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":535
 *     retval["m"] = m
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["ldu"] = ldu
 *     retval["ldvt"] = ldvt
 */
  __pyx_t_3 = __pyx_v_lda;
  __site_setindex_535_10->Target(__site_setindex_535_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":536
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["ldu"] = ldu             # <<<<<<<<<<<<<<
 *     retval["ldvt"] = ldvt
 *     retval["lwork"] = lwork
 */
  __pyx_t_3 = __pyx_v_ldu;
  __site_setindex_536_10->Target(__site_setindex_536_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldu"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":537
 *     retval["lda"] = lda
 *     retval["ldu"] = ldu
 *     retval["ldvt"] = ldvt             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 */
  __pyx_t_3 = __pyx_v_ldvt;
  __site_setindex_537_10->Target(__site_setindex_537_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldvt"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":538
 *     retval["ldu"] = ldu
 *     retval["ldvt"] = ldvt
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_3 = __pyx_v_lwork;
  __site_setindex_538_10->Target(__site_setindex_538_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":539
 *     retval["ldvt"] = ldvt
 *     retval["lwork"] = lwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_3 = __pyx_v_info;
  __site_setindex_539_10->Target(__site_setindex_539_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":540
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 *     return retval             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = ((System::Object^)__pyx_v_retval);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":543
 * 
 * 
 * def zgetrf(int m, int n, np.ndarray a, int lda, np.ndarray ipiv, int info):             # <<<<<<<<<<<<<<
 *     cdef int lapack_lite_status__
 * 
 */

static System::Object^ zgetrf(System::Object^ m, System::Object^ n, System::Object^ a, System::Object^ lda, System::Object^ ipiv, System::Object^ info) {
  int __pyx_v_m;
  int __pyx_v_n;
  NumpyDotNet::ndarray^ __pyx_v_a = nullptr;
  int __pyx_v_lda;
  NumpyDotNet::ndarray^ __pyx_v_ipiv = nullptr;
  int __pyx_v_info;
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  PythonDictionary^ __pyx_t_2;
  System::Object^ __pyx_t_3 = nullptr;
  __pyx_v_m = __site_cvt_cvt_int_543_0->Target(__site_cvt_cvt_int_543_0, m);
  __pyx_v_n = __site_cvt_cvt_int_543_0_1->Target(__site_cvt_cvt_int_543_0_1, n);
  __pyx_v_a = ((NumpyDotNet::ndarray^)a);
  __pyx_v_lda = __site_cvt_cvt_int_543_0_2->Target(__site_cvt_cvt_int_543_0_2, lda);
  __pyx_v_ipiv = ((NumpyDotNet::ndarray^)ipiv);
  __pyx_v_info = __site_cvt_cvt_int_543_0_3->Target(__site_cvt_cvt_int_543_0_3, info);
  __pyx_v_retval = nullptr;
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr)) {
    throw PythonOps::TypeError("Argument 'a' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_ipiv) == nullptr)) {
    throw PythonOps::TypeError("Argument 'ipiv' has incorrect type");
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":546
 *     cdef int lapack_lite_status__
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgetrf"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(ipiv,np.NPY_INT,"ipiv","np.NPY_INT","zgetrf"): return None
 * 
 */
  __pyx_t_1 = (!check_object(__pyx_v_a, NPY_CDOUBLE, __pyx_k__a, __pyx_k_3, __pyx_k__zgetrf));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":547
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgetrf"): return None
 *     if not check_object(ipiv,np.NPY_INT,"ipiv","np.NPY_INT","zgetrf"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_zgetrf(&m,&n,
 */
  __pyx_t_1 = (!check_object(__pyx_v_ipiv, NPY_INT, __pyx_k__ipiv, __pyx_k_2, __pyx_k__zgetrf));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":551
 *     lapack_lite_status__ = lapack_zgetrf(&m,&n,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
 *                                          <int *>np.PyArray_DATA(ipiv),&info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(zgetrf_)((&__pyx_v_m), (&__pyx_v_n), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_a)), (&__pyx_v_lda), ((int *)PyArray_DATA(__pyx_v_ipiv)), (&__pyx_v_info));

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":553
 *                                          <int *>np.PyArray_DATA(ipiv),&info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["zgetrf_"] = lapack_lite_status__
 *     retval["m"] = m
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":554
 * 
 *     retval = {}
 *     retval["zgetrf_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["m"] = m
 *     retval["n"] = n
 */
  __pyx_t_3 = __pyx_v_lapack_lite_status__;
  __site_setindex_554_10->Target(__site_setindex_554_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"zgetrf_"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":555
 *     retval = {}
 *     retval["zgetrf_"] = lapack_lite_status__
 *     retval["m"] = m             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_3 = __pyx_v_m;
  __site_setindex_555_10->Target(__site_setindex_555_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"m"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":556
 *     retval["zgetrf_"] = lapack_lite_status__
 *     retval["m"] = m
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["info"] = info
 */
  __pyx_t_3 = __pyx_v_n;
  __site_setindex_556_10->Target(__site_setindex_556_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":557
 *     retval["m"] = m
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_3 = __pyx_v_lda;
  __site_setindex_557_10->Target(__site_setindex_557_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":558
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_3 = __pyx_v_info;
  __site_setindex_558_10->Target(__site_setindex_558_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":559
 *     retval["lda"] = lda
 *     retval["info"] = info
 *     return retval             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = ((System::Object^)__pyx_v_retval);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":562
 * 
 * 
 * def zpotrf(uplo, int n, np.ndarray a, int lda, int info):             # <<<<<<<<<<<<<<
 *     cdef int  lapack_lite_status__
 *     cdef char uplo_char = ord(uplo[0])
 */

static System::Object^ zpotrf(System::Object^ uplo, System::Object^ n, System::Object^ a, System::Object^ lda, System::Object^ info) {
  System::Object^ __pyx_v_uplo = nullptr;
  int __pyx_v_n;
  NumpyDotNet::ndarray^ __pyx_v_a = nullptr;
  int __pyx_v_lda;
  int __pyx_v_info;
  int __pyx_v_lapack_lite_status__;
  char __pyx_v_uplo_char;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  char __pyx_t_4;
  int __pyx_t_5;
  PythonDictionary^ __pyx_t_6;
  __pyx_v_uplo = uplo;
  __pyx_v_n = __site_cvt_cvt_int_562_0->Target(__site_cvt_cvt_int_562_0, n);
  __pyx_v_a = ((NumpyDotNet::ndarray^)a);
  __pyx_v_lda = __site_cvt_cvt_int_562_0_1->Target(__site_cvt_cvt_int_562_0_1, lda);
  __pyx_v_info = __site_cvt_cvt_int_562_0_2->Target(__site_cvt_cvt_int_562_0_2, info);
  __pyx_v_retval = nullptr;
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr)) {
    throw PythonOps::TypeError("Argument 'a' has incorrect type");
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":564
 * def zpotrf(uplo, int n, np.ndarray a, int lda, int info):
 *     cdef int  lapack_lite_status__
 *     cdef char uplo_char = ord(uplo[0])             # <<<<<<<<<<<<<<
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zpotrf"): return None
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "ord");
  __pyx_t_2 = __site_getindex_564_34->Target(__site_getindex_564_34, __pyx_v_uplo, ((System::Object^)0));
  __pyx_t_3 = __site_call1_564_29->Target(__site_call1_564_29, __pyx_context, __pyx_t_1, __pyx_t_2);
  __pyx_t_1 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_t_4 = __site_cvt_cvt_char_564_29->Target(__site_cvt_cvt_char_564_29, __pyx_t_3);
  __pyx_t_3 = nullptr;
  __pyx_v_uplo_char = __pyx_t_4;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":566
 *     cdef char uplo_char = ord(uplo[0])
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zpotrf"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_zpotrf(&uplo_char,&n,
 */
  __pyx_t_5 = (!check_object(__pyx_v_a, NPY_CDOUBLE, __pyx_k__a, __pyx_k_3, __pyx_k__zpotrf));
  if (__pyx_t_5) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":570
 *     lapack_lite_status__ = lapack_zpotrf(&uplo_char,&n,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
 *                                          &info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(zpotrf_)((&__pyx_v_uplo_char), (&__pyx_v_n), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_a)), (&__pyx_v_lda), (&__pyx_v_info));

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":572
 *                                          &info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["zpotrf_"] = lapack_lite_status__
 *     retval["n"] = n
 */
  __pyx_t_6 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_6;
  __pyx_t_6 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":573
 * 
 *     retval = {}
 *     retval["zpotrf_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_3 = __pyx_v_lapack_lite_status__;
  __site_setindex_573_10->Target(__site_setindex_573_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"zpotrf_"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":574
 *     retval = {}
 *     retval["zpotrf_"] = lapack_lite_status__
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["info"] = info
 */
  __pyx_t_3 = __pyx_v_n;
  __site_setindex_574_10->Target(__site_setindex_574_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":575
 *     retval["zpotrf_"] = lapack_lite_status__
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_3 = __pyx_v_lda;
  __site_setindex_575_10->Target(__site_setindex_575_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":576
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_3 = __pyx_v_info;
  __site_setindex_576_10->Target(__site_setindex_576_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":577
 *     retval["lda"] = lda
 *     retval["info"] = info
 *     return retval             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = ((System::Object^)__pyx_v_retval);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":580
 * 
 * 
 * def zgeqrf(int m, int n, np.ndarray a, int lda,             # <<<<<<<<<<<<<<
 *             np.ndarray tau, np.ndarray work, int lwork, int info):
 *     cdef int lapack_lite_status__
 */

static System::Object^ zgeqrf(System::Object^ m, System::Object^ n, System::Object^ a, System::Object^ lda, System::Object^ tau, System::Object^ work, System::Object^ lwork, System::Object^ info) {
  int __pyx_v_m;
  int __pyx_v_n;
  NumpyDotNet::ndarray^ __pyx_v_a = nullptr;
  int __pyx_v_lda;
  NumpyDotNet::ndarray^ __pyx_v_tau = nullptr;
  NumpyDotNet::ndarray^ __pyx_v_work = nullptr;
  int __pyx_v_lwork;
  int __pyx_v_info;
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  PythonDictionary^ __pyx_t_2;
  System::Object^ __pyx_t_3 = nullptr;
  __pyx_v_m = __site_cvt_cvt_int_580_0->Target(__site_cvt_cvt_int_580_0, m);
  __pyx_v_n = __site_cvt_cvt_int_580_0_1->Target(__site_cvt_cvt_int_580_0_1, n);
  __pyx_v_a = ((NumpyDotNet::ndarray^)a);
  __pyx_v_lda = __site_cvt_cvt_int_580_0_2->Target(__site_cvt_cvt_int_580_0_2, lda);
  __pyx_v_tau = ((NumpyDotNet::ndarray^)tau);
  __pyx_v_work = ((NumpyDotNet::ndarray^)work);
  __pyx_v_lwork = __site_cvt_cvt_int_580_0_3->Target(__site_cvt_cvt_int_580_0_3, lwork);
  __pyx_v_info = __site_cvt_cvt_int_580_0_4->Target(__site_cvt_cvt_int_580_0_4, info);
  __pyx_v_retval = nullptr;
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr)) {
    throw PythonOps::TypeError("Argument 'a' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_tau) == nullptr)) {
    throw PythonOps::TypeError("Argument 'tau' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr)) {
    throw PythonOps::TypeError("Argument 'work' has incorrect type");
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":585
 * 
 *     # check objects and convert to right storage order
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgeqrf"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(tau,np.NPY_CDOUBLE,"tau","np.NPY_CDOUBLE","zgeqrf"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgeqrf"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_a, NPY_CDOUBLE, __pyx_k__a, __pyx_k_3, __pyx_k__zgeqrf));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":586
 *     # check objects and convert to right storage order
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgeqrf"): return None
 *     if not check_object(tau,np.NPY_CDOUBLE,"tau","np.NPY_CDOUBLE","zgeqrf"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgeqrf"): return None
 * 
 */
  __pyx_t_1 = (!check_object(__pyx_v_tau, NPY_CDOUBLE, __pyx_k__tau, __pyx_k_3, __pyx_k__zgeqrf));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":587
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgeqrf"): return None
 *     if not check_object(tau,np.NPY_CDOUBLE,"tau","np.NPY_CDOUBLE","zgeqrf"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgeqrf"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_zgeqrf(&m, &n,
 */
  __pyx_t_1 = (!check_object(__pyx_v_work, NPY_CDOUBLE, __pyx_k__work, __pyx_k_3, __pyx_k__zgeqrf));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":593
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(tau),
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work), &lwork,
 *                                          &info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(zgeqrf_)((&__pyx_v_m), (&__pyx_v_n), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_a)), (&__pyx_v_lda), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_tau)), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_work)), (&__pyx_v_lwork), (&__pyx_v_info));

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":595
 *                                          &info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["zgeqrf_"] = lapack_lite_status__
 *     retval["m"] = m
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":596
 * 
 *     retval = {}
 *     retval["zgeqrf_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["m"] = m
 *     retval["n"] = n
 */
  __pyx_t_3 = __pyx_v_lapack_lite_status__;
  __site_setindex_596_10->Target(__site_setindex_596_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"zgeqrf_"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":597
 *     retval = {}
 *     retval["zgeqrf_"] = lapack_lite_status__
 *     retval["m"] = m             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_3 = __pyx_v_m;
  __site_setindex_597_10->Target(__site_setindex_597_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"m"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":598
 *     retval["zgeqrf_"] = lapack_lite_status__
 *     retval["m"] = m
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork
 */
  __pyx_t_3 = __pyx_v_n;
  __site_setindex_598_10->Target(__site_setindex_598_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":599
 *     retval["m"] = m
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 */
  __pyx_t_3 = __pyx_v_lda;
  __site_setindex_599_10->Target(__site_setindex_599_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":600
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_3 = __pyx_v_lwork;
  __site_setindex_600_10->Target(__site_setindex_600_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":601
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_3 = __pyx_v_info;
  __site_setindex_601_10->Target(__site_setindex_601_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":602
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 *     return retval             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = ((System::Object^)__pyx_v_retval);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":605
 * 
 * 
 * def zungqr(int m, int n, int k, np.ndarray a, int lda,             # <<<<<<<<<<<<<<
 *             np.ndarray tau, np.ndarray work, int lwork, int info):
 *     cdef int  lapack_lite_status__
 */

static System::Object^ zungqr(System::Object^ m, System::Object^ n, System::Object^ k, System::Object^ a, System::Object^ lda, System::Object^ tau, System::Object^ work, System::Object^ lwork, System::Object^ info) {
  int __pyx_v_m;
  int __pyx_v_n;
  int __pyx_v_k;
  NumpyDotNet::ndarray^ __pyx_v_a = nullptr;
  int __pyx_v_lda;
  NumpyDotNet::ndarray^ __pyx_v_tau = nullptr;
  NumpyDotNet::ndarray^ __pyx_v_work = nullptr;
  int __pyx_v_lwork;
  int __pyx_v_info;
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  PythonDictionary^ __pyx_t_2;
  System::Object^ __pyx_t_3 = nullptr;
  __pyx_v_m = __site_cvt_cvt_int_605_0->Target(__site_cvt_cvt_int_605_0, m);
  __pyx_v_n = __site_cvt_cvt_int_605_0_1->Target(__site_cvt_cvt_int_605_0_1, n);
  __pyx_v_k = __site_cvt_cvt_int_605_0_2->Target(__site_cvt_cvt_int_605_0_2, k);
  __pyx_v_a = ((NumpyDotNet::ndarray^)a);
  __pyx_v_lda = __site_cvt_cvt_int_605_0_3->Target(__site_cvt_cvt_int_605_0_3, lda);
  __pyx_v_tau = ((NumpyDotNet::ndarray^)tau);
  __pyx_v_work = ((NumpyDotNet::ndarray^)work);
  __pyx_v_lwork = __site_cvt_cvt_int_605_0_4->Target(__site_cvt_cvt_int_605_0_4, lwork);
  __pyx_v_info = __site_cvt_cvt_int_605_0_5->Target(__site_cvt_cvt_int_605_0_5, info);
  __pyx_v_retval = nullptr;
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr)) {
    throw PythonOps::TypeError("Argument 'a' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_tau) == nullptr)) {
    throw PythonOps::TypeError("Argument 'tau' has incorrect type");
  }
  if (unlikely(dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr)) {
    throw PythonOps::TypeError("Argument 'work' has incorrect type");
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":609
 *     cdef int  lapack_lite_status__
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zungqr"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(tau,np.NPY_CDOUBLE,"tau","np.NPY_CDOUBLE","zungqr"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zungqr"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_a, NPY_CDOUBLE, __pyx_k__a, __pyx_k_3, __pyx_k__zungqr));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":610
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zungqr"): return None
 *     if not check_object(tau,np.NPY_CDOUBLE,"tau","np.NPY_CDOUBLE","zungqr"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zungqr"): return None
 * 
 */
  __pyx_t_1 = (!check_object(__pyx_v_tau, NPY_CDOUBLE, __pyx_k__tau, __pyx_k_3, __pyx_k__zungqr));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":611
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zungqr"): return None
 *     if not check_object(tau,np.NPY_CDOUBLE,"tau","np.NPY_CDOUBLE","zungqr"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zungqr"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_zungqr(&m, &n, &k,
 */
  __pyx_t_1 = (!check_object(__pyx_v_work, NPY_CDOUBLE, __pyx_k__work, __pyx_k_3, __pyx_k__zungqr));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":616
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a), &lda,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(tau),
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,&info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(zungqr_)((&__pyx_v_m), (&__pyx_v_n), (&__pyx_v_k), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_a)), (&__pyx_v_lda), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_tau)), ((f2c_doublecomplex *)PyArray_DATA(__pyx_v_work)), (&__pyx_v_lwork), (&__pyx_v_info));

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":618
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,&info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["zungqr_"] = lapack_lite_status__
 *     retval["info"] = info
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":619
 * 
 *     retval = {}
 *     retval["zungqr_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_3 = __pyx_v_lapack_lite_status__;
  __site_setindex_619_10->Target(__site_setindex_619_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"zungqr_"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":620
 *     retval = {}
 *     retval["zungqr_"] = lapack_lite_status__
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_3 = __pyx_v_info;
  __site_setindex_620_10->Target(__site_setindex_620_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":621
 *     retval["zungqr_"] = lapack_lite_status__
 *     retval["info"] = info
 *     return retval             # <<<<<<<<<<<<<<
 * 
 */
  __pyx_r = ((System::Object^)__pyx_v_retval);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":292
 *     dtype NpyArray_FindArrayType_3args "NumpyDotNet::NpyArray::FindArrayType" (object src, dtype minitype, int max)
 * 
 * cdef inline dtype NpyArray_FindArrayType_2args(object src, dtype minitype):             # <<<<<<<<<<<<<<
 *     return NpyArray_FindArrayType_3args(src, minitype, NPY_MAXDIMS)
 * 
 */

static CYTHON_INLINE NumpyDotNet::dtype^ NpyArray_FindArrayType_2args(System::Object^ __pyx_v_src, NumpyDotNet::dtype^ __pyx_v_minitype) {
  NumpyDotNet::dtype^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":293
 * 
 * cdef inline dtype NpyArray_FindArrayType_2args(object src, dtype minitype):
 *     return NpyArray_FindArrayType_3args(src, minitype, NPY_MAXDIMS)             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_t_1 = ((System::Object^)NumpyDotNet::NpyArray::FindArrayType(__pyx_v_src, __pyx_v_minitype, NPY_MAXDIMS)); 
  __pyx_r = ((NumpyDotNet::dtype^)__pyx_t_1);
  __pyx_t_1 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":312
 * ctypedef void (*PyArray_CopySwapFunc)(void *, void *, int, NpyArray *)
 * 
 * cdef inline object PyUFunc_FromFuncAndData(PyUFuncGenericFunction* func, void** data,             # <<<<<<<<<<<<<<
 *         char* types, int ntypes, int nin, int nout,
 *         int identity, char* name, char* doc, int c):
 */

static CYTHON_INLINE System::Object^ PyUFunc_FromFuncAndData(__pyx_t_5numpy_6linalg_5numpy_PyUFuncGenericFunction *__pyx_v_func, void **__pyx_v_data, char *__pyx_v_types, int __pyx_v_ntypes, int __pyx_v_nin, int __pyx_v_nout, int __pyx_v_identity, char *__pyx_v_name, char *__pyx_v_doc, int __pyx_v_c) {
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":315
 *         char* types, int ntypes, int nin, int nout,
 *         int identity, char* name, char* doc, int c):
 *    return Npy_INTERFACE_ufunc(NpyUFunc_FromFuncAndDataAndSignature(func, data, types, ntypes, nin, nout, identity, name, doc, c, NULL))             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_DescrFromType(int typenum):
 */
  __pyx_t_1 = Npy_INTERFACE_OBJECT(NpyUFunc_FromFuncAndDataAndSignature(__pyx_v_func, __pyx_v_data, __pyx_v_types, __pyx_v_ntypes, __pyx_v_nin, __pyx_v_nout, __pyx_v_identity, __pyx_v_name, __pyx_v_doc, __pyx_v_c, NULL)); 
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":317
 *    return Npy_INTERFACE_ufunc(NpyUFunc_FromFuncAndDataAndSignature(func, data, types, ntypes, nin, nout, identity, name, doc, c, NULL))
 * 
 * cdef inline object PyArray_DescrFromType(int typenum):             # <<<<<<<<<<<<<<
 *     return Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum))
 * 
 */

static CYTHON_INLINE System::Object^ PyArray_DescrFromType(int __pyx_v_typenum) {
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":318
 * 
 * cdef inline object PyArray_DescrFromType(int typenum):
 *     return Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum))             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_t_1 = ((System::Object^)Npy_INTERFACE_OBJECT(NpyArray_DescrFromType(__pyx_v_typenum))); 
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":321
 * 
 * 
 * cdef inline object PyArray_ZEROS(int ndim, npy_intp *shape, int typenum, int fortran):             # <<<<<<<<<<<<<<
 *     shape_list = []
 *     cdef int i
 */

static CYTHON_INLINE System::Object^ PyArray_ZEROS(int __pyx_v_ndim, __pyx_t_5numpy_6linalg_5numpy_npy_intp *__pyx_v_shape, int __pyx_v_typenum, int __pyx_v_fortran) {
  System::Object^ __pyx_v_shape_list;
  int __pyx_v_i;
  System::Object^ __pyx_v_numpy;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  int __pyx_t_2;
  int __pyx_t_3;
  System::Object^ __pyx_t_4 = nullptr;
  System::Object^ __pyx_t_5 = nullptr;
  System::Object^ __pyx_t_6 = nullptr;
  __pyx_v_shape_list = nullptr;
  __pyx_v_numpy = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":322
 * 
 * cdef inline object PyArray_ZEROS(int ndim, npy_intp *shape, int typenum, int fortran):
 *     shape_list = []             # <<<<<<<<<<<<<<
 *     cdef int i
 *     for i in range(ndim):
 */
  __pyx_t_1 = PythonOps::MakeListNoCopy(gcnew array<System::Object^>{});
  __pyx_v_shape_list = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":324
 *     shape_list = []
 *     cdef int i
 *     for i in range(ndim):             # <<<<<<<<<<<<<<
 *         shape_list.append(shape[i])
 *     import numpy
 */
  __pyx_t_2 = __pyx_v_ndim;
  for (__pyx_t_3 = 0; __pyx_t_3 < __pyx_t_2; __pyx_t_3+=1) {
    __pyx_v_i = __pyx_t_3;

    /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":325
 *     cdef int i
 *     for i in range(ndim):
 *         shape_list.append(shape[i])             # <<<<<<<<<<<<<<
 *     import numpy
 *     return numpy.zeros(shape_list, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), 'F' if fortran else 'C')
 */
    __pyx_t_1 = __site_get_append_325_18->Target(__site_get_append_325_18, ((System::Object^)__pyx_v_shape_list), __pyx_context);
    __pyx_t_4 = (__pyx_v_shape[__pyx_v_i]);
    __pyx_t_5 = __site_call1_325_25->Target(__site_call1_325_25, __pyx_context, __pyx_t_1, __pyx_t_4);
    __pyx_t_1 = nullptr;
    __pyx_t_4 = nullptr;
    __pyx_t_5 = nullptr;
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":326
 *     for i in range(ndim):
 *         shape_list.append(shape[i])
 *     import numpy             # <<<<<<<<<<<<<<
 *     return numpy.zeros(shape_list, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), 'F' if fortran else 'C')
 * 
 */
  __pyx_t_5 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "numpy", -1));
  __pyx_v_numpy = __pyx_t_5;
  __pyx_t_5 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":327
 *         shape_list.append(shape[i])
 *     import numpy
 *     return numpy.zeros(shape_list, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), 'F' if fortran else 'C')             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_EMPTY(int ndim, npy_intp *shape, int typenum, int fortran):
 */
  __pyx_t_5 = __site_get_zeros_327_16->Target(__site_get_zeros_327_16, __pyx_v_numpy, __pyx_context);
  __pyx_t_4 = ((System::Object^)Npy_INTERFACE_OBJECT(NpyArray_DescrFromType(__pyx_v_typenum))); 
  if (__pyx_v_fortran) {
    __pyx_t_1 = "F";
  } else {
    __pyx_t_1 = "C";
  }
  __pyx_t_6 = __site_call3_327_22->Target(__site_call3_327_22, __pyx_context, __pyx_t_5, ((System::Object^)__pyx_v_shape_list), __pyx_t_4, ((System::Object^)__pyx_t_1));
  __pyx_t_5 = nullptr;
  __pyx_t_4 = nullptr;
  __pyx_t_1 = nullptr;
  __pyx_r = __pyx_t_6;
  __pyx_t_6 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":329
 *     return numpy.zeros(shape_list, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), 'F' if fortran else 'C')
 * 
 * cdef inline object PyArray_EMPTY(int ndim, npy_intp *shape, int typenum, int fortran):             # <<<<<<<<<<<<<<
 *     shape_list = []
 *     cdef int i
 */

static CYTHON_INLINE System::Object^ PyArray_EMPTY(int __pyx_v_ndim, __pyx_t_5numpy_6linalg_5numpy_npy_intp *__pyx_v_shape, int __pyx_v_typenum, int __pyx_v_fortran) {
  System::Object^ __pyx_v_shape_list;
  int __pyx_v_i;
  System::Object^ __pyx_v_numpy;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  int __pyx_t_2;
  int __pyx_t_3;
  System::Object^ __pyx_t_4 = nullptr;
  System::Object^ __pyx_t_5 = nullptr;
  System::Object^ __pyx_t_6 = nullptr;
  __pyx_v_shape_list = nullptr;
  __pyx_v_numpy = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":330
 * 
 * cdef inline object PyArray_EMPTY(int ndim, npy_intp *shape, int typenum, int fortran):
 *     shape_list = []             # <<<<<<<<<<<<<<
 *     cdef int i
 *     for i in range(ndim):
 */
  __pyx_t_1 = PythonOps::MakeListNoCopy(gcnew array<System::Object^>{});
  __pyx_v_shape_list = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":332
 *     shape_list = []
 *     cdef int i
 *     for i in range(ndim):             # <<<<<<<<<<<<<<
 *         shape_list.append(shape[i])
 *     import numpy
 */
  __pyx_t_2 = __pyx_v_ndim;
  for (__pyx_t_3 = 0; __pyx_t_3 < __pyx_t_2; __pyx_t_3+=1) {
    __pyx_v_i = __pyx_t_3;

    /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":333
 *     cdef int i
 *     for i in range(ndim):
 *         shape_list.append(shape[i])             # <<<<<<<<<<<<<<
 *     import numpy
 *     return numpy.empty(shape_list, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), 'F' if fortran else 'C')
 */
    __pyx_t_1 = __site_get_append_333_18->Target(__site_get_append_333_18, ((System::Object^)__pyx_v_shape_list), __pyx_context);
    __pyx_t_4 = (__pyx_v_shape[__pyx_v_i]);
    __pyx_t_5 = __site_call1_333_25->Target(__site_call1_333_25, __pyx_context, __pyx_t_1, __pyx_t_4);
    __pyx_t_1 = nullptr;
    __pyx_t_4 = nullptr;
    __pyx_t_5 = nullptr;
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":334
 *     for i in range(ndim):
 *         shape_list.append(shape[i])
 *     import numpy             # <<<<<<<<<<<<<<
 *     return numpy.empty(shape_list, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), 'F' if fortran else 'C')
 * 
 */
  __pyx_t_5 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "numpy", -1));
  __pyx_v_numpy = __pyx_t_5;
  __pyx_t_5 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":335
 *         shape_list.append(shape[i])
 *     import numpy
 *     return numpy.empty(shape_list, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), 'F' if fortran else 'C')             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_Empty(int nd, npy_intp *dims, dtype descr, int fortran):
 */
  __pyx_t_5 = __site_get_empty_335_16->Target(__site_get_empty_335_16, __pyx_v_numpy, __pyx_context);
  __pyx_t_4 = ((System::Object^)Npy_INTERFACE_OBJECT(NpyArray_DescrFromType(__pyx_v_typenum))); 
  if (__pyx_v_fortran) {
    __pyx_t_1 = "F";
  } else {
    __pyx_t_1 = "C";
  }
  __pyx_t_6 = __site_call3_335_22->Target(__site_call3_335_22, __pyx_context, __pyx_t_5, ((System::Object^)__pyx_v_shape_list), __pyx_t_4, ((System::Object^)__pyx_t_1));
  __pyx_t_5 = nullptr;
  __pyx_t_4 = nullptr;
  __pyx_t_1 = nullptr;
  __pyx_r = __pyx_t_6;
  __pyx_t_6 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":337
 *     return numpy.empty(shape_list, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), 'F' if fortran else 'C')
 * 
 * cdef inline object PyArray_Empty(int nd, npy_intp *dims, dtype descr, int fortran):             # <<<<<<<<<<<<<<
 *     shape_list = []
 *     cdef int i
 */

static CYTHON_INLINE System::Object^ PyArray_Empty(int __pyx_v_nd, __pyx_t_5numpy_6linalg_5numpy_npy_intp *__pyx_v_dims, NumpyDotNet::dtype^ __pyx_v_descr, int __pyx_v_fortran) {
  System::Object^ __pyx_v_shape_list;
  int __pyx_v_i;
  System::Object^ __pyx_v_numpy;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  int __pyx_t_2;
  int __pyx_t_3;
  System::Object^ __pyx_t_4 = nullptr;
  System::Object^ __pyx_t_5 = nullptr;
  __pyx_v_shape_list = nullptr;
  __pyx_v_numpy = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":338
 * 
 * cdef inline object PyArray_Empty(int nd, npy_intp *dims, dtype descr, int fortran):
 *     shape_list = []             # <<<<<<<<<<<<<<
 *     cdef int i
 *     for i in range(nd):
 */
  __pyx_t_1 = PythonOps::MakeListNoCopy(gcnew array<System::Object^>{});
  __pyx_v_shape_list = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":340
 *     shape_list = []
 *     cdef int i
 *     for i in range(nd):             # <<<<<<<<<<<<<<
 *         shape_list.append(dims[i])
 *     import numpy
 */
  __pyx_t_2 = __pyx_v_nd;
  for (__pyx_t_3 = 0; __pyx_t_3 < __pyx_t_2; __pyx_t_3+=1) {
    __pyx_v_i = __pyx_t_3;

    /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":341
 *     cdef int i
 *     for i in range(nd):
 *         shape_list.append(dims[i])             # <<<<<<<<<<<<<<
 *     import numpy
 *     return numpy.empty(shape_list, descr, 'F' if fortran else 'C')
 */
    __pyx_t_1 = __site_get_append_341_18->Target(__site_get_append_341_18, ((System::Object^)__pyx_v_shape_list), __pyx_context);
    __pyx_t_4 = (__pyx_v_dims[__pyx_v_i]);
    __pyx_t_5 = __site_call1_341_25->Target(__site_call1_341_25, __pyx_context, __pyx_t_1, __pyx_t_4);
    __pyx_t_1 = nullptr;
    __pyx_t_4 = nullptr;
    __pyx_t_5 = nullptr;
  }

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":342
 *     for i in range(nd):
 *         shape_list.append(dims[i])
 *     import numpy             # <<<<<<<<<<<<<<
 *     return numpy.empty(shape_list, descr, 'F' if fortran else 'C')
 * 
 */
  __pyx_t_5 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "numpy", -1));
  __pyx_v_numpy = __pyx_t_5;
  __pyx_t_5 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":343
 *         shape_list.append(dims[i])
 *     import numpy
 *     return numpy.empty(shape_list, descr, 'F' if fortran else 'C')             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_t_5 = __site_get_empty_343_16->Target(__site_get_empty_343_16, __pyx_v_numpy, __pyx_context);
  if (__pyx_v_fortran) {
    __pyx_t_4 = "F";
  } else {
    __pyx_t_4 = "C";
  }
  __pyx_t_1 = __site_call3_343_22->Target(__site_call3_343_22, __pyx_context, __pyx_t_5, ((System::Object^)__pyx_v_shape_list), ((System::Object^)__pyx_v_descr), ((System::Object^)__pyx_t_4));
  __pyx_t_5 = nullptr;
  __pyx_t_4 = nullptr;
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":346
 * 
 * 
 * cdef inline object PyArray_New(void *subtype, int nd, npy_intp *dims, int type_num, npy_intp *strides, void *data, int itemsize, int flags, void *obj):             # <<<<<<<<<<<<<<
 *     assert subtype == NULL
 *     assert obj == NULL
 */

static CYTHON_INLINE System::Object^ PyArray_New(void *__pyx_v_subtype, int __pyx_v_nd, __pyx_t_5numpy_6linalg_5numpy_npy_intp *__pyx_v_dims, int __pyx_v_type_num, __pyx_t_5numpy_6linalg_5numpy_npy_intp *__pyx_v_strides, void *__pyx_v_data, int __pyx_v_itemsize, int __pyx_v_flags, void *__pyx_v_obj) {
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":347
 * 
 * cdef inline object PyArray_New(void *subtype, int nd, npy_intp *dims, int type_num, npy_intp *strides, void *data, int itemsize, int flags, void *obj):
 *     assert subtype == NULL             # <<<<<<<<<<<<<<
 *     assert obj == NULL
 *     return Npy_INTERFACE_array(NpyArray_New(subtype, nd, dims, type_num, strides, data, itemsize, flags, obj))
 */
  #ifndef PYREX_WITHOUT_ASSERTIONS
  if (unlikely(!(__pyx_v_subtype == NULL))) {
    PythonOps::RaiseAssertionError(nullptr);
  }
  #endif

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":348
 * cdef inline object PyArray_New(void *subtype, int nd, npy_intp *dims, int type_num, npy_intp *strides, void *data, int itemsize, int flags, void *obj):
 *     assert subtype == NULL
 *     assert obj == NULL             # <<<<<<<<<<<<<<
 *     return Npy_INTERFACE_array(NpyArray_New(subtype, nd, dims, type_num, strides, data, itemsize, flags, obj))
 * 
 */
  #ifndef PYREX_WITHOUT_ASSERTIONS
  if (unlikely(!(__pyx_v_obj == NULL))) {
    PythonOps::RaiseAssertionError(nullptr);
  }
  #endif

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":349
 *     assert subtype == NULL
 *     assert obj == NULL
 *     return Npy_INTERFACE_array(NpyArray_New(subtype, nd, dims, type_num, strides, data, itemsize, flags, obj))             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_SimpleNew(int nd, npy_intp *dims, int type_num):
 */
  __pyx_t_1 = ((System::Object^)Npy_INTERFACE_OBJECT(NpyArray_New(__pyx_v_subtype, __pyx_v_nd, __pyx_v_dims, __pyx_v_type_num, __pyx_v_strides, __pyx_v_data, __pyx_v_itemsize, __pyx_v_flags, __pyx_v_obj))); 
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":351
 *     return Npy_INTERFACE_array(NpyArray_New(subtype, nd, dims, type_num, strides, data, itemsize, flags, obj))
 * 
 * cdef inline object PyArray_SimpleNew(int nd, npy_intp *dims, int type_num):             # <<<<<<<<<<<<<<
 *     return PyArray_New(NULL, nd, dims, type_num, NULL, NULL, 0, NPY_CARRAY, NULL)
 * 
 */

static CYTHON_INLINE System::Object^ PyArray_SimpleNew(int __pyx_v_nd, __pyx_t_5numpy_6linalg_5numpy_npy_intp *__pyx_v_dims, int __pyx_v_type_num) {
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":352
 * 
 * cdef inline object PyArray_SimpleNew(int nd, npy_intp *dims, int type_num):
 *     return PyArray_New(NULL, nd, dims, type_num, NULL, NULL, 0, NPY_CARRAY, NULL)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_SimpleNewFromData(int nd, npy_intp *dims, int type_num, void *data):
 */
  __pyx_t_1 = PyArray_New(NULL, __pyx_v_nd, __pyx_v_dims, __pyx_v_type_num, NULL, NULL, 0, NPY_CARRAY, NULL); 
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":354
 *     return PyArray_New(NULL, nd, dims, type_num, NULL, NULL, 0, NPY_CARRAY, NULL)
 * 
 * cdef inline object PyArray_SimpleNewFromData(int nd, npy_intp *dims, int type_num, void *data):             # <<<<<<<<<<<<<<
 *     return PyArray_New(NULL, nd, dims, type_num, NULL, data, 0, NPY_CARRAY, NULL)
 * 
 */

static CYTHON_INLINE System::Object^ PyArray_SimpleNewFromData(int __pyx_v_nd, __pyx_t_5numpy_6linalg_5numpy_npy_intp *__pyx_v_dims, int __pyx_v_type_num, void *__pyx_v_data) {
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":355
 * 
 * cdef inline object PyArray_SimpleNewFromData(int nd, npy_intp *dims, int type_num, void *data):
 *     return PyArray_New(NULL, nd, dims, type_num, NULL, data, 0, NPY_CARRAY, NULL)             # <<<<<<<<<<<<<<
 * 
 * cdef inline bint PyArray_CHKFLAGS(ndarray n, int flags):
 */
  __pyx_t_1 = PyArray_New(NULL, __pyx_v_nd, __pyx_v_dims, __pyx_v_type_num, NULL, __pyx_v_data, 0, NPY_CARRAY, NULL); 
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":357
 *     return PyArray_New(NULL, nd, dims, type_num, NULL, data, 0, NPY_CARRAY, NULL)
 * 
 * cdef inline bint PyArray_CHKFLAGS(ndarray n, int flags):             # <<<<<<<<<<<<<<
 *     return  NpyArray_CHKFLAGS(<NpyArray*> <npy_intp>n.Array, flags)
 * 
 */

static CYTHON_INLINE int PyArray_CHKFLAGS(NumpyDotNet::ndarray^ __pyx_v_n, int __pyx_v_flags) {
  int __pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  __pyx_t_5numpy_6linalg_5numpy_npy_intp __pyx_t_2;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":358
 * 
 * cdef inline bint PyArray_CHKFLAGS(ndarray n, int flags):
 *     return  NpyArray_CHKFLAGS(<NpyArray*> <npy_intp>n.Array, flags)             # <<<<<<<<<<<<<<
 * 
 * cdef inline void* PyArray_DATA(ndarray n) nogil:
 */
  __pyx_t_1 = __site_get_Array_358_53->Target(__site_get_Array_358_53, ((System::Object^)__pyx_v_n), __pyx_context);
  __pyx_t_2 = __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_358_53->Target(__site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_358_53, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_r = NpyArray_CHKFLAGS(((NpyArray *)((__pyx_t_5numpy_6linalg_5numpy_npy_intp)__pyx_t_2)), __pyx_v_flags);
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":360
 *     return  NpyArray_CHKFLAGS(<NpyArray*> <npy_intp>n.Array, flags)
 * 
 * cdef inline void* PyArray_DATA(ndarray n) nogil:             # <<<<<<<<<<<<<<
 *     return NpyArray_DATA(<NpyArray*> <npy_intp>n.Array)
 * 
 */

static CYTHON_INLINE void *PyArray_DATA(NumpyDotNet::ndarray^ __pyx_v_n) {
  void *__pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  __pyx_t_5numpy_6linalg_5numpy_npy_intp __pyx_t_2;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":361
 * 
 * cdef inline void* PyArray_DATA(ndarray n) nogil:
 *     return NpyArray_DATA(<NpyArray*> <npy_intp>n.Array)             # <<<<<<<<<<<<<<
 * 
 * cdef inline intp_t* PyArray_DIMS(ndarray n) nogil:
 */
  __pyx_t_1 = __site_get_Array_361_48->Target(__site_get_Array_361_48, ((System::Object^)__pyx_v_n), __pyx_context);
  __pyx_t_2 = __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_361_48->Target(__site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_361_48, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_r = NpyArray_DATA(((NpyArray *)((__pyx_t_5numpy_6linalg_5numpy_npy_intp)__pyx_t_2)));
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":363
 *     return NpyArray_DATA(<NpyArray*> <npy_intp>n.Array)
 * 
 * cdef inline intp_t* PyArray_DIMS(ndarray n) nogil:             # <<<<<<<<<<<<<<
 *     return NpyArray_DIMS(<NpyArray*> <npy_intp>n.Array)
 * 
 */

static CYTHON_INLINE __pyx_t_5numpy_6linalg_5numpy_intp_t *PyArray_DIMS(NumpyDotNet::ndarray^ __pyx_v_n) {
  __pyx_t_5numpy_6linalg_5numpy_intp_t *__pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  __pyx_t_5numpy_6linalg_5numpy_npy_intp __pyx_t_2;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":364
 * 
 * cdef inline intp_t* PyArray_DIMS(ndarray n) nogil:
 *     return NpyArray_DIMS(<NpyArray*> <npy_intp>n.Array)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_DESCR(ndarray n):
 */
  __pyx_t_1 = __site_get_Array_364_48->Target(__site_get_Array_364_48, ((System::Object^)__pyx_v_n), __pyx_context);
  __pyx_t_2 = __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_364_48->Target(__site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_364_48, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_r = NpyArray_DIMS(((NpyArray *)((__pyx_t_5numpy_6linalg_5numpy_npy_intp)__pyx_t_2)));
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":366
 *     return NpyArray_DIMS(<NpyArray*> <npy_intp>n.Array)
 * 
 * cdef inline object PyArray_DESCR(ndarray n):             # <<<<<<<<<<<<<<
 *     return Npy_INTERFACE_descr(NpyArray_DESCR(<NpyArray*> <npy_intp>n.Array))
 * 
 */

static CYTHON_INLINE System::Object^ PyArray_DESCR(NumpyDotNet::ndarray^ __pyx_v_n) {
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  __pyx_t_5numpy_6linalg_5numpy_npy_intp __pyx_t_2;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":367
 * 
 * cdef inline object PyArray_DESCR(ndarray n):
 *     return Npy_INTERFACE_descr(NpyArray_DESCR(<NpyArray*> <npy_intp>n.Array))             # <<<<<<<<<<<<<<
 * 
 * cdef inline int PyArray_ITEMSIZE(ndarray n):
 */
  __pyx_t_1 = __site_get_Array_367_69->Target(__site_get_Array_367_69, ((System::Object^)__pyx_v_n), __pyx_context);
  __pyx_t_2 = __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_367_69->Target(__site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_367_69, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = ((System::Object^)Npy_INTERFACE_OBJECT(NpyArray_DESCR(((NpyArray *)((__pyx_t_5numpy_6linalg_5numpy_npy_intp)__pyx_t_2))))); 
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":369
 *     return Npy_INTERFACE_descr(NpyArray_DESCR(<NpyArray*> <npy_intp>n.Array))
 * 
 * cdef inline int PyArray_ITEMSIZE(ndarray n):             # <<<<<<<<<<<<<<
 *     return NpyArray_ITEMSIZE(<NpyArray*> <npy_intp>n.Array)
 * 
 */

static CYTHON_INLINE int PyArray_ITEMSIZE(NumpyDotNet::ndarray^ __pyx_v_n) {
  int __pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  __pyx_t_5numpy_6linalg_5numpy_npy_intp __pyx_t_2;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":370
 * 
 * cdef inline int PyArray_ITEMSIZE(ndarray n):
 *     return NpyArray_ITEMSIZE(<NpyArray*> <npy_intp>n.Array)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_Return(arr):
 */
  __pyx_t_1 = __site_get_Array_370_52->Target(__site_get_Array_370_52, ((System::Object^)__pyx_v_n), __pyx_context);
  __pyx_t_2 = __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_370_52->Target(__site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_370_52, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_r = NpyArray_ITEMSIZE(((NpyArray *)((__pyx_t_5numpy_6linalg_5numpy_npy_intp)__pyx_t_2)));
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":372
 *     return NpyArray_ITEMSIZE(<NpyArray*> <npy_intp>n.Array)
 * 
 * cdef inline object PyArray_Return(arr):             # <<<<<<<<<<<<<<
 *     if arr is None:
 *         return None
 */

static CYTHON_INLINE System::Object^ PyArray_Return(System::Object^ __pyx_v_arr) {
  System::Object^ __pyx_v_clr;
  System::Object^ __pyx_v_NumpyDotNet;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  __pyx_v_clr = nullptr;
  __pyx_v_NumpyDotNet = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":373
 * 
 * cdef inline object PyArray_Return(arr):
 *     if arr is None:             # <<<<<<<<<<<<<<
 *         return None
 *     import clr
 */
  __pyx_t_1 = (__pyx_v_arr == nullptr);
  if (__pyx_t_1) {

    /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":374
 * cdef inline object PyArray_Return(arr):
 *     if arr is None:
 *         return None             # <<<<<<<<<<<<<<
 *     import clr
 *     import NumpyDotNet.ndarray
 */
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":375
 *     if arr is None:
 *         return None
 *     import clr             # <<<<<<<<<<<<<<
 *     import NumpyDotNet.ndarray
 *     return NumpyDotNet.ndarray.ArrayReturn(arr)
 */
  __pyx_t_2 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "clr", -1));
  __pyx_v_clr = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":376
 *         return None
 *     import clr
 *     import NumpyDotNet.ndarray             # <<<<<<<<<<<<<<
 *     return NumpyDotNet.ndarray.ArrayReturn(arr)
 * 
 */
  __pyx_t_2 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "NumpyDotNet.ndarray", -1));
  __pyx_v_NumpyDotNet = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":377
 *     import clr
 *     import NumpyDotNet.ndarray
 *     return NumpyDotNet.ndarray.ArrayReturn(arr)             # <<<<<<<<<<<<<<
 * 
 * cdef inline intp_t PyArray_DIM(ndarray n, int dim):
 */
  __pyx_t_2 = __site_get_ndarray_377_22->Target(__site_get_ndarray_377_22, __pyx_v_NumpyDotNet, __pyx_context);
  __pyx_t_3 = __site_get_ArrayReturn_377_30->Target(__site_get_ArrayReturn_377_30, __pyx_t_2, __pyx_context);
  __pyx_t_2 = nullptr;
  __pyx_t_2 = __site_call1_377_42->Target(__site_call1_377_42, __pyx_context, __pyx_t_3, __pyx_v_arr);
  __pyx_t_3 = nullptr;
  __pyx_r = __pyx_t_2;
  __pyx_t_2 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":379
 *     return NumpyDotNet.ndarray.ArrayReturn(arr)
 * 
 * cdef inline intp_t PyArray_DIM(ndarray n, int dim):             # <<<<<<<<<<<<<<
 *     return NpyArray_DIM(<NpyArray*><long long>n.Array, dim)
 * 
 */

static CYTHON_INLINE __pyx_t_5numpy_6linalg_5numpy_intp_t PyArray_DIM(NumpyDotNet::ndarray^ __pyx_v_n, int __pyx_v_dim) {
  __pyx_t_5numpy_6linalg_5numpy_intp_t __pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  PY_LONG_LONG __pyx_t_2;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":380
 * 
 * cdef inline intp_t PyArray_DIM(ndarray n, int dim):
 *     return NpyArray_DIM(<NpyArray*><long long>n.Array, dim)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_NDIM(ndarray obj):
 */
  __pyx_t_1 = __site_get_Array_380_47->Target(__site_get_Array_380_47, ((System::Object^)__pyx_v_n), __pyx_context);
  __pyx_t_2 = __site_cvt_cvt_PY_LONG_LONG_380_47->Target(__site_cvt_cvt_PY_LONG_LONG_380_47, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_r = NpyArray_DIM(((NpyArray *)((PY_LONG_LONG)__pyx_t_2)), __pyx_v_dim);
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":382
 *     return NpyArray_DIM(<NpyArray*><long long>n.Array, dim)
 * 
 * cdef inline object PyArray_NDIM(ndarray obj):             # <<<<<<<<<<<<<<
 *     return obj.ndim
 * 
 */

static CYTHON_INLINE System::Object^ PyArray_NDIM(NumpyDotNet::ndarray^ __pyx_v_obj) {
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":383
 * 
 * cdef inline object PyArray_NDIM(ndarray obj):
 *     return obj.ndim             # <<<<<<<<<<<<<<
 * 
 * cdef inline intp_t PyArray_SIZE(ndarray n):
 */
  __pyx_t_1 = __site_get_ndim_383_14->Target(__site_get_ndim_383_14, ((System::Object^)__pyx_v_obj), __pyx_context);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":385
 *     return obj.ndim
 * 
 * cdef inline intp_t PyArray_SIZE(ndarray n):             # <<<<<<<<<<<<<<
 *     return NpyArray_SIZE(<NpyArray*> <npy_intp>n.Array)
 * 
 */

static CYTHON_INLINE __pyx_t_5numpy_6linalg_5numpy_intp_t PyArray_SIZE(NumpyDotNet::ndarray^ __pyx_v_n) {
  __pyx_t_5numpy_6linalg_5numpy_intp_t __pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  __pyx_t_5numpy_6linalg_5numpy_npy_intp __pyx_t_2;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":386
 * 
 * cdef inline intp_t PyArray_SIZE(ndarray n):
 *     return NpyArray_SIZE(<NpyArray*> <npy_intp>n.Array)             # <<<<<<<<<<<<<<
 * 
 * cdef inline npy_intp* PyArray_STRIDES(ndarray n):
 */
  __pyx_t_1 = __site_get_Array_386_48->Target(__site_get_Array_386_48, ((System::Object^)__pyx_v_n), __pyx_context);
  __pyx_t_2 = __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_386_48->Target(__site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_386_48, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_r = NpyArray_SIZE(((NpyArray *)((__pyx_t_5numpy_6linalg_5numpy_npy_intp)__pyx_t_2)));
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":388
 *     return NpyArray_SIZE(<NpyArray*> <npy_intp>n.Array)
 * 
 * cdef inline npy_intp* PyArray_STRIDES(ndarray n):             # <<<<<<<<<<<<<<
 *     return NpyArray_STRIDES(<NpyArray*> <npy_intp>n.Array)
 * 
 */

static CYTHON_INLINE __pyx_t_5numpy_6linalg_5numpy_npy_intp *PyArray_STRIDES(NumpyDotNet::ndarray^ __pyx_v_n) {
  __pyx_t_5numpy_6linalg_5numpy_npy_intp *__pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  __pyx_t_5numpy_6linalg_5numpy_npy_intp __pyx_t_2;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":389
 * 
 * cdef inline npy_intp* PyArray_STRIDES(ndarray n):
 *     return NpyArray_STRIDES(<NpyArray*> <npy_intp>n.Array)             # <<<<<<<<<<<<<<
 * 
 * cdef inline npy_intp PyArray_NBYTES(ndarray n):
 */
  __pyx_t_1 = __site_get_Array_389_51->Target(__site_get_Array_389_51, ((System::Object^)__pyx_v_n), __pyx_context);
  __pyx_t_2 = __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_389_51->Target(__site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_389_51, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_r = NpyArray_STRIDES(((NpyArray *)((__pyx_t_5numpy_6linalg_5numpy_npy_intp)__pyx_t_2)));
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":391
 *     return NpyArray_STRIDES(<NpyArray*> <npy_intp>n.Array)
 * 
 * cdef inline npy_intp PyArray_NBYTES(ndarray n):             # <<<<<<<<<<<<<<
 *     return NpyArray_NBYTES(<NpyArray *><long long>n.Array)
 * 
 */

static CYTHON_INLINE __pyx_t_5numpy_6linalg_5numpy_npy_intp PyArray_NBYTES(NumpyDotNet::ndarray^ __pyx_v_n) {
  __pyx_t_5numpy_6linalg_5numpy_npy_intp __pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  PY_LONG_LONG __pyx_t_2;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":392
 * 
 * cdef inline npy_intp PyArray_NBYTES(ndarray n):
 *     return NpyArray_NBYTES(<NpyArray *><long long>n.Array)             # <<<<<<<<<<<<<<
 * 
 * cdef inline NpyArray *PyArray_ARRAY(ndarray n):
 */
  __pyx_t_1 = __site_get_Array_392_51->Target(__site_get_Array_392_51, ((System::Object^)__pyx_v_n), __pyx_context);
  __pyx_t_2 = __site_cvt_cvt_PY_LONG_LONG_392_51->Target(__site_cvt_cvt_PY_LONG_LONG_392_51, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_r = NpyArray_NBYTES(((NpyArray *)((PY_LONG_LONG)__pyx_t_2)));
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":394
 *     return NpyArray_NBYTES(<NpyArray *><long long>n.Array)
 * 
 * cdef inline NpyArray *PyArray_ARRAY(ndarray n):             # <<<<<<<<<<<<<<
 *     return <NpyArray*> <npy_intp>n.Array
 * 
 */

static CYTHON_INLINE NpyArray *PyArray_ARRAY(NumpyDotNet::ndarray^ __pyx_v_n) {
  NpyArray *__pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  __pyx_t_5numpy_6linalg_5numpy_npy_intp __pyx_t_2;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":395
 * 
 * cdef inline NpyArray *PyArray_ARRAY(ndarray n):
 *     return <NpyArray*> <npy_intp>n.Array             # <<<<<<<<<<<<<<
 * 
 * cdef inline int PyArray_TYPE(ndarray n):
 */
  __pyx_t_1 = __site_get_Array_395_34->Target(__site_get_Array_395_34, ((System::Object^)__pyx_v_n), __pyx_context);
  __pyx_t_2 = __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_395_34->Target(__site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_395_34, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_r = ((NpyArray *)((__pyx_t_5numpy_6linalg_5numpy_npy_intp)__pyx_t_2));
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":397
 *     return <NpyArray*> <npy_intp>n.Array
 * 
 * cdef inline int PyArray_TYPE(ndarray n):             # <<<<<<<<<<<<<<
 *     return NpyArray_TYPE(<NpyArray*> <npy_intp>n.Array)
 * 
 */

static CYTHON_INLINE int PyArray_TYPE(NumpyDotNet::ndarray^ __pyx_v_n) {
  int __pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  __pyx_t_5numpy_6linalg_5numpy_npy_intp __pyx_t_2;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":398
 * 
 * cdef inline int PyArray_TYPE(ndarray n):
 *     return NpyArray_TYPE(<NpyArray*> <npy_intp>n.Array)             # <<<<<<<<<<<<<<
 * 
 * cdef inline void *PyArray_Zero(arr):
 */
  __pyx_t_1 = __site_get_Array_398_48->Target(__site_get_Array_398_48, ((System::Object^)__pyx_v_n), __pyx_context);
  __pyx_t_2 = __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_398_48->Target(__site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_398_48, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_r = NpyArray_TYPE(((NpyArray *)((__pyx_t_5numpy_6linalg_5numpy_npy_intp)__pyx_t_2)));
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":400
 *     return NpyArray_TYPE(<NpyArray*> <npy_intp>n.Array)
 * 
 * cdef inline void *PyArray_Zero(arr):             # <<<<<<<<<<<<<<
 *     import clr
 *     import NumpyDotNet.NpyArray
 */

static CYTHON_INLINE void *PyArray_Zero(System::Object^ __pyx_v_arr) {
  System::Object^ __pyx_v_clr;
  System::Object^ __pyx_v_NumpyDotNet;
  void *__pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_t_5numpy_6linalg_5numpy_npy_intp __pyx_t_3;
  __pyx_v_clr = nullptr;
  __pyx_v_NumpyDotNet = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":401
 * 
 * cdef inline void *PyArray_Zero(arr):
 *     import clr             # <<<<<<<<<<<<<<
 *     import NumpyDotNet.NpyArray
 *     return <void *><npy_intp>NumpyDotNet.NpyArray.Zero(arr)
 */
  __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "clr", -1));
  __pyx_v_clr = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":402
 * cdef inline void *PyArray_Zero(arr):
 *     import clr
 *     import NumpyDotNet.NpyArray             # <<<<<<<<<<<<<<
 *     return <void *><npy_intp>NumpyDotNet.NpyArray.Zero(arr)
 * 
 */
  __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "NumpyDotNet.NpyArray", -1));
  __pyx_v_NumpyDotNet = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":403
 *     import clr
 *     import NumpyDotNet.NpyArray
 *     return <void *><npy_intp>NumpyDotNet.NpyArray.Zero(arr)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object NpyArray_Return(NpyArray *arr):
 */
  __pyx_t_1 = __site_get_NpyArray_403_40->Target(__site_get_NpyArray_403_40, __pyx_v_NumpyDotNet, __pyx_context);
  __pyx_t_2 = __site_get_Zero_403_49->Target(__site_get_Zero_403_49, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call1_403_54->Target(__site_call1_403_54, __pyx_context, __pyx_t_2, __pyx_v_arr);
  __pyx_t_2 = nullptr;
  __pyx_t_3 = __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_403_54->Target(__site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_403_54, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_r = ((void *)((__pyx_t_5numpy_6linalg_5numpy_npy_intp)__pyx_t_3));
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":405
 *     return <void *><npy_intp>NumpyDotNet.NpyArray.Zero(arr)
 * 
 * cdef inline object NpyArray_Return(NpyArray *arr):             # <<<<<<<<<<<<<<
 *     ret = Npy_INTERFACE_array(arr)
 *     Npy_DECREF(arr)
 */

static CYTHON_INLINE System::Object^ NpyArray_Return(NpyArray *__pyx_v_arr) {
  NumpyDotNet::ndarray^ __pyx_v_ret;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  __pyx_v_ret = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":406
 * 
 * cdef inline object NpyArray_Return(NpyArray *arr):
 *     ret = Npy_INTERFACE_array(arr)             # <<<<<<<<<<<<<<
 *     Npy_DECREF(arr)
 *     return ret
 */
  __pyx_t_1 = ((System::Object^)Npy_INTERFACE_OBJECT(__pyx_v_arr)); 
  __pyx_v_ret = ((NumpyDotNet::ndarray^)__pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":407
 * cdef inline object NpyArray_Return(NpyArray *arr):
 *     ret = Npy_INTERFACE_array(arr)
 *     Npy_DECREF(arr)             # <<<<<<<<<<<<<<
 *     return ret
 * 
 */
  Npy_DECREF(__pyx_v_arr);

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":408
 *     ret = Npy_INTERFACE_array(arr)
 *     Npy_DECREF(arr)
 *     return ret             # <<<<<<<<<<<<<<
 * 
 * cdef inline int PyDataType_TYPE_NUM(dtype t):
 */
  __pyx_r = ((System::Object^)__pyx_v_ret);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":410
 *     return ret
 * 
 * cdef inline int PyDataType_TYPE_NUM(dtype t):             # <<<<<<<<<<<<<<
 *     return NpyDataType_TYPE_NUM(<NpyArray_Descr *><long long>t.Dtype)
 * 
 */

static CYTHON_INLINE int PyDataType_TYPE_NUM(NumpyDotNet::dtype^ __pyx_v_t) {
  int __pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  PY_LONG_LONG __pyx_t_2;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":411
 * 
 * cdef inline int PyDataType_TYPE_NUM(dtype t):
 *     return NpyDataType_TYPE_NUM(<NpyArray_Descr *><long long>t.Dtype)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_FromAny(op, newtype, min_depth, max_depth, flags, context):
 */
  __pyx_t_1 = __site_get_Dtype_411_62->Target(__site_get_Dtype_411_62, ((System::Object^)__pyx_v_t), __pyx_context);
  __pyx_t_2 = __site_cvt_cvt_PY_LONG_LONG_411_62->Target(__site_cvt_cvt_PY_LONG_LONG_411_62, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_r = NpyDataType_TYPE_NUM(((NpyArray_Descr *)((PY_LONG_LONG)__pyx_t_2)));
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":413
 *     return NpyDataType_TYPE_NUM(<NpyArray_Descr *><long long>t.Dtype)
 * 
 * cdef inline object PyArray_FromAny(op, newtype, min_depth, max_depth, flags, context):             # <<<<<<<<<<<<<<
 *     import clr
 *     import NumpyDotNet.NpyArray
 */

static CYTHON_INLINE System::Object^ PyArray_FromAny(System::Object^ __pyx_v_op, System::Object^ __pyx_v_newtype, System::Object^ __pyx_v_min_depth, System::Object^ __pyx_v_max_depth, System::Object^ __pyx_v_flags, System::Object^ __pyx_v_context) {
  System::Object^ __pyx_v_clr;
  System::Object^ __pyx_v_NumpyDotNet;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_clr = nullptr;
  __pyx_v_NumpyDotNet = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":414
 * 
 * cdef inline object PyArray_FromAny(op, newtype, min_depth, max_depth, flags, context):
 *     import clr             # <<<<<<<<<<<<<<
 *     import NumpyDotNet.NpyArray
 *     return NumpyDotNet.NpyArray.FromAny(op, newtype, min_depth, max_depth, flags, context)
 */
  __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "clr", -1));
  __pyx_v_clr = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":415
 * cdef inline object PyArray_FromAny(op, newtype, min_depth, max_depth, flags, context):
 *     import clr
 *     import NumpyDotNet.NpyArray             # <<<<<<<<<<<<<<
 *     return NumpyDotNet.NpyArray.FromAny(op, newtype, min_depth, max_depth, flags, context)
 * 
 */
  __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "NumpyDotNet.NpyArray", -1));
  __pyx_v_NumpyDotNet = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":416
 *     import clr
 *     import NumpyDotNet.NpyArray
 *     return NumpyDotNet.NpyArray.FromAny(op, newtype, min_depth, max_depth, flags, context)             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_t_1 = __site_get_NpyArray_416_22->Target(__site_get_NpyArray_416_22, __pyx_v_NumpyDotNet, __pyx_context);
  __pyx_t_2 = __site_get_FromAny_416_31->Target(__site_get_FromAny_416_31, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call6_416_39->Target(__site_call6_416_39, __pyx_context, __pyx_t_2, __pyx_v_op, __pyx_v_newtype, __pyx_v_min_depth, __pyx_v_max_depth, __pyx_v_flags, __pyx_v_context);
  __pyx_t_2 = nullptr;
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":419
 * 
 * 
 * cdef inline object PyArray_CopyFromObject(op, descr, min_depth, max_depth):             # <<<<<<<<<<<<<<
 *     return PyArray_FromAny(op, descr, min_depth, max_depth,
 *                            NPY_ENSURECOPY | NPY_DEFAULT | NPY_ENSUREARRAY, NULL)
 */

static CYTHON_INLINE System::Object^ PyArray_CopyFromObject(System::Object^ __pyx_v_op, System::Object^ __pyx_v_descr, System::Object^ __pyx_v_min_depth, System::Object^ __pyx_v_max_depth) {
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":421
 * cdef inline object PyArray_CopyFromObject(op, descr, min_depth, max_depth):
 *     return PyArray_FromAny(op, descr, min_depth, max_depth,
 *                            NPY_ENSURECOPY | NPY_DEFAULT | NPY_ENSUREARRAY, NULL)             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_t_1 = (System::Object^)(long long)(((NPY_ENSURECOPY | NPY_DEFAULT) | NPY_ENSUREARRAY));
  __pyx_t_2 = NULL;
  __pyx_t_3 = PyArray_FromAny(__pyx_v_op, __pyx_v_descr, __pyx_v_min_depth, __pyx_v_max_depth, __pyx_t_1, __pyx_t_2); 
  __pyx_t_1 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_r = __pyx_t_3;
  __pyx_t_3 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":424
 * 
 * 
 * cdef inline object PyArray_FROMANY(m, type, min, max, flags):             # <<<<<<<<<<<<<<
 *     if flags & NPY_ENSURECOPY:
 *         flags |= NPY_DEFAULT
 */

static CYTHON_INLINE System::Object^ PyArray_FROMANY(System::Object^ __pyx_v_m, System::Object^ __pyx_v_type, System::Object^ __pyx_v_min, System::Object^ __pyx_v_max, System::Object^ __pyx_v_flags) {
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  int __pyx_t_3;
  int __pyx_t_4;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":425
 * 
 * cdef inline object PyArray_FROMANY(m, type, min, max, flags):
 *     if flags & NPY_ENSURECOPY:             # <<<<<<<<<<<<<<
 *         flags |= NPY_DEFAULT
 *     return PyArray_FromAny(m, Npy_INTERFACE_descr(NpyArray_DescrFromType(type)), min, max, flags, None)
 */
  __pyx_t_1 = (System::Object^)(long long)(NPY_ENSURECOPY);
  __pyx_t_2 = __site_op_and_425_13->Target(__site_op_and_425_13, __pyx_v_flags, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_t_3 = __site_istrue_425_13->Target(__site_istrue_425_13, __pyx_t_2);
  __pyx_t_2 = nullptr;
  if (__pyx_t_3) {

    /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":426
 * cdef inline object PyArray_FROMANY(m, type, min, max, flags):
 *     if flags & NPY_ENSURECOPY:
 *         flags |= NPY_DEFAULT             # <<<<<<<<<<<<<<
 *     return PyArray_FromAny(m, Npy_INTERFACE_descr(NpyArray_DescrFromType(type)), min, max, flags, None)
 * 
 */
    __pyx_t_2 = (System::Object^)(long long)(NPY_DEFAULT);
    __pyx_t_1 = __site_op_ior_426_14->Target(__site_op_ior_426_14, __pyx_v_flags, __pyx_t_2);
    __pyx_t_2 = nullptr;
    __pyx_v_flags = __pyx_t_1;
    __pyx_t_1 = nullptr;
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":427
 *     if flags & NPY_ENSURECOPY:
 *         flags |= NPY_DEFAULT
 *     return PyArray_FromAny(m, Npy_INTERFACE_descr(NpyArray_DescrFromType(type)), min, max, flags, None)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_ContiguousFromObject(op, type, minDepth, maxDepth):
 */
  __pyx_t_4 = __site_cvt_cvt_int_427_77->Target(__site_cvt_cvt_int_427_77, __pyx_v_type);
  __pyx_t_1 = ((System::Object^)Npy_INTERFACE_OBJECT(NpyArray_DescrFromType(__pyx_t_4))); 
  __pyx_t_2 = PyArray_FromAny(__pyx_v_m, __pyx_t_1, __pyx_v_min, __pyx_v_max, __pyx_v_flags, nullptr); 
  __pyx_t_1 = nullptr;
  __pyx_r = __pyx_t_2;
  __pyx_t_2 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":429
 *     return PyArray_FromAny(m, Npy_INTERFACE_descr(NpyArray_DescrFromType(type)), min, max, flags, None)
 * 
 * cdef inline object PyArray_ContiguousFromObject(op, type, minDepth, maxDepth):             # <<<<<<<<<<<<<<
 *     return PyArray_FromAny(op, Npy_INTERFACE_descr(NpyArray_DescrFromType(type)), minDepth, maxDepth,
 *                            NPY_DEFAULT | NPY_ENSUREARRAY, NULL)
 */

static CYTHON_INLINE System::Object^ PyArray_ContiguousFromObject(System::Object^ __pyx_v_op, System::Object^ __pyx_v_type, System::Object^ __pyx_v_minDepth, System::Object^ __pyx_v_maxDepth) {
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  System::Object^ __pyx_t_4 = nullptr;
  System::Object^ __pyx_t_5 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":430
 * 
 * cdef inline object PyArray_ContiguousFromObject(op, type, minDepth, maxDepth):
 *     return PyArray_FromAny(op, Npy_INTERFACE_descr(NpyArray_DescrFromType(type)), minDepth, maxDepth,             # <<<<<<<<<<<<<<
 *                            NPY_DEFAULT | NPY_ENSUREARRAY, NULL)
 * 
 */
  __pyx_t_1 = __site_cvt_cvt_int_430_78->Target(__site_cvt_cvt_int_430_78, __pyx_v_type);
  __pyx_t_2 = ((System::Object^)Npy_INTERFACE_OBJECT(NpyArray_DescrFromType(__pyx_t_1))); 

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":431
 * cdef inline object PyArray_ContiguousFromObject(op, type, minDepth, maxDepth):
 *     return PyArray_FromAny(op, Npy_INTERFACE_descr(NpyArray_DescrFromType(type)), minDepth, maxDepth,
 *                            NPY_DEFAULT | NPY_ENSUREARRAY, NULL)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_CheckFromAny(op, newtype, min_depth, max_depth, flags, context):
 */
  __pyx_t_3 = (System::Object^)(long long)((NPY_DEFAULT | NPY_ENSUREARRAY));
  __pyx_t_4 = NULL;
  __pyx_t_5 = PyArray_FromAny(__pyx_v_op, __pyx_t_2, __pyx_v_minDepth, __pyx_v_maxDepth, __pyx_t_3, __pyx_t_4); 
  __pyx_t_2 = nullptr;
  __pyx_t_3 = nullptr;
  __pyx_t_4 = nullptr;
  __pyx_r = __pyx_t_5;
  __pyx_t_5 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":433
 *                            NPY_DEFAULT | NPY_ENSUREARRAY, NULL)
 * 
 * cdef inline object PyArray_CheckFromAny(op, newtype, min_depth, max_depth, flags, context):             # <<<<<<<<<<<<<<
 *     import clr
 *     import NumpyDotNet.NpyArray
 */

static CYTHON_INLINE System::Object^ PyArray_CheckFromAny(System::Object^ __pyx_v_op, System::Object^ __pyx_v_newtype, System::Object^ __pyx_v_min_depth, System::Object^ __pyx_v_max_depth, System::Object^ __pyx_v_flags, System::Object^ __pyx_v_context) {
  System::Object^ __pyx_v_clr;
  System::Object^ __pyx_v_NumpyDotNet;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_clr = nullptr;
  __pyx_v_NumpyDotNet = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":434
 * 
 * cdef inline object PyArray_CheckFromAny(op, newtype, min_depth, max_depth, flags, context):
 *     import clr             # <<<<<<<<<<<<<<
 *     import NumpyDotNet.NpyArray
 *     return NumpyDotNet.NpyArray.CheckFromAny(op, newtype, min_depth, max_depth, flags, context)
 */
  __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "clr", -1));
  __pyx_v_clr = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":435
 * cdef inline object PyArray_CheckFromAny(op, newtype, min_depth, max_depth, flags, context):
 *     import clr
 *     import NumpyDotNet.NpyArray             # <<<<<<<<<<<<<<
 *     return NumpyDotNet.NpyArray.CheckFromAny(op, newtype, min_depth, max_depth, flags, context)
 * 
 */
  __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "NumpyDotNet.NpyArray", -1));
  __pyx_v_NumpyDotNet = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":436
 *     import clr
 *     import NumpyDotNet.NpyArray
 *     return NumpyDotNet.NpyArray.CheckFromAny(op, newtype, min_depth, max_depth, flags, context)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_Check(obj):
 */
  __pyx_t_1 = __site_get_NpyArray_436_22->Target(__site_get_NpyArray_436_22, __pyx_v_NumpyDotNet, __pyx_context);
  __pyx_t_2 = __site_get_CheckFromAny_436_31->Target(__site_get_CheckFromAny_436_31, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call6_436_44->Target(__site_call6_436_44, __pyx_context, __pyx_t_2, __pyx_v_op, __pyx_v_newtype, __pyx_v_min_depth, __pyx_v_max_depth, __pyx_v_flags, __pyx_v_context);
  __pyx_t_2 = nullptr;
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":438
 *     return NumpyDotNet.NpyArray.CheckFromAny(op, newtype, min_depth, max_depth, flags, context)
 * 
 * cdef inline object PyArray_Check(obj):             # <<<<<<<<<<<<<<
 *     import numpy as np
 *     return isinstance(obj, np.ndarray)
 */

static CYTHON_INLINE System::Object^ PyArray_Check(System::Object^ __pyx_v_obj) {
  System::Object^ __pyx_v_np;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  __pyx_v_np = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":439
 * 
 * cdef inline object PyArray_Check(obj):
 *     import numpy as np             # <<<<<<<<<<<<<<
 *     return isinstance(obj, np.ndarray)
 * 
 */
  __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "numpy", -1));
  __pyx_v_np = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":440
 * cdef inline object PyArray_Check(obj):
 *     import numpy as np
 *     return isinstance(obj, np.ndarray)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_Cast(arr, typenum):
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "isinstance");
  __pyx_t_2 = __site_get_ndarray_440_29->Target(__site_get_ndarray_440_29, __pyx_v_np, __pyx_context);
  __pyx_t_3 = __site_call2_440_21->Target(__site_call2_440_21, __pyx_context, __pyx_t_1, __pyx_v_obj, __pyx_t_2);
  __pyx_t_1 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_r = __pyx_t_3;
  __pyx_t_3 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":442
 *     return isinstance(obj, np.ndarray)
 * 
 * cdef inline object PyArray_Cast(arr, typenum):             # <<<<<<<<<<<<<<
 *     import clr
 *     import NumpyDotNet.NpyCoreApi
 */

static CYTHON_INLINE System::Object^ PyArray_Cast(System::Object^ __pyx_v_arr, System::Object^ __pyx_v_typenum) {
  System::Object^ __pyx_v_clr;
  System::Object^ __pyx_v_NumpyDotNet;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  int __pyx_t_3;
  System::Object^ __pyx_t_4 = nullptr;
  System::Object^ __pyx_t_5 = nullptr;
  __pyx_v_clr = nullptr;
  __pyx_v_NumpyDotNet = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":443
 * 
 * cdef inline object PyArray_Cast(arr, typenum):
 *     import clr             # <<<<<<<<<<<<<<
 *     import NumpyDotNet.NpyCoreApi
 *     return NumpyDotNet.NpyCoreApi.CastToType(arr, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), False)
 */
  __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "clr", -1));
  __pyx_v_clr = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":444
 * cdef inline object PyArray_Cast(arr, typenum):
 *     import clr
 *     import NumpyDotNet.NpyCoreApi             # <<<<<<<<<<<<<<
 *     return NumpyDotNet.NpyCoreApi.CastToType(arr, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), False)
 * 
 */
  __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "NumpyDotNet.NpyCoreApi", -1));
  __pyx_v_NumpyDotNet = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":445
 *     import clr
 *     import NumpyDotNet.NpyCoreApi
 *     return NumpyDotNet.NpyCoreApi.CastToType(arr, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), False)             # <<<<<<<<<<<<<<
 * 
 * cdef inline void import_array():
 */
  __pyx_t_1 = __site_get_NpyCoreApi_445_22->Target(__site_get_NpyCoreApi_445_22, __pyx_v_NumpyDotNet, __pyx_context);
  __pyx_t_2 = __site_get_CastToType_445_33->Target(__site_get_CastToType_445_33, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_3 = __site_cvt_cvt_int_445_100->Target(__site_cvt_cvt_int_445_100, __pyx_v_typenum);
  __pyx_t_1 = ((System::Object^)Npy_INTERFACE_OBJECT(NpyArray_DescrFromType(__pyx_t_3))); 
  __pyx_t_4 = 0;
  __pyx_t_5 = __site_call3_445_44->Target(__site_call3_445_44, __pyx_context, __pyx_t_2, __pyx_v_arr, __pyx_t_1, __pyx_t_4);
  __pyx_t_2 = nullptr;
  __pyx_t_1 = nullptr;
  __pyx_t_4 = nullptr;
  __pyx_r = __pyx_t_5;
  __pyx_t_5 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":447
 *     return NumpyDotNet.NpyCoreApi.CastToType(arr, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), False)
 * 
 * cdef inline void import_array():             # <<<<<<<<<<<<<<
 *     pass
 * 
 */

static CYTHON_INLINE void import_array(void) {

}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":450
 *     pass
 * 
 * cdef inline object PyArray_DescrConverter(obj):             # <<<<<<<<<<<<<<
 *     import clr
 *     import NumpyDotNet.NpyDescr
 */

static CYTHON_INLINE System::Object^ PyArray_DescrConverter(System::Object^ __pyx_v_obj) {
  System::Object^ __pyx_v_clr;
  System::Object^ __pyx_v_NumpyDotNet;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_clr = nullptr;
  __pyx_v_NumpyDotNet = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":451
 * 
 * cdef inline object PyArray_DescrConverter(obj):
 *     import clr             # <<<<<<<<<<<<<<
 *     import NumpyDotNet.NpyDescr
 *     return NumpyDotNet.NpyDescr.DescrConverter(obj)
 */
  __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "clr", -1));
  __pyx_v_clr = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":452
 * cdef inline object PyArray_DescrConverter(obj):
 *     import clr
 *     import NumpyDotNet.NpyDescr             # <<<<<<<<<<<<<<
 *     return NumpyDotNet.NpyDescr.DescrConverter(obj)
 * 
 */
  __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "NumpyDotNet.NpyDescr", -1));
  __pyx_v_NumpyDotNet = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":453
 *     import clr
 *     import NumpyDotNet.NpyDescr
 *     return NumpyDotNet.NpyDescr.DescrConverter(obj)             # <<<<<<<<<<<<<<
 * 
 * cdef inline PyNumber_Check(o):
 */
  __pyx_t_1 = __site_get_NpyDescr_453_22->Target(__site_get_NpyDescr_453_22, __pyx_v_NumpyDotNet, __pyx_context);
  __pyx_t_2 = __site_get_DescrConverter_453_31->Target(__site_get_DescrConverter_453_31, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call1_453_46->Target(__site_call1_453_46, __pyx_context, __pyx_t_2, __pyx_v_obj);
  __pyx_t_2 = nullptr;
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":455
 *     return NumpyDotNet.NpyDescr.DescrConverter(obj)
 * 
 * cdef inline PyNumber_Check(o):             # <<<<<<<<<<<<<<
 *     import clr
 *     import NumpyDotNet.ScalarGeneric
 */

static CYTHON_INLINE System::Object^ PyNumber_Check(System::Object^ __pyx_v_o) {
  System::Object^ __pyx_v_clr;
  System::Object^ __pyx_v_NumpyDotNet;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  System::Object^ __pyx_t_4 = nullptr;
  System::Object^ __pyx_t_5 = nullptr;
  int __pyx_t_6;
  __pyx_v_clr = nullptr;
  __pyx_v_NumpyDotNet = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":456
 * 
 * cdef inline PyNumber_Check(o):
 *     import clr             # <<<<<<<<<<<<<<
 *     import NumpyDotNet.ScalarGeneric
 *     return isinstance(o, (int, long, float)) or isinstance(o, NumpyDotNet.ScalarGeneric)
 */
  __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "clr", -1));
  __pyx_v_clr = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":457
 * cdef inline PyNumber_Check(o):
 *     import clr
 *     import NumpyDotNet.ScalarGeneric             # <<<<<<<<<<<<<<
 *     return isinstance(o, (int, long, float)) or isinstance(o, NumpyDotNet.ScalarGeneric)
 * 
 */
  __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "NumpyDotNet.ScalarGeneric", -1));
  __pyx_v_NumpyDotNet = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":458
 *     import clr
 *     import NumpyDotNet.ScalarGeneric
 *     return isinstance(o, (int, long, float)) or isinstance(o, NumpyDotNet.ScalarGeneric)             # <<<<<<<<<<<<<<
 * 
 * cdef inline NpyArrayIterObject *PyArray_IterNew(ndarray n):
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "isinstance");
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "int");
  __pyx_t_3 = PythonOps::GetGlobal(__pyx_context, "long");
  __pyx_t_4 = PythonOps::GetGlobal(__pyx_context, "float");
  __pyx_t_5 = PythonOps::MakeTuple(gcnew array<System::Object^>{((System::Object^)__pyx_t_2), ((System::Object^)__pyx_t_3), ((System::Object^)__pyx_t_4)});
  __pyx_t_2 = nullptr;
  __pyx_t_3 = nullptr;
  __pyx_t_4 = nullptr;
  __pyx_t_4 = __site_call2_458_21->Target(__site_call2_458_21, __pyx_context, __pyx_t_1, __pyx_v_o, __pyx_t_5);
  __pyx_t_1 = nullptr;
  __pyx_t_5 = nullptr;
  __pyx_t_6 = __site_cvt_bool_458_45->Target(__site_cvt_bool_458_45, __pyx_t_4);
  if (!__pyx_t_6) {
    __pyx_t_4 = nullptr;
    __pyx_t_5 = PythonOps::GetGlobal(__pyx_context, "isinstance");
    __pyx_t_1 = __site_get_ScalarGeneric_458_73->Target(__site_get_ScalarGeneric_458_73, __pyx_v_NumpyDotNet, __pyx_context);
    __pyx_t_3 = __site_call2_458_58->Target(__site_call2_458_58, __pyx_context, __pyx_t_5, __pyx_v_o, __pyx_t_1);
    __pyx_t_5 = nullptr;
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __pyx_t_3;
    __pyx_t_3 = nullptr;
  } else {
    __pyx_t_1 = __pyx_t_4;
    __pyx_t_4 = nullptr;
  }
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":460
 *     return isinstance(o, (int, long, float)) or isinstance(o, NumpyDotNet.ScalarGeneric)
 * 
 * cdef inline NpyArrayIterObject *PyArray_IterNew(ndarray n):             # <<<<<<<<<<<<<<
 *     return NpyArray_IterNew(<NpyArray*> <npy_intp>n.Array)
 * 
 */

static CYTHON_INLINE NpyArrayIterObject *PyArray_IterNew(NumpyDotNet::ndarray^ __pyx_v_n) {
  NpyArrayIterObject *__pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  __pyx_t_5numpy_6linalg_5numpy_npy_intp __pyx_t_2;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":461
 * 
 * cdef inline NpyArrayIterObject *PyArray_IterNew(ndarray n):
 *     return NpyArray_IterNew(<NpyArray*> <npy_intp>n.Array)             # <<<<<<<<<<<<<<
 * 
 * cdef inline NpyArrayIterObject *PyArray_IterAllButAxis(ndarray n, int *inaxis):
 */
  __pyx_t_1 = __site_get_Array_461_51->Target(__site_get_Array_461_51, ((System::Object^)__pyx_v_n), __pyx_context);
  __pyx_t_2 = __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_461_51->Target(__site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_461_51, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_r = NpyArray_IterNew(((NpyArray *)((__pyx_t_5numpy_6linalg_5numpy_npy_intp)__pyx_t_2)));
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":463
 *     return NpyArray_IterNew(<NpyArray*> <npy_intp>n.Array)
 * 
 * cdef inline NpyArrayIterObject *PyArray_IterAllButAxis(ndarray n, int *inaxis):             # <<<<<<<<<<<<<<
 *     return NpyArray_IterAllButAxis(<NpyArray*> <npy_intp>n.Array, inaxis)
 * 
 */

static CYTHON_INLINE NpyArrayIterObject *PyArray_IterAllButAxis(NumpyDotNet::ndarray^ __pyx_v_n, int *__pyx_v_inaxis) {
  NpyArrayIterObject *__pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  __pyx_t_5numpy_6linalg_5numpy_npy_intp __pyx_t_2;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":464
 * 
 * cdef inline NpyArrayIterObject *PyArray_IterAllButAxis(ndarray n, int *inaxis):
 *     return NpyArray_IterAllButAxis(<NpyArray*> <npy_intp>n.Array, inaxis)             # <<<<<<<<<<<<<<
 * 
 * cdef inline void PyArray_ITER_NEXT(NpyArrayIterObject *obj):
 */
  __pyx_t_1 = __site_get_Array_464_58->Target(__site_get_Array_464_58, ((System::Object^)__pyx_v_n), __pyx_context);
  __pyx_t_2 = __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_464_58->Target(__site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_464_58, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_r = NpyArray_IterAllButAxis(((NpyArray *)((__pyx_t_5numpy_6linalg_5numpy_npy_intp)__pyx_t_2)), __pyx_v_inaxis);
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":466
 *     return NpyArray_IterAllButAxis(<NpyArray*> <npy_intp>n.Array, inaxis)
 * 
 * cdef inline void PyArray_ITER_NEXT(NpyArrayIterObject *obj):             # <<<<<<<<<<<<<<
 *     NpyArray_ITER_NEXT(obj)
 * 
 */

static CYTHON_INLINE void PyArray_ITER_NEXT(NpyArrayIterObject *__pyx_v_obj) {

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":467
 * 
 * cdef inline void PyArray_ITER_NEXT(NpyArrayIterObject *obj):
 *     NpyArray_ITER_NEXT(obj)             # <<<<<<<<<<<<<<
 * 
 * cdef inline void PyArray_ITER_RESET(NpyArrayIterObject *obj):
 */
  NpyArray_ITER_NEXT(__pyx_v_obj);

}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":469
 *     NpyArray_ITER_NEXT(obj)
 * 
 * cdef inline void PyArray_ITER_RESET(NpyArrayIterObject *obj):             # <<<<<<<<<<<<<<
 *     NpyArray_ITER_RESET(obj)
 * 
 */

static CYTHON_INLINE void PyArray_ITER_RESET(NpyArrayIterObject *__pyx_v_obj) {

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":470
 * 
 * cdef inline void PyArray_ITER_RESET(NpyArrayIterObject *obj):
 *     NpyArray_ITER_RESET(obj)             # <<<<<<<<<<<<<<
 * 
 * cdef inline void * PyArray_ITER_DATA(NpyArrayIterObject *obj):
 */
  NpyArray_ITER_RESET(__pyx_v_obj);

}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":472
 *     NpyArray_ITER_RESET(obj)
 * 
 * cdef inline void * PyArray_ITER_DATA(NpyArrayIterObject *obj):             # <<<<<<<<<<<<<<
 *     return NpyArray_ITER_DATA(obj)
 * 
 */

static CYTHON_INLINE void *PyArray_ITER_DATA(NpyArrayIterObject *__pyx_v_obj) {
  void *__pyx_r;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":473
 * 
 * cdef inline void * PyArray_ITER_DATA(NpyArrayIterObject *obj):
 *     return NpyArray_ITER_DATA(obj)             # <<<<<<<<<<<<<<
 * 
 * cdef inline NpyArrayNeighborhoodIterObject* PyArray_NeighborhoodIterNew(NpyArrayIterObject *obj,
 */
  __pyx_r = NpyArray_ITER_DATA(__pyx_v_obj);
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":475
 *     return NpyArray_ITER_DATA(obj)
 * 
 * cdef inline NpyArrayNeighborhoodIterObject* PyArray_NeighborhoodIterNew(NpyArrayIterObject *obj,             # <<<<<<<<<<<<<<
 *                                                                         npy_intp *bounds,
 *                                                                         int mode,
 */

static CYTHON_INLINE NpyArrayNeighborhoodIterObject *PyArray_NeighborhoodIterNew(NpyArrayIterObject *__pyx_v_obj, __pyx_t_5numpy_6linalg_5numpy_npy_intp *__pyx_v_bounds, int __pyx_v_mode, void *__pyx_v_fill, npy_free_func __pyx_v_fillfree) {
  NpyArrayNeighborhoodIterObject *__pyx_r;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":480
 *                                                                         void *fill,
 *                                                                         npy_free_func fillfree):
 *     return NpyArray_NeighborhoodIterNew(obj, bounds, mode, fill, fillfree)             # <<<<<<<<<<<<<<
 * 
 * cdef inline int PyArrayNeighborhoodIter_Reset(NpyArrayNeighborhoodIterObject* iter):
 */
  __pyx_r = NpyArray_NeighborhoodIterNew(__pyx_v_obj, __pyx_v_bounds, __pyx_v_mode, __pyx_v_fill, __pyx_v_fillfree);
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":482
 *     return NpyArray_NeighborhoodIterNew(obj, bounds, mode, fill, fillfree)
 * 
 * cdef inline int PyArrayNeighborhoodIter_Reset(NpyArrayNeighborhoodIterObject* iter):             # <<<<<<<<<<<<<<
 *     return NpyArrayNeighborhoodIter_Reset(iter)
 * 
 */

static CYTHON_INLINE int PyArrayNeighborhoodIter_Reset(NpyArrayNeighborhoodIterObject *__pyx_v_iter) {
  int __pyx_r;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":483
 * 
 * cdef inline int PyArrayNeighborhoodIter_Reset(NpyArrayNeighborhoodIterObject* iter):
 *     return NpyArrayNeighborhoodIter_Reset(iter)             # <<<<<<<<<<<<<<
 * 
 * cdef inline int PyArrayNeighborhoodIter_Next(NpyArrayNeighborhoodIterObject* iter):
 */
  __pyx_r = NpyArrayNeighborhoodIter_Reset(__pyx_v_iter);
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":485
 *     return NpyArrayNeighborhoodIter_Reset(iter)
 * 
 * cdef inline int PyArrayNeighborhoodIter_Next(NpyArrayNeighborhoodIterObject* iter):             # <<<<<<<<<<<<<<
 *     return NpyArrayNeighborhoodIter_Next(iter)
 * 
 */

static CYTHON_INLINE int PyArrayNeighborhoodIter_Next(NpyArrayNeighborhoodIterObject *__pyx_v_iter) {
  int __pyx_r;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":486
 * 
 * cdef inline int PyArrayNeighborhoodIter_Next(NpyArrayNeighborhoodIterObject* iter):
 *     return NpyArrayNeighborhoodIter_Next(iter)             # <<<<<<<<<<<<<<
 * 
 * cdef inline ndarray NpyIter_ARRAY(NpyArrayIterObject *iter):
 */
  __pyx_r = NpyArrayNeighborhoodIter_Next(__pyx_v_iter);
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":488
 *     return NpyArrayNeighborhoodIter_Next(iter)
 * 
 * cdef inline ndarray NpyIter_ARRAY(NpyArrayIterObject *iter):             # <<<<<<<<<<<<<<
 *     return Npy_INTERFACE_array(iter.ao)
 */

static CYTHON_INLINE NumpyDotNet::ndarray^ NpyIter_ARRAY(NpyArrayIterObject *__pyx_v_iter) {
  NumpyDotNet::ndarray^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":489
 * 
 * cdef inline ndarray NpyIter_ARRAY(NpyArrayIterObject *iter):
 *     return Npy_INTERFACE_array(iter.ao)             # <<<<<<<<<<<<<<
 */
  __pyx_t_1 = ((System::Object^)Npy_INTERFACE_OBJECT(__pyx_v_iter->ao)); 
  __pyx_r = ((NumpyDotNet::ndarray^)__pyx_t_1);
  __pyx_t_1 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}
// XXX skipping all typeobj definitions
/* Cython code section 'pystring_table' */
/* Cython code section 'cached_builtins' */
/* Cython code section 'init_globals' */

static int __Pyx_InitGlobals(void) {

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
  __site_op_mod_37_77 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Modulo));
  __site_call1_37_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_mod_39_77 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Modulo));
  __site_call1_39_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_byteorder_40_29 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "byteorder", false));
  __site_op_ne_40_40 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::NotEqual));
  __site_istrue_40_40 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_get_byteorder_40_71 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "byteorder", false));
  __site_op_ne_40_82 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::NotEqual));
  __site_istrue_40_82 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_op_mod_41_85 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Modulo));
  __site_call1_41_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_cvt_int_46_0 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_46_0_1 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_46_0_2 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_46_0_3 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_46_0_4 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_46_0_5 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_getindex_50_36 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_call1_50_30 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_cvt_char_50_30 = CallSite< System::Func< CallSite^, System::Object^, char >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, char::typeid, ConversionResultKind::ExplicitCast));
  __site_getindex_51_36 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_call1_51_30 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_cvt_char_51_30 = CallSite< System::Func< CallSite^, System::Object^, char >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, char::typeid, ConversionResultKind::ExplicitCast));
  __site_setindex_70_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_71_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_72_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_73_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_74_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_75_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_76_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_77_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_78_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_cvt_cvt_int_83_0 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_83_0_1 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_83_0_2 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_83_0_3 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_83_0_4 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_getindex_124_34 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_call1_124_29 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_cvt_char_124_29 = CallSite< System::Func< CallSite^, System::Object^, char >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, char::typeid, ConversionResultKind::ExplicitCast));
  __site_getindex_125_34 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_call1_125_29 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_cvt_char_125_29 = CallSite< System::Func< CallSite^, System::Object^, char >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, char::typeid, ConversionResultKind::ExplicitCast));
  __site_setindex_139_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_140_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_141_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_142_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_143_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_144_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_145_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_146_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_cvt_cvt_int_150_0 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_150_0_1 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_150_0_2 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_150_0_3 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_150_0_4 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_150_0_5 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_getindex_194_34 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_call1_194_29 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_cvt_char_194_29 = CallSite< System::Func< CallSite^, System::Object^, char >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, char::typeid, ConversionResultKind::ExplicitCast));
  __site_getindex_195_34 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_call1_195_29 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_cvt_char_195_29 = CallSite< System::Func< CallSite^, System::Object^, char >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, char::typeid, ConversionResultKind::ExplicitCast));
  __site_setindex_211_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_212_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_213_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_214_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_215_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_216_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_217_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_218_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_219_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_cvt_cvt_int_223_0 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_223_0_1 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_223_0_2 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_223_0_3 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_223_0_4 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_double_223_0 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_223_0_5 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_223_0_6 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_223_0_7 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_setindex_242_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_243_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_244_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_245_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_246_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_247_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_248_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_249_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_250_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_251_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_cvt_cvt_int_255_0 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_255_0_1 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_255_0_2 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_255_0_3 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_255_0_4 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_setindex_270_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_271_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_272_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_273_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_274_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_275_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_cvt_cvt_int_279_0 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_279_0_1 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_279_0_2 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_279_0_3 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_279_0_4 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_279_0_5 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_279_0_6 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_getindex_283_34 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_call1_283_29 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_cvt_char_283_29 = CallSite< System::Func< CallSite^, System::Object^, char >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, char::typeid, ConversionResultKind::ExplicitCast));
  __site_op_eq_311_16 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Equal));
  __site_istrue_311_16 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_op_eq_313_18 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Equal));
  __site_istrue_313_18 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_op_eq_315_18 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Equal));
  __site_istrue_315_18 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_op_eq_315_33 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Equal));
  __site_istrue_315_33 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_setindex_321_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_322_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_323_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_324_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_325_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_326_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_327_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_328_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_329_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_cvt_cvt_int_333_0 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_333_0_1 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_333_0_2 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_333_0_3 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_setindex_343_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_344_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_345_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_346_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_347_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_cvt_cvt_int_351_0 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_351_0_1 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_351_0_2 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_getindex_353_34 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_call1_353_29 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_cvt_char_353_29 = CallSite< System::Func< CallSite^, System::Object^, char >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, char::typeid, ConversionResultKind::ExplicitCast));
  __site_setindex_362_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_363_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_364_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_365_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_cvt_cvt_int_369_0 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_369_0_1 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_369_0_2 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_369_0_3 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_369_0_4 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_setindex_385_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_386_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_387_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_388_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_389_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_390_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_cvt_cvt_int_394_0 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_394_0_1 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_394_0_2 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_394_0_3 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_394_0_4 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_394_0_5 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_setindex_409_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_410_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_cvt_cvt_int_414_0 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_414_0_1 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_414_0_2 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_414_0_3 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_414_0_4 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_414_0_5 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_getindex_418_36 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_call1_418_30 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_cvt_char_418_30 = CallSite< System::Func< CallSite^, System::Object^, char >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, char::typeid, ConversionResultKind::ExplicitCast));
  __site_getindex_419_36 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_call1_419_30 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_cvt_char_419_30 = CallSite< System::Func< CallSite^, System::Object^, char >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, char::typeid, ConversionResultKind::ExplicitCast));
  __site_setindex_437_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_438_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_439_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_440_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_441_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_442_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_443_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_444_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_445_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_cvt_cvt_int_449_0 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_449_0_1 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_449_0_2 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_449_0_3 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_449_0_4 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_double_449_0 = CallSite< System::Func< CallSite^, System::Object^, double >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, double::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_449_0_5 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_449_0_6 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_449_0_7 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_setindex_471_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_472_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_473_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_474_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_475_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_476_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_477_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_478_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_479_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_cvt_cvt_int_483_0 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_483_0_1 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_483_0_2 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_483_0_3 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_483_0_4 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_setindex_498_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_499_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_500_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_501_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_502_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_503_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_cvt_cvt_int_507_0 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_507_0_1 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_507_0_2 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_507_0_3 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_507_0_4 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_507_0_5 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_507_0_6 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_getindex_511_34 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_call1_511_29 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_cvt_char_511_29 = CallSite< System::Func< CallSite^, System::Object^, char >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, char::typeid, ConversionResultKind::ExplicitCast));
  __site_setindex_531_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_532_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_533_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_534_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_535_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_536_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_537_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_538_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_539_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_cvt_cvt_int_543_0 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_543_0_1 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_543_0_2 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_543_0_3 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_setindex_554_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_555_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_556_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_557_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_558_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_cvt_cvt_int_562_0 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_562_0_1 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_562_0_2 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_getindex_564_34 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeGetIndexAction(__pyx_context, 2));
  __site_call1_564_29 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_cvt_char_564_29 = CallSite< System::Func< CallSite^, System::Object^, char >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, char::typeid, ConversionResultKind::ExplicitCast));
  __site_setindex_573_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_574_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_575_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_576_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_cvt_cvt_int_580_0 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_580_0_1 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_580_0_2 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_580_0_3 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_580_0_4 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_setindex_596_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_597_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_598_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_599_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_600_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_601_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_cvt_cvt_int_605_0 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_605_0_1 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_605_0_2 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_605_0_3 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_605_0_4 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_605_0_5 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_setindex_619_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_620_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_get_append_325_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "append", false));
  __site_call1_325_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_zeros_327_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "zeros", false));
  __site_call3_327_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(3)));
  __site_get_append_333_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "append", false));
  __site_call1_333_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_empty_335_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_call3_335_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(3)));
  __site_get_append_341_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "append", false));
  __site_call1_341_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_empty_343_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "empty", false));
  __site_call3_343_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(3)));
  __site_get_Array_358_53 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Array", false));
  __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_358_53 = CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, __pyx_t_5numpy_6linalg_5numpy_npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_Array_361_48 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Array", false));
  __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_361_48 = CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, __pyx_t_5numpy_6linalg_5numpy_npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_Array_364_48 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Array", false));
  __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_364_48 = CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, __pyx_t_5numpy_6linalg_5numpy_npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_Array_367_69 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Array", false));
  __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_367_69 = CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, __pyx_t_5numpy_6linalg_5numpy_npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_Array_370_52 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Array", false));
  __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_370_52 = CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, __pyx_t_5numpy_6linalg_5numpy_npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_ndarray_377_22 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "ndarray", false));
  __site_get_ArrayReturn_377_30 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "ArrayReturn", false));
  __site_call1_377_42 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_Array_380_47 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Array", false));
  __site_cvt_cvt_PY_LONG_LONG_380_47 = CallSite< System::Func< CallSite^, System::Object^, PY_LONG_LONG >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, PY_LONG_LONG::typeid, ConversionResultKind::ExplicitCast));
  __site_get_ndim_383_14 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "ndim", false));
  __site_get_Array_386_48 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Array", false));
  __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_386_48 = CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, __pyx_t_5numpy_6linalg_5numpy_npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_Array_389_51 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Array", false));
  __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_389_51 = CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, __pyx_t_5numpy_6linalg_5numpy_npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_Array_392_51 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Array", false));
  __site_cvt_cvt_PY_LONG_LONG_392_51 = CallSite< System::Func< CallSite^, System::Object^, PY_LONG_LONG >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, PY_LONG_LONG::typeid, ConversionResultKind::ExplicitCast));
  __site_get_Array_395_34 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Array", false));
  __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_395_34 = CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, __pyx_t_5numpy_6linalg_5numpy_npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_Array_398_48 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Array", false));
  __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_398_48 = CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, __pyx_t_5numpy_6linalg_5numpy_npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_NpyArray_403_40 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "NpyArray", false));
  __site_get_Zero_403_49 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Zero", false));
  __site_call1_403_54 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_403_54 = CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, __pyx_t_5numpy_6linalg_5numpy_npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_Dtype_411_62 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Dtype", false));
  __site_cvt_cvt_PY_LONG_LONG_411_62 = CallSite< System::Func< CallSite^, System::Object^, PY_LONG_LONG >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, PY_LONG_LONG::typeid, ConversionResultKind::ExplicitCast));
  __site_get_NpyArray_416_22 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "NpyArray", false));
  __site_get_FromAny_416_31 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "FromAny", false));
  __site_call6_416_39 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(6)));
  __site_op_and_425_13 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::And));
  __site_istrue_425_13 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_op_ior_426_14 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::OrAssign));
  __site_cvt_cvt_int_427_77 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_cvt_int_430_78 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_get_NpyArray_436_22 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "NpyArray", false));
  __site_get_CheckFromAny_436_31 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "CheckFromAny", false));
  __site_call6_436_44 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(6)));
  __site_get_ndarray_440_29 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "ndarray", false));
  __site_call2_440_21 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_NpyCoreApi_445_22 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "NpyCoreApi", false));
  __site_get_CastToType_445_33 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "CastToType", false));
  __site_cvt_cvt_int_445_100 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_call3_445_44 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(3)));
  __site_get_NpyDescr_453_22 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "NpyDescr", false));
  __site_get_DescrConverter_453_31 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "DescrConverter", false));
  __site_call1_453_46 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_call2_458_21 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_cvt_bool_458_45 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_get_ScalarGeneric_458_73 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "ScalarGeneric", false));
  __site_call2_458_58 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_Array_461_51 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Array", false));
  __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_461_51 = CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, __pyx_t_5numpy_6linalg_5numpy_npy_intp::typeid, ConversionResultKind::ExplicitCast));
  __site_get_Array_464_58 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Array", false));
  __site_cvt_cvt___pyx_t_5numpy_6linalg_5numpy_npy_intp_464_58 = CallSite< System::Func< CallSite^, System::Object^, __pyx_t_5numpy_6linalg_5numpy_npy_intp >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, __pyx_t_5numpy_6linalg_5numpy_npy_intp::typeid, ConversionResultKind::ExplicitCast));
}
[SpecialName]
static void PerformModuleReload(PythonContext^ context, PythonDictionary^ dict) {
  dict["__builtins__"] = context->BuiltinModuleInstance;
  __pyx_context = (gcnew ModuleContext(dict, context))->GlobalContext;
  __Pyx_InitSites(__pyx_context);
  __Pyx_InitGlobals();
  /*--- Type init code ---*/
  /*--- Type import code ---*/
  // XXX skipping type ptr assignment for NumpyDotNet::ndarray
  // XXX skipping type ptr assignment for NumpyDotNet::dtype
  /*--- Create function pointers ---*/
  /*--- Execution code ---*/
  PythonDictionary^ __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":29
 * 
 * cimport numpy as np
 * np.import_array()             # <<<<<<<<<<<<<<
 * 
 * class LapackError(Exception):
 */
  import_array();

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":31
 * np.import_array()
 * 
 * class LapackError(Exception):             # <<<<<<<<<<<<<<
 *     pass
 * 
 */
  __pyx_t_1 = PythonOps::MakeEmptyDict();
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "Exception");
  __pyx_t_3 = PythonOps::MakeTuple(gcnew array<System::Object^>{__pyx_t_2});
  __pyx_t_2 = nullptr;
  FunctionCode^ func_code_LapackError = PythonOps::MakeFunctionCode(__pyx_context, "func_code_LapackError", nullptr, gcnew array<System::String^>{"arg0"}, FunctionAttributes::None, 0, 0, "", gcnew System::Func<CodeContext^, CodeContext^>(mk_empty_context), gcnew array<System::String^>(0), gcnew array<System::String^>(0), gcnew array<System::String^>(0), gcnew array<System::String^>(0), 0);
  PythonTuple^ tbases_LapackError = safe_cast<PythonTuple^>(__pyx_t_3);
  array<System::Object^>^ bases_LapackError = gcnew array<System::Object^>(tbases_LapackError->Count);
  tbases_LapackError->CopyTo(bases_LapackError, 0);
  __pyx_t_2 = PythonOps::MakeClass(func_code_LapackError, nullptr, __pyx_context, "LapackError", bases_LapackError, "");
  __pyx_t_3 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "LapackError", __pyx_t_2);
  __pyx_t_2 = nullptr;
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\lapack_lite.pyx":1
 * """ Cythonized version of lapack_litemodule.c             # <<<<<<<<<<<<<<
 * """
 * 
 */
  __pyx_t_1 = PythonOps::MakeEmptyDict();
  PythonOps::SetGlobal(__pyx_context, "__test__", ((System::Object^)__pyx_t_1));
  __pyx_t_1 = nullptr;

  /* "Z:\dev\numpy-refactor\numpy\linalg\numpy.pxd":488
 *     return NpyArrayNeighborhoodIter_Next(iter)
 * 
 * cdef inline ndarray NpyIter_ARRAY(NpyArrayIterObject *iter):             # <<<<<<<<<<<<<<
 *     return Npy_INTERFACE_array(iter.ao)
 */
}
/* Cython code section 'cleanup_globals' */
/* Cython code section 'cleanup_module' */
/* Cython code section 'main_method' */
/* Cython code section 'dotnet_globals' */



static Types::PythonType^ __pyx_ptype_5numpy_6linalg_5numpy_ndarray = nullptr;
static Types::PythonType^ __pyx_ptype_5numpy_6linalg_5numpy_dtype = nullptr;

/* Cython code section 'utility_code_def' */

/* Runtime support code */

#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float x, float y) {
      return ::std::complex< float >(x, y);
    }
  #else
    static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float x, float y) {
      return x + y*(__pyx_t_float_complex)_Complex_I;
    }
  #endif
#else
    static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float x, float y) {
      __pyx_t_float_complex z;
      z.real = x;
      z.imag = y;
      return z;
    }
#endif

#if CYTHON_CCOMPLEX
#else
    static CYTHON_INLINE int __Pyx_c_eqf(__pyx_t_float_complex a, __pyx_t_float_complex b) {
       return (a.real == b.real) && (a.imag == b.imag);
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_sumf(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        __pyx_t_float_complex z;
        z.real = a.real + b.real;
        z.imag = a.imag + b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_difff(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        __pyx_t_float_complex z;
        z.real = a.real - b.real;
        z.imag = a.imag - b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_prodf(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        __pyx_t_float_complex z;
        z.real = a.real * b.real - a.imag * b.imag;
        z.imag = a.real * b.imag + a.imag * b.real;
        return z;
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_quotf(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        __pyx_t_float_complex z;
        float denom = b.real * b.real + b.imag * b.imag;
        z.real = (a.real * b.real + a.imag * b.imag) / denom;
        z.imag = (a.imag * b.real - a.real * b.imag) / denom;
        return z;
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_negf(__pyx_t_float_complex a) {
        __pyx_t_float_complex z;
        z.real = -a.real;
        z.imag = -a.imag;
        return z;
    }
    static CYTHON_INLINE int __Pyx_c_is_zerof(__pyx_t_float_complex a) {
       return (a.real == 0) && (a.imag == 0);
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_conjf(__pyx_t_float_complex a) {
        __pyx_t_float_complex z;
        z.real =  a.real;
        z.imag = -a.imag;
        return z;
    }
/*
    static CYTHON_INLINE float __Pyx_c_absf(__pyx_t_float_complex z) {
#if HAVE_HYPOT
        return hypotf(z.real, z.imag);
#else
        return sqrtf(z.real*z.real + z.imag*z.imag);
#endif
    }
*/
#endif

#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double x, double y) {
      return ::std::complex< double >(x, y);
    }
  #else
    static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double x, double y) {
      return x + y*(__pyx_t_double_complex)_Complex_I;
    }
  #endif
#else
    static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double x, double y) {
      __pyx_t_double_complex z;
      z.real = x;
      z.imag = y;
      return z;
    }
#endif

#if CYTHON_CCOMPLEX
#else
    static CYTHON_INLINE int __Pyx_c_eq(__pyx_t_double_complex a, __pyx_t_double_complex b) {
       return (a.real == b.real) && (a.imag == b.imag);
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_sum(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        __pyx_t_double_complex z;
        z.real = a.real + b.real;
        z.imag = a.imag + b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_diff(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        __pyx_t_double_complex z;
        z.real = a.real - b.real;
        z.imag = a.imag - b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_prod(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        __pyx_t_double_complex z;
        z.real = a.real * b.real - a.imag * b.imag;
        z.imag = a.real * b.imag + a.imag * b.real;
        return z;
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_quot(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        __pyx_t_double_complex z;
        double denom = b.real * b.real + b.imag * b.imag;
        z.real = (a.real * b.real + a.imag * b.imag) / denom;
        z.imag = (a.imag * b.real - a.real * b.imag) / denom;
        return z;
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_neg(__pyx_t_double_complex a) {
        __pyx_t_double_complex z;
        z.real = -a.real;
        z.imag = -a.imag;
        return z;
    }
    static CYTHON_INLINE int __Pyx_c_is_zero(__pyx_t_double_complex a) {
       return (a.real == 0) && (a.imag == 0);
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_conj(__pyx_t_double_complex a) {
        __pyx_t_double_complex z;
        z.real =  a.real;
        z.imag = -a.imag;
        return z;
    }
/*
    static CYTHON_INLINE double __Pyx_c_abs(__pyx_t_double_complex z) {
#if HAVE_HYPOT
        return hypot(z.real, z.imag);
#else
        return sqrt(z.real*z.real + z.imag*z.imag);
#endif
    }
*/
#endif
/* Cython code section 'end' */
};
[assembly: PythonModule("numpy__linalg__lapack_lite", module_lapack_lite::typeid)];
};
