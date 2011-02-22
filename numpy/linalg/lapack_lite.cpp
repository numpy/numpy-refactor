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
#include "npy_defs.h"
#include "npy_arrayobject.h"
#include "npy_descriptor.h"
#include "npy_ufunc_object.h"
#include "npy_api.h"
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

/* Cython code section 'filename_table' */

static const char *__pyx_f[] = {
  0
};
/* Cython code section 'utility_code_proto_before_types' */
/* Cython code section 'numeric_typedefs' */

typedef int __pyx_t_5numpy_6linalg_5numpy_npy_int;

typedef double __pyx_t_5numpy_6linalg_5numpy_double_t;

typedef int __pyx_t_5numpy_6linalg_5numpy_npy_intp;

typedef signed char __pyx_t_5numpy_6linalg_5numpy_npy_int8;

typedef signed short __pyx_t_5numpy_6linalg_5numpy_npy_int16;

typedef signed int __pyx_t_5numpy_6linalg_5numpy_npy_int32;

typedef signed PY_LONG_LONG __pyx_t_5numpy_6linalg_5numpy_npy_int64;

typedef unsigned char __pyx_t_5numpy_6linalg_5numpy_npy_uint8;

typedef unsigned short __pyx_t_5numpy_6linalg_5numpy_npy_uint16;

typedef unsigned int __pyx_t_5numpy_6linalg_5numpy_npy_uint32;

typedef unsigned PY_LONG_LONG __pyx_t_5numpy_6linalg_5numpy_npy_uint64;

typedef float __pyx_t_5numpy_6linalg_5numpy_npy_float32;

typedef double __pyx_t_5numpy_6linalg_5numpy_npy_float64;

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
/* Cython code section 'type_declarations' */

/* Type declarations */

typedef void (*__pyx_t_5numpy_6linalg_5numpy_PyUFuncGenericFunction)(char **, __pyx_t_5numpy_6linalg_5numpy_npy_intp *, __pyx_t_5numpy_6linalg_5numpy_npy_intp *, void *);
/* Cython code section 'utility_code_proto' */
/* Cython code section 'module_declarations' */
/* Module declarations from numpy */
/* Module declarations from numpy.linalg.numpy */
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_5numpy_PyUFunc_FromFuncAndData(__pyx_t_5numpy_6linalg_5numpy_PyUFuncGenericFunction *, void **, char *, int, int, int, int, char *, char *, int);
static CYTHON_INLINE System::Object^ PyUFunc_FromFuncAndData(__pyx_t_5numpy_6linalg_5numpy_PyUFuncGenericFunction *, void **, char *, int, int, int, int, char *, char *, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_5numpy_PyArray_ZEROS(int, __pyx_t_5numpy_6linalg_5numpy_intp_t *, int, int);
static CYTHON_INLINE System::Object^ PyArray_ZEROS(int, __pyx_t_5numpy_6linalg_5numpy_intp_t *, int, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_5numpy_PyArray_New(void *, int, __pyx_t_5numpy_6linalg_5numpy_npy_intp *, int, __pyx_t_5numpy_6linalg_5numpy_npy_intp *, void *, int, int, void *);
static CYTHON_INLINE System::Object^ PyArray_New(void *, int, __pyx_t_5numpy_6linalg_5numpy_npy_intp *, int, __pyx_t_5numpy_6linalg_5numpy_npy_intp *, void *, int, int, void *); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate int __pyx_delegate_t_5numpy_6linalg_5numpy_PyArray_CHKFLAGS(NumpyDotNet::ndarray^, int);
static CYTHON_INLINE int PyArray_CHKFLAGS(NumpyDotNet::ndarray^, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate void *__pyx_delegate_t_5numpy_6linalg_5numpy_PyArray_DATA(NumpyDotNet::ndarray^);
static CYTHON_INLINE void *PyArray_DATA(NumpyDotNet::ndarray^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate __pyx_t_5numpy_6linalg_5numpy_intp_t *__pyx_delegate_t_5numpy_6linalg_5numpy_PyArray_DIMS(NumpyDotNet::ndarray^);
static CYTHON_INLINE __pyx_t_5numpy_6linalg_5numpy_intp_t *PyArray_DIMS(NumpyDotNet::ndarray^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate __pyx_t_5numpy_6linalg_5numpy_intp_t __pyx_delegate_t_5numpy_6linalg_5numpy_PyArray_SIZE(NumpyDotNet::ndarray^);
static CYTHON_INLINE __pyx_t_5numpy_6linalg_5numpy_intp_t PyArray_SIZE(NumpyDotNet::ndarray^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_5numpy_PyArray_FromAny(System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^);
static CYTHON_INLINE System::Object^ PyArray_FromAny(System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_5numpy_PyArray_FROMANY(System::Object^, System::Object^, System::Object^, System::Object^, System::Object^);
static CYTHON_INLINE System::Object^ PyArray_FROMANY(System::Object^, System::Object^, System::Object^, System::Object^, System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_5numpy_PyArray_Check(System::Object^);
static CYTHON_INLINE System::Object^ PyArray_Check(System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_5numpy_PyArray_NDIM(System::Object^);
static CYTHON_INLINE System::Object^ PyArray_NDIM(System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate void __pyx_delegate_t_5numpy_6linalg_5numpy_import_array(void);
static CYTHON_INLINE void import_array(void); /*proto*/
/* Module declarations from numpy.linalg.lapack_lite */
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate int __pyx_delegate_t_5numpy_6linalg_11lapack_lite_check_object(System::Object^, int, char *, char *, char *);
static int check_object(System::Object^, int, char *, char *, char *); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_11lapack_lite_dgeev(char, char, int, System::Object^, int, System::Object^, System::Object^, System::Object^, int, System::Object^, int, System::Object^, int, int);
static System::Object^ dgeev(char, char, int, System::Object^, int, System::Object^, System::Object^, System::Object^, int, System::Object^, int, System::Object^, int, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_11lapack_lite_dsyevd(char, char, int, System::Object^, int, System::Object^, System::Object^, int, System::Object^, int, int);
static System::Object^ dsyevd(char, char, int, System::Object^, int, System::Object^, System::Object^, int, System::Object^, int, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_11lapack_lite_zheevd(char, char, int, System::Object^, int, System::Object^, System::Object^, int, System::Object^, int, System::Object^, int, int);
static System::Object^ zheevd(char, char, int, System::Object^, int, System::Object^, System::Object^, int, System::Object^, int, System::Object^, int, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_11lapack_lite_dgelsd(int, int, int, System::Object^, int, System::Object^, int, System::Object^, double, int, System::Object^, int, System::Object^, int);
static System::Object^ dgelsd(int, int, int, System::Object^, int, System::Object^, int, System::Object^, double, int, System::Object^, int, System::Object^, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_11lapack_lite_dgesv(int, int, System::Object^, int, System::Object^, System::Object^, int, int);
static System::Object^ dgesv(int, int, System::Object^, int, System::Object^, System::Object^, int, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_11lapack_lite_dgesdd(char, int, int, System::Object^, int, System::Object^, System::Object^, int, System::Object^, int, System::Object^, int, System::Object^, int);
static System::Object^ dgesdd(char, int, int, System::Object^, int, System::Object^, System::Object^, int, System::Object^, int, System::Object^, int, System::Object^, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_11lapack_lite_dgetrf(int, int, System::Object^, int, System::Object^, int);
static System::Object^ dgetrf(int, int, System::Object^, int, System::Object^, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_11lapack_lite_dpotrf(char, int, System::Object^, int, int);
static System::Object^ dpotrf(char, int, System::Object^, int, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_11lapack_lite_dgeqrf(int, int, System::Object^, int, System::Object^, System::Object^, int, int);
static System::Object^ dgeqrf(int, int, System::Object^, int, System::Object^, System::Object^, int, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_11lapack_lite_dorgqr(int, int, int, System::Object^, int, System::Object^, System::Object^, int, int);
static System::Object^ dorgqr(int, int, int, System::Object^, int, System::Object^, System::Object^, int, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_11lapack_lite_zgeev(char, char, int, System::Object^, int, System::Object^, System::Object^, int, System::Object^, int, System::Object^, int, System::Object^, int);
static System::Object^ zgeev(char, char, int, System::Object^, int, System::Object^, System::Object^, int, System::Object^, int, System::Object^, int, System::Object^, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_11lapack_lite_zgelsd(int, int, int, System::Object^, int, System::Object^, int, System::Object^, double, int, System::Object^, int, System::Object^, System::Object^, int);
static System::Object^ zgelsd(int, int, int, System::Object^, int, System::Object^, int, System::Object^, double, int, System::Object^, int, System::Object^, System::Object^, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_11lapack_lite_zgesv(int, int, System::Object^, int, System::Object^, System::Object^, int, int);
static System::Object^ zgesv(int, int, System::Object^, int, System::Object^, System::Object^, int, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_11lapack_lite_zgesdd(char, int, int, System::Object^, int, System::Object^, System::Object^, int, System::Object^, int, System::Object^, int, System::Object^, System::Object^, int);
static System::Object^ zgesdd(char, int, int, System::Object^, int, System::Object^, System::Object^, int, System::Object^, int, System::Object^, int, System::Object^, System::Object^, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_11lapack_lite_zgetrf(int, int, System::Object^, int, System::Object^, int);
static System::Object^ zgetrf(int, int, System::Object^, int, System::Object^, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_11lapack_lite_zpotrf(char, int, System::Object^, int, int);
static System::Object^ zpotrf(char, int, System::Object^, int, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_11lapack_lite_zgeqrf(int, int, System::Object^, int, System::Object^, System::Object^, int, int);
static System::Object^ zgeqrf(int, int, System::Object^, int, System::Object^, System::Object^, int, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_6linalg_11lapack_lite_zungqr(int, int, int, System::Object^, int, System::Object^, System::Object^, int, int);
static System::Object^ zungqr(int, int, int, System::Object^, int, System::Object^, System::Object^, int, int); /*proto*/
/* Cython code section 'typeinfo' */
/* Cython code section 'before_global_var' */
#define __Pyx_MODULE_NAME "numpy.linalg.lapack_lite"

/* Implementation of numpy.linalg.lapack_lite */
namespace clr_lapack_lite {
  public ref class module_lapack_lite sealed abstract {
/* Cython code section 'global_var' */
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_36_27;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_mod_37_81;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_37_25;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_PyArray_FLAGS_38_16;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_38_30;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_and_38_35;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_38_35;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_mod_39_77;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_39_25;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_PyArray_TYPE_40_11;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_40_24;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ne_40_29;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_40_29;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_mod_41_77;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_41_25;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_PyArray_DESCR_42_11;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_42_25;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_byteorder_42_29;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ne_42_40;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_42_40;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_PyArray_DESCR_42_53;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_42_67;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_byteorder_42_71;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ne_42_82;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_42_82;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_mod_43_85;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_43_25;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_70_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_71_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_72_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_73_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_74_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_75_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_76_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_77_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_78_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_137_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_138_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_139_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_140_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_141_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_142_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_143_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_144_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_207_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_208_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_209_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_210_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_211_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_212_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_213_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_214_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_215_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_238_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_239_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_240_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_241_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_242_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_243_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_244_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_245_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_246_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_247_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_266_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_267_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_268_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_269_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_270_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_271_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_316_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_317_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_318_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_319_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_320_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_321_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_322_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_323_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_324_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_338_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_339_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_340_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_341_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_342_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_356_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_357_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_358_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_359_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_378_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_379_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_380_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_381_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_382_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_383_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_401_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_402_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_427_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_428_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_429_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_430_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_431_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_432_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_433_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_434_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_435_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_461_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_462_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_463_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_464_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_465_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_466_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_467_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_468_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_469_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_487_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_488_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_489_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_490_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_491_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_492_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_519_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_520_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_521_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_522_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_523_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_524_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_525_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_526_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_527_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_542_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_543_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_544_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_545_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_546_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_560_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_561_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_562_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_563_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_582_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_583_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_584_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_585_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_586_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_587_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_604_10;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_setindex_605_10;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_append_198_18;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_198_25;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_zeros_200_16;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call3_200_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Array_209_54;
static  CallSite< System::Func< CallSite^, System::Object^, PY_LONG_LONG >^ >^ __site_cvt_PY_LONG_LONG_209_54;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Array_213_49;
static  CallSite< System::Func< CallSite^, System::Object^, PY_LONG_LONG >^ >^ __site_cvt_PY_LONG_LONG_213_49;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Array_217_49;
static  CallSite< System::Func< CallSite^, System::Object^, PY_LONG_LONG >^ >^ __site_cvt_PY_LONG_LONG_217_49;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_Array_221_49;
static  CallSite< System::Func< CallSite^, System::Object^, PY_LONG_LONG >^ >^ __site_cvt_PY_LONG_LONG_221_49;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_NpyArray_226_22;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_FromAny_226_31;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call6_226_39;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_and_229_13;
static  CallSite< System::Func< CallSite^, System::Object^, bool >^ >^ __site_istrue_229_13;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_ior_230_14;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_int_231_77;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call2_234_21;
static  CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >^ __site_get_ndim_237_14;
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":35
 * 
 * 
 * cdef int check_object(object ob, int t, char *obname, char *tname, char *funname):             # <<<<<<<<<<<<<<
 *     if not np.PyArray_Check(ob):
 *         raise LapackError("Expected an array for parameter %s in lapack_lite.%s" % (obname, funname))
 */

static  int check_object(System::Object^ __pyx_v_ob, int __pyx_v_t, char *__pyx_v_obname, char *__pyx_v_tname, char *__pyx_v_funname) {
  int __pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  int __pyx_t_2;
  int __pyx_t_3;
  System::Object^ __pyx_t_4 = nullptr;
  System::Object^ __pyx_t_5 = nullptr;
  System::Object^ __pyx_t_6 = nullptr;
  System::Object^ __pyx_t_7 = nullptr;
  int __pyx_t_8;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":36
 * 
 * cdef int check_object(object ob, int t, char *obname, char *tname, char *funname):
 *     if not np.PyArray_Check(ob):             # <<<<<<<<<<<<<<
 *         raise LapackError("Expected an array for parameter %s in lapack_lite.%s" % (obname, funname))
 *     elif not (np.PyArray_FLAGS(ob) & np.NPY_CONTIGUOUS):
 */
  __pyx_t_1 = PyArray_Check(__pyx_v_ob); 
  __pyx_t_2 = __site_istrue_36_27->Target(__site_istrue_36_27, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_t_3 = (!__pyx_t_2);
  if (__pyx_t_3) {

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":37
 * cdef int check_object(object ob, int t, char *obname, char *tname, char *funname):
 *     if not np.PyArray_Check(ob):
 *         raise LapackError("Expected an array for parameter %s in lapack_lite.%s" % (obname, funname))             # <<<<<<<<<<<<<<
 *     elif not (np.PyArray_FLAGS(ob) & np.NPY_CONTIGUOUS):
 *         raise LapackError("Parameter %s is not contiguous in lapack_lite.%s" % (obname, funname))
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "LapackError");
    __pyx_t_4 = gcnew System::String(__pyx_v_obname);
    __pyx_t_5 = gcnew System::String(__pyx_v_funname);
    __pyx_t_6 = PythonOps::MakeTuple(gcnew array<System::Object^>{((System::Object^)__pyx_t_4), ((System::Object^)__pyx_t_5)});
    __pyx_t_4 = nullptr;
    __pyx_t_5 = nullptr;
    __pyx_t_5 = __site_op_mod_37_81->Target(__site_op_mod_37_81, ((System::Object^)"Expected an array for parameter %s in lapack_lite.%s"), __pyx_t_6);
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_call1_37_25->Target(__site_call1_37_25, __pyx_context, __pyx_t_1, ((System::Object^)__pyx_t_5));
    __pyx_t_1 = nullptr;
    __pyx_t_5 = nullptr;
    throw PythonOps::MakeException(__pyx_context, __pyx_t_6, nullptr, nullptr);
    __pyx_t_6 = nullptr;
    goto __pyx_L3;
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":38
 *     if not np.PyArray_Check(ob):
 *         raise LapackError("Expected an array for parameter %s in lapack_lite.%s" % (obname, funname))
 *     elif not (np.PyArray_FLAGS(ob) & np.NPY_CONTIGUOUS):             # <<<<<<<<<<<<<<
 *         raise LapackError("Parameter %s is not contiguous in lapack_lite.%s" % (obname, funname))
 *     elif np.PyArray_TYPE(ob) != t:
 */
  __pyx_t_6 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_5 = __site_get_PyArray_FLAGS_38_16->Target(__site_get_PyArray_FLAGS_38_16, __pyx_t_6, __pyx_context);
  __pyx_t_6 = nullptr;
  __pyx_t_6 = __site_call1_38_30->Target(__site_call1_38_30, __pyx_context, __pyx_t_5, __pyx_v_ob);
  __pyx_t_5 = nullptr;
  __pyx_t_5 = (System::Object^)(long long)(NPY_CONTIGUOUS);
  __pyx_t_1 = __site_op_and_38_35->Target(__site_op_and_38_35, __pyx_t_6, __pyx_t_5);
  __pyx_t_6 = nullptr;
  __pyx_t_5 = nullptr;
  __pyx_t_3 = __site_istrue_38_35->Target(__site_istrue_38_35, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_t_2 = (!__pyx_t_3);
  if (__pyx_t_2) {

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":39
 *         raise LapackError("Expected an array for parameter %s in lapack_lite.%s" % (obname, funname))
 *     elif not (np.PyArray_FLAGS(ob) & np.NPY_CONTIGUOUS):
 *         raise LapackError("Parameter %s is not contiguous in lapack_lite.%s" % (obname, funname))             # <<<<<<<<<<<<<<
 *     elif np.PyArray_TYPE(ob) != t:
 *         raise LapackError("Parameter %s is not of type %s in lapack_lite.%s" % (obname, tname, funname))
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "LapackError");
    __pyx_t_5 = gcnew System::String(__pyx_v_obname);
    __pyx_t_6 = gcnew System::String(__pyx_v_funname);
    __pyx_t_4 = PythonOps::MakeTuple(gcnew array<System::Object^>{((System::Object^)__pyx_t_5), ((System::Object^)__pyx_t_6)});
    __pyx_t_5 = nullptr;
    __pyx_t_6 = nullptr;
    __pyx_t_6 = __site_op_mod_39_77->Target(__site_op_mod_39_77, ((System::Object^)"Parameter %s is not contiguous in lapack_lite.%s"), __pyx_t_4);
    __pyx_t_4 = nullptr;
    __pyx_t_4 = __site_call1_39_25->Target(__site_call1_39_25, __pyx_context, __pyx_t_1, ((System::Object^)__pyx_t_6));
    __pyx_t_1 = nullptr;
    __pyx_t_6 = nullptr;
    throw PythonOps::MakeException(__pyx_context, __pyx_t_4, nullptr, nullptr);
    __pyx_t_4 = nullptr;
    goto __pyx_L3;
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":40
 *     elif not (np.PyArray_FLAGS(ob) & np.NPY_CONTIGUOUS):
 *         raise LapackError("Parameter %s is not contiguous in lapack_lite.%s" % (obname, funname))
 *     elif np.PyArray_TYPE(ob) != t:             # <<<<<<<<<<<<<<
 *         raise LapackError("Parameter %s is not of type %s in lapack_lite.%s" % (obname, tname, funname))
 *     elif np.PyArray_DESCR(ob).byteorder != '=' and np.PyArray_DESCR(ob).byteorder != '|':
 */
  __pyx_t_4 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_6 = __site_get_PyArray_TYPE_40_11->Target(__site_get_PyArray_TYPE_40_11, __pyx_t_4, __pyx_context);
  __pyx_t_4 = nullptr;
  __pyx_t_4 = __site_call1_40_24->Target(__site_call1_40_24, __pyx_context, __pyx_t_6, __pyx_v_ob);
  __pyx_t_6 = nullptr;
  __pyx_t_6 = __pyx_v_t;
  __pyx_t_1 = __site_op_ne_40_29->Target(__site_op_ne_40_29, __pyx_t_4, __pyx_t_6);
  __pyx_t_4 = nullptr;
  __pyx_t_6 = nullptr;
  __pyx_t_2 = __site_istrue_40_29->Target(__site_istrue_40_29, __pyx_t_1);
  __pyx_t_1 = nullptr;
  if (__pyx_t_2) {

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":41
 *         raise LapackError("Parameter %s is not contiguous in lapack_lite.%s" % (obname, funname))
 *     elif np.PyArray_TYPE(ob) != t:
 *         raise LapackError("Parameter %s is not of type %s in lapack_lite.%s" % (obname, tname, funname))             # <<<<<<<<<<<<<<
 *     elif np.PyArray_DESCR(ob).byteorder != '=' and np.PyArray_DESCR(ob).byteorder != '|':
 *         raise LapackError("Parameter %s has non-native byte order in lapack_lite.%s" % (obname, funname))
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "LapackError");
    __pyx_t_6 = gcnew System::String(__pyx_v_obname);
    __pyx_t_4 = gcnew System::String(__pyx_v_tname);
    __pyx_t_5 = gcnew System::String(__pyx_v_funname);
    __pyx_t_7 = PythonOps::MakeTuple(gcnew array<System::Object^>{((System::Object^)__pyx_t_6), ((System::Object^)__pyx_t_4), ((System::Object^)__pyx_t_5)});
    __pyx_t_6 = nullptr;
    __pyx_t_4 = nullptr;
    __pyx_t_5 = nullptr;
    __pyx_t_5 = __site_op_mod_41_77->Target(__site_op_mod_41_77, ((System::Object^)"Parameter %s is not of type %s in lapack_lite.%s"), __pyx_t_7);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_call1_41_25->Target(__site_call1_41_25, __pyx_context, __pyx_t_1, ((System::Object^)__pyx_t_5));
    __pyx_t_1 = nullptr;
    __pyx_t_5 = nullptr;
    throw PythonOps::MakeException(__pyx_context, __pyx_t_7, nullptr, nullptr);
    __pyx_t_7 = nullptr;
    goto __pyx_L3;
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":42
 *     elif np.PyArray_TYPE(ob) != t:
 *         raise LapackError("Parameter %s is not of type %s in lapack_lite.%s" % (obname, tname, funname))
 *     elif np.PyArray_DESCR(ob).byteorder != '=' and np.PyArray_DESCR(ob).byteorder != '|':             # <<<<<<<<<<<<<<
 *         raise LapackError("Parameter %s has non-native byte order in lapack_lite.%s" % (obname, funname))
 * 
 */
  __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
  __pyx_t_5 = __site_get_PyArray_DESCR_42_11->Target(__site_get_PyArray_DESCR_42_11, __pyx_t_7, __pyx_context);
  __pyx_t_7 = nullptr;
  __pyx_t_7 = __site_call1_42_25->Target(__site_call1_42_25, __pyx_context, __pyx_t_5, __pyx_v_ob);
  __pyx_t_5 = nullptr;
  __pyx_t_5 = __site_get_byteorder_42_29->Target(__site_get_byteorder_42_29, __pyx_t_7, __pyx_context);
  __pyx_t_7 = nullptr;
  __pyx_t_7 = __site_op_ne_42_40->Target(__site_op_ne_42_40, __pyx_t_5, ((System::Object^)"="));
  __pyx_t_5 = nullptr;
  __pyx_t_2 = __site_istrue_42_40->Target(__site_istrue_42_40, __pyx_t_7);
  __pyx_t_7 = nullptr;
  if (__pyx_t_2) {
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "np");
    __pyx_t_5 = __site_get_PyArray_DESCR_42_53->Target(__site_get_PyArray_DESCR_42_53, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_call1_42_67->Target(__site_call1_42_67, __pyx_context, __pyx_t_5, __pyx_v_ob);
    __pyx_t_5 = nullptr;
    __pyx_t_5 = __site_get_byteorder_42_71->Target(__site_get_byteorder_42_71, __pyx_t_7, __pyx_context);
    __pyx_t_7 = nullptr;
    __pyx_t_7 = __site_op_ne_42_82->Target(__site_op_ne_42_82, __pyx_t_5, ((System::Object^)"|"));
    __pyx_t_5 = nullptr;
    __pyx_t_3 = __site_istrue_42_82->Target(__site_istrue_42_82, __pyx_t_7);
    __pyx_t_7 = nullptr;
    __pyx_t_8 = __pyx_t_3;
  } else {
    __pyx_t_8 = __pyx_t_2;
  }
  if (__pyx_t_8) {

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":43
 *         raise LapackError("Parameter %s is not of type %s in lapack_lite.%s" % (obname, tname, funname))
 *     elif np.PyArray_DESCR(ob).byteorder != '=' and np.PyArray_DESCR(ob).byteorder != '|':
 *         raise LapackError("Parameter %s has non-native byte order in lapack_lite.%s" % (obname, funname))             # <<<<<<<<<<<<<<
 * 
 *     return 1
 */
    __pyx_t_7 = PythonOps::GetGlobal(__pyx_context, "LapackError");
    __pyx_t_5 = gcnew System::String(__pyx_v_obname);
    __pyx_t_1 = gcnew System::String(__pyx_v_funname);
    __pyx_t_4 = PythonOps::MakeTuple(gcnew array<System::Object^>{((System::Object^)__pyx_t_5), ((System::Object^)__pyx_t_1)});
    __pyx_t_5 = nullptr;
    __pyx_t_1 = nullptr;
    __pyx_t_1 = __site_op_mod_43_85->Target(__site_op_mod_43_85, ((System::Object^)"Parameter %s has non-native byte order in lapack_lite.%s"), __pyx_t_4);
    __pyx_t_4 = nullptr;
    __pyx_t_4 = __site_call1_43_25->Target(__site_call1_43_25, __pyx_context, __pyx_t_7, ((System::Object^)__pyx_t_1));
    __pyx_t_7 = nullptr;
    __pyx_t_1 = nullptr;
    throw PythonOps::MakeException(__pyx_context, __pyx_t_4, nullptr, nullptr);
    __pyx_t_4 = nullptr;
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":45
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":48
 * 
 * 
 * cdef dgeev(char jobvl, char jobvr, int n, object a, int lda,             # <<<<<<<<<<<<<<
 *            object wr, object wi, object vl,
 *            int ldvl, object vr, int ldvr, object work, int lwork, int info):
 */

static  System::Object^ dgeev(char __pyx_v_jobvl, char __pyx_v_jobvr, int __pyx_v_n, System::Object^ __pyx_v_a, int __pyx_v_lda, System::Object^ __pyx_v_wr, System::Object^ __pyx_v_wi, System::Object^ __pyx_v_vl, int __pyx_v_ldvl, System::Object^ __pyx_v_vr, int __pyx_v_ldvr, System::Object^ __pyx_v_work, int __pyx_v_lwork, int __pyx_v_info) {
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_retval = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":53
 *     cdef int lapack_lite_status__
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(wr,np.NPY_DOUBLE,"wr","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(wi,np.NPY_DOUBLE,"wi","np.NPY_DOUBLE","dgeev"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_a, NPY_DOUBLE, __pyx_k__a, __pyx_k_1, __pyx_k__dgeev));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":54
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(wr,np.NPY_DOUBLE,"wr","np.NPY_DOUBLE","dgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(wi,np.NPY_DOUBLE,"wi","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(vl,np.NPY_DOUBLE,"vl","np.NPY_DOUBLE","dgeev"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_wr, NPY_DOUBLE, __pyx_k__wr, __pyx_k_1, __pyx_k__dgeev));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":55
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(wr,np.NPY_DOUBLE,"wr","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(wi,np.NPY_DOUBLE,"wi","np.NPY_DOUBLE","dgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(vl,np.NPY_DOUBLE,"vl","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(vr,np.NPY_DOUBLE,"vr","np.NPY_DOUBLE","dgeev"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_wi, NPY_DOUBLE, __pyx_k__wi, __pyx_k_1, __pyx_k__dgeev));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":56
 *     if not check_object(wr,np.NPY_DOUBLE,"wr","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(wi,np.NPY_DOUBLE,"wi","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(vl,np.NPY_DOUBLE,"vl","np.NPY_DOUBLE","dgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(vr,np.NPY_DOUBLE,"vr","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgeev"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_vl, NPY_DOUBLE, __pyx_k__vl, __pyx_k_1, __pyx_k__dgeev));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":57
 *     if not check_object(wi,np.NPY_DOUBLE,"wi","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(vl,np.NPY_DOUBLE,"vl","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(vr,np.NPY_DOUBLE,"vr","np.NPY_DOUBLE","dgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgeev"): return None
 * 
 */
  __pyx_t_1 = (!check_object(__pyx_v_vr, NPY_DOUBLE, __pyx_k__vr, __pyx_k_1, __pyx_k__dgeev));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":58
 *     if not check_object(vl,np.NPY_DOUBLE,"vl","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(vr,np.NPY_DOUBLE,"vr","np.NPY_DOUBLE","dgeev"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgeev"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_dgeev(&jobvl,&jobvr,&n,
 */
  __pyx_t_1 = (!check_object(__pyx_v_work, NPY_DOUBLE, __pyx_k__work, __pyx_k_1, __pyx_k__dgeev));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L8;
  }
  __pyx_L8:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":61
 * 
 *     lapack_lite_status__ = lapack_dgeev(&jobvl,&jobvr,&n,
 *                                         <double *>np.PyArray_DATA(a),&lda,             # <<<<<<<<<<<<<<
 *                                         <double *>np.PyArray_DATA(wr),
 *                                         <double *>np.PyArray_DATA(wi),
 */
  if (__pyx_v_a == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":62
 *     lapack_lite_status__ = lapack_dgeev(&jobvl,&jobvr,&n,
 *                                         <double *>np.PyArray_DATA(a),&lda,
 *                                         <double *>np.PyArray_DATA(wr),             # <<<<<<<<<<<<<<
 *                                         <double *>np.PyArray_DATA(wi),
 *                                         <double *>np.PyArray_DATA(vl),&ldvl,
 */
  if (__pyx_v_wr == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_wr) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":63
 *                                         <double *>np.PyArray_DATA(a),&lda,
 *                                         <double *>np.PyArray_DATA(wr),
 *                                         <double *>np.PyArray_DATA(wi),             # <<<<<<<<<<<<<<
 *                                         <double *>np.PyArray_DATA(vl),&ldvl,
 *                                         <double *>np.PyArray_DATA(vr),&ldvr,
 */
  if (__pyx_v_wi == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_wi) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":64
 *                                         <double *>np.PyArray_DATA(wr),
 *                                         <double *>np.PyArray_DATA(wi),
 *                                         <double *>np.PyArray_DATA(vl),&ldvl,             # <<<<<<<<<<<<<<
 *                                         <double *>np.PyArray_DATA(vr),&ldvr,
 *                                         <double *>np.PyArray_DATA(work),&lwork,
 */
  if (__pyx_v_vl == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_vl) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":65
 *                                         <double *>np.PyArray_DATA(wi),
 *                                         <double *>np.PyArray_DATA(vl),&ldvl,
 *                                         <double *>np.PyArray_DATA(vr),&ldvr,             # <<<<<<<<<<<<<<
 *                                         <double *>np.PyArray_DATA(work),&lwork,
 *                                         &info)
 */
  if (__pyx_v_vr == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_vr) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":66
 *                                         <double *>np.PyArray_DATA(vl),&ldvl,
 *                                         <double *>np.PyArray_DATA(vr),&ldvr,
 *                                         <double *>np.PyArray_DATA(work),&lwork,             # <<<<<<<<<<<<<<
 *                                         &info)
 * 
 */
  if (__pyx_v_work == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":67
 *                                         <double *>np.PyArray_DATA(vr),&ldvr,
 *                                         <double *>np.PyArray_DATA(work),&lwork,
 *                                         &info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(dgeev_)((&__pyx_v_jobvl), (&__pyx_v_jobvr), (&__pyx_v_n), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_a))), (&__pyx_v_lda), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_wr))), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_wi))), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_vl))), (&__pyx_v_ldvl), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_vr))), (&__pyx_v_ldvr), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_work))), (&__pyx_v_lwork), (&__pyx_v_info));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":69
 *                                         &info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["dgeev_"] = lapack_lite_status__
 *     retval["jobvl"] = jobvl
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":70
 * 
 *     retval = {}
 *     retval["dgeev_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["jobvl"] = jobvl
 *     retval["jobvr"] = jobvr
 */
  __pyx_t_2 = __pyx_v_lapack_lite_status__;
  __site_setindex_70_10->Target(__site_setindex_70_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"dgeev_"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":71
 *     retval = {}
 *     retval["dgeev_"] = lapack_lite_status__
 *     retval["jobvl"] = jobvl             # <<<<<<<<<<<<<<
 *     retval["jobvr"] = jobvr
 *     retval["n"] = n
 */
  __pyx_t_2 = __pyx_v_jobvl;
  __site_setindex_71_10->Target(__site_setindex_71_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"jobvl"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":72
 *     retval["dgeev_"] = lapack_lite_status__
 *     retval["jobvl"] = jobvl
 *     retval["jobvr"] = jobvr             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_2 = __pyx_v_jobvr;
  __site_setindex_72_10->Target(__site_setindex_72_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"jobvr"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":73
 *     retval["jobvl"] = jobvl
 *     retval["jobvr"] = jobvr
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["ldvl"] = ldvl
 */
  __pyx_t_2 = __pyx_v_n;
  __site_setindex_73_10->Target(__site_setindex_73_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":74
 *     retval["jobvr"] = jobvr
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["ldvl"] = ldvl
 *     retval["ldvr"] = ldvr
 */
  __pyx_t_2 = __pyx_v_lda;
  __site_setindex_74_10->Target(__site_setindex_74_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":75
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["ldvl"] = ldvl             # <<<<<<<<<<<<<<
 *     retval["ldvr"] = ldvr
 *     retval["lwork"] = lwork
 */
  __pyx_t_2 = __pyx_v_ldvl;
  __site_setindex_75_10->Target(__site_setindex_75_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldvl"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":76
 *     retval["lda"] = lda
 *     retval["ldvl"] = ldvl
 *     retval["ldvr"] = ldvr             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 */
  __pyx_t_2 = __pyx_v_ldvr;
  __site_setindex_76_10->Target(__site_setindex_76_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldvr"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":77
 *     retval["ldvl"] = ldvl
 *     retval["ldvr"] = ldvr
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 * 
 */
  __pyx_t_2 = __pyx_v_lwork;
  __site_setindex_77_10->Target(__site_setindex_77_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":78
 *     retval["ldvr"] = ldvr
 *     retval["lwork"] = lwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 * 
 *     return retval
 */
  __pyx_t_2 = __pyx_v_info;
  __site_setindex_78_10->Target(__site_setindex_78_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":80
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":83
 * 
 * 
 * cdef dsyevd(char jobz, char uplo, int n, object a, int lda,             # <<<<<<<<<<<<<<
 *             object w, object work, int lwork, object iwork, int liwork, int info):
 *     """ Arguments
 */

static  System::Object^ dsyevd(char __pyx_v_jobz, char __pyx_v_uplo, int __pyx_v_n, System::Object^ __pyx_v_a, int __pyx_v_lda, System::Object^ __pyx_v_w, System::Object^ __pyx_v_work, int __pyx_v_lwork, System::Object^ __pyx_v_iwork, int __pyx_v_liwork, int __pyx_v_info) {
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_retval = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":125
 *     cdef int lapack_lite_status__
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dsyevd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(w,np.NPY_DOUBLE,"w","np.NPY_DOUBLE","dsyevd"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dsyevd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_a, NPY_DOUBLE, __pyx_k__a, __pyx_k_1, __pyx_k__dsyevd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":126
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dsyevd"): return None
 *     if not check_object(w,np.NPY_DOUBLE,"w","np.NPY_DOUBLE","dsyevd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dsyevd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","dsyevd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_w, NPY_DOUBLE, __pyx_k__w, __pyx_k_1, __pyx_k__dsyevd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":127
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dsyevd"): return None
 *     if not check_object(w,np.NPY_DOUBLE,"w","np.NPY_DOUBLE","dsyevd"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dsyevd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","dsyevd"): return None
 * 
 */
  __pyx_t_1 = (!check_object(__pyx_v_work, NPY_DOUBLE, __pyx_k__work, __pyx_k_1, __pyx_k__dsyevd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":128
 *     if not check_object(w,np.NPY_DOUBLE,"w","np.NPY_DOUBLE","dsyevd"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dsyevd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","dsyevd"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_dsyevd(&jobz,&uplo,&n,
 */
  __pyx_t_1 = (!check_object(__pyx_v_iwork, NPY_INT, __pyx_k__iwork, __pyx_k_2, __pyx_k__dsyevd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":131
 * 
 *     lapack_lite_status__ = lapack_dsyevd(&jobz,&uplo,&n,
 *                                          <double *>np.PyArray_DATA(a),&lda,             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(w),
 *                                          <double *>np.PyArray_DATA(work),&lwork,
 */
  if (__pyx_v_a == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":132
 *     lapack_lite_status__ = lapack_dsyevd(&jobz,&uplo,&n,
 *                                          <double *>np.PyArray_DATA(a),&lda,
 *                                          <double *>np.PyArray_DATA(w),             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(work),&lwork,
 *                                          <int *>np.PyArray_DATA(iwork),&liwork,&info)
 */
  if (__pyx_v_w == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_w) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":133
 *                                          <double *>np.PyArray_DATA(a),&lda,
 *                                          <double *>np.PyArray_DATA(w),
 *                                          <double *>np.PyArray_DATA(work),&lwork,             # <<<<<<<<<<<<<<
 *                                          <int *>np.PyArray_DATA(iwork),&liwork,&info)
 * 
 */
  if (__pyx_v_work == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":134
 *                                          <double *>np.PyArray_DATA(w),
 *                                          <double *>np.PyArray_DATA(work),&lwork,
 *                                          <int *>np.PyArray_DATA(iwork),&liwork,&info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  if (__pyx_v_iwork == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_iwork) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(dsyevd_)((&__pyx_v_jobz), (&__pyx_v_uplo), (&__pyx_v_n), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_a))), (&__pyx_v_lda), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_w))), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_work))), (&__pyx_v_lwork), ((int *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_iwork))), (&__pyx_v_liwork), (&__pyx_v_info));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":136
 *                                          <int *>np.PyArray_DATA(iwork),&liwork,&info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["dsyevd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":137
 * 
 *     retval = {}
 *     retval["dsyevd_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["jobz"] = jobz
 *     retval["uplo"] = uplo
 */
  __pyx_t_2 = __pyx_v_lapack_lite_status__;
  __site_setindex_137_10->Target(__site_setindex_137_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"dsyevd_"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":138
 *     retval = {}
 *     retval["dsyevd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz             # <<<<<<<<<<<<<<
 *     retval["uplo"] = uplo
 *     retval["n"] = n
 */
  __pyx_t_2 = __pyx_v_jobz;
  __site_setindex_138_10->Target(__site_setindex_138_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"jobz"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":139
 *     retval["dsyevd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz
 *     retval["uplo"] = uplo             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_2 = __pyx_v_uplo;
  __site_setindex_139_10->Target(__site_setindex_139_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"uplo"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":140
 *     retval["jobz"] = jobz
 *     retval["uplo"] = uplo
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork
 */
  __pyx_t_2 = __pyx_v_n;
  __site_setindex_140_10->Target(__site_setindex_140_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":141
 *     retval["uplo"] = uplo
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["liwork"] = liwork
 */
  __pyx_t_2 = __pyx_v_lda;
  __site_setindex_141_10->Target(__site_setindex_141_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":142
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["liwork"] = liwork
 *     retval["info"] = info
 */
  __pyx_t_2 = __pyx_v_lwork;
  __site_setindex_142_10->Target(__site_setindex_142_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":143
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork
 *     retval["liwork"] = liwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_2 = __pyx_v_liwork;
  __site_setindex_143_10->Target(__site_setindex_143_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"liwork"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":144
 *     retval["lwork"] = lwork
 *     retval["liwork"] = liwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_2 = __pyx_v_info;
  __site_setindex_144_10->Target(__site_setindex_144_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":145
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":148
 * 
 * 
 * cdef zheevd(char jobz, char uplo, int n, object a, int lda,             # <<<<<<<<<<<<<<
 *             object w, object work, int lwork,
 *             object rwork, int lrwork,
 */

static  System::Object^ zheevd(char __pyx_v_jobz, char __pyx_v_uplo, int __pyx_v_n, System::Object^ __pyx_v_a, int __pyx_v_lda, System::Object^ __pyx_v_w, System::Object^ __pyx_v_work, int __pyx_v_lwork, System::Object^ __pyx_v_rwork, int __pyx_v_lrwork, System::Object^ __pyx_v_iwork, int __pyx_v_liwork, int __pyx_v_info) {
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_retval = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":193
 *     cdef int lapack_lite_status__
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zheevd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(w,np.NPY_DOUBLE,"w","np.NPY_DOUBLE","zheevd"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zheevd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_a, NPY_CDOUBLE, __pyx_k__a, __pyx_k_3, __pyx_k__zheevd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":194
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zheevd"): return None
 *     if not check_object(w,np.NPY_DOUBLE,"w","np.NPY_DOUBLE","zheevd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zheevd"): return None
 *     if not check_object(w,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zheevd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_w, NPY_DOUBLE, __pyx_k__w, __pyx_k_1, __pyx_k__zheevd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":195
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zheevd"): return None
 *     if not check_object(w,np.NPY_DOUBLE,"w","np.NPY_DOUBLE","zheevd"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zheevd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(w,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zheevd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","zheevd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_work, NPY_CDOUBLE, __pyx_k__work, __pyx_k_3, __pyx_k__zheevd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":196
 *     if not check_object(w,np.NPY_DOUBLE,"w","np.NPY_DOUBLE","zheevd"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zheevd"): return None
 *     if not check_object(w,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zheevd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","zheevd"): return None
 * 
 */
  __pyx_t_1 = (!check_object(__pyx_v_w, NPY_DOUBLE, __pyx_k__rwork, __pyx_k_1, __pyx_k__zheevd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":197
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zheevd"): return None
 *     if not check_object(w,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zheevd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","zheevd"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_zheevd(&jobz,&uplo,&n,
 */
  __pyx_t_1 = (!check_object(__pyx_v_iwork, NPY_INT, __pyx_k__iwork, __pyx_k_2, __pyx_k__zheevd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":200
 * 
 *     lapack_lite_status__ = lapack_zheevd(&jobz,&uplo,&n,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(w),
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 */
  if (__pyx_v_a == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":201
 *     lapack_lite_status__ = lapack_zheevd(&jobz,&uplo,&n,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
 *                                          <double *>np.PyArray_DATA(w),             # <<<<<<<<<<<<<<
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 *                                          <double *>np.PyArray_DATA(rwork),&lrwork,
 */
  if (__pyx_v_w == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_w) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":202
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
 *                                          <double *>np.PyArray_DATA(w),
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(rwork),&lrwork,
 *                                          <int *>np.PyArray_DATA(iwork),&liwork,&info)
 */
  if (__pyx_v_work == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":203
 *                                          <double *>np.PyArray_DATA(w),
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 *                                          <double *>np.PyArray_DATA(rwork),&lrwork,             # <<<<<<<<<<<<<<
 *                                          <int *>np.PyArray_DATA(iwork),&liwork,&info)
 * 
 */
  if (__pyx_v_rwork == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_rwork) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":204
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 *                                          <double *>np.PyArray_DATA(rwork),&lrwork,
 *                                          <int *>np.PyArray_DATA(iwork),&liwork,&info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  if (__pyx_v_iwork == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_iwork) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(zheevd_)((&__pyx_v_jobz), (&__pyx_v_uplo), (&__pyx_v_n), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_a))), (&__pyx_v_lda), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_w))), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_work))), (&__pyx_v_lwork), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_rwork))), (&__pyx_v_lrwork), ((int *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_iwork))), (&__pyx_v_liwork), (&__pyx_v_info));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":206
 *                                          <int *>np.PyArray_DATA(iwork),&liwork,&info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["zheevd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":207
 * 
 *     retval = {}
 *     retval["zheevd_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["jobz"] = jobz
 *     retval["uplo"] = uplo
 */
  __pyx_t_2 = __pyx_v_lapack_lite_status__;
  __site_setindex_207_10->Target(__site_setindex_207_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"zheevd_"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":208
 *     retval = {}
 *     retval["zheevd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz             # <<<<<<<<<<<<<<
 *     retval["uplo"] = uplo
 *     retval["n"] = n
 */
  __pyx_t_2 = __pyx_v_jobz;
  __site_setindex_208_10->Target(__site_setindex_208_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"jobz"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":209
 *     retval["zheevd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz
 *     retval["uplo"] = uplo             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_2 = __pyx_v_uplo;
  __site_setindex_209_10->Target(__site_setindex_209_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"uplo"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":210
 *     retval["jobz"] = jobz
 *     retval["uplo"] = uplo
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork
 */
  __pyx_t_2 = __pyx_v_n;
  __site_setindex_210_10->Target(__site_setindex_210_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":211
 *     retval["uplo"] = uplo
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["lrwork"] = lrwork
 */
  __pyx_t_2 = __pyx_v_lda;
  __site_setindex_211_10->Target(__site_setindex_211_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":212
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["lrwork"] = lrwork
 *     retval["liwork"] = liwork
 */
  __pyx_t_2 = __pyx_v_lwork;
  __site_setindex_212_10->Target(__site_setindex_212_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":213
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork
 *     retval["lrwork"] = lrwork             # <<<<<<<<<<<<<<
 *     retval["liwork"] = liwork
 *     retval["info"] = info
 */
  __pyx_t_2 = __pyx_v_lrwork;
  __site_setindex_213_10->Target(__site_setindex_213_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lrwork"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":214
 *     retval["lwork"] = lwork
 *     retval["lrwork"] = lrwork
 *     retval["liwork"] = liwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_2 = __pyx_v_liwork;
  __site_setindex_214_10->Target(__site_setindex_214_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"liwork"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":215
 *     retval["lrwork"] = lrwork
 *     retval["liwork"] = liwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_2 = __pyx_v_info;
  __site_setindex_215_10->Target(__site_setindex_215_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":216
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":219
 * 
 * 
 * cdef dgelsd(int m, int n, int nrhs, object a, int lda, object b, int ldb,             # <<<<<<<<<<<<<<
 *             object s, double rcond, int rank,
 *             object work, int lwork, object iwork, int info):
 */

static  System::Object^ dgelsd(int __pyx_v_m, int __pyx_v_n, int __pyx_v_nrhs, System::Object^ __pyx_v_a, int __pyx_v_lda, System::Object^ __pyx_v_b, int __pyx_v_ldb, System::Object^ __pyx_v_s, double __pyx_v_rcond, int __pyx_v_rank, System::Object^ __pyx_v_work, int __pyx_v_lwork, System::Object^ __pyx_v_iwork, int __pyx_v_info) {
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_retval = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":224
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
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":225
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
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":226
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
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":227
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
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":228
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
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":231
 * 
 *     lapack_lite_status__ = lapack_dgelsd(&m,&n,&nrhs,
 *                                          <double *>np.PyArray_DATA(a),&lda,             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(b),&ldb,
 *                                          <double *>np.PyArray_DATA(s),&rcond,&rank,
 */
  if (__pyx_v_a == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":232
 *     lapack_lite_status__ = lapack_dgelsd(&m,&n,&nrhs,
 *                                          <double *>np.PyArray_DATA(a),&lda,
 *                                          <double *>np.PyArray_DATA(b),&ldb,             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(s),&rcond,&rank,
 *                                          <double *>np.PyArray_DATA(work),&lwork,
 */
  if (__pyx_v_b == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_b) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":233
 *                                          <double *>np.PyArray_DATA(a),&lda,
 *                                          <double *>np.PyArray_DATA(b),&ldb,
 *                                          <double *>np.PyArray_DATA(s),&rcond,&rank,             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(work),&lwork,
 *                                          <int *>np.PyArray_DATA(iwork),&info)
 */
  if (__pyx_v_s == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_s) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":234
 *                                          <double *>np.PyArray_DATA(b),&ldb,
 *                                          <double *>np.PyArray_DATA(s),&rcond,&rank,
 *                                          <double *>np.PyArray_DATA(work),&lwork,             # <<<<<<<<<<<<<<
 *                                          <int *>np.PyArray_DATA(iwork),&info)
 * 
 */
  if (__pyx_v_work == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":235
 *                                          <double *>np.PyArray_DATA(s),&rcond,&rank,
 *                                          <double *>np.PyArray_DATA(work),&lwork,
 *                                          <int *>np.PyArray_DATA(iwork),&info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  if (__pyx_v_iwork == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_iwork) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(dgelsd_)((&__pyx_v_m), (&__pyx_v_n), (&__pyx_v_nrhs), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_a))), (&__pyx_v_lda), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_b))), (&__pyx_v_ldb), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_s))), (&__pyx_v_rcond), (&__pyx_v_rank), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_work))), (&__pyx_v_lwork), ((int *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_iwork))), (&__pyx_v_info));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":237
 *                                          <int *>np.PyArray_DATA(iwork),&info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["dgelsd_"] = lapack_lite_status__
 *     retval["m"] = m
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":238
 * 
 *     retval = {}
 *     retval["dgelsd_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["m"] = m
 *     retval["n"] = n
 */
  __pyx_t_2 = __pyx_v_lapack_lite_status__;
  __site_setindex_238_10->Target(__site_setindex_238_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"dgelsd_"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":239
 *     retval = {}
 *     retval["dgelsd_"] = lapack_lite_status__
 *     retval["m"] = m             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs
 */
  __pyx_t_2 = __pyx_v_m;
  __site_setindex_239_10->Target(__site_setindex_239_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"m"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":240
 *     retval["dgelsd_"] = lapack_lite_status__
 *     retval["m"] = m
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda
 */
  __pyx_t_2 = __pyx_v_n;
  __site_setindex_240_10->Target(__site_setindex_240_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":241
 *     retval["m"] = m
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb
 */
  __pyx_t_2 = __pyx_v_nrhs;
  __site_setindex_241_10->Target(__site_setindex_241_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"nrhs"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":242
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["ldb"] = ldb
 *     retval["rcond"] = rcond
 */
  __pyx_t_2 = __pyx_v_lda;
  __site_setindex_242_10->Target(__site_setindex_242_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":243
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb             # <<<<<<<<<<<<<<
 *     retval["rcond"] = rcond
 *     retval["rank"] = rank
 */
  __pyx_t_2 = __pyx_v_ldb;
  __site_setindex_243_10->Target(__site_setindex_243_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldb"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":244
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb
 *     retval["rcond"] = rcond             # <<<<<<<<<<<<<<
 *     retval["rank"] = rank
 *     retval["lwork"] = lwork
 */
  __pyx_t_2 = __pyx_v_rcond;
  __site_setindex_244_10->Target(__site_setindex_244_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"rcond"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":245
 *     retval["ldb"] = ldb
 *     retval["rcond"] = rcond
 *     retval["rank"] = rank             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 */
  __pyx_t_2 = __pyx_v_rank;
  __site_setindex_245_10->Target(__site_setindex_245_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"rank"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":246
 *     retval["rcond"] = rcond
 *     retval["rank"] = rank
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_2 = __pyx_v_lwork;
  __site_setindex_246_10->Target(__site_setindex_246_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":247
 *     retval["rank"] = rank
 *     retval["lwork"] = lwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_2 = __pyx_v_info;
  __site_setindex_247_10->Target(__site_setindex_247_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":248
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":251
 * 
 * 
 * cdef dgesv(int n, int nrhs, object a, int lda, object ipiv,             # <<<<<<<<<<<<<<
 *            object b, int ldb, int info):
 *     cdef int lapack_lite_status__
 */

static  System::Object^ dgesv(int __pyx_v_n, int __pyx_v_nrhs, System::Object^ __pyx_v_a, int __pyx_v_lda, System::Object^ __pyx_v_ipiv, System::Object^ __pyx_v_b, int __pyx_v_ldb, int __pyx_v_info) {
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_retval = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":255
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
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":256
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
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":257
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
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":260
 * 
 *     lapack_lite_status__ = lapack_dgesv(&n,&nrhs,
 *                                         <double *>np.PyArray_DATA(a),&lda,             # <<<<<<<<<<<<<<
 *                                         <int *>np.PyArray_DATA(ipiv),
 *                                         <double *>np.PyArray_DATA(b),&ldb,
 */
  if (__pyx_v_a == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":261
 *     lapack_lite_status__ = lapack_dgesv(&n,&nrhs,
 *                                         <double *>np.PyArray_DATA(a),&lda,
 *                                         <int *>np.PyArray_DATA(ipiv),             # <<<<<<<<<<<<<<
 *                                         <double *>np.PyArray_DATA(b),&ldb,
 *                                         &info)
 */
  if (__pyx_v_ipiv == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_ipiv) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":262
 *                                         <double *>np.PyArray_DATA(a),&lda,
 *                                         <int *>np.PyArray_DATA(ipiv),
 *                                         <double *>np.PyArray_DATA(b),&ldb,             # <<<<<<<<<<<<<<
 *                                         &info)
 * 
 */
  if (__pyx_v_b == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_b) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":263
 *                                         <int *>np.PyArray_DATA(ipiv),
 *                                         <double *>np.PyArray_DATA(b),&ldb,
 *                                         &info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(dgesv_)((&__pyx_v_n), (&__pyx_v_nrhs), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_a))), (&__pyx_v_lda), ((int *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_ipiv))), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_b))), (&__pyx_v_ldb), (&__pyx_v_info));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":265
 *                                         &info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["dgesv_"] = lapack_lite_status__
 *     retval["n"] = n
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":266
 * 
 *     retval = {}
 *     retval["dgesv_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs
 */
  __pyx_t_2 = __pyx_v_lapack_lite_status__;
  __site_setindex_266_10->Target(__site_setindex_266_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"dgesv_"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":267
 *     retval = {}
 *     retval["dgesv_"] = lapack_lite_status__
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda
 */
  __pyx_t_2 = __pyx_v_n;
  __site_setindex_267_10->Target(__site_setindex_267_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":268
 *     retval["dgesv_"] = lapack_lite_status__
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb
 */
  __pyx_t_2 = __pyx_v_nrhs;
  __site_setindex_268_10->Target(__site_setindex_268_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"nrhs"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":269
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["ldb"] = ldb
 *     retval["info"] = info
 */
  __pyx_t_2 = __pyx_v_lda;
  __site_setindex_269_10->Target(__site_setindex_269_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":270
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_2 = __pyx_v_ldb;
  __site_setindex_270_10->Target(__site_setindex_270_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldb"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":271
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_2 = __pyx_v_info;
  __site_setindex_271_10->Target(__site_setindex_271_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":272
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":275
 * 
 * 
 * cdef dgesdd(char jobz, int m, int n, object a, int lda,             # <<<<<<<<<<<<<<
 *             object s, object u, int ldu, object vt, int ldvt,
 *             object work, int lwork, object iwork, int info):
 */

static  System::Object^ dgesdd(char __pyx_v_jobz, int __pyx_v_m, int __pyx_v_n, System::Object^ __pyx_v_a, int __pyx_v_lda, System::Object^ __pyx_v_s, System::Object^ __pyx_v_u, int __pyx_v_ldu, System::Object^ __pyx_v_vt, int __pyx_v_ldvt, System::Object^ __pyx_v_work, int __pyx_v_lwork, System::Object^ __pyx_v_iwork, int __pyx_v_info) {
  int __pyx_v_lapack_lite_status__;
  long __pyx_v_work0;
  int __pyx_v_mn;
  int __pyx_v_mx;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  int __pyx_t_2;
  int __pyx_t_3;
  int __pyx_t_4;
  int __pyx_t_5;
  int __pyx_t_6;
  long __pyx_t_7;
  long __pyx_t_8;
  long __pyx_t_9;
  long __pyx_t_10;
  System::Object^ __pyx_t_11 = nullptr;
  __pyx_v_retval = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":282
 *     cdef int mn, mx
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(u,np.NPY_DOUBLE,"u","np.NPY_DOUBLE","dgesdd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_a, NPY_DOUBLE, __pyx_k__a, __pyx_k_1, __pyx_k__dgesdd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":283
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","dgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(u,np.NPY_DOUBLE,"u","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(vt,np.NPY_DOUBLE,"vt","np.NPY_DOUBLE","dgesdd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_s, NPY_DOUBLE, __pyx_k__s, __pyx_k_1, __pyx_k__dgesdd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":284
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(u,np.NPY_DOUBLE,"u","np.NPY_DOUBLE","dgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(vt,np.NPY_DOUBLE,"vt","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgesdd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_u, NPY_DOUBLE, __pyx_k__u, __pyx_k_1, __pyx_k__dgesdd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":285
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(u,np.NPY_DOUBLE,"u","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(vt,np.NPY_DOUBLE,"vt","np.NPY_DOUBLE","dgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","dgesdd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_vt, NPY_DOUBLE, __pyx_k__vt, __pyx_k_1, __pyx_k__dgesdd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":286
 *     if not check_object(u,np.NPY_DOUBLE,"u","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(vt,np.NPY_DOUBLE,"vt","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","dgesdd"): return None
 * 
 */
  __pyx_t_1 = (!check_object(__pyx_v_work, NPY_DOUBLE, __pyx_k__work, __pyx_k_1, __pyx_k__dgesdd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":287
 *     if not check_object(vt,np.NPY_DOUBLE,"vt","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(work,np.NPY_DOUBLE,"work","np.NPY_DOUBLE","dgesdd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","dgesdd"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_dgesdd(&jobz,&m,&n,
 */
  __pyx_t_1 = (!check_object(__pyx_v_iwork, NPY_INT, __pyx_k__iwork, __pyx_k_2, __pyx_k__dgesdd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L8;
  }
  __pyx_L8:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":290
 * 
 *     lapack_lite_status__ = lapack_dgesdd(&jobz,&m,&n,
 *                                          <double *>np.PyArray_DATA(a),&lda,             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(s),
 *                                          <double *>np.PyArray_DATA(u),&ldu,
 */
  if (__pyx_v_a == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":291
 *     lapack_lite_status__ = lapack_dgesdd(&jobz,&m,&n,
 *                                          <double *>np.PyArray_DATA(a),&lda,
 *                                          <double *>np.PyArray_DATA(s),             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(u),&ldu,
 *                                          <double *>np.PyArray_DATA(vt),&ldvt,
 */
  if (__pyx_v_s == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_s) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":292
 *                                          <double *>np.PyArray_DATA(a),&lda,
 *                                          <double *>np.PyArray_DATA(s),
 *                                          <double *>np.PyArray_DATA(u),&ldu,             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(vt),&ldvt,
 *                                          <double *>np.PyArray_DATA(work),&lwork,
 */
  if (__pyx_v_u == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_u) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":293
 *                                          <double *>np.PyArray_DATA(s),
 *                                          <double *>np.PyArray_DATA(u),&ldu,
 *                                          <double *>np.PyArray_DATA(vt),&ldvt,             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(work),&lwork,
 *                                          <int *>np.PyArray_DATA(iwork),&info)
 */
  if (__pyx_v_vt == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_vt) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":294
 *                                          <double *>np.PyArray_DATA(u),&ldu,
 *                                          <double *>np.PyArray_DATA(vt),&ldvt,
 *                                          <double *>np.PyArray_DATA(work),&lwork,             # <<<<<<<<<<<<<<
 *                                          <int *>np.PyArray_DATA(iwork),&info)
 * 
 */
  if (__pyx_v_work == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":295
 *                                          <double *>np.PyArray_DATA(vt),&ldvt,
 *                                          <double *>np.PyArray_DATA(work),&lwork,
 *                                          <int *>np.PyArray_DATA(iwork),&info)             # <<<<<<<<<<<<<<
 * 
 *     if info == 0 and lwork == -1:
 */
  if (__pyx_v_iwork == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_iwork) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(dgesdd_)((&__pyx_v_jobz), (&__pyx_v_m), (&__pyx_v_n), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_a))), (&__pyx_v_lda), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_s))), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_u))), (&__pyx_v_ldu), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_vt))), (&__pyx_v_ldvt), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_work))), (&__pyx_v_lwork), ((int *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_iwork))), (&__pyx_v_info));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":297
 *                                          <int *>np.PyArray_DATA(iwork),&info)
 * 
 *     if info == 0 and lwork == -1:             # <<<<<<<<<<<<<<
 *         # We need to check the result because
 *         # sometimes the "optimal" value is actually
 */
  __pyx_t_1 = (__pyx_v_info == 0);
  if (__pyx_t_1) {
    __pyx_t_2 = (__pyx_v_lwork == -1);
    __pyx_t_3 = __pyx_t_2;
  } else {
    __pyx_t_3 = __pyx_t_1;
  }
  if (__pyx_t_3) {

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":302
 *         # too small.
 *         # Change it to the maximum of the minimum and the optimal.
 *         work0 = <long>(<double *>np.PyArray_DATA(work))[0]             # <<<<<<<<<<<<<<
 *         mn = min(m,n)
 *         mx = max(m,n)
 */
    if (__pyx_v_work == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr) {
      throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
    }
    __pyx_v_work0 = ((long)(((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_work)))[0]));

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":303
 *         # Change it to the maximum of the minimum and the optimal.
 *         work0 = <long>(<double *>np.PyArray_DATA(work))[0]
 *         mn = min(m,n)             # <<<<<<<<<<<<<<
 *         mx = max(m,n)
 * 
 */
    __pyx_t_4 = __pyx_v_n;
    __pyx_t_5 = __pyx_v_m;
    if ((__pyx_t_4 < __pyx_t_5)) {
      __pyx_t_6 = __pyx_t_4;
    } else {
      __pyx_t_6 = __pyx_t_5;
    }
    __pyx_v_mn = __pyx_t_6;

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":304
 *         work0 = <long>(<double *>np.PyArray_DATA(work))[0]
 *         mn = min(m,n)
 *         mx = max(m,n)             # <<<<<<<<<<<<<<
 * 
 *         if jobz == 'N':
 */
    __pyx_t_6 = __pyx_v_n;
    __pyx_t_4 = __pyx_v_m;
    if ((__pyx_t_6 > __pyx_t_4)) {
      __pyx_t_5 = __pyx_t_6;
    } else {
      __pyx_t_5 = __pyx_t_4;
    }
    __pyx_v_mx = __pyx_t_5;

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":306
 *         mx = max(m,n)
 * 
 *         if jobz == 'N':             # <<<<<<<<<<<<<<
 *             work0 = max(work0,3*mn + max(mx,6*mn)+500)
 *         elif jobz == 'O':
 */
    __pyx_t_3 = (__pyx_v_jobz == 'N');
    if (__pyx_t_3) {

      /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":307
 * 
 *         if jobz == 'N':
 *             work0 = max(work0,3*mn + max(mx,6*mn)+500)             # <<<<<<<<<<<<<<
 *         elif jobz == 'O':
 *             work0 = max(work0,3*mn*mn + max(mx,5*mn*mn+4*mn+500))
 */
      __pyx_t_7 = (6 * __pyx_v_mn);
      __pyx_t_5 = __pyx_v_mx;
      if ((__pyx_t_7 > __pyx_t_5)) {
        __pyx_t_8 = __pyx_t_7;
      } else {
        __pyx_t_8 = __pyx_t_5;
      }
      __pyx_t_7 = (((3 * __pyx_v_mn) + __pyx_t_8) + 500);
      __pyx_t_9 = __pyx_v_work0;
      if ((__pyx_t_7 > __pyx_t_9)) {
        __pyx_t_10 = __pyx_t_7;
      } else {
        __pyx_t_10 = __pyx_t_9;
      }
      __pyx_v_work0 = __pyx_t_10;
      goto __pyx_L10;
    }

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":308
 *         if jobz == 'N':
 *             work0 = max(work0,3*mn + max(mx,6*mn)+500)
 *         elif jobz == 'O':             # <<<<<<<<<<<<<<
 *             work0 = max(work0,3*mn*mn + max(mx,5*mn*mn+4*mn+500))
 *         elif jobz == 'S' or jobz == 'A':
 */
    __pyx_t_3 = (__pyx_v_jobz == 'O');
    if (__pyx_t_3) {

      /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":309
 *             work0 = max(work0,3*mn + max(mx,6*mn)+500)
 *         elif jobz == 'O':
 *             work0 = max(work0,3*mn*mn + max(mx,5*mn*mn+4*mn+500))             # <<<<<<<<<<<<<<
 *         elif jobz == 'S' or jobz == 'A':
 *             work0 = max(work0,3*mn*mn + max(mx,4*mn*(mn+1))+500)
 */
      __pyx_t_10 = ((((5 * __pyx_v_mn) * __pyx_v_mn) + (4 * __pyx_v_mn)) + 500);
      __pyx_t_5 = __pyx_v_mx;
      if ((__pyx_t_10 > __pyx_t_5)) {
        __pyx_t_8 = __pyx_t_10;
      } else {
        __pyx_t_8 = __pyx_t_5;
      }
      __pyx_t_10 = (((3 * __pyx_v_mn) * __pyx_v_mn) + __pyx_t_8);
      __pyx_t_7 = __pyx_v_work0;
      if ((__pyx_t_10 > __pyx_t_7)) {
        __pyx_t_9 = __pyx_t_10;
      } else {
        __pyx_t_9 = __pyx_t_7;
      }
      __pyx_v_work0 = __pyx_t_9;
      goto __pyx_L10;
    }

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":310
 *         elif jobz == 'O':
 *             work0 = max(work0,3*mn*mn + max(mx,5*mn*mn+4*mn+500))
 *         elif jobz == 'S' or jobz == 'A':             # <<<<<<<<<<<<<<
 *             work0 = max(work0,3*mn*mn + max(mx,4*mn*(mn+1))+500)
 * 
 */
    __pyx_t_3 = (__pyx_v_jobz == 'S');
    if (!__pyx_t_3) {
      __pyx_t_1 = (__pyx_v_jobz == 'A');
      __pyx_t_2 = __pyx_t_1;
    } else {
      __pyx_t_2 = __pyx_t_3;
    }
    if (__pyx_t_2) {

      /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":311
 *             work0 = max(work0,3*mn*mn + max(mx,5*mn*mn+4*mn+500))
 *         elif jobz == 'S' or jobz == 'A':
 *             work0 = max(work0,3*mn*mn + max(mx,4*mn*(mn+1))+500)             # <<<<<<<<<<<<<<
 * 
 *         (<double *>np.PyArray_DATA(work))[0] = <double>work0
 */
      __pyx_t_9 = ((4 * __pyx_v_mn) * (__pyx_v_mn + 1));
      __pyx_t_5 = __pyx_v_mx;
      if ((__pyx_t_9 > __pyx_t_5)) {
        __pyx_t_8 = __pyx_t_9;
      } else {
        __pyx_t_8 = __pyx_t_5;
      }
      __pyx_t_9 = ((((3 * __pyx_v_mn) * __pyx_v_mn) + __pyx_t_8) + 500);
      __pyx_t_10 = __pyx_v_work0;
      if ((__pyx_t_9 > __pyx_t_10)) {
        __pyx_t_7 = __pyx_t_9;
      } else {
        __pyx_t_7 = __pyx_t_10;
      }
      __pyx_v_work0 = __pyx_t_7;
      goto __pyx_L10;
    }
    __pyx_L10:;

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":313
 *             work0 = max(work0,3*mn*mn + max(mx,4*mn*(mn+1))+500)
 * 
 *         (<double *>np.PyArray_DATA(work))[0] = <double>work0             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
    if (__pyx_v_work == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr) {
      throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
    }
    (((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_work)))[0]) = ((double)__pyx_v_work0);
    goto __pyx_L9;
  }
  __pyx_L9:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":315
 *         (<double *>np.PyArray_DATA(work))[0] = <double>work0
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["dgesdd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz
 */
  __pyx_t_11 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_11;
  __pyx_t_11 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":316
 * 
 *     retval = {}
 *     retval["dgesdd_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["jobz"] = jobz
 *     retval["m"] = m
 */
  __pyx_t_11 = __pyx_v_lapack_lite_status__;
  __site_setindex_316_10->Target(__site_setindex_316_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"dgesdd_"), __pyx_t_11);
  __pyx_t_11 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":317
 *     retval = {}
 *     retval["dgesdd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz             # <<<<<<<<<<<<<<
 *     retval["m"] = m
 *     retval["n"] = n
 */
  __pyx_t_11 = __pyx_v_jobz;
  __site_setindex_317_10->Target(__site_setindex_317_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"jobz"), __pyx_t_11);
  __pyx_t_11 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":318
 *     retval["dgesdd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz
 *     retval["m"] = m             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_11 = __pyx_v_m;
  __site_setindex_318_10->Target(__site_setindex_318_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"m"), __pyx_t_11);
  __pyx_t_11 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":319
 *     retval["jobz"] = jobz
 *     retval["m"] = m
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["ldu"] = ldu
 */
  __pyx_t_11 = __pyx_v_n;
  __site_setindex_319_10->Target(__site_setindex_319_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_11);
  __pyx_t_11 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":320
 *     retval["m"] = m
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["ldu"] = ldu
 *     retval["ldvt"] = ldvt
 */
  __pyx_t_11 = __pyx_v_lda;
  __site_setindex_320_10->Target(__site_setindex_320_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_11);
  __pyx_t_11 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":321
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["ldu"] = ldu             # <<<<<<<<<<<<<<
 *     retval["ldvt"] = ldvt
 *     retval["lwork"] = lwork
 */
  __pyx_t_11 = __pyx_v_ldu;
  __site_setindex_321_10->Target(__site_setindex_321_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldu"), __pyx_t_11);
  __pyx_t_11 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":322
 *     retval["lda"] = lda
 *     retval["ldu"] = ldu
 *     retval["ldvt"] = ldvt             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 */
  __pyx_t_11 = __pyx_v_ldvt;
  __site_setindex_322_10->Target(__site_setindex_322_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldvt"), __pyx_t_11);
  __pyx_t_11 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":323
 *     retval["ldu"] = ldu
 *     retval["ldvt"] = ldvt
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_11 = __pyx_v_lwork;
  __site_setindex_323_10->Target(__site_setindex_323_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_11);
  __pyx_t_11 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":324
 *     retval["ldvt"] = ldvt
 *     retval["lwork"] = lwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_11 = __pyx_v_info;
  __site_setindex_324_10->Target(__site_setindex_324_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_11);
  __pyx_t_11 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":325
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":328
 * 
 * 
 * cdef dgetrf(int m, int n, object a, int lda, object ipiv, int info):             # <<<<<<<<<<<<<<
 *     cdef int lapack_lite_status__
 * 
 */

static  System::Object^ dgetrf(int __pyx_v_m, int __pyx_v_n, System::Object^ __pyx_v_a, int __pyx_v_lda, System::Object^ __pyx_v_ipiv, int __pyx_v_info) {
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_retval = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":331
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
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":332
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
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":334
 *     if not check_object(ipiv,np.NPY_INT,"ipiv","np.NPY_INT","dgetrf"): return None
 * 
 *     lapack_lite_status__ = lapack_dgetrf(&m,&n,<double *>np.PyArray_DATA(a),&lda,             # <<<<<<<<<<<<<<
 *                                          <int *>np.PyArray_DATA(ipiv),&info)
 * 
 */
  if (__pyx_v_a == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":335
 * 
 *     lapack_lite_status__ = lapack_dgetrf(&m,&n,<double *>np.PyArray_DATA(a),&lda,
 *                                          <int *>np.PyArray_DATA(ipiv),&info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  if (__pyx_v_ipiv == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_ipiv) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(dgetrf_)((&__pyx_v_m), (&__pyx_v_n), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_a))), (&__pyx_v_lda), ((int *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_ipiv))), (&__pyx_v_info));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":337
 *                                          <int *>np.PyArray_DATA(ipiv),&info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["dgetrf_"] = lapack_lite_status__
 *     retval["m"] = m
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":338
 * 
 *     retval = {}
 *     retval["dgetrf_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["m"] = m
 *     retval["n"] = n
 */
  __pyx_t_2 = __pyx_v_lapack_lite_status__;
  __site_setindex_338_10->Target(__site_setindex_338_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"dgetrf_"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":339
 *     retval = {}
 *     retval["dgetrf_"] = lapack_lite_status__
 *     retval["m"] = m             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_2 = __pyx_v_m;
  __site_setindex_339_10->Target(__site_setindex_339_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"m"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":340
 *     retval["dgetrf_"] = lapack_lite_status__
 *     retval["m"] = m
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["info"] = info
 */
  __pyx_t_2 = __pyx_v_n;
  __site_setindex_340_10->Target(__site_setindex_340_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":341
 *     retval["m"] = m
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_2 = __pyx_v_lda;
  __site_setindex_341_10->Target(__site_setindex_341_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":342
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_2 = __pyx_v_info;
  __site_setindex_342_10->Target(__site_setindex_342_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":343
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":346
 * 
 * 
 * cdef dpotrf(char uplo, int n, object a, int lda, int info):             # <<<<<<<<<<<<<<
 *     cdef int lapack_lite_status__
 * 
 */

static  System::Object^ dpotrf(char __pyx_v_uplo, int __pyx_v_n, System::Object^ __pyx_v_a, int __pyx_v_lda, int __pyx_v_info) {
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_retval = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":349
 *     cdef int lapack_lite_status__
 * 
 *     if not check_object(a,np.NPY_DOUBLE,"a","np.NPY_DOUBLE","dpotrf"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_dpotrf(&uplo,&n,
 */
  __pyx_t_1 = (!check_object(__pyx_v_a, NPY_DOUBLE, __pyx_k__a, __pyx_k_1, __pyx_k__dpotrf));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":352
 * 
 *     lapack_lite_status__ = lapack_dpotrf(&uplo,&n,
 *                                          <double *>np.PyArray_DATA(a),&lda,             # <<<<<<<<<<<<<<
 *                                          &info)
 * 
 */
  if (__pyx_v_a == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":353
 *     lapack_lite_status__ = lapack_dpotrf(&uplo,&n,
 *                                          <double *>np.PyArray_DATA(a),&lda,
 *                                          &info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(dpotrf_)((&__pyx_v_uplo), (&__pyx_v_n), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_a))), (&__pyx_v_lda), (&__pyx_v_info));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":355
 *                                          &info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["dpotrf_"] = lapack_lite_status__
 *     retval["n"] = n
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":356
 * 
 *     retval = {}
 *     retval["dpotrf_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_2 = __pyx_v_lapack_lite_status__;
  __site_setindex_356_10->Target(__site_setindex_356_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"dpotrf_"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":357
 *     retval = {}
 *     retval["dpotrf_"] = lapack_lite_status__
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["info"] = info
 */
  __pyx_t_2 = __pyx_v_n;
  __site_setindex_357_10->Target(__site_setindex_357_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":358
 *     retval["dpotrf_"] = lapack_lite_status__
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_2 = __pyx_v_lda;
  __site_setindex_358_10->Target(__site_setindex_358_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":359
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_2 = __pyx_v_info;
  __site_setindex_359_10->Target(__site_setindex_359_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":360
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":363
 * 
 * 
 * cdef dgeqrf(int m, int n, object a, int lda, object tau, object work, int lwork, int info):             # <<<<<<<<<<<<<<
 *     cdef int  lapack_lite_status__
 * 
 */

static  System::Object^ dgeqrf(int __pyx_v_m, int __pyx_v_n, System::Object^ __pyx_v_a, int __pyx_v_lda, System::Object^ __pyx_v_tau, System::Object^ __pyx_v_work, int __pyx_v_lwork, int __pyx_v_info) {
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_retval = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":367
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
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":368
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
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":369
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
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":372
 * 
 *     lapack_lite_status__ = lapack_dgeqrf(&m, &n,
 *                                          <double *>np.PyArray_DATA(a), &lda,             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(tau),
 *                                          <double *>np.PyArray_DATA(work), &lwork,
 */
  if (__pyx_v_a == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":373
 *     lapack_lite_status__ = lapack_dgeqrf(&m, &n,
 *                                          <double *>np.PyArray_DATA(a), &lda,
 *                                          <double *>np.PyArray_DATA(tau),             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(work), &lwork,
 *                                          &info)
 */
  if (__pyx_v_tau == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_tau) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":374
 *                                          <double *>np.PyArray_DATA(a), &lda,
 *                                          <double *>np.PyArray_DATA(tau),
 *                                          <double *>np.PyArray_DATA(work), &lwork,             # <<<<<<<<<<<<<<
 *                                          &info)
 * 
 */
  if (__pyx_v_work == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":375
 *                                          <double *>np.PyArray_DATA(tau),
 *                                          <double *>np.PyArray_DATA(work), &lwork,
 *                                          &info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(dgeqrf_)((&__pyx_v_m), (&__pyx_v_n), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_a))), (&__pyx_v_lda), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_tau))), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_work))), (&__pyx_v_lwork), (&__pyx_v_info));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":377
 *                                          &info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["dgeqrf_"] = lapack_lite_status__
 *     retval["m"] = m
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":378
 * 
 *     retval = {}
 *     retval["dgeqrf_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["m"] = m
 *     retval["n"] = n
 */
  __pyx_t_2 = __pyx_v_lapack_lite_status__;
  __site_setindex_378_10->Target(__site_setindex_378_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"dgeqrf_"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":379
 *     retval = {}
 *     retval["dgeqrf_"] = lapack_lite_status__
 *     retval["m"] = m             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_2 = __pyx_v_m;
  __site_setindex_379_10->Target(__site_setindex_379_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"m"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":380
 *     retval["dgeqrf_"] = lapack_lite_status__
 *     retval["m"] = m
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork
 */
  __pyx_t_2 = __pyx_v_n;
  __site_setindex_380_10->Target(__site_setindex_380_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":381
 *     retval["m"] = m
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 */
  __pyx_t_2 = __pyx_v_lda;
  __site_setindex_381_10->Target(__site_setindex_381_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":382
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_2 = __pyx_v_lwork;
  __site_setindex_382_10->Target(__site_setindex_382_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":383
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_2 = __pyx_v_info;
  __site_setindex_383_10->Target(__site_setindex_383_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":384
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":387
 * 
 * 
 * cdef dorgqr(int m, int n, int k, object a, int lda, object tau, object work, int lwork, int info):             # <<<<<<<<<<<<<<
 *     cdef int  lapack_lite_status__
 * 
 */

static  System::Object^ dorgqr(int __pyx_v_m, int __pyx_v_n, int __pyx_v_k, System::Object^ __pyx_v_a, int __pyx_v_lda, System::Object^ __pyx_v_tau, System::Object^ __pyx_v_work, int __pyx_v_lwork, int __pyx_v_info) {
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_retval = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":390
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
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":391
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
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":392
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
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":395
 * 
 *     lapack_lite_status__ = lapack_dorgqr(&m, &n, &k,
 *                                          <double *>np.PyArray_DATA(a), &lda,             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(tau),
 *                                          <double *>np.PyArray_DATA(work), &lwork,
 */
  if (__pyx_v_a == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":396
 *     lapack_lite_status__ = lapack_dorgqr(&m, &n, &k,
 *                                          <double *>np.PyArray_DATA(a), &lda,
 *                                          <double *>np.PyArray_DATA(tau),             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(work), &lwork,
 *                                          &info)
 */
  if (__pyx_v_tau == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_tau) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":397
 *                                          <double *>np.PyArray_DATA(a), &lda,
 *                                          <double *>np.PyArray_DATA(tau),
 *                                          <double *>np.PyArray_DATA(work), &lwork,             # <<<<<<<<<<<<<<
 *                                          &info)
 * 
 */
  if (__pyx_v_work == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":398
 *                                          <double *>np.PyArray_DATA(tau),
 *                                          <double *>np.PyArray_DATA(work), &lwork,
 *                                          &info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(dorgqr_)((&__pyx_v_m), (&__pyx_v_n), (&__pyx_v_k), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_a))), (&__pyx_v_lda), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_tau))), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_work))), (&__pyx_v_lwork), (&__pyx_v_info));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":400
 *                                          &info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["dorgqr_"] = lapack_lite_status__
 *     retval["info"] = info
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":401
 * 
 *     retval = {}
 *     retval["dorgqr_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_2 = __pyx_v_lapack_lite_status__;
  __site_setindex_401_10->Target(__site_setindex_401_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"dorgqr_"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":402
 *     retval = {}
 *     retval["dorgqr_"] = lapack_lite_status__
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_2 = __pyx_v_info;
  __site_setindex_402_10->Target(__site_setindex_402_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":403
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":406
 * 
 * 
 * cdef zgeev(char jobvl, char jobvr, int n, object a, int lda,             # <<<<<<<<<<<<<<
 *            object w, object vl, int ldvl, object vr, int ldvr,
 *            object work, int lwork, object rwork, int info):
 */

static  System::Object^ zgeev(char __pyx_v_jobvl, char __pyx_v_jobvr, int __pyx_v_n, System::Object^ __pyx_v_a, int __pyx_v_lda, System::Object^ __pyx_v_w, System::Object^ __pyx_v_vl, int __pyx_v_ldvl, System::Object^ __pyx_v_vr, int __pyx_v_ldvr, System::Object^ __pyx_v_work, int __pyx_v_lwork, System::Object^ __pyx_v_rwork, int __pyx_v_info) {
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_retval = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":411
 *     cdef int lapack_lite_status__
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(w,np.NPY_CDOUBLE,"w","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(vl,np.NPY_CDOUBLE,"vl","np.NPY_CDOUBLE","zgeev"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_a, NPY_CDOUBLE, __pyx_k__a, __pyx_k_3, __pyx_k__zgeev));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":412
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(w,np.NPY_CDOUBLE,"w","np.NPY_CDOUBLE","zgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(vl,np.NPY_CDOUBLE,"vl","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(vr,np.NPY_CDOUBLE,"vr","np.NPY_CDOUBLE","zgeev"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_w, NPY_CDOUBLE, __pyx_k__w, __pyx_k_3, __pyx_k__zgeev));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":413
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(w,np.NPY_CDOUBLE,"w","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(vl,np.NPY_CDOUBLE,"vl","np.NPY_CDOUBLE","zgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(vr,np.NPY_CDOUBLE,"vr","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgeev"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_vl, NPY_CDOUBLE, __pyx_k__vl, __pyx_k_3, __pyx_k__zgeev));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":414
 *     if not check_object(w,np.NPY_CDOUBLE,"w","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(vl,np.NPY_CDOUBLE,"vl","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(vr,np.NPY_CDOUBLE,"vr","np.NPY_CDOUBLE","zgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgeev"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_vr, NPY_CDOUBLE, __pyx_k__vr, __pyx_k_3, __pyx_k__zgeev));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":415
 *     if not check_object(vl,np.NPY_CDOUBLE,"vl","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(vr,np.NPY_CDOUBLE,"vr","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgeev"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgeev"): return None
 * 
 */
  __pyx_t_1 = (!check_object(__pyx_v_work, NPY_CDOUBLE, __pyx_k__work, __pyx_k_3, __pyx_k__zgeev));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":416
 *     if not check_object(vr,np.NPY_CDOUBLE,"vr","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgeev"): return None
 *     if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgeev"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_zgeev(&jobvl,&jobvr,&n,
 */
  __pyx_t_1 = (!check_object(__pyx_v_rwork, NPY_DOUBLE, __pyx_k__rwork, __pyx_k_1, __pyx_k__zgeev));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L8;
  }
  __pyx_L8:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":419
 * 
 *     lapack_lite_status__ = lapack_zgeev(&jobvl,&jobvr,&n,
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,             # <<<<<<<<<<<<<<
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(w),
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(vl),&ldvl,
 */
  if (__pyx_v_a == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":420
 *     lapack_lite_status__ = lapack_zgeev(&jobvl,&jobvr,&n,
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(w),             # <<<<<<<<<<<<<<
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(vl),&ldvl,
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(vr),&ldvr,
 */
  if (__pyx_v_w == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_w) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":421
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(w),
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(vl),&ldvl,             # <<<<<<<<<<<<<<
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(vr),&ldvr,
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 */
  if (__pyx_v_vl == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_vl) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":422
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(w),
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(vl),&ldvl,
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(vr),&ldvr,             # <<<<<<<<<<<<<<
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 *                                         <double *>np.PyArray_DATA(rwork),&info)
 */
  if (__pyx_v_vr == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_vr) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":423
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(vl),&ldvl,
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(vr),&ldvr,
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,             # <<<<<<<<<<<<<<
 *                                         <double *>np.PyArray_DATA(rwork),&info)
 * 
 */
  if (__pyx_v_work == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":424
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(vr),&ldvr,
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 *                                         <double *>np.PyArray_DATA(rwork),&info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  if (__pyx_v_rwork == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_rwork) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(zgeev_)((&__pyx_v_jobvl), (&__pyx_v_jobvr), (&__pyx_v_n), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_a))), (&__pyx_v_lda), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_w))), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_vl))), (&__pyx_v_ldvl), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_vr))), (&__pyx_v_ldvr), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_work))), (&__pyx_v_lwork), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_rwork))), (&__pyx_v_info));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":426
 *                                         <double *>np.PyArray_DATA(rwork),&info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["zgeev_"] = lapack_lite_status__
 *     retval["jobvl"] = jobvl
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":427
 * 
 *     retval = {}
 *     retval["zgeev_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["jobvl"] = jobvl
 *     retval["jobvr"] = jobvr
 */
  __pyx_t_2 = __pyx_v_lapack_lite_status__;
  __site_setindex_427_10->Target(__site_setindex_427_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"zgeev_"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":428
 *     retval = {}
 *     retval["zgeev_"] = lapack_lite_status__
 *     retval["jobvl"] = jobvl             # <<<<<<<<<<<<<<
 *     retval["jobvr"] = jobvr
 *     retval["n"] = n
 */
  __pyx_t_2 = __pyx_v_jobvl;
  __site_setindex_428_10->Target(__site_setindex_428_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"jobvl"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":429
 *     retval["zgeev_"] = lapack_lite_status__
 *     retval["jobvl"] = jobvl
 *     retval["jobvr"] = jobvr             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_2 = __pyx_v_jobvr;
  __site_setindex_429_10->Target(__site_setindex_429_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"jobvr"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":430
 *     retval["jobvl"] = jobvl
 *     retval["jobvr"] = jobvr
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["ldvl"] = ldvl
 */
  __pyx_t_2 = __pyx_v_n;
  __site_setindex_430_10->Target(__site_setindex_430_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":431
 *     retval["jobvr"] = jobvr
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["ldvl"] = ldvl
 *     retval["ldvr"] = ldvr
 */
  __pyx_t_2 = __pyx_v_lda;
  __site_setindex_431_10->Target(__site_setindex_431_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":432
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["ldvl"] = ldvl             # <<<<<<<<<<<<<<
 *     retval["ldvr"] = ldvr
 *     retval["lwork"] = lwork
 */
  __pyx_t_2 = __pyx_v_ldvl;
  __site_setindex_432_10->Target(__site_setindex_432_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldvl"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":433
 *     retval["lda"] = lda
 *     retval["ldvl"] = ldvl
 *     retval["ldvr"] = ldvr             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 */
  __pyx_t_2 = __pyx_v_ldvr;
  __site_setindex_433_10->Target(__site_setindex_433_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldvr"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":434
 *     retval["ldvl"] = ldvl
 *     retval["ldvr"] = ldvr
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_2 = __pyx_v_lwork;
  __site_setindex_434_10->Target(__site_setindex_434_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":435
 *     retval["ldvr"] = ldvr
 *     retval["lwork"] = lwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_2 = __pyx_v_info;
  __site_setindex_435_10->Target(__site_setindex_435_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":436
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":439
 * 
 * 
 * cdef zgelsd(int m, int n, int nrhs, object a, int lda,             # <<<<<<<<<<<<<<
 *             object b, int ldb, object s, double rcond,
 *             int rank, object work, int lwork,
 */

static  System::Object^ zgelsd(int __pyx_v_m, int __pyx_v_n, int __pyx_v_nrhs, System::Object^ __pyx_v_a, int __pyx_v_lda, System::Object^ __pyx_v_b, int __pyx_v_ldb, System::Object^ __pyx_v_s, double __pyx_v_rcond, int __pyx_v_rank, System::Object^ __pyx_v_work, int __pyx_v_lwork, System::Object^ __pyx_v_rwork, System::Object^ __pyx_v_iwork, int __pyx_v_info) {
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_retval = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":445
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
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":446
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
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":447
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
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":448
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
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":449
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
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":450
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
    goto __pyx_L8;
  }
  __pyx_L8:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":453
 * 
 *     lapack_lite_status__ = lapack_zgelsd(&m,&n,&nrhs,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,             # <<<<<<<<<<<<<<
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(b),&ldb,
 *                                          <double *>np.PyArray_DATA(s),&rcond,&rank,
 */
  if (__pyx_v_a == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":454
 *     lapack_lite_status__ = lapack_zgelsd(&m,&n,&nrhs,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(b),&ldb,             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(s),&rcond,&rank,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 */
  if (__pyx_v_b == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_b) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":455
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(b),&ldb,
 *                                          <double *>np.PyArray_DATA(s),&rcond,&rank,             # <<<<<<<<<<<<<<
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 *                                          <double *>np.PyArray_DATA(rwork),
 */
  if (__pyx_v_s == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_s) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":456
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(b),&ldb,
 *                                          <double *>np.PyArray_DATA(s),&rcond,&rank,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(rwork),
 *                                          <int *>np.PyArray_DATA(iwork),&info)
 */
  if (__pyx_v_work == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":457
 *                                          <double *>np.PyArray_DATA(s),&rcond,&rank,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 *                                          <double *>np.PyArray_DATA(rwork),             # <<<<<<<<<<<<<<
 *                                          <int *>np.PyArray_DATA(iwork),&info)
 * 
 */
  if (__pyx_v_rwork == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_rwork) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":458
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 *                                          <double *>np.PyArray_DATA(rwork),
 *                                          <int *>np.PyArray_DATA(iwork),&info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  if (__pyx_v_iwork == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_iwork) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(zgelsd_)((&__pyx_v_m), (&__pyx_v_n), (&__pyx_v_nrhs), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_a))), (&__pyx_v_lda), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_b))), (&__pyx_v_ldb), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_s))), (&__pyx_v_rcond), (&__pyx_v_rank), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_work))), (&__pyx_v_lwork), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_rwork))), ((int *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_iwork))), (&__pyx_v_info));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":460
 *                                          <int *>np.PyArray_DATA(iwork),&info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["zgelsd_"] = lapack_lite_status__
 *     retval["m"] = m
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":461
 * 
 *     retval = {}
 *     retval["zgelsd_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["m"] = m
 *     retval["n"] = n
 */
  __pyx_t_2 = __pyx_v_lapack_lite_status__;
  __site_setindex_461_10->Target(__site_setindex_461_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"zgelsd_"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":462
 *     retval = {}
 *     retval["zgelsd_"] = lapack_lite_status__
 *     retval["m"] = m             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs
 */
  __pyx_t_2 = __pyx_v_m;
  __site_setindex_462_10->Target(__site_setindex_462_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"m"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":463
 *     retval["zgelsd_"] = lapack_lite_status__
 *     retval["m"] = m
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda
 */
  __pyx_t_2 = __pyx_v_n;
  __site_setindex_463_10->Target(__site_setindex_463_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":464
 *     retval["m"] = m
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb
 */
  __pyx_t_2 = __pyx_v_nrhs;
  __site_setindex_464_10->Target(__site_setindex_464_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"nrhs"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":465
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["ldb"] = ldb
 *     retval["rank"] = rank
 */
  __pyx_t_2 = __pyx_v_lda;
  __site_setindex_465_10->Target(__site_setindex_465_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":466
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb             # <<<<<<<<<<<<<<
 *     retval["rank"] = rank
 *     retval["lwork"] = lwork
 */
  __pyx_t_2 = __pyx_v_ldb;
  __site_setindex_466_10->Target(__site_setindex_466_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldb"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":467
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb
 *     retval["rank"] = rank             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 */
  __pyx_t_2 = __pyx_v_rank;
  __site_setindex_467_10->Target(__site_setindex_467_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"rank"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":468
 *     retval["ldb"] = ldb
 *     retval["rank"] = rank
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_2 = __pyx_v_lwork;
  __site_setindex_468_10->Target(__site_setindex_468_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":469
 *     retval["rank"] = rank
 *     retval["lwork"] = lwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_2 = __pyx_v_info;
  __site_setindex_469_10->Target(__site_setindex_469_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":470
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":473
 * 
 * 
 * cdef zgesv(int n, int nrhs, object a, int lda, object ipiv, object b, int ldb, int info):             # <<<<<<<<<<<<<<
 *     cdef int lapack_lite_status__
 * 
 */

static  System::Object^ zgesv(int __pyx_v_n, int __pyx_v_nrhs, System::Object^ __pyx_v_a, int __pyx_v_lda, System::Object^ __pyx_v_ipiv, System::Object^ __pyx_v_b, int __pyx_v_ldb, int __pyx_v_info) {
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_retval = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":476
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
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":477
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
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":478
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
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":481
 * 
 *     lapack_lite_status__ = lapack_zgesv(&n,&nrhs,
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,             # <<<<<<<<<<<<<<
 *                                         <int *>np.PyArray_DATA(ipiv),
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(b),&ldb,
 */
  if (__pyx_v_a == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":482
 *     lapack_lite_status__ = lapack_zgesv(&n,&nrhs,
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
 *                                         <int *>np.PyArray_DATA(ipiv),             # <<<<<<<<<<<<<<
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(b),&ldb,
 *                                         &info)
 */
  if (__pyx_v_ipiv == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_ipiv) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":483
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
 *                                         <int *>np.PyArray_DATA(ipiv),
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(b),&ldb,             # <<<<<<<<<<<<<<
 *                                         &info)
 * 
 */
  if (__pyx_v_b == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_b) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":484
 *                                         <int *>np.PyArray_DATA(ipiv),
 *                                         <f2c_doublecomplex *>np.PyArray_DATA(b),&ldb,
 *                                         &info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(zgesv_)((&__pyx_v_n), (&__pyx_v_nrhs), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_a))), (&__pyx_v_lda), ((int *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_ipiv))), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_b))), (&__pyx_v_ldb), (&__pyx_v_info));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":486
 *                                         &info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["zgesv_"] = lapack_lite_status__
 *     retval["n"] = n
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":487
 * 
 *     retval = {}
 *     retval["zgesv_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs
 */
  __pyx_t_2 = __pyx_v_lapack_lite_status__;
  __site_setindex_487_10->Target(__site_setindex_487_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"zgesv_"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":488
 *     retval = {}
 *     retval["zgesv_"] = lapack_lite_status__
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda
 */
  __pyx_t_2 = __pyx_v_n;
  __site_setindex_488_10->Target(__site_setindex_488_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":489
 *     retval["zgesv_"] = lapack_lite_status__
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb
 */
  __pyx_t_2 = __pyx_v_nrhs;
  __site_setindex_489_10->Target(__site_setindex_489_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"nrhs"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":490
 *     retval["n"] = n
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["ldb"] = ldb
 *     retval["info"] = info
 */
  __pyx_t_2 = __pyx_v_lda;
  __site_setindex_490_10->Target(__site_setindex_490_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":491
 *     retval["nrhs"] = nrhs
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_2 = __pyx_v_ldb;
  __site_setindex_491_10->Target(__site_setindex_491_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldb"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":492
 *     retval["lda"] = lda
 *     retval["ldb"] = ldb
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_2 = __pyx_v_info;
  __site_setindex_492_10->Target(__site_setindex_492_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":493
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":496
 * 
 * 
 * cdef zgesdd(char jobz, int m, int n, object a, int lda,             # <<<<<<<<<<<<<<
 *             object s, object u, int ldu, object vt, int ldvt,
 *             object work, int lwork, object rwork, object iwork, int info):
 */

static  System::Object^ zgesdd(char __pyx_v_jobz, int __pyx_v_m, int __pyx_v_n, System::Object^ __pyx_v_a, int __pyx_v_lda, System::Object^ __pyx_v_s, System::Object^ __pyx_v_u, int __pyx_v_ldu, System::Object^ __pyx_v_vt, int __pyx_v_ldvt, System::Object^ __pyx_v_work, int __pyx_v_lwork, System::Object^ __pyx_v_rwork, System::Object^ __pyx_v_iwork, int __pyx_v_info) {
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_retval = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":501
 *     cdef int lapack_lite_status__
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","zgesdd"): return None
 *     if not check_object(u,np.NPY_CDOUBLE,"u","np.NPY_CDOUBLE","zgesdd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_a, NPY_CDOUBLE, __pyx_k__a, __pyx_k_3, __pyx_k__zgesdd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":502
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","zgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(u,np.NPY_CDOUBLE,"u","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(vt,np.NPY_CDOUBLE,"vt","np.NPY_CDOUBLE","zgesdd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_s, NPY_DOUBLE, __pyx_k__s, __pyx_k_1, __pyx_k__zgesdd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":503
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","zgesdd"): return None
 *     if not check_object(u,np.NPY_CDOUBLE,"u","np.NPY_CDOUBLE","zgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(vt,np.NPY_CDOUBLE,"vt","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgesdd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_u, NPY_CDOUBLE, __pyx_k__u, __pyx_k_3, __pyx_k__zgesdd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":504
 *     if not check_object(s,np.NPY_DOUBLE,"s","np.NPY_DOUBLE","zgesdd"): return None
 *     if not check_object(u,np.NPY_CDOUBLE,"u","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(vt,np.NPY_CDOUBLE,"vt","np.NPY_CDOUBLE","zgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgesdd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_vt, NPY_CDOUBLE, __pyx_k__vt, __pyx_k_3, __pyx_k__zgesdd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":505
 *     if not check_object(u,np.NPY_CDOUBLE,"u","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(vt,np.NPY_CDOUBLE,"vt","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgesdd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","zgesdd"): return None
 */
  __pyx_t_1 = (!check_object(__pyx_v_work, NPY_CDOUBLE, __pyx_k__work, __pyx_k_3, __pyx_k__zgesdd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":506
 *     if not check_object(vt,np.NPY_CDOUBLE,"vt","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgesdd"): return None             # <<<<<<<<<<<<<<
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","zgesdd"): return None
 * 
 */
  __pyx_t_1 = (!check_object(__pyx_v_rwork, NPY_DOUBLE, __pyx_k__rwork, __pyx_k_1, __pyx_k__zgesdd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L8;
  }
  __pyx_L8:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":507
 *     if not check_object(work,np.NPY_CDOUBLE,"work","np.NPY_CDOUBLE","zgesdd"): return None
 *     if not check_object(rwork,np.NPY_DOUBLE,"rwork","np.NPY_DOUBLE","zgesdd"): return None
 *     if not check_object(iwork,np.NPY_INT,"iwork","np.NPY_INT","zgesdd"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_zgesdd(&jobz,&m,&n,
 */
  __pyx_t_1 = (!check_object(__pyx_v_iwork, NPY_INT, __pyx_k__iwork, __pyx_k_2, __pyx_k__zgesdd));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L9;
  }
  __pyx_L9:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":510
 * 
 *     lapack_lite_status__ = lapack_zgesdd(&jobz,&m,&n,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(s),
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(u),&ldu,
 */
  if (__pyx_v_a == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":511
 *     lapack_lite_status__ = lapack_zgesdd(&jobz,&m,&n,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
 *                                          <double *>np.PyArray_DATA(s),             # <<<<<<<<<<<<<<
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(u),&ldu,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(vt),&ldvt,
 */
  if (__pyx_v_s == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_s) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":512
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
 *                                          <double *>np.PyArray_DATA(s),
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(u),&ldu,             # <<<<<<<<<<<<<<
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(vt),&ldvt,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 */
  if (__pyx_v_u == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_u) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":513
 *                                          <double *>np.PyArray_DATA(s),
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(u),&ldu,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(vt),&ldvt,             # <<<<<<<<<<<<<<
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 *                                          <double *>np.PyArray_DATA(rwork),
 */
  if (__pyx_v_vt == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_vt) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":514
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(u),&ldu,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(vt),&ldvt,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,             # <<<<<<<<<<<<<<
 *                                          <double *>np.PyArray_DATA(rwork),
 *                                          <int *>np.PyArray_DATA(iwork),&info)
 */
  if (__pyx_v_work == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":515
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(vt),&ldvt,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 *                                          <double *>np.PyArray_DATA(rwork),             # <<<<<<<<<<<<<<
 *                                          <int *>np.PyArray_DATA(iwork),&info)
 * 
 */
  if (__pyx_v_rwork == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_rwork) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":516
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,
 *                                          <double *>np.PyArray_DATA(rwork),
 *                                          <int *>np.PyArray_DATA(iwork),&info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  if (__pyx_v_iwork == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_iwork) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(zgesdd_)((&__pyx_v_jobz), (&__pyx_v_m), (&__pyx_v_n), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_a))), (&__pyx_v_lda), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_s))), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_u))), (&__pyx_v_ldu), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_vt))), (&__pyx_v_ldvt), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_work))), (&__pyx_v_lwork), ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_rwork))), ((int *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_iwork))), (&__pyx_v_info));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":518
 *                                          <int *>np.PyArray_DATA(iwork),&info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["zgesdd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":519
 * 
 *     retval = {}
 *     retval["zgesdd_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["jobz"] = jobz
 *     retval["m"] = m
 */
  __pyx_t_2 = __pyx_v_lapack_lite_status__;
  __site_setindex_519_10->Target(__site_setindex_519_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"zgesdd_"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":520
 *     retval = {}
 *     retval["zgesdd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz             # <<<<<<<<<<<<<<
 *     retval["m"] = m
 *     retval["n"] = n
 */
  __pyx_t_2 = __pyx_v_jobz;
  __site_setindex_520_10->Target(__site_setindex_520_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"jobz"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":521
 *     retval["zgesdd_"] = lapack_lite_status__
 *     retval["jobz"] = jobz
 *     retval["m"] = m             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_2 = __pyx_v_m;
  __site_setindex_521_10->Target(__site_setindex_521_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"m"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":522
 *     retval["jobz"] = jobz
 *     retval["m"] = m
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["ldu"] = ldu
 */
  __pyx_t_2 = __pyx_v_n;
  __site_setindex_522_10->Target(__site_setindex_522_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":523
 *     retval["m"] = m
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["ldu"] = ldu
 *     retval["ldvt"] = ldvt
 */
  __pyx_t_2 = __pyx_v_lda;
  __site_setindex_523_10->Target(__site_setindex_523_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":524
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["ldu"] = ldu             # <<<<<<<<<<<<<<
 *     retval["ldvt"] = ldvt
 *     retval["lwork"] = lwork
 */
  __pyx_t_2 = __pyx_v_ldu;
  __site_setindex_524_10->Target(__site_setindex_524_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldu"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":525
 *     retval["lda"] = lda
 *     retval["ldu"] = ldu
 *     retval["ldvt"] = ldvt             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 */
  __pyx_t_2 = __pyx_v_ldvt;
  __site_setindex_525_10->Target(__site_setindex_525_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"ldvt"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":526
 *     retval["ldu"] = ldu
 *     retval["ldvt"] = ldvt
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_2 = __pyx_v_lwork;
  __site_setindex_526_10->Target(__site_setindex_526_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":527
 *     retval["ldvt"] = ldvt
 *     retval["lwork"] = lwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_2 = __pyx_v_info;
  __site_setindex_527_10->Target(__site_setindex_527_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":528
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":531
 * 
 * 
 * cdef zgetrf(int m, int n, object a, int lda, object ipiv, int info):             # <<<<<<<<<<<<<<
 *     cdef int lapack_lite_status__
 * 
 */

static  System::Object^ zgetrf(int __pyx_v_m, int __pyx_v_n, System::Object^ __pyx_v_a, int __pyx_v_lda, System::Object^ __pyx_v_ipiv, int __pyx_v_info) {
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_retval = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":534
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
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":535
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
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":538
 * 
 *     lapack_lite_status__ = lapack_zgetrf(&m,&n,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,             # <<<<<<<<<<<<<<
 *                                          <int *>np.PyArray_DATA(ipiv),&info)
 * 
 */
  if (__pyx_v_a == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":539
 *     lapack_lite_status__ = lapack_zgetrf(&m,&n,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
 *                                          <int *>np.PyArray_DATA(ipiv),&info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  if (__pyx_v_ipiv == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_ipiv) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(zgetrf_)((&__pyx_v_m), (&__pyx_v_n), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_a))), (&__pyx_v_lda), ((int *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_ipiv))), (&__pyx_v_info));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":541
 *                                          <int *>np.PyArray_DATA(ipiv),&info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["zgetrf_"] = lapack_lite_status__
 *     retval["m"] = m
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":542
 * 
 *     retval = {}
 *     retval["zgetrf_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["m"] = m
 *     retval["n"] = n
 */
  __pyx_t_2 = __pyx_v_lapack_lite_status__;
  __site_setindex_542_10->Target(__site_setindex_542_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"zgetrf_"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":543
 *     retval = {}
 *     retval["zgetrf_"] = lapack_lite_status__
 *     retval["m"] = m             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_2 = __pyx_v_m;
  __site_setindex_543_10->Target(__site_setindex_543_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"m"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":544
 *     retval["zgetrf_"] = lapack_lite_status__
 *     retval["m"] = m
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["info"] = info
 */
  __pyx_t_2 = __pyx_v_n;
  __site_setindex_544_10->Target(__site_setindex_544_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":545
 *     retval["m"] = m
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_2 = __pyx_v_lda;
  __site_setindex_545_10->Target(__site_setindex_545_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":546
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_2 = __pyx_v_info;
  __site_setindex_546_10->Target(__site_setindex_546_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":547
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":550
 * 
 * 
 * cdef zpotrf(char uplo, int n, object a, int lda, int info):             # <<<<<<<<<<<<<<
 *     cdef int  lapack_lite_status__
 * 
 */

static  System::Object^ zpotrf(char __pyx_v_uplo, int __pyx_v_n, System::Object^ __pyx_v_a, int __pyx_v_lda, int __pyx_v_info) {
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_retval = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":553
 *     cdef int  lapack_lite_status__
 * 
 *     if not check_object(a,np.NPY_CDOUBLE,"a","np.NPY_CDOUBLE","zpotrf"): return None             # <<<<<<<<<<<<<<
 * 
 *     lapack_lite_status__ = lapack_zpotrf(&uplo,&n,
 */
  __pyx_t_1 = (!check_object(__pyx_v_a, NPY_CDOUBLE, __pyx_k__a, __pyx_k_3, __pyx_k__zpotrf));
  if (__pyx_t_1) {
    __pyx_r = nullptr;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":556
 * 
 *     lapack_lite_status__ = lapack_zpotrf(&uplo,&n,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,             # <<<<<<<<<<<<<<
 *                                          &info)
 * 
 */
  if (__pyx_v_a == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":557
 *     lapack_lite_status__ = lapack_zpotrf(&uplo,&n,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a),&lda,
 *                                          &info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(zpotrf_)((&__pyx_v_uplo), (&__pyx_v_n), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_a))), (&__pyx_v_lda), (&__pyx_v_info));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":559
 *                                          &info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["zpotrf_"] = lapack_lite_status__
 *     retval["n"] = n
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":560
 * 
 *     retval = {}
 *     retval["zpotrf_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_2 = __pyx_v_lapack_lite_status__;
  __site_setindex_560_10->Target(__site_setindex_560_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"zpotrf_"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":561
 *     retval = {}
 *     retval["zpotrf_"] = lapack_lite_status__
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["info"] = info
 */
  __pyx_t_2 = __pyx_v_n;
  __site_setindex_561_10->Target(__site_setindex_561_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":562
 *     retval["zpotrf_"] = lapack_lite_status__
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_2 = __pyx_v_lda;
  __site_setindex_562_10->Target(__site_setindex_562_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":563
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_2 = __pyx_v_info;
  __site_setindex_563_10->Target(__site_setindex_563_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":564
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":567
 * 
 * 
 * cdef zgeqrf(int m, int n, object a, int lda, object tau, object work, int lwork, int info):             # <<<<<<<<<<<<<<
 *     cdef int lapack_lite_status__
 * 
 */

static  System::Object^ zgeqrf(int __pyx_v_m, int __pyx_v_n, System::Object^ __pyx_v_a, int __pyx_v_lda, System::Object^ __pyx_v_tau, System::Object^ __pyx_v_work, int __pyx_v_lwork, int __pyx_v_info) {
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_retval = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":571
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
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":572
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
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":573
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
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":576
 * 
 *     lapack_lite_status__ = lapack_zgeqrf(&m, &n,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a), &lda,             # <<<<<<<<<<<<<<
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(tau),
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work), &lwork,
 */
  if (__pyx_v_a == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":577
 *     lapack_lite_status__ = lapack_zgeqrf(&m, &n,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a), &lda,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(tau),             # <<<<<<<<<<<<<<
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work), &lwork,
 *                                          &info)
 */
  if (__pyx_v_tau == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_tau) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":578
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a), &lda,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(tau),
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work), &lwork,             # <<<<<<<<<<<<<<
 *                                          &info)
 * 
 */
  if (__pyx_v_work == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":579
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(tau),
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work), &lwork,
 *                                          &info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(zgeqrf_)((&__pyx_v_m), (&__pyx_v_n), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_a))), (&__pyx_v_lda), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_tau))), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_work))), (&__pyx_v_lwork), (&__pyx_v_info));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":581
 *                                          &info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["zgeqrf_"] = lapack_lite_status__
 *     retval["m"] = m
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":582
 * 
 *     retval = {}
 *     retval["zgeqrf_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["m"] = m
 *     retval["n"] = n
 */
  __pyx_t_2 = __pyx_v_lapack_lite_status__;
  __site_setindex_582_10->Target(__site_setindex_582_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"zgeqrf_"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":583
 *     retval = {}
 *     retval["zgeqrf_"] = lapack_lite_status__
 *     retval["m"] = m             # <<<<<<<<<<<<<<
 *     retval["n"] = n
 *     retval["lda"] = lda
 */
  __pyx_t_2 = __pyx_v_m;
  __site_setindex_583_10->Target(__site_setindex_583_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"m"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":584
 *     retval["zgeqrf_"] = lapack_lite_status__
 *     retval["m"] = m
 *     retval["n"] = n             # <<<<<<<<<<<<<<
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork
 */
  __pyx_t_2 = __pyx_v_n;
  __site_setindex_584_10->Target(__site_setindex_584_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"n"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":585
 *     retval["m"] = m
 *     retval["n"] = n
 *     retval["lda"] = lda             # <<<<<<<<<<<<<<
 *     retval["lwork"] = lwork
 *     retval["info"] = info
 */
  __pyx_t_2 = __pyx_v_lda;
  __site_setindex_585_10->Target(__site_setindex_585_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lda"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":586
 *     retval["n"] = n
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_2 = __pyx_v_lwork;
  __site_setindex_586_10->Target(__site_setindex_586_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"lwork"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":587
 *     retval["lda"] = lda
 *     retval["lwork"] = lwork
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_2 = __pyx_v_info;
  __site_setindex_587_10->Target(__site_setindex_587_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":588
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":591
 * 
 * 
 * cdef zungqr(int m, int n, int k, object a, int lda, object tau, object work, int lwork, int info):             # <<<<<<<<<<<<<<
 *     cdef int  lapack_lite_status__
 * 
 */

static  System::Object^ zungqr(int __pyx_v_m, int __pyx_v_n, int __pyx_v_k, System::Object^ __pyx_v_a, int __pyx_v_lda, System::Object^ __pyx_v_tau, System::Object^ __pyx_v_work, int __pyx_v_lwork, int __pyx_v_info) {
  int __pyx_v_lapack_lite_status__;
  System::Object^ __pyx_v_retval;
  System::Object^ __pyx_r = nullptr;
  int __pyx_t_1;
  System::Object^ __pyx_t_2 = nullptr;
  __pyx_v_retval = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":594
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
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":595
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
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":596
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
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":599
 * 
 *     lapack_lite_status__ = lapack_zungqr(&m, &n, &k,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a), &lda,             # <<<<<<<<<<<<<<
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(tau),
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,&info)
 */
  if (__pyx_v_a == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_a) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":600
 *     lapack_lite_status__ = lapack_zungqr(&m, &n, &k,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a), &lda,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(tau),             # <<<<<<<<<<<<<<
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,&info)
 * 
 */
  if (__pyx_v_tau == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_tau) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":601
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(a), &lda,
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(tau),
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,&info)             # <<<<<<<<<<<<<<
 * 
 *     retval = {}
 */
  if (__pyx_v_work == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_work) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_lapack_lite_status__ = GLOBALFUNC(zungqr_)((&__pyx_v_m), (&__pyx_v_n), (&__pyx_v_k), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_a))), (&__pyx_v_lda), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_tau))), ((f2c_doublecomplex *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_work))), (&__pyx_v_lwork), (&__pyx_v_info));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":603
 *                                          <f2c_doublecomplex *>np.PyArray_DATA(work),&lwork,&info)
 * 
 *     retval = {}             # <<<<<<<<<<<<<<
 *     retval["zungqr_"] = lapack_lite_status__
 *     retval["info"] = info
 */
  __pyx_t_2 = PythonOps::MakeEmptyDict();
  __pyx_v_retval = __pyx_t_2;
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":604
 * 
 *     retval = {}
 *     retval["zungqr_"] = lapack_lite_status__             # <<<<<<<<<<<<<<
 *     retval["info"] = info
 *     return retval
 */
  __pyx_t_2 = __pyx_v_lapack_lite_status__;
  __site_setindex_604_10->Target(__site_setindex_604_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"zungqr_"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":605
 *     retval = {}
 *     retval["zungqr_"] = lapack_lite_status__
 *     retval["info"] = info             # <<<<<<<<<<<<<<
 *     return retval
 * 
 */
  __pyx_t_2 = __pyx_v_info;
  __site_setindex_605_10->Target(__site_setindex_605_10, ((System::Object^)__pyx_v_retval), ((System::Object^)"info"), __pyx_t_2);
  __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":606
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":189
 *     object Npy_INTERFACE_array "Npy_INTERFACE_OBJECT" (NpyArray*)
 * 
 * cdef inline object PyUFunc_FromFuncAndData(PyUFuncGenericFunction* func, void** data,             # <<<<<<<<<<<<<<
 *         char* types, int ntypes, int nin, int nout,
 *         int identity, char* name, char* doc, int c):
 */

static CYTHON_INLINE System::Object^ PyUFunc_FromFuncAndData(__pyx_t_5numpy_6linalg_5numpy_PyUFuncGenericFunction *__pyx_v_func, void **__pyx_v_data, char *__pyx_v_types, int __pyx_v_ntypes, int __pyx_v_nin, int __pyx_v_nout, int __pyx_v_identity, char *__pyx_v_name, char *__pyx_v_doc, int __pyx_v_c) {
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":192
 *         char* types, int ntypes, int nin, int nout,
 *         int identity, char* name, char* doc, int c):
 *    return Npy_INTERFACE_ufunc(NpyUFunc_FromFuncAndDataAndSignature(func, data, types, ntypes, nin, nout, identity, name, doc, c, NULL))             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_ZEROS(int ndim, intp_t *shape, int typenum, int fortran):
 */
  __pyx_t_1 = Npy_INTERFACE_OBJECT(NpyUFunc_FromFuncAndDataAndSignature(__pyx_v_func, __pyx_v_data, __pyx_v_types, __pyx_v_ntypes, __pyx_v_nin, __pyx_v_nout, __pyx_v_identity, __pyx_v_name, __pyx_v_doc, __pyx_v_c, NULL)); 
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":194
 *    return Npy_INTERFACE_ufunc(NpyUFunc_FromFuncAndDataAndSignature(func, data, types, ntypes, nin, nout, identity, name, doc, c, NULL))
 * 
 * cdef inline object PyArray_ZEROS(int ndim, intp_t *shape, int typenum, int fortran):             # <<<<<<<<<<<<<<
 *     shape_list = []
 *     cdef int i
 */

static CYTHON_INLINE System::Object^ PyArray_ZEROS(int __pyx_v_ndim, __pyx_t_5numpy_6linalg_5numpy_intp_t *__pyx_v_shape, int __pyx_v_typenum, int __pyx_v_fortran) {
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

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":195
 * 
 * cdef inline object PyArray_ZEROS(int ndim, intp_t *shape, int typenum, int fortran):
 *     shape_list = []             # <<<<<<<<<<<<<<
 *     cdef int i
 *     for i in range(ndim):
 */
  __pyx_t_1 = PythonOps::MakeListNoCopy(gcnew array<System::Object^>{});
  __pyx_v_shape_list = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":197
 *     shape_list = []
 *     cdef int i
 *     for i in range(ndim):             # <<<<<<<<<<<<<<
 *         shape_list.append(shape[i])
 *     import numpy
 */
  __pyx_t_2 = __pyx_v_ndim;
  for (__pyx_t_3 = 0; __pyx_t_3 < __pyx_t_2; __pyx_t_3+=1) {
    __pyx_v_i = __pyx_t_3;

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":198
 *     cdef int i
 *     for i in range(ndim):
 *         shape_list.append(shape[i])             # <<<<<<<<<<<<<<
 *     import numpy
 *     return numpy.zeros(shape_list, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), 'F' if fortran else 'C')
 */
    __pyx_t_1 = __site_get_append_198_18->Target(__site_get_append_198_18, ((System::Object^)__pyx_v_shape_list), __pyx_context);
    __pyx_t_4 = (__pyx_v_shape[__pyx_v_i]);
    __pyx_t_5 = __site_call1_198_25->Target(__site_call1_198_25, __pyx_context, __pyx_t_1, __pyx_t_4);
    __pyx_t_1 = nullptr;
    __pyx_t_4 = nullptr;
    __pyx_t_5 = nullptr;
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":199
 *     for i in range(ndim):
 *         shape_list.append(shape[i])
 *     import numpy             # <<<<<<<<<<<<<<
 *     return numpy.zeros(shape_list, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), 'F' if fortran else 'C')
 * 
 */
  __pyx_t_5 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "numpy", -1));
  __pyx_v_numpy = __pyx_t_5;
  __pyx_t_5 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":200
 *         shape_list.append(shape[i])
 *     import numpy
 *     return numpy.zeros(shape_list, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), 'F' if fortran else 'C')             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_New(void *subtype, int nd, npy_intp *dims, int type_num, npy_intp *strides, void *data, int itemsize, int flags, void *obj):
 */
  __pyx_t_5 = __site_get_zeros_200_16->Target(__site_get_zeros_200_16, __pyx_v_numpy, __pyx_context);
  __pyx_t_4 = Npy_INTERFACE_OBJECT(NpyArray_DescrFromType(__pyx_v_typenum)); 
  if (__pyx_v_fortran) {
    __pyx_t_1 = "F";
  } else {
    __pyx_t_1 = "C";
  }
  __pyx_t_6 = __site_call3_200_22->Target(__site_call3_200_22, __pyx_context, __pyx_t_5, ((System::Object^)__pyx_v_shape_list), __pyx_t_4, ((System::Object^)__pyx_t_1));
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":202
 *     return numpy.zeros(shape_list, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), 'F' if fortran else 'C')
 * 
 * cdef inline object PyArray_New(void *subtype, int nd, npy_intp *dims, int type_num, npy_intp *strides, void *data, int itemsize, int flags, void *obj):             # <<<<<<<<<<<<<<
 *     assert subtype == NULL
 *     assert obj == NULL
 */

static CYTHON_INLINE System::Object^ PyArray_New(void *__pyx_v_subtype, int __pyx_v_nd, __pyx_t_5numpy_6linalg_5numpy_npy_intp *__pyx_v_dims, int __pyx_v_type_num, __pyx_t_5numpy_6linalg_5numpy_npy_intp *__pyx_v_strides, void *__pyx_v_data, int __pyx_v_itemsize, int __pyx_v_flags, void *__pyx_v_obj) {
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":203
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

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":204
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

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":205
 *     assert subtype == NULL
 *     assert obj == NULL
 *     return Npy_INTERFACE_array(NpyArray_New(subtype, nd, dims, type_num, strides, data, itemsize, flags, obj))             # <<<<<<<<<<<<<<
 * 
 * cdef inline bint PyArray_CHKFLAGS(ndarray n, int flags):
 */
  __pyx_t_1 = Npy_INTERFACE_OBJECT(NpyArray_New(__pyx_v_subtype, __pyx_v_nd, __pyx_v_dims, __pyx_v_type_num, __pyx_v_strides, __pyx_v_data, __pyx_v_itemsize, __pyx_v_flags, __pyx_v_obj)); 
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":207
 *     return Npy_INTERFACE_array(NpyArray_New(subtype, nd, dims, type_num, strides, data, itemsize, flags, obj))
 * 
 * cdef inline bint PyArray_CHKFLAGS(ndarray n, int flags):             # <<<<<<<<<<<<<<
 *      # XXX "long long" is wrong type
 *     return  NpyArray_CHKFLAGS(<NpyArray*> <long long>n.Array, flags)
 */

static CYTHON_INLINE int PyArray_CHKFLAGS(NumpyDotNet::ndarray^ __pyx_v_n, int __pyx_v_flags) {
  int __pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  PY_LONG_LONG __pyx_t_2;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":209
 * cdef inline bint PyArray_CHKFLAGS(ndarray n, int flags):
 *      # XXX "long long" is wrong type
 *     return  NpyArray_CHKFLAGS(<NpyArray*> <long long>n.Array, flags)             # <<<<<<<<<<<<<<
 * 
 * cdef inline void* PyArray_DATA(ndarray n):
 */
  __pyx_t_1 = __site_get_Array_209_54->Target(__site_get_Array_209_54, ((System::Object^)__pyx_v_n), __pyx_context);
  __pyx_t_2 = __site_cvt_PY_LONG_LONG_209_54->Target(__site_cvt_PY_LONG_LONG_209_54, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_r = NpyArray_CHKFLAGS(((NpyArray *)((PY_LONG_LONG)__pyx_t_2)), __pyx_v_flags);
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":211
 *     return  NpyArray_CHKFLAGS(<NpyArray*> <long long>n.Array, flags)
 * 
 * cdef inline void* PyArray_DATA(ndarray n):             # <<<<<<<<<<<<<<
 *     # XXX "long long" is wrong type
 *     return NpyArray_DATA(<NpyArray*> <long long>n.Array)
 */

static CYTHON_INLINE void *PyArray_DATA(NumpyDotNet::ndarray^ __pyx_v_n) {
  void *__pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  PY_LONG_LONG __pyx_t_2;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":213
 * cdef inline void* PyArray_DATA(ndarray n):
 *     # XXX "long long" is wrong type
 *     return NpyArray_DATA(<NpyArray*> <long long>n.Array)             # <<<<<<<<<<<<<<
 * 
 * cdef inline intp_t* PyArray_DIMS(ndarray n):
 */
  __pyx_t_1 = __site_get_Array_213_49->Target(__site_get_Array_213_49, ((System::Object^)__pyx_v_n), __pyx_context);
  __pyx_t_2 = __site_cvt_PY_LONG_LONG_213_49->Target(__site_cvt_PY_LONG_LONG_213_49, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_r = NpyArray_DATA(((NpyArray *)((PY_LONG_LONG)__pyx_t_2)));
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":215
 *     return NpyArray_DATA(<NpyArray*> <long long>n.Array)
 * 
 * cdef inline intp_t* PyArray_DIMS(ndarray n):             # <<<<<<<<<<<<<<
 *     # XXX "long long" is wrong type
 *     return NpyArray_DIMS(<NpyArray*> <long long>n.Array)
 */

static CYTHON_INLINE __pyx_t_5numpy_6linalg_5numpy_intp_t *PyArray_DIMS(NumpyDotNet::ndarray^ __pyx_v_n) {
  __pyx_t_5numpy_6linalg_5numpy_intp_t *__pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  PY_LONG_LONG __pyx_t_2;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":217
 * cdef inline intp_t* PyArray_DIMS(ndarray n):
 *     # XXX "long long" is wrong type
 *     return NpyArray_DIMS(<NpyArray*> <long long>n.Array)             # <<<<<<<<<<<<<<
 * 
 * cdef inline intp_t PyArray_SIZE(ndarray n):
 */
  __pyx_t_1 = __site_get_Array_217_49->Target(__site_get_Array_217_49, ((System::Object^)__pyx_v_n), __pyx_context);
  __pyx_t_2 = __site_cvt_PY_LONG_LONG_217_49->Target(__site_cvt_PY_LONG_LONG_217_49, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_r = NpyArray_DIMS(((NpyArray *)((PY_LONG_LONG)__pyx_t_2)));
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":219
 *     return NpyArray_DIMS(<NpyArray*> <long long>n.Array)
 * 
 * cdef inline intp_t PyArray_SIZE(ndarray n):             # <<<<<<<<<<<<<<
 *     # XXX "long long" is wrong type
 *     return NpyArray_SIZE(<NpyArray*> <long long>n.Array)
 */

static CYTHON_INLINE __pyx_t_5numpy_6linalg_5numpy_intp_t PyArray_SIZE(NumpyDotNet::ndarray^ __pyx_v_n) {
  __pyx_t_5numpy_6linalg_5numpy_intp_t __pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  PY_LONG_LONG __pyx_t_2;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":221
 * cdef inline intp_t PyArray_SIZE(ndarray n):
 *     # XXX "long long" is wrong type
 *     return NpyArray_SIZE(<NpyArray*> <long long>n.Array)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_FromAny(op, newtype, min_depth, max_depth, flags, context):
 */
  __pyx_t_1 = __site_get_Array_221_49->Target(__site_get_Array_221_49, ((System::Object^)__pyx_v_n), __pyx_context);
  __pyx_t_2 = __site_cvt_PY_LONG_LONG_221_49->Target(__site_cvt_PY_LONG_LONG_221_49, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_r = NpyArray_SIZE(((NpyArray *)((PY_LONG_LONG)__pyx_t_2)));
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":223
 *     return NpyArray_SIZE(<NpyArray*> <long long>n.Array)
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

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":224
 * 
 * cdef inline object PyArray_FromAny(op, newtype, min_depth, max_depth, flags, context):
 *     import clr             # <<<<<<<<<<<<<<
 *     import NumpyDotNet.NpyArray
 *     return NumpyDotNet.NpyArray.FromAny(op, newtype, min_depth, max_depth, flags, context)
 */
  __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "clr", -1));
  __pyx_v_clr = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":225
 * cdef inline object PyArray_FromAny(op, newtype, min_depth, max_depth, flags, context):
 *     import clr
 *     import NumpyDotNet.NpyArray             # <<<<<<<<<<<<<<
 *     return NumpyDotNet.NpyArray.FromAny(op, newtype, min_depth, max_depth, flags, context)
 * 
 */
  __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "NumpyDotNet.NpyArray", -1));
  __pyx_v_NumpyDotNet = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":226
 *     import clr
 *     import NumpyDotNet.NpyArray
 *     return NumpyDotNet.NpyArray.FromAny(op, newtype, min_depth, max_depth, flags, context)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_FROMANY(m, type, min, max, flags):
 */
  __pyx_t_1 = __site_get_NpyArray_226_22->Target(__site_get_NpyArray_226_22, __pyx_v_NumpyDotNet, __pyx_context);
  __pyx_t_2 = __site_get_FromAny_226_31->Target(__site_get_FromAny_226_31, __pyx_t_1, __pyx_context);
  __pyx_t_1 = nullptr;
  __pyx_t_1 = __site_call6_226_39->Target(__site_call6_226_39, __pyx_context, __pyx_t_2, __pyx_v_op, __pyx_v_newtype, __pyx_v_min_depth, __pyx_v_max_depth, __pyx_v_flags, __pyx_v_context);
  __pyx_t_2 = nullptr;
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":228
 *     return NumpyDotNet.NpyArray.FromAny(op, newtype, min_depth, max_depth, flags, context)
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

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":229
 * 
 * cdef inline object PyArray_FROMANY(m, type, min, max, flags):
 *     if flags & NPY_ENSURECOPY:             # <<<<<<<<<<<<<<
 *         flags |= NPY_DEFAULT
 *     return PyArray_FromAny(m, Npy_INTERFACE_descr(NpyArray_DescrFromType(type)), min, max, flags, None)
 */
  __pyx_t_1 = (System::Object^)(long long)(NPY_ENSURECOPY);
  __pyx_t_2 = __site_op_and_229_13->Target(__site_op_and_229_13, __pyx_v_flags, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_t_3 = __site_istrue_229_13->Target(__site_istrue_229_13, __pyx_t_2);
  __pyx_t_2 = nullptr;
  if (__pyx_t_3) {

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":230
 * cdef inline object PyArray_FROMANY(m, type, min, max, flags):
 *     if flags & NPY_ENSURECOPY:
 *         flags |= NPY_DEFAULT             # <<<<<<<<<<<<<<
 *     return PyArray_FromAny(m, Npy_INTERFACE_descr(NpyArray_DescrFromType(type)), min, max, flags, None)
 * 
 */
    __pyx_t_2 = (System::Object^)(long long)(NPY_DEFAULT);
    __pyx_t_1 = __site_op_ior_230_14->Target(__site_op_ior_230_14, __pyx_v_flags, __pyx_t_2);
    __pyx_t_2 = nullptr;
    __pyx_v_flags = __pyx_t_1;
    __pyx_t_1 = nullptr;
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":231
 *     if flags & NPY_ENSURECOPY:
 *         flags |= NPY_DEFAULT
 *     return PyArray_FromAny(m, Npy_INTERFACE_descr(NpyArray_DescrFromType(type)), min, max, flags, None)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_Check(obj):
 */
  __pyx_t_4 = __site_cvt_int_231_77->Target(__site_cvt_int_231_77, __pyx_v_type);
  __pyx_t_1 = Npy_INTERFACE_OBJECT(NpyArray_DescrFromType(__pyx_t_4)); 
  __pyx_t_2 = PyArray_FromAny(__pyx_v_m, __pyx_t_1, __pyx_v_min, __pyx_v_max, __pyx_v_flags, nullptr); 
  __pyx_t_1 = nullptr;
  __pyx_r = __pyx_t_2;
  __pyx_t_2 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":233
 *     return PyArray_FromAny(m, Npy_INTERFACE_descr(NpyArray_DescrFromType(type)), min, max, flags, None)
 * 
 * cdef inline object PyArray_Check(obj):             # <<<<<<<<<<<<<<
 *     return isinstance(obj, ndarray)
 * 
 */

static CYTHON_INLINE System::Object^ PyArray_Check(System::Object^ __pyx_v_obj) {
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":234
 * 
 * cdef inline object PyArray_Check(obj):
 *     return isinstance(obj, ndarray)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_NDIM(obj):
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "isinstance");
  __pyx_t_2 = __site_call2_234_21->Target(__site_call2_234_21, __pyx_context, __pyx_t_1, __pyx_v_obj, ((System::Object^)((System::Object^)__pyx_ptype_5numpy_6linalg_5numpy_ndarray)));
  __pyx_t_1 = nullptr;
  __pyx_r = __pyx_t_2;
  __pyx_t_2 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":236
 *     return isinstance(obj, ndarray)
 * 
 * cdef inline object PyArray_NDIM(obj):             # <<<<<<<<<<<<<<
 *     return obj.ndim
 * 
 */

static CYTHON_INLINE System::Object^ PyArray_NDIM(System::Object^ __pyx_v_obj) {
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":237
 * 
 * cdef inline object PyArray_NDIM(obj):
 *     return obj.ndim             # <<<<<<<<<<<<<<
 * 
 * cdef inline void import_array():
 */
  __pyx_t_1 = __site_get_ndim_237_14->Target(__site_get_ndim_237_14, __pyx_v_obj, __pyx_context);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":239
 *     return obj.ndim
 * 
 * cdef inline void import_array():             # <<<<<<<<<<<<<<
 *     pass
 */

static CYTHON_INLINE void import_array(void) {

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
  __site_istrue_36_27 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_op_mod_37_81 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Modulo));
  __site_call1_37_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_PyArray_FLAGS_38_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "PyArray_FLAGS", false));
  __site_call1_38_30 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_and_38_35 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::And));
  __site_istrue_38_35 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_op_mod_39_77 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Modulo));
  __site_call1_39_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_PyArray_TYPE_40_11 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "PyArray_TYPE", false));
  __site_call1_40_24 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_ne_40_29 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::NotEqual));
  __site_istrue_40_29 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_op_mod_41_77 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Modulo));
  __site_call1_41_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_PyArray_DESCR_42_11 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "PyArray_DESCR", false));
  __site_call1_42_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_byteorder_42_29 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "byteorder", false));
  __site_op_ne_42_40 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::NotEqual));
  __site_istrue_42_40 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_get_PyArray_DESCR_42_53 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "PyArray_DESCR", false));
  __site_call1_42_67 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_byteorder_42_71 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "byteorder", false));
  __site_op_ne_42_82 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::NotEqual));
  __site_istrue_42_82 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_op_mod_43_85 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Modulo));
  __site_call1_43_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_setindex_70_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_71_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_72_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_73_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_74_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_75_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_76_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_77_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_78_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_137_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_138_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_139_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_140_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_141_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_142_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_143_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_144_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_207_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_208_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_209_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_210_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_211_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_212_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_213_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_214_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_215_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_238_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_239_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_240_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_241_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_242_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_243_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_244_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_245_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_246_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_247_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_266_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_267_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_268_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_269_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_270_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_271_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_316_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_317_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_318_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_319_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_320_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_321_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_322_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_323_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_324_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_338_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_339_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_340_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_341_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_342_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_356_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_357_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_358_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_359_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_378_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_379_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_380_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_381_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_382_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_383_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_401_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_402_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_427_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_428_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_429_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_430_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_431_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_432_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_433_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_434_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_435_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_461_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_462_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_463_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_464_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_465_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_466_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_467_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_468_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_469_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_487_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_488_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_489_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_490_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_491_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_492_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_519_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_520_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_521_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_522_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_523_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_524_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_525_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_526_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_527_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_542_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_543_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_544_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_545_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_546_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_560_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_561_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_562_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_563_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_582_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_583_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_584_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_585_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_586_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_587_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_604_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_setindex_605_10 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeSetIndexAction(__pyx_context, 2));
  __site_get_append_198_18 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "append", false));
  __site_call1_198_25 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_get_zeros_200_16 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "zeros", false));
  __site_call3_200_22 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(3)));
  __site_get_Array_209_54 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Array", false));
  __site_cvt_PY_LONG_LONG_209_54 = CallSite< System::Func< CallSite^, System::Object^, PY_LONG_LONG >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, PY_LONG_LONG::typeid, ConversionResultKind::ExplicitCast));
  __site_get_Array_213_49 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Array", false));
  __site_cvt_PY_LONG_LONG_213_49 = CallSite< System::Func< CallSite^, System::Object^, PY_LONG_LONG >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, PY_LONG_LONG::typeid, ConversionResultKind::ExplicitCast));
  __site_get_Array_217_49 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Array", false));
  __site_cvt_PY_LONG_LONG_217_49 = CallSite< System::Func< CallSite^, System::Object^, PY_LONG_LONG >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, PY_LONG_LONG::typeid, ConversionResultKind::ExplicitCast));
  __site_get_Array_221_49 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "Array", false));
  __site_cvt_PY_LONG_LONG_221_49 = CallSite< System::Func< CallSite^, System::Object^, PY_LONG_LONG >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, PY_LONG_LONG::typeid, ConversionResultKind::ExplicitCast));
  __site_get_NpyArray_226_22 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "NpyArray", false));
  __site_get_FromAny_226_31 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "FromAny", false));
  __site_call6_226_39 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(6)));
  __site_op_and_229_13 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::And));
  __site_istrue_229_13 = CallSite< System::Func< CallSite^, System::Object^, bool >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, bool::typeid, ConversionResultKind::ExplicitCast));
  __site_op_ior_230_14 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::OrAssign));
  __site_cvt_int_231_77 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_call2_234_21 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(2)));
  __site_get_ndim_237_14 = CallSite< System::Func< CallSite^, System::Object^, CodeContext^, System::Object^ >^ >::Create(PythonOps::MakeGetAction(__pyx_context, "ndim", false));
}
[SpecialName]
static void PerformModuleReload(PythonContext^ context, PythonDictionary^ dict) {
  dict["__builtins__"] = context->BuiltinModuleInstance;
  __pyx_context = (gcnew ModuleContext(dict, context))->GlobalContext;
  __Pyx_InitSites(__pyx_context);
  __Pyx_InitGlobals();
  /*--- Type init code ---*/
  /*--- Create function pointers ---*/
  /*--- Execution code ---*/
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":29
 * 
 * cimport numpy as np
 * np.import_array()             # <<<<<<<<<<<<<<
 * 
 * class LapackError(Exception):
 */
  import_array();

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":31
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

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\lapack_lite.pyx":1
 * """ Cythonized version of lapack_litemodule.c             # <<<<<<<<<<<<<<
 * """
 * 
 */
  __pyx_t_1 = PythonOps::MakeEmptyDict();
  PythonOps::SetGlobal(__pyx_context, "__test__", ((System::Object^)__pyx_t_1));
  __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\linalg\numpy.pxd":239
 *     return obj.ndim
 * 
 * cdef inline void import_array():             # <<<<<<<<<<<<<<
 *     pass
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
/* Cython code section 'end' */
};
[assembly: PythonModule("numpy__linalg__lapack_lite", module_lapack_lite::typeid)];
};
