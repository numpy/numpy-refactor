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
  dict["__module__"] = "numpy.fft.fftpack_cython";
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
#define __PYX_HAVE_API__numpy__fft__fftpack_cython
#include "string.h"
#include "fftpack.h"
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

typedef int __pyx_t_5numpy_3fft_5numpy_npy_int;

typedef double __pyx_t_5numpy_3fft_5numpy_double_t;

typedef int __pyx_t_5numpy_3fft_5numpy_npy_intp;

typedef signed char __pyx_t_5numpy_3fft_5numpy_npy_int8;

typedef signed short __pyx_t_5numpy_3fft_5numpy_npy_int16;

typedef signed int __pyx_t_5numpy_3fft_5numpy_npy_int32;

typedef signed PY_LONG_LONG __pyx_t_5numpy_3fft_5numpy_npy_int64;

typedef unsigned char __pyx_t_5numpy_3fft_5numpy_npy_uint8;

typedef unsigned short __pyx_t_5numpy_3fft_5numpy_npy_uint16;

typedef unsigned int __pyx_t_5numpy_3fft_5numpy_npy_uint32;

typedef unsigned PY_LONG_LONG __pyx_t_5numpy_3fft_5numpy_npy_uint64;

typedef float __pyx_t_5numpy_3fft_5numpy_npy_float32;

typedef double __pyx_t_5numpy_3fft_5numpy_npy_float64;

typedef __pyx_t_5numpy_3fft_5numpy_npy_intp __pyx_t_5numpy_3fft_5numpy_intp_t;

typedef __pyx_t_5numpy_3fft_5numpy_npy_int8 __pyx_t_5numpy_3fft_5numpy_int8_t;

typedef __pyx_t_5numpy_3fft_5numpy_npy_int16 __pyx_t_5numpy_3fft_5numpy_int16_t;

typedef __pyx_t_5numpy_3fft_5numpy_npy_int32 __pyx_t_5numpy_3fft_5numpy_int32_t;

typedef __pyx_t_5numpy_3fft_5numpy_npy_int64 __pyx_t_5numpy_3fft_5numpy_int64_t;

typedef __pyx_t_5numpy_3fft_5numpy_npy_uint8 __pyx_t_5numpy_3fft_5numpy_uint8_t;

typedef __pyx_t_5numpy_3fft_5numpy_npy_uint16 __pyx_t_5numpy_3fft_5numpy_uint16_t;

typedef __pyx_t_5numpy_3fft_5numpy_npy_uint32 __pyx_t_5numpy_3fft_5numpy_uint32_t;

typedef __pyx_t_5numpy_3fft_5numpy_npy_uint64 __pyx_t_5numpy_3fft_5numpy_uint64_t;

typedef __pyx_t_5numpy_3fft_5numpy_npy_float32 __pyx_t_5numpy_3fft_5numpy_float32_t;

typedef __pyx_t_5numpy_3fft_5numpy_npy_float64 __pyx_t_5numpy_3fft_5numpy_float64_t;
/* Cython code section 'complex_type_declarations' */
/* Cython code section 'type_declarations' */

/* Type declarations */

typedef void (*__pyx_t_5numpy_3fft_5numpy_PyUFuncGenericFunction)(char **, __pyx_t_5numpy_3fft_5numpy_npy_intp *, __pyx_t_5numpy_3fft_5numpy_npy_intp *, void *);
/* Cython code section 'utility_code_proto' */

static CYTHON_INLINE long __Pyx_div_long(long, long); /* proto */
/* Cython code section 'module_declarations' */
/* Module declarations from numpy */
/* Module declarations from numpy.fft.numpy */
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_3fft_5numpy_PyUFunc_FromFuncAndData(__pyx_t_5numpy_3fft_5numpy_PyUFuncGenericFunction *, void **, char *, int, int, int, int, char *, char *, int);
static CYTHON_INLINE System::Object^ PyUFunc_FromFuncAndData(__pyx_t_5numpy_3fft_5numpy_PyUFuncGenericFunction *, void **, char *, int, int, int, int, char *, char *, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_3fft_5numpy_PyArray_ZEROS(int, __pyx_t_5numpy_3fft_5numpy_intp_t *, int, int);
static CYTHON_INLINE System::Object^ PyArray_ZEROS(int, __pyx_t_5numpy_3fft_5numpy_intp_t *, int, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_3fft_5numpy_PyArray_New(void *, int, __pyx_t_5numpy_3fft_5numpy_npy_intp *, int, __pyx_t_5numpy_3fft_5numpy_npy_intp *, void *, int, int, void *);
static CYTHON_INLINE System::Object^ PyArray_New(void *, int, __pyx_t_5numpy_3fft_5numpy_npy_intp *, int, __pyx_t_5numpy_3fft_5numpy_npy_intp *, void *, int, int, void *); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate int __pyx_delegate_t_5numpy_3fft_5numpy_PyArray_CHKFLAGS(NumpyDotNet::ndarray^, int);
static CYTHON_INLINE int PyArray_CHKFLAGS(NumpyDotNet::ndarray^, int); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate void *__pyx_delegate_t_5numpy_3fft_5numpy_PyArray_DATA(NumpyDotNet::ndarray^);
static CYTHON_INLINE void *PyArray_DATA(NumpyDotNet::ndarray^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate __pyx_t_5numpy_3fft_5numpy_intp_t *__pyx_delegate_t_5numpy_3fft_5numpy_PyArray_DIMS(NumpyDotNet::ndarray^);
static CYTHON_INLINE __pyx_t_5numpy_3fft_5numpy_intp_t *PyArray_DIMS(NumpyDotNet::ndarray^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate __pyx_t_5numpy_3fft_5numpy_intp_t __pyx_delegate_t_5numpy_3fft_5numpy_PyArray_SIZE(NumpyDotNet::ndarray^);
static CYTHON_INLINE __pyx_t_5numpy_3fft_5numpy_intp_t PyArray_SIZE(NumpyDotNet::ndarray^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_3fft_5numpy_PyArray_FromAny(System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^);
static CYTHON_INLINE System::Object^ PyArray_FromAny(System::Object^, System::Object^, System::Object^, System::Object^, System::Object^, System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_3fft_5numpy_PyArray_FROMANY(System::Object^, System::Object^, System::Object^, System::Object^, System::Object^);
static CYTHON_INLINE System::Object^ PyArray_FROMANY(System::Object^, System::Object^, System::Object^, System::Object^, System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_3fft_5numpy_PyArray_Check(System::Object^);
static CYTHON_INLINE System::Object^ PyArray_Check(System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_3fft_5numpy_PyArray_NDIM(System::Object^);
static CYTHON_INLINE System::Object^ PyArray_NDIM(System::Object^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate void __pyx_delegate_t_5numpy_3fft_5numpy_import_array(void);
static CYTHON_INLINE void import_array(void); /*proto*/
/* Module declarations from numpy.fft.fftpack_cython */
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_3fft_14fftpack_cython_cfftf(NumpyDotNet::ndarray^, NumpyDotNet::ndarray^);
static System::Object^ cfftf(NumpyDotNet::ndarray^, NumpyDotNet::ndarray^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_3fft_14fftpack_cython_cfftb(NumpyDotNet::ndarray^, NumpyDotNet::ndarray^);
static System::Object^ cfftb(NumpyDotNet::ndarray^, NumpyDotNet::ndarray^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_3fft_14fftpack_cython_cffti(long);
static System::Object^ cffti(long); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_3fft_14fftpack_cython_rfftf(NumpyDotNet::ndarray^, NumpyDotNet::ndarray^);
static System::Object^ rfftf(NumpyDotNet::ndarray^, NumpyDotNet::ndarray^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_3fft_14fftpack_cython_rfftb(NumpyDotNet::ndarray^, NumpyDotNet::ndarray^);
static System::Object^ rfftb(NumpyDotNet::ndarray^, NumpyDotNet::ndarray^); /*proto*/
[InteropServices::UnmanagedFunctionPointer(InteropServices::CallingConvention::Cdecl)]
public delegate System::Object^ __pyx_delegate_t_5numpy_3fft_14fftpack_cython_rffti(long);
static System::Object^ rffti(long); /*proto*/
/* Cython code section 'typeinfo' */
/* Cython code section 'before_global_var' */
#define __Pyx_MODULE_NAME "numpy.fft.fftpack_cython"

/* Implementation of numpy.fft.fftpack_cython */
namespace clr_fftpack_cython {
  public ref class module_fftpack_cython sealed abstract {
/* Cython code section 'global_var' */
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_sub_35_55;
static  CallSite< System::Func< CallSite^, System::Object^, Py_ssize_t >^ >^ __site_cvt_Py_ssize_t_35_55;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_37_19;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_sub_62_55;
static  CallSite< System::Func< CallSite^, System::Object^, Py_ssize_t >^ >^ __site_cvt_Py_ssize_t_62_55;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_64_19;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_sub_101_54;
static  CallSite< System::Func< CallSite^, System::Object^, Py_ssize_t >^ >^ __site_cvt_Py_ssize_t_101_54;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_sub_102_48;
static  CallSite< System::Func< CallSite^, System::Object^, Py_ssize_t >^ >^ __site_cvt_Py_ssize_t_102_48;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_int_103_42;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_sub_104_48;
static  CallSite< System::Func< CallSite^, System::Object^, Py_ssize_t >^ >^ __site_cvt_Py_ssize_t_104_48;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_sub_106_54;
static  CallSite< System::Func< CallSite^, System::Object^, Py_ssize_t >^ >^ __site_cvt_Py_ssize_t_106_54;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_110_19;
static  CallSite< System::Func< CallSite^, System::Object^, int >^ >^ __site_cvt_int_139_42;
static  CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >^ __site_op_sub_141_54;
static  CallSite< System::Func< CallSite^, System::Object^, Py_ssize_t >^ >^ __site_cvt_Py_ssize_t_141_54;
static  CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >^ __site_call1_144_19;
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
/* Cython code section 'dotnet_globals' */


static Types::PythonType^ __pyx_ptype_5numpy_3fft_5numpy_ndarray = nullptr;
static Types::PythonType^ __pyx_ptype_5numpy_3fft_5numpy_dtype = nullptr;

/* Cython code section 'decls' */
static int^ __pyx_int_0;
static int^ __pyx_int_1;
/* Cython code section 'all_the_rest' */
public:
static System::String^ __module__ = __Pyx_MODULE_NAME;

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":22
 *     pass
 * 
 * cdef cfftf(np.ndarray op1, np.ndarray op2):             # <<<<<<<<<<<<<<
 *     cdef double *wsave, *dptr
 *     cdef np.intp_t nsave
 */

static  System::Object^ cfftf(NumpyDotNet::ndarray^ __pyx_v_op1, NumpyDotNet::ndarray^ __pyx_v_op2) {
  double *__pyx_v_wsave;
  double *__pyx_v_dptr;
  __pyx_t_5numpy_3fft_5numpy_intp_t __pyx_v_nsave;
  int __pyx_v_npts;
  int __pyx_v_nrepeats;
  int __pyx_v_i;
  NumpyDotNet::ndarray^ __pyx_v_data;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  int __pyx_t_4;
  Py_ssize_t __pyx_t_5;
  int __pyx_t_6;
  int __pyx_t_7;
  __pyx_v_data = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":28
 *     cdef np.ndarray data
 * 
 *     data = np.PyArray_FROMANY(op1, np.NPY_CDOUBLE, 1, 0, np.NPY_ENSURECOPY|np.NPY_C_CONTIGUOUS)             # <<<<<<<<<<<<<<
 * 
 *     # Force op2 to be contiguous in place of calling PyArray_AsCArray()
 */
  __pyx_t_1 = (System::Object^)(long long)(NPY_CDOUBLE);
  __pyx_t_2 = (System::Object^)(long long)((NPY_ENSURECOPY | NPY_C_CONTIGUOUS));
  __pyx_t_3 = PyArray_FROMANY(((System::Object^)__pyx_v_op1), __pyx_t_1, __pyx_int_1, __pyx_int_0, __pyx_t_2); 
  __pyx_t_1 = nullptr;
  __pyx_t_2 = nullptr;
  if (__pyx_t_3 == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_t_3) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_data = ((NumpyDotNet::ndarray^)__pyx_t_3);
  __pyx_t_3 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":31
 * 
 *     # Force op2 to be contiguous in place of calling PyArray_AsCArray()
 *     if not np.PyArray_CHKFLAGS(op2, np.NPY_CONTIGUOUS):             # <<<<<<<<<<<<<<
 *         op2 = np.PyArray_FROMANY(op2, np.NPY_CDOUBLE, 1, 0, np.NPY_ENSURECOPY|np.NPY_C_CONTIGUOUS)
 * 
 */
  __pyx_t_4 = (!PyArray_CHKFLAGS(__pyx_v_op2, NPY_CONTIGUOUS));
  if (__pyx_t_4) {

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":32
 *     # Force op2 to be contiguous in place of calling PyArray_AsCArray()
 *     if not np.PyArray_CHKFLAGS(op2, np.NPY_CONTIGUOUS):
 *         op2 = np.PyArray_FROMANY(op2, np.NPY_CDOUBLE, 1, 0, np.NPY_ENSURECOPY|np.NPY_C_CONTIGUOUS)             # <<<<<<<<<<<<<<
 * 
 *     nsave = np.PyArray_DIMS(op2)[0]
 */
    __pyx_t_3 = (System::Object^)(long long)(NPY_CDOUBLE);
    __pyx_t_2 = (System::Object^)(long long)((NPY_ENSURECOPY | NPY_C_CONTIGUOUS));
    __pyx_t_1 = PyArray_FROMANY(((System::Object^)__pyx_v_op2), __pyx_t_3, __pyx_int_1, __pyx_int_0, __pyx_t_2); 
    __pyx_t_3 = nullptr;
    __pyx_t_2 = nullptr;
    if (__pyx_t_1 == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_t_1) == nullptr) {
      throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
    }
    __pyx_v_op2 = ((NumpyDotNet::ndarray^)__pyx_t_1);
    __pyx_t_1 = nullptr;
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":34
 *         op2 = np.PyArray_FROMANY(op2, np.NPY_CDOUBLE, 1, 0, np.NPY_ENSURECOPY|np.NPY_C_CONTIGUOUS)
 * 
 *     nsave = np.PyArray_DIMS(op2)[0]             # <<<<<<<<<<<<<<
 *     npts = np.PyArray_DIMS(data)[np.PyArray_NDIM(data) - 1]
 *     if nsave != npts*4 + 15:
 */
  __pyx_v_nsave = (PyArray_DIMS(__pyx_v_op2)[0]);

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":35
 * 
 *     nsave = np.PyArray_DIMS(op2)[0]
 *     npts = np.PyArray_DIMS(data)[np.PyArray_NDIM(data) - 1]             # <<<<<<<<<<<<<<
 *     if nsave != npts*4 + 15:
 *         raise error("invalid work array for fft size")
 */
  __pyx_t_1 = PyArray_NDIM(((System::Object^)__pyx_v_data)); 
  __pyx_t_2 = __site_op_sub_35_55->Target(__site_op_sub_35_55, __pyx_t_1, __pyx_int_1);
  __pyx_t_1 = nullptr;
  __pyx_t_5 = __site_cvt_Py_ssize_t_35_55->Target(__site_cvt_Py_ssize_t_35_55, __pyx_t_2);
  __pyx_t_2 = nullptr;
  __pyx_v_npts = (PyArray_DIMS(__pyx_v_data)[__pyx_t_5]);

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":36
 *     nsave = np.PyArray_DIMS(op2)[0]
 *     npts = np.PyArray_DIMS(data)[np.PyArray_NDIM(data) - 1]
 *     if nsave != npts*4 + 15:             # <<<<<<<<<<<<<<
 *         raise error("invalid work array for fft size")
 * 
 */
  __pyx_t_4 = (__pyx_v_nsave != ((__pyx_v_npts * 4) + 15));
  if (__pyx_t_4) {

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":37
 *     npts = np.PyArray_DIMS(data)[np.PyArray_NDIM(data) - 1]
 *     if nsave != npts*4 + 15:
 *         raise error("invalid work array for fft size")             # <<<<<<<<<<<<<<
 * 
 *     nrepeats = np.PyArray_SIZE(data)
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "error");
    __pyx_t_1 = __site_call1_37_19->Target(__site_call1_37_19, __pyx_context, __pyx_t_2, ((System::Object^)"invalid work array for fft size"));
    __pyx_t_2 = nullptr;
    throw PythonOps::MakeException(__pyx_context, __pyx_t_1, nullptr, nullptr);
    __pyx_t_1 = nullptr;
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":39
 *         raise error("invalid work array for fft size")
 * 
 *     nrepeats = np.PyArray_SIZE(data)             # <<<<<<<<<<<<<<
 *     nrepeats /= npts
 *     dptr = <double *>np.PyArray_DATA(data)
 */
  __pyx_v_nrepeats = PyArray_SIZE(__pyx_v_data);

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":40
 * 
 *     nrepeats = np.PyArray_SIZE(data)
 *     nrepeats /= npts             # <<<<<<<<<<<<<<
 *     dptr = <double *>np.PyArray_DATA(data)
 *     wsave = <double *>np.PyArray_DATA(op2)
 */
  __pyx_v_nrepeats /= __pyx_v_npts;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":41
 *     nrepeats = np.PyArray_SIZE(data)
 *     nrepeats /= npts
 *     dptr = <double *>np.PyArray_DATA(data)             # <<<<<<<<<<<<<<
 *     wsave = <double *>np.PyArray_DATA(op2)
 * 
 */
  __pyx_v_dptr = ((double *)PyArray_DATA(__pyx_v_data));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":42
 *     nrepeats /= npts
 *     dptr = <double *>np.PyArray_DATA(data)
 *     wsave = <double *>np.PyArray_DATA(op2)             # <<<<<<<<<<<<<<
 * 
 *     for i in range(nrepeats):
 */
  __pyx_v_wsave = ((double *)PyArray_DATA(__pyx_v_op2));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":44
 *     wsave = <double *>np.PyArray_DATA(op2)
 * 
 *     for i in range(nrepeats):             # <<<<<<<<<<<<<<
 *         fftpack_cfftf(npts, dptr, wsave)
 *         dptr += npts*2
 */
  __pyx_t_6 = __pyx_v_nrepeats;
  for (__pyx_t_7 = 0; __pyx_t_7 < __pyx_t_6; __pyx_t_7+=1) {
    __pyx_v_i = __pyx_t_7;

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":45
 * 
 *     for i in range(nrepeats):
 *         fftpack_cfftf(npts, dptr, wsave)             # <<<<<<<<<<<<<<
 *         dptr += npts*2
 * 
 */
    GLOBALFUNC(cfftf)(__pyx_v_npts, __pyx_v_dptr, __pyx_v_wsave);

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":46
 *     for i in range(nrepeats):
 *         fftpack_cfftf(npts, dptr, wsave)
 *         dptr += npts*2             # <<<<<<<<<<<<<<
 * 
 *     return data
 */
    __pyx_v_dptr += (__pyx_v_npts * 2);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":48
 *         dptr += npts*2
 * 
 *     return data             # <<<<<<<<<<<<<<
 * 
 * cdef cfftb(np.ndarray op1, np.ndarray op2):
 */
  __pyx_r = ((System::Object^)__pyx_v_data);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":50
 *     return data
 * 
 * cdef cfftb(np.ndarray op1, np.ndarray op2):             # <<<<<<<<<<<<<<
 *     cdef double *wsave, *dptr
 *     cdef np.intp_t nsave
 */

static  System::Object^ cfftb(NumpyDotNet::ndarray^ __pyx_v_op1, NumpyDotNet::ndarray^ __pyx_v_op2) {
  double *__pyx_v_wsave;
  double *__pyx_v_dptr;
  __pyx_t_5numpy_3fft_5numpy_intp_t __pyx_v_nsave;
  int __pyx_v_npts;
  int __pyx_v_nrepeats;
  int __pyx_v_i;
  System::Object^ __pyx_v_data;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  int __pyx_t_4;
  Py_ssize_t __pyx_t_5;
  int __pyx_t_6;
  int __pyx_t_7;
  __pyx_v_data = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":55
 *     cdef int npts, nrepeats, i
 * 
 *     data = np.PyArray_FROMANY(op1, np.NPY_CDOUBLE, 1, 0, np.NPY_ENSURECOPY|np.NPY_C_CONTIGUOUS)             # <<<<<<<<<<<<<<
 * 
 *     # Force op2 to be contiguous in place of calling PyArray_AsCArray()
 */
  __pyx_t_1 = (System::Object^)(long long)(NPY_CDOUBLE);
  __pyx_t_2 = (System::Object^)(long long)((NPY_ENSURECOPY | NPY_C_CONTIGUOUS));
  __pyx_t_3 = PyArray_FROMANY(((System::Object^)__pyx_v_op1), __pyx_t_1, __pyx_int_1, __pyx_int_0, __pyx_t_2); 
  __pyx_t_1 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_v_data = __pyx_t_3;
  __pyx_t_3 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":58
 * 
 *     # Force op2 to be contiguous in place of calling PyArray_AsCArray()
 *     if not np.PyArray_CHKFLAGS(op2, np.NPY_CONTIGUOUS):             # <<<<<<<<<<<<<<
 *         op2 = np.PyArray_FROMANY(op2, np.NPY_CDOUBLE, 1, 0, np.NPY_ENSURECOPY|np.NPY_C_CONTIGUOUS)
 * 
 */
  __pyx_t_4 = (!PyArray_CHKFLAGS(__pyx_v_op2, NPY_CONTIGUOUS));
  if (__pyx_t_4) {

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":59
 *     # Force op2 to be contiguous in place of calling PyArray_AsCArray()
 *     if not np.PyArray_CHKFLAGS(op2, np.NPY_CONTIGUOUS):
 *         op2 = np.PyArray_FROMANY(op2, np.NPY_CDOUBLE, 1, 0, np.NPY_ENSURECOPY|np.NPY_C_CONTIGUOUS)             # <<<<<<<<<<<<<<
 * 
 *     nsave = np.PyArray_DIMS(op2)[0]
 */
    __pyx_t_3 = (System::Object^)(long long)(NPY_CDOUBLE);
    __pyx_t_2 = (System::Object^)(long long)((NPY_ENSURECOPY | NPY_C_CONTIGUOUS));
    __pyx_t_1 = PyArray_FROMANY(((System::Object^)__pyx_v_op2), __pyx_t_3, __pyx_int_1, __pyx_int_0, __pyx_t_2); 
    __pyx_t_3 = nullptr;
    __pyx_t_2 = nullptr;
    if (__pyx_t_1 == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_t_1) == nullptr) {
      throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
    }
    __pyx_v_op2 = ((NumpyDotNet::ndarray^)__pyx_t_1);
    __pyx_t_1 = nullptr;
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":61
 *         op2 = np.PyArray_FROMANY(op2, np.NPY_CDOUBLE, 1, 0, np.NPY_ENSURECOPY|np.NPY_C_CONTIGUOUS)
 * 
 *     nsave = np.PyArray_DIMS(op2)[0]             # <<<<<<<<<<<<<<
 *     npts = np.PyArray_DIMS(data)[np.PyArray_NDIM(data) - 1]
 *     if nsave != npts*4 + 15:
 */
  __pyx_v_nsave = (PyArray_DIMS(__pyx_v_op2)[0]);

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":62
 * 
 *     nsave = np.PyArray_DIMS(op2)[0]
 *     npts = np.PyArray_DIMS(data)[np.PyArray_NDIM(data) - 1]             # <<<<<<<<<<<<<<
 *     if nsave != npts*4 + 15:
 *         raise error("invalid work array for fft size")
 */
  if (__pyx_v_data == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_data) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_t_1 = PyArray_NDIM(__pyx_v_data); 
  __pyx_t_2 = __site_op_sub_62_55->Target(__site_op_sub_62_55, __pyx_t_1, __pyx_int_1);
  __pyx_t_1 = nullptr;
  __pyx_t_5 = __site_cvt_Py_ssize_t_62_55->Target(__site_cvt_Py_ssize_t_62_55, __pyx_t_2);
  __pyx_t_2 = nullptr;
  __pyx_v_npts = (PyArray_DIMS(((NumpyDotNet::ndarray^)__pyx_v_data))[__pyx_t_5]);

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":63
 *     nsave = np.PyArray_DIMS(op2)[0]
 *     npts = np.PyArray_DIMS(data)[np.PyArray_NDIM(data) - 1]
 *     if nsave != npts*4 + 15:             # <<<<<<<<<<<<<<
 *         raise error("invalid work array for fft size")
 * 
 */
  __pyx_t_4 = (__pyx_v_nsave != ((__pyx_v_npts * 4) + 15));
  if (__pyx_t_4) {

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":64
 *     npts = np.PyArray_DIMS(data)[np.PyArray_NDIM(data) - 1]
 *     if nsave != npts*4 + 15:
 *         raise error("invalid work array for fft size")             # <<<<<<<<<<<<<<
 * 
 *     nrepeats = np.PyArray_SIZE(data)
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "error");
    __pyx_t_1 = __site_call1_64_19->Target(__site_call1_64_19, __pyx_context, __pyx_t_2, ((System::Object^)"invalid work array for fft size"));
    __pyx_t_2 = nullptr;
    throw PythonOps::MakeException(__pyx_context, __pyx_t_1, nullptr, nullptr);
    __pyx_t_1 = nullptr;
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":66
 *         raise error("invalid work array for fft size")
 * 
 *     nrepeats = np.PyArray_SIZE(data)             # <<<<<<<<<<<<<<
 *     nrepeats /= npts
 *     dptr = <double *>np.PyArray_DATA(data)
 */
  if (__pyx_v_data == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_data) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_nrepeats = PyArray_SIZE(((NumpyDotNet::ndarray^)__pyx_v_data));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":67
 * 
 *     nrepeats = np.PyArray_SIZE(data)
 *     nrepeats /= npts             # <<<<<<<<<<<<<<
 *     dptr = <double *>np.PyArray_DATA(data)
 *     wsave = <double *>np.PyArray_DATA(op2)
 */
  __pyx_v_nrepeats /= __pyx_v_npts;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":68
 *     nrepeats = np.PyArray_SIZE(data)
 *     nrepeats /= npts
 *     dptr = <double *>np.PyArray_DATA(data)             # <<<<<<<<<<<<<<
 *     wsave = <double *>np.PyArray_DATA(op2)
 * 
 */
  if (__pyx_v_data == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_data) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_dptr = ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_data)));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":69
 *     nrepeats /= npts
 *     dptr = <double *>np.PyArray_DATA(data)
 *     wsave = <double *>np.PyArray_DATA(op2)             # <<<<<<<<<<<<<<
 * 
 *     for i in range(nrepeats):
 */
  __pyx_v_wsave = ((double *)PyArray_DATA(__pyx_v_op2));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":71
 *     wsave = <double *>np.PyArray_DATA(op2)
 * 
 *     for i in range(nrepeats):             # <<<<<<<<<<<<<<
 *         fftpack_cfftb(npts, dptr, wsave)
 *         dptr += npts*2
 */
  __pyx_t_6 = __pyx_v_nrepeats;
  for (__pyx_t_7 = 0; __pyx_t_7 < __pyx_t_6; __pyx_t_7+=1) {
    __pyx_v_i = __pyx_t_7;

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":72
 * 
 *     for i in range(nrepeats):
 *         fftpack_cfftb(npts, dptr, wsave)             # <<<<<<<<<<<<<<
 *         dptr += npts*2
 * 
 */
    GLOBALFUNC(cfftb)(__pyx_v_npts, __pyx_v_dptr, __pyx_v_wsave);

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":73
 *     for i in range(nrepeats):
 *         fftpack_cfftb(npts, dptr, wsave)
 *         dptr += npts*2             # <<<<<<<<<<<<<<
 * 
 *     return data
 */
    __pyx_v_dptr += (__pyx_v_npts * 2);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":75
 *         dptr += npts*2
 * 
 *     return data             # <<<<<<<<<<<<<<
 * 
 * cdef cffti(long n):
 */
  __pyx_r = __pyx_v_data;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":77
 *     return data
 * 
 * cdef cffti(long n):             # <<<<<<<<<<<<<<
 *     cdef np.intp_t dim
 *     cdef np.ndarray op
 */

static  System::Object^ cffti(long __pyx_v_n) {
  __pyx_t_5numpy_3fft_5numpy_intp_t __pyx_v_dim;
  NumpyDotNet::ndarray^ __pyx_v_op;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  __pyx_v_op = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":82
 * 
 *     # Magic size needed by cffti
 *     dim = 4*n + 15;             # <<<<<<<<<<<<<<
 *     # Create a 1 dimensional array of dimensions of type double
 *     op = np.PyArray_New(NULL, 1, &dim, np.NPY_DOUBLE, NULL, NULL, 0, 0, NULL)
 */
  __pyx_v_dim = ((4 * __pyx_v_n) + 15);

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":84
 *     dim = 4*n + 15;
 *     # Create a 1 dimensional array of dimensions of type double
 *     op = np.PyArray_New(NULL, 1, &dim, np.NPY_DOUBLE, NULL, NULL, 0, 0, NULL)             # <<<<<<<<<<<<<<
 * 
 *     fftpack_cffti(n, <double *>np.PyArray_DATA(op))
 */
  __pyx_t_1 = PyArray_New(NULL, 1, (&__pyx_v_dim), NPY_DOUBLE, NULL, NULL, 0, 0, NULL); 
  if (__pyx_t_1 == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_t_1) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_op = ((NumpyDotNet::ndarray^)__pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":86
 *     op = np.PyArray_New(NULL, 1, &dim, np.NPY_DOUBLE, NULL, NULL, 0, 0, NULL)
 * 
 *     fftpack_cffti(n, <double *>np.PyArray_DATA(op))             # <<<<<<<<<<<<<<
 * 
 *     return op
 */
  GLOBALFUNC(cffti)(__pyx_v_n, ((double *)PyArray_DATA(__pyx_v_op)));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":88
 *     fftpack_cffti(n, <double *>np.PyArray_DATA(op))
 * 
 *     return op             # <<<<<<<<<<<<<<
 * 
 * cdef rfftf(np.ndarray op1, np.ndarray op2):
 */
  __pyx_r = ((System::Object^)__pyx_v_op);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":90
 *     return op
 * 
 * cdef rfftf(np.ndarray op1, np.ndarray op2):             # <<<<<<<<<<<<<<
 *     cdef double *wsave, *dptr, *rptr
 *     cdef np.intp_t nsave
 */

static  System::Object^ rfftf(NumpyDotNet::ndarray^ __pyx_v_op1, NumpyDotNet::ndarray^ __pyx_v_op2) {
  double *__pyx_v_wsave;
  double *__pyx_v_dptr;
  double *__pyx_v_rptr;
  __pyx_t_5numpy_3fft_5numpy_intp_t __pyx_v_nsave;
  int __pyx_v_npts;
  int __pyx_v_nrepeats;
  int __pyx_v_rstep;
  int __pyx_v_i;
  System::Object^ __pyx_v_data;
  System::Object^ __pyx_v_ret;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  int __pyx_t_4;
  Py_ssize_t __pyx_t_5;
  Py_ssize_t __pyx_t_6;
  int __pyx_t_7;
  Py_ssize_t __pyx_t_8;
  Py_ssize_t __pyx_t_9;
  int __pyx_t_10;
  int __pyx_t_11;
  __pyx_v_data = nullptr;
  __pyx_v_ret = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":95
 *     cdef int npts, nrepeats, rstep, i
 * 
 *     data = np.PyArray_FROMANY(op1, np.NPY_DOUBLE, 1, 0, np.NPY_C_CONTIGUOUS)             # <<<<<<<<<<<<<<
 * 
 *     # Force op2 to be contiguous in place of calling PyArray_AsCArray()
 */
  __pyx_t_1 = (System::Object^)(long long)(NPY_DOUBLE);
  __pyx_t_2 = (System::Object^)(long long)(NPY_C_CONTIGUOUS);
  __pyx_t_3 = PyArray_FROMANY(((System::Object^)__pyx_v_op1), __pyx_t_1, __pyx_int_1, __pyx_int_0, __pyx_t_2); 
  __pyx_t_1 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_v_data = __pyx_t_3;
  __pyx_t_3 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":98
 * 
 *     # Force op2 to be contiguous in place of calling PyArray_AsCArray()
 *     if not np.PyArray_CHKFLAGS(op2, np.NPY_CONTIGUOUS):             # <<<<<<<<<<<<<<
 *         op2 = np.PyArray_FROMANY(op2, np.NPY_DOUBLE, 1, 0, np.NPY_ENSURECOPY|np.NPY_C_CONTIGUOUS)
 * 
 */
  __pyx_t_4 = (!PyArray_CHKFLAGS(__pyx_v_op2, NPY_CONTIGUOUS));
  if (__pyx_t_4) {

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":99
 *     # Force op2 to be contiguous in place of calling PyArray_AsCArray()
 *     if not np.PyArray_CHKFLAGS(op2, np.NPY_CONTIGUOUS):
 *         op2 = np.PyArray_FROMANY(op2, np.NPY_DOUBLE, 1, 0, np.NPY_ENSURECOPY|np.NPY_C_CONTIGUOUS)             # <<<<<<<<<<<<<<
 * 
 *     npts = np.PyArray_DIMS(data)[np.PyArray_NDIM(data)-1]
 */
    __pyx_t_3 = (System::Object^)(long long)(NPY_DOUBLE);
    __pyx_t_2 = (System::Object^)(long long)((NPY_ENSURECOPY | NPY_C_CONTIGUOUS));
    __pyx_t_1 = PyArray_FROMANY(((System::Object^)__pyx_v_op2), __pyx_t_3, __pyx_int_1, __pyx_int_0, __pyx_t_2); 
    __pyx_t_3 = nullptr;
    __pyx_t_2 = nullptr;
    if (__pyx_t_1 == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_t_1) == nullptr) {
      throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
    }
    __pyx_v_op2 = ((NumpyDotNet::ndarray^)__pyx_t_1);
    __pyx_t_1 = nullptr;
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":101
 *         op2 = np.PyArray_FROMANY(op2, np.NPY_DOUBLE, 1, 0, np.NPY_ENSURECOPY|np.NPY_C_CONTIGUOUS)
 * 
 *     npts = np.PyArray_DIMS(data)[np.PyArray_NDIM(data)-1]             # <<<<<<<<<<<<<<
 *     np.PyArray_DIMS(data)[np.PyArray_NDIM(data) - 1] = npts/2 + 1
 *     ret = np.PyArray_ZEROS(np.PyArray_NDIM(data), np.PyArray_DIMS(data), np.NPY_CDOUBLE, 0)
 */
  if (__pyx_v_data == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_data) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_t_1 = PyArray_NDIM(__pyx_v_data); 
  __pyx_t_2 = __site_op_sub_101_54->Target(__site_op_sub_101_54, __pyx_t_1, __pyx_int_1);
  __pyx_t_1 = nullptr;
  __pyx_t_5 = __site_cvt_Py_ssize_t_101_54->Target(__site_cvt_Py_ssize_t_101_54, __pyx_t_2);
  __pyx_t_2 = nullptr;
  __pyx_v_npts = (PyArray_DIMS(((NumpyDotNet::ndarray^)__pyx_v_data))[__pyx_t_5]);

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":102
 * 
 *     npts = np.PyArray_DIMS(data)[np.PyArray_NDIM(data)-1]
 *     np.PyArray_DIMS(data)[np.PyArray_NDIM(data) - 1] = npts/2 + 1             # <<<<<<<<<<<<<<
 *     ret = np.PyArray_ZEROS(np.PyArray_NDIM(data), np.PyArray_DIMS(data), np.NPY_CDOUBLE, 0)
 *     np.PyArray_DIMS(data)[np.PyArray_NDIM(data) - 1] = npts
 */
  if (__pyx_v_data == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_data) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_t_2 = PyArray_NDIM(__pyx_v_data); 
  __pyx_t_1 = __site_op_sub_102_48->Target(__site_op_sub_102_48, __pyx_t_2, __pyx_int_1);
  __pyx_t_2 = nullptr;
  __pyx_t_6 = __site_cvt_Py_ssize_t_102_48->Target(__site_cvt_Py_ssize_t_102_48, __pyx_t_1);
  __pyx_t_1 = nullptr;
  (PyArray_DIMS(((NumpyDotNet::ndarray^)__pyx_v_data))[__pyx_t_6]) = (__Pyx_div_long(__pyx_v_npts, 2) + 1);

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":103
 *     npts = np.PyArray_DIMS(data)[np.PyArray_NDIM(data)-1]
 *     np.PyArray_DIMS(data)[np.PyArray_NDIM(data) - 1] = npts/2 + 1
 *     ret = np.PyArray_ZEROS(np.PyArray_NDIM(data), np.PyArray_DIMS(data), np.NPY_CDOUBLE, 0)             # <<<<<<<<<<<<<<
 *     np.PyArray_DIMS(data)[np.PyArray_NDIM(data) - 1] = npts
 * 
 */
  __pyx_t_1 = PyArray_NDIM(__pyx_v_data); 
  __pyx_t_7 = __site_cvt_int_103_42->Target(__site_cvt_int_103_42, __pyx_t_1);
  __pyx_t_1 = nullptr;
  if (__pyx_v_data == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_data) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_t_1 = PyArray_ZEROS(__pyx_t_7, PyArray_DIMS(((NumpyDotNet::ndarray^)__pyx_v_data)), NPY_CDOUBLE, 0); 
  __pyx_v_ret = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":104
 *     np.PyArray_DIMS(data)[np.PyArray_NDIM(data) - 1] = npts/2 + 1
 *     ret = np.PyArray_ZEROS(np.PyArray_NDIM(data), np.PyArray_DIMS(data), np.NPY_CDOUBLE, 0)
 *     np.PyArray_DIMS(data)[np.PyArray_NDIM(data) - 1] = npts             # <<<<<<<<<<<<<<
 * 
 *     rstep = np.PyArray_DIMS(ret)[np.PyArray_NDIM(ret) - 1]*2
 */
  if (__pyx_v_data == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_data) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_t_1 = PyArray_NDIM(__pyx_v_data); 
  __pyx_t_2 = __site_op_sub_104_48->Target(__site_op_sub_104_48, __pyx_t_1, __pyx_int_1);
  __pyx_t_1 = nullptr;
  __pyx_t_8 = __site_cvt_Py_ssize_t_104_48->Target(__site_cvt_Py_ssize_t_104_48, __pyx_t_2);
  __pyx_t_2 = nullptr;
  (PyArray_DIMS(((NumpyDotNet::ndarray^)__pyx_v_data))[__pyx_t_8]) = __pyx_v_npts;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":106
 *     np.PyArray_DIMS(data)[np.PyArray_NDIM(data) - 1] = npts
 * 
 *     rstep = np.PyArray_DIMS(ret)[np.PyArray_NDIM(ret) - 1]*2             # <<<<<<<<<<<<<<
 * 
 *     nsave = np.PyArray_DIMS(op2)[0]
 */
  if (__pyx_v_ret == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_ret) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_t_2 = PyArray_NDIM(__pyx_v_ret); 
  __pyx_t_1 = __site_op_sub_106_54->Target(__site_op_sub_106_54, __pyx_t_2, __pyx_int_1);
  __pyx_t_2 = nullptr;
  __pyx_t_9 = __site_cvt_Py_ssize_t_106_54->Target(__site_cvt_Py_ssize_t_106_54, __pyx_t_1);
  __pyx_t_1 = nullptr;
  __pyx_v_rstep = ((PyArray_DIMS(((NumpyDotNet::ndarray^)__pyx_v_ret))[__pyx_t_9]) * 2);

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":108
 *     rstep = np.PyArray_DIMS(ret)[np.PyArray_NDIM(ret) - 1]*2
 * 
 *     nsave = np.PyArray_DIMS(op2)[0]             # <<<<<<<<<<<<<<
 *     if nsave != npts*2+15:
 *         raise error("invalid work array for fft size")
 */
  __pyx_v_nsave = (PyArray_DIMS(__pyx_v_op2)[0]);

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":109
 * 
 *     nsave = np.PyArray_DIMS(op2)[0]
 *     if nsave != npts*2+15:             # <<<<<<<<<<<<<<
 *         raise error("invalid work array for fft size")
 * 
 */
  __pyx_t_4 = (__pyx_v_nsave != ((__pyx_v_npts * 2) + 15));
  if (__pyx_t_4) {

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":110
 *     nsave = np.PyArray_DIMS(op2)[0]
 *     if nsave != npts*2+15:
 *         raise error("invalid work array for fft size")             # <<<<<<<<<<<<<<
 * 
 *     nrepeats = np.PyArray_SIZE(data)
 */
    __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "error");
    __pyx_t_2 = __site_call1_110_19->Target(__site_call1_110_19, __pyx_context, __pyx_t_1, ((System::Object^)"invalid work array for fft size"));
    __pyx_t_1 = nullptr;
    throw PythonOps::MakeException(__pyx_context, __pyx_t_2, nullptr, nullptr);
    __pyx_t_2 = nullptr;
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":112
 *         raise error("invalid work array for fft size")
 * 
 *     nrepeats = np.PyArray_SIZE(data)             # <<<<<<<<<<<<<<
 *     nrepeats /= npts
 *     rptr = <double *>np.PyArray_DATA(ret)
 */
  if (__pyx_v_data == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_data) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_nrepeats = PyArray_SIZE(((NumpyDotNet::ndarray^)__pyx_v_data));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":113
 * 
 *     nrepeats = np.PyArray_SIZE(data)
 *     nrepeats /= npts             # <<<<<<<<<<<<<<
 *     rptr = <double *>np.PyArray_DATA(ret)
 *     dptr = <double *>np.PyArray_DATA(data)
 */
  __pyx_v_nrepeats /= __pyx_v_npts;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":114
 *     nrepeats = np.PyArray_SIZE(data)
 *     nrepeats /= npts
 *     rptr = <double *>np.PyArray_DATA(ret)             # <<<<<<<<<<<<<<
 *     dptr = <double *>np.PyArray_DATA(data)
 *     wsave = <double *>np.PyArray_DATA(op2)
 */
  if (__pyx_v_ret == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_ret) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_rptr = ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_ret)));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":115
 *     nrepeats /= npts
 *     rptr = <double *>np.PyArray_DATA(ret)
 *     dptr = <double *>np.PyArray_DATA(data)             # <<<<<<<<<<<<<<
 *     wsave = <double *>np.PyArray_DATA(op2)
 * 
 */
  if (__pyx_v_data == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_data) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_dptr = ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_data)));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":116
 *     rptr = <double *>np.PyArray_DATA(ret)
 *     dptr = <double *>np.PyArray_DATA(data)
 *     wsave = <double *>np.PyArray_DATA(op2)             # <<<<<<<<<<<<<<
 * 
 *     for i in range(nrepeats):
 */
  __pyx_v_wsave = ((double *)PyArray_DATA(__pyx_v_op2));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":118
 *     wsave = <double *>np.PyArray_DATA(op2)
 * 
 *     for i in range(nrepeats):             # <<<<<<<<<<<<<<
 *         memcpy(<char *>(rptr+1), dptr, npts*sizeof(double))
 *         fftpack_rfftf(npts, rptr+1, wsave)
 */
  __pyx_t_10 = __pyx_v_nrepeats;
  for (__pyx_t_11 = 0; __pyx_t_11 < __pyx_t_10; __pyx_t_11+=1) {
    __pyx_v_i = __pyx_t_11;

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":119
 * 
 *     for i in range(nrepeats):
 *         memcpy(<char *>(rptr+1), dptr, npts*sizeof(double))             # <<<<<<<<<<<<<<
 *         fftpack_rfftf(npts, rptr+1, wsave)
 *         rptr[0] = rptr[1]
 */
    memcpy(((char *)(__pyx_v_rptr + 1)), __pyx_v_dptr, (__pyx_v_npts * (sizeof(double))));

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":120
 *     for i in range(nrepeats):
 *         memcpy(<char *>(rptr+1), dptr, npts*sizeof(double))
 *         fftpack_rfftf(npts, rptr+1, wsave)             # <<<<<<<<<<<<<<
 *         rptr[0] = rptr[1]
 *         rptr[1] = 0.0
 */
    GLOBALFUNC(rfftf)(__pyx_v_npts, (__pyx_v_rptr + 1), __pyx_v_wsave);

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":121
 *         memcpy(<char *>(rptr+1), dptr, npts*sizeof(double))
 *         fftpack_rfftf(npts, rptr+1, wsave)
 *         rptr[0] = rptr[1]             # <<<<<<<<<<<<<<
 *         rptr[1] = 0.0
 *         rptr += rstep
 */
    (__pyx_v_rptr[0]) = (__pyx_v_rptr[1]);

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":122
 *         fftpack_rfftf(npts, rptr+1, wsave)
 *         rptr[0] = rptr[1]
 *         rptr[1] = 0.0             # <<<<<<<<<<<<<<
 *         rptr += rstep
 *         dptr += npts
 */
    (__pyx_v_rptr[1]) = 0.0;

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":123
 *         rptr[0] = rptr[1]
 *         rptr[1] = 0.0
 *         rptr += rstep             # <<<<<<<<<<<<<<
 *         dptr += npts
 * 
 */
    __pyx_v_rptr += __pyx_v_rstep;

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":124
 *         rptr[1] = 0.0
 *         rptr += rstep
 *         dptr += npts             # <<<<<<<<<<<<<<
 * 
 *     return ret
 */
    __pyx_v_dptr += __pyx_v_npts;
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":126
 *         dptr += npts
 * 
 *     return ret             # <<<<<<<<<<<<<<
 * 
 * cdef rfftb(np.ndarray op1, np.ndarray op2):
 */
  __pyx_r = __pyx_v_ret;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":128
 *     return ret
 * 
 * cdef rfftb(np.ndarray op1, np.ndarray op2):             # <<<<<<<<<<<<<<
 *     cdef double *wsave, *dptr, *rptr
 *     cdef np.intp_t nsave
 */

static  System::Object^ rfftb(NumpyDotNet::ndarray^ __pyx_v_op1, NumpyDotNet::ndarray^ __pyx_v_op2) {
  double *__pyx_v_wsave;
  double *__pyx_v_dptr;
  double *__pyx_v_rptr;
  __pyx_t_5numpy_3fft_5numpy_intp_t __pyx_v_nsave;
  int __pyx_v_npts;
  int __pyx_v_nrepeats;
  int __pyx_v_i;
  System::Object^ __pyx_v_data;
  System::Object^ __pyx_v_ret;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  System::Object^ __pyx_t_2 = nullptr;
  System::Object^ __pyx_t_3 = nullptr;
  int __pyx_t_4;
  int __pyx_t_5;
  Py_ssize_t __pyx_t_6;
  int __pyx_t_7;
  int __pyx_t_8;
  __pyx_v_data = nullptr;
  __pyx_v_ret = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":133
 *     cdef int npts, nrepeats, i
 * 
 *     data = np.PyArray_FROMANY(op1, np.NPY_CDOUBLE, 1, 0, np.NPY_C_CONTIGUOUS)             # <<<<<<<<<<<<<<
 * 
 *      # Force op2 to be contiguous in place of calling PyArray_AsCArray()
 */
  __pyx_t_1 = (System::Object^)(long long)(NPY_CDOUBLE);
  __pyx_t_2 = (System::Object^)(long long)(NPY_C_CONTIGUOUS);
  __pyx_t_3 = PyArray_FROMANY(((System::Object^)__pyx_v_op1), __pyx_t_1, __pyx_int_1, __pyx_int_0, __pyx_t_2); 
  __pyx_t_1 = nullptr;
  __pyx_t_2 = nullptr;
  __pyx_v_data = __pyx_t_3;
  __pyx_t_3 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":136
 * 
 *      # Force op2 to be contiguous in place of calling PyArray_AsCArray()
 *     if not np.PyArray_CHKFLAGS(op2, np.NPY_CONTIGUOUS):             # <<<<<<<<<<<<<<
 *         op2 = np.PyArray_FROMANY(op2, np.NPY_DOUBLE, 1, 0, np.NPY_ENSURECOPY|np.NPY_C_CONTIGUOUS)
 * 
 */
  __pyx_t_4 = (!PyArray_CHKFLAGS(__pyx_v_op2, NPY_CONTIGUOUS));
  if (__pyx_t_4) {

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":137
 *      # Force op2 to be contiguous in place of calling PyArray_AsCArray()
 *     if not np.PyArray_CHKFLAGS(op2, np.NPY_CONTIGUOUS):
 *         op2 = np.PyArray_FROMANY(op2, np.NPY_DOUBLE, 1, 0, np.NPY_ENSURECOPY|np.NPY_C_CONTIGUOUS)             # <<<<<<<<<<<<<<
 * 
 *     ret = np.PyArray_ZEROS(np.PyArray_NDIM(data), np.PyArray_DIMS(data), np.NPY_DOUBLE, 0)
 */
    __pyx_t_3 = (System::Object^)(long long)(NPY_DOUBLE);
    __pyx_t_2 = (System::Object^)(long long)((NPY_ENSURECOPY | NPY_C_CONTIGUOUS));
    __pyx_t_1 = PyArray_FROMANY(((System::Object^)__pyx_v_op2), __pyx_t_3, __pyx_int_1, __pyx_int_0, __pyx_t_2); 
    __pyx_t_3 = nullptr;
    __pyx_t_2 = nullptr;
    if (__pyx_t_1 == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_t_1) == nullptr) {
      throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
    }
    __pyx_v_op2 = ((NumpyDotNet::ndarray^)__pyx_t_1);
    __pyx_t_1 = nullptr;
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":139
 *         op2 = np.PyArray_FROMANY(op2, np.NPY_DOUBLE, 1, 0, np.NPY_ENSURECOPY|np.NPY_C_CONTIGUOUS)
 * 
 *     ret = np.PyArray_ZEROS(np.PyArray_NDIM(data), np.PyArray_DIMS(data), np.NPY_DOUBLE, 0)             # <<<<<<<<<<<<<<
 * 
 *     npts = np.PyArray_DIMS(data)[np.PyArray_NDIM(data)-1]
 */
  __pyx_t_1 = PyArray_NDIM(__pyx_v_data); 
  __pyx_t_5 = __site_cvt_int_139_42->Target(__site_cvt_int_139_42, __pyx_t_1);
  __pyx_t_1 = nullptr;
  if (__pyx_v_data == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_data) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_t_1 = PyArray_ZEROS(__pyx_t_5, PyArray_DIMS(((NumpyDotNet::ndarray^)__pyx_v_data)), NPY_DOUBLE, 0); 
  __pyx_v_ret = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":141
 *     ret = np.PyArray_ZEROS(np.PyArray_NDIM(data), np.PyArray_DIMS(data), np.NPY_DOUBLE, 0)
 * 
 *     npts = np.PyArray_DIMS(data)[np.PyArray_NDIM(data)-1]             # <<<<<<<<<<<<<<
 *     nsave = np.PyArray_DIMS(op2)[0]
 *     if nsave != npts*2 + 15:
 */
  if (__pyx_v_data == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_data) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_t_1 = PyArray_NDIM(__pyx_v_data); 
  __pyx_t_2 = __site_op_sub_141_54->Target(__site_op_sub_141_54, __pyx_t_1, __pyx_int_1);
  __pyx_t_1 = nullptr;
  __pyx_t_6 = __site_cvt_Py_ssize_t_141_54->Target(__site_cvt_Py_ssize_t_141_54, __pyx_t_2);
  __pyx_t_2 = nullptr;
  __pyx_v_npts = (PyArray_DIMS(((NumpyDotNet::ndarray^)__pyx_v_data))[__pyx_t_6]);

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":142
 * 
 *     npts = np.PyArray_DIMS(data)[np.PyArray_NDIM(data)-1]
 *     nsave = np.PyArray_DIMS(op2)[0]             # <<<<<<<<<<<<<<
 *     if nsave != npts*2 + 15:
 *         raise error("invalid work array for fft size")
 */
  __pyx_v_nsave = (PyArray_DIMS(__pyx_v_op2)[0]);

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":143
 *     npts = np.PyArray_DIMS(data)[np.PyArray_NDIM(data)-1]
 *     nsave = np.PyArray_DIMS(op2)[0]
 *     if nsave != npts*2 + 15:             # <<<<<<<<<<<<<<
 *         raise error("invalid work array for fft size")
 * 
 */
  __pyx_t_4 = (__pyx_v_nsave != ((__pyx_v_npts * 2) + 15));
  if (__pyx_t_4) {

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":144
 *     nsave = np.PyArray_DIMS(op2)[0]
 *     if nsave != npts*2 + 15:
 *         raise error("invalid work array for fft size")             # <<<<<<<<<<<<<<
 * 
 *     nrepeats = np.PyArray_SIZE(ret)
 */
    __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "error");
    __pyx_t_1 = __site_call1_144_19->Target(__site_call1_144_19, __pyx_context, __pyx_t_2, ((System::Object^)"invalid work array for fft size"));
    __pyx_t_2 = nullptr;
    throw PythonOps::MakeException(__pyx_context, __pyx_t_1, nullptr, nullptr);
    __pyx_t_1 = nullptr;
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":146
 *         raise error("invalid work array for fft size")
 * 
 *     nrepeats = np.PyArray_SIZE(ret)             # <<<<<<<<<<<<<<
 *     nrepeats /= npts
 *     rptr = <double *>np.PyArray_DATA(ret)
 */
  if (__pyx_v_ret == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_ret) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_nrepeats = PyArray_SIZE(((NumpyDotNet::ndarray^)__pyx_v_ret));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":147
 * 
 *     nrepeats = np.PyArray_SIZE(ret)
 *     nrepeats /= npts             # <<<<<<<<<<<<<<
 *     rptr = <double *>np.PyArray_DATA(ret)
 *     dptr = <double *>np.PyArray_DATA(data)
 */
  __pyx_v_nrepeats /= __pyx_v_npts;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":148
 *     nrepeats = np.PyArray_SIZE(ret)
 *     nrepeats /= npts
 *     rptr = <double *>np.PyArray_DATA(ret)             # <<<<<<<<<<<<<<
 *     dptr = <double *>np.PyArray_DATA(data)
 *     wsave = <double *>np.PyArray_DATA(data)
 */
  if (__pyx_v_ret == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_ret) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_rptr = ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_ret)));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":149
 *     nrepeats /= npts
 *     rptr = <double *>np.PyArray_DATA(ret)
 *     dptr = <double *>np.PyArray_DATA(data)             # <<<<<<<<<<<<<<
 *     wsave = <double *>np.PyArray_DATA(data)
 * 
 */
  if (__pyx_v_data == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_data) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_dptr = ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_data)));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":150
 *     rptr = <double *>np.PyArray_DATA(ret)
 *     dptr = <double *>np.PyArray_DATA(data)
 *     wsave = <double *>np.PyArray_DATA(data)             # <<<<<<<<<<<<<<
 * 
 *     for i in range(nrepeats):
 */
  if (__pyx_v_data == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_v_data) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_wsave = ((double *)PyArray_DATA(((NumpyDotNet::ndarray^)__pyx_v_data)));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":152
 *     wsave = <double *>np.PyArray_DATA(data)
 * 
 *     for i in range(nrepeats):             # <<<<<<<<<<<<<<
 *         memcpy(<char *>(rptr + 1), (dptr + 2), (npts - 1)*sizeof(double))
 *         rptr[0] = dptr[0]
 */
  __pyx_t_7 = __pyx_v_nrepeats;
  for (__pyx_t_8 = 0; __pyx_t_8 < __pyx_t_7; __pyx_t_8+=1) {
    __pyx_v_i = __pyx_t_8;

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":153
 * 
 *     for i in range(nrepeats):
 *         memcpy(<char *>(rptr + 1), (dptr + 2), (npts - 1)*sizeof(double))             # <<<<<<<<<<<<<<
 *         rptr[0] = dptr[0]
 *         fftpack_rfftb(npts, rptr, wsave)
 */
    memcpy(((char *)(__pyx_v_rptr + 1)), (__pyx_v_dptr + 2), ((__pyx_v_npts - 1) * (sizeof(double))));

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":154
 *     for i in range(nrepeats):
 *         memcpy(<char *>(rptr + 1), (dptr + 2), (npts - 1)*sizeof(double))
 *         rptr[0] = dptr[0]             # <<<<<<<<<<<<<<
 *         fftpack_rfftb(npts, rptr, wsave)
 *         rptr += npts
 */
    (__pyx_v_rptr[0]) = (__pyx_v_dptr[0]);

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":155
 *         memcpy(<char *>(rptr + 1), (dptr + 2), (npts - 1)*sizeof(double))
 *         rptr[0] = dptr[0]
 *         fftpack_rfftb(npts, rptr, wsave)             # <<<<<<<<<<<<<<
 *         rptr += npts
 *         dptr += npts*2
 */
    GLOBALFUNC(rfftb)(__pyx_v_npts, __pyx_v_rptr, __pyx_v_wsave);

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":156
 *         rptr[0] = dptr[0]
 *         fftpack_rfftb(npts, rptr, wsave)
 *         rptr += npts             # <<<<<<<<<<<<<<
 *         dptr += npts*2
 * 
 */
    __pyx_v_rptr += __pyx_v_npts;

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":157
 *         fftpack_rfftb(npts, rptr, wsave)
 *         rptr += npts
 *         dptr += npts*2             # <<<<<<<<<<<<<<
 * 
 *     return ret
 */
    __pyx_v_dptr += (__pyx_v_npts * 2);
  }

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":159
 *         dptr += npts*2
 * 
 *     return ret             # <<<<<<<<<<<<<<
 * 
 * cdef rffti(long n):
 */
  __pyx_r = __pyx_v_ret;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":161
 *     return ret
 * 
 * cdef rffti(long n):             # <<<<<<<<<<<<<<
 *     cdef np.intp_t dim
 *     cdef np.ndarray op
 */

static  System::Object^ rffti(long __pyx_v_n) {
  __pyx_t_5numpy_3fft_5numpy_intp_t __pyx_v_dim;
  NumpyDotNet::ndarray^ __pyx_v_op;
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;
  __pyx_v_op = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":166
 * 
 *     # Magic size needed by rffti
 *     dim = 2*n + 15             # <<<<<<<<<<<<<<
 *     # Create a 1 dimensional array of dimensions of type double
 *     op = np.PyArray_New(NULL, 1, &dim, np.NPY_DOUBLE, NULL, NULL, 0, 0, NULL)
 */
  __pyx_v_dim = ((2 * __pyx_v_n) + 15);

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":168
 *     dim = 2*n + 15
 *     # Create a 1 dimensional array of dimensions of type double
 *     op = np.PyArray_New(NULL, 1, &dim, np.NPY_DOUBLE, NULL, NULL, 0, 0, NULL)             # <<<<<<<<<<<<<<
 * 
 *     fftpack_rffti(n, <double *>np.PyArray_DATA(op))
 */
  __pyx_t_1 = PyArray_New(NULL, 1, (&__pyx_v_dim), NPY_DOUBLE, NULL, NULL, 0, 0, NULL); 
  if (__pyx_t_1 == nullptr || dynamic_cast<NumpyDotNet::ndarray^>(__pyx_t_1) == nullptr) {
    throw PythonOps::MakeException(__pyx_context, PythonOps::GetGlobal(__pyx_context, "TypeError"), "type error", nullptr);
  }
  __pyx_v_op = ((NumpyDotNet::ndarray^)__pyx_t_1);
  __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":170
 *     op = np.PyArray_New(NULL, 1, &dim, np.NPY_DOUBLE, NULL, NULL, 0, 0, NULL)
 * 
 *     fftpack_rffti(n, <double *>np.PyArray_DATA(op))             # <<<<<<<<<<<<<<
 * 
 *     return op
 */
  GLOBALFUNC(rffti)(__pyx_v_n, ((double *)PyArray_DATA(__pyx_v_op)));

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":172
 *     fftpack_rffti(n, <double *>np.PyArray_DATA(op))
 * 
 *     return op             # <<<<<<<<<<<<<<
 */
  __pyx_r = ((System::Object^)__pyx_v_op);
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":189
 *     object Npy_INTERFACE_array "Npy_INTERFACE_OBJECT" (NpyArray*)
 * 
 * cdef inline object PyUFunc_FromFuncAndData(PyUFuncGenericFunction* func, void** data,             # <<<<<<<<<<<<<<
 *         char* types, int ntypes, int nin, int nout,
 *         int identity, char* name, char* doc, int c):
 */

static CYTHON_INLINE System::Object^ PyUFunc_FromFuncAndData(__pyx_t_5numpy_3fft_5numpy_PyUFuncGenericFunction *__pyx_v_func, void **__pyx_v_data, char *__pyx_v_types, int __pyx_v_ntypes, int __pyx_v_nin, int __pyx_v_nout, int __pyx_v_identity, char *__pyx_v_name, char *__pyx_v_doc, int __pyx_v_c) {
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":192
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":194
 *    return Npy_INTERFACE_ufunc(NpyUFunc_FromFuncAndDataAndSignature(func, data, types, ntypes, nin, nout, identity, name, doc, c, NULL))
 * 
 * cdef inline object PyArray_ZEROS(int ndim, intp_t *shape, int typenum, int fortran):             # <<<<<<<<<<<<<<
 *     shape_list = []
 *     cdef int i
 */

static CYTHON_INLINE System::Object^ PyArray_ZEROS(int __pyx_v_ndim, __pyx_t_5numpy_3fft_5numpy_intp_t *__pyx_v_shape, int __pyx_v_typenum, int __pyx_v_fortran) {
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

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":195
 * 
 * cdef inline object PyArray_ZEROS(int ndim, intp_t *shape, int typenum, int fortran):
 *     shape_list = []             # <<<<<<<<<<<<<<
 *     cdef int i
 *     for i in range(ndim):
 */
  __pyx_t_1 = PythonOps::MakeListNoCopy(gcnew array<System::Object^>{});
  __pyx_v_shape_list = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":197
 *     shape_list = []
 *     cdef int i
 *     for i in range(ndim):             # <<<<<<<<<<<<<<
 *         shape_list.append(shape[i])
 *     import numpy
 */
  __pyx_t_2 = __pyx_v_ndim;
  for (__pyx_t_3 = 0; __pyx_t_3 < __pyx_t_2; __pyx_t_3+=1) {
    __pyx_v_i = __pyx_t_3;

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":198
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

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":199
 *     for i in range(ndim):
 *         shape_list.append(shape[i])
 *     import numpy             # <<<<<<<<<<<<<<
 *     return numpy.zeros(shape_list, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), 'F' if fortran else 'C')
 * 
 */
  __pyx_t_5 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "numpy", -1));
  __pyx_v_numpy = __pyx_t_5;
  __pyx_t_5 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":200
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":202
 *     return numpy.zeros(shape_list, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), 'F' if fortran else 'C')
 * 
 * cdef inline object PyArray_New(void *subtype, int nd, npy_intp *dims, int type_num, npy_intp *strides, void *data, int itemsize, int flags, void *obj):             # <<<<<<<<<<<<<<
 *     assert subtype == NULL
 *     assert obj == NULL
 */

static CYTHON_INLINE System::Object^ PyArray_New(void *__pyx_v_subtype, int __pyx_v_nd, __pyx_t_5numpy_3fft_5numpy_npy_intp *__pyx_v_dims, int __pyx_v_type_num, __pyx_t_5numpy_3fft_5numpy_npy_intp *__pyx_v_strides, void *__pyx_v_data, int __pyx_v_itemsize, int __pyx_v_flags, void *__pyx_v_obj) {
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":203
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

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":204
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

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":205
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":207
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

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":209
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":211
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

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":213
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":215
 *     return NpyArray_DATA(<NpyArray*> <long long>n.Array)
 * 
 * cdef inline intp_t* PyArray_DIMS(ndarray n):             # <<<<<<<<<<<<<<
 *     # XXX "long long" is wrong type
 *     return NpyArray_DIMS(<NpyArray*> <long long>n.Array)
 */

static CYTHON_INLINE __pyx_t_5numpy_3fft_5numpy_intp_t *PyArray_DIMS(NumpyDotNet::ndarray^ __pyx_v_n) {
  __pyx_t_5numpy_3fft_5numpy_intp_t *__pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  PY_LONG_LONG __pyx_t_2;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":217
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":219
 *     return NpyArray_DIMS(<NpyArray*> <long long>n.Array)
 * 
 * cdef inline intp_t PyArray_SIZE(ndarray n):             # <<<<<<<<<<<<<<
 *     # XXX "long long" is wrong type
 *     return NpyArray_SIZE(<NpyArray*> <long long>n.Array)
 */

static CYTHON_INLINE __pyx_t_5numpy_3fft_5numpy_intp_t PyArray_SIZE(NumpyDotNet::ndarray^ __pyx_v_n) {
  __pyx_t_5numpy_3fft_5numpy_intp_t __pyx_r;
  System::Object^ __pyx_t_1 = nullptr;
  PY_LONG_LONG __pyx_t_2;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":221
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":223
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

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":224
 * 
 * cdef inline object PyArray_FromAny(op, newtype, min_depth, max_depth, flags, context):
 *     import clr             # <<<<<<<<<<<<<<
 *     import NumpyDotNet.NpyArray
 *     return NumpyDotNet.NpyArray.FromAny(op, newtype, min_depth, max_depth, flags, context)
 */
  __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "clr", -1));
  __pyx_v_clr = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":225
 * cdef inline object PyArray_FromAny(op, newtype, min_depth, max_depth, flags, context):
 *     import clr
 *     import NumpyDotNet.NpyArray             # <<<<<<<<<<<<<<
 *     return NumpyDotNet.NpyArray.FromAny(op, newtype, min_depth, max_depth, flags, context)
 * 
 */
  __pyx_t_1 = LightExceptions::CheckAndThrow(PythonOps::ImportTop(__pyx_context, "NumpyDotNet.NpyArray", -1));
  __pyx_v_NumpyDotNet = __pyx_t_1;
  __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":226
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":228
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

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":229
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

    /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":230
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

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":231
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":233
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

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":234
 * 
 * cdef inline object PyArray_Check(obj):
 *     return isinstance(obj, ndarray)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_NDIM(obj):
 */
  __pyx_t_1 = PythonOps::GetGlobal(__pyx_context, "isinstance");
  __pyx_t_2 = __site_call2_234_21->Target(__site_call2_234_21, __pyx_context, __pyx_t_1, __pyx_v_obj, ((System::Object^)((System::Object^)__pyx_ptype_5numpy_3fft_5numpy_ndarray)));
  __pyx_t_1 = nullptr;
  __pyx_r = __pyx_t_2;
  __pyx_t_2 = nullptr;
  goto __pyx_L0;

  __pyx_r = nullptr;
  __pyx_L0:;
  return __pyx_r;
}

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":236
 *     return isinstance(obj, ndarray)
 * 
 * cdef inline object PyArray_NDIM(obj):             # <<<<<<<<<<<<<<
 *     return obj.ndim
 * 
 */

static CYTHON_INLINE System::Object^ PyArray_NDIM(System::Object^ __pyx_v_obj) {
  System::Object^ __pyx_r = nullptr;
  System::Object^ __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":237
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

/* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":239
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
  __pyx_int_0 = 0;
  __pyx_int_1 = 1;

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
  __site_op_sub_35_55 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Subtract));
  __site_cvt_Py_ssize_t_35_55 = CallSite< System::Func< CallSite^, System::Object^, Py_ssize_t >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, Py_ssize_t::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_37_19 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_sub_62_55 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Subtract));
  __site_cvt_Py_ssize_t_62_55 = CallSite< System::Func< CallSite^, System::Object^, Py_ssize_t >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, Py_ssize_t::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_64_19 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_op_sub_101_54 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Subtract));
  __site_cvt_Py_ssize_t_101_54 = CallSite< System::Func< CallSite^, System::Object^, Py_ssize_t >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, Py_ssize_t::typeid, ConversionResultKind::ExplicitCast));
  __site_op_sub_102_48 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Subtract));
  __site_cvt_Py_ssize_t_102_48 = CallSite< System::Func< CallSite^, System::Object^, Py_ssize_t >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, Py_ssize_t::typeid, ConversionResultKind::ExplicitCast));
  __site_cvt_int_103_42 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_op_sub_104_48 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Subtract));
  __site_cvt_Py_ssize_t_104_48 = CallSite< System::Func< CallSite^, System::Object^, Py_ssize_t >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, Py_ssize_t::typeid, ConversionResultKind::ExplicitCast));
  __site_op_sub_106_54 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Subtract));
  __site_cvt_Py_ssize_t_106_54 = CallSite< System::Func< CallSite^, System::Object^, Py_ssize_t >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, Py_ssize_t::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_110_19 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
  __site_cvt_int_139_42 = CallSite< System::Func< CallSite^, System::Object^, int >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, int::typeid, ConversionResultKind::ExplicitCast));
  __site_op_sub_141_54 = CallSite< System::Func< CallSite^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeBinaryOperationAction(__pyx_context, ExpressionType::Subtract));
  __site_cvt_Py_ssize_t_141_54 = CallSite< System::Func< CallSite^, System::Object^, Py_ssize_t >^ >::Create(PythonOps::MakeConversionAction(__pyx_context, Py_ssize_t::typeid, ConversionResultKind::ExplicitCast));
  __site_call1_144_19 = CallSite< System::Func< CallSite^, CodeContext^, System::Object^, System::Object^, System::Object^ >^ >::Create(PythonOps::MakeInvokeAction(__pyx_context, CallSignature(1)));
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

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":17
 * 
 * cimport numpy as np
 * np.import_array()             # <<<<<<<<<<<<<<
 * 
 * class error(Exception):
 */
  import_array();

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":19
 * np.import_array()
 * 
 * class error(Exception):             # <<<<<<<<<<<<<<
 *     pass
 * 
 */
  __pyx_t_1 = PythonOps::MakeEmptyDict();
  __pyx_t_2 = PythonOps::GetGlobal(__pyx_context, "Exception");
  __pyx_t_3 = PythonOps::MakeTuple(gcnew array<System::Object^>{__pyx_t_2});
  __pyx_t_2 = nullptr;
  FunctionCode^ func_code_error = PythonOps::MakeFunctionCode(__pyx_context, "func_code_error", nullptr, gcnew array<System::String^>{"arg0"}, FunctionAttributes::None, 0, 0, "", gcnew System::Func<CodeContext^, CodeContext^>(mk_empty_context), gcnew array<System::String^>(0), gcnew array<System::String^>(0), gcnew array<System::String^>(0), gcnew array<System::String^>(0), 0);
  PythonTuple^ tbases_error = safe_cast<PythonTuple^>(__pyx_t_3);
  array<System::Object^>^ bases_error = gcnew array<System::Object^>(tbases_error->Count);
  tbases_error->CopyTo(bases_error, 0);
  __pyx_t_2 = PythonOps::MakeClass(func_code_error, nullptr, __pyx_context, "error", bases_error, "");
  __pyx_t_3 = nullptr;
  PythonOps::SetGlobal(__pyx_context, "error", __pyx_t_2);
  __pyx_t_2 = nullptr;
  __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\fftpack_cython.pyx":1
 * """ Cythonized version of fftpack_litemodule.c             # <<<<<<<<<<<<<<
 * """
 * 
 */
  __pyx_t_1 = PythonOps::MakeEmptyDict();
  PythonOps::SetGlobal(__pyx_context, "__test__", ((System::Object^)__pyx_t_1));
  __pyx_t_1 = nullptr;

  /* "C:\Users\jwiggins\source\jwiggins-numpy-refactor\numpy\fft\numpy.pxd":239
 *     return obj.ndim
 * 
 * cdef inline void import_array():             # <<<<<<<<<<<<<<
 *     pass
 */
}
/* Cython code section 'cleanup_globals' */
/* Cython code section 'cleanup_module' */
/* Cython code section 'main_method' */
/* Cython code section 'utility_code_def' */

/* Runtime support code */

static CYTHON_INLINE long __Pyx_div_long(long a, long b) {
    long q = a / b;
    long r = a - q*b;
    q -= ((r != 0) & ((r ^ b) < 0));
    return q;
}
/* Cython code section 'end' */
};
[assembly: PythonModule("numpy__fft__fftpack_cython", module_fftpack_cython::typeid)];
};
