#ifndef _NPY_COMMON_H_
#define _NPY_COMMON_H_

#include "npy_config.h"

#if defined(__cplusplus)
extern "C" {
#endif
    

#if defined(_MSC_VER)
        #define NPY_INLINE __inline
#elif defined(__GNUC__)
        #define NPY_INLINE inline
#else
        #define NPY_INLINE
#endif

#ifdef _WIN32
#ifdef BUILDING_NDARRAY
#define NDARRAY_API __declspec(dllexport)
#else
#define NDARRAY_API __declspec(dllimport)
#endif
#else
#define NDARRAY_API
#endif


/* enums for detected endianness */
enum {
    NPY_CPU_UNKNOWN_ENDIAN,
    NPY_CPU_LITTLE,
    NPY_CPU_BIG,
};

/* Some platforms don't define bool, long long, or long double.
   Handle that here.
*/

#define NPY_BYTE_FMT "hhd"
#define NPY_UBYTE_FMT "hhu"
#define NPY_SHORT_FMT "hd"
#define NPY_USHORT_FMT "hu"
#define NPY_INT_FMT "d"
#define NPY_UINT_FMT "u"
#define NPY_LONG_FMT "ld"
#define NPY_ULONG_FMT "lu"
#define NPY_FLOAT_FMT "g"
#define NPY_DOUBLE_FMT "g"

#ifdef NPY_HAVE_LONGLONG
typedef long long             npy_longlong;
typedef unsigned long long    npy_ulonglong;
#  ifdef _MSC_VER
#    define NPY_LONGLONG_FMT         "I64d"
#    define NPY_ULONGLONG_FMT        "I64u"
#    define NPY_LONGLONG_SUFFIX(x)   (x##i64)
#    define NPY_ULONGLONG_SUFFIX(x)  (x##Ui64)
#  else
        /* #define LONGLONG_FMT   "lld"      Another possible variant
           #define ULONGLONG_FMT  "llu"

           #define LONGLONG_FMT   "qd"   -- BSD perhaps?
           #define ULONGLONG_FMT  "qu"
        */
#    define NPY_LONGLONG_FMT         "Ld"
#    define NPY_ULONGLONG_FMT        "Lu"
#    define NPY_LONGLONG_SUFFIX(x)   (x##LL)
#    define NPY_ULONGLONG_SUFFIX(x)  (x##ULL)
#  endif
#else
typedef long npy_longlong;
typedef unsigned long npy_ulonglong;
#  define NPY_LONGLONG_SUFFIX(x)  (x##L)
#  define NPY_ULONGLONG_SUFFIX(x) (x##UL)
#endif /* ifdef NPY_HAVE_LONGLONG */


typedef unsigned char npy_bool;
#define NPY_FALSE 0
#define NPY_TRUE 1


#if NPY_SIZEOF_LONGDOUBLE == NPY_SIZEOF_DOUBLE
        typedef double npy_longdouble;
        #define NPY_LONGDOUBLE_FMT "g"
#else
        typedef long double npy_longdouble;
        #define NPY_LONGDOUBLE_FMT "Lg"
#endif


typedef signed char npy_byte;
typedef unsigned char npy_ubyte;
typedef unsigned short npy_ushort;
typedef unsigned int npy_uint;
typedef unsigned long npy_ulong;

/* These are for completeness */
typedef float npy_float;
typedef double npy_double;
typedef short npy_short;
typedef int npy_int;
typedef long npy_long;

/*
 * Disabling C99 complex usage: a lot of C code in numpy/scipy rely on being
 * able to do .real/.imag. Will have to convert code first.
 */
#if 0
#if NPY_HAVE_COMPLEX_H && defined(NPY_HAVE_COMPLEX_DOUBLE)
typedef complex npy_cdouble;
#else
typedef struct { double real, imag; } npy_cdouble;
#endif

#if NPY_HAVE_COMPLEX_H && defined(NPY_HAVE_COMPLEX_FLOAT)
typedef complex float npy_cfloat;
#else
typedef struct { float real, imag; } npy_cfloat;
#endif

#if NPY_HAVE_COMPLEX_H && defined(NPY_HAVE_COMPLEX_LONG_DOUBLE)
typedef complex long double npy_clongdouble;
#else
typedef struct {npy_longdouble real, imag;} npy_clongdouble;
#endif
#endif

#if NPY_HAVE_COMPLEX_H && NPY_SIZEOF_COMPLEX_DOUBLE != 2 * NPY_SIZEOF_DOUBLE
#error npy_cdouble definition is not compatible with C99 complex definition ! \
        Please contact Numpy maintainers and give detailed information about \
        your compiler and platform
#endif
typedef struct { double real, imag; } npy_cdouble;

#if NPY_HAVE_COMPLEX_H && NPY_SIZEOF_COMPLEX_FLOAT != 2 * NPY_SIZEOF_FLOAT
#error npy_cfloat definition is not compatible with C99 complex definition ! \
        Please contact Numpy maintainers and give detailed information about \
        your compiler and platform
#endif
typedef struct { float real, imag; } npy_cfloat;

#if NPY_HAVE_COMPLEX_H && \
           NPY_SIZEOF_COMPLEX_LONGDOUBLE != 2 * NPY_SIZEOF_LONGDOUBLE
#error npy_clongdouble definition is not compatible with C99 complex \
        definition !  Please contact Numpy maintainers and give detailed \
        information about your compiler and platform
#endif
typedef struct { npy_longdouble real, imag; } npy_clongdouble;

/*
 * numarray-style bit-width typedefs
 */
#define NPY_MAX_INT8 127
#define NPY_MIN_INT8 -128
#define NPY_MAX_UINT8 255
#define NPY_MAX_INT16 32767
#define NPY_MIN_INT16 -32768
#define NPY_MAX_UINT16 65535
#define NPY_MAX_INT32 2147483647
#define NPY_MIN_INT32 (-NPY_MAX_INT32 - 1)
#define NPY_MAX_UINT32 4294967295U
#define NPY_MAX_INT64 NPY_LONGLONG_SUFFIX(9223372036854775807)
#define NPY_MIN_INT64 (-NPY_MAX_INT64 - NPY_LONGLONG_SUFFIX(1))
#define NPY_MAX_UINT64 NPY_ULONGLONG_SUFFIX(18446744073709551615)
#define NPY_MAX_INT128 NPY_LONGLONG_SUFFIX(85070591730234615865843651857942052864)
#define NPY_MIN_INT128 (-NPY_MAX_INT128 - NPY_LONGLONG_SUFFIX(1))
#define NPY_MAX_UINT128 NPY_ULONGLONG_SUFFIX(170141183460469231731687303715884105728)
#define NPY_MAX_INT256 NPY_LONGLONG_SUFFIX(57896044618658097711785492504343953926634992332820282019728792003956564819967)
#define NPY_MIN_INT256 (-NPY_MAX_INT256 - NPY_LONGLONG_SUFFIX(1))
#define NPY_MAX_UINT256 NPY_ULONGLONG_SUFFIX(115792089237316195423570985008687907853269984665640564039457584007913129639935)
#define NPY_MIN_DATETIME NPY_MIN_INT64
#define NPY_MAX_DATETIME NPY_MAX_INT64
#define NPY_MIN_TIMEDELTA NPY_MIN_INT64
#define NPY_MAX_TIMEDELTA NPY_MAX_INT64

        /* Need to find the number of bits for each type and
           make definitions accordingly.

           C states that sizeof(char) == 1 by definition

           So, just using the sizeof keyword won't help.

           It also looks like Python itself uses sizeof(char) quite a
           bit, which by definition should be 1 all the time.

           Idea: Make Use of CHAR_BIT which should tell us how many
           BITS per CHARACTER
        */

        /* Include platform definitions -- These are in the C89/90 standard */
#include <limits.h>
#define NPY_MAX_BYTE SCHAR_MAX
#define NPY_MIN_BYTE SCHAR_MIN
#define NPY_MAX_UBYTE UCHAR_MAX
#define NPY_MAX_SHORT SHRT_MAX
#define NPY_MIN_SHORT SHRT_MIN
#define NPY_MAX_USHORT USHRT_MAX
#define NPY_MAX_INT   INT_MAX
#ifndef INT_MIN
#define INT_MIN (-INT_MAX - 1)
#endif
#define NPY_MIN_INT   INT_MIN
#define NPY_MAX_UINT  UINT_MAX
#define NPY_MAX_LONG  LONG_MAX
#define NPY_MIN_LONG  LONG_MIN
#define NPY_MAX_ULONG  ULONG_MAX

#define NPY_SIZEOF_DATETIME 8
#define NPY_SIZEOF_TIMEDELTA 8

#define NPY_BITSOF_BOOL (sizeof(npy_bool)*CHAR_BIT)
#define NPY_BITSOF_CHAR CHAR_BIT
#define NPY_BITSOF_SHORT (NPY_SIZEOF_SHORT * CHAR_BIT)
#define NPY_BITSOF_INT (NPY_SIZEOF_INT * CHAR_BIT)
#define NPY_BITSOF_LONG (NPY_SIZEOF_LONG * CHAR_BIT)
#define NPY_BITSOF_LONGLONG (NPY_SIZEOF_LONGLONG * CHAR_BIT)
#define NPY_BITSOF_FLOAT (NPY_SIZEOF_FLOAT * CHAR_BIT)
#define NPY_BITSOF_DOUBLE (NPY_SIZEOF_DOUBLE * CHAR_BIT)
#define NPY_BITSOF_LONGDOUBLE (NPY_SIZEOF_LONGDOUBLE * CHAR_BIT)
#define NPY_BITSOF_DATETIME (NPY_SIZEOF_DATETIME * CHAR_BIT)
#define NPY_BITSOF_TIMEDELTA (NPY_SIZEOF_TIMEDELTA * CHAR_BIT)

#if NPY_BITSOF_LONG == 8
#define NPY_INT8 NPY_LONG
#define NPY_UINT8 NPY_ULONG
        typedef long npy_int8;
        typedef unsigned long npy_uint8;
#define NPY_INT8_FMT NPY_LONG_FMT
#define NPY_UINT8_FMT NPY_ULONG_FMT
#elif NPY_BITSOF_LONG == 16
#define NPY_INT16 NPY_LONG
#define NPY_UINT16 NPY_ULONG
        typedef long npy_int16;
        typedef unsigned long npy_uint16;
#define NPY_INT16_FMT NPY_LONG_FMT
#define NPY_UINT16_FMT NPY_ULONG_FMT
#elif NPY_BITSOF_LONG == 32
#define NPY_INT32 NPY_LONG
#define NPY_UINT32 NPY_ULONG
        typedef long npy_int32;
        typedef unsigned long npy_uint32;
        typedef unsigned long npy_ucs4;
#define NPY_INT32_FMT NPY_LONG_FMT
#define NPY_UINT32_FMT NPY_ULONG_FMT
#elif NPY_BITSOF_LONG == 64
#define NPY_INT64 NPY_LONG
#define NPY_UINT64 NPY_ULONG
        typedef long npy_int64;
        typedef unsigned long npy_uint64;
#define NPY_INT64_FMT NPY_LONG_FMT
#define NPY_UINT64_FMT NPY_ULONG_FMT
#elif NPY_BITSOF_LONG == 128
#define NPY_INT128 NPY_LONG
#define NPY_UINT128 NPY_ULONG
        typedef long npy_int128;
        typedef unsigned long npy_uint128;
#define NPY_INT128_FMT NPY_LONG_FMT
#define NPY_UINT128_FMT NPY_ULONG_FMT
#endif

#if NPY_BITSOF_LONGLONG == 8
#  ifndef NPY_INT8
#    define NPY_INT8 NPY_LONGLONG
#    define NPY_UINT8 NPY_ULONGLONG
        typedef npy_longlong npy_int8;
        typedef npy_ulonglong npy_uint8;
#define NPY_INT8_FMT NPY_LONGLONG_FMT
#define NPY_UINT8_FMT NPY_ULONGLONG_FMT
#  endif
#  define NPY_MAX_LONGLONG NPY_MAX_INT8
#  define NPY_MIN_LONGLONG NPY_MIN_INT8
#  define NPY_MAX_ULONGLONG NPY_MAX_UINT8
#elif NPY_BITSOF_LONGLONG == 16
#  ifndef NPY_INT16
#    define NPY_INT16 NPY_LONGLONG
#    define NPY_UINT16 NPY_ULONGLONG
        typedef npy_longlong npy_int16;
        typedef npy_ulonglong npy_uint16;
#define NPY_INT16_FMT NPY_LONGLONG_FMT
#define NPY_UINT16_FMT NPY_ULONGLONG_FMT
#  endif
#  define NPY_MAX_LONGLONG NPY_MAX_INT16
#  define NPY_MIN_LONGLONG NPY_MIN_INT16
#  define NPY_MAX_ULONGLONG NPY_MAX_UINT16
#elif NPY_BITSOF_LONGLONG == 32
#  ifndef NPY_INT32
#    define NPY_INT32 NPY_LONGLONG
#    define NPY_UINT32 NPY_ULONGLONG
        typedef npy_longlong npy_int32;
        typedef npy_ulonglong npy_uint32;
        typedef npy_ulonglong npy_ucs4;
#define NPY_INT32_FMT NPY_LONGLONG_FMT
#define NPY_UINT32_FMT NPY_ULONGLONG_FMT
#  endif
#  define NPY_MAX_LONGLONG NPY_MAX_INT32
#  define NPY_MIN_LONGLONG NPY_MIN_INT32
#  define NPY_MAX_ULONGLONG NPY_MAX_UINT32
#elif NPY_BITSOF_LONGLONG == 64
#  ifndef NPY_INT64
#    define NPY_INT64 NPY_LONGLONG
#    define NPY_UINT64 NPY_ULONGLONG
        typedef npy_longlong npy_int64;
        typedef npy_ulonglong npy_uint64;
#define NPY_INT64_FMT NPY_LONGLONG_FMT
#define NPY_UINT64_FMT NPY_ULONGLONG_FMT
#  endif
#  define NPY_MAX_LONGLONG NPY_MAX_INT64
#  define NPY_MIN_LONGLONG NPY_MIN_INT64
#  define NPY_MAX_ULONGLONG NPY_MAX_UINT64
#elif NPY_BITSOF_LONGLONG == 128
#  ifndef NPY_INT128
#    define NPY_INT128 NPY_LONGLONG
#    define NPY_UINT128 NPY_ULONGLONG
        typedef npy_longlong npy_int128;
        typedef npy_ulonglong npy_uint128;
#define NPY_INT128_FMT NPY_LONGLONG_FMT
#define NPY_UINT128_FMT NPY_ULONGLONG_FMT
#  endif
#  define NPY_MAX_LONGLONG NPY_MAX_INT128
#  define NPY_MIN_LONGLONG NPY_MIN_INT128
#  define NPY_MAX_ULONGLONG NPY_MAX_UINT128
#elif NPY_BITSOF_LONGLONG == 256
#  define NPY_INT256 NPY_LONGLONG
#  define NPY_UINT256 NPY_ULONGLONG
        typedef npy_longlong npy_int256;
        typedef npy_ulonglong npy_uint256;
#define NPY_INT256_FMT NPY_LONGLONG_FMT
#define NPY_UINT256_FMT NPY_ULONGLONG_FMT
#  define NPY_MAX_LONGLONG NPY_MAX_INT256
#  define NPY_MIN_LONGLONG NPY_MIN_INT256
#  define NPY_MAX_ULONGLONG NPY_MAX_UINT256
#endif

#if NPY_BITSOF_INT == 8
#ifndef NPY_INT8
#define NPY_INT8 NPY_INT
#define NPY_UINT8 NPY_UINT
        typedef int npy_int8;
        typedef unsigned int npy_uint8;
#define NPY_INT8_FMT NPY_INT_FMT
#define NPY_UINT8_FMT NPY_UINT_FMT
#endif
#elif NPY_BITSOF_INT == 16
#ifndef NPY_INT16
#define NPY_INT16 NPY_INT
#define NPY_UINT16 NPY_UINT
        typedef int npy_int16;
        typedef unsigned int npy_uint16;
#define NPY_INT16_FMT NPY_INT_FMT
#define NPY_UINT16_FMT NPY_UINT_FMT
#endif
#elif NPY_BITSOF_INT == 32
#ifndef NPY_INT32
#define NPY_INT32 NPY_INT
#define NPY_UINT32 NPY_UINT
        typedef int npy_int32;
        typedef unsigned int npy_uint32;
        typedef unsigned int npy_ucs4;
#define NPY_INT32_FMT NPY_INT_FMT
#define NPY_UINT32_FMT NPY_UINT_FMT
#endif
#elif NPY_BITSOF_INT == 64
#ifndef NPY_INT64
#define NPY_INT64 NPY_INT
#define NPY_UINT64 NPY_UINT
        typedef int npy_int64;
        typedef unsigned int npy_uint64;
#define NPY_INT64_FMT NPY_INT_FMT
#define NPY_UINT64_FMT NPY_UINT_FMT
#endif
#elif NPY_BITSOF_INT == 128
#ifndef NPY_INT128
#define NPY_INT128 NPY_INT
#define NPY_UINT128 NPY_UINT
        typedef int npy_int128;
        typedef unsigned int npy_uint128;
#define NPY_INT128_FMT NPY_INT_FMT
#define NPY_UINT128_FMT NPY_UINT_FMT
#endif
#endif

#if NPY_BITSOF_SHORT == 8
#ifndef NPY_INT8
#define NPY_INT8 NPY_SHORT
#define NPY_UINT8 NPY_USHORT
        typedef short npy_int8;
        typedef unsigned short npy_uint8;
#define NPY_INT8_FMT NPY_SHORT_FMT
#define NPY_UINT8_FMT NPY_USHORT_FMT
#endif
#elif NPY_BITSOF_SHORT == 16
#ifndef NPY_INT16
#define NPY_INT16 NPY_SHORT
#define NPY_UINT16 NPY_USHORT
        typedef short npy_int16;
        typedef unsigned short npy_uint16;
#define NPY_INT16_FMT NPY_SHORT_FMT
#define NPY_UINT16_FMT NPY_USHORT_FMT
#endif
#elif NPY_BITSOF_SHORT == 32
#ifndef NPY_INT32
#define NPY_INT32 NPY_SHORT
#define NPY_UINT32 NPY_USHORT
        typedef short npy_int32;
        typedef unsigned short npy_uint32;
        typedef unsigned short npy_ucs4;
#define NPY_INT32_FMT NPY_SHORT_FMT
#define NPY_UINT32_FMT NPY_USHORT_FMT
#endif
#elif NPY_BITSOF_SHORT == 64
#ifndef NPY_INT64
#define NPY_INT64 NPY_SHORT
#define NPY_UINT64 NPY_USHORT
        typedef short npy_int64;
        typedef unsigned short npy_uint64;
#define NPY_INT64_FMT NPY_SHORT_FMT
#define NPY_UINT64_FMT NPY_USHORT_FMT
#endif
#elif NPY_BITSOF_SHORT == 128
#ifndef NPY_INT128
#define NPY_INT128 NPY_SHORT
#define NPY_UINT128 NPY_USHORT
        typedef short npy_int128;
        typedef unsigned short npy_uint128;
#define NPY_INT128_FMT NPY_SHORT_FMT
#define NPY_UINT128_FMT NPY_USHORT_FMT
#endif
#endif

#if NPY_BITSOF_CHAR == 8
#ifndef NPY_INT8
#define NPY_INT8 NPY_BYTE
#define NPY_UINT8 NPY_UBYTE
        typedef signed char npy_int8;
        typedef unsigned char npy_uint8;
#define NPY_INT8_FMT NPY_BYTE_FMT
#define NPY_UINT8_FMT NPY_UBYTE_FMT
#endif
#elif NPY_BITSOF_CHAR == 16
#ifndef NPY_INT16
#define NPY_INT16 NPY_BYTE
#define NPY_UINT16 NPY_UBYTE
        typedef signed char npy_int16;
        typedef unsigned char npy_uint16;
#define NPY_INT16_FMT NPY_BYTE_FMT
#define NPY_UINT16_FMT NPY_UBYTE_FMT
#endif
#elif NPY_BITSOF_CHAR == 32
#ifndef NPY_INT32
#define NPY_INT32 NPY_BYTE
#define NPY_UINT32 NPY_UBYTE
        typedef signed char npy_int32;
        typedef unsigned char npy_uint32;
        typedef unsigned char npy_ucs4;
#define NPY_INT32_FMT NPY_BYTE_FMT
#define NPY_UINT32_FMT NPY_UBYTE_FMT
#endif
#elif NPY_BITSOF_CHAR == 64
#ifndef NPY_INT64
#define NPY_INT64 NPY_BYTE
#define NPY_UINT64 NPY_UBYTE
        typedef signed char npy_int64;
        typedef unsigned char npy_uint64;
#define NPY_INT64_FMT NPY_BYTE_FMT
#define NPY_UINT64_FMT NPY_UBYTE_FMT
#endif
#elif NPY_BITSOF_CHAR == 128
#ifndef NPY_INT128
#define NPY_INT128 NPY_BYTE
#define NPY_UINT128 NPY_UBYTE
        typedef signed char npy_int128;
        typedef unsigned char npy_uint128;
#define NPY_INT128_FMT NPY_BYTE_FMT
#define NPY_UINT128_FMT NPY_UBYTE_FMT
#endif
#endif

#if NPY_BITSOF_DOUBLE == 16
#ifndef NPY_FLOAT16
#define NPY_FLOAT16 NPY_DOUBLE
#define NPY_COMPLEX32 NPY_CDOUBLE
        typedef  double npy_float16;
        typedef npy_cdouble npy_complex32;
#define NPY_FLOAT16_FMT NPY_DOUBLE_FMT
#define NPY_COMPLEX32_FMT NPY_CDOUBLE_FMT
#endif
#elif NPY_BITSOF_DOUBLE == 32
#ifndef NPY_FLOAT32
#define NPY_FLOAT32 NPY_DOUBLE
#define NPY_COMPLEX64 NPY_CDOUBLE
        typedef double npy_float32;
        typedef npy_cdouble npy_complex64;
#define NPY_FLOAT32_FMT NPY_DOUBLE_FMT
#define NPY_COMPLEX64_FMT NPY_CDOUBLE_FMT
#endif
#elif NPY_BITSOF_DOUBLE == 64
#ifndef NPY_FLOAT64
#define NPY_FLOAT64 NPY_DOUBLE
#define NPY_COMPLEX128 NPY_CDOUBLE
        typedef double npy_float64;
        typedef npy_cdouble npy_complex128;
#define NPY_FLOAT64_FMT NPY_DOUBLE_FMT
#define NPY_COMPLEX128_FMT NPY_CDOUBLE_FMT
#endif
#elif NPY_BITSOF_DOUBLE == 80
#ifndef NPY_FLOAT80
#define NPY_FLOAT80 NPY_DOUBLE
#define NPY_COMPLEX160 NPY_CDOUBLE
        typedef double npy_float80;
        typedef npy_cdouble npy_complex160;
#define NPY_FLOAT80_FMT NPY_DOUBLE_FMT
#define NPY_COMPLEX160_FMT NPY_CDOUBLE_FMT
#endif
#elif NPY_BITSOF_DOUBLE == 96
#ifndef NPY_FLOAT96
#define NPY_FLOAT96 NPY_DOUBLE
#define NPY_COMPLEX192 NPY_CDOUBLE
        typedef double npy_float96;
        typedef npy_cdouble npy_complex192;
#define NPY_FLOAT96_FMT NPY_DOUBLE_FMT
#define NPY_COMPLEX192_FMT NPY_CDOUBLE_FMT
#endif
#elif NPY_BITSOF_DOUBLE == 128
#ifndef NPY_FLOAT128
#define NPY_FLOAT128 NPY_DOUBLE
#define NPY_COMPLEX256 NPY_CDOUBLE
        typedef double npy_float128;
        typedef npy_cdouble npy_complex256;
#define NPY_FLOAT128_FMT NPY_DOUBLE_FMT
#define NPY_COMPLEX256_FMT NPY_CDOUBLE_FMT
#endif
#endif


#if NPY_BITSOF_FLOAT == 16
#ifndef NPY_FLOAT16
#define NPY_FLOAT16 NPY_FLOAT
#define NPY_COMPLEX32 NPY_CFLOAT
        typedef float npy_float16;
        typedef npy_cfloat npy_complex32;
#define NPY_FLOAT16_FMT NPY_FLOAT_FMT
#define NPY_COMPLEX32_FMT NPY_CFLOAT_FMT
#endif
#elif NPY_BITSOF_FLOAT == 32
#ifndef NPY_FLOAT32
#define NPY_FLOAT32 NPY_FLOAT
#define NPY_COMPLEX64 NPY_CFLOAT
        typedef float npy_float32;
        typedef npy_cfloat npy_complex64;
#define NPY_FLOAT32_FMT NPY_FLOAT_FMT
#define NPY_COMPLEX64_FMT NPY_CFLOAT_FMT
#endif
#elif NPY_BITSOF_FLOAT == 64
#ifndef NPY_FLOAT64
#define NPY_FLOAT64 NPY_FLOAT
#define NPY_COMPLEX128 NPY_CFLOAT
        typedef float npy_float64;
        typedef npy_cfloat npy_complex128;
#define NPY_FLOAT64_FMT NPY_FLOAT_FMT
#define NPY_COMPLEX128_FMT NPY_CFLOAT_FMT
#endif
#elif NPY_BITSOF_FLOAT == 80
#ifndef NPY_FLOAT80
#define NPY_FLOAT80 NPY_FLOAT
#define NPY_COMPLEX160 NPY_CFLOAT
        typedef float npy_float80;
        typedef npy_cfloat npy_complex160;
#define NPY_FLOAT80_FMT NPY_FLOAT_FMT
#define NPY_COMPLEX160_FMT NPY_CFLOAT_FMT
#endif
#elif NPY_BITSOF_FLOAT == 96
#ifndef NPY_FLOAT96
#define NPY_FLOAT96 NPY_FLOAT
#define NPY_COMPLEX192 NPY_CFLOAT
        typedef float npy_float96;
        typedef npy_cfloat npy_complex192;
#define NPY_FLOAT96_FMT NPY_FLOAT_FMT
#define NPY_COMPLEX192_FMT NPY_CFLOAT_FMT
#endif
#elif NPY_BITSOF_FLOAT == 128
#ifndef NPY_FLOAT128
#define NPY_FLOAT128 NPY_FLOAT
#define NPY_COMPLEX256 NPY_CFLOAT
        typedef float npy_float128;
        typedef npy_cfloat npy_complex256;
#define NPY_FLOAT128_FMT NPY_FLOAT_FMT
#define NPY_COMPLEX256_FMT NPY_CFLOAT_FMT
#endif
#endif

#if NPY_BITSOF_LONGDOUBLE == 16
#ifndef NPY_FLOAT16
#define NPY_FLOAT16 NPY_LONGDOUBLE
#define NPY_COMPLEX32 NPY_CLONGDOUBLE
        typedef npy_longdouble npy_float16;
        typedef npy_clongdouble npy_complex32;
#define NPY_FLOAT16_FMT NPY_LONGDOUBLE_FMT
#define NPY_COMPLEX32_FMT NPY_CLONGDOUBLE_FMT
#endif
#elif NPY_BITSOF_LONGDOUBLE == 32
#ifndef NPY_FLOAT32
#define NPY_FLOAT32 NPY_LONGDOUBLE
#define NPY_COMPLEX64 NPY_CLONGDOUBLE
        typedef npy_longdouble npy_float32;
        typedef npy_clongdouble npy_complex64;
#define NPY_FLOAT32_FMT NPY_LONGDOUBLE_FMT
#define NPY_COMPLEX64_FMT NPY_CLONGDOUBLE_FMT
#endif
#elif NPY_BITSOF_LONGDOUBLE == 64
#ifndef NPY_FLOAT64
#define NPY_FLOAT64 NPY_LONGDOUBLE
#define NPY_COMPLEX128 NPY_CLONGDOUBLE
        typedef npy_longdouble npy_float64;
        typedef npy_clongdouble npy_complex128;
#define NPY_FLOAT64_FMT NPY_LONGDOUBLE_FMT
#define NPY_COMPLEX128_FMT NPY_CLONGDOUBLE_FMT
#endif
#elif NPY_BITSOF_LONGDOUBLE == 80
#ifndef NPY_FLOAT80
#define NPY_FLOAT80 NPY_LONGDOUBLE
#define NPY_COMPLEX160 NPY_CLONGDOUBLE
        typedef npy_longdouble npy_float80;
        typedef npy_clongdouble npy_complex160;
#define NPY_FLOAT80_FMT NPY_LONGDOUBLE_FMT
#define NPY_COMPLEX160_FMT NPY_CLONGDOUBLE_FMT
#endif
#elif NPY_BITSOF_LONGDOUBLE == 96
#ifndef NPY_FLOAT96
#define NPY_FLOAT96 NPY_LONGDOUBLE
#define NPY_COMPLEX192 NPY_CLONGDOUBLE
        typedef npy_longdouble npy_float96;
        typedef npy_clongdouble npy_complex192;
#define NPY_FLOAT96_FMT NPY_LONGDOUBLE_FMT
#define NPY_COMPLEX192_FMT NPY_CLONGDOUBLE_FMT
#endif
#elif NPY_BITSOF_LONGDOUBLE == 128
#ifndef NPY_FLOAT128
#define NPY_FLOAT128 NPY_LONGDOUBLE
#define NPY_COMPLEX256 NPY_CLONGDOUBLE
        typedef npy_longdouble npy_float128;
        typedef npy_clongdouble npy_complex256;
#define NPY_FLOAT128_FMT NPY_LONGDOUBLE_FMT
#define NPY_COMPLEX256_FMT NPY_CLONGDOUBLE_FMT
#endif
#elif NPY_BITSOF_LONGDOUBLE == 256
#define NPY_FLOAT256 NPY_LONGDOUBLE
#define NPY_COMPLEX512 NPY_CLONGDOUBLE
        typedef npy_longdouble npy_float256;
        typedef npy_clongdouble npy_complex512;
#define NPY_FLOAT256_FMT NPY_LONGDOUBLE_FMT
#define NPY_COMPLEX512_FMT NPY_CLONGDOUBLE_FMT
#endif

/* datetime typedefs */
typedef npy_int64 npy_timedelta;
typedef npy_int64 npy_datetime;
#define NPY_DATETIME_FMT NPY_INT64_FMT
#define NPY_TIMEDELTA_FMT NPY_INT64_FMT

/* End of typedefs for numarray style bit-width names */

#if defined(__cplusplus)
}
#endif

#endif
