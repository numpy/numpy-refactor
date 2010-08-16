#ifndef _NPY_NOPREFIX_H_
#define _NPY_NOPREFIX_H_


#define longlong    npy_longlong
#define ulonglong   npy_ulonglong
#define Bool        npy_bool
#define longdouble  npy_longdouble
#define byte        npy_byte

#ifndef _BSD_SOURCE
#define ushort      npy_ushort
#define uint        npy_uint
#define ulong       npy_ulong
#endif

#define ubyte       npy_ubyte
#define ushort      npy_ushort
#define uint        npy_uint
#define ulong       npy_ulong
#define cfloat      npy_cfloat
#define cdouble     npy_cdouble
#define clongdouble npy_clongdouble
#define Int8        npy_int8
#define UInt8       npy_uint8
#define Int16       npy_int16
#define UInt16      npy_uint16
#define Int32       npy_int32
#define UInt32      npy_uint32
#define Int64       npy_int64
#define UInt64      npy_uint64
#define Int128      npy_int128
#define UInt128     npy_uint128
#define Int256      npy_int256
#define UInt256     npy_uint256
#define Float16     npy_float16
#define Complex32   npy_complex32
#define Float32     npy_float32
#define Complex64   npy_complex64
#define Float64     npy_float64
#define Complex128  npy_complex128
#define Float80     npy_float80
#define Complex160  npy_complex160
#define Float96     npy_float96
#define Complex192  npy_complex192
#define Float128    npy_float128
#define Complex256  npy_complex256
#define intp        npy_intp
#define uintp       npy_uintp
#define datetime    npy_datetime
#define timedelta   npy_timedelta

#define LONGLONG_FMT NPY_LONGLONG_FMT
#define ULONGLONG_FMT NPY_ULONGLONG_FMT
#define LONGLONG_SUFFIX NPY_LONGLONG_SUFFIX
#define ULONGLONG_SUFFIX NPY_ULONGLONG_SUFFIX(x)

#define PyArray_UCS4 npy_ucs4

#endif
