#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.61])
AC_INIT([libndarray], [1.0], [info@enthought.com])
AM_INIT_AUTOMAKE([foreign subdir-objects])
AC_CONFIG_FILES([Makefile])
AC_CONFIG_SRCDIR([src/npy_arrayobject.c])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CC
AC_PROG_LIBTOOL

# Define version constants
AC_DEFINE([API_VERSION], [2.0], [Core library API version])
AC_DEFINE([ABI_VERSION], [2.0], [Core library ABI version])

# Checks for header files.
AC_CHECK_HEADERS([complex.h])

# Checks for sizes
AC_CHECK_SIZEOF(short)
AC_CHECK_SIZEOF(int)
AC_CHECK_SIZEOF(long)
AC_CHECK_SIZEOF(float)
AC_CHECK_SIZEOF(double)
AC_CHECK_SIZEOF(long double)
AC_CHECK_SIZEOF(long long)
AC_CHECK_SIZEOF(void *)
AC_CHECK_SIZEOF(float complex,, [#include <complex.h>])
AC_CHECK_SIZEOF(double complex,, [#include <complex.h>])
AC_CHECK_SIZEOF(long double complex,, [#include <complex.h>])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_TYPE_UINTPTR_T()
AC_TYPE_LONG_LONG_INT()

# Other checks
AC_CHECK_DECLS([isnan, isfinite, isinf])
AC_CHECK_DECLS([sin, sinf, sinl, cos, cosf, cosl, tan, tanf, tanl,
                sinh, sinhf, sinhl, cosh, coshf, coshl, tanh, tanhf, tanhl,
                fabs, fabsf, fabsl, floor, floorf, floorl, ceil, ceilf, ceill,
                rint, rintf, rintl, trunc, truncf, truncl, sqrt, sqrtf, sqrtl,
                log10, log10f, log10l, log, logf, logl, exp, expf, expl,
                expm1, expm1f, expm1l, asin, asinf, asinl, acos, acosf, acosl,
                atan, atanf, atanl, asinh, asinhf, asinhl,
                acosh, acoshf, acoshl, atanh, atanhf, atanhl,
                log1p, log1pf, log1pl, exp2, exp2f, exp2l, log2, log2f, log2l,
                atan2, atan2f, atan2l, hypot, hypotf, hypotl, pow, powf, powl,
                fmod, fmodf, fmodl, copysign, copysignf, copysignl,
                modf, modff, modfl],,, [#include <math.h>])
AC_CHECK_DECLS([cabsf, cabsl, cargf, cargl, cexpf, cexpl,
                clogf, clogl, csqrtf, csqrtl, cpowf, cpowl, ccosf, ccosl,
                csinf, csinl],,, [#include <complex.h>])
AC_WORDS_BIGENDIAN()

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([localeconv memmove memset strdup strpbrk])

# AC_OUTPUT
AC_OUTPUT
