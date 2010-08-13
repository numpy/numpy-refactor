#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.61])
AC_INIT([libndarray], [1.0], [info@enthought.com])
AM_INIT_AUTOMAKE([foreign subdir-objects])
AC_CONFIG_FILES([Makefile])
AC_CONFIG_SRCDIR([src/npy_arrayobject.c])
AC_CONFIG_HEADERS([npy_config.h])

# Checks for programs.
AC_PROG_CC
AC_PROG_LIBTOOL

# Checks for sizes
AC_CHECK_SIZEOF(short)
AC_CHECK_SIZEOF(int)
AC_CHECK_SIZEOF(long)
AC_CHECK_SIZEOF(float)
AC_CHECK_SIZEOF(double)
AC_CHECK_SIZEOF(long double)
AC_CHECK_SIZEOF(long long)
AC_CHECK_SIZEOF(void *)

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_TYPE_UINTPTR_T()
AC_TYPE_LONG_LONG_INT()

# Other checks
AC_CHECK_DECLS([isnan, isfinite, isinf])
AC_WORDS_BIGENDIAN()

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([localeconv memmove memset strdup strpbrk])

# AC_OUTPUT
AC_OUTPUT
