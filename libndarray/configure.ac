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

# Checks for libraries.

# Checks for header files.
AC_CHECK_HEADERS([locale.h memory.h stdint.h stdlib.h string.h strings.h])


AC_DEFINE(WORDS_BIGENDIAN, 1, [
    Define if your processor stores words with the most significant byte
    first (like Motorola and SPARC, unlike Intel).])
AC_DEFINE(HAVE_UINTPTR_T, 1, [Define if you have the \`uintptr_t' type.])
AC_CHECK_SIZEOF(long)



# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([localeconv memmove memset strdup strpbrk])

# AC_OUTPUT
AC_OUTPUT
