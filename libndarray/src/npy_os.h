#ifndef _NPY_OS_H_
#define _NPY_OS_H_

#include <stdlib.h>
#include <stdio.h>


#if defined(linux) || defined(__linux) || defined(__linux__)
    #define NPY_OS_LINUX
#elif defined(__FreeBSD__) || defined(__NetBSD__) || \
            defined(__OpenBSD__) || defined(__DragonFly__)
    #define NPY_OS_BSD
    #ifdef __FreeBSD__
        #define NPY_OS_FREEBSD
    #elif defined(__NetBSD__)
        #define NPY_OS_NETBSD
    #elif defined(__OpenBSD__)
        #define NPY_OS_OPENBSD
    #elif defined(__DragonFly__)
        #define NPY_OS_DRAGONFLY
    #endif
#elif defined(sun) || defined(__sun)
    #define NPY_OS_SOLARIS
#elif defined(__CYGWIN__)
    #define NPY_OS_CYGWIN
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    #define NPY_OS_WIN32
#elif defined(__APPLE__)
    #define NPY_OS_DARWIN
#else
    #define NPY_OS_UNKNOWN
#endif


int _npy_signbit_d(double x);

#define npy_signbit(x)                                   \
    (sizeof(x) == sizeof(double) ? _npy_signbit_d(x)     \
     : _npy_signbit_d((double) x))


#ifdef NPY_OS_WIN32
    #define NpyOS_snprintf _snprintf
#else
    #define NpyOS_snprintf snprintf
#endif

#define NpyOS_strtol strtol
#define NpyOS_strtoul strtoul

#define Npy_CHARMASK(c)  ((c) & 0xff)


char *
NpyOS_ascii_formatd(char *buffer, size_t buf_size,
                    const char *format,
                    double val, int decimal);

char *
NpyOS_ascii_formatf(char *buffer, size_t buf_size,
                    const char *format,
                    float val, int decimal);

char *
NpyOS_ascii_formatl(char *buffer, size_t buf_size,
                    const char *format,
                    long double val, int decimal);

double
NpyOS_ascii_strtod(const char *s, char** endptr);

int
NpyOS_ascii_ftolf(FILE *fp, double *value);

int
NpyOS_ascii_isspace(char c);

#endif
