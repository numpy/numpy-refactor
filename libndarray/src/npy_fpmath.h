#ifndef _NPY_FPMATH_H_
#define _NPY_FPMATH_H_

#include "npy_cpu.h"
#include "npy_common.h"


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

#ifdef NPY_OS_DARWIN
    /* This hardcoded logic is fragile, but universal builds makes it
       difficult to detect arch-specific features */

    /* MAC OS X < 10.4 and gcc < 4 does not support proper long double, and
       is the same as double on those platforms */
    #if NPY_BITSOF_LONGDOUBLE == NPY_BITSOF_DOUBLE
        /* This assumes that FPU and ALU have the same endianness */
        #if NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN
            #define HAVE_LDOUBLE_IEEE_DOUBLE_LE
        #elif NPY_BYTE_ORDER == NPY_BIG_ENDIAN
            #define HAVE_LDOUBLE_IEEE_DOUBLE_BE
        #else
            #error Endianness undefined ?
        #endif
    #else
        #if defined(NPY_CPU_X86)
            #define HAVE_LDOUBLE_INTEL_EXTENDED_12_BYTES_LE
        #elif defined(NPY_CPU_AMD64)
            #define HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE
        #elif defined(NPY_CPU_PPC) || defined(NPY_CPU_PPC64)
            #define HAVE_LDOUBLE_IEEE_DOUBLE_16_BYTES_BE
        #endif
    #endif
#else
    /* TODO: There is code in scons_support.py which checks for this, which
             still need to be converted such that configure defines this.
             Just use this definition for now */
    #define HAVE_LDOUBLE_INTEL_EXTENDED_12_BYTES_LE
#endif

#if !(defined(HAVE_LDOUBLE_IEEE_QUAD_BE) || \
      defined(HAVE_LDOUBLE_IEEE_QUAD_LE) || \
      defined(HAVE_LDOUBLE_IEEE_DOUBLE_LE) || \
      defined(HAVE_LDOUBLE_IEEE_DOUBLE_BE) || \
      defined(HAVE_LDOUBLE_IEEE_DOUBLE_16_BYTES_BE) || \
      defined(HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE) || \
      defined(HAVE_LDOUBLE_INTEL_EXTENDED_12_BYTES_LE))
    #error No long double representation defined
#endif


#endif
