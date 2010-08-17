#ifndef _NPY_NPY_OS_H_
#define _NPY_NPY_OS_H_

#include <stdlib.h>
#include <stdio.h>


int _npy_signbit_d(double x);

#define npy_signbit(x)                                             \
    (sizeof(x) == sizeof(double) ? _npy_signbit_d(x)               \
     : _npy_signbit_d((double) x))


#define NpyOS_snprintf snprintf
#define NpyOS_strtol strtol
#define NpyOS_strtoul strtoul


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
