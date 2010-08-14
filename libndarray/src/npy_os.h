#ifndef _NPY_NPY_OS_H_
#define _NPY_NPY_OS_H_

#include <stdlib.h>
#include <stdio.h>


int _npy_signbit_d(double x);

#define npy_signbit(x)                                             \
    (sizeof(x) == sizeof(double) ? _npy_signbit_d(x)               \
     : _npy_signbit_d((double) x))


#define NumPyOS_snprintf snprintf
#define NumPyOS_strtol strtol
#define NumPyOS_strtoul strtoul


char *
NumPyOS_ascii_formatd(char *buffer, size_t buf_size,
                      const char *format,
                      double val, int decimal);

char *
NumPyOS_ascii_formatf(char *buffer, size_t buf_size,
                      const char *format,
                      float val, int decimal);

char *
NumPyOS_ascii_formatl(char *buffer, size_t buf_size,
                      const char *format,
                      long double val, int decimal);

double
NumPyOS_ascii_strtod(const char *s, char** endptr);

int
NumPyOS_ascii_ftolf(FILE *fp, double *value);

int
NumPyOS_ascii_isspace(char c);

#endif
