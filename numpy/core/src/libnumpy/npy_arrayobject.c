
#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"

/*
 * Compute the size of an array (in number of items)
 */
npy_intp
NpyArray_Size(NpyArray *op)
{
    return NpyArray_SIZE(op);
}

int
NpyArray_CompareUCS4(npy_ucs4 *s1, npy_ucs4 *s2, size_t len)
{
    npy_ucs4 c1, c2;
    while(len-- > 0) {
        c1 = *s1++;
        c2 = *s2++;
        if (c1 != c2) {
            return (c1 < c2) ? -1 : 1;
        }
    }
    return 0;
}

int
NpyArray_CompareString(char *s1, char *s2, size_t len)
{
    const unsigned char *c1 = (unsigned char *)s1;
    const unsigned char *c2 = (unsigned char *)s2;
    size_t i;

    for(i = 0; i < len; ++i) {
        if (c1[i] != c2[i]) {
            return (c1[i] > c2[i]) ? 1 : -1;
        }
    }
    return 0;
}

int
NpyArray_ElementStrides(NpyArray *arr)
{
    int itemsize = NpyArray_ITEMSIZE(arr);
    int i, N = NpyArray_NDIM(arr);
    npy_intp *strides = PyArray_STRIDES(arr);

    for (i = 0; i < N; i++) {
        if ((strides[i] % itemsize) != 0) {
            return 0;
        }
    }
    return 1;
}


/*
 * This routine checks to see if newstrides (of length nd) will not
 * ever be able to walk outside of the memory implied numbytes and offset.
 *
 * The available memory is assumed to start at -offset and proceed
 * to numbytes-offset.  The strides are checked to ensure
 * that accessing memory using striding will not try to reach beyond
 * this memory for any of the axes.
 *
 * If numbytes is 0 it will be calculated using the dimensions and
 * element-size.
 *
 * This function checks for walking beyond the beginning and right-end
 * of the buffer and therefore works for any integer stride (positive
 * or negative).
 */
npy_bool
NpyArray_CheckStrides(int elsize, int nd, npy_intp numbytes, npy_intp offset,
                      npy_intp *dims, npy_intp *newstrides)
{
    int i;
    npy_intp byte_begin;
    npy_intp begin;
    npy_intp end;

    if (numbytes == 0) {
        numbytes = NpyArray_MultiplyList(dims, nd) * elsize;
    }
    begin = -offset;
    end = numbytes - offset - elsize;
    for (i = 0; i < nd; i++) {
        byte_begin = newstrides[i]*(dims[i] - 1);
        if ((byte_begin < begin) || (byte_begin > end)) {
            return NPY_FALSE;
        }
    }
    return NPY_TRUE;
}
