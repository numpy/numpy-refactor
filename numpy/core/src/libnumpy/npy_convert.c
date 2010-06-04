/*
 *  npy_convert.c - 
 *  
 */

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"





int NpyArray_ToBinaryFile(NpyArray *self, FILE *fp)
{
    npy_intp size;
    npy_intp n;
    PyArrayIterObject *it;
        
    /* binary data */
    if (NpyDataType_FLAGCHK(self->descr, NPY_LIST_PICKLE)) {
        NpyErr_SetString(NpyExc_ValueError, "cannot write " \
                         "object arrays to a file in "   \
                         "binary mode");
        return -1;
    }
    
    if (NpyArray_ISCONTIGUOUS(self)) {
        size = NpyArray_SIZE(self);
        NPY_BEGIN_ALLOW_THREADS;
        n = fwrite((const void *)self->data,
                   (size_t) self->descr->elsize,
                   (size_t) size, fp);
        NPY_END_ALLOW_THREADS;
        if (n < size) {
            NpyErr_Format(NpyExc_ValueError,
                          "%ld requested and %ld written",
                          (long) size, (long) n);
            return -1;
        }
    }
    else {
        NPY_BEGIN_THREADS_DEF;
        
        it = NpyArray_IterNew(self);
        NPY_BEGIN_THREADS;
        while (it->index < it->size) {
            if (fwrite((const void *)it->dataptr,
                       (size_t) self->descr->elsize,
                       1, fp) < 1) {
                NPY_END_THREADS;
                NpyErr_Format(NpyExc_IOError,
                              "problem writing element"\
                              " %"NPY_INTP_FMT" to file",
                              it->index);
                Py_DECREF(it);
                return -1;
            }
            NpyArray_ITER_NEXT(it);
        }
        NPY_END_THREADS;
        Npy_DECREF(it);
    }
    return 0;
}