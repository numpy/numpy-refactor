/*
 *  npy_common.c - 
 *  
 */

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"



int 
Npy_IsAligned(NpyArray *ap)
{
    int i, alignment, aligned = 1;
    npy_intp ptr;
    
    /* The special casing for STRING and VOID types was removed
     * in accordance with http://projects.scipy.org/numpy/ticket/1227
     * It used to be that IsAligned always returned True for these
     * types, which is indeed the case when they are created using
     * PyArray_DescrConverter(), but not necessarily when using
     * PyArray_DescrAlignConverter(). */
    
    alignment = ap->descr->alignment;
    if (alignment == 1) {
        return 1;
    }
    ptr = (npy_intp) ap->data;
    aligned = (ptr % alignment) == 0;
    for (i = 0; i < ap->nd; i++) {
        aligned &= ((ap->strides[i] % alignment) == 0);
    }
    return aligned != 0;
}


npy_bool 
Npy_IsWriteable(NpyArray *ap)
{
    NpyArray *base = ap->base;
    void *dummy;
    size_t n;
    
    /* If we own our own data, then no-problem */
    if ((base == NULL) || (ap->flags & NPY_OWNDATA)) {
        return NPY_TRUE;
    }
    /*
     * Get to the final base object
     * If it is a writeable array, then return TRUE
     * If we can find an array object
     * or a writeable buffer object as the final base object
     * or a string object (for pickling support memory savings).
     * - this last could be removed if a proper pickleable
     * buffer was added to Python.
     */
    
    while(NpyArray_Check(base)) {
        if (NpyArray_CHKFLAGS(base, NPY_OWNDATA)) {
            return (npy_bool) (NpyArray_ISWRITEABLE(base));
        }
        base = NpyArray_BASE(base);
    }
    
    /*
     * here so pickle support works seamlessly
     * and unpickled array can be set and reset writeable
     * -- could be abused --
     */
    /* TODO: Handling for string and buffers as the array base, related to pickling. */
    if (NpyString_Check(base)) {
        return NPY_TRUE;
    }
    if (NpyObject_AsWriteBuffer(base, &dummy, &n) < 0) {
        return NPY_FALSE;
    }
    return NPY_TRUE;
}
