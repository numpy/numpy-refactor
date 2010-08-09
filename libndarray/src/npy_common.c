/*
 *  npy_common.c -
 *
 */

#define _MULTIARRAYMODULE
#include "npy_config.h"
#include "numpy/numpy_api.h"
#include "numpy/npy_arrayobject.h"


extern int PyArray_INCREF(void *);     /* TODO: Make these into interface functions */
extern int PyArray_XDECREF(void *);


/* TODO: We should be getting this from an include. */
#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif

int
Npy_IsAligned(NpyArray *ap)
{
    int i, alignment, aligned = 1;
    npy_intp ptr;

    /* The special casing for STRING and VOID types was removed
     * in accordance with http://projects.scipy.org/numpy/ticket/1227
     * It used to be that IsAligned always returned True for these
     * types, which is indeed the case when they are created using
     * NpyArray_DescrConverter(), but not necessarily when using
     * NpyArray_DescrAlignConverter(). */

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

/* TODO: Remove these declarations once pickling code below is refactored into
 * the interface
#include <Python.h>
*/

npy_bool
Npy_IsWriteable(NpyArray *ap)
{
    NpyArray *base_arr = ap->base_arr;
    void *base_obj = ap->base_obj;

    /* If we own our own data, then no-problem */
    if ((base_arr == NULL && NULL == base_obj) || (ap->flags & NPY_OWNDATA)) {
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

    while (NULL != base_arr) {
        if (NpyArray_CHKFLAGS(base_arr, NPY_OWNDATA)) {
            return (npy_bool) (NpyArray_ISWRITEABLE(base_arr));
        }
        base_arr = base_arr->base_arr;
        base_obj = base_arr->base_obj;
    }

    /*
     * here so pickle support works seamlessly
     * and unpickled array can be set and reset writeable
     * -- could be abused --
     */
#if 0  /* XXX */
    /* TODO: How is this related to pickling? Need to promote to interface layer to determine if opaque obj is writable. */
    if (NpyString_Check(base_obj)) {
        return NPY_TRUE;
    }
    if (NpyObject_AsWriteBuffer(base_obj, &dummy, &n) < 0) {
        return NPY_FALSE;
    }
#endif
    return NPY_TRUE;
}

/*
 * new reference
 * doesn't alter refcount of chktype or mintype ---
 * unless one of them is returned
 * TODO: Come up with a name that means something.
 */
#if 0
/* TODO: Dead code, duplicate in npy_common.c */
NpyArray_Descr *
NpyArray_SmallType(NpyArray_Descr *chktype, NpyArray_Descr *mintype)
{
    NpyArray_Descr *outtype;
    int outtype_num, save_num;

    if (NpyArray_EquivTypes(chktype, mintype)) {
        _Npy_INCREF(mintype);
        return mintype;
    }


    if (chktype->type_num > mintype->type_num) {
        outtype_num = chktype->type_num;
    }
    else {
        if (NpyDataType_ISOBJECT(chktype) &&
            NpyDataType_ISSTRING(mintype)) {
            return NpyArray_DescrFromType(NPY_OBJECT);
        }
        else {
            outtype_num = mintype->type_num;
        }
    }

    save_num = outtype_num;
    while (outtype_num < NPY_NTYPES &&
          !(NpyArray_CanCastSafely(chktype->type_num, outtype_num)
            && NpyArray_CanCastSafely(mintype->type_num, outtype_num))) {
        outtype_num++;
    }
    if (outtype_num == NPY_NTYPES) {
        outtype = NpyArray_DescrFromType(save_num);
    }
    else {
        outtype = NpyArray_DescrFromType(outtype_num);
    }
    if (NpyTypeNum_ISEXTENDED(outtype->type_num)) {
        int testsize = outtype->elsize;
        int chksize, minsize;
        chksize = chktype->elsize;
        minsize = mintype->elsize;
        /*
         * Handle string->unicode case separately
         * because string itemsize is 4* as large
         */
        if (outtype->type_num == NPY_UNICODE &&
            mintype->type_num == NPY_STRING) {
            testsize = MAX(chksize, 4*minsize);
        }
        else if (chktype->type_num == NPY_STRING &&
                 mintype->type_num == NPY_UNICODE) {
            testsize = MAX(chksize*4, minsize);
        }
        else {
            testsize = MAX(chksize, minsize);
        }
        if (testsize != outtype->elsize) {
            NpyArray_DESCR_REPLACE(outtype);
            outtype->elsize = testsize;
            NpyArray_DescrDeallocNamesAndFields(outtype);
        }
    }
    return outtype;
}
#endif

char *
NpyArray_Index2Ptr(NpyArray *mp, npy_intp i)
{
    npy_intp dim0;

    if (NpyArray_NDIM(mp) == 0) {
        NpyErr_SetString(NpyExc_IndexError, "0-d arrays can't be indexed");
        return NULL;
    }
    dim0 = NpyArray_DIM(mp, 0);
    if (i < 0) {
        i += dim0;
    }
    if (i == 0 && dim0 > 0) {
        return NpyArray_BYTES(mp);
    }
    if (i > 0 && i < dim0) {
        return NpyArray_BYTES(mp)+i*NpyArray_STRIDE(mp, 0);
    }
    NpyErr_SetString(NpyExc_IndexError,"index out of bounds");
    return NULL;
}
