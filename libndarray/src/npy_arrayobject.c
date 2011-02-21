/* npy_arrayobject.c */

#include <stdlib.h>
#include <string.h>
#include "npy_config.h"
#include "npy_api.h"
#include "npy_arrayobject.h"
#include "npy_iterators.h"
#include "npy_internal.h"

#include "npy_dict.h"

/* TODO: Make these into interface functions */
extern int PyArray_INCREF(void *);
extern int PyArray_XDECREF(void *);

/*
 * Compute the size of an array (in number of items)
 */
NDARRAY_API npy_intp
NpyArray_Size(NpyArray *op)
{
    assert(NPY_VALID_MAGIC == op->nob_magic_number);
    return NpyArray_SIZE(op);
}

NDARRAY_API int
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

NDARRAY_API int
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

NDARRAY_API int
NpyArray_ElementStrides(NpyArray *arr)
{
    int itemsize = NpyArray_ITEMSIZE(arr);
    int i, N = NpyArray_NDIM(arr);
    npy_intp *strides = NpyArray_STRIDES(arr);

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
NDARRAY_API npy_bool
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

NDARRAY_API void
NpyArray_ForceUpdate(NpyArray* self)
{
    if ((self->flags&NPY_UPDATEIFCOPY) && self->base_arr != NULL) {
        /*
         * UPDATEIFCOPY means that base points to an
         * array that should be updated with the contents
         * of this array upon destruction.
         * self->base->flags must have been WRITEABLE
         * (checked previously) and it was locked here
         * thus, unlock it.
         */
        if (self->flags & NPY_UPDATEIFCOPY) {
            self->flags &= ~NPY_UPDATEIFCOPY;
            self->base_arr->flags |= NPY_WRITEABLE;
            Npy_INCREF(self); /* hold on to self in next call */
            if (NpyArray_CopyAnyInto(self->base_arr, self) < 0) {
                /* NpyErr_Print(); */
                NpyErr_Clear();
            }
            Npy_DECREF(self);
            Npy_DECREF(self->base_arr);
            self->base_arr = NULL;
        }
    }
}

/* This also handles possibly mis-aligned data */
/* Compare s1 and s2 which are not necessarily NULL-terminated.
   s1 is of length len1
   s2 is of length len2
   If they are NULL terminated, then stop comparison.
*/
static int
_myunincmp(NpyArray_UCS4 *s1, NpyArray_UCS4 *s2, int len1, int len2)
{
    NpyArray_UCS4 *sptr;
    NpyArray_UCS4 *s1t=s1, *s2t=s2;
    int val;
    npy_intp size;
    int diff;

    if ((npy_intp)s1 % sizeof(NpyArray_UCS4) != 0) {
        size = len1*sizeof(NpyArray_UCS4);
        s1t = malloc(size);
        memcpy(s1t, s1, size);
    }
    if ((npy_intp)s2 % sizeof(NpyArray_UCS4) != 0) {
        size = len2*sizeof(NpyArray_UCS4);
        s2t = malloc(size);
        memcpy(s2t, s2, size);
    }
    val = NpyArray_CompareUCS4(s1t, s2t, NpyArray_MIN(len1,len2));
    if ((val != 0) || (len1 == len2)) {
        goto finish;
    }
    if (len2 > len1) {
        sptr = s2t+len1;
        val = -1;
        diff = len2-len1;
    }
    else {
        sptr = s1t+len2;
        val = 1;
        diff=len1-len2;
    }
    while (diff--) {
        if (*sptr != 0) {
            goto finish;
        }
        sptr++;
    }
    val = 0;

 finish:
    if (s1t != s1) {
        free(s1t);
    }
    if (s2t != s2) {
        free(s2t);
    }
    return val;
}




/*
 * Compare s1 and s2 which are not necessarily NULL-terminated.
 * s1 is of length len1
 * s2 is of length len2
 * If they are NULL terminated, then stop comparison.
 */
static int
_mystrncmp(char *s1, char *s2, int len1, int len2)
{
    char *sptr;
    int val;
    int diff;

    val = memcmp(s1, s2, NpyArray_MIN(len1, len2));
    if ((val != 0) || (len1 == len2)) {
        return val;
    }
    if (len2 > len1) {
        sptr = s2 + len1;
        val = -1;
        diff = len2 - len1;
    }
    else {
        sptr = s1 + len2;
        val = 1;
        diff = len1 - len2;
    }
    while (diff--) {
        if (*sptr != 0) {
            return val;
        }
        sptr++;
    }
    return 0; /* Only happens if NULLs are everywhere */
}


/* Borrowed from Numarray */

#define SMALL_STRING 2048

#if defined(isspace)
#undef isspace
#define isspace(c)  ((c==' ')||(c=='\t')||(c=='\n')||(c=='\r')||(c=='\v')||(c=='\f'))
#endif

static void _rstripw(char *s, int n)
{
    int i;
    for (i = n - 1; i >= 1; i--) { /* Never strip to length 0. */
        int c = s[i];

        if (!c || isspace(c)) {
            s[i] = 0;
        }
        else {
            break;
        }
    }
}

static void _unistripw(NpyArray_UCS4 *s, int n)
{
    int i;
    for (i = n - 1; i >= 1; i--) { /* Never strip to length 0. */
        NpyArray_UCS4 c = s[i];
        if (!c || isspace(c)) {
            s[i] = 0;
        }
        else {
            break;
        }
    }
}


static char *
_char_copy_n_strip(char *original, char *temp, int nc)
{
    if (nc > SMALL_STRING) {
        temp = (char*)malloc(nc);
        if (!temp) {
            NpyErr_MEMORY;
            return NULL;
        }
    }
    memcpy(temp, original, nc);
    _rstripw(temp, nc);
    return temp;
}

static void
_char_release(char *ptr, int nc)
{
    if (nc > SMALL_STRING) {
        free(ptr);
    }
}

static char *
_uni_copy_n_strip(char *original, char *temp, int nc)
{
    if (nc*sizeof(NpyArray_UCS4) > SMALL_STRING) {
        temp = (char*)malloc(nc*sizeof(NpyArray_UCS4));
        if (!temp) {
            NpyErr_MEMORY;
            return NULL;
        }
    }
    memcpy(temp, original, nc*sizeof(NpyArray_UCS4));
    _unistripw((NpyArray_UCS4 *)temp, nc);
    return temp;
}

static void
_uni_release(char *ptr, int nc)
{
    if (nc*sizeof(NpyArray_UCS4) > SMALL_STRING) {
        free(ptr);
    }
}

/* End borrowed from numarray */

#define _rstrip_loop(CMP) {                                     \
        void *aptr, *bptr;                                      \
        char atemp[SMALL_STRING], btemp[SMALL_STRING];          \
        while(size--) {                                         \
            aptr = stripfunc(iself->dataptr, atemp, N1);        \
            if (!aptr) return -1;                               \
            bptr = stripfunc(iother->dataptr, btemp, N2);       \
            if (!bptr) {                                        \
                relfunc(aptr, N1);                              \
                return -1;                                      \
            }                                                   \
            val = cmpfunc(aptr, bptr, N1, N2);                  \
            *dptr = (val CMP 0);                                \
            NpyArray_ITER_NEXT(iself);                          \
            NpyArray_ITER_NEXT(iother);                         \
            dptr += 1;                                          \
            relfunc(aptr, N1);                                  \
            relfunc(bptr, N2);                                  \
        }                                                       \
    }

#define _reg_loop(CMP) {                                \
        while(size--) {                                 \
            val = cmpfunc((void *)iself->dataptr,       \
                          (void *)iother->dataptr,      \
                          N1, N2);                      \
            *dptr = (val CMP 0);                        \
            NpyArray_ITER_NEXT(iself);                  \
            NpyArray_ITER_NEXT(iother);                 \
            dptr += 1;                                  \
        }                                               \
    }

#define _loop(CMP) if (rstrip) _rstrip_loop(CMP)        \
        else _reg_loop(CMP)

static int
_compare_strings(NpyArray *result, NpyArrayMultiIterObject *multi,
                 int cmp_op, void *func, int rstrip)
{
    NpyArrayIterObject *iself, *iother;
    npy_bool *dptr;
    npy_intp size;
    int val;
    int N1, N2;
    int (*cmpfunc)(void *, void *, int, int);
    void (*relfunc)(char *, int);
    char* (*stripfunc)(char *, char *, int);

    cmpfunc = func;
    dptr = (npy_bool *)NpyArray_DATA(result);
    iself = multi->iters[0];
    iother = multi->iters[1];
    size = multi->size;
    N1 = iself->ao->descr->elsize;
    N2 = iother->ao->descr->elsize;
    if ((void *)cmpfunc == (void *)_myunincmp) {
        N1 >>= 2;
        N2 >>= 2;
        stripfunc = _uni_copy_n_strip;
        relfunc = _uni_release;
    }
    else {
        stripfunc = _char_copy_n_strip;
        relfunc = _char_release;
    }
    switch (cmp_op) {
    case NPY_EQ:
        _loop(==)
            break;
    case NPY_NE:
        _loop(!=)
            break;
    case NPY_LT:
        _loop(<)
            break;
    case NPY_LE:
        _loop(<=)
            break;
    case NPY_GT:
        _loop(>)
            break;
    case NPY_GE:
        _loop(>=)
            break;
    default:
        NpyErr_SetString(NpyExc_RuntimeError, "bad comparison operator");
        return -1;
    }
    return 0;
}

#undef _loop
#undef _reg_loop
#undef _rstrip_loop
#undef SMALL_STRING

NDARRAY_API NpyArray*
NpyArray_CompareStringArrays(NpyArray* a1, NpyArray* a2, int cmp_op, int rstrip) 
{
    NpyArray* result;
    int val;
    NpyArrayMultiIterObject* mit;

    int t1 = NpyArray_TYPE(a1);
    int t2 = NpyArray_TYPE(a2);

    if (NpyArray_TYPE(a1) != NpyArray_TYPE(a2) || 
        (t1 != NPY_UNICODE && t1 != NPY_STRING && t1 != NPY_VOID)) {
        NpyErr_SetString(NpyExc_ValueError, 
                         "Arrays must be of the same string type.");
        return NULL;
    }

    /* Broad-cast the arrays to a common shape */
    mit = NpyArray_MultiIterFromArrays(NULL, 0, 2, a1, a2);
    if (mit == NULL) {
        return NULL;
    }

    result = NpyArray_NewFromDescr(NpyArray_DescrFromType(NPY_BOOL),
                                   mit->nd,
                                   mit->dimensions, NULL,
                                   NULL, 0, NPY_TRUE, NULL, NULL);
    if (result == NULL) {
        goto finish;
    }

    if (t1 == NPY_UNICODE) {
        val = _compare_strings(result, mit, cmp_op, _myunincmp, rstrip);
    }
    else {
        val = _compare_strings(result, mit, cmp_op, _mystrncmp, rstrip);
    }

    if (val < 0) {
        Npy_DECREF(result);
        result = NULL;
    }

 finish:
    Npy_DECREF(mit);
    return result;
}

/* Deallocs & destroy's the array object.
 *  Returns whether or not we did an artificial incref
 *  so we can keep track of the total refcount for debugging.
 */
/* TODO: For now caller is expected to call _array_dealloc_buffer_info
         and clear weak refs.  Need to revisit. */
NDARRAY_API int
NpyArray_dealloc(NpyArray *self)
{
    int i;

    int result = 0;

    assert(NPY_VALID_MAGIC == self->nob_magic_number);
    assert(NULL == self->base_arr ||
           NPY_VALID_MAGIC == self->base_arr->nob_magic_number);

    if (NULL != self->base_arr) {
        /*
         * UPDATEIFCOPY means that base points to an
         * array that should be updated with the contents
         * of this array upon destruction.
         * self->base->flags must have been WRITEABLE
         * (checked previously) and it was locked here
         * thus, unlock it.
         */
        if (self->flags & NPY_UPDATEIFCOPY) {
            self->base_arr->flags |= NPY_WRITEABLE;
            Npy_INCREF(self); /* hold on to self in next call */
            if (NpyArray_CopyAnyInto(self->base_arr, self) < 0) {
                /* NpyErr_Print(); */
                NpyErr_Clear();
            }
            /*
             * Don't need to DECREF -- because we are deleting
             *self already...
             */
            result = 1;
        }
        /*
         * In any case base is pointing to something that we need
         * to DECREF -- either a view or a buffer object
         */
        Npy_DECREF(self->base_arr);
        self->base_arr = NULL;
    } else if (NULL != self->base_obj) {
        NpyInterface_DECREF(self->base_obj);
        self->base_obj = NULL;
    }

    if ((self->flags & NPY_OWNDATA) && self->data) {
        /* Free internal references if an Object array */
        if (NpyDataType_FLAGCHK(self->descr, NPY_ITEM_REFCOUNT)) {
            Npy_INCREF(self); /* hold on to self in next call */
            NpyArray_XDECREF(self);
            /*
             * Don't need to DECREF -- because we are deleting
             * self already...
             */
            if (self->nob_refcnt == 1) {
                result = 1;
            }
        }
        fflush(stdout);

        NpyDataMem_FREE(self->data);
        self->data =NULL;
    }

    if (NULL != self->dimensions) {
        NpyDimMem_FREE(self->dimensions);
    }
    
    Npy_DECREF(self->descr);
    /* Flag that this object is now deallocated. */
    self->nob_magic_number = NPY_INVALID_MAGIC;

    NpyArray_free(self);

    return result;
}


NpyTypeObject NpyArray_Type = {
    (npy_destructor)NpyArray_dealloc,
    NULL
};
