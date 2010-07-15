/*
 *  npy_ctors.c - 
 *  
 * */

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"







static void 
_unaligned_strided_byte_move(char *dst, npy_intp outstrides, char *src,
                             npy_intp instrides, npy_intp N, int elsize)
{
    npy_intp i;
    char *tout = dst;
    char *tin = src;
    
    
#define _MOVE_N_SIZE(size)                      \
for(i=0; i<N; i++) {                       \
memmove(tout, tin, size);               \
tin += instrides;                       \
tout += outstrides;                     \
}                                           \
return
    
    switch(elsize) {
        case 8:
            _MOVE_N_SIZE(8);
        case 4:
            _MOVE_N_SIZE(4);
        case 1:
            _MOVE_N_SIZE(1);
        case 2:
            _MOVE_N_SIZE(2);
        case 16:
            _MOVE_N_SIZE(16);
        default:
            _MOVE_N_SIZE(elsize);
    }
#undef _MOVE_N_SIZE
    
}

void 
_unaligned_strided_byte_copy(char *dst, npy_intp outstrides, char *src,
                             npy_intp instrides, npy_intp N, int elsize)
{
    npy_intp i;
    char *tout = dst;
    char *tin = src;
    
#define _COPY_N_SIZE(size)                      \
for(i=0; i<N; i++) {                       \
memcpy(tout, tin, size);                \
tin += instrides;                       \
tout += outstrides;                     \
}                                           \
return
    
    switch(elsize) {
        case 8:
            _COPY_N_SIZE(8);
        case 4:
            _COPY_N_SIZE(4);
        case 1:
            _COPY_N_SIZE(1);
        case 2:
            _COPY_N_SIZE(2);
        case 16:
            _COPY_N_SIZE(16);
        default:
            _COPY_N_SIZE(elsize);
    }
#undef _COPY_N_SIZE
    
}


static void
_strided_byte_copy(char *dst, npy_intp outstrides, char *src, npy_intp instrides,
                   npy_intp N, int elsize)
{
    npy_intp i, j;
    char *tout = dst;
    char *tin = src;
    
#define _FAST_MOVE(_type_)                              \
for(i=0; i<N; i++) {                               \
((_type_ *)tout)[0] = ((_type_ *)tin)[0];       \
tin += instrides;                               \
tout += outstrides;                             \
}                                                   \
return
    
    switch(elsize) {
        case 8:
            _FAST_MOVE(npy_int64);
        case 4:
            _FAST_MOVE(npy_int32);
        case 1:
            _FAST_MOVE(npy_int8);
        case 2:
            _FAST_MOVE(npy_int16);
        case 16:
            for (i = 0; i < N; i++) {
                ((npy_int64 *)tout)[0] = ((npy_int64 *)tin)[0];
                ((npy_int64 *)tout)[1] = ((npy_int64 *)tin)[1];
                tin += instrides;
                tout += outstrides;
            }
            return;
        default:
            for(i = 0; i < N; i++) {
                for(j=0; j<elsize; j++) {
                    *tout++ = *tin++;
                }
                tin = tin + instrides - elsize;
                tout = tout + outstrides - elsize;
            }
    }
#undef _FAST_MOVE
    
}



void 
_strided_byte_swap(void *p, npy_intp stride, npy_intp n, int size)
{
    char *a, *b, c = 0;
    int j, m;
    
    switch(size) {
        case 1: /* no byteswap necessary */
            break;
        case 4:
            for (a = (char*)p; n > 0; n--, a += stride - 1) {
                b = a + 3;
                c = *a; *a++ = *b; *b-- = c;
                c = *a; *a = *b; *b   = c;
            }
            break;
        case 8:
            for (a = (char*)p; n > 0; n--, a += stride - 3) {
                b = a + 7;
                c = *a; *a++ = *b; *b-- = c;
                c = *a; *a++ = *b; *b-- = c;
                c = *a; *a++ = *b; *b-- = c;
                c = *a; *a = *b; *b   = c;
            }
            break;
        case 2:
            for (a = (char*)p; n > 0; n--, a += stride) {
                b = a + 1;
                c = *a; *a = *b; *b = c;
            }
            break;
        default:
            m = size/2;
            for (a = (char *)p; n > 0; n--, a += stride - m) {
                b = a + (size - 1);
                for (j = 0; j < m; j++) {
                    c=*a; *a++ = *b; *b-- = c;
                }
            }
            break;
    }
}


NPY_NO_EXPORT void
byte_swap_vector(void *p, npy_intp n, int size)
{
    _strided_byte_swap(p, (npy_intp) size, n, size);
    return;
}



static int
_copy_from_same_shape(NpyArray *dest, NpyArray *src,
                      void (*myfunc)(char *, npy_intp, char *, npy_intp, npy_intp, int),
                      int swap)
{
    int maxaxis = -1, elsize;
    npy_intp maxdim;
    NpyArrayIterObject *dit, *sit;
    NPY_BEGIN_THREADS_DEF;
    
    dit = NpyArray_IterAllButAxis(dest, &maxaxis);
    sit = NpyArray_IterAllButAxis(src, &maxaxis);
    
    maxdim = dest->dimensions[maxaxis];
    
    if ((dit == NULL) || (sit == NULL)) {
        _Npy_XDECREF(dit);
        _Npy_XDECREF(sit);
        return -1;
    }
    elsize = NpyArray_ITEMSIZE(dest);
    
    /* Refcount note: src and dst have the same size */
    NpyArray_INCREF(src);
    NpyArray_XDECREF(dest);
    
    NPY_BEGIN_THREADS;
    while(dit->index < dit->size) {
        /* strided copy of elsize bytes */
        myfunc(dit->dataptr, dest->strides[maxaxis],
               sit->dataptr, src->strides[maxaxis],
               maxdim, elsize);
        if (swap) {
            _strided_byte_swap(dit->dataptr,
                               dest->strides[maxaxis],
                               dest->dimensions[maxaxis],
                               elsize);
        }
        NpyArray_ITER_NEXT(dit);
        NpyArray_ITER_NEXT(sit);
    }
    NPY_END_THREADS;
    
    _Npy_DECREF(sit);
    _Npy_DECREF(dit);
    return 0;
}




static int
_broadcast_copy(NpyArray *dest, NpyArray *src,
                void (*myfunc)(char *, npy_intp, char *, npy_intp, npy_intp, int),
                int swap)
{
    int elsize;
    NpyArrayMultiIterObject *multi;
    int maxaxis; 
    npy_intp maxdim;
    NPY_BEGIN_THREADS_DEF;
    
    elsize = NpyArray_ITEMSIZE(dest);
    multi = NpyArray_MultiIterFromArrays(NULL, 0, 2, dest, src);
    if (multi == NULL) {
        return -1;
    }
    
    if (multi->size != NpyArray_SIZE(dest)) {
        NpyErr_SetString(NpyExc_ValueError,
                        "array dimensions are not "\
                        "compatible for copy");
        _Npy_DECREF(multi);
        return -1;
    }
    
    maxaxis = NpyArray_RemoveSmallest(multi);
    if (maxaxis < 0) {
        /*
         * copy 1 0-d array to another
         * Refcount note: src and dst have the same size
         */
        NpyArray_INCREF(src);
        NpyArray_XDECREF(dest);
        memcpy(dest->data, src->data, elsize);
        if (swap) {
            byte_swap_vector(dest->data, 1, elsize);
        }
        return 0;
    }
    maxdim = multi->dimensions[maxaxis];
    
    /*
     * Increment the source and decrement the destination
     * reference counts
     *
     * Refcount note: src and dest may have different sizes
     */
    NpyArray_INCREF(src);
    NpyArray_XDECREF(dest);
    
    NPY_BEGIN_THREADS;
    while(multi->index < multi->size) {
        myfunc(multi->iters[0]->dataptr,
               multi->iters[0]->strides[maxaxis],
               multi->iters[1]->dataptr,
               multi->iters[1]->strides[maxaxis],
               maxdim, elsize);
        if (swap) {
            _strided_byte_swap(multi->iters[0]->dataptr,
                               multi->iters[0]->strides[maxaxis],
                               maxdim, elsize);
        }
        NpyArray_MultiIter_NEXT(multi);
    }
    NPY_END_THREADS;
    
    NpyArray_INCREF(dest);
    NpyArray_XDECREF(src);
    
    _Npy_DECREF(multi);
    return 0;
}


static int
_copy_from0d(NpyArray *dest, NpyArray *src, int usecopy, int swap)
{
    char *aligned = NULL;
    char *sptr;
    npy_intp numcopies, nbytes;
    void (*myfunc)(char *, npy_intp, char *, npy_intp, npy_intp, int);
    int retval = -1;
    NPY_BEGIN_THREADS_DEF;
    
    numcopies = NpyArray_SIZE(dest);
    if (numcopies < 1) {
        return 0;
    }
    nbytes = NpyArray_ITEMSIZE(src);
    
    if (!NpyArray_ISALIGNED(src)) {
        aligned = malloc((size_t)nbytes);
        if (aligned == NULL) {
            NpyErr_NoMemory();
            return -1;
        }
        memcpy(aligned, src->data, (size_t) nbytes);
        usecopy = 1;
        sptr = aligned;
    }
    else {
        sptr = src->data;
    }
    if (NpyArray_SAFEALIGNEDCOPY(dest)) {
        myfunc = _strided_byte_copy;
    }
    else if (usecopy) {
        myfunc = _unaligned_strided_byte_copy;
    }
    else {
        myfunc = _unaligned_strided_byte_move;
    }
    
    if ((dest->nd < 2) || NpyArray_ISONESEGMENT(dest)) {
        char *dptr;
        npy_intp dstride;
        
        dptr = dest->data;
        if (dest->nd == 1) {
            dstride = dest->strides[0];
        }
        else {
            dstride = nbytes;
        }
        
        /* Refcount note: src and dest may have different sizes */
        NpyArray_INCREF(src);
        NpyArray_XDECREF(dest);
        NPY_BEGIN_THREADS;
        myfunc(dptr, dstride, sptr, 0, numcopies, (int) nbytes);
        if (swap) {
            _strided_byte_swap(dptr, dstride, numcopies, (int) nbytes);
        }
        NPY_END_THREADS;
        NpyArray_INCREF(dest);
        NpyArray_XDECREF(src);
    }
    else {
        NpyArrayIterObject *dit;
        int axis = -1;
        
        dit = NpyArray_IterAllButAxis(dest, &axis);
        if (dit == NULL) {
            goto finish;
        }
        /* Refcount note: src and dest may have different sizes */
        NpyArray_INCREF(src);
        NpyArray_XDECREF(dest);
        NPY_BEGIN_THREADS;
        while(dit->index < dit->size) {
            myfunc(dit->dataptr, NpyArray_STRIDE(dest, axis), sptr, 0,
                   NpyArray_DIM(dest, axis), nbytes);
            if (swap) {
                _strided_byte_swap(dit->dataptr, NpyArray_STRIDE(dest, axis),
                                   NpyArray_DIM(dest, axis), nbytes);
            }
            NpyArray_ITER_NEXT(dit);
        }
        NPY_END_THREADS;
        NpyArray_INCREF(dest);
        NpyArray_XDECREF(src);
        _Npy_DECREF(dit);
    }
    retval = 0;
    
finish:
    if (aligned != NULL) {
        free(aligned);
    }
    return retval;
}




/*
 * Special-case of NpyArray_CopyInto when dst is 1-d
 * and contiguous (and aligned).
 * NpyArray_CopyInto requires broadcastable arrays while
 * this one is a flattening operation...
 */
int 
_flat_copyinto(NpyArray *dst, NpyArray *src, NPY_ORDER order)
{
    NpyArrayIterObject *it;
    NpyArray *orig_src;
    void (*myfunc)(char *, npy_intp, char *, npy_intp, npy_intp, int);
    char *dptr;
    int axis;
    int elsize;
    npy_intp nbytes;
    NPY_BEGIN_THREADS_DEF;
    
    
    orig_src = src;
    if (NpyArray_NDIM(src) == 0) {
        /* Refcount note: src and dst have the same size */
        NpyArray_INCREF(src);
        NpyArray_XDECREF(dst);
        NPY_BEGIN_THREADS;
        memcpy(NpyArray_BYTES(dst), NpyArray_BYTES(src),
               NpyArray_ITEMSIZE(src));
        NPY_END_THREADS;
        return 0;
    }
    
    axis = NpyArray_NDIM(src)-1;
    
    if (order == NPY_FORTRANORDER) {
        if (NpyArray_NDIM(src) <= 2) {
            axis = 0;
        }
        /* fall back to a more general method */
        else {
            src = NpyArray_Transpose(orig_src, NULL);
        }
    }
    
    it = NpyArray_IterAllButAxis(src, &axis);
    if (it == NULL) {
        if (src != orig_src) {
            Py_DECREF(src);
        }
        return -1;
    }
    
    if (NpyArray_SAFEALIGNEDCOPY(src)) {
        myfunc = _strided_byte_copy;
    }
    else {
        myfunc = _unaligned_strided_byte_copy;
    }
    
    dptr = NpyArray_BYTES(dst);
    elsize = NpyArray_ITEMSIZE(dst);
    nbytes = elsize * NpyArray_DIM(src, axis);
    
    /* Refcount note: src and dst have the same size */
    NpyArray_INCREF(src);
    NpyArray_XDECREF(dst);
    NPY_BEGIN_THREADS;
    while(it->index < it->size) {
        myfunc(dptr, elsize, it->dataptr, NpyArray_STRIDE(src,axis),
               NpyArray_DIM(src,axis), elsize);
        dptr += nbytes;
        NpyArray_ITER_NEXT(it);
    }
    NPY_END_THREADS;
    
    if (src != orig_src) {
        Npy_DECREF(src);
    }
    _Npy_DECREF(it);
    return 0;
}



/*
 * This is the main array creation routine.
 *
 * Flags argument has multiple related meanings
 * depending on data and strides:
 *
 * If data is given, then flags is flags associated with data.
 * If strides is not given, then a contiguous strides array will be created
 * and the CONTIGUOUS bit will be set.  If the flags argument
 * has the FORTRAN bit set, then a FORTRAN-style strides array will be
 * created (and of course the FORTRAN flag bit will be set).
 *
 * If data is not given but created here, then flags will be DEFAULT
 * and a non-zero flags argument can be used to indicate a FORTRAN style
 * array is desired.
 */

size_t 
_array_fill_strides(npy_intp *strides, npy_intp *dims, int nd, size_t itemsize,
                    int inflag, int *objflags)
{
    int i;
    /* Only make Fortran strides if not contiguous as well */
    if ((inflag & NPY_FORTRAN) && !(inflag & NPY_CONTIGUOUS)) {
        for (i = 0; i < nd; i++) {
            strides[i] = itemsize;
            itemsize *= dims[i] ? dims[i] : 1;
        }
        *objflags |= NPY_FORTRAN;
        if (nd > 1) {
            *objflags &= ~NPY_CONTIGUOUS;
        }
        else {
            *objflags |= NPY_CONTIGUOUS;
        }
    }
    else {
        for (i = nd - 1; i >= 0; i--) {
            strides[i] = itemsize;
            itemsize *= dims[i] ? dims[i] : 1;
        }
        *objflags |= NPY_CONTIGUOUS;
        if (nd > 1) {
            *objflags &= ~NPY_FORTRAN;
        }
        else {
            *objflags |= NPY_FORTRAN;
        }
    }
    return itemsize;
}



/*
 * Change a sub-array field to the base descriptor
 *
 * and update the dimensions and strides
 * appropriately.  Dimensions and strides are added
 * to the end unless we have a FORTRAN array
 * and then they are added to the beginning
 *
 * Strides are only added if given (because data is given).
 */
static int 
_update_descr_and_dimensions(NpyArray_Descr **des, npy_intp *newdims,
                             npy_intp *newstrides, int oldnd, int isfortran)
{
    NpyArray_Descr *old;
    int newnd;
    int numnew;
    npy_intp *mydim;
    int i;
    
    old = *des;
    *des = old->subarray->base;
    
    /* TODO: NpyArray, NpyArray_Descr use Python tuples and other object types. Still needs to be refactored. */
    mydim = newdims + oldnd;
    numnew = old->subarray->shape_num_dims;
        
    newnd = oldnd + numnew;
    if (newnd > NPY_MAXDIMS) {
        goto finish;
    }
    if (isfortran) {
        memmove(newdims+numnew, newdims, oldnd*sizeof(npy_intp));
        mydim = newdims;
    }
    for (i = 0; i < numnew; i++) {
        mydim[i] = old->subarray->shape_dims[i];
    }
    
    if (newstrides) {
        npy_intp tempsize;
        npy_intp *mystrides;
        
        mystrides = newstrides + oldnd;
        if (isfortran) {
            memmove(newstrides+numnew, newstrides, oldnd*sizeof(npy_intp));
            mystrides = newstrides;
        }
        /* Make new strides -- alwasy C-contiguous */
        tempsize = (*des)->elsize;
        for (i = numnew - 1; i >= 0; i--) {
            mystrides[i] = tempsize;
            tempsize *= mydim[i] ? mydim[i] : 1;
        }
    }
    
finish:
    Npy_INCREF(*des);
    Npy_DECREF(old);
    return newnd;
}


/* If destination is not the right type, then src
 will be cast to destination -- this requires
 src and dest to have the same shape
 */

/* Requires arrays to have broadcastable shapes
 
 The arrays are assumed to have the same number of elements
 They can be different sizes and have different types however.
 */

static int 
_array_copy_into(NpyArray *dest, NpyArray *src, int usecopy)
{
    int swap;
    void (*myfunc)(char *, npy_intp, char *, npy_intp, npy_intp, int);
    int simple;
    int same;
    NPY_BEGIN_THREADS_DEF;
    
    
    if (!NpyArray_EquivArrTypes(dest, src)) {
        return NpyArray_CastTo(dest, src);
    }
    if (!NpyArray_ISWRITEABLE(dest)) {
        NpyErr_SetString(NpyExc_RuntimeError,
                        "cannot write to array");
        return -1;
    }
    same = NpyArray_SAMESHAPE(dest, src);
    simple = same && ((NpyArray_ISCARRAY_RO(src) && NpyArray_ISCARRAY(dest)) ||
                      (NpyArray_ISFARRAY_RO(src) && NpyArray_ISFARRAY(dest)));
    
    if (simple) {
        /* Refcount note: src and dest have the same size */
        NpyArray_INCREF(src);
        NpyArray_XDECREF(dest);
        NPY_BEGIN_THREADS;
        if (usecopy) {
            memcpy(dest->data, src->data, NpyArray_NBYTES(dest));
        }
        else {
            memmove(dest->data, src->data, NpyArray_NBYTES(dest));
        }
        NPY_END_THREADS;
        return 0;
    }
    
    swap = NpyArray_ISNOTSWAPPED(dest) != NpyArray_ISNOTSWAPPED(src);
    
    if (src->nd == 0) {
        return _copy_from0d(dest, src, usecopy, swap);
    }
    
    if (NpyArray_SAFEALIGNEDCOPY(dest) && NpyArray_SAFEALIGNEDCOPY(src)) {
        myfunc = _strided_byte_copy;
    }
    else if (usecopy) {
        myfunc = _unaligned_strided_byte_copy;
    }
    else {
        myfunc = _unaligned_strided_byte_move;
    }
    /*
     * Could combine these because _broadcasted_copy would work as well.
     * But, same-shape copying is so common we want to speed it up.
     */
    if (same) {
        return _copy_from_same_shape(dest, src, myfunc, swap);
    }
    else {
        return _broadcast_copy(dest, src, myfunc, swap);
    }
}

/*NUMPY_API
 * Move the memory of one array into another.
 */
int 
NpyArray_MoveInto(NpyArray *dest, NpyArray *src)
{
    return _array_copy_into(dest, src, 0);
}


/*NUMPY_API
 * steals a reference to descr -- accepts NULL
 */
NpyArray *
NpyArray_CheckFromArray(NpyArray *arr, PyArray_Descr *descr, int requires)
{
    NpyArray *obj;

    if (requires & NPY_NOTSWAPPED) {
        if (!descr && NpyArray_Check(arr) &&
            !NpyArray_ISNBO(NpyArray_DESCR(arr)->byteorder)) {
            descr = NpyArray_DescrNew(NpyArray_DESCR(arr));
        }
        else if (descr && !NpyArray_ISNBO(descr->byteorder)) {
            NpyArray_DESCR_REPLACE(descr);
        }
        if (descr) {
            descr->byteorder = NPY_NATIVE;
        }
    }
    
    obj = NpyArray_FromArray(arr, descr, requires);
    if (obj == NULL) {
        return NULL;
    }
    if ((requires & NPY_ELEMENTSTRIDES) &&
        !NpyArray_ElementStrides(obj)) {
        NpyArray *new;
        new = NpyArray_NewCopy(obj, PyArray_ANYORDER);
        Npy_DECREF(obj);
        obj = new;
    }
    return obj;
}


NpyArray *
NpyArray_CheckAxis(NpyArray *arr, int *axis, int flags)
{
    NpyArray *temp1, *temp2;
    int n = arr->nd;
    
    if (*axis == NPY_MAXDIMS || n == 0) {
        if (n != 1) {
            temp1 = NpyArray_Ravel(arr,0);
            if (temp1 == NULL) {
                *axis = 0;
                return NULL;
            }
            if (*axis == NPY_MAXDIMS) {
                *axis = NpyArray_NDIM(temp1)-1;
            }
        }
        else {
            temp1 = arr;
            Npy_INCREF(temp1);
            *axis = 0;
        }
        if (!flags && *axis == 0) {
            return temp1;
        }
    }
    else {
        temp1 = arr;
        Npy_INCREF(temp1);
    }
    if (flags) {
        temp2 = NpyArray_CheckFromArray(temp1, NULL, flags);
        Npy_DECREF(temp1);
        if (temp2 == NULL) {
            return NULL;
        }
    }
    else {
        temp2 = temp1;
    }
    n = NpyArray_NDIM(temp2);
    if (*axis < 0) {
        *axis += n;
    }
    if ((*axis < 0) || (*axis >= n)) {
        NpyErr_Format(NpyExc_ValueError,
                     "axis(=%d) out of bounds", *axis);
        Npy_DECREF(temp2);
        return NULL;
    }
    return temp2;
}






/*NUMPY_API
 * Generic new array creation routine.
 *
 * Array type algorithm: IF
 *  ensureArray             - use base array type
 *  subtype != NULL         - use subtype
 *  interfaceData != NULL   - use type of interface data
 *  default                 - use base array type
 * steals a reference to descr (even on failure)
 */
NpyArray *
NpyArray_NewFromDescr(NpyArray_Descr *descr, int nd,
                      npy_intp *dims, npy_intp *strides, void *data,
                      int flags, int ensureArray, void *subtype, 
                      void *interfaceData)
{
    NpyArray *self;
    int i;
    size_t sd;
    npy_intp largest;
    npy_intp size;
    PyTypeObject *subtypeHack = NULL;
    
    if (descr->subarray) {
        NpyArray *ret;
        npy_intp newdims[2*NPY_MAXDIMS];
        npy_intp *newstrides = NULL;
        int isfortran = 0;
        isfortran = (data && (flags & NPY_FORTRAN) && !(flags & NPY_CONTIGUOUS)) ||
        (!data && flags);
        memcpy(newdims, dims, nd*sizeof(npy_intp));
        if (strides) {
            newstrides = newdims + NPY_MAXDIMS;
            memcpy(newstrides, strides, nd*sizeof(npy_intp));
        }
        nd =_update_descr_and_dimensions(&descr, newdims,
                                         newstrides, nd, isfortran);
        ret = NpyArray_NewFromDescr(descr, nd, newdims,
                                    newstrides,
                                    data, flags, ensureArray, subtype, interfaceData);
        return ret;
    }
    if (nd < 0) {
        NpyErr_SetString(NpyExc_ValueError,
                        "number of dimensions must be >=0");
        Npy_DECREF(descr);
        return NULL;
    }
    if (nd > NPY_MAXDIMS) {
        NpyErr_Format(NpyExc_ValueError,
                     "maximum number of dimensions is %d", NPY_MAXDIMS);
        Npy_DECREF(descr);
        return NULL;
    }
    
    /* Check dimensions */
    size = 1;
    sd = (size_t) descr->elsize;
    if (sd == 0) {
        if (!NpyDataType_ISSTRING(descr)) {
            NpyErr_SetString(NpyExc_ValueError, "Empty data-type");
            Npy_DECREF(descr);
            return NULL;
        }
        NpyArray_DESCR_REPLACE(descr);
        if (descr->type_num == NPY_STRING) {
            descr->elsize = 1;
        }
        else {
            descr->elsize = sizeof(NpyArray_UCS4);
        }
        sd = descr->elsize;
    }
    
    largest = NPY_MAX_INTP / sd;
    for (i = 0; i < nd; i++) {
        npy_intp dim = dims[i];
        
        if (dim == 0) {
            /*
             * Compare to PyArray_OverflowMultiplyList that
             * returns 0 in this case.
             */
            continue;
        }
        if (dim < 0) {
            NpyErr_SetString(NpyExc_ValueError,
                            "negative dimensions are not allowed");
            Npy_DECREF(descr);
            return NULL;
        }
        if (dim > largest) {
            NpyErr_SetString(NpyExc_ValueError,
                            "array is too big.");
            Npy_DECREF(descr);
            return NULL;
        }
        size *= dim;
        largest /= dim;
    }
    
    
    /* TODO: This code should go away as soon as we split the array object from the Python wrapper. */
    if (NPY_TRUE == ensureArray) {
        subtypeHack = &PyArray_Type;
    } else if (NULL != subtype) {
        subtypeHack = (PyTypeObject *)subtype;
        assert(PyType_Check((PyObject *)subtypeHack));
    } else if (NULL != interfaceData) {
        assert(PyArray_Check(interfaceData));
        subtypeHack = Py_TYPE((PyObject *)interfaceData);
    } else {
        subtypeHack = &PyArray_Type;
    }
    self = (NpyArray *)subtypeHack->tp_alloc(subtypeHack, 0);
    //self = (NpyArray *) NpyArray_malloc(sizeof(NpyArray));
    
    
    if (self == NULL) {
        Npy_DECREF(descr);
        return NULL;
    }
    self->magic_number = NPY_VALID_MAGIC;
    self->nd = nd;
    self->dimensions = NULL;
    self->data = NULL;
    if (data == NULL) {
        self->flags = NPY_DEFAULT;
        if (flags) {
            self->flags |= NPY_FORTRAN;
            if (nd > 1) {
                self->flags &= ~NPY_CONTIGUOUS;
            }
            flags = NPY_FORTRAN;
        }
    }
    else {
        self->flags = (flags & ~NPY_UPDATEIFCOPY);
    }
    self->nob_interface = NULL;
    self->descr = descr;
    self->base_arr = NULL;
    self->base_obj = NULL;
    self->weakreflist = (PyObject *)NULL;   // TODO: Check this type
    
    if (nd > 0) {
        self->dimensions = NpyDimMem_NEW(2*nd);
        if (self->dimensions == NULL) {
            NpyErr_NoMemory();
            goto fail;
        }
        self->strides = self->dimensions + nd;
        memcpy(self->dimensions, dims, sizeof(npy_intp)*nd);
        if (strides == NULL) { /* fill it in */
            sd = _array_fill_strides(self->strides, dims, nd, sd,
                                     flags, &(self->flags));
        }
        else {
            /*
             * we allow strides even when we create
             * the memory, but be careful with this...
             */
            memcpy(self->strides, strides, sizeof(npy_intp)*nd);
            sd *= size;
        }
    }
    else {
        self->dimensions = self->strides = NULL;
    }
    
    if (data == NULL) {
        /*
         * Allocate something even for zero-space arrays
         * e.g. shape=(0,) -- otherwise buffer exposure
         * (a.data) doesn't work as it should.
         */
        
        if (sd == 0) {
            sd = descr->elsize;
        }
        if ((data = NpyDataMem_NEW(sd)) == NULL) {
            NpyErr_NoMemory();
            goto fail;
        }
        self->flags |= NPY_OWNDATA;
        
        /*
         * It is bad to have unitialized OBJECT pointers
         * which could also be sub-fields of a VOID array
         */
        if (NpyDataType_FLAGCHK(descr, NPY_NEEDS_INIT)) {
            memset(data, 0, sd);
        }
    }
    else {
        /*
         * If data is passed in, this object won't own it by default.
         * Caller must arrange for this to be reset if truly desired
         */
        self->flags &= ~NPY_OWNDATA;
    }
    self->data = data;
    
    /*
     * call the __array_finalize__
     * method if a subtype.
     * If obj is NULL, then call method with Py_None
     */
    if (NPY_FALSE == NpyInterface_ArrayNewWrapper(self, ensureArray, 
                                                  (NULL != strides), 
                                                  subtype, interfaceData, &self->nob_interface)) {
        Npy_INTERFACE(self) = NULL;
        Npy_DECREF(self);
        return NULL;
    }
    return self;
    
fail:
    Npy_DECREF(self);
    return NULL;
}




/*NUMPY_API
 * Generic new array creation routine.
 */
NpyArray *
NpyArray_New(NpyTypeObject *subtype, int nd, npy_intp *dims, int type_num,
             npy_intp *strides, void *data, int itemsize, int flags,
             NpyObject *obj)
{
    NpyArray_Descr *descr;
    NpyArray *new;
    
    descr = NpyArray_DescrFromType(type_num);
    if (descr == NULL) {
        return NULL;
    }
    if (descr->elsize == 0) {
        if (itemsize < 1) {
            NpyErr_SetString(NpyExc_ValueError,
                            "data type must provide an itemsize");
            Npy_DECREF(descr);
            return NULL;
        }
        NpyArray_DESCR_REPLACE(descr);
        descr->elsize = itemsize;
    }
    new = NpyArray_NewFromDescr(descr, nd, dims, strides,
                                data, flags, NPY_FALSE, subtype, obj);
    return new;
}




/*NUMPY_API
 * steals reference to newtype --- acc. NULL
 */
NpyArray *
NpyArray_FromArray(NpyArray *arr, NpyArray_Descr *newtype, int flags)
{
    NpyArray *ret = NULL;
    int itemsize;
    int copy = 0;
    int arrflags;
    NpyArray_Descr *oldtype;
    char *msg = "cannot copy back to a read-only array";
    int ensureArray = NPY_FALSE;
    
    oldtype = NpyArray_DESCR(arr);
    if (newtype == NULL) {
        newtype = oldtype; 
        Npy_INCREF(oldtype);
    }
    itemsize = newtype->elsize;
    if (itemsize == 0) {
        NpyArray_DESCR_REPLACE(newtype);
        if (newtype == NULL) {
            return NULL;
        }
        newtype->elsize = oldtype->elsize;
        itemsize = newtype->elsize;
    }
    
    /*
     * Can't cast unless ndim-0 array, FORCECAST is specified
     * or the cast is safe.
     */
    if (!(flags & NPY_FORCECAST) && !NpyArray_NDIM(arr) == 0 &&
        !NpyArray_CanCastTo(oldtype, newtype)) {
        Npy_DECREF(newtype);
        NpyErr_SetString(NpyExc_TypeError,
                        "array cannot be safely cast "  \
                        "to required type");
        return NULL;
    }
    
    /* Don't copy if sizes are compatible */
    if ((flags & NPY_ENSURECOPY) || NpyArray_EquivTypes(oldtype, newtype)) {
        arrflags = arr->flags;
        copy = (flags & NPY_ENSURECOPY) ||
        ((flags & NPY_CONTIGUOUS) && (!(arrflags & NPY_CONTIGUOUS)))
        || ((flags & NPY_ALIGNED) && (!(arrflags & NPY_ALIGNED)))
        || (arr->nd > 1 &&
            ((flags & NPY_FORTRAN) && (!(arrflags & NPY_FORTRAN))))
        || ((flags & NPY_WRITEABLE) && (!(arrflags & NPY_WRITEABLE)));
        
        if (copy) {
            if ((flags & NPY_UPDATEIFCOPY) &&
                (!NpyArray_ISWRITEABLE(arr))) {
                Npy_DECREF(newtype);
                NpyErr_SetString(NpyExc_ValueError, msg);
                return NULL;
            }
            if ((flags & NPY_ENSUREARRAY)) {
                ensureArray = NPY_TRUE;
            }
            ret = (PyArrayObject *)
            NpyArray_NewFromDescr(newtype,
                                  arr->nd,
                                  arr->dimensions,
                                  NULL, NULL,
                                  flags & NPY_FORTRAN,
                                  ensureArray, NULL,
                                  (PyObject *)arr);
            if (ret == NULL) {
                return NULL;
            }
            if (NpyArray_CopyInto(ret, arr) == -1) {
                Npy_DECREF(ret);
                return NULL;
            }
            if (flags & NPY_UPDATEIFCOPY)  {
                ret->flags |= NPY_UPDATEIFCOPY;
                ret->base_arr = arr;
                assert(NULL == ret->base_arr || NULL == ret->base_obj);
                NpyArray_FLAGS(ret->base_arr) &= ~NPY_WRITEABLE;
                Npy_INCREF(arr);
            }
        }
        /*
         * If no copy then just increase the reference
         * count and return the input
         */
        else {
            Npy_DECREF(newtype);
            if ((flags & NPY_ENSUREARRAY) &&
                !NpyArray_CheckExact(arr)) {
                Npy_INCREF(arr->descr);
                ret = NpyArray_NewFromDescr(arr->descr,
                                            arr->nd,
                                            arr->dimensions,
                                            arr->strides,
                                            arr->data,
                                            arr->flags, 
                                            NPY_TRUE, NULL, NULL);
                if (ret == NULL) {
                    return NULL;
                }
                ret->base_arr = arr;
                assert(NULL == ret->base_arr || NULL == ret->base_obj);
            }
            else {
                ret = arr;
            }
            Npy_INCREF(arr);
        }
    }
    
    /*
     * The desired output type is different than the input
     * array type and copy was not specified
     */
    else {
        if ((flags & NPY_UPDATEIFCOPY) &&
            (!NpyArray_ISWRITEABLE(arr))) {
            Npy_DECREF(newtype);
            NpyErr_SetString(NpyExc_ValueError, msg);
            return NULL;
        }
        if ((flags & NPY_ENSUREARRAY)) {
            ensureArray = NPY_TRUE;
        }
        ret = (NpyArray *)
        NpyArray_NewFromDescr(newtype,
                              arr->nd, arr->dimensions,
                              NULL, NULL,
                              flags & NPY_FORTRAN,
                              ensureArray, NULL, arr);
        if (ret == NULL) {
            return NULL;
        }
        if (NpyArray_CastTo(ret, arr) < 0) {
            Npy_DECREF(ret);
            return NULL;
        }
        if (flags & NPY_UPDATEIFCOPY)  {
            ret->flags |= NPY_UPDATEIFCOPY;
            ret->base_arr = arr;
            NpyArray_FLAGS(ret->base_arr) &= ~NPY_WRITEABLE;
            Npy_INCREF(arr);
        }
    }
    return ret;
}




/*NUMPY_API
 * Copy an Array into another array -- memory must not overlap
 * Does not require src and dest to have "broadcastable" shapes
 * (only the same number of elements).
 */
int 
NpyArray_CopyAnyInto(NpyArray *dest, NpyArray *src)
{
    int elsize, simple;
    NpyArrayIterObject *idest, *isrc;
    void (*myfunc)(char *, npy_intp, char *, npy_intp, npy_intp, int);
    NPY_BEGIN_THREADS_DEF;
    
    if (!NpyArray_EquivArrTypes(dest, src)) {
        return NpyArray_CastAnyTo(dest, src);
    }
    if (!NpyArray_ISWRITEABLE(dest)) {
        NpyErr_SetString(NpyExc_RuntimeError,
                         "cannot write to array");
        return -1;
    }
    if (NpyArray_SIZE(dest) != NpyArray_SIZE(src)) {
        NpyErr_SetString(NpyExc_ValueError,
                         "arrays must have the same number of elements"
                         " for copy");
        return -1;
    }
    
    simple = ((NpyArray_ISCARRAY_RO(src) && NpyArray_ISCARRAY(dest)) ||
              (NpyArray_ISFARRAY_RO(src) && NpyArray_ISFARRAY(dest)));
    if (simple) {
        /* Refcount note: src and dest have the same size */
        NpyArray_INCREF(src);
        NpyArray_XDECREF(dest);
        NPY_BEGIN_THREADS;
        memcpy(dest->data, src->data, NpyArray_NBYTES(dest));
        NPY_END_THREADS;
        return 0;
    }
    
    if (NpyArray_SAMESHAPE(dest, src)) {
        int swap;
        
        if (NpyArray_SAFEALIGNEDCOPY(dest) && NpyArray_SAFEALIGNEDCOPY(src)) {
            myfunc = _strided_byte_copy;
        }
        else {
            myfunc = _unaligned_strided_byte_copy;
        }
        swap = NpyArray_ISNOTSWAPPED(dest) != NpyArray_ISNOTSWAPPED(src);
        return _copy_from_same_shape(dest, src, myfunc, swap);
    }
    
    /* Otherwise we have to do an iterator-based copy */
    idest = NpyArray_IterNew(dest);
    if (idest == NULL) {
        return -1;
    }
    isrc = NpyArray_IterNew(src);
    if (isrc == NULL) {
        _Npy_DECREF(idest);
        return -1;
    }
    elsize = dest->descr->elsize;

    /* Refcount note: src and dest have the same size */
    NpyArray_INCREF(src);
    NpyArray_XDECREF(dest);
    NPY_BEGIN_THREADS;
    while(idest->index < idest->size) {
        memcpy(idest->dataptr, isrc->dataptr, elsize);
        NpyArray_ITER_NEXT(idest);
        NpyArray_ITER_NEXT(isrc);
    }
    NPY_END_THREADS;
    _Npy_DECREF(idest);
    _Npy_DECREF(isrc);
    return 0;
}


int 
NpyArray_CopyInto(NpyArray *dest, NpyArray *src)
{
    return _array_copy_into(dest, src, 1);
}


static NpyArray *array_fromfile_binary(FILE *fp, PyArray_Descr *dtype, npy_intp num, size_t *nread)
{
    NpyArray *r;
    npy_intp start, numbytes;
    
    if (num < 0) {
        int fail = 0;
        
        start = (npy_intp )ftell(fp);
        if (start < 0) {
            fail = 1;
        }
        if (fseek(fp, 0, SEEK_END) < 0) {
            fail = 1;
        }
        numbytes = (npy_intp) ftell(fp);
        if (numbytes < 0) {
            fail = 1;
        }
        numbytes -= start;
        if (fseek(fp, start, SEEK_SET) < 0) {
            fail = 1;
        }
        if (fail) {
            NpyErr_SetString(NpyExc_IOError,
                             "could not seek in file");
            Npy_DECREF(dtype);
            return NULL;
        }
        num = numbytes / dtype->elsize;
    }
    r = (NpyArray *)NpyArray_NewFromDescr(dtype,
                                          1, &num,
                                          NULL, NULL,
                                          0, NPY_TRUE, NULL, NULL);
    if (r == NULL) {
        return NULL;
    }
    NPY_BEGIN_ALLOW_THREADS;
    *nread = fread(r->data, dtype->elsize, num, fp);
    NPY_END_ALLOW_THREADS;
    return r;
}




NpyArray *
NpyArray_FromBinaryFile(FILE *fp, PyArray_Descr *dtype, npy_intp num)
{
    NpyArray *ret;
    size_t nread = 0;
    
    if (NpyDataType_REFCHK(dtype)) {
        NpyErr_SetString(NpyExc_ValueError,
                         "Cannot read into object array");
        Npy_DECREF(dtype);
        return NULL;
    }
    if (dtype->elsize == 0) {
        NpyErr_SetString(NpyExc_ValueError,
                         "The elements are 0-sized.");
        Npy_DECREF(dtype);
        return NULL;
    }

    ret = array_fromfile_binary(fp, dtype, num, &nread);
    if (ret == NULL) {
        Npy_DECREF(dtype);
        return NULL;
    }
    if (((npy_intp) nread) < num) {
        /* Realloc memory for smaller number of elements */
        const size_t nsize = NPY_MAX(nread,1)*ret->descr->elsize;
        char *tmp;
        
        if((tmp = NpyDataMem_RENEW(ret->data, nsize)) == NULL) {
            Npy_DECREF(ret);
            NpyErr_NoMemory();
            return NULL;
        }
        ret->data = tmp;
        NpyArray_DIM(ret,0) = nread;
    }
    return ret;
}



NpyArray *
NpyArray_FromBinaryString(char *data, npy_intp slen, PyArray_Descr *dtype, npy_intp num)
{
    int itemsize;
    NpyArray *ret;
    
    if (dtype == NULL) {
        dtype=NpyArray_DescrFromType(PyArray_DEFAULT);
    }
    if (NpyDataType_FLAGCHK(dtype, NPY_ITEM_IS_POINTER)) {
        NpyErr_SetString(NpyExc_ValueError,
                         "Cannot create an object array from"    \
                         " a string");
        Npy_DECREF(dtype);
        return NULL;
    }
    itemsize = dtype->elsize;
    if (itemsize == 0) {
        NpyErr_SetString(NpyExc_ValueError, "zero-valued itemsize");
        Npy_DECREF(dtype);
        return NULL;
    }
    
    if (num < 0 ) {
        if (slen % itemsize != 0) {
            NpyErr_SetString(NpyExc_ValueError,
                             "string size must be a "\
                             "multiple of element size");
            Npy_DECREF(dtype);
            return NULL;
        }
        num = slen/itemsize;
    }
    else {
        if (slen < num*itemsize) {
            NpyErr_SetString(NpyExc_ValueError,
                             "string is smaller than " \
                             "requested size");
            Npy_DECREF(dtype);
            return NULL;
        }
    }
    
    
    ret = NpyArray_NewFromDescr(dtype,
                                1, &num, NULL, NULL,
                                0, NPY_TRUE, NULL, NULL);
    if (ret == NULL) {
        return NULL;
    }
    memcpy(ret->data, data, num*dtype->elsize);
    return ret;
}

NpyArray_Descr *
NpyArray_DescrFromArray(NpyArray* array, NpyArray_Descr* mintype)
{
    NpyArray_Descr *result;
    if (mintype == NULL) {
        result = array->descr;
        Npy_INCREF(result);
    } else {
        result = NpyArray_SmallType(array->descr, mintype);
    }
    return result;
}
