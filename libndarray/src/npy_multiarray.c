/*
 *  npy_multiarray.c -
 *
 */

#include <stdlib.h>
#include <memory.h>
#include "npy_config.h"
#include "npy_api.h"
#include "npy_arrayobject.h"
#include "npy_dict.h"
#include "npy_internal.h"



npy_tp_error_set NpyErr_SetString;
npy_tp_error_occurred NpyErr_Occurred;
npy_tp_error_clear NpyErr_Clear;
npy_tp_cmp_priority Npy_CmpPriority;

/* Defined in npy_arraytypes.c.src */
extern void _init_builtin_descr_wrappers(struct NpyArray_FunctionDefs *);


npy_interface_incref _NpyInterface_Incref = NULL;
npy_interface_decref _NpyInterface_Decref = NULL;
struct NpyInterface_WrapperFuncs _NpyArrayWrapperFuncs = {
    NULL, NULL, NULL, NULL, NULL, NULL
};


/* Initializes the library at startup.
   This functions must be called exactly once by the interface layer.*/
void initlibnumpy(struct NpyArray_FunctionDefs *functionDefs,
                  struct NpyInterface_WrapperFuncs *wrapperFuncs,
                  npy_tp_error_set error_set,
                  npy_tp_error_occurred error_occurred,
                  npy_tp_error_clear error_clear,
                  npy_tp_cmp_priority cmp_priority,
                  npy_interface_incref incref, npy_interface_decref decref)
{
    if (NULL != wrapperFuncs) {
        memmove(&_NpyArrayWrapperFuncs, wrapperFuncs, sizeof(struct NpyInterface_WrapperFuncs));
    }
    NpyErr_SetString = error_set;
    NpyErr_Occurred = error_occurred;
    NpyErr_Clear = error_clear;
    Npy_CmpPriority = cmp_priority;

    _NpyInterface_Incref = incref;
    _NpyInterface_Decref = decref;

    /* Must be last because it uses some of the above functions. */
    _init_builtin_descr_wrappers(functionDefs);
}


/*NUMPY_API
 * Multiply a List of ints
 */
int
NpyArray_MultiplyIntList(int *l1, int n)
{
    int s = 1;

    while (n--) {
        s *= (*l1++);
    }
    return s;
}


/*NUMPY_API
 * Multiply a List
 */
npy_intp
NpyArray_MultiplyList(npy_intp *l1, int n)
{
    npy_intp s = 1;

    while (n--) {
        s *= (*l1++);
    }
    return s;
}


/*NUMPY_API
 * Multiply a List of Non-negative numbers with over-flow detection.
 */
npy_intp
NpyArray_OverflowMultiplyList(npy_intp *l1, int n)
{
    npy_intp prod = 1;
    npy_intp imax = NPY_MAX_INTP;
    int i;

    for (i = 0; i < n; i++) {
        npy_intp dim = l1[i];

        if (dim == 0) {
            return 0;
        }
        if (dim > imax) {
            return -1;
        }
        imax /= dim;
        prod *= dim;
    }
    return prod;
}

/*NUMPY_API
 * Produce a pointer into array
 */
void *
NpyArray_GetPtr(NpyArray *obj, npy_intp *ind)
{
    int n = obj->nd;
    npy_intp *strides = obj->strides;
    char *dptr = obj->data;

    while (n--) {
        dptr += (*strides++) * (*ind++);
    }
    return (void *)dptr;
}


/*NUMPY_API
 * Compare Lists
 */
int
NpyArray_CompareLists(npy_intp *l1, npy_intp *l2, int n)
{
    int i;

    for (i = 0; i < n; i++) {
        if (l1[i] != l2[i]) {
            return 0;
        }
    }
    return 1;
}

/*
 * simulates a C-style 1-3 dimensional array which can be accesed using
 * ptr[i]  or ptr[i][j] or ptr[i][j][k] -- requires pointer allocation
 * for 2-d and 3-d.
 *
 * For 2-d and up, ptr is NOT equivalent to a statically defined
 * 2-d or 3-d array.  In particular, it cannot be passed into a
 * function that requires a true pointer to a fixed-size array.
 */

/*NUMPY_API
 * Simulate a C-array
 * steals a reference to typedescr -- can be NULL
 */
int
NpyArray_AsCArray(NpyArray **apIn, void *ptr, npy_intp *dims, int nd,
                  NpyArray_Descr* typedescr)
{
    NpyArray *ap;
    npy_intp n, m, i, j;
    char **ptr2;
    char ***ptr3;

    if ((nd < 1) || (nd > 3)) {
        NpyErr_SetString(NpyExc_ValueError,
                         "C arrays of only 1-3 dimensions available");
        _Npy_XDECREF(typedescr);
        return -1;
    }
    if ((ap = NpyArray_FromArray(*apIn, typedescr, NPY_CARRAY)) == NULL) {
        return -1;
    }
    switch(nd) {
        case 1:
            *((char **)ptr) = ap->data;
            break;
        case 2:
            n = ap->dimensions[0];
            ptr2 = (char **)NpyArray_malloc(n * sizeof(char *));
            if (!ptr2) {
                goto fail;
            }
            for (i = 0; i < n; i++) {
                ptr2[i] = ap->data + i*ap->strides[0];
            }
            *((char ***)ptr) = ptr2;
            break;
        case 3:
            n = ap->dimensions[0];
            m = ap->dimensions[1];
            ptr3 = (char ***)NpyArray_malloc(n*(m+1) * sizeof(char *));
            if (!ptr3) {
                goto fail;
            }
            for (i = 0; i < n; i++) {
                ptr3[i] = ptr3[n + (m-1)*i];
                for (j = 0; j < m; j++) {
                    ptr3[i][j] = ap->data + i*ap->strides[0] + j*ap->strides[1];
                }
            }
            *((char ****)ptr) = ptr3;
    }
    memcpy(dims, ap->dimensions, nd*sizeof(npy_intp));
    *apIn = ap;
    return 0;

fail:
    NpyErr_SetString(NpyExc_MemoryError, "no memory");
    return -1;
}





/*NUMPY_API
 * Free pointers created if As2D is called
 */
int
NpyArray_Free(NpyArray *ap, void *ptr)
{
    if ((ap->nd < 1) || (ap->nd > 3)) {
        return -1;
    }
    if (ap->nd >= 2) {
        /* TODO: Notice lower case 'f' - points to define that translate to
                 free or something. */
        NpyArray_free(ptr);
    }
    _Npy_DECREF(ap);
    return 0;
}




static int
_signbit_set(NpyArray *arr)
{
    static char bitmask = (char) 0x80;
    char *ptr;  /* points to the byte to test */
    char byteorder;
    int elsize;

    elsize = arr->descr->elsize;
    byteorder = arr->descr->byteorder;
    ptr = arr->data;
    if (elsize > 1 &&
        (byteorder == NPY_LITTLE ||
         (byteorder == NPY_NATIVE &&
          NpyArray_ISNBO(NPY_LITTLE)))) {
             ptr += elsize - 1;
         }
    return ((*ptr & bitmask) != 0);
}


/*NUMPY_API
 * ScalarKind
 */
NPY_SCALARKIND
NpyArray_ScalarKind(int typenum, NpyArray **arr)
{
    if (NpyTypeNum_ISSIGNED(typenum)) {
        if (arr && _signbit_set(*arr)) {
            return NPY_INTNEG_SCALAR;
        }
        else {
            return NPY_INTPOS_SCALAR;
        }
    }
    if (NpyTypeNum_ISFLOAT(typenum)) {
        return NPY_FLOAT_SCALAR;
    }
    if (NpyTypeNum_ISUNSIGNED(typenum)) {
        return NPY_INTPOS_SCALAR;
    }
    if (NpyTypeNum_ISCOMPLEX(typenum)) {
        return NPY_COMPLEX_SCALAR;
    }
    if (NpyTypeNum_ISBOOL(typenum)) {
        return NPY_BOOL_SCALAR;
    }

    if (NpyTypeNum_ISUSERDEF(typenum)) {
        NPY_SCALARKIND retval;
        NpyArray_Descr* descr = NpyArray_DescrFromType(typenum);

        if (descr->f->scalarkind) {
            retval = descr->f->scalarkind((arr ? *arr : NULL));
        }
        else {
            retval = NPY_NOSCALAR;
        }
        _Npy_DECREF(descr);
        return retval;
    }
    return NPY_OBJECT_SCALAR;
}

/*NUMPY_API*/
int
NpyArray_CanCoerceScalar(int thistype, int neededtype,
                         NPY_SCALARKIND scalar)
{
    NpyArray_Descr* from;
    int *castlist;

    if (scalar == NPY_NOSCALAR) {
        return NpyArray_CanCastSafely(thistype, neededtype);
    }
    from = NpyArray_DescrFromType(thistype);
    if (from->f->cancastscalarkindto
        && (castlist = from->f->cancastscalarkindto[scalar])) {
        while (*castlist != NPY_NOTYPE) {
            if (*castlist++ == neededtype) {
                _Npy_DECREF(from);
                return 1;
            }
        }
    }
    _Npy_DECREF(from);

    switch(scalar) {
        case NPY_BOOL_SCALAR:
        case NPY_OBJECT_SCALAR:
            return NpyArray_CanCastSafely(thistype, neededtype);
        default:
            if (NpyTypeNum_ISUSERDEF(neededtype)) {
                return NPY_FALSE;
            }
            switch(scalar) {
                case NPY_INTPOS_SCALAR:
                    return (neededtype >= NPY_BYTE);
                case NPY_INTNEG_SCALAR:
                    return (neededtype >= NPY_BYTE)
                    && !(NpyTypeNum_ISUNSIGNED(neededtype));
                case NPY_FLOAT_SCALAR:
                    return (neededtype >= NPY_FLOAT);
                case NPY_COMPLEX_SCALAR:
                    return (neededtype >= NPY_CFLOAT);
                default:
                    /* should never get here... */
                    return 1;
            }
    }
}




/*
 * Make a new empty array, of the passed size, of a type that takes the
 * priority of ap1 and ap2 into account.
 */
static NpyArray *
new_array_for_sum(NpyArray *ap1, NpyArray *ap2,
                  int nd, npy_intp dimensions[], int typenum)
{
    int tmp;

    /*
     * Need to choose an output array that can hold a sum
     */
    tmp = Npy_CmpPriority(Npy_INTERFACE(ap1), Npy_INTERFACE(ap2));

    return NpyArray_New(NULL, nd, dimensions,
                        typenum, NULL, NULL, 0, 0,
                        Npy_INTERFACE(tmp ? ap2 : ap1));
}


/*NUMPY_API
 * Numeric.innerproduct(a,v)
 */
NpyArray *
NpyArray_InnerProduct(NpyArray *ap1, NpyArray *ap2, int typenum)
{
    NpyArray *ret = NULL;
    NpyArrayIterObject *it1, *it2;
    npy_intp i, j, l;
    int nd, axis;
    npy_intp is1, is2, os;
    char *op;
    npy_intp dimensions[NPY_MAXDIMS];
    NpyArray_DotFunc *dot;
    NPY_BEGIN_THREADS_DEF;

    l = ap1->dimensions[ap1->nd - 1];
    if (ap2->dimensions[ap2->nd - 1] != l) {
        NpyErr_SetString(NpyExc_ValueError, "matrices are not aligned");
        return NULL;
    }

    nd = ap1->nd + ap2->nd - 2;
    j = 0;
    for (i = 0; i < ap1->nd - 1; i++) {
        dimensions[j++] = ap1->dimensions[i];
    }
    for (i = 0; i < ap2->nd - 1; i++) {
        dimensions[j++] = ap2->dimensions[i];
    }

    /*
     * Need to choose an output array that can hold a sum
     * -- use priority to determine which subtype.
     */
    ret = new_array_for_sum(ap1, ap2, nd, dimensions, typenum);
    if (ret == NULL) {
        return NULL;
    }
    dot = ret->descr->f->dotfunc;
    if (dot == NULL) {
        NpyErr_SetString(NpyExc_ValueError, "dot not available for this type");
        goto fail;
    }
    is1 = ap1->strides[ap1->nd - 1];
    is2 = ap2->strides[ap2->nd - 1];
    op = ret->data;
    os = ret->descr->elsize;
    axis = ap1->nd - 1;
    it1 = NpyArray_IterAllButAxis(ap1, &axis);
    axis = ap2->nd - 1;
    it2 = NpyArray_IterAllButAxis(ap2, &axis);
    NPY_BEGIN_THREADS_DESCR(ap2->descr);
    while (1) {
        while (it2->index < it2->size) {
            dot(it1->dataptr, is1, it2->dataptr, is2, op, l, ret);
            op += os;
            NpyArray_ITER_NEXT(it2);
        }
        NpyArray_ITER_NEXT(it1);
        if (it1->index >= it1->size) {
            break;
        }
        NpyArray_ITER_RESET(it2);
    }
    NPY_END_THREADS_DESCR(ap2->descr);
    _Npy_DECREF(it1);
    _Npy_DECREF(it2);
    if (NpyErr_Occurred()) {
        goto fail;
    }
    return ret;

 fail:
    _Npy_DECREF(ret);
    return NULL;
}


/*NUMPY_API
 *Numeric.matrixproduct(a,v)
 * just like inner product but does the swapaxes stuff on the fly
 */
NpyArray *
NpyArray_MatrixProduct(NpyArray *ap1, NpyArray *ap2, int typenum)
{
    NpyArray *ret = NULL;
    NpyArrayIterObject *it1, *it2;
    npy_intp i, j, l, is1, is2, os, dimensions[NPY_MAXDIMS];
    int nd, axis, matchDim;
    char *op;
    NpyArray_DotFunc *dot;
    NPY_BEGIN_THREADS_DEF;

    if (ap2->nd > 1) {
        matchDim = ap2->nd - 2;
    }
    else {
        matchDim = 0;
    }
    l = ap1->dimensions[ap1->nd - 1];
    if (ap2->dimensions[matchDim] != l) {
        NpyErr_SetString(NpyExc_ValueError, "objects are not aligned");
        return NULL;
    }
    nd = ap1->nd + ap2->nd - 2;
    if (nd > NPY_MAXDIMS) {
        NpyErr_SetString(NpyExc_ValueError,
                         "dot: too many dimensions in result");
        return NULL;
    }
    j = 0;
    for (i = 0; i < ap1->nd - 1; i++) {
        dimensions[j++] = ap1->dimensions[i];
    }
    for (i = 0; i < ap2->nd - 2; i++) {
        dimensions[j++] = ap2->dimensions[i];
    }
    if(ap2->nd > 1) {
        dimensions[j++] = ap2->dimensions[ap2->nd - 1];
    }
    is1 = ap1->strides[ap1->nd - 1];
    is2 = ap2->strides[matchDim];
    /* Choose which subtype to return */
    ret = new_array_for_sum(ap1, ap2, nd, dimensions, typenum);
    if (ret == NULL) {
        return NULL;
    }
    /* Ensure that multiarray.dot(<Nx0>,<0xM>) -> zeros((N,M)) */
    if (NpyArray_SIZE(ap1) == 0 && NpyArray_SIZE(ap2) == 0) {
        memset(NpyArray_DATA(ret), 0, NpyArray_NBYTES(ret));
    }
    else {
        /* Ensure that multiarray.dot([],[]) -> 0 */
        memset(NpyArray_DATA(ret), 0, NpyArray_ITEMSIZE(ret));
    }

    dot = ret->descr->f->dotfunc;
    if (dot == NULL) {
        NpyErr_SetString(NpyExc_ValueError, "dot not available for this type");
        goto fail;
    }

    op = NpyArray_BYTES(ret);
    os = NpyArray_ITEMSIZE(ret);
    axis = ap1->nd - 1;
    it1 = NpyArray_IterAllButAxis(ap1, &axis);
    it2 = NpyArray_IterAllButAxis(ap2, &matchDim);
    NPY_BEGIN_THREADS_DESCR(ap2->descr);
    while (1) {
        while (it2->index < it2->size) {
            dot(it1->dataptr, is1, it2->dataptr, is2, op, l, ret);
            op += os;
            NpyArray_ITER_NEXT(it2);
        }
        NpyArray_ITER_NEXT(it1);
        if (it1->index >= it1->size) {
            break;
        }
        NpyArray_ITER_RESET(it2);
    }
    NPY_END_THREADS_DESCR(ap2->descr);
    _Npy_DECREF(it1);
    _Npy_DECREF(it2);
    if (NpyErr_Occurred()) {
        goto fail;
    }
    return ret;

fail:
    _Npy_XDECREF(ret);
    return NULL;
}


/*NUMPY_API
 * Fast Copy and Transpose
 */
NpyArray *
NpyArray_CopyAndTranspose(NpyArray *arr)
{
    NpyArray *ret, *tmp;
    int nd, eltsize, stride2;
    npy_intp dims[2], i, j;
    char *iptr, *optr;

    /* make sure it is well-behaved */
    tmp = NpyArray_ContiguousFromArray(arr, NpyArray_TYPE(arr));
    if (tmp == NULL) {
        return NULL;
    }
    arr = tmp;

    nd = NpyArray_NDIM(arr);
    if (nd == 1) {
        /* we will give in to old behavior */
        _Npy_DECREF(tmp);
        return arr;
    }
    else if (nd != 2) {
        _Npy_DECREF(tmp);
        NpyErr_SetString(NpyExc_ValueError, "only 2-d arrays are allowed");
        return NULL;
    }

    /* Now construct output array */
    dims[0] = NpyArray_DIM(arr, 1);
    dims[1] = NpyArray_DIM(arr, 0);
    eltsize = NpyArray_ITEMSIZE(arr);
    _Npy_INCREF(arr->descr);
    ret = NpyArray_Alloc(arr->descr, 2, dims, NPY_FALSE, NULL);
    if (ret == NULL) {
        _Npy_DECREF(tmp);
        return NULL;
    }

    /* do 2-d loop */
    NPY_BEGIN_ALLOW_THREADS;
    optr = NpyArray_DATA(ret);
    stride2 = eltsize * dims[0];
    for (i = 0; i < dims[0]; i++) {
        iptr = NpyArray_BYTES(arr) + i * eltsize;
        for (j = 0; j < dims[1]; j++) {
            /* optr[i,j] = iptr[j,i] */
            memcpy(optr, iptr, eltsize);
            optr += eltsize;
            iptr += stride2;
        }
    }
    NPY_END_ALLOW_THREADS;

    _Npy_DECREF(tmp);
    return ret;
}


/*
 * Implementation which is common between
 * NpyArray_Correlate and NpyArray_Correlate2
 *
 * inverted is set to 1 if computed correlate(ap2, ap1), 0 otherwise
 */
static NpyArray *
_npyarray_correlate(NpyArray *ap1, NpyArray *ap2,
                    int typenum, int mode, int *inverted)
{
    NpyArray *ret;
    npy_intp length, i, n1, n2, n, n_left, n_right, is1, is2, os;
    char *ip1, *ip2, *op;
    NpyArray_DotFunc *dot;
    NPY_BEGIN_THREADS_DEF;

    n1 = NpyArray_DIM(ap1, 0);
    n2 = NpyArray_DIM(ap2, 0);
    if (n1 < n2) {
        ret = ap1;
        ap1 = ap2;
        ap2 = ret;
        ret = NULL;
        i = n1;
        n1 = n2;
        n2 = i;
        *inverted = 1;
    } else {
        *inverted = 0;
    }

    length = n1;
    n = n2;
    switch(mode) {
    case 0:
        length = length - n + 1;
        n_left = n_right = 0;
        break;
    case 1:
        n_left = (npy_intp)(n / 2);
        n_right = n - n_left - 1;
        break;
    case 2:
        n_right = n - 1;
        n_left = n - 1;
        length = length + n - 1;
        break;
    default:
        NpyErr_SetString(NpyExc_ValueError, "mode must be 0, 1, or 2");
        return NULL;
    }

    /*
     * Need to choose an output array that can hold a sum
     * -- use priority to determine which subtype.
     */
    ret = new_array_for_sum(ap1, ap2, 1, &length, typenum);
    if (ret == NULL) {
        return NULL;
    }
    dot = NpyArray_DESCR(ret)->f->dotfunc;
    if (dot == NULL) {
        NpyErr_SetString(NpyExc_ValueError,
                         "function not available for this data type");
        goto clean_ret;
    }

    NPY_BEGIN_THREADS_DESCR(NpyArray_DESCR(ret));
    is1 = NpyArray_STRIDE(ap1, 0);
    is2 = NpyArray_STRIDE(ap2, 0);
    op = NpyArray_BYTES(ret);
    os = NpyArray_ITEMSIZE(ret);
    ip1 = NpyArray_BYTES(ap1);
    ip2 = NpyArray_BYTES(ap2) + n_left * is2;
    n -= n_left;
    for (i = 0; i < n_left; i++) {
        dot(ip1, is1, ip2, is2, op, n, ret);
        n++;
        ip2 -= is2;
        op += os;
    }
    for (i = 0; i < (n1 - n2 + 1); i++) {
        dot(ip1, is1, ip2, is2, op, n, ret);
        ip1 += is1;
        op += os;
    }
    for (i = 0; i < n_right; i++) {
        n--;
        dot(ip1, is1, ip2, is2, op, n, ret);
        ip1 += is1;
        op += os;
    }

    NPY_END_THREADS_DESCR(NpyArray_DESCR(ret));
    if (NpyErr_Occurred()) {
        goto clean_ret;
    }

    return ret;

clean_ret:
    _Npy_DECREF(ret);
    return NULL;
}


/*
 * Revert a one dimensional array in-place
 *
 * Return 0 on success, other value on failure
 */
static int
_npyarray_revert(NpyArray *ret)
{
    npy_intp length, i, os;
    NpyArray_CopySwapFunc *copyswap;
    char *tmp = NULL, *sw1, *sw2,  *op;

    length = NpyArray_DIM(ret, 0);
    copyswap = NpyArray_DESCR(ret)->f->copyswap;

    tmp = NpyArray_malloc(NpyArray_ITEMSIZE(ret));
    if (tmp == NULL) {
        return -1;
    }

    os = NpyArray_ITEMSIZE(ret);
    op = NpyArray_BYTES(ret);
    sw1 = op;
    sw2 = op + (length - 1) * os;
    if (NpyArray_ISFLEXIBLE(ret) || NpyArray_ISOBJECT(ret)) {
        for(i = 0; i < length / 2; i++) {
            memmove(tmp, sw1, os);
            copyswap(tmp, NULL, 0, NULL);
            memmove(sw1, sw2, os);
            copyswap(sw1, NULL, 0, NULL);
            memmove(sw2, tmp, os);
            copyswap(sw2, NULL, 0, NULL);
            sw1 += os;
            sw2 -= os;
        }
    } else {
        for(i = 0; i < length / 2; i++) {
            memcpy(tmp, sw1, os);
            memcpy(sw1, sw2, os);
            memcpy(sw2, tmp, os);
            sw1 += os;
            sw2 -= os;
        }
    }

    NpyArray_free(tmp);
    return 0;
}


/* TODO: Remove this declaration once PyArray_Conjugate is refactored */
extern NpyArray *NpyArray_Conjugate(NpyArray *self, NpyArray *out);

/*NUMPY_API
 * correlate(a1, a2 ,typenum, mode)
 *
 * This function computes the usual correlation (correlate(a1, a2) !=
 * correlate(a2, a1), and conjugate the second argument for complex inputs
 */
NpyArray *
NpyArray_Correlate2(NpyArray *ap1, NpyArray *ap2, int typenum, int mode)
{
    NpyArray *ret = NULL;
    NpyArray *cap2 = NULL;
    int inverted, status;

    if (NpyArray_ISCOMPLEX(ap2)) {
        /* FIXME: PyArray_Conjugate need to be replaced by NpyArray_Conjugate,
                  once NpyArray_Conjugate is created, which can be done once
                  the ufunc stuff is in the core.
         */
        ap2 = NpyArray_Conjugate(ap2, NULL);
        if (NULL == ap2) {
            return NULL;
        }
        cap2 = ap2;
    }

    ret = _npyarray_correlate(ap1, ap2, typenum, mode, &inverted);
    if (ret == NULL) {
        goto done;
    }

    /* If we inverted input orders, we need to reverse the output array (i.e.
       ret = ret[::-1]) */
    if (inverted) {
        status = _npyarray_revert(ret);
        if(status) {
            _Npy_DECREF(ret);
            ret = NULL;
            goto done;
        }
    }
 done:
    _Npy_XDECREF(cap2);
    return ret;
}


/*NUMPY_API
 * Numeric.correlate(a1, a2 ,typenum, mode)
 */
NpyArray *
NpyArray_Correlate(NpyArray *ap1, NpyArray *ap2, int typenum, int mode)
{
    int unused;

    return _npyarray_correlate(ap1, ap2, typenum, mode, &unused);
}


/*
 * compare the field dictionary for two types
 * return 1 if the same or 0 if not
 */
static int
_equivalent_fields(NpyDict *field1, NpyDict *field2)
{
    NpyDict_Iter pos;
    NpyArray_DescrField *value1, *value2;
    const char *key;
    int same=1;

    if (field1 == field2) {
        return 1;
    }
    if (field1 == NULL || field2 == NULL) {
        return 0;
    }
    if (NpyDict_Size(field1) != NpyDict_Size(field2)) {
        same = 0;
    }

    NpyDict_IterInit(&pos);
    while (same && NpyDict_IterNext(field1, &pos, (void **)&key,
                                    (void **)&value1)) {
        value2 = NpyDict_Get(field2, key);
        if (NULL == value2 || value1->offset != value2->offset ||
            ((NULL == value1->title && NULL != value2->title) ||
             (NULL != value1->title && NULL == value2->title) ||
             (NULL != value1->title && NULL != value2->title &&
              strcmp(value1->title, value2->title)))) {
            same = 0;
        } else if (!NpyArray_EquivTypes(value1->descr, value2->descr)) {
            same = 0;
        }
    }
    return same;
}


/* compare the metadata for two date-times
 * return 1 if they are the same, or 0 if not
 */
static int
_equivalent_units(NpyArray_DateTimeInfo *info1, NpyArray_DateTimeInfo *info2)
{
    /* Same meta object */
    return ((info1 == info2)
            || ((info1->base == info2->base)
            && (info1->num == info2->num)
            && (info1->den == info2->den)
            && (info1->events == info2->events)));
}


/*NUMPY_API
 *
 * This function returns true if the two typecodes are
 * equivalent (same basic kind and same itemsize).
 */
unsigned char
NpyArray_EquivTypes(NpyArray_Descr *typ1, NpyArray_Descr *typ2)
{
    int typenum1 = typ1->type_num;
    int typenum2 = typ2->type_num;
    int size1 = typ1->elsize;
    int size2 = typ2->elsize;

    if (size1 != size2) {
        return NPY_FALSE;
    }
    if (NpyArray_ISNBO(typ1->byteorder) != NpyArray_ISNBO(typ2->byteorder)) {
        return NPY_FALSE;
    }
    if (typenum1 == NPY_VOID
        || typenum2 == NPY_VOID) {
        return ((typenum1 == typenum2)
                && _equivalent_fields(typ1->fields, typ2->fields));
    }
    if (typenum1 == NPY_DATETIME
        || typenum1 == NPY_DATETIME
        || typenum2 == NPY_TIMEDELTA
        || typenum2 == NPY_TIMEDELTA) {
        return ((typenum1 == typenum2)
                && _equivalent_units(typ1->dtinfo, typ2->dtinfo));
    }
    return typ1->kind == typ2->kind;
}


/*NUMPY_API*/
unsigned char
NpyArray_EquivTypenums(int typenum1, int typenum2)
{
    NpyArray_Descr *d1, *d2;
    npy_bool ret;

    d1 = NpyArray_DescrFromType(typenum1);
    d2 = NpyArray_DescrFromType(typenum2);
    ret = NpyArray_EquivTypes(d1, d2);
    _Npy_DECREF(d1);
    _Npy_DECREF(d2);
    return ret;
}


/*NUMPY_API
*/
int
NpyArray_GetEndianness(void)
{
    const union {
        npy_uint32 i;
        char c[4];
    } bint = {0x01020304};

    if (bint.c[0] == 1) {
        return NPY_CPU_BIG;
    }
    else if (bint.c[0] == 4) {
        return NPY_CPU_LITTLE;
    }
    else {
        return NPY_CPU_UNKNOWN_ENDIAN;
    }
}
