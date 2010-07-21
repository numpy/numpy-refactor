/*
 *  npy_multiarraymodule.c -
 *
 */

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"






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
        Npy_XDECREF(typedescr);
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
        NpyArray_free(ptr);     /* TODO: Notice lower case 'f' - points to define that translate to free or something. */
    }
    Npy_DECREF(ap);
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
        Npy_DECREF(descr);
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
                Npy_DECREF(from);
                return 1;
            }
        }
    }
    Npy_DECREF(from);

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
    NpyArray *ret;
    double prior1, prior2;

    /*
     * Need to choose an output array that can hold a sum
     */
    if (Py_TYPE(ap2) != Py_TYPE(ap1)) {
        /* TODO: We can't get priority from the core object.
           We need to refactor this and probably move this
           funciton to the interface layer. */
        prior2 = PyArray_GetPriority((PyObject*)ap2, 0.0);
        prior1 = PyArray_GetPriority((PyObject*)ap1, 0.0);
    }
    else {
        prior1 = prior2 = 0.0;
    }

    ret = NpyArray_New(NULL, nd, dimensions,
                       typenum, NULL, NULL, 0, 0,
                       (NpyObject *)(prior2 > prior1 ? ap2 : ap1));
    return ret;
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
    Py_XDECREF(ret);
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
    if (PyErr_Occurred()) {
        goto fail;
    }
    return ret;

fail:
    Py_XDECREF(ret);
    return NULL;
}


/*NUMPY_API
 * Fast Copy and Transpose
 */
NpyArray *
NpyArray_CopyAndTranspose(NpyArray *arr)
{
    NpyArray *ret;
    int nd, eltsize, stride2;
    npy_intp dims[2], i, j;
    char *iptr, *optr;

    /* make sure it is well-behaved */
    arr = NpyArray_ContiguousFromArray(arr, NpyArray_TYPE(arr));
    if (arr == NULL) {
        return NULL;
    }
    nd = NpyArray_NDIM(arr);
    if (nd == 1) {
        /* we will give in to old behavior */
        Npy_DECREF(arr);
        return arr;
    }
    else if (nd != 2) {
        Npy_DECREF(arr);
        NpyErr_SetString(NpyExc_ValueError, "only 2-d arrays are allowed");
        return NULL;
    }

    /* Now construct output array */
    dims[0] = NpyArray_DIM(arr, 1);
    dims[1] = NpyArray_DIM(arr, 0);
    eltsize = NpyArray_ITEMSIZE(arr);
    Npy_INCREF(arr);
    ret = NpyArray_NewFromDescr(NpyArray_DESCR(arr), 2, dims,
                                NULL, NULL, 0, NPY_FALSE, NULL, arr);
    if (ret == NULL) {
        Npy_DECREF(arr);
        return NULL;
    }

    /* do 2-d loop */
    NPY_BEGIN_ALLOW_THREADS;
    optr = NpyArray_DATA(ret);
    stride2 = eltsize * dims[0];
    for (i = 0; i < dims[0]; i++) {
        iptr = PyArray_BYTES(arr) + i * eltsize;
        for (j = 0; j < dims[1]; j++) {
            /* optr[i,j] = iptr[j,i] */
            memcpy(optr, iptr, eltsize);
            optr += eltsize;
            iptr += stride2;
        }
    }
    NPY_END_ALLOW_THREADS;
    Npy_DECREF(arr);

    return ret;
}
