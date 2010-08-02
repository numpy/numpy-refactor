

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include "npy_config.h"
#include "numpy/npy_object.h"
#include "numpy/numpy_api.h"

static int
_check_ones(NpyArray *self, int newnd, npy_intp* newdims, npy_intp *strides);

static int
_fix_unknown_dimension(NpyArray_Dims *newshape, npy_intp s_original);

static int
_attempt_nocopy_reshape(NpyArray *self, int newnd, npy_intp* newdims,
                        npy_intp *newstrides, int fortran);


/*
 * Resize (reallocate data).  Only works if nothing else is referencing this
 * array and it is contiguous.  If refcheck is 0, then the reference count is
 * not checked and assumed to be 1.  You still must own this data and have no
 * weak-references and no base object.
 */
int
NpyArray_Resize(NpyArray *self, NpyArray_Dims *newshape, int refcheck,
               NPY_ORDER fortran)
{
    npy_intp oldsize, newsize;
    int new_nd=newshape->len, k, elsize;
    int refcnt;
    npy_intp* new_dimensions=newshape->ptr;
    npy_intp new_strides[NPY_MAXDIMS];
    size_t sd;
    npy_intp *dimptr;
    char *new_data;
    npy_intp largest;

    if (!NpyArray_ISONESEGMENT(self)) {
        NpyErr_SetString(NpyExc_ValueError,
                         "resize only works on single-segment arrays");
        return -1;
    }

    if (self->descr->elsize == 0) {
        NpyErr_SetString(NpyExc_ValueError,
                         "Bad data-type size.");
        return -1;
    }
    newsize = 1;
    largest = NPY_MAX_INTP / self->descr->elsize;
    for(k = 0; k < new_nd; k++) {
        if (new_dimensions[k] == 0) {
            break;
        }
        if (new_dimensions[k] < 0) {
            NpyErr_SetString(NpyExc_ValueError,
                    "negative dimensions not allowed");
            return -1;
        }
        newsize *= new_dimensions[k];
        if (newsize <= 0 || newsize > largest) {
            NpyErr_NoMemory();
            return -1;
        }
    }
    oldsize = NpyArray_SIZE(self);

    if (oldsize != newsize) {
        if (!(self->flags & NPY_OWNDATA)) {
            NpyErr_SetString(NpyExc_ValueError,
                    "cannot resize this array: it does not own its data");
            return -1;
        }

        /* TODO: This isn't right for usage from C.  I think we
           need to revisit the refcounts so we don't have counts
           of 0. */
        if (refcheck) {
            refcnt = self->nob_refcnt;
        }
        else {
            refcnt = 0;
        }
        if ((refcnt > 0)
            || (self->base_arr != NULL) || (NULL != self->base_obj)) {
            NpyErr_SetString(NpyExc_ValueError,
                    "cannot resize an array references or is referenced\n"\
                    "by another array in this way.  Use the resize function");
            return -1;
        }

        if (newsize == 0) {
            sd = self->descr->elsize;
        }
        else {
            sd = newsize*self->descr->elsize;
        }
        /* Reallocate space if needed */
        new_data = NpyDataMem_RENEW(self->data, sd);
        if (new_data == NULL) {
            NpyErr_SetString(NpyExc_MemoryError,
                    "cannot allocate memory for array");
            return -1;
        }
        self->data = new_data;
    }

    if ((newsize > oldsize) && NpyArray_ISWRITEABLE(self)) {
        /* Fill new memory with zeros */
        elsize = self->descr->elsize;
        memset(self->data+oldsize*elsize, 0, (newsize-oldsize)*elsize);
    }

    if (self->nd != new_nd) {
        /* Different number of dimensions. */
        self->nd = new_nd;
        /* Need new dimensions and strides arrays */
        dimptr = NpyDimMem_RENEW(self->dimensions, 2*new_nd);
        if (dimptr == NULL) {
            NpyErr_SetString(NpyExc_MemoryError,
                    "cannot allocate memory for array");
            return -1;
        }
        self->dimensions = dimptr;
        self->strides = dimptr + new_nd;
    }

    /* make new_strides variable */
    sd = (size_t) self->descr->elsize;
    sd = (size_t) _array_fill_strides(new_strides, new_dimensions, new_nd, sd,
            self->flags, &(self->flags));
    memmove(self->dimensions, new_dimensions, new_nd*sizeof(npy_intp));
    memmove(self->strides, new_strides, new_nd*sizeof(npy_intp));
    return 0;
}

/*
 * Returns a new array
 * with the new shape from the data
 * in the old array --- order-perspective depends on fortran argument.
 * copy-only-if-necessary
 */

/*
 * New shape for an array
 */
NpyArray*
NpyArray_Newshape(NpyArray* self, NpyArray_Dims *newdims,
                  NPY_ORDER fortran)
{
    npy_intp i;
    npy_intp *dimensions = newdims->ptr;
    NpyArray *ret;
    int n = newdims->len;
    npy_bool same, incref = NPY_TRUE;
    npy_intp *strides = NULL;
    npy_intp newstrides[NPY_MAXDIMS];
    int flags;

    if (fortran == NPY_ANYORDER) {
        fortran = NpyArray_ISFORTRAN(self);
    }
    /*  Quick check to make sure anything actually needs to be done */
    if (n == self->nd) {
        same = NPY_TRUE;
        i = 0;
        while (same && i < n) {
            if (NpyArray_DIM(self,i) != dimensions[i]) {
                same=NPY_FALSE;
            }
            i++;
        }
        if (same) {
            return NpyArray_View(self, NULL, NULL);
        }
    }

    /*
     * Returns a pointer to an appropriate strides array
     * if all we are doing is inserting ones into the shape,
     * or removing ones from the shape
     * or doing a combination of the two
     * In this case we don't need to do anything but update strides and
     * dimensions.  So, we can handle non single-segment cases.
     */
    i = _check_ones(self, n, dimensions, newstrides);
    if (i == 0) {
        strides = newstrides;
    }
    flags = self->flags;

    if (strides == NULL) {
        /*
         * we are really re-shaping not just adding ones to the shape somewhere
         * fix any -1 dimensions and check new-dimensions against old size
         */
        if (_fix_unknown_dimension(newdims, NpyArray_SIZE(self)) < 0) {
            return NULL;
        }
        /*
         * sometimes we have to create a new copy of the array
         * in order to get the right orientation and
         * because we can't just re-use the buffer with the
         * data in the order it is in.
         */
        if (!(NpyArray_ISONESEGMENT(self)) ||
            (((NpyArray_CHKFLAGS(self, NPY_CONTIGUOUS) &&
               fortran == NPY_FORTRANORDER) ||
              (NpyArray_CHKFLAGS(self, NPY_FORTRAN) &&
                  fortran == NPY_CORDER)) && (self->nd > 1))) {
            int success = 0;
            success = _attempt_nocopy_reshape(self,n,dimensions,
                                              newstrides,fortran);
            if (success) {
                /* no need to copy the array after all */
                strides = newstrides;
                flags = self->flags;
            }
            else {
                NpyArray *new;
                new = NpyArray_NewCopy(self, fortran);
                if (new == NULL) {
                    return NULL;
                }
                incref = NPY_FALSE;
                self = new;
                flags = self->flags;
            }
        }

        /* We always have to interpret the contiguous buffer correctly */

        /* Make sure the flags argument is set. */
        if (n > 1) {
            if (fortran == NPY_FORTRANORDER) {
                flags &= ~NPY_CONTIGUOUS;
                flags |= NPY_FORTRAN;
            }
            else {
                flags &= ~NPY_FORTRAN;
                flags |= NPY_CONTIGUOUS;
            }
        }
    }
    else if (n > 0) {
        /*
         * replace any 0-valued strides with
         * appropriate value to preserve contiguousness
         */
        if (fortran == NPY_FORTRANORDER) {
            if (strides[0] == 0) {
                strides[0] = self->descr->elsize;
            }
            for (i = 1; i < n; i++) {
                if (strides[i] == 0) {
                    strides[i] = strides[i-1] * dimensions[i-1];
                }
            }
        }
        else {
            if (strides[n-1] == 0) {
                strides[n-1] = self->descr->elsize;
            }
            for (i = n - 2; i > -1; i--) {
                if (strides[i] == 0) {
                    strides[i] = strides[i+1] * dimensions[i+1];
                }
            }
        }
    }

    _Npy_INCREF(self->descr);
    ret = NpyArray_NewFromDescr(self->descr,
                                n, dimensions,
                                strides,
                                self->data,
                                flags, NPY_FALSE, NULL,
                                Npy_INTERFACE(self));

    if (ret == NULL) {
        goto fail;
    }
    if (incref) {
        _Npy_INCREF(self);
    }
    ret->base_arr = self;
    NpyArray_UpdateFlags(ret, NPY_CONTIGUOUS | NPY_FORTRAN);
    assert(NULL == ret->base_arr || NULL == ret->base_obj);
    return ret;

 fail:
    if (!incref) {
        _Npy_DECREF(self);
    }
    return NULL;
}


/*
 * return a new view of the array object with all of its unit-length
 * dimensions squeezed out if needed, otherwise
 * return the same array.
 */
NpyArray*
NpyArray_Squeeze(NpyArray *self)
{
    int nd = self->nd;
    int newnd = nd;
    npy_intp dimensions[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];
    int i, j;
    NpyArray *ret;

    if (nd == 0) {
        _Npy_INCREF(self);
        return self;
    }
    for (j = 0, i = 0; i < nd; i++) {
        if (self->dimensions[i] == 1) {
            newnd -= 1;
        }
        else {
            dimensions[j] = self->dimensions[i];
            strides[j++] = self->strides[i];
        }
    }

    _Npy_INCREF(self->descr);
    ret = NpyArray_NewFromDescr(self->descr,
                                newnd, dimensions,
                                strides, self->data,
                                self->flags,
                                NPY_FALSE, NULL,
                                Npy_INTERFACE(self));
    if (ret == NULL) {
        return NULL;
    }
    NpyArray_FLAGS(ret) &= ~NPY_OWNDATA;
    ret->base_arr = self;
    _Npy_INCREF(self);
    assert(NULL == ret->base_arr || NULL == ret->base_obj);
    return ret;
}

/*
 * SwapAxes
 */
NpyArray*
NpyArray_SwapAxes(NpyArray *ap, int a1, int a2)
{
    NpyArray_Dims new_axes;
    npy_intp dims[NPY_MAXDIMS];
    int n, i, val;
    NpyArray *ret;

    if (a1 == a2) {
        _Npy_INCREF(ap);
        return ap;
    }

    n = ap->nd;
    if (n <= 1) {
        _Npy_INCREF(ap);
        return ap;
    }

    if (a1 < 0) {
        a1 += n;
    }
    if (a2 < 0) {
        a2 += n;
    }
    if ((a1 < 0) || (a1 >= n)) {
        NpyErr_SetString(NpyExc_ValueError,
                        "bad axis1 argument to swapaxes");
        return NULL;
    }
    if ((a2 < 0) || (a2 >= n)) {
        NpyErr_SetString(NpyExc_ValueError,
                        "bad axis2 argument to swapaxes");
        return NULL;
    }
    new_axes.ptr = dims;
    new_axes.len = n;

    for (i = 0; i < n; i++) {
        if (i == a1) {
            val = a2;
        }
        else if (i == a2) {
            val = a1;
        }
        else {
            val = i;
        }
        new_axes.ptr[i] = val;
    }
    ret = NpyArray_Transpose(ap, &new_axes);
    return ret;
}

/*
 * Return Transpose.
 */
NpyArray*
NpyArray_Transpose(NpyArray *ap, NpyArray_Dims *permute)
{
    npy_intp *axes, axis;
    npy_intp i, n;
    npy_intp permutation[NPY_MAXDIMS], reverse_permutation[NPY_MAXDIMS];
    NpyArray *ret = NULL;

    if (permute == NULL) {
        n = ap->nd;
        for (i = 0; i < n; i++) {
            permutation[i] = n-1-i;
        }
    }
    else {
        n = permute->len;
        axes = permute->ptr;
        if (n != ap->nd) {
            NpyErr_SetString(NpyExc_ValueError,
                            "axes don't match array");
            return NULL;
        }
        for (i = 0; i < n; i++) {
            reverse_permutation[i] = -1;
        }
        for (i = 0; i < n; i++) {
            axis = axes[i];
            if (axis < 0) {
                axis = ap->nd + axis;
            }
            if (axis < 0 || axis >= ap->nd) {
                NpyErr_SetString(NpyExc_ValueError,
                                "invalid axis for this array");
                return NULL;
            }
            if (reverse_permutation[axis] != -1) {
                NpyErr_SetString(NpyExc_ValueError,
                                "repeated axis in transpose");
                return NULL;
            }
            reverse_permutation[axis] = i;
            permutation[i] = axis;
        }
        for (i = 0; i < n; i++) {
        }
    }

    /*
     * this allocates memory for dimensions and strides (but fills them
     * incorrectly), sets up descr, and points data at ap->data.
     */
    _Npy_INCREF(ap->descr);
    ret = NpyArray_NewFromDescr(ap->descr,
                                n, ap->dimensions,
                                NULL, ap->data, ap->flags,
                                NPY_FALSE, NULL,
                                Npy_INTERFACE(ap));
    if (ret == NULL) {
        return NULL;
    }
    /* point at true owner of memory: */
    ret->base_arr = ap;
    assert(NULL == ret->base_arr || NULL == ret->base_obj);
    _Npy_INCREF(ap);

    /* fix the dimensions and strides of the return-array */
    for (i = 0; i < n; i++) {
        ret->dimensions[i] = ap->dimensions[permutation[i]];
        ret->strides[i] = ap->strides[permutation[i]];
    }
    NpyArray_UpdateFlags(ret, NPY_CONTIGUOUS | NPY_FORTRAN);
    return ret;
}

/*
 * Ravel
 * Returns a contiguous array
 */
NpyArray*
NpyArray_Ravel(NpyArray *a, NPY_ORDER fortran)
{
    NpyArray_Dims newdim = {NULL,1};
    npy_intp val[1] = {-1};

    if (fortran == NPY_ANYORDER) {
        fortran = NpyArray_ISFORTRAN(a);
    }
    newdim.ptr = val;
    if (!fortran && NpyArray_ISCONTIGUOUS(a)) {
        return NpyArray_Newshape(a, &newdim, NPY_CORDER);
    }
    else if (fortran && NpyArray_ISFORTRAN(a)) {
        return NpyArray_Newshape(a, &newdim, NPY_FORTRANORDER);
    }
    else {
        return NpyArray_Flatten(a, fortran);
    }
}

/*
 * Flatten
 */
NpyArray *
NpyArray_Flatten(NpyArray *a, NPY_ORDER order)
{
    NpyArray *ret;
    npy_intp size;

    if (order == NPY_ANYORDER) {
        order = NpyArray_ISFORTRAN(a);
    }
    _Npy_INCREF(a->descr);
    size = NpyArray_SIZE(a);
    ret = NpyArray_NewFromDescr(a->descr,
                                1, &size,
                                NULL,
                                NULL,
                                0,
                                NPY_FALSE, NULL,
                                Npy_INTERFACE(a));

    if (ret == NULL) {
        return NULL;
    }
    if (_flat_copyinto(ret, a, order) < 0) {
        _Npy_DECREF(ret);
        return NULL;
    }
    return ret;
}


/* inserts 0 for strides where dimension will be 1 */
static int
_check_ones(NpyArray *self, int newnd, npy_intp* newdims, npy_intp *strides)
{
    int nd;
    npy_intp *dims;
    npy_bool done=NPY_FALSE;
    int j, k;

    nd = self->nd;
    dims = self->dimensions;

    for (k = 0, j = 0; !done && (j < nd || k < newnd);) {
        if ((j<nd) && (k<newnd) && (newdims[k] == dims[j])) {
            strides[k] = self->strides[j];
            j++;
            k++;
        }
        else if ((k < newnd) && (newdims[k] == 1)) {
            strides[k] = 0;
            k++;
        }
        else if ((j<nd) && (dims[j] == 1)) {
            j++;
        }
        else {
            done = NPY_TRUE;
        }
    }
    if (done) {
        return -1;
    }
    return 0;
}

/*
 * attempt to reshape an array without copying data
 *
 * This function should correctly handle all reshapes, including
 * axes of length 1. Zero strides should work but are untested.
 *
 * If a copy is needed, returns 0
 * If no copy is needed, returns 1 and fills newstrides
 *     with appropriate strides
 *
 * The "fortran" argument describes how the array should be viewed
 * during the reshape, not how it is stored in memory (that
 * information is in self->strides).
 *
 * If some output dimensions have length 1, the strides assigned to
 * them are arbitrary. In the current implementation, they are the
 * stride of the next-fastest index.
 */
static int
_attempt_nocopy_reshape(NpyArray *self, int newnd, npy_intp* newdims,
                        npy_intp *newstrides, int fortran)
{
    int oldnd;
    npy_intp olddims[NPY_MAXDIMS];
    npy_intp oldstrides[NPY_MAXDIMS];
    int oi, oj, ok, ni, nj, nk;
    int np, op;

    oldnd = 0;
    for (oi = 0; oi < self->nd; oi++) {
        if (self->dimensions[oi]!= 1) {
            olddims[oldnd] = self->dimensions[oi];
            oldstrides[oldnd] = self->strides[oi];
            oldnd++;
        }
    }

    /*
      fprintf(stderr, "_attempt_nocopy_reshape( (");
      for (oi=0; oi<oldnd; oi++)
      fprintf(stderr, "(%d,%d), ", olddims[oi], oldstrides[oi]);
      fprintf(stderr, ") -> (");
      for (ni=0; ni<newnd; ni++)
      fprintf(stderr, "(%d,*), ", newdims[ni]);
      fprintf(stderr, "), fortran=%d)\n", fortran);
    */


    np = 1;
    for (ni = 0; ni < newnd; ni++) {
        np *= newdims[ni];
    }
    op = 1;
    for (oi = 0; oi < oldnd; oi++) {
        op *= olddims[oi];
    }
    if (np != op) {
        /* different total sizes; no hope */
        return 0;
    }
    /* the current code does not handle 0-sized arrays, so give up */
    if (np == 0) {
        return 0;
    }

    oi = 0;
    oj = 1;
    ni = 0;
    nj = 1;
    while(ni < newnd && oi < oldnd) {
        np = newdims[ni];
        op = olddims[oi];

        while (np != op) {
            if (np < op) {
                np *= newdims[nj++];
            } else {
                op *= olddims[oj++];
            }
        }

        for (ok = oi; ok < oj - 1; ok++) {
            if (fortran) {
                if (oldstrides[ok+1] != olddims[ok]*oldstrides[ok]) {
                     /* not contiguous enough */
                    return 0;
                }
            }
            else {
                /* C order */
                if (oldstrides[ok] != olddims[ok+1]*oldstrides[ok+1]) {
                    /* not contiguous enough */
                    return 0;
                }
            }
        }

        if (fortran) {
            newstrides[ni] = oldstrides[oi];
            for (nk = ni + 1; nk < nj; nk++) {
                newstrides[nk] = newstrides[nk - 1]*newdims[nk - 1];
            }
        }
        else {
            /* C order */
            newstrides[nj - 1] = oldstrides[oj - 1];
            for (nk = nj - 1; nk > ni; nk--) {
                newstrides[nk - 1] = newstrides[nk]*newdims[nk];
            }
        }
        ni = nj++;
        oi = oj++;
    }

    /*
      fprintf(stderr, "success: _attempt_nocopy_reshape (");
      for (oi=0; oi<oldnd; oi++)
      fprintf(stderr, "(%d,%d), ", olddims[oi], oldstrides[oi]);
      fprintf(stderr, ") -> (");
      for (ni=0; ni<newnd; ni++)
      fprintf(stderr, "(%d,%d), ", newdims[ni], newstrides[ni]);
      fprintf(stderr, ")\n");
    */

    return 1;
}

static int
_fix_unknown_dimension(NpyArray_Dims *newshape, npy_intp s_original)
{
    npy_intp *dimensions;
    npy_intp i_unknown, s_known;
    int i, n;
    static char msg[] = "total size of new array must be unchanged";

    dimensions = newshape->ptr;
    n = newshape->len;
    s_known = 1;
    i_unknown = -1;

    for (i = 0; i < n; i++) {
        if (dimensions[i] < 0) {
            if (i_unknown == -1) {
                i_unknown = i;
            }
            else {
                NpyErr_SetString(NpyExc_ValueError,
                                 "can only specify one" \
                                 " unknown dimension");
                return -1;
            }
        }
        else {
            s_known *= dimensions[i];
        }
    }

    if (i_unknown >= 0) {
        if ((s_known == 0) || (s_original % s_known != 0)) {
            NpyErr_SetString(NpyExc_ValueError, msg);
            return -1;
        }
        dimensions[i_unknown] = s_original/s_known;
    }
    else {
        if (s_original != s_known) {
            NpyErr_SetString(NpyExc_ValueError, msg);
            return -1;
        }
    }
    return 0;
}

