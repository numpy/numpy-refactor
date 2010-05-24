/*
  Provide multidimensional arrays as a basic object type in python.

  Based on Original Numeric implementation
  Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu

  with contributions from many Numeric Python developers 1995-2004

  Heavily modified in 2005 with inspiration from Numarray

  by

  Travis Oliphant,  oliphant@ee.byu.edu
  Brigham Young Univeristy


maintainer email:  oliphant.travis@ieee.org

  Numarray design (which provided guidance) by
  Space Science Telescope Institute
  (J. Todd Miller, Perry Greenfield, Rick White)
*/

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"

NpyArray_Descr **npy_userdescrs=NULL;

static int *
_append_new(int *types, int insert)
{
    int n = 0;
    int *newtypes;

    while (types[n] != NpyArray_NOTYPE) {
        n++;
    }
    newtypes = (int *)realloc(types, (n + 2)*sizeof(int));
    newtypes[n] = insert;
    newtypes[n + 1] = NpyArray_NOTYPE;
    return newtypes;
}

static npy_bool
_default_nonzero(void *ip, void *arr)
{
    int elsize = NpyArray_ITEMSIZE(arr);
    char *ptr = ip;
    while (elsize--) {
        if (*ptr++ != 0) {
            return NPY_TRUE;
        }
    }
    return NPY_FALSE;
}

static void
_default_copyswapn(void *dst, npy_intp dstride, void *src,
                   npy_intp sstride, npy_intp n, int swap, void *arr)
{
    npy_intp i;
    NpyArray_CopySwapFunc *copyswap;
    char *dstptr = dst;
    char *srcptr = src;

    copyswap = NpyArray_DESCR(arr)->f->copyswap;

    for (i = 0; i < n; i++) {
        copyswap(dstptr, srcptr, swap, arr);
        dstptr += dstride;
        srcptr += sstride;
    }
}

/*NUMPY_API
  Initialize arrfuncs to NULL
*/
void
NpyArray_InitArrFuncs(NpyArray_ArrFuncs *f)
{
    int i;

    for(i = 0; i < PyArray_NTYPES; i++) {
        f->cast[i] = NULL;
    }
    f->getitem = NULL;
    f->setitem = NULL;
    f->copyswapn = NULL;
    f->copyswap = NULL;
    f->compare = NULL;
    f->argmax = NULL;
    f->dotfunc = NULL;
    f->scanfunc = NULL;
    f->fromstr = NULL;
    f->nonzero = NULL;
    f->fill = NULL;
    f->fillwithscalar = NULL;
    for(i = 0; i < PyArray_NSORTS; i++) {
        f->sort[i] = NULL;
        f->argsort[i] = NULL;
    }
    f->castdict = NULL;
    f->scalarkind = NULL;
    f->cancastscalarkindto = NULL;
    f->cancastto = NULL;
}


/*
  returns typenum to associate with this type >=PyArray_USERDEF.
  needs the userdecrs table and PyArray_NUMUSER variables
  defined in arraytypes.inc
*/
/*
  Register Data type
  Does not change the reference count of descr
*/
int
NpyArray_RegisterDataType(NpyArray_Descr *descr)
{
    NpyArray_Descr *descr2;
    int typenum;
    int i;
    NpyArray_ArrFuncs *f;

    /* See if this type is already registered */
    for (i = 0; i < NPY_NUMUSERTYPES; i++) {
        descr2 = npy_userdescrs[i];
        if (descr2 == descr) {
            return descr->type_num;
        }
    }
    typenum = PyArray_USERDEF + NPY_NUMUSERTYPES;
    descr->type_num = typenum;
    if (descr->elsize == 0) {
        NpyErr_SetString(NpyExc_ValueError, "cannot register a" \
                         "flexible data-type");
        return -1;
    }
    f = descr->f;
    if (f->nonzero == NULL) {
        f->nonzero = _default_nonzero;
    }
    if (f->copyswapn == NULL) {
        f->copyswapn = _default_copyswapn;
    }
    if (f->copyswap == NULL || f->getitem == NULL ||
        f->setitem == NULL) {
        NpyErr_SetString(NpyExc_ValueError, "a required array function"   \
                         " is missing.");
        return -1;
    }
    if (descr->typeobj == NULL) {
        NpyErr_SetString(NpyExc_ValueError, "missing typeobject");
        return -1;
    }
    npy_userdescrs = realloc(userdescrs,
                             (NPY_NUMUSERTYPES+1)*sizeof(void *));
    if (npy_userdescrs == NULL) {
        NpyErr_SetString(NpyExc_MemoryError, "RegisterDataType");
        return -1;
    }
    npy_userdescrs[NPY_NUMUSERTYPES++] = descr;
    return typenum;
}
