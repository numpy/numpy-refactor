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

#include <stdlib.h>
#include <memory.h>
#include "npy_config.h"
#include "npy_api.h"
#include "npy_arrayobject.h"


static int numusertypes = 0;

NpyArray_Descr **npy_userdescrs=NULL;

static NpyArray_CastFuncsItem*
castfuncs_new(void)
{
    NpyArray_CastFuncsItem* result =
        (NpyArray_CastFuncsItem *) malloc(sizeof(NpyArray_CastFuncsItem));
    result[0].totype = NPY_NOTYPE;
    return result;
}

static NpyArray_CastFuncsItem*
castfuncs_append(NpyArray_CastFuncsItem* items,
                 int totype, NpyArray_VectorUnaryFunc* func)
{
    int n = 0;

    while (items[n].totype != NPY_NOTYPE) {
        n++;
    }
    items = (NpyArray_CastFuncsItem *)
        realloc(items, (n + 2) * sizeof(NpyArray_CastFuncsItem));
    items[n].totype = totype;
    items[n].castfunc = func;
    items[n + 1].totype = NPY_NOTYPE;

    return items;
}

static int *
_append_new(int *types, int insert)
{
    int n = 0;
    int *newtypes;

    while (types[n] != NPY_NOTYPE) {
        n++;
    }
    newtypes = (int *)realloc(types, (n + 2) * sizeof(int));
    newtypes[n] = insert;
    newtypes[n + 1] = NPY_NOTYPE;
    return newtypes;
}

static npy_bool
_default_nonzero(void *ip, NpyArray *arr)
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
                   npy_intp sstride, npy_intp n, int swap, NpyArray *arr)
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

/*
  Initialize arrfuncs to NULL
*/
void
NpyArray_InitArrFuncs(NpyArray_ArrFuncs *f)
{
    int i;

    for(i = 0; i < NPY_NTYPES; i++) {
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
    for(i = 0; i < NPY_NSORTS; i++) {
        f->sort[i] = NULL;
        f->argsort[i] = NULL;
    }
    f->castfuncs = NULL;
    f->scalarkind = NULL;
    f->cancastscalarkindto = NULL;
    f->cancastto = NULL;
}


int
NpyArray_GetNumusertypes(void)
{
    return numusertypes;
}


/*
  returns typenum to associate with this type >=NPY_USERDEF.
  needs the userdecrs table and NPY_NUMUSER variables
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
    for (i = 0; i < numusertypes; i++) {
        descr2 = npy_userdescrs[i];
        if (descr2 == descr) {
            return descr->type_num;
        }
    }
    typenum = NPY_USERDEF + numusertypes;
    descr->type_num = typenum;
    if (descr->elsize == 0) {
        NpyErr_SetString(NpyExc_ValueError, "cannot register a"
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
        NpyErr_SetString(NpyExc_ValueError, "a required array function"
                         " is missing.");
        return -1;
    }
    /* TODO: Can't check typeobj down here in the core.  Do we need a
       callback or check on the way in? */
 /*   if (descr->typeobj == NULL) {
        NpyErr_SetString(NpyExc_ValueError, "missing typeobject");
        return -1;
    } */
    npy_userdescrs = realloc(npy_userdescrs,
                             (numusertypes + 1) * sizeof(void *));
    if (npy_userdescrs == NULL) {
        NpyErr_MEMORY;
        return -1;
    }
    npy_userdescrs[numusertypes++] = descr;
    return typenum;
}

/*
  Register Casting Function
  Replaces any function currently stored.
*/
int
NpyArray_RegisterCastFunc(NpyArray_Descr *descr, int totype,
                          NpyArray_VectorUnaryFunc *castfunc)
{
    if (totype < NPY_NTYPES) {
        descr->f->cast[totype] = castfunc;
        return 0;
    }
    if (!NpyTypeNum_ISUSERDEF(totype)) {
        NpyErr_SetString(NpyExc_TypeError, "invalid type number.");
        return -1;
    }
    if (descr->f->castfuncs == NULL) {
        descr->f->castfuncs = castfuncs_new();
        if (descr->f->castfuncs == NULL) {
            return -1;
        }
    }
    descr->f->castfuncs =
        castfuncs_append(descr->f->castfuncs, totype, castfunc);
    return 0;
}

/*
 * Register a type number indicating that a descriptor can be cast
 * to it safely
 */
int
NpyArray_RegisterCanCast(NpyArray_Descr *descr, int totype,
                         NPY_SCALARKIND scalar)
{
    if (scalar == NPY_NOSCALAR) {
        /*
         * register with cancastto
         * These lists won't be freed once created
         * -- they become part of the data-type
         */
        if (descr->f->cancastto == NULL) {
            descr->f->cancastto = (int *)NpyArray_malloc(1 * sizeof(int));
            descr->f->cancastto[0] = NPY_NOTYPE;
        }
        descr->f->cancastto = _append_new(descr->f->cancastto, totype);
    }
    else {
        /* register with cancastscalarkindto */
        if (descr->f->cancastscalarkindto == NULL) {
            int i;
            descr->f->cancastscalarkindto =
                (int **)NpyArray_malloc(NPY_NSCALARKINDS * sizeof(int *));
            for (i = 0; i < NPY_NSCALARKINDS; i++) {
                descr->f->cancastscalarkindto[i] = NULL;
            }
        }
        if (descr->f->cancastscalarkindto[scalar] == NULL) {
            descr->f->cancastscalarkindto[scalar] =
                (int *)NpyArray_malloc(1 * sizeof(int));
            descr->f->cancastscalarkindto[scalar][0] = NPY_NOTYPE;
        }
        descr->f->cancastscalarkindto[scalar] =
            _append_new(descr->f->cancastscalarkindto[scalar], totype);
    }
    return 0;
}


NpyArray_Descr*
NpyArray_UserDescrFromTypeNum(int typenum)
{
    return npy_userdescrs[typenum - NPY_USERDEF];
}
