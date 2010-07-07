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

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "npy_config.h"
#include "numpy/numpy_api.h"
#include "npy_3kcompat.h"

NpyArray_Descr **npy_userdescrs=NULL;

static PyArray_CastFuncsItem* 
castfuncs_new(void)
{
    PyArray_CastFuncsItem* result = 
        (PyArray_CastFuncsItem *) malloc(sizeof(PyArray_CastFuncsItem));
    result[0].totype = NPY_NOTYPE;
    return result;
}

static PyArray_CastFuncsItem* 
castfuncs_append(PyArray_CastFuncsItem* items,
                 int totype, PyArray_VectorUnaryFunc* func)
{
    int n = 0;

    while (items[n].totype != NPY_NOTYPE) {
        n++;
    }
    items = (PyArray_CastFuncsItem *)
        realloc(items, (n + 2)*sizeof(PyArray_CastFuncsItem));
    items[n].totype = totype;
    items[n].castfunc = func;
    items[n+1].totype = NPY_NOTYPE;

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
    newtypes = (int *)realloc(types, (n + 2)*sizeof(int));
    newtypes[n] = insert;
    newtypes[n + 1] = NPY_NOTYPE;
    return newtypes;
}

static npy_bool
_default_nonzero(void *ip, void *arr)
{
    int elsize = NpyArray_ITEMSIZE((NpyArray *)arr);
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

    copyswap = NpyArray_DESCR((NpyArray *)arr)->f->copyswap;

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
    for (i = 0; i < NPY_NUMUSERTYPES; i++) {
        descr2 = npy_userdescrs[i];
        if (descr2 == descr) {
            return descr->type_num;
        }
    }
    typenum = NPY_USERDEF + NPY_NUMUSERTYPES;
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
    npy_userdescrs = realloc(npy_userdescrs,
                             (NPY_NUMUSERTYPES+1)*sizeof(void *));
    if (npy_userdescrs == NULL) {
        NpyErr_SetString(NpyExc_MemoryError, "RegisterDataType");
        return -1;
    }
    npy_userdescrs[NPY_NUMUSERTYPES++] = descr;
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
            descr->f->cancastto = (int *)malloc(1*sizeof(int));
            descr->f->cancastto[0] = NPY_NOTYPE;
        }
        descr->f->cancastto = _append_new(descr->f->cancastto,
                                          totype);
    }
    else {
        /* register with cancastscalarkindto */
        if (descr->f->cancastscalarkindto == NULL) {
            int i;
            descr->f->cancastscalarkindto =
                (int **)malloc(NPY_NSCALARKINDS* sizeof(int*));
            for (i = 0; i < NPY_NSCALARKINDS; i++) {
                descr->f->cancastscalarkindto[i] = NULL;
            }
        }
        if (descr->f->cancastscalarkindto[scalar] == NULL) {
            descr->f->cancastscalarkindto[scalar] =
                (int *)malloc(1*sizeof(int));
            descr->f->cancastscalarkindto[scalar][0] =
                NPY_NOTYPE;
        }
        descr->f->cancastscalarkindto[scalar] =
            _append_new(descr->f->cancastscalarkindto[scalar], totype);
    }
    return 0;
}

int
NpyArray_TypeNumFromName(char *str)
{
    int i;
    NpyArray_Descr *descr;

    /* TODO: This looks at the python type and needs to change. */
    for (i = 0; i < NPY_NUMUSERTYPES; i++) {
        descr = npy_userdescrs[i];
        if (strcmp(descr->typeobj->tp_name, str) == 0) {
            return descr->type_num;
        }
    }
    return NPY_NOTYPE;
}

int
NpyArray_TypeNumFromTypeObj(void* typeobj)
{
    int i;
    NpyArray_Descr *descr;

    /* TODO: This looks at the python type and needs to change. */
    for (i = 0; i < NPY_NUMUSERTYPES; i++) {
        descr = npy_userdescrs[i];
        if (descr->typeobj == typeobj) {
            return descr->type_num;
        }
    }
    return NPY_NOTYPE;
}

NpyArray_Descr*
NpyArray_UserDescrFromTypeNum(int typenum)
{
    return npy_userdescrs[typenum - NPY_USERDEF];
}
