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
#include "structmember.h"

/*#include <stdio.h>*/
#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "common.h"

#include "numpy_3kcompat.h"

#include "usertypes.h"

#include "numpy/numpy_api.h"

/*NUMPY_API
  Initialize arrfuncs to NULL
*/
NPY_NO_EXPORT void
PyArray_InitArrFuncs(PyArray_ArrFuncs *f)
{
    NpyArray_InitArrFuncs(f);
}


int
PyArray_TypeNumFromTypeObj(PyTypeObject *typeobj)
{
    int i;
    NpyArray_Descr *descr;
    
    /* TODO: This looks at the python type and needs to change. */
    for (i = 0; i < NpyArray_GetNumusertypes(); i++) {
        descr = npy_userdescrs[i];
        if (PyArray_Descr_WRAP(descr)->typeobj == typeobj) {
            return descr->type_num;
        }
    }
    return NPY_NOTYPE;
}


/*
 returns typenum to associate with this type >=PyArray_USERDEF.
  needs the userdecrs table and PyArray_NUMUSER variables
  defined in arraytypes.inc
*/
/*NUMPY_API
  Register Data type
  Does not change the reference count of descr
*/
NPY_NO_EXPORT int
PyArray_RegisterDataType(PyArray_Descr *descr)
{
    return NpyArray_RegisterDataType(descr->descr);
}

/*NUMPY_API
  Register Casting Function
  Replaces any function currently stored.
*/
NPY_NO_EXPORT int
PyArray_RegisterCastFunc(PyArray_Descr *descr, int totype,
                         PyArray_VectorUnaryFunc *castfunc)
{
    return NpyArray_RegisterCastFunc(descr->descr, totype, castfunc);
}

/*NUMPY_API
 * Register a type number indicating that a descriptor can be cast
 * to it safely
 */
NPY_NO_EXPORT int
PyArray_RegisterCanCast(PyArray_Descr *descr, int totype,
                        NPY_SCALARKIND scalar)
{
    return NpyArray_RegisterCanCast(descr->descr, totype, scalar);
}

