#ifndef _NPY_PRIVATE_USERTYPES_H_
#define _NPY_PRIVATE_USERTYPES_H_

extern PyArray_Descr **userdescrs;

NPY_NO_EXPORT void
PyArray_InitArrFuncs(PyArray_ArrFuncs *f);

NPY_NO_EXPORT int
PyArray_RegisterCanCast(PyArray_Descr *descr, int totype,
                        NPY_SCALARKIND scalar);

NPY_NO_EXPORT int
PyArray_RegisterDataType(PyArray_Descr *descr);

NPY_NO_EXPORT int
PyArray_RegisterCastFunc(PyArray_Descr *descr, int totype,
                         PyArray_VectorUnaryFunc *castfunc);

#endif
