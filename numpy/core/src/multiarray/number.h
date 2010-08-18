#ifndef _NPY_ARRAY_NUMBER_H_
#define _NPY_ARRAY_NUMBER_H_

#include <npy_ufunc_object.h>
#include "numpy/ufuncobject.h"


extern NPY_NO_EXPORT PyNumberMethods array_as_number;


NPY_NO_EXPORT PyObject *
array_int(PyArrayObject *v);

NPY_NO_EXPORT PyObject *
PyArray_GetNumericOp(enum NpyArray_Ops op);
                     
NPY_NO_EXPORT int
PyArray_SetNumericOps(PyObject *dict);

NPY_NO_EXPORT PyObject *
PyArray_GetNumericOps(void);

NPY_NO_EXPORT PyObject *
PyArray_GenericBinaryFunction(PyArrayObject *m1, PyObject *m2, PyObject *op);

NPY_NO_EXPORT PyObject *
PyArray_GenericUnaryFunction(PyArrayObject *m1, PyObject *op);

NPY_NO_EXPORT PyObject *
PyArray_GenericReduceFunction(PyArrayObject *m1, PyObject *op, int axis,
                              int rtype, PyArrayObject *out);

NPY_NO_EXPORT PyObject *
PyArray_GenericAccumulateFunction(PyArrayObject *m1, PyObject *op, int axis,
                                  int rtype, PyArrayObject *out);

#endif
