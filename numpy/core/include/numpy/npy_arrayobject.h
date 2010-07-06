#ifndef _NPY_ARRAYOBJECT_H_
#define _NPY_ARRAYOBJECT_H_

#include "npy_object.h"
#include "npy_defs.h"

npy_intp NpyArray_MultiplyList(npy_intp *l1, int n);
int NpyArray_CompareLists(npy_intp *l1, npy_intp *l2, int n);


#define NpyArray_CHKFLAGS(m, FLAGS)                              \
    (((m)->flags & (FLAGS)) == (FLAGS))

#define NpyArray_ISCONTIGUOUS(m) NpyArray_CHKFLAGS(m, NPY_CONTIGUOUS)
#define NpyArray_ISWRITEABLE(m) NpyArray_CHKFLAGS(m, NPY_WRITEABLE)
#define NpyArray_ISALIGNED(m) NpyArray_CHKFLAGS(m, NPY_ALIGNED)


#define NpyArray_NDIM(obj) ((obj)->nd)
#define NpyArray_ISONESEGMENT(m) (NpyArray_NDIM(m) == 0 ||              \
                                 NpyArray_CHKFLAGS(m, NPY_CONTIGUOUS) || \
                                 NpyArray_CHKFLAGS(m, NPY_FORTRAN))

#define NpyArray_ISFORTRAN(m) (NpyArray_CHKFLAGS(m, NPY_FORTRAN) &&     \
                               (NpyArray_NDIM(m) > 1))

#define NpyArray_FORTRAN_IF(m) ((PyArray_CHKFLAGS(m, NPY_FORTRAN) ?     \
                                 NPY_FORTRAN : 0))

#define NpyArray_DATA(obj) ((void *)((obj)->data))
#define NpyArray_BYTES(obj) ((obj)->data)
#define NpyArray_DIMS(obj) ((obj)->dimensions)
#define NpyArray_STRIDES(obj) ((obj)->strides)
#define NpyArray_DIM(obj,n) (NpyArray_DIMS(obj)[n])
#define NpyArray_STRIDE(obj,n) (NpyArray_STRIDES(obj)[n])
#define NpyArray_DESCR(obj) ((obj)->descr)
#define NpyArray_FLAGS(obj) ((obj)->flags)
#define NpyArray_ITEMSIZE(obj) ((obj)->descr->elsize)
#define NpyArray_TYPE(obj) ((obj)->descr->type_num)

#define NpyArray_GETITEM(obj,itemptr)                           \
        (obj)->descr->f->getitem((char *)(itemptr),             \
                                 (PyArrayObject *)(obj))

#define NpyArray_SETITEM(obj,itemptr,v)                         \
        (obj)->descr->f->setitem((PyObject *)(v),               \
                                 (char *)(itemptr),             \
                                 (PyArrayObject *)(obj))


#define NpyArray_SIZE(m) NpyArray_MultiplyList(PyArray_DIMS(m), PyArray_NDIM(m))
#define NpyArray_NBYTES(m) (NpyArray_ITEMSIZE(m) * NpyArray_SIZE(m))

#define NpyArray_SAMESHAPE(a1,a2) ((NpyArray_NDIM(a1) == NpyArray_NDIM(a2)) && \
                                   NpyArray_CompareLists(NpyArray_DIMS(a1), \
                                                         NpyArray_DIMS(a2), \
                                                       NpyArray_NDIM(a1)))




#endif
