#ifndef _NPY_ARRAYOBJECT_H_
#define _NPY_ARRAYOBJECT_H_

#include "npy_object.h"
#include "npy_defs.h"

struct _NpyArray {
    NpyObject_HEAD
    int magic_number;       /* Initialized to NPY_VALID_MAGIC initialization and NPY_INVALID_MAGIC on dealloc */
    char *data;             /* pointer to raw data buffer */
    int nd;                 /* number of dimensions, also called ndim */
    npy_intp *dimensions;   /* size in each dimension */
    npy_intp *strides;      /*
                             * bytes to jump to get to the
                             * next element in each dimension
                             */
    struct _NpyArray *base_arr; /* Base when it's specifically an array object */
    void *base_obj;         /* Base when it's an opaque interface object */
    
    struct _PyArray_Descr *descr;   /* Pointer to type structure */
    int flags;              /* Flags describing array -- see below */
};

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
#define NpyArray_BASE_ARRAY(obj) ((obj)->base_arr)
#define NpyArray_BASE(obj) ((obj)->base_obj)

#define NpyArray_GETITEM(obj,itemptr)                           \
        (obj)->descr->f->getitem((char *)(itemptr),             \
                                 (PyArrayObject *)(obj))

#define NpyArray_SETITEM(obj,itemptr,v)                         \
        (obj)->descr->f->setitem((PyObject *)(v),               \
                                 (char *)(itemptr),             \
                                 (obj))


#define NpyArray_SIZE(m) NpyArray_MultiplyList(NpyArray_DIMS(m), NpyArray_NDIM(m))
#define NpyArray_NBYTES(m) (NpyArray_ITEMSIZE(m) * NpyArray_SIZE(m))

#define NpyArray_SAMESHAPE(a1,a2) ((NpyArray_NDIM(a1) == NpyArray_NDIM(a2)) && \
                                   NpyArray_CompareLists(NpyArray_DIMS(a1), \
                                                         NpyArray_DIMS(a2), \
                                                       NpyArray_NDIM(a1)))


#define NpyArray_ISBOOL(obj) NpyTypeNum_ISBOOL(NpyArray_TYPE(obj))
#define NpyArray_ISUNSIGNED(obj) NpyTypeNum_ISUNSIGNED(NpyArray_TYPE(obj))
#define NpyArray_ISSIGNED(obj) NpyTypeNum_ISSIGNED(NpyArray_TYPE(obj))
#define NpyArray_ISINTEGER(obj) NpyTypeNum_ISINTEGER(NpyArray_TYPE(obj))
#define NpyArray_ISFLOAT(obj) NpyTypeNum_ISFLOAT(NpyArray_TYPE(obj))
#define NpyArray_ISNUMBER(obj) NpyTypeNum_ISNUMBER(NpyArray_TYPE(obj))
#define NpyArray_ISSTRING(obj) NpyTypeNum_ISSTRING(NpyArray_TYPE(obj))
#define NpyArray_ISCOMPLEX(obj) NpyTypeNum_ISCOMPLEX(NpyArray_TYPE(obj))
#define NpyArray_ISPYTHON(obj) NpyTypeNum_ISPYTHON(NpyArray_TYPE(obj))
#define NpyArray_ISFLEXIBLE(obj) NpyTypeNum_ISFLEXIBLE(NpyArray_TYPE(obj))
#define NpyArray_ISDATETIME(obj) NpyTypeNum_ISDATETIME(NpyArray_TYPE(obj))
#define NpyArray_ISUSERDEF(obj) NpyTypeNum_ISUSERDEF(NpyArray_TYPE(obj))
#define NpyArray_ISEXTENDED(obj) NpyTypeNum_ISEXTENDED(NpyArray_TYPE(obj))
#define NpyArray_ISOBJECT(obj) NpyTypeNum_ISOBJECT(NpyArray_TYPE(obj))
#define NpyArray_HASFIELDS(obj) (NpyArray_DESCR(obj)->fields != NULL)


    /*
     * FIXME: This should check for a flag on the data-type that
     * states whether or not it is variable length.  Because the
     * ISFLEXIBLE check is hard-coded to the built-in data-types.
     */
#define NpyArray_ISVARIABLE(obj) NpyTypeNum_ISFLEXIBLE(NpyArray_TYPE(obj))

#define NpyArray_SAFEALIGNEDCOPY(obj) (NpyArray_ISALIGNED(obj) && !NpyArray_ISVARIABLE(obj))


#define NpyArray_ISNOTSWAPPED(m) NpyArray_ISNBO(NpyArray_DESCR(m)->byteorder)
#define NpyArray_ISBYTESWAPPED(m) (!NpyArray_ISNOTSWAPPED(m))

#define NpyArray_FLAGSWAP(m, flags) (NpyArray_CHKFLAGS(m, flags) &&       \
                                    NpyArray_ISNOTSWAPPED(m))

#define NpyArray_ISCARRAY(m) NpyArray_FLAGSWAP(m, NPY_CARRAY)
#define NpyArray_ISCARRAY_RO(m) NpyArray_FLAGSWAP(m, NPY_CARRAY_RO)
#define NpyArray_ISFARRAY(m) NpyArray_FLAGSWAP(m, NPY_FARRAY)
#define NpyArray_ISFARRAY_RO(m) NpyArray_FLAGSWAP(m, NPY_FARRAY_RO)
#define NpyArray_ISBEHAVED(m) NpyArray_FLAGSWAP(m, NPY_BEHAVED)
#define NpyArray_ISBEHAVED_RO(m) NpyArray_FLAGSWAP(m, NPY_ALIGNED)

#endif
