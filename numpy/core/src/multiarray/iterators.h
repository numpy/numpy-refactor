#ifndef _NPY_ARRAYITERATORS_H_
#define _NPY_ARRAYITERATORS_H_

NPY_NO_EXPORT intp
parse_subindex(PyObject *op, intp *step_size, intp *n_steps, intp max);

NPY_NO_EXPORT int
parse_index(PyArrayObject *self, PyObject *op,
            intp *dimensions, intp *strides, intp *offset_ptr,
            int* axismap, intp* starts);

NPY_NO_EXPORT PyObject
*iter_subscript(PyArrayIterObject *, PyObject *);

NPY_NO_EXPORT PyObject*
npy_iter_subscript(NpyArrayIterObject* self, PyObject* ind);

NPY_NO_EXPORT int
iter_ass_subscript(PyArrayIterObject *, PyObject *, PyObject *);

NPY_NO_EXPORT int
npy_iter_ass_subscript(NpyArrayIterObject* self, PyObject* ind, PyObject* val);

NPY_NO_EXPORT int
slice_GetIndices(PySliceObject *r, intp length,
                 intp *start, intp *stop, intp *step,
                 intp *slicelength);

NPY_NO_EXPORT int
NpyInterface_IterNewWrapper(NpyArrayIterObject *iter, void **interfaceRet);

NPY_NO_EXPORT int
NpyInterface_MultiIterNewWrapper(NpyArrayMultiIterObject *iter, void **interfaceRet);

NPY_NO_EXPORT int
NpyInterface_NeighborhoodIterNewWrapper(NpyArrayNeighborhoodIterObject *iter, void **interfaceRet);


#endif
