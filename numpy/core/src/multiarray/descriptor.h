#ifndef _NPY_ARRAYDESCR_H_
#define _NPY_ARRAYDESCR_H_

#include "npy_dict.h"
#include "npy_api.h"


NPY_NO_EXPORT PyObject *npy_arraydescr_protocol_typestr_get(NpyArray_Descr *);
NPY_NO_EXPORT PyObject *arraydescr_protocol_typestr_get(PyArray_Descr *);
NPY_NO_EXPORT PyObject *npy_arraydescr_protocol_descr_get(NpyArray_Descr *self);
NPY_NO_EXPORT PyObject *arraydescr_protocol_descr_get(PyArray_Descr *self);

NPY_NO_EXPORT PyObject *
array_set_typeDict(PyObject *NPY_UNUSED(ignored), PyObject *args);

NPY_NO_EXPORT PyArray_Descr *
_arraydescr_fromobj(PyObject *obj);

NPY_NO_EXPORT char **
arraydescr_seq_to_nameslist(PyObject *seq);

NPY_NO_EXPORT NpyDict *
arraydescr_fields_from_pydict(PyObject *dict);


#endif
