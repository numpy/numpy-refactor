#ifndef _NPY_ARRAYTYPES_H_
#define _NPY_ARRAYTYPES_H_

extern NpyArray_Descr npy_LONGLONG_Descr;
extern NpyArray_Descr npy_LONG_Descr;
extern NpyArray_Descr npy_INT_Descr;

NPY_NO_EXPORT int
set_typeinfo(PyObject *dict);

#endif
