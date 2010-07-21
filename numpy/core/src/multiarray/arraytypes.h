#ifndef _NPY_ARRAYTYPES_H_
#define _NPY_ARRAYTYPES_H_

#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT NpyArray_Descr LONGLONG_Descr;
extern NPY_NO_EXPORT NpyArray_Descr LONG_Descr;
extern NPY_NO_EXPORT NpyArray_Descr INT_Descr;
#endif

NPY_NO_EXPORT int
set_typeinfo(PyObject *dict);

#endif
