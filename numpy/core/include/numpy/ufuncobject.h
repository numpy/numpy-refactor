#ifndef _UFUNCOBJECT_H_
#define _UFUNCOBJECT_H_

#include "npy_ufunc_object.h"

#ifdef __cplusplus
extern "C" {
#endif

extern PyTypeObject PyUFunc_Type;
    
typedef struct {
    PyObject_HEAD
    int magic_number;

    struct NpyUFuncObject *ufunc;
    PyObject *func_obj;     /* Used for managing refcnt of function stored in core 'ptr' field */
} PyUFuncObject;

#define PyUFunc_Check(obj) (&PyUFunc_Type == (obj)->ob_type)
#define PyUFunc_UFUNC(obj) \
    (assert(NPY_VALID_MAGIC == (obj)->magic_number &&   \
    NPY_VALID_MAGIC == (obj)->ufunc->nob_magic_number), (obj)->ufunc)
#define PyUFunc_WRAP(obj) \
    (assert(NPY_VALID_MAGIC == (obj)->nob_magic_number), (PyUFuncObject *)Npy_INTERFACE(obj))
    
#include "arrayobject.h"


#if NPY_ALLOW_THREADS
#define NPY_LOOP_BEGIN_THREADS do {if (!(loop->obj & UFUNC_OBJ_NEEDS_API)) _save = PyEval_SaveThread();} while (0)
#define NPY_LOOP_END_THREADS   do {if (!(loop->obj & UFUNC_OBJ_NEEDS_API)) PyEval_RestoreThread(_save);} while (0)
#else
#define NPY_LOOP_BEGIN_THREADS
#define NPY_LOOP_END_THREADS
#endif


    
#include "__ufunc_api.h"

#define NPY_UFUNC_PYVALS_NAME "UFUNC_PYVALS"


  /* Make sure it gets defined if it isn't already */
#ifndef UFUNC_NOFPE
#define UFUNC_NOFPE
#endif


#ifdef __cplusplus
}
#endif
#endif /* _UFUNCOBJECT_H_ */
