#ifndef _NPY_OBJECT_H_
#define _NPY_OBJECT_H_

#include "npy_defs.h"

/* Simple object model for numpy objects. 
   This is similar to the Python object model. */

typedef struct _NpyObject _NpyObject;

typedef void (*npy_destructor)(_NpyObject *);

typedef struct _NpyTypeObject {
    npy_destructor ntp_dealloc;
} _NpyTypeObject;

#define NpyObject_HEAD                          \
    npy_intp nob_refcnt;                        \
    _NpyTypeObject* nob_type;

struct _NpyObject {
    NpyObject_HEAD;
};

#define _Npy_INCREF(a)                          \
       (((_NpyObject*)(a))->nob_refcnt++)

#define _Npy_DECREF(a)                                          \
        if (--((_NpyObject*)(a))->nob_refcnt == 0)               \
            ((_NpyObject*)(a))->nob_type->ntp_dealloc((_NpyObject*)a)

#define _Npy_XINCREF(a) if ((a) == NULL) ; else _Npy_INCREF(a)
#define _Npy_XDECREF(a) if ((a) == NULL) ; else _Npy_DECREF(a)

#define NpyObject_TypeCheck(a, t) ((a)->nob_type == t)

#define _NpyObject_Init(a, t)                   \
    do {                                        \
        (a)->nob_type = t;                      \
        (a)->nob_refcnt = 1;                    \
    } while (0)


#define NpyObject_Wrapper(a) ((a)->interface)


#endif
