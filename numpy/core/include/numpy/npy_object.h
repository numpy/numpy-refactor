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
    _NpyTypeObject* nob_type;                   \
    void *nob_interface;

struct _NpyObject {
    NpyObject_HEAD;
};

#define Npy_INTERFACE(a) ((a)->nob_interface)


#define _Npy_INCREF(a)                                                \
       do {                                                           \
            if (0 == (a)->nob_refcnt && NULL != Npy_INTERFACE(a))     \
                NpyInterface_Incref(Npy_INTERFACE(a));                \
            (a)->nob_refcnt++;                                        \
       } while(0)                                                     \


#define _Npy_DECREF(a)                                          \
        if (--(a)->nob_refcnt == 0) {                           \
            if (NULL != Npy_INTERFACE(a))                       \
                NpyInterface_Decref(Npy_INTERFACE(a));          \
            else (a)->nob_type->ntp_dealloc((_NpyObject*)a);    \
        }


#define _Npy_XINCREF(a) if ((a) == NULL) ; else _Npy_INCREF(a)
#define _Npy_XDECREF(a) if ((a) == NULL) ; else _Npy_DECREF(a)


/* To be called by interface objects that are managing the lifetime of
   a core object. */
#define Npy_DEALLOC(a) (a)->nob_type->ntp_dealloc((_NpyObject*)a)

#define NpyObject_TypeCheck(a, t) ((a)->nob_type == t)

#define _NpyObject_Init(a, t)                   \
    do {                                        \
        (a)->nob_refcnt = 1;                    \
        (a)->nob_type = t;                      \
        (a)->nob_interface = NULL;              \
    } while (0)

#define _NpyObject_HEAD_INIT(t)                 1, t, NULL,

#define NpyObject_Wrapper(a) ((a)->interface)


#endif
