#ifndef _NPY_OBJECT_H_
#define _NPY_OBJECT_H_

#include "npy_defs.h"

/* Simple object model for numpy objects.
   This is similar to the Python object model. */

typedef struct _NpyObject _NpyObject;

typedef void (*npy_destructor)(_NpyObject *);

typedef struct NpyTypeObject {
    npy_destructor ntp_dealloc;
} NpyTypeObject;

#define NpyObject_HEAD                          \
    npy_intp nob_refcnt;                        \
    NpyTypeObject* nob_type;                   \
    void *nob_interface;

struct _NpyObject {
    NpyObject_HEAD;
};

#define Npy_INTERFACE(a) ((a)->nob_interface)


#define Npy_INCREF(a)                                                \
       do {                                                           \
            if (0 == (a)->nob_refcnt && NULL != Npy_INTERFACE(a))     \
                NpyInterface_INCREF(Npy_INTERFACE(a));                \
            (a)->nob_refcnt++;                                        \
       } while(0)                                                     \


#define Npy_DECREF(a)                                          \
        if (--(a)->nob_refcnt == 0) {                           \
            if (NULL != Npy_INTERFACE(a))                       \
                NpyInterface_DECREF(Npy_INTERFACE(a));          \
            else (a)->nob_type->ntp_dealloc((_NpyObject*)a);    \
        }


#define Npy_XINCREF(a) if ((a) == NULL) ; else Npy_INCREF(a)
#define Npy_XDECREF(a) if ((a) == NULL) ; else Npy_DECREF(a)


/* To be called by interface objects that are managing the lifetime of
   a core object. */
#define Npy_DEALLOC(a) (a)->nob_type->ntp_dealloc((_NpyObject*)a)

#define NpyObject_TypeCheck(a, t) ((a)->nob_type == t)

#define NpyObject_Init(a, t)                   \
    do {                                        \
        (a)->nob_refcnt = 1;                    \
        (a)->nob_type = t;                      \
        (a)->nob_interface = NULL;              \
    } while (0)

#define NpyObject_HEAD_INIT(t)                 1, t, NULL,

#define NpyObject_Wrapper(a) ((a)->interface)


#endif
