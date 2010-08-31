#ifndef _NPY_OBJECT_H_
#define _NPY_OBJECT_H_

#if defined(_WIN32)
#include <Windows.h>
#endif

#include "npy_defs.h"

/* Simple object model for numpy objects.
   This is similar to the Python object model. */

typedef struct _NpyObject _NpyObject;

typedef void (*npy_destructor)(_NpyObject *);

typedef struct NpyTypeObject {
    npy_destructor ntp_dealloc;
} NpyTypeObject;

#define NpyObject_HEAD                          \
    npy_uintp nob_refcnt;                        \
    NpyTypeObject* nob_type;                   \
    void *nob_interface;

struct _NpyObject {
    NpyObject_HEAD
};
typedef struct _NpyObject NpyObject;

#define Npy_INTERFACE(a) ((a)->nob_interface)


/* These are platform-dependent macros implementing thread-safe
 * atomic increment/decrement behavior to make the reference counting
 * re-entrant. Both macros return the modified value. */
#if defined(_WIN32)
#define AtomicIncrement(i) InterlockedIncrement(&(i))
#define AtomicDecrement(i) InterlockedDecrement(&(i))

#else
/* NOT THREAD SAFE! */
#define AtomicIncrement(i) (++(i))
#define AtomicDecrement(i) (--(i))

#endif


#define Npy_INCREF(a)                                                 \
       do {                                                           \
            if (1 == AtomicIncrement((a)->nob_refcnt) && NULL != Npy_INTERFACE(a))   \
                _NpyInterface_Incref(Npy_INTERFACE(a), &((a)->nob_interface));  \
       } while(0)


#define Npy_DECREF(a)                                           \
        if (0 == AtomicDecrement((a)->nob_refcnt)) {            \
            if (NULL != Npy_INTERFACE(a))                       \
                _NpyInterface_Decref(Npy_INTERFACE(a), &((a)->nob_interface));      \
            else                                                \
                (a)->nob_type->ntp_dealloc((_NpyObject*)a);     \
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
