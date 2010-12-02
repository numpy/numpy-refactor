#ifndef _NPY_OBJECT_H_
#define _NPY_OBJECT_H_

#if defined(_WIN32)
#include <Windows.h>
#endif

#include <assert.h>
#include "npy_defs.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* Simple object model for numpy objects.
   This is similar to the Python object model. */

typedef struct _NpyObject _NpyObject;

typedef void (*npy_destructor)(_NpyObject *);
typedef int (*npy_wrapper_construct)(void *, void **);

typedef struct NpyTypeObject {
    npy_destructor ntp_dealloc;
    npy_wrapper_construct ntp_interface_alloc;
} NpyTypeObject;


#define NpyObject_HEAD                         \
    npy_uintp nob_refcnt;                      \
    NpyTypeObject* nob_type;                   \
    void *nob_interface;                       \
    int nob_magic_number;        /* Initialized to NPY_VALID_MAGIC initialization and NPY_INVALID_MAGIC on dealloc */

struct _NpyObject {
    NpyObject_HEAD
};
typedef struct _NpyObject NpyObject;

/* This defines the size of the NpyObject structure in bytes.  sizeof(struct _NpyObject) does not
   work because the structure may be padded, but if it's inserted at the head of another structure
   that padding will not be present. */
#define NpyObject_SIZE_OFFSET \
    ((npy_intp)(&((struct _NpyObject *)0)->nob_magic_number) + sizeof(((struct _NpyObject *)0)->nob_magic_number))


/* Returns the interface pointer for the object.  If the interface pointer is null and the interface allocator
   function is defined, the interface is created and that instance is returned.  This allows types such as
   iterators that typically don't need a wrapper to skip that step until needed. */
#define Npy_INTERFACE(a)                                                                     \
    (NULL != (a)->nob_interface ?                                                            \
       (a)->nob_interface :                                                                  \
       ((NULL != (a)->nob_type->ntp_interface_alloc) ?                                       \
          (a)->nob_type->ntp_interface_alloc((a), &(a)->nob_interface), (a)->nob_interface : \
          NULL))


/* These are platform-dependent macros implementing thread-safe
 * atomic increment/decrement behavior to make the reference counting
 * re-entrant. Both macros return the modified value. */
#if defined(_WIN32)
NDARRAY_API CRITICAL_SECTION Npy_RefCntLock;

/* NOTE: Do not use Npy_INTERFACE macro in these macros as it can trigger
   construction of a new interface object. */
#define Npy_INCREF(a)                                                      \
    do {                                                                   \
        EnterCriticalSection(&Npy_RefCntLock);                             \
        if (1 == ++(a)->nob_refcnt &&                                      \
                  NULL != (a)->nob_interface)                              \
           _NpyInterface_Incref((a)->nob_interface, &((a)->nob_interface));  \
        LeaveCriticalSection(&Npy_RefCntLock);                             \
    } while(0)

#define Npy_DECREF(a)                                                       \
    do {                                                                    \
        EnterCriticalSection(&Npy_RefCntLock);                              \
        assert((a)->nob_refcnt > 0);                                        \
        if (0 == --(a)->nob_refcnt) {                                       \
            if (NULL != (a)->nob_interface)                                 \
                _NpyInterface_Decref((a)->nob_interface, &((a)->nob_interface));  \
            else                                                            \
               (a)->nob_type->ntp_dealloc((_NpyObject*)a);                  \
        }                                                                   \
        LeaveCriticalSection(&Npy_RefCntLock);                              \
    } while(0);

#else
/* NOTE: Do not use Npy_INTERFACE macro in these macros as it can trigger
   construction of a new interface object. */
#define Npy_INCREF(a)                                                      \
    do {                                                                   \
        if (1 == ++(a)->nob_refcnt &&                                      \
                  NULL != (a)->nob_interface)                              \
           _NpyInterface_Incref((a)->nob_interface, &((a)->nob_interface));  \
    } while(0)

#define Npy_DECREF(a)                                                       \
    if (0 == --(a)->nob_refcnt) {                                           \
        if (NULL != (a)->nob_interface)                                     \
            _NpyInterface_Decref((a)->nob_interface, &((a)->nob_interface));  \
        else                                                                \
           (a)->nob_type->ntp_dealloc((_NpyObject*)a);                      \
    }

#endif




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
        (a)->nob_magic_number = NPY_VALID_MAGIC; \
    } while (0)

#define NpyObject_HEAD_INIT(t)                 1, t, NULL, NPY_VALID_MAGIC,

#define NpyObject_Wrapper(a) ((a)->interface)

#if defined(__cplusplus)
}
#endif

#endif
