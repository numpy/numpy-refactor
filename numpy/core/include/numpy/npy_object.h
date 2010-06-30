#ifndef _NPY_OBJECT_H_
#define _NPY_OBJECT_H_

/* Simple object model for numpy objects. 
   This is similar to the Python object model. */

typedef void (*npy_destructor)(NpyObject *);

typedef struct NpyTypeObject {
    npy_destructor ntp_dealloc;
} NpyTypeObject;

#define NpyObject_HEAD                          \
    npy_intp nob_refcnt;                         \
    NpyTypeObject* nob_type;

typedef struct _NpyObject {
    NpyObject_HEAD;
} _NpyObject;

#define _Npy_INCREF(a)                          \
       (((NpyObject*)(a))->nob_refcnt++)

#define _Npy_DECREF(a)                                          \
        if (--((NpyObject*)(a))->nob_refcnt == 0)               \
            ((NpyObject*)(a))->ntp_dealloc((NpyObject*)a)

#define _Npy_XINCREF(a) if ((a) == NULL) ; else _Npy_INCREF(a)
#define _Npy_XDECREF(a) if ((a) == NULL) ; else _Npy_DECREF(a)


#endif
