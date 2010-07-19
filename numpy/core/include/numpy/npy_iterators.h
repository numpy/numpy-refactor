#ifndef _NPY_ITERATORS_H_
#define _NPY_ITERATORS_H_

#include "npy_object.h"

typedef struct NpyArrayIterObject NpyArrayIterObject;

/*
 * type of the function which translates a set of coordinates to a
 * pointer to the data
 */
typedef char* (*npy_iter_get_dataptr_t)(NpyArrayIterObject* iter, npy_intp*);

struct NpyArrayIterObject {
        NpyObject_HEAD
        int               magic_number;       /* Initialized to NPY_VALID_MAGIC initialization and NPY_INVALID_MAGIC on dealloc */
    
        int               nd_m1;            /* number of dimensions - 1 */
        npy_intp          index, size;
        npy_intp          coordinates[NPY_MAXDIMS];/* N-dimensional loop */
        npy_intp          dims_m1[NPY_MAXDIMS];    /* ao->dimensions - 1 */
        npy_intp          strides[NPY_MAXDIMS];    /* ao->strides or fake */
        npy_intp          backstrides[NPY_MAXDIMS];/* how far to jump back */
        npy_intp          factors[NPY_MAXDIMS];     /* shape factors */
        struct PyArrayObject          *ao;
        char              *dataptr;        /* pointer to current item*/
        npy_bool          contiguous;

        npy_intp          bounds[NPY_MAXDIMS][2];
        npy_intp          limits[NPY_MAXDIMS][2];
        npy_intp          limits_sizes[NPY_MAXDIMS];
        npy_iter_get_dataptr_t translate;
};

extern _NpyTypeObject NpyArrayIter_Type;


/* Iterator API */


NpyArrayIterObject *
NpyArray_IterNew(struct PyArrayObject *ao);

NpyArrayIterObject *
NpyArray_IterAllButAxis(struct PyArrayObject* obj, int *inaxis);

NpyArrayIterObject *
NpyArray_BroadcastToShape(struct PyArrayObject *ao, npy_intp *dims, int nd);

#define NpyArrayIter_Check(op) NpyObject_TypeCheck(op, &PyArrayIter_Type)

#define NpyArray_ITER_RESET(it) {                                        \
        assert( NPY_VALID_MAGIC == (it)->magic_number );                \
        (it)->index = 0;                                          \
        (it)->dataptr = (it)->ao->data;                     \
        memset((it)->coordinates, 0,                              \
               ((it)->nd_m1+1)*sizeof(npy_intp));                 \
}

#define _NpyArray_ITER_NEXT1(it) {                                       \
        assert( NPY_VALID_MAGIC == (it)->magic_number );                \
        (it)->dataptr += (it)->strides[0];                        \
        (it)->coordinates[0]++;                                         \
}

#define _NpyArray_ITER_NEXT2(it) {                                       \
        assert( NPY_VALID_MAGIC == (it)->magic_number );                \
        if ((it)->coordinates[1] < (it)->dims_m1[1]) {                  \
                (it)->coordinates[1]++;                                 \
                (it)->dataptr += (it)->strides[1];                      \
        }                                                               \
        else {                                                          \
                (it)->coordinates[1] = 0;                               \
                (it)->coordinates[0]++;                                 \
                (it)->dataptr += (it)->strides[0] -                     \
                        (it)->backstrides[1];                           \
        }                                                               \
}

#define _NpyArray_ITER_NEXT3(it) {                                       \
        assert( NPY_VALID_MAGIC == (it)->magic_number );                \
        if ((it)->coordinates[2] < (it)->dims_m1[2]) {                  \
                (it)->coordinates[2]++;                                 \
                (it)->dataptr += (it)->strides[2];                      \
        }                                                               \
        else {                                                          \
                (it)->coordinates[2] = 0;                               \
                (it)->dataptr -= (it)->backstrides[2];                  \
                if ((it)->coordinates[1] < (it)->dims_m1[1]) {          \
                        (it)->coordinates[1]++;                         \
                        (it)->dataptr += (it)->strides[1];              \
                }                                                       \
                else {                                                  \
                        (it)->coordinates[1] = 0;                       \
                        (it)->coordinates[0]++;                         \
                        (it)->dataptr += (it)->strides[0] -             \
                                (it)->backstrides[1];                   \
                }                                                       \
        }                                                               \
}

#define NpyArray_ITER_NEXT(it) {                                            \
        assert( NPY_VALID_MAGIC == (it)->magic_number );                \
        (it)->index++;                                               \
        if ((it)->nd_m1 == 0) {                                      \
                _NpyArray_ITER_NEXT1((it));                           \
        }                                                                  \
        else if ((it)->contiguous)                                   \
                (it)->dataptr += (it)->ao->descr->elsize;      \
        else if ((it)->nd_m1 == 1) {                                 \
                _NpyArray_ITER_NEXT2((it));                           \
        }                                                                  \
        else {                                                             \
                int __npy_i;                                               \
                for (__npy_i=(it)->nd_m1; __npy_i >= 0; __npy_i--) { \
                        if ((it)->coordinates[__npy_i] <             \
                            (it)->dims_m1[__npy_i]) {                \
                                (it)->coordinates[__npy_i]++;        \
                                (it)->dataptr +=                     \
                                        (it)->strides[__npy_i];      \
                                break;                                     \
                        }                                                  \
                        else {                                             \
                                (it)->coordinates[__npy_i] = 0;      \
                                (it)->dataptr -=                     \
                                        (it)->backstrides[__npy_i];  \
                        }                                                  \
                }                                                          \
        }                                                                  \
}

#define NpyArray_ITER_GOTO(it, destination) {                            \
        int __npy_i;                                                    \
        assert( NPY_VALID_MAGIC == (it)->magic_number );                \
        (it)->index = 0;                                          \
        (it)->dataptr = (it)->ao->data;                     \
        for (__npy_i = (it)->nd_m1; __npy_i>=0; __npy_i--) {      \
                if (destination[__npy_i] < 0) {                         \
                        destination[__npy_i] +=                         \
                                (it)->dims_m1[__npy_i]+1;         \
                }                                                       \
                (it)->dataptr += destination[__npy_i] *           \
                        (it)->strides[__npy_i];                   \
                (it)->coordinates[__npy_i] =                      \
                        destination[__npy_i];                           \
                (it)->index += destination[__npy_i] *             \
                        ( __npy_i==(it)->nd_m1 ? 1 :              \
                          (it)->dims_m1[__npy_i+1]+1) ;           \
        }                                                               \
}

#define NpyArray_ITER_GOTO1D(it, ind) {                                     \
        int __npy_i;                                                       \
        npy_intp __npy_ind = (npy_intp) (ind);                             \
        assert( NPY_VALID_MAGIC == (it)->magic_number );                \
        if (__npy_ind < 0) __npy_ind += (it)->size;                  \
        (it)->index = __npy_ind;                                     \
        if ((it)->nd_m1 == 0) {                                      \
                (it)->dataptr = (it)->ao->data +               \
                        __npy_ind * (it)->strides[0];                \
        }                                                                  \
        else if ((it)->contiguous)                                   \
                (it)->dataptr = (it)->ao->data +               \
                        __npy_ind * (it)->ao->descr->elsize;         \
        else {                                                             \
                (it)->dataptr = (it)->ao->data;                \
                for (__npy_i = 0; __npy_i<=(it)->nd_m1;              \
                     __npy_i++) {                                          \
                        (it)->dataptr +=                             \
                                (__npy_ind / (it)->factors[__npy_i]) \
                                * (it)->strides[__npy_i];            \
                        __npy_ind %= (it)->factors[__npy_i];         \
                }                                                          \
        }                                                                  \
}

#define NpyArray_ITER_DATA(it) ((void *)((it)->dataptr))

#define NpyArray_ITER_NOTDONE(it) ((it)->index < (it)->size)


/*
 * Any object passed to NpyArray_Broadcast must be binary compatible
 * with this structure.
 */
typedef struct NpyArrayMultiIterObject {
        NpyObject_HEAD
        /* DANGER - this must be in sync with MyUFuncLoopObject in ufuncobject.h */
        int                  magic_number;            /* Initialized to NPY_VALID_MAGIC initialization and NPY_INVALID_MAGIC on dealloc */

        int                  numiter;                 /* number of iters */
        npy_intp             size;                    /* broadcasted size */
        npy_intp             index;                   /* current index */
        int                  nd;                      /* number of dims */
        npy_intp             dimensions[NPY_MAXDIMS]; /* dimensions */
        NpyArrayIterObject    *iters[NPY_MAXARGS];     /* iterators */
} NpyArrayMultiIterObject;

extern _NpyTypeObject NpyArrayMultiIter_Type;

NpyArrayMultiIterObject *
NpyArray_MultiIterFromArrays(struct PyArrayObject **mps, int n, int nadd, ...);

NpyArrayMultiIterObject *
NpyArray_MultiIterNew(void);

int
NpyArray_RemoveSmallest(NpyArrayMultiIterObject *multi);
int
NpyArray_Broadcast(NpyArrayMultiIterObject *mit);


#define NpyArray_MultiIter_RESET(multi) {                               \
        int __npy_mi;                                                   \
        assert( NPY_VALID_MAGIC == (multi)->magic_number );             \
        (multi)->index = 0;                                             \
        for (__npy_mi=0; __npy_mi < (multi)->numiter;  __npy_mi++) {    \
                NpyArray_ITER_RESET((multi)->iters[__npy_mi]);          \
        }                                                               \
}

#define NpyArray_MultiIter_NEXT(multi) {                                \
        int __npy_mi;                                                   \
        assert( NPY_VALID_MAGIC == (multi)->magic_number );             \
        (multi)->index++;                                               \
        for (__npy_mi=0; __npy_mi < (multi)->numiter;   __npy_mi++) {   \
                NpyArray_ITER_NEXT((multi)->iters[__npy_mi]);           \
        }                                                               \
}

#define NpyArray_MultiIter_GOTO(multi, dest) {                               \
        int __npy_mi;                                                       \
        assert( NPY_VALID_MAGIC == (multi)->magic_number );             \
        for (__npy_mi=0; __npy_mi < (multi)->numiter; __npy_mi++) {   \
                NpyArray_ITER_GOTO((multi)->iters[__npy_mi], dest);    \
        }                                                                   \
        (multi)->index = (multi)->iters[0]->index;              \
}

#define NpyArray_MultiIter_GOTO1D(multi, ind) {                             \
        int __npy_mi;                                                      \
        assert( NPY_VALID_MAGIC == (multi)->magic_number );             \
        for (__npy_mi=0; __npy_mi < (multi)->numiter; __npy_mi++) {  \
                NpyArray_ITER_GOTO1D((multi)->iters[__npy_mi], ind);  \
        }                                                                  \
        (multi)->index = (multi)->iters[0]->index;             \
}

#define NpyArray_MultiIter_DATA(multi, i)                \
        ((void *)((multi)->iters[i]->dataptr))

#define NpyArray_MultiIter_NEXTi(multi, i)               \
        NpyArray_ITER_NEXT((multi)->iters[i])

#define NpyArray_MultiIter_NOTDONE(multi)                \
        ((multi)->index < (multi)->size)

/* Store the information needed for fancy-indexing over an array */

typedef struct {
        NpyObject_HEAD
        /*
         * Multi-iterator portion --- needs to be present in this
         * order to work with NpyArray_Broadcast
         */
        int                   magic_number;            /* Initialized to NPY_VALID_MAGIC initialization and NPY_INVALID_MAGIC on dealloc */

        int                   numiter;                 /* number of index-array
                                                          iterators */
        npy_intp              size;                    /* size of broadcasted
                                                          result */
        npy_intp              index;                   /* current index */
        int                   nd;                      /* number of dims */
        npy_intp              dimensions[NPY_MAXDIMS]; /* dimensions */
        NpyArrayIterObject     *iters[NPY_MAXDIMS];     /* index object
                                                          iterators */
        NpyArrayIterObject     *ait;                    /* flat Iterator for
                                                          underlying array */

        /* flat iterator for subspace (when numiter < nd) */
        NpyArrayIterObject     *subspace;

        /*
         * if subspace iteration, then this is the array of axes in
         * the underlying array represented by the index objects
         */
        int                   iteraxes[NPY_MAXDIMS];
        /*
         * if subspace iteration, the these are the coordinates to the
         * start of the subspace.
         */
        npy_intp              bscoord[NPY_MAXDIMS];

        PyObject            *indexobj;               /* creating obj */     /* TODO: Refactor me: indexobj of map iterator */
        int                   consec;
        char                  *dataptr;

} NpyArrayMapIterObject;

/*
 * TODO: We should have both PY and NPY level modes since the
 * NPY level doesn't support 0 and 1.
 */
enum {
    NPY_NEIGHBORHOOD_ITER_ZERO_PADDING,
    NPY_NEIGHBORHOOD_ITER_ONE_PADDING,
    NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING,
    NPY_NEIGHBORHOOD_ITER_CIRCULAR_PADDING,
    NPY_NEIGHBORHOOD_ITER_MIRROR_PADDING
};

typedef struct {
    NpyObject_HEAD
    int               magic_number;       /* Initialized to NPY_VALID_MAGIC initialization and NPY_INVALID_MAGIC on dealloc */
    
    /*
     * NpyArrayIterObject part: keep this in this exact order
     */
    int               nd_m1;            /* number of dimensions - 1 */
    npy_intp          index, size;
    npy_intp          coordinates[NPY_MAXDIMS];/* N-dimensional loop */
    npy_intp          dims_m1[NPY_MAXDIMS];    /* ao->dimensions - 1 */
    npy_intp          strides[NPY_MAXDIMS];    /* ao->strides or fake */
    npy_intp          backstrides[NPY_MAXDIMS];/* how far to jump back */
    npy_intp          factors[NPY_MAXDIMS];     /* shape factors */
    struct PyArrayObject          *ao;
    char              *dataptr;        /* pointer to current item*/
    npy_bool          contiguous;

    npy_intp          bounds[NPY_MAXDIMS][2];
    npy_intp          limits[NPY_MAXDIMS][2];
    npy_intp          limits_sizes[NPY_MAXDIMS];
    npy_iter_get_dataptr_t translate;

    /*
     * New members
     */
    npy_intp nd;

    /* Dimensions is the dimension of the array */
    npy_intp dimensions[NPY_MAXDIMS];

    /*
     * Neighborhood points coordinates are computed relatively to the
     * point pointed by _internal_iter
     */
    NpyArrayIterObject* _internal_iter;
    /*
     * To keep a reference to the representation of the constant value
     * for constant padding
     */
    char* constant;
    npy_free_func constant_free;

    int mode;
} NpyArrayNeighborhoodIterObject;


extern _NpyTypeObject NpyArrayNeighborhoodIter_Type;

/*
 * Neighborhood iterator API
 */

NpyArrayNeighborhoodIterObject*
NpyArray_NeighborhoodIterNew(NpyArrayIterObject *x, npy_intp *bounds,
                             int mode, void *fill,  npy_free_func fillfree);

/* General: those work for any mode */
static NPY_INLINE int
NpyArrayNeighborhoodIter_Reset(NpyArrayNeighborhoodIterObject* iter);
static NPY_INLINE int
NpyArrayNeighborhoodIter_Next(NpyArrayNeighborhoodIterObject* iter);
#if 0
static NPY_INLINE int
NpyArrayNeighborhoodIter_Next2D(NpyArrayNeighborhoodIterObject* iter);
#endif

/*
 * Include inline implementations - functions defined there are not
 * considered public API
 */
#define _NPY_INCLUDE_NEIGHBORHOOD_IMP
#include "_neighborhood_iterator_imp.h"
#undef _NPY_INCLUDE_NEIGHBORHOOD_IMP

#endif
