#ifndef _NPY_INCLUDE_NEIGHBORHOOD_IMP
#error You should not include this header directly
#endif

#ifndef _NEIGHBORHOOD_ITERATOR_IMP_H_
#define _NEIGHBORHOOD_ITERATOR_IMP_H_

#include "npy_iterators.h"

/*
 * Private API (here for inline)
 */
static NPY_INLINE int
_NpyArrayNeighborhoodIter_IncrCoord(NpyArrayNeighborhoodIterObject* iter);

/*
 * Update to next item of the iterator
 *
 * Note: this simply increment the coordinates vector, last dimension
 * incremented first , i.e, for dimension 3
 * ...
 * -1, -1, -1
 * -1, -1,  0
 * -1, -1,  1
 *  ....
 * -1,  0, -1
 * -1,  0,  0
 *  ....
 * 0,  -1, -1
 * 0,  -1,  0
 *  ....
 */
#define _UPDATE_COORD_ITER(c) \
    wb = iter->coordinates[c] < iter->bounds[c][1]; \
    if (wb) { \
        iter->coordinates[c] += 1; \
        return 0; \
    } \
    else { \
        iter->coordinates[c] = iter->bounds[c][0]; \
    }

static NPY_INLINE int
_NpyArrayNeighborhoodIter_IncrCoord(NpyArrayNeighborhoodIterObject* iter)
{
    npy_intp i, wb;

    for (i = iter->nd - 1; i >= 0; --i) {
        _UPDATE_COORD_ITER(i)
    }

    return 0;
}

/*
 * Version optimized for 2d arrays, manual loop unrolling
 */
static NPY_INLINE int
_NpyArrayNeighborhoodIter_IncrCoord2D(NpyArrayNeighborhoodIterObject* iter)
{
    npy_intp wb;

    _UPDATE_COORD_ITER(1)
    _UPDATE_COORD_ITER(0)

    return 0;
}
#undef _UPDATE_COORD_ITER

/*
 * Advance to the next neighbour
 */
static NPY_INLINE int
NpyArrayNeighborhoodIter_Next(NpyArrayNeighborhoodIterObject* iter)
{
    _NpyArrayNeighborhoodIter_IncrCoord (iter);
    iter->dataptr = iter->translate((NpyArrayIterObject*)iter, iter->coordinates);

    return 0;
}

/*
 * Reset functions
 */
static NPY_INLINE int
NpyArrayNeighborhoodIter_Reset(NpyArrayNeighborhoodIterObject* iter)
{
    npy_intp i;

    for (i = 0; i < iter->nd; ++i) {
        iter->coordinates[i] = iter->bounds[i][0];
    }
    iter->dataptr = iter->translate((NpyArrayIterObject*)iter, iter->coordinates);

    return 0;
}

#endif
