#ifndef _NPY_INCLUDE_NEIGHBORHOOD_IMP
#error You should not include this header directly
#endif
/*
 * Private API (here for inline)
 */
static NPY_INLINE int
_PyArrayNeighborhoodIter_IncrCoord(PyArrayNeighborhoodIterObject* iter);
static NPY_INLINE int
_PyArrayNeighborhoodIter_SetPtr(PyArrayNeighborhoodIterObject* iter);

/*
 * Inline implementations
 */
static NPY_INLINE int PyArrayNeighborhoodIter_Reset(PyArrayNeighborhoodIterObject* iter)
{
    int i;

    for (i = 0; i < iter->nd; ++i) {
        iter->coordinates[i] = iter->bounds[i][0];
    }
    _PyArrayNeighborhoodIter_SetPtr(iter);

    return 0;
}

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

static NPY_INLINE int _PyArrayNeighborhoodIter_IncrCoord(PyArrayNeighborhoodIterObject* iter)
{
    int i, wb;

    for (i = iter->nd - 1; i >= 0; --i) {
        _UPDATE_COORD_ITER(i)
    }

    return 0;
}

/*
 * Version optimized for 2d arrays, manual loop unrolling
 */
static NPY_INLINE int _PyArrayNeighborhoodIter_IncrCoord2D(PyArrayNeighborhoodIterObject* iter)
{
    int wb;

    _UPDATE_COORD_ITER(1)
    _UPDATE_COORD_ITER(0)

    return 0;
}
#undef _UPDATE_COORD_ITER

#define _INF_SET_PTR(c) \
    bd = iter->coordinates[c] + iter->_internal_iter->coordinates[c]; \
    if (bd < 0 || bd > iter->dimensions[c]) { \
        iter->dataptr = iter->constant; \
        return 1; \
    } \
    offset = iter->coordinates[c] * iter->strides[c]; \
    iter->dataptr += offset;

/* set the dataptr from its current coordinates */
static NPY_INLINE int _PyArrayNeighborhoodIter_SetPtr(PyArrayNeighborhoodIterObject* iter)
{
    int i;
    npy_intp offset, bd;

    iter->dataptr = iter->_internal_iter->dataptr;

    for(i = 0; i < iter->nd; ++i) {
        _INF_SET_PTR(i)
    }

    return 0;
}

static NPY_INLINE int _PyArrayNeighborhoodIter_SetPtr2D(PyArrayNeighborhoodIterObject* iter)
{
    npy_intp offset, bd;

    iter->dataptr = iter->_internal_iter->dataptr;

    _INF_SET_PTR(0)
    _INF_SET_PTR(1)

    return 0;
}
#undef _INF_SET_PTR

/*
 * Advance to the next neighbour
 */
static NPY_INLINE int PyArrayNeighborhoodIter_Next2D(PyArrayNeighborhoodIterObject* iter)
{
    _PyArrayNeighborhoodIter_IncrCoord2D(iter);
    _PyArrayNeighborhoodIter_SetPtr2D(iter);

    return 0;
}

static NPY_INLINE int PyArrayNeighborhoodIter_Next(PyArrayNeighborhoodIterObject* iter)
{
    _PyArrayNeighborhoodIter_IncrCoord (iter);
    _PyArrayNeighborhoodIter_SetPtr(iter);

    return 0;
}