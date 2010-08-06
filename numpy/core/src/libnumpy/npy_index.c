
#include <numpy/npy_index.h>
#include <numpy/npy_object.h>
#include <numpy/npy_arrayobject.h>
#include <numpy/numpy_api.h>

void
NpyArray_IndexDealloc(NpyIndex*  indexes, int n)
{
    int i;
    NpyIndex *index = indexes;

    for (i=0; i<n; i++) {
        switch(index->type) {
        case NPY_INDEX_INTP_ARRAY:
            _Npy_DECREF(index->index.intp_array);
            break;
        case NPY_INDEX_BOOL_ARRAY:
            _Npy_DECREF(index->index.bool_array);
            break;
        default:
            break;
        }
        index++;
    }
}

/*
 * Returns the number of non-new indices.  Boolean arrays are
 * counted as if they are expanded.
 */
int count_nonnew(NpyIndex* indexes, int n)
{
    int i;
    int result = 0;

    for (i=0; i<n; i++) {
        switch (indexes[i].type) {
        case NPY_INDEX_NEWAXIS:
            break;
        case NPY_INDEX_BOOL_ARRAY:
            result += indexes[i].index.bool_array->nd;
            break;
        default:
            result++;
            break;
        }
    }
    return result;
}

/*
 * Converts indexes int out_indexes appropriate for an array by:
 *
 * 1. Expanding any ellipses.
 * 2. Setting slice start/stop/step appropriately for the array dims.
 * 3. Expanding any boolean arrays to intp arrays of non-zero indices.
 *
 * Returns the number of indices in out_indexes, or -1 on error.
 */
int NpyArray_IndexBind(NpyArray* array, NpyIndex* indexes,
                       int n, NpyIndex* out_indexes)
{
    int i, n2;
    int result = 0;

    for (i=0; i<n; i++) {
        if (result >= array->nd) {
            NpyErr_SetString(NpyExc_IndexError,
                             "too many indices");
            NpyArray_IndexDealloc(out_indexes, result);
            return -1;
        }

        switch (indexes[i].type) {
        case NPY_INDEX_ELLIPSIS:
            {
                /* Expand the ellipsis. */
                int j, n2;
                n2 = array->nd - count_nonnew(&indexes[i], n-i) - result;
                if (n2 < 0) {
                    NpyErr_SetString(NpyExc_IndexError,
                                     "too many indices");
                    NpyArray_IndexDealloc(out_indexes, result);
                    return -1;
                }
                /* Fill with full slices. */
                for (j=0; j<n2; i++) {
                    NpyIndex *out = &out_indexes[result];
                    out->type = NPY_INDEX_SLICE;
                    out->index.slice.start = 0;
                    out->index.slice.stop = NpyArray_DIM(array, result);
                    out->index.slice.has_stop = NPY_TRUE;
                    out->index.slice.step = 1;
                    result++;
                }
            }
            break;
        case NPY_INDEX_BOOL_ARRAY:
            {
                /* Convert to intp array on non-zero indexes. */
                NpyArray *index_arrays[NPY_MAXDIMS];
                NpyArray *bool_array = indexes[i].index.bool_array;
                int j;

                if (result + bool_array->nd >= array->nd) {
                    NpyErr_SetString(NpyExc_IndexError,
                                     "too many indices");
                    NpyArray_IndexDealloc(out_indexes, result);
                    return -1;
                }
                if (NpyArray_NonZero(bool_array, index_arrays, NULL) < 0) {
                    NpyArray_IndexDealloc(out_indexes, result);
                    return -1;
                }
                for (j=0; j<bool_array->nd; j++) {
                    out_indexes[result].type = NPY_INDEX_INTP_ARRAY;
                    out_indexes[result].index.intp_array = index_arrays[j];
                    result++;
                }
            }
            break;

        case NPY_INDEX_SLICE:
            {
                /* Sets the slice values based on the array. */
                npy_intp dim; 
                NpyIndexSlice *slice;

                dim = NpyArray_DIM(array, result);
                out_indexes[result].type = NPY_INDEX_SLICE;
                out_indexes[result].index.slice = indexes[i].index.slice;
                slice = &out_indexes[result].index.slice;

                if (slice->start < 0) {
                    slice->start += dim;
                }
                if (slice->start < 0) {
                    slice->start = 0;
                    if (slice->step < 0) {
                        slice->start -= 1;
                    }
                }
                if (slice->start >= dim) {
                    slice->start = dim;
                    if (slice->step < 0) {
                        slice->start -= 1;
                    }
                }

                if (!slice->has_stop) {
                    if (slice->step > 0) {
                        slice->stop = dim-1;
                    } else {
                        slice->stop = 0;
                    }
                    slice->has_stop = NPY_TRUE;
                }

                if (slice->stop < 0) {
                    slice->stop += dim;
                }

                if (slice->stop < 0) {
                    slice->stop = -1;
                }
                if (slice->stop > dim) {
                    slice->stop = dim;
                }

                result++;
            }
            break;
        default:
            /* Copy anything else. */
            out_indexes[result++] = indexes[i];
            break;
        }
    }

    return result;
}

