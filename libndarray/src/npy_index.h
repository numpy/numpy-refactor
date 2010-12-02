#ifndef _NPY_INDEX_H_
#define _NPY_INDEX_H_

#include "npy_defs.h"

#if defined(__cplusplus)
extern "C" {
#endif

/*
 * Structure for describing a slice without a stop.
 */
typedef struct NpyIndexSliceNoStop {
    npy_intp start;
    npy_intp step;
} NpyIndexSliceNoStop;

/*
 * Structure for describing a slice.
 */
typedef struct NpyIndexSlice {
    npy_intp start;
    npy_intp step;
    npy_intp stop;
} NpyIndexSlice;

/*
 * Enum for index types.
 */
typedef enum NpyIndexType {
    NPY_INDEX_INTP,
    NPY_INDEX_BOOL,
    NPY_INDEX_SLICE_NOSTOP,
    NPY_INDEX_SLICE,
    NPY_INDEX_STRING,
    NPY_INDEX_BOOL_ARRAY,
    NPY_INDEX_INTP_ARRAY,
    NPY_INDEX_ELLIPSIS,
    NPY_INDEX_NEWAXIS,
} NpyIndexType;

typedef struct NpyIndex {
    NpyIndexType type;
    union {
        npy_intp intp;
        npy_bool boolean;
        NpyIndexSlice slice;
        NpyIndexSliceNoStop slice_nostop;
        char *string;
        NpyArray *bool_array;
        NpyArray *intp_array;
    } index;
} NpyIndex;


NDARRAY_API void NpyArray_IndexDealloc(NpyIndex* indexes, int n);

NDARRAY_API int NpyArray_IndexExpandBool(NpyIndex *indexes, int n, NpyIndex *out_indexes);

NDARRAY_API int NpyArray_IndexBind(NpyIndex* indexes, int n,
                                   npy_intp *dimensions, int nd,
                                   NpyIndex* out_indexes);

NDARRAY_API int NpyArray_IndexToDimsEtc(NpyArray* array, NpyIndex* indexes, int n,
                                        npy_intp *dimensions, npy_intp* strides,
                                        npy_intp* offset_ptr, npy_bool allow_arrays);

NDARRAY_API npy_intp NpyArray_SliceSteps(NpyIndexSlice *slice);

#if defined(__cplusplus)
}
#endif

#endif
