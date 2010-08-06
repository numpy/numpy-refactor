#ifndef _NPY_INDEX_H_
#define _NPY_INDEX_H_

#include <numpy/npy_defs.h>

/*
 * Structure for describing a slice.
 */
typedef struct NpyIndexSlice {
    npy_intp start;
    npy_intp stop;
    npy_intp step;
    npy_bool has_stop;
} NpyIndexSlice;

/*
 * Enum for index types.
 */
typedef enum NpyIndexType {
    NPY_INDEX_INTP,
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
        NpyIndexSlice slice;
        char *string;
        NpyArray *bool_array;
        NpyArray *intp_array;
    } index;
} NpyIndex;


void NpyArray_IndexDealloc(NpyIndex* indexes, int n);

int NpyArray_IndexBind(NpyArray* array, NpyIndex* indexes,
                       int n, NpyIndex* out_indexes);

#endif
