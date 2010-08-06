
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

