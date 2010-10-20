/* npy_buffer.h */

#if !defined(_NPY_BUFFER_H)
#define _NPY_BUFFER_H

/* Additional per-array data required for providing the buffer interface */
typedef struct {
    char *format;
    int ndim;
    size_t *strides;
    size_t *shape;
} _buffer_info_t;



/* Internal support routines, intended for use by various interface layers. */
extern NDARRAY_API size_t
npy_array_getsegcount(NpyArray *self, size_t *lenp);

extern NDARRAY_API size_t
npy_array_getreadbuf(NpyArray *self, size_t segment, void **ptrptr);

extern NDARRAY_API size_t
npy_array_getwritebuf(NpyArray *self, size_t segment, void **ptrptr);

extern NDARRAY_API size_t
npy_array_getcharbuf(NpyArray *self, size_t segment, char **ptrptr);

extern NDARRAY_API _buffer_info_t*
npy_buffer_info_new(NpyArray *arr);

extern NDARRAY_API size_t
npy_buffer_info_cmp(_buffer_info_t *a, _buffer_info_t *b);

extern NDARRAY_API void
npy_buffer_info_free(_buffer_info_t *info);

#endif
