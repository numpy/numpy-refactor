/* npy_buffer.h */

#if !defined(_NPY_BUFFER_H)
#define _NPY_BUFFER_H

#if defined(__cplusplus)
extern "C" {
#endif

/* Additional per-array data required for providing the buffer interface */
typedef struct {
    char *format;
    int ndim;
    size_t *strides;
    size_t *shape;
} npy_buffer_info_t;

/* Fast string 'class' */
typedef struct {
    char *s;
    int allocated;
    int pos;
} npy_tmp_string_t;



/* Internal support routines, intended for use by various interface layers. */
extern NDARRAY_API int
npy_buffer_format_string(NpyArray_Descr *descr, npy_tmp_string_t *str,
                         NpyArray *arr, size_t *offset,
                         char *active_byteorder);
extern NDARRAY_API int
npy_append_char(npy_tmp_string_t *s, char c);

extern NDARRAY_API size_t
npy_array_getsegcount(NpyArray *self, size_t *lenp);

extern NDARRAY_API size_t
npy_array_getreadbuf(NpyArray *self, size_t segment, void **ptrptr);

extern NDARRAY_API size_t
npy_array_getwritebuf(NpyArray *self, size_t segment, void **ptrptr);

extern NDARRAY_API size_t
npy_array_getcharbuf(NpyArray *self, size_t segment, char **ptrptr);

extern NDARRAY_API npy_buffer_info_t*
npy_buffer_info_new(NpyArray *arr);

extern NDARRAY_API size_t
npy_buffer_info_cmp(npy_buffer_info_t *a, npy_buffer_info_t *b);

extern NDARRAY_API void
npy_buffer_info_free(npy_buffer_info_t *info);

#if defined(__cplusplus)
}
#endif

#endif
