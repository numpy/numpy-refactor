/* npy_buffer.c */

#include <stdlib.h>
#include "npy_api.h"
#include "npy_dict.h"
#include "npy_buffer.h"
#include "npy_arrayobject.h"
#include "npy_internal.h"
#include "npy_os.h"


/* 
 * This module doesn't specifically implement the buffer protocol but provides supporting
 * functions for each interface that implements it.
 */


/* removed multiple segment interface */

NDARRAY_API size_t
npy_array_getsegcount(NpyArray *self, size_t *lenp)
{
    if (lenp) {
        *lenp = NpyArray_NBYTES(self);
    }
    if (NpyArray_ISONESEGMENT(self)) {
        return 1;
    }
    if (lenp) {
        *lenp = 0;
    }
    return 0;
}

NDARRAY_API size_t
npy_array_getreadbuf(NpyArray *self, size_t segment, void **ptrptr)
{
    if (segment != 0) {
        NpyErr_SetString(NpyExc_ValueError,
                         "accessing non-existing array segment");
        return -1;
    }
    if (NpyArray_ISONESEGMENT(self)) {
        *ptrptr = NpyArray_BYTES(self);
        return NpyArray_NBYTES(self);
    }
    NpyErr_SetString(NpyExc_ValueError, "array is not a single segment");
    *ptrptr = NULL;
    return -1;
}


NDARRAY_API size_t
npy_array_getwritebuf(NpyArray *self, size_t segment, void **ptrptr)
{
    if (NpyArray_CHKFLAGS(self, NPY_WRITEABLE)) {
        return npy_array_getreadbuf(self, segment, (void **) ptrptr);
    }
    else {
        NpyErr_SetString(NpyExc_ValueError, "array cannot be "
                         "accessed as a writeable buffer");
        return -1;
    }
}

NDARRAY_API size_t
npy_array_getcharbuf(NpyArray *self, size_t segment, char **ptrptr)
{
    return npy_array_getreadbuf(self, segment, (void **) ptrptr);
}


/*************************************************************************
 * PEP 3118 buffer protocol
 *
 * Implementing PEP 3118 is somewhat convoluted because of the desirata:
 *
 * - Don't add new members to ndarray or descr structs, to preserve binary
 *   compatibility. (Also, adding the items is actually not very useful,
 *   since mutability issues prevent an 1 to 1 relationship between arrays
 *   and buffer views.)
 *
 * - Don't use bf_releasebuffer, because it prevents PyArg_ParseTuple("s#", ...
 *   from working. Breaking this would cause several backward compatibility
 *   issues already on Python 2.6.
 *
 * - Behave correctly when array is reshaped in-place, or it's dtype is
 *   altered.
 *
 * The solution taken below is to manually track memory allocated for
 * Py_buffers.
 *************************************************************************/

/*
 * Format string translator
 *
 * Translate PyArray_Descr to a PEP 3118 format string.
 */

NDARRAY_API int
npy_append_char(npy_tmp_string_t *s, char c)
{
    char *p;
    if (s->s == NULL) {
        s->s = (char*)malloc(16);
        s->pos = 0;
        s->allocated = 16;
    }
    if (s->pos >= s->allocated) {
        p = (char*)realloc(s->s, 2*s->allocated);
        if (p == NULL) {
            NpyErr_SetString(NpyExc_MemoryError, "memory allocation failed");
            return -1;
        }
        s->s = p;
        s->allocated *= 2;
    }
    s->s[s->pos] = c;
    ++s->pos;
    return 0;
}

NDARRAY_API int
npy_append_str(npy_tmp_string_t *s, char *c)
{
    while (*c != '\0') {
        if (npy_append_char(s, *c)) return -1;
        ++c;
    }
    return 0;
}

/*
 * Return non-zero if a type is aligned in each item in the given array,
 * AND, the descr element size is a multiple of the alignment,
 * AND, the array data is positioned to alignment granularity.
 */
static int
_is_natively_aligned_at(NpyArray_Descr *descr,
                        NpyArray *arr, size_t offset)
{
    int k;
    
    if ((size_t)(NpyArray_BYTES(arr)) % descr->alignment != 0) {
        return 0;
    }
    
    if (offset % descr->alignment != 0) {
        return 0;
    }
    
    if (descr->elsize % descr->alignment) {
        return 0;
    }
    
    for (k = 0; k < NpyArray_NDIM(arr); ++k) {
        if (NpyArray_DIM(arr, k) > 1) {
            if (NpyArray_STRIDE(arr, k) % descr->alignment != 0) {
                return 0;
            }
        }
    }
    
    return 1;
}

NDARRAY_API int
npy_buffer_format_string(NpyArray_Descr *descr, npy_tmp_string_t *str,
                         NpyArray *arr, size_t *offset,
                         char *active_byteorder)
{
    int k;
    char _active_byteorder = '@';
    size_t _offset = 0;
    
    if (active_byteorder == NULL) {
        active_byteorder = &_active_byteorder;
    }
    if (offset == NULL) {
        offset = &_offset;
    }
    
    if (descr->subarray) {
        size_t total_count = 1;
        char buf[128];
        int old_offset;
        int ret;
        
        npy_append_char(str, '(');
        for (k = 0; k < descr->subarray->shape_num_dims; ++k) {
            if (k > 0) {
                npy_append_char(str, ',');
            }
            NpyOS_snprintf(buf, sizeof(buf), "%ld", (long)descr->subarray->shape_dims[k]);
            npy_append_str(str, buf);
            total_count *= descr->subarray->shape_dims[k];
        }
        npy_append_char(str, ')');
        old_offset = *offset;
        ret = npy_buffer_format_string(descr->subarray->base, str, arr, offset,
                                       active_byteorder);
        *offset = old_offset + (*offset - old_offset) * total_count;
        return ret;
    }
    else if (NpyDataType_HASFIELDS(descr)) {
        int n;
        
        npy_append_str(str, "T{");
        for (n=0; NULL != descr->names[n]; n++) ;
        for (k = 0; k < n; ++k) {
            const char *name;
            NpyArray_DescrField *item = NULL;
            NpyArray_Descr *child;
            const char *p;
            int new_offset;
            
            name = descr->names[k];
            item = (NpyArray_DescrField *)NpyDict_Get(descr->fields, name);
            
            child = item->descr;
            new_offset = item->offset;
            
            /* Insert padding manually */
            while (*offset < new_offset) {
                npy_append_char(str, 'x');
                ++*offset;
            }
            *offset += child->elsize;
            
            /* Insert child item */
            npy_buffer_format_string(child, str, arr, offset,
                                     active_byteorder);
            
            npy_append_char(str, ':');
            p = name;
            while (*p) {
                if (*p == ':') {
                    NpyErr_SetString(NpyExc_ValueError,
                                     "':' is not an allowed character in buffer "
                                     "field names");
                    return -1;
                }
                npy_append_char(str, *p);
                ++p;
            }
            npy_append_char(str, ':');
        }
        npy_append_char(str, '}');
    }
    else {
        int is_standard_size = 1;
        int is_native_only_type = (descr->type_num == NPY_LONGDOUBLE ||
                                   descr->type_num == NPY_CLONGDOUBLE);
#if NPY_SIZEOF_LONGLONG != 8
        is_native_only_type = is_native_only_type || (
                                                      descr->type_num == NPY_LONGLONG ||
                                                      descr->type_num == NPY_ULONGLONG);
#endif
        
        if (descr->byteorder == '=' &&
            _is_natively_aligned_at(descr, arr, *offset)) {
            /* Prefer native types, to cater for Cython */
            is_standard_size = 0;
            if (*active_byteorder != '@') {
                npy_append_char(str, '@');
                *active_byteorder = '@';
            }
        }
        else if (descr->byteorder == '=' && is_native_only_type) {
            /* Data types that have no standard size */
            is_standard_size = 0;
            if (*active_byteorder != '^') {
                npy_append_char(str, '^');
                *active_byteorder = '^';
            }
        }
        else if (descr->byteorder == '<' || descr->byteorder == '>' ||
                 descr->byteorder == '=') {
            is_standard_size = 1;
            if (*active_byteorder != descr->byteorder) {
                npy_append_char(str, descr->byteorder);
                *active_byteorder = descr->byteorder;
            }
            
            if (is_native_only_type) {
                /* It's not possible to express native-only data types
                 in non-native byte orders */
                char buffer[512];
                NpyOS_snprintf(buffer, sizeof(buffer),
                              "cannot expose native-only dtype '%c' in "
                              "non-native byte order '%c' via buffer interface",
                              descr->type, descr->byteorder);
                NpyErr_SetString(NpyExc_ValueError, buffer);
            }
        }
        
        switch (descr->type_num) {
            case NPY_BOOL:         if (npy_append_char(str, '?')) return -1; break;
            case NPY_BYTE:         if (npy_append_char(str, 'b')) return -1; break;
            case NPY_UBYTE:        if (npy_append_char(str, 'B')) return -1; break;
            case NPY_SHORT:        if (npy_append_char(str, 'h')) return -1; break;
            case NPY_USHORT:       if (npy_append_char(str, 'H')) return -1; break;
            case NPY_INT:          if (npy_append_char(str, 'i')) return -1; break;
            case NPY_UINT:         if (npy_append_char(str, 'I')) return -1; break;
            case NPY_LONG:
                if (is_standard_size && (NPY_SIZEOF_LONG == 8)) {
                    if (npy_append_char(str, 'q')) return -1;
                }
                else {
                    if (npy_append_char(str, 'l')) return -1;
                }
                break;
            case NPY_ULONG:
                if (is_standard_size && (NPY_SIZEOF_LONG == 8)) {
                    if (npy_append_char(str, 'Q')) return -1;
                }
                else {
                    if (npy_append_char(str, 'L')) return -1;
                }
                break;
            case NPY_LONGLONG:     if (npy_append_char(str, 'q')) return -1; break;
            case NPY_ULONGLONG:    if (npy_append_char(str, 'Q')) return -1; break;
            case NPY_FLOAT:        if (npy_append_char(str, 'f')) return -1; break;
            case NPY_DOUBLE:       if (npy_append_char(str, 'd')) return -1; break;
            case NPY_LONGDOUBLE:   if (npy_append_char(str, 'g')) return -1; break;
            case NPY_CFLOAT:       if (npy_append_str(str, "Zf")) return -1; break;
            case NPY_CDOUBLE:      if (npy_append_str(str, "Zd")) return -1; break;
            case NPY_CLONGDOUBLE:  if (npy_append_str(str, "Zg")) return -1; break;
                /* XXX: datetime */
                /* XXX: timedelta */
            case NPY_OBJECT:       if (npy_append_char(str, 'O')) return -1; break;
            case NPY_STRING: {
                char buf[128];
                NpyOS_snprintf(buf, sizeof(buf), "%ds", descr->elsize);
                if (npy_append_str(str, buf)) return -1;
                break;
            }
            case NPY_UNICODE: {
                /* Numpy Unicode is always 4-byte */
                char buf[128];
                assert(descr->elsize % 4 == 0);
                NpyOS_snprintf(buf, sizeof(buf), "%dw", descr->elsize / 4);
                if (npy_append_str(str, buf)) return -1;
                break;
            }
            case NPY_VOID: {
                /* Insert padding bytes */
                char buf[128];
                NpyOS_snprintf(buf, sizeof(buf), "%dx", descr->elsize);
                if (npy_append_str(str, buf)) return -1;
                break;
            }
            default: {
                char buffer[512];
                
                NpyOS_snprintf(buffer, sizeof(buffer),
                              "cannot include dtype '%c' in a buffer",
                              descr->type);
                NpyErr_SetString(NpyExc_ValueError, buffer);
                return -1;
            }
        }
    }
    
    return 0;
}


/*
 * Global information about all active buffers
 *
 * Note: because for backward compatibility we cannot define bf_releasebuffer,
 * we must manually keep track of the additional data required by the buffers.
 */



/* Fill in the info structure */
NDARRAY_API npy_buffer_info_t*
npy_buffer_info_new(NpyArray *arr)
{
    npy_buffer_info_t *info;
    npy_tmp_string_t fmt = {0,0,0};
    int k;
    
    info = (npy_buffer_info_t*)npy_malloc(sizeof(npy_buffer_info_t));
    
    /* Fill in format */
    if (npy_buffer_format_string(NpyArray_DESCR(arr), &fmt, arr, NULL, NULL) != 0) {
        npy_free(info);
        return NULL;
    }
    npy_append_char(&fmt, '\0');
    info->format = fmt.s;
    
    /* Fill in shape and strides */
    info->ndim = NpyArray_NDIM(arr);
    
    if (info->ndim == 0) {
        info->shape = NULL;
        info->strides = NULL;
    }
    else {
        info->shape = (size_t*)npy_malloc(sizeof(size_t)
                                          * NpyArray_NDIM(arr) * 2 + 1);
        info->strides = info->shape + NpyArray_NDIM(arr);
        for (k = 0; k < NpyArray_NDIM(arr); ++k) {
            info->shape[k] = NpyArray_DIM(arr, k);
            info->strides[k] = NpyArray_STRIDE(arr, k);
        }
    }
    
    return info;
}

/* Compare two info structures */
NDARRAY_API size_t
npy_buffer_info_cmp(npy_buffer_info_t *a, npy_buffer_info_t *b)
{
    size_t c;
    int k;
    
    c = strcmp(a->format, b->format);
    if (c != 0) return c;
    
    c = a->ndim - b->ndim;
    if (c != 0) return c;
    
    for (k = 0; k < a->ndim; ++k) {
        c = a->shape[k] - b->shape[k];
        if (c != 0) return c;
        c = a->strides[k] - b->strides[k];
        if (c != 0) return c;
    }
    
    return 0;
}

NDARRAY_API void
npy_buffer_info_free(npy_buffer_info_t *info)
{
    if (info->format) {
        npy_free(info->format);
    }
    if (info->shape) {
        npy_free(info->shape);
    }
    npy_free(info);
}

