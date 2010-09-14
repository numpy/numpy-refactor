
#include <assert.h>

extern "C" {
#include <npy_api.h>
#include <npy_defs.h>
#include <npy_arrayobject.h>
#include <npy_descriptor.h>
#include <npy_object.h>
#include <npy_dict.h>
}


///
/// This library provides a set of native access functions used by NumpyDotNet
/// for accessing the core library.
///


#define offsetof(type, member) ( (int) & ((type*)0) -> member )

extern "C" __declspec(dllexport)
void _cdecl NpyArrayAccess_Incref(NpyObject *obj)
{
    assert(NPY_VALID_MAGIC == obj->nob_magic_number);
    Npy_INCREF(obj);
}

extern "C" __declspec(dllexport)
void _cdecl NpyArrayAccess_Decref(NpyObject *obj)
{
    assert(NPY_VALID_MAGIC == obj->nob_magic_number);
    Npy_DECREF(obj);
}


extern "C" __declspec(dllexport)
void _cdecl NpyArrayAccess_ArrayGetOffsets(int *magic_number, int *descr, int *nd, 
                                           int *flags, int *data)
{
    NpyArray *ptr = NULL;

    *magic_number = offsetof(NpyArray, nob_magic_number);
    *descr = offsetof(NpyArray, descr);
    *nd = offsetof(NpyArray, nd);
    *flags = offsetof(NpyArray, flags);
    *data = offsetof(NpyArray, data);
}


extern "C" __declspec(dllexport)
void _cdecl NpyArrayAccess_DescrGetOffsets(int* magicNumOffset,
            int* kindOffset, int* typeOffset, int* byteorderOffset,
            int* flagsOffset, int* typenumOffset, int* elsizeOffset, 
            int* alignmentOffset, int* namesOffset, int* subarrayOffset)
{
    NpyArray *ptr = NULL;

    *magicNumOffset = offsetof(NpyArray_Descr, nob_magic_number);
    *kindOffset = offsetof(NpyArray_Descr, kind);
    *typeOffset = offsetof(NpyArray_Descr, type);
    *byteorderOffset = offsetof(NpyArray_Descr, byteorder);
    *flagsOffset = offsetof(NpyArray_Descr, flags);
    *typenumOffset = offsetof(NpyArray_Descr, type_num);
    *elsizeOffset = offsetof(NpyArray_Descr, elsize);
    *alignmentOffset = offsetof(NpyArray_Descr, alignment);
    *namesOffset = offsetof(NpyArray_Descr, names);
    *subarrayOffset = offsetof(NpyArray_Descr, subarray);
}


extern "C" __declspec(dllexport)
void _cdecl NpyArrayAccess_ArraySetDescr(void *arrTmp, void *newDescrTmp) 
{
    NpyArray *arr = (NpyArray *)arrTmp;
    NpyArray_Descr *newDescr = (NpyArray_Descr *)newDescrTmp;
    assert(NPY_VALID_MAGIC == arr->nob_magic_number);
    assert(NPY_VALID_MAGIC == newDescr->nob_magic_number);

    NpyArray_Descr *oldDescr = arr->descr;
    Npy_INCREF(newDescr);
    arr->descr = newDescr;
    Npy_DECREF(oldDescr);
}


// Returns the native byte order code for this platform and size of types
// that vary platform-to-playform.
extern "C" __declspec(dllexport)
    char _cdecl NpyArrayAccess_GetNativeTypeInfo(int *intSize, int *longSize, 
        int *longlongSize)
{
    *intSize = sizeof(npy_int);
    *longSize = sizeof(npy_long);
    *longlongSize = sizeof(npy_longlong);
    return NPY_NATBYTE;
}


// Fills in an int64 array with the dimensions of the array.
extern "C" __declspec(dllexport)
    bool _cdecl NpyArrayAccess_GetArrayDimsOrStrides(void *arrTmp, int ndims, bool dims, 
        npy_int64 *retPtr)
{
    NpyArray *arr = (NpyArray *)arrTmp;
    assert(NPY_VALID_MAGIC == arr->nob_magic_number);

    npy_intp *srcPtr = dims ? arr->dimensions : arr->strides;

    if (ndims != arr->nd) return false;
    if (sizeof(npy_int64) == sizeof(npy_intp)) {
        // Fast if sizes are the same.
        memcpy(retPtr, srcPtr, sizeof(npy_intp) * arr->nd);
    } else {
        // Slower, but converts between sizes.
        for (int i = 0; i < arr->nd; i++) { 
            retPtr[i] = srcPtr[i];
        }
    }
    return true;
}



// Trivial wrapper around NpyArray_Alloc.  The only reason for this is that .NET doesn't
// define an equivalent of npy_intp that changes for 32-bit or 64-bit systems.  So this
// converts for 32-bit systems.
extern "C" __declspec(dllexport)
    void * _cdecl NpyArrayAccess_AllocArray(void *descr, int numdims, npy_int64 *dimensions,
            bool fortran)
{
    npy_intp *dims = NULL;

    if (sizeof(npy_int64) != sizeof(npy_intp)) {
        // Dimensions uses a different type so we need to copy it.
        dims = (npy_intp *)malloc(sizeof(npy_intp) * numdims);
        for (int i=0; i < numdims; i++) {
            dims[i] = (npy_intp)dimensions[i];
        }
    } else {
        dims = (npy_intp *)dimensions;
    }
    
    NpyArray *arr = NpyArray_Alloc((NpyArray_Descr *)descr, numdims, dims, fortran, NULL);

    if (sizeof(npy_int64) != sizeof(npy_intp)) {
        free(dims);
    }
    return arr;
}


extern "C" __declspec(dllexport)
    npy_int64 _cdecl NpyArrayAccess_GetArrayStride(NpyArray *arr, int dim)
{
    return NpyArray_STRIDE(arr, dim);
}

extern "C" __declspec(dllexport)
	void _cdecl NpyArrayAccess_GetIndexInfo(int *unionOffset, int* indexSize, int* maxDims)
{
	*unionOffset = offsetof(NpyIndex, index);
	*indexSize = sizeof(NpyIndex);
	*maxDims = NPY_MAXDIMS;
}

extern "C" __declspec(dllexport)
	int _cdecl NpyArrayAccess_BindIndex(NpyArray* arr, NpyIndex* indexes, int n, NpyIndex* bound_indexes)
{
	return NpyArray_IndexBind(indexes, n, arr->dimensions, arr->nd, bound_indexes);
}

// Returns the offset for a field or -1 if there is no field that name.
extern "C" __declspec(dllexport)
	int _cdecl NpyArrayAccess_GetFieldOffset(NpyArray_Descr* descr, const char* fieldName, NpyArray_Descr** pDescr)
{
	if (descr->names == NULL) {
		return -1;
	}
	NpyArray_DescrField *value = (NpyArray_DescrField*) NpyDict_Get(descr->fields, fieldName);
	if (value == NULL) {
		return -1;
	}
	*pDescr = value->descr;
	return value->offset;
}

// Deallocates a numpy object.
extern "C" __declspec(dllexport)
	void _cdecl NpyArrayAccess_Dealloc(_NpyObject *obj)
{
	obj->nob_type->ntp_dealloc(obj);
}

// Gets the offsets for iterator fields.
extern "C" __declspec(dllexport)
	void _cdecl NpyArrayAccess_IterGetOffsets(int* off_size, int* off_index)
{
	*off_size = offsetof(NpyArrayIterObject, size);
	*off_index = offsetof(NpyArrayIterObject, index);
}

// Returns a pointer to the current data and advances the iterator.
// Returns NULL if the iterator is already past the end.
extern "C" __declspec(dllexport)
	void* _cdecl NpyArrayAccess_IterNext(NpyArrayIterObject* it)
{
	void* result = NULL;

	if (it->index < it->size) {
		result = it->dataptr;
		NpyArray_ITER_NEXT(it);
	}
	return result;
}

// Resets the iterator to the first element in the array.
extern "C" __declspec(dllexport)
	void _cdecl NpyArrayAccess_IterReset(NpyArrayIterObject* it)
{
	NpyArray_ITER_RESET(it);
}

// Returns the array for the iterator.
extern "C" __declspec(dllexport)
	NpyArray* _cdecl NpyArrayAccess_IterArray(NpyArrayIterObject* it)
{
	NpyArray* result = it->ao;
	Npy_INCREF(result);
	return result;
}

extern "C" __declspec(dllexport)
	npy_intp* _cdecl NpyArrayAccess_IterCoords(NpyArrayIterObject* self)
{
	if (self->contiguous) {
        /*
         * coordinates not kept track of ---
         * need to generate from index
         */
        npy_intp val;
		int nd = self->ao->nd;
        int i;
        val = self->index;
        for (i = 0; i < nd; i++) {
            if (self->factors[i] != 0) {
                self->coordinates[i] = val / self->factors[i];
                val = val % self->factors[i];
            } else {
                self->coordinates[i] = 0;
            }
        }
    }
	return self->coordinates;
}

// Moves the iterator to the location and returns a pointer to the data.
// Returns NULL if the index is invalid.
extern "C" __declspec(dllexport)
	void* _cdecl NpyArrayAccess_IterGoto1D(NpyArrayIterObject* it, npy_intp index)
{
	if (index < 0) {
		index += it->size;
	}
	if (index < 0 || index >= it->size) {
		char buf[1024];
		sprintf_s<sizeof(buf)>(buf, "index out of bounds 0<=index<%ld", (long)index);
		NpyErr_SetString(NpyExc_IndexError, buf);
		return NULL;
	}
	NpyArray_ITER_RESET(it);
	NpyArray_ITER_GOTO1D(it, index);
	return it->dataptr;
}