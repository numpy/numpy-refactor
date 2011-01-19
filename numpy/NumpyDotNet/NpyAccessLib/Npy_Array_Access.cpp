#include <assert.h>

extern "C" {
#include <npy_api.h>
#include <npy_defs.h>
#include <npy_os.h>
#include <npy_buffer.h>
#include <npy_arrayobject.h>
#include <npy_descriptor.h>
#include <npy_object.h>
#include <npy_dict.h>
#include <npy_ufunc_object.h>
#include <npy_iterators.h>
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


// This function is here because the Npy_INTERFACE macro does some
// magic with creating interface objects on an as-needed basis so it's
// more code than simply reading the nob_interface field.
extern "C" __declspec(dllexport)
void * _cdecl NpyArrayAccess_ToInterface(NpyObject *obj)
{
    assert(NPY_VALID_MAGIC == obj->nob_magic_number);
    return Npy_INTERFACE(obj);
}


extern "C" __declspec(dllexport)
void _cdecl NpyArrayAccess_ArrayGetOffsets(int *magic_number, int *descr, int *nd, 
                                           int *dimensions, int *strides,
                                           int *flags, int *data, int* base_obj, 
                                           int* base_array)
{
    NpyArray *ptr = NULL;

    *magic_number = offsetof(NpyArray, nob_magic_number);
    *descr = offsetof(NpyArray, descr);
    *nd = offsetof(NpyArray, nd);
    *dimensions = offsetof(NpyArray, dimensions);
    *strides = offsetof(NpyArray, strides);
    *flags = offsetof(NpyArray, flags);
    *data = offsetof(NpyArray, data);
    *base_obj = offsetof(NpyArray, base_obj);
    *base_array = offsetof(NpyArray, base_arr);
}


extern "C" __declspec(dllexport)
void _cdecl NpyArrayAccess_DescrGetOffsets(int* magicNumOffset,
            int* kindOffset, int* typeOffset, int* byteorderOffset,
            int* flagsOffset, int* typenumOffset, int* elsizeOffset, 
            int* alignmentOffset, int* namesOffset, int* subarrayOffset,
            int* fieldsOffset, int* dtinfoOffset, int* fieldsOffsetOffset, 
            int *fieldsDescrOffset,
            int* fieldsTitleOffset)
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
    *fieldsOffset = offsetof(NpyArray_Descr, fields);
    *dtinfoOffset = offsetof(NpyArray_Descr, dtinfo);
    *fieldsOffsetOffset = offsetof(NpyArray_DescrField, offset);
    *fieldsDescrOffset = offsetof(NpyArray_DescrField, descr);
    *fieldsTitleOffset = offsetof(NpyArray_DescrField, title);
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


extern "C" __declspec(dllexport)
    float _cdecl NpyArrayAccess_GetAbiVersion()
{
    return NPY_ABI_VERSION;
}


// Returns the native byte order code for this platform and size of types
// that vary platform-to-playform.
extern "C" __declspec(dllexport)
    char _cdecl NpyArrayAccess_GetNativeTypeInfo(int *intSize, int *longSize, 
        int *longlongSize, int* longDoubleSize)
{
    *intSize = sizeof(npy_int);
    *longSize = sizeof(npy_long);
    *longlongSize = sizeof(npy_longlong);
    *longDoubleSize = sizeof(npy_longdouble);
    return NPY_OPPBYTE;
}


// Fills in an int64 array with the dimensions of the array.
extern "C" __declspec(dllexport)
    bool _cdecl NpyArrayAccess_GetIntpArray(npy_intp *srcPtr, int len, npy_int64 *retPtr)
{
    if (sizeof(npy_int64) == sizeof(npy_intp)) {
        // Fast if sizes are the same.
        memcpy(retPtr, srcPtr, sizeof(npy_intp) * len);
    } else {
        // Slower, but converts between sizes.
        for (int i = 0; i < len; i++) { 
            retPtr[i] = srcPtr[i];
        }
    }
    return true;
}

extern "C" __declspec(dllexport)
    void _cdecl NpyArrayAccess_ZeroFill(NpyArray* arr, npy_intp offset) 
{
    int itemsize = NpyArray_ITEMSIZE(arr);
    npy_intp size = NpyArray_SIZE(arr)*itemsize;
    npy_intp off = offset * itemsize;
    npy_intp fill_size = size-off;
    char* p = arr->data + off;
    memset(p, 0, fill_size);
}




// Trivial wrapper around NpyArray_Alloc.  The only reason for this is that .NET doesn't
// define an equivalent of npy_intp that changes for 32-bit or 64-bit systems.  So this
// converts for 32-bit systems.
extern "C" __declspec(dllexport)
    void * _cdecl NpyArrayAccess_AllocArray(void *descr, int numdims, npy_int64 *dimensions,
            bool fortran)
{
    npy_intp *dims = NULL;
    npy_intp dimMem[NPY_MAXDIMS];

    if (sizeof(npy_int64) != sizeof(npy_intp)) {
        // Dimensions uses a different type so we need to copy it.
        for (int i=0; i < numdims; i++) {
            dimMem[i] = (npy_intp)dimensions[i];
        }
        dims = dimMem;
    } else {
        dims = (npy_intp *)dimensions;
    }
    NpyArray* result = NpyArray_Alloc((NpyArray_Descr *)descr, numdims, dims, fortran, NULL);
    return result;
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

extern "C" __declspec(dllexport)
    int _cdecl NpyArrayAccess_GetDescrField(NpyArray_Descr* descr, const char* fieldName, NpyArray_DescrField* pField)
{
    if (descr->names == NULL) {
        return -1;
    }
    NpyArray_DescrField *value = (NpyArray_DescrField*) NpyDict_Get(descr->fields, fieldName);
    if (value == NULL) {
        return -1;
    }
    *pField = *value;
    return 0;
}

// Deallocates a numpy object.
extern "C" __declspec(dllexport)
	void _cdecl NpyArrayAccess_Dealloc(_NpyObject *obj)
{
    assert(NPY_VALID_MAGIC == obj->nob_magic_number);
    assert(0 == obj->nob_refcnt);

    // Clear the interface pointer because some deallocators temporarily increment the
    // reference count on the object for some reason.  This causes callbacks into the
    // managed world that we do not want (and is added overhead).
    obj->nob_interface = NULL;

    // Calls the type-specific deallocation route.  This routine releases any references
    // held by the object itself prior to actually freeing the memory. 
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
		sprintf_s<sizeof(buf)>(buf, "index out of bounds 0<=index<%ld", (long)it->size);
		NpyErr_SetString(NpyExc_IndexError, buf);
		return NULL;
	}
	NpyArray_ITER_RESET(it);
	NpyArray_ITER_GOTO1D(it, index);
	return it->dataptr;
}

// A simple translation routine that handles resizing the long[] types to either
// int or long depending on the sizes of the C types.
extern "C" __declspec(dllexport)
    void * _cdecl NpyArrayAccess_NewFromDescrThunk(NpyArray_Descr *descr, int nd,
    int flags, npy_int64 *dimsLong, npy_int64 *stridesLong, void *data, void *interfaceData)
{
    npy_intp *dims = NULL;
    npy_intp *strides = NULL;
    npy_intp dimMem[NPY_MAXDIMS], strideMem[NPY_MAXDIMS];

    assert(NPY_VALID_MAGIC == descr->nob_magic_number);

    if (sizeof(npy_int64) != sizeof(npy_intp)) {
        // Dimensions uses a different type so we need to copy it.
        for (int i=0; i < nd; i++) {
            dimMem[i] = (npy_intp)(dimsLong[i]);
            if (NULL != stridesLong) strideMem[i] = (npy_intp)stridesLong[i];
        }
        dims = dimMem;
        strides = (NULL != stridesLong) ? strideMem : NULL;
    } else {
        dims = (npy_intp *)dimsLong;
        strides = (npy_intp *)stridesLong;
    }
    return NpyArray_NewFromDescr(descr, nd, dims, strides, data, flags, 
        NPY_FALSE, NULL, interfaceData);
}

extern "C" __declspec(dllexport)
    NpyArrayMultiIterObject* NpyArrayAccess_MultiIterFromArrays(NpyArray** arrays, int n)
{
    return NpyArray_MultiIterFromArrays(arrays, n, 0);
}

extern "C" __declspec(dllexport)
    void NpyArrayAccess_MultiIterGetOffsets(int* off_numiter, int* off_size, int* off_index, int* off_nd, int* off_dimensions, int* off_iters)
{
    *off_numiter = offsetof(NpyArrayMultiIterObject, numiter);
    *off_size = offsetof(NpyArrayMultiIterObject, size);
    *off_index = offsetof(NpyArrayMultiIterObject, index);
    *off_nd = offsetof(NpyArrayMultiIterObject, nd);
    *off_dimensions = offsetof(NpyArrayMultiIterObject, dimensions);
    *off_iters = offsetof(NpyArrayMultiIterObject, iters);
}


// Sets the dimensions and strides of the array and reallocates/resizes the array memory
// in preparation for being loaded with new data.  Typically used after unpickling.
// If srcPtr is set, it points to 16-bit unicode memory of which only the low-order 8-bits
// in each wide char is a byte of data in the array.
extern "C" __declspec(dllexport)
    void NpyArrayAccess_SetState(NpyArray *self, int ndim, npy_intp *dims, NPY_ORDER order, const wchar_t *srcPtr, int srcLen)
{
    assert(NPY_VALID_MAGIC == self->nob_magic_number);
    assert(NULL != dims);
    assert(0 <= ndim);

    // Clear existing data and references.  Typically these will be empty.
    if (NpyArray_FLAGS(self) & NPY_OWNDATA) {
        if (NULL != NpyArray_BYTES(self)) {
            NpyArray_free(NpyArray_BYTES(self));
        }
        NpyArray_FLAGS(self) &= ~NPY_OWNDATA;
    }
    Npy_XDECREF(NpyArray_BASE_ARRAY(self));
    NpyInterface_DECREF(NpyArray_BASE(self));
    NpyArray_BASE_ARRAY(self) = NULL;
    NpyArray_BASE(self) = NULL;

    if (NULL != NpyArray_DIMS(self)) {
        NpyDimMem_FREE(NpyArray_DIMS(self));
        NpyArray_DIMS(self) = NULL;
    }

    self->flags = NPY_DEFAULT;
    self->nd = ndim;
    if (0 < ndim) {
        NpyArray_DIMS(self) = NpyDimMem_NEW(ndim*2);
        NpyArray_STRIDES(self) = NpyArray_DIMS(self) + ndim;
        memcpy(NpyArray_DIMS(self), dims, sizeof(npy_int64)*ndim);
        npy_array_fill_strides(NpyArray_STRIDES(self), dims, ndim,
            NpyArray_ITEMSIZE(self), order, &(NpyArray_FLAGS(self)));
    }

    npy_intp bytes = NpyArray_ITEMSIZE(self) * NpyArray_SIZE(self);
    NpyArray_BYTES(self) = (char *)NpyArray_malloc(bytes);
    NpyArray_FLAGS(self) |= NPY_OWNDATA;

    if (NULL != srcPtr) {
        // This is unpleasantly inefficent.  The input is a .NET string, which is 16-bit
        // unicode. Thus the data is encoded into alternating bytes so we can't use memcpy.
        char *destPtr = NpyArray_BYTES(self);
        char *destEnd = destPtr + bytes;
        const wchar_t *srcEnd = srcPtr + srcLen;
        while (destPtr < destEnd && srcPtr < srcEnd) *(destPtr++) = (char)*(srcPtr++);
    } else {
        memset(NpyArray_BYTES(self), 0, bytes);
    }
}


extern "C" __declspec(dllexport)
    NpyArray* NpyArrayAccess_Newshape(NpyArray* self, int ndim, npy_intp* dims, NPY_ORDER order)
{
    NpyArray_Dims newdims;
    newdims.len = ndim;
    newdims.ptr = dims;
    return NpyArray_Newshape(self, &newdims, order);
}

extern "C" __declspec(dllexport)
    int NpyArrayAccess_SetShape(NpyArray* self, int ndim, npy_intp* dims)
{
    NpyArray_Dims newdims;
    newdims.len = ndim;
    newdims.ptr = dims;
    return NpyArray_SetShape(self, &newdims);
}

extern "C" __declspec(dllexport)
    int NpyArrayAccess_Resize(NpyArray *self, int ndim, npy_intp* newshape, int refcheck, NPY_ORDER fortran)
{
    NpyArray_Dims newdims;
    newdims.len = ndim;
    newdims.ptr = newshape;
    return NpyArray_Resize(self, &newdims, refcheck, fortran);
}

extern "C" __declspec(dllexport)
    NpyArray* NpyArrayAccess_Transpose(NpyArray *self, int ndim, npy_intp* permute)
{
    if (ndim == 0) {
        return NpyArray_Transpose(self, NULL);
    } else {
        NpyArray_Dims p;
        p.len = ndim;
        p.ptr = permute;
        return NpyArray_Transpose(self, &p);
    }
}

extern "C" __declspec(dllexport)
    void NpyArrayAccess_ClearUPDATEIFCOPY(NpyArray *self) 
{
        if (self->flags | NPY_UPDATEIFCOPY) {
            if (self->base_arr != NULL) {
                self->base_arr->flags &= ~NPY_WRITEABLE;
                Npy_DECREF(self->base_arr);
                self->base_arr = NULL;
            }
            self->flags &= ~NPY_UPDATEIFCOPY;
        }
}

extern "C" __declspec(dllexport)
    void NpyArrayAccess_DescrDestroyNames(char** names, int n) 
{
    for (int i=0; i<n; i++) {
        if (names[i]) {
            free(names[i]);
        }
    }
    npy_free(names);
}

extern "C" __declspec(dllexport)
    int NpyArrayAccess_AddField(NpyDict* fields, char** names, int i, 
    char* name, NpyArray_Descr* fieldType, int offset, char* title)
{
    if (NpyDict_ContainsKey(fields, name)) {
        NpyErr_SetString(NpyExc_ValueError, "two fields with the same name");
        return -1;
    }
    names[i] = _strdup(name);
    NpyArray_DescrSetField(fields, names[i], fieldType, offset, title);
    return 0;
}

extern "C" __declspec(dllexport)
    NpyArray_Descr* NpyArrayAccess_DescrNewVoid(NpyDict* fields, char **names, int elsize, int flags, int alignment)
{
    NpyArray_Descr* result = NpyArray_DescrNewFromType(NPY_VOID);
    result->fields = fields;
    result->names = names;
    result->elsize = elsize;
    result->flags = flags;
    result->alignment = alignment;
    return result;
}


// Replaces/sets the subarray field of an existing descriptor object.
extern "C" __declspec(dllexport)
    void NpyArrayAccess_DescrReplaceSubarray(NpyArray_Descr* descr, NpyArray_Descr *baseDescr, int ndim, npy_intp* dims)
{
    assert(NULL != descr && NPY_VALID_MAGIC == descr->nob_magic_number);
    assert(NULL != baseDescr && NPY_VALID_MAGIC == baseDescr->nob_magic_number);

    if (NULL != descr->subarray) {
        NpyArray_DestroySubarray(descr->subarray);
    }

    descr->subarray = (NpyArray_ArrayDescr*)NpyArray_malloc(sizeof(NpyArray_ArrayDescr));
    descr->subarray->base = baseDescr;
    Npy_INCREF(baseDescr);
    descr->subarray->shape_num_dims = ndim;
    descr->subarray->shape_dims = (npy_intp*) NpyArray_malloc(ndim*sizeof(npy_intp));
    memcpy(descr->subarray->shape_dims, dims, ndim*sizeof(npy_intp));
}


// Allocates a new VOID descriptor and sets the subarray data.
extern "C" __declspec(dllexport)
    NpyArray_Descr* NpyArrayAccess_DescrNewSubarray(NpyArray_Descr* baseDescr, int ndim, npy_intp* dims)
{
    NpyArray_Descr* result = NpyArray_DescrNewFromType(NPY_VOID);
    if (result == NULL) {
        return result;
    }
    result->elsize = baseDescr->elsize;
    result->elsize *= NpyArray_MultiplyList(dims, ndim);
    NpyArrayAccess_DescrReplaceSubarray(result, baseDescr, ndim, dims);
    result->flags = baseDescr->flags;
    /* I'm not sure why or if the next call is needed, but it is in the CPython code. */
    NpyArray_DescrDeallocNamesAndFields(result);

    return result;
}


extern "C" __declspec(dllexport)
    void NpyArrayAccess_DescrReplaceFields(NpyArray_Descr *descr, char **nameslist, NpyDict *fields)
{
    assert(NULL != descr && NPY_VALID_MAGIC == descr->nob_magic_number);

    if (NULL != descr->names) {
        NpyArray_DescrDeallocNamesAndFields(descr);
    }
    descr->names = nameslist;
    descr->fields = fields;
}

//
// UFunc access methods
//

extern "C" __declspec(dllexport)
void _cdecl NpyArrayAccess_UFuncGetOffsets(int *ninOffset, int *noutOffset,
    int *nargsOffset, int *coreEnabledOffset, int *identityOffset, int *ntypesOffset, 
    int *checkRetOffset, int *nameOffset, int *typesOffset, int *coreSigOffset)
{
  	*ninOffset = offsetof(NpyUFuncObject, nin);
  	*noutOffset = offsetof(NpyUFuncObject, nout);
    *nargsOffset = offsetof(NpyUFuncObject, nargs);
    *coreEnabledOffset = offsetof(NpyUFuncObject, core_enabled);
  	*identityOffset = offsetof(NpyUFuncObject, identity);
  	*ntypesOffset = offsetof(NpyUFuncObject, ntypes);
  	*checkRetOffset = offsetof(NpyUFuncObject, check_return);
  	*nameOffset = offsetof(NpyUFuncObject, name);
  	*typesOffset = offsetof(NpyUFuncObject, types);
  	*coreSigOffset = offsetof(NpyUFuncObject, core_signature);
}

extern "C" __declspec(dllexport)
    int _cdecl NpyArrayAccess_GetBytes(NpyArray* arr, char* buffer, npy_int64 length, NPY_ORDER order)
{
    npy_intp numbytes;


    if (order == NPY_ANYORDER) {
        order = NpyArray_ISFORTRAN(arr) ? NPY_FORTRANORDER : NPY_CORDER;
    }

    numbytes = NpyArray_NBYTES(arr);
    if (length != (npy_int64)numbytes) {
        NpyErr_SetString(NpyExc_ValueError, "length is not the size of the array.");
        return -1;
    }

    if (NpyArray_ISCONTIGUOUS(arr) && order == NPY_CORDER ||
        NpyArray_ISFORTRAN(arr) && order == NPY_FORTRANORDER) {
            memcpy(buffer, arr->data, numbytes);
    } else {
        NpyArray *src;
        NpyArrayIterObject *it;
        npy_intp index;
        int elsize;

        if (order == NPY_FORTRANORDER) {
            src = NpyArray_Transpose(arr, NULL);
        } else {
            src = arr;
            Npy_INCREF(src);
        }

        it = NpyArray_IterNew(src);
        Npy_DECREF(src);
        if (it == NULL) {
            return -1;
        }
        index = it->size;
        elsize = NpyArray_ITEMSIZE(arr);
        while (index--) {
            memcpy(buffer, it->dataptr, elsize);
            buffer += elsize;
            NpyArray_ITER_NEXT(it);
        }
        Npy_DECREF(it);
    }
    return 0;
}

extern "C" __declspec(dllexport)
    int _cdecl NpyArrayAccess_Fill(NpyArray* arr)
{
    NpyArray_FillFunc* fill = arr->descr->f->fill;
    if (fill == NULL) {
        NpyErr_SetString(NpyExc_ValueError, "no fill-function for data-type");
        return -1;
    }
    fill(arr->data, NpyArray_SIZE(arr), arr);
    if (NpyErr_Occurred()) {
        return -1;
    }
    return 0;
}

extern "C" __declspec(dllexport)
    void _cdecl NpyArrayAccess_CopySwapIn(NpyArray* arr, npy_int64 offset, void* data, int swap)
{
    arr->descr->f->copyswap(arr->data+offset, data, swap, arr);
}

extern "C" __declspec(dllexport)
    void _cdecl NpyArrayAccess_CopySwapOut(NpyArray* arr, npy_int64 offset, void* data, int swap)
{
    arr->descr->f->copyswap(data, arr->data+offset, swap, arr);
}


// Similar to above, but does not handle string, void or other flexibly sized types because it can't pass
// an array pointer in.  This is specifically used for fixed scalar types.
extern "C" __declspec(dllexport)
    void _cdecl NpyArrayAccess_CopySwapScalar(NpyArray_Descr* descr, void *dest, void* src, int swap)
{
    descr->f->copyswap(dest, src, swap, NULL);
}



extern "C" __declspec(dllexport)
    int _cdecl NpyArrayAccess_SetDateTimeInfo(NpyArray_Descr* descr, const char* units, int num, int den, int events)
{
    NpyArray_DateTimeInfo* info = NpyArray_DateTimeInfoNew(units, num, den, events);
    if (info == NULL) {
        return -1;
    }
    if (descr->dtinfo != NULL) {
        NpyArray_free(descr->dtinfo);
    }
    descr->dtinfo = info;
    return NULL;
}

extern "C" __declspec(dllexport)
    NpyArray_Descr* _cdecl NpyArrayAccess_InheritDescriptor(NpyArray_Descr* type, NpyArray_Descr* conv)
{
    NpyArray_Descr* nw = NpyArray_DescrNew(type);
    if (nw->elsize && nw->elsize != conv->elsize) {
        NpyErr_SetString(NpyExc_ValueError, "mismatch in size of old and new data-descriptor");
        Npy_DECREF(nw);
        return NULL;
    }
    nw->elsize = conv->elsize;
    if (conv->names != NULL) {
        nw->names = NpyArray_DescrNamesCopy(conv->names);
        nw->fields = NpyArray_DescrFieldsCopy(conv->fields);
    }
    nw->flags = conv->flags;
    return nw;
}


extern "C" __declspec(dllexport)
    const char * _cdecl NpyArrayAccess_GetBufferFormatString(NpyArray *arr)
{
    assert(NULL != arr && NPY_VALID_MAGIC == arr->nob_magic_number);

    npy_tmp_string_t fmt = {0, 0, 0};
    if (npy_buffer_format_string(NpyArray_DESCR(arr), &fmt, arr, NULL, NULL) != 0) {
        return NULL;
    }
    npy_append_char(&fmt, '\0');

    // Note: caller must release returned string. Done by Marshal.PtrToStringAnsi().
    return fmt.s;
}


// Stupid hack for freeing the string returned by NpyArrayAccess_GetBufferFormatString
// only because I can't find anything that definitively says where Marshal.FreeHGlobal
// is compatible with free() or not.  I am assuming not and we know the string was 
// allocated in the core using malloc.
extern "C" __declspec(dllexport)
    void NpyArrayAccess_Free(void *p)
{
    free(p);
}


extern "C" __declspec(dllexport)
    NpyArray *NpyArrayAccess_FromFile(const char *fileName, NpyArray_Descr *dtype,
        int count, const char *sep) 
{
    FILE *fp = NULL;
    NpyArray* result;
    errno_t errno;

    assert(NULL != dtype && NPY_VALID_MAGIC == dtype->nob_magic_number);
    
    errno = fopen_s(&fp, fileName, "rb");
    if (errno != 0) {
        NpyErr_SetString(NpyExc_IOError, "unable to open file");
        return NULL;
    }

    Npy_INCREF(dtype);
    result = (NULL == sep) ?
        NpyArray_FromBinaryFile(fp, dtype, count) :
        NpyArray_FromTextFile(fp, dtype, count, const_cast<char *>(sep));
    fclose(fp);
    return result;
}


extern "C" __declspec(dllexport)
    void NpyArrayAccess_SetNamesList(NpyArray_Descr *dtype, const char **nameslist, int len)
{
    char **nl2 = NpyArray_DescrAllocNames(len);

    for (int i = 0; i < len; i++) {
        nl2[i] = strdup(nameslist[i]);
    }
    NpyArray_DescrReplaceNames(dtype, nl2);
}


//
// The following three routines are used to iterate over an NpyDict structure
// from the managed world.
//

extern "C" __declspec(dllexport)
    NpyDict_Iter *NpyArrayAccess_DictAllocIter()
{
    NpyDict_Iter *iter = (NpyDict_Iter *)npy_malloc(sizeof(NpyDict_Iter));
    NpyDict_IterInit(iter);
    return iter;
}

extern "C" __declspec(dllexport)
    void NpyArrayAccess_DictFreeIter(NpyDict_Iter *iter)
{
    npy_free(iter);
}

extern "C" __declspec(dllexport)
    bool NpyArrayAccess_DictNext(NpyDict *dict, NpyDict_Iter *iter, void **key, void **value)
{
    return (NpyDict_IterNext(dict, iter, key, value) != 0);
}

//
// Returns a view of a using prototype as the interfaceData when creating the wrapper.
// This will return the same subtype as prototype and use prototype in the __array_finalize__ call.
//
extern "C" __declspec(dllexport)
    NpyArray* NpyArrayAccess_ViewLike(NpyArray* a, NpyArray* prototype) 
{
    Npy_INCREF(a->descr);
    return NpyArray_NewFromDescr(a->descr, a->nd, a->dimensions, a->strides, a->data, a->flags, NPY_FALSE, NULL, Npy_INTERFACE(prototype));
}


extern "C" __declspec(dllexport)
    char *NpyArrayAccess_FormatLongFloat(double val, int precision)
{
    const int buflen = 64;
    char *buf = (char *)CoTaskMemAlloc(buflen);
    char format[12];

    NpyOS_snprintf(format, sizeof(format), "%%.%i%s", precision, NPY_LONGDOUBLE_FMT);
    NpyOS_ascii_formatl(buf, buflen, format, val, 0);

    // If nothing but digits after sign, append ".0"
    int cnt = strlen(buf);
    int i;
    for (i = (val < 0) ? 1 : 0; i < cnt; i++) {
        if (!isdigit(Npy_CHARMASK(buf[i]))) {
            break;
        }
    }
    if (i == cnt && buflen >= cnt + 3) {
        strcpy_s(&buf[cnt], buflen-cnt, ".0");
    }
    return buf;
}

