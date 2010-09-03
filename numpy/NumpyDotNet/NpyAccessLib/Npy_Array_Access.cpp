
#include <assert.h>

extern "C" {
#include <npy_api.h>
#include <npy_defs.h>
#include <npy_arrayobject.h>
#include <npy_descriptor.h>
#include <npy_object.h>
}

extern "C" _declspec(dllimport) npy_interface_incref _NpyInterface_Incref;
extern "C" _declspec(dllimport) npy_interface_decref _NpyInterface_Decref;


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
    //Npy_DECREF(obj);
    if (0 == InterlockedDecrement(&(obj)->nob_refcnt)) {                            
        if (NULL != Npy_INTERFACE(obj))                                       
            _NpyInterface_Decref(Npy_INTERFACE(obj), &((obj)->nob_interface));  
        else                                                                
           (obj)->nob_type->ntp_dealloc((_NpyObject*)obj);                      
    }
}


extern "C" __declspec(dllexport)
void _cdecl NpyArrayAccess_ArrayGetOffsets(int *magic_number, int *descr, int *nd, int *flags)
{
    NpyArray *ptr = NULL;

    *magic_number = offsetof(NpyArray, nob_magic_number);
    *descr = offsetof(NpyArray, descr);
    *nd = offsetof(NpyArray, nd);
    *flags = offsetof(NpyArray, flags);
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


// Returns the native byte order code for this platform. Used during
// initialization.
extern "C" __declspec(dllexport)
    char _cdecl NpyArrayAccess_GetNativeByteOrder()
{
    return NPY_NATBYTE;
}


// Fills in an int64 array with the dimensions of the array.
extern "C" __declspec(dllexport)
    bool _cdecl NpyArrayAccess_GetArrayDims(void *arrTmp, int ndims, npy_int64 *dims)
{
    NpyArray *arr = (NpyArray *)arrTmp;
    assert(NPY_VALID_MAGIC == arr->nob_magic_number);

    if (ndims != arr->nd) return false;
    if (sizeof(npy_int64) == sizeof(npy_intp)) {
        // Fast if sizes are the same.
        memcpy(dims, arr->dimensions, sizeof(npy_intp) * arr->nd);
    } else {
        // Slower, but converts between sizes.
        for (int i = 0; i < arr->nd; i++) { 
            dims[i] = arr->dimensions[i];
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
