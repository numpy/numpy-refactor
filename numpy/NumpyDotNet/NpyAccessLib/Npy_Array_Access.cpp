
#include <assert.h>

extern "C" {
#include <npy_api.h>
#include <npy_arrayobject.h>
#include <npy_descriptor.h>
#include <npy_object.h>
}


///
/// This library provides a set of native access functions used by NumpyDotNet
/// for accessing the core library.
///


#define offsetof(type, member) ( (int) & ((type*)0) -> member )

extern "C" __declspec(dllexport)
void _cdecl NpyArrayAccess_Incref(NpyObject *obj)
{
    //assert(NPY_VALID_MAGIC == obj->nob_magic_number);
    Npy_INCREF(obj);
}

extern "C" __declspec(dllexport)
void _cdecl NpyArrayAccess_Decref(NpyObject *obj)
{
    //assert(NPY_VALID_MAGIC == obj->nob_magic_number);
    Npy_DECREF(obj);
}


extern "C" __declspec(dllexport)
void _cdecl NpyArrayAccess_ArrayGetOffsets(int *magic_number, int *descr, int *nd, int *flags)
{
    NpyArray *ptr = NULL;

    *magic_number = offsetof(NpyArray, magic_number);
    *descr = offsetof(NpyArray, descr);
    *nd = offsetof(NpyArray, nd);
    *flags = offsetof(NpyArray, flags);
}


extern "C" __declspec(dllexport)
void _cdecl NpyArrayAccess_ArraySetDescr(void *arrTmp, void *newDescrTmp) 
{
    NpyArray *arr = (NpyArray *)arrTmp;
    NpyArray_Descr *newDescr = (NpyArray_Descr *)newDescrTmp;
    assert(NPY_VALID_MAGIC == arr->magic_number);
    assert(NPY_VALID_MAGIC == newDescr->magic_number);

    NpyArray_Descr *oldDescr = arr->descr;
    Npy_INCREF(newDescr);
    arr->descr = newDescr;
    Npy_DECREF(oldDescr);
}
