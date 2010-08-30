
#include <npy_arrayobject.h>

///
/// This library provides a set of native access functions used by NumpyDotNet
/// for accessing the core library.
///

extern "C" __declspec(dllexport)
    void _cdecl NpyArray_GetOffsets(int *magic_number, int *descr, int *flags);

#define offsetof(type, member) ( (int) & ((type*)0) -> member )


void NpyArray_GetOffsets(int *magic_number, int *descr, int *flags)
{
    NpyArray *ptr = NULL;

    *magic_number = offsetof(NpyArray, magic_number);
    *descr = offsetof(NpyArray, descr);
    *flags = offsetof(NpyArray, flags);
}

