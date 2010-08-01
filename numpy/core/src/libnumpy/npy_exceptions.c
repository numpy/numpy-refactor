/*
 *  npy_exceptions.c -
 *
 */
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>


enum {
    NpyExc_NOERROR,
    NpyExc_ValueError,
    NpyExc_MemoryError,
    NpyExc_IOError,
    NpyExc_TypeError,
    NpyExc_IndexError,
    NpyExc_RuntimeError,
    NpyExc_AttributeError,
};


int _type = NpyExc_NOERROR;
char _msg[256];


void NpyErr_Clear()
{
    _type = NpyExc_NOERROR;
}


int NpyErr_Occurred()
{
    return _type == NpyExc_NOERROR ? 0 : 1;
}


void NpyErr_SetString(int exc, const char *str)
{
    _type = exc;
    assert(strlen(str) < 256);
    strcpy(_msg, str);
}


void NpyErr_Format(int exc, const char *format, ...)
{
    va_list vargs;

    _type = exc;
    va_start(vargs, format);
    vsprintf(_msg, format, vargs);
    va_end(vargs);
}


void NpyErr_NoMemory()
{
    _type = NpyExc_MemoryError;
}


void NpyErr_Print()
{
    assert(NpyErr_Occurred());
    fprintf(stderr, "%d: %s\n", _type, _msg);
}


int main()
{
    NpyErr_SetString(NpyExc_TypeError, "something has wrong type");
    NpyErr_Print();
    NpyErr_Format(NpyExc_ValueError, "too large");
    NpyErr_Print();
    NpyErr_Format(NpyExc_ValueError, "too large (%d)", 32);
    NpyErr_Print();
    NpyErr_Format(NpyExc_ValueError, "too large (%d) %s", 32, "bad");
    NpyErr_Print();

    return 0;
}
