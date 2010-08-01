/*
 *  npy_exceptions.c -
 *
 */
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

#define MSG_SIZE  1024


enum {
    NpyExc_NOERROR = 0,
    NpyExc_MemoryError,
    NpyExc_IOError,
    NpyExc_ValueError,
    NpyExc_TypeError,
    NpyExc_IndexError,
    NpyExc_RuntimeError,
    NpyExc_AttributeError,
};


int _type = NpyExc_NOERROR;
char _msg[MSG_SIZE];


void NpyErr_Clear()
{
    _type = NpyExc_NOERROR;
    _msg[0] = '\0';
}


int NpyErr_Occurred()
{
    return _type;
}


int NpyErr_ExceptionMatches(int exc)
{
    return exc == _type ? 1 : 0;
}


void NpyErr_SetString(int exc, const char *str)
{
    _type = exc;
    assert(strlen(str) < MSG_SIZE);
    strcpy(_msg, str);
}


void NpyErr_Format(int exc, const char *format, ...)
{
    va_list vargs;

    _type = exc;
    assert(strlen(format) < MSG_SIZE);
    va_start(vargs, format);
    vsprintf(_msg, format, vargs);
    va_end(vargs);
}


void NpyErr_NoMemory()
{
    _type = NpyExc_MemoryError;
    _msg[0] = '\0';
}


void NpyErr_Print()
{
    assert(NpyErr_Occurred());
#define CP(t)   if (_type == t) fprintf(stderr, "%s: %s\n", # t, _msg);
    CP(NpyExc_MemoryError);
    CP(NpyExc_IOError);
    CP(NpyExc_ValueError);
    CP(NpyExc_TypeError);
    CP(NpyExc_IndexError);
    CP(NpyExc_RuntimeError);
    CP(NpyExc_AttributeError);
#undef CP
}


int main()
{
    NpyErr_SetString(NpyExc_TypeError, "something has wrong type");
    NpyErr_Print();
    NpyErr_Format(NpyExc_ValueError, "too large");
    NpyErr_Print();
    NpyErr_Format(NpyExc_ValueError, "too large (%d)", 32);
    NpyErr_Print();
    NpyErr_Format(NpyExc_AttributeError, "too large (%d) %s", 32, "bad");
    NpyErr_Print();
    NpyErr_NoMemory();
    NpyErr_Print();
    assert(NpyErr_ExceptionMatches(NpyExc_MemoryError));
    assert(!NpyErr_ExceptionMatches(NpyExc_IndexError));

    return 0;
}
