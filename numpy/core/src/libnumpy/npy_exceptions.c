/*
 *  npy_exceptions.c -
 *
 */
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

#define MSG_SIZE  1024


enum npyexc_type {
    NpyExc_NOERROR = 0,
    NpyExc_MemoryError,
    NpyExc_IOError,
    NpyExc_ValueError,
    NpyExc_TypeError,
    NpyExc_IndexError,
    NpyExc_RuntimeError,
    NpyExc_AttributeError,
};


static enum npyexc_type cur = NpyExc_NOERROR;
static char msg[MSG_SIZE];


void NpyErr_Clear()
{
    cur = NpyExc_NOERROR;
    msg[0] = '\0';
}


int NpyErr_Occurred()
{
    return cur;
}


/* Return the current error message.  Call this function only when
   NpyErr_Occurred() returned non-zero. */
char *NpyErr_OccurredString()
{
    assert(NpyErr_Occurred());
    return msg;
}


int NpyErr_ExceptionMatches(int exc)
{
    return (exc == cur) ? 1 : 0;
}


void NpyErr_SetString(int exc, const char *str)
{
    cur = exc;
    assert(strlen(str) < MSG_SIZE);
    strcpy(msg, str);
}


void NpyErr_Format(int exc, const char *format, ...)
{
    va_list vargs;

    cur = exc;
    assert(strlen(format) < MSG_SIZE);
    va_start(vargs, format);
    vsprintf(msg, format, vargs);
    va_end(vargs);
}


void NpyErr_NoMemory()
{
    cur = NpyExc_MemoryError;
    msg[0] = '\0';
}


void NpyErr_Print()
{
    assert(NpyErr_Occurred());
#define CP(t)   if (cur == t) fprintf(stderr, "%s: %s\n", # t, msg);
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
    printf("String = '%s'\n", NpyErr_OccurredString());
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
