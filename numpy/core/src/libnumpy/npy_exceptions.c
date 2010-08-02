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
    _NpyExc_NOERROR = 0,
    _NpyExc_MemoryError,
    _NpyExc_IOError,
    _NpyExc_ValueError,
    _NpyExc_TypeError,
    _NpyExc_IndexError,
    _NpyExc_RuntimeError,
    _NpyExc_AttributeError,
};


static enum npyexc_type cur = _NpyExc_NOERROR;
static char msg[MSG_SIZE];


void _NpyErr_Clear()
{
    cur = _NpyExc_NOERROR;
    msg[0] = '\0';
}


int _NpyErr_Occurred()
{
    return cur;
}


/* Return the current error message.  Call this function only when
   NpyErr_Occurred() returned non-zero. */
char *_NpyErr_OccurredString()
{
    assert(_NpyErr_Occurred());
    return msg;
}


int _NpyErr_ExceptionMatches(int exc)
{
    return (exc == cur) ? 1 : 0;
}


void _NpyErr_SetString(int exc, const char *str)
{
    cur = exc;
    assert(strlen(str) < MSG_SIZE);
    strcpy(msg, str);
}


void _NpyErr_Format(int exc, const char *format, ...)
{
    va_list vargs;

    cur = exc;
    assert(strlen(format) < MSG_SIZE);
    va_start(vargs, format);
    vsprintf(msg, format, vargs);
    va_end(vargs);
}


void _NpyErr_NoMemory()
{
    cur = _NpyExc_MemoryError;
    msg[0] = '\0';
}


void _NpyErr_Print()
{
    assert(_NpyErr_Occurred());
#define CP(t)   if (cur == t) fprintf(stderr, "%s: %s\n", # t, msg);
    CP(_NpyExc_MemoryError);
    CP(_NpyExc_IOError);
    CP(_NpyExc_ValueError);
    CP(_NpyExc_TypeError);
    CP(_NpyExc_IndexError);
    CP(_NpyExc_RuntimeError);
    CP(_NpyExc_AttributeError);
#undef CP
}


/*
int main()
{
    _NpyErr_SetString(_NpyExc_TypeError, "something has wrong type");
    printf("String = '%s'\n", _NpyErr_OccurredString());
    _NpyErr_Print();
    _NpyErr_Format(_NpyExc_ValueError, "too large");
    _NpyErr_Print();
    _NpyErr_Format(_NpyExc_ValueError, "too large (%d)", 32);
    _NpyErr_Print();
    _NpyErr_Format(_NpyExc_AttributeError, "too large (%d) %s", 32, "bad");
    _NpyErr_Print();
    _NpyErr_NoMemory();
    _NpyErr_Print();
    assert(_NpyErr_ExceptionMatches(_NpyExc_MemoryError));
    assert(!_NpyErr_ExceptionMatches(_NpyExc_IndexError));

    return 0;
}
*/
