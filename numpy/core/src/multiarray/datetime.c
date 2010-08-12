#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <datetime.h>

#include <time.h>

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include <numpy/ndarrayobject.h>
#include <npy_api.h>



/*
 * NOTE: Datetime implementation moved to libnumpy/npy_datetime.c
 */

/*NUMPY_API
 * Create a datetime value from a filled datetime struct and resolution unit.
 */
NPY_NO_EXPORT npy_datetime
PyArray_DatetimeStructToDatetime(NPY_DATETIMEUNIT fr, npy_datetimestruct *d)
{
    return NpyArray_DatetimeStructToDatetime(fr, d);
}


/*NUMPY_API
 * Create a timdelta value from a filled timedelta struct and resolution unit.
 */
NPY_NO_EXPORT npy_datetime
PyArray_TimedeltaStructToTimedelta(NPY_DATETIMEUNIT fr, npy_timedeltastruct *d)
{
    return NpyArray_TimedeltaStructToTimedelta(fr, d);
}



/*NUMPY_API
 * Fill the datetime struct from the value and resolution unit.
 */
NPY_NO_EXPORT void
PyArray_DatetimeToDatetimeStruct(npy_datetime val, NPY_DATETIMEUNIT fr,
                                 npy_datetimestruct *result)
{
    NpyArray_DatetimeToDatetimeStruct(val, fr, result);
}


/*
 * FIXME: Overflow is not handled at all
 *   To convert from Years, Months, and Business Days, multiplication by the average is done
 */

/*NUMPY_API
 * Fill the timedelta struct from the timedelta value and resolution unit.
 */
NPY_NO_EXPORT void
PyArray_TimedeltaToTimedeltaStruct(npy_timedelta val, NPY_DATETIMEUNIT fr,
                                 npy_timedeltastruct *result)
{
    NpyArray_TimedeltaToTimedeltaStruct(val, fr, result);
}
