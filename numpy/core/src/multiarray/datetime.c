#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <datetime.h>

#include <time.h>

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include <numpy/ndarrayobject.h>
#include <npy_api.h>

#include "npy_config.h"

#include "numpy/npy_3kcompat.h"

#include "_datetime.h"

/* For defaults and errors */
#define NPY_FR_ERR  -1

/* Offset for number of days between Dec 31, 1969 and Jan 1, 0001
*  Assuming Gregorian calendar was always in effect (proleptic Gregorian calendar)
*/

/* Calendar Structure for Parsing Long -> Date */
typedef struct {
    int year, month, day;
} ymdstruct;

typedef struct {
    int hour, min, sec;
} hmsstruct;


/*
  ====================================================
  == Beginning of section borrowed from mx.DateTime ==
  ====================================================
*/

/*
 * Functions in the following section are borrowed from mx.DateTime version
 * 2.0.6, and hence this code is subject to the terms of the egenix public
 * license version 1.0.0
 */

#define Py_AssertWithArg(x,errortype,errorstr,a1) {if (!(x)) {PyErr_Format(errortype,errorstr,a1);goto onError;}}

/* Table with day offsets for each month (0-based, without and with leap) */
static int month_offset[2][13] = {
    { 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365 },
    { 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366 }
};

/* Table of number of days in a month (0-based, without and with leap) */
static int days_in_month[2][12] = {
    { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
    { 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
};

/* Return 1/0 iff year points to a leap year in calendar. */
static int
is_leapyear(long year)
{
    return (year % 4 == 0) && ((year % 100 != 0) || (year % 400 == 0));
}


/*
 * Return the day of the week for the given absolute date.
 * Monday is 0 and Sunday is 6
 */
static int
day_of_week(npy_longlong absdate)
{
    /* Add in four for the Thursday on Jan 1, 1970 (epoch offset)*/
    absdate += 4;

    if (absdate >= 0) {
        return absdate % 7;
    }
    else {
        return 6 + (absdate + 1) % 7;
    }
}

/*
 * Return the year offset, that is the absolute date of the day
 * 31.12.(year-1) since 31.12.1969 in the proleptic Gregorian calendar.
 */
static npy_longlong
year_offset(npy_longlong year)
{
    /* Note that 477 == 1969/4 - 1969/100 + 1969/400 */
    year--;
    if (year >= 0 || -1/4 == -1)
        return (year-1969)*365 + year/4 - year/100 + year/400 - 477;
    else
        return (year-1969)*365 + (year-3)/4 - (year-99)/100 + (year-399)/400 - 477;
}

/*
 * Modified version of mxDateTime function
 * Returns absolute number of days since Jan 1, 1970
 * assuming a proleptic Gregorian Calendar
 * Raises a ValueError if out of range month or day
 * day -1 is Dec 31, 1969, day 0 is Jan 1, 1970, day 1 is Jan 2, 1970
 */
static npy_longlong
days_from_ymd(int year, int month, int day)
{

    /* Calculate the absolute date */
    int leap;
    npy_longlong yearoffset, absdate;

    /* Is it a leap year ? */
    leap = is_leapyear(year);

    /* Negative month values indicate months relative to the years end */
    if (month < 0) month += 13;
    Py_AssertWithArg(month >= 1 && month <= 12,
                     PyExc_ValueError,
                     "month out of range (1-12): %i",
                     month);

    /* Negative values indicate days relative to the months end */
    if (day < 0) day += days_in_month[leap][month - 1] + 1;
    Py_AssertWithArg(day >= 1 && day <= days_in_month[leap][month - 1],
                     PyExc_ValueError,
                     "day out of range: %i",
                     day);

    /*
     * Number of days between Dec 31, (year - 1) and Dec 31, 1969
     *    (can be negative).
     */
    yearoffset = year_offset(year);

    if (PyErr_Occurred()) goto onError;

    /*
     * Calculate the number of days using yearoffset
     * Jan 1, 1970 is day 0 and thus Dec. 31, 1969 is day -1
     */
    absdate = day-1 + month_offset[leap][month - 1] + yearoffset;

    return absdate;

 onError:
    return 0;

}

/* Returns absolute seconds from an hour, minute, and second
 */
#define secs_from_hms(hour, min, sec, multiplier) (\
  ((hour)*3600 + (min)*60 + (sec)) * (npy_int64)(multiplier)\
)

/*
 * Takes a number of days since Jan 1, 1970 (positive or negative)
 * and returns the year. month, and day in the proleptic
 * Gregorian calendar
 *
 * Examples:
 *
 * -1 returns 1969, 12, 31
 * 0  returns 1970, 1, 1
 * 1  returns 1970, 1, 2
 */

static ymdstruct
days_to_ymdstruct(npy_datetime dlong)
{
    ymdstruct ymd;
    long year;
    npy_longlong yearoffset;
    int leap, dayoffset;
    int month = 1, day = 1;
    int *monthoffset;

    dlong += 1;

    /* Approximate year */
    year = 1970 + dlong / 365.2425;

    /* Apply corrections to reach the correct year */
    while (1) {
        /* Calculate the year offset */
        yearoffset = year_offset(year);

        /*
         * Backward correction: absdate must be greater than the
         * yearoffset
         */
        if (yearoffset >= dlong) {
            year--;
            continue;
        }

        dayoffset = dlong - yearoffset;
        leap = is_leapyear(year);

        /* Forward correction: non leap years only have 365 days */
        if (dayoffset > 365 && !leap) {
            year++;
            continue;
        }
        break;
    }

    /* Now iterate to find the month */
    monthoffset = month_offset[leap];
    for (month = 1; month < 13; month++) {
        if (monthoffset[month] >= dayoffset)
            break;
    }
    day = dayoffset - month_offset[leap][month-1];

    ymd.year  = year;
    ymd.month = month;
    ymd.day   = day;

    return ymd;
}


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
