from os import path
import numpy as np
from numpy.testing import *
import datetime

class TestDateTime(TestCase):
    def test_creation(self):
        for unit in ['Y', 'M', 'W', 'B', 'D',
                     'h', 'm', 's', 'ms', 'us',
                     'ns', 'ps', 'fs', 'as']:
            dt1 = np.dtype('M8[750%s]' % unit)
            assert dt1 == np.dtype('datetime64[750%s]' % unit)
            dt2 = np.dtype('m8[%s]' % unit)
            assert dt2 == np.dtype('timedelta64[%s]' % unit)

    def test_divisor_conversion_year(self):
        assert np.dtype('M8[Y/4]') == np.dtype('M8[3M]')
        assert np.dtype('M8[Y/13]') == np.dtype('M8[4W]')
        assert np.dtype('M8[3Y/73]') == np.dtype('M8[15D]')

    def test_divisor_conversion_month(self):
        assert np.dtype('M8[M/2]') == np.dtype('M8[2W]')
        assert np.dtype('M8[M/15]') == np.dtype('M8[2D]')
        assert np.dtype('M8[3M/40]') == np.dtype('M8[54h]')

    def test_divisor_conversion_week(self):
        assert np.dtype('m8[W/5]') == np.dtype('m8[B]')
        assert np.dtype('m8[W/7]') == np.dtype('m8[D]')
        assert np.dtype('m8[3W/14]') == np.dtype('m8[36h]')
        assert np.dtype('m8[5W/140]') == np.dtype('m8[360m]')

    def test_divisor_conversion_bday(self):
        assert np.dtype('M8[B/12]') == np.dtype('M8[2h]')
        assert np.dtype('M8[B/120]') == np.dtype('M8[12m]')
        assert np.dtype('M8[3B/960]') == np.dtype('M8[270s]')

    def test_divisor_conversion_day(self):
        assert np.dtype('M8[D/12]') == np.dtype('M8[2h]')
        assert np.dtype('M8[D/120]') == np.dtype('M8[12m]')
        assert np.dtype('M8[3D/960]') == np.dtype('M8[270s]')

    def test_divisor_conversion_hour(self):
        assert np.dtype('m8[h/30]') == np.dtype('m8[2m]')
        assert np.dtype('m8[3h/300]') == np.dtype('m8[36s]')

    def test_divisor_conversion_minute(self):
        assert np.dtype('m8[m/30]') == np.dtype('m8[2s]')
        assert np.dtype('m8[3m/300]') == np.dtype('m8[600ms]')

    def test_divisor_conversion_second(self):
        assert np.dtype('m8[s/100]') == np.dtype('m8[10ms]')
        assert np.dtype('m8[3s/10000]') == np.dtype('m8[300us]')

    def test_divisor_conversion_fs(self):
        assert np.dtype('M8[fs/100]') == np.dtype('M8[10as]')
        self.assertRaises(ValueError, lambda : np.dtype('M8[3fs/10000]'))

    def test_divisor_conversion_as(self):
        self.assertRaises(ValueError, lambda : np.dtype('M8[as/10]'))

    def test_creation_overflow(self):
        date = '1980-03-23 20:00:00'
        timesteps = np.array([date], dtype='datetime64[s]')[0].astype(np.int64)
        for unit in ['ms', 'us', 'ns']:
            timesteps *= 1000
            x = np.array([date], dtype='datetime64[%s]' % unit)

            assert_equal(timesteps, x[0].astype(np.int64),
                         err_msg='Datetime conversion error for unit %s' % unit)

        assert_equal(x[0].astype(np.int64), 322689600000000000)


class TestDateTimeModulo(TestCase):
    def test_modulo_years(self):
        timesteps = np.array([0,1,2], dtype='datetime64[Y]//10')
        assert timesteps[0] == np.datetime64('1970')
        assert timesteps[1] == np.datetime64('1980')
        assert timesteps[2] == np.datetime64('1990')

    def test_modulo_months(self):
        timesteps = np.array([0,1,2], dtype='datetime64[M]//10')
        assert timesteps[0] == np.datetime64('1970-01')
        assert timesteps[1] == np.datetime64('1970-11')
        assert timesteps[2] == np.datetime64('1971-09')

    def test_modulo_weeks(self):
        timesteps = np.array([0,1,2], dtype='datetime64[W]//3')
        assert timesteps[0] == np.datetime64('1970-01-01')
        assert timesteps[1] == np.datetime64('1970-01-22')
        assert timesteps[2] == np.datetime64('1971-02-12')
        
    def test_modulo_business_days(self):
        timesteps = np.array([0,1,2], dtype='datetime64[B]//4')
        assert timesteps[0] == np.datetime64('1970-01-01')
        assert timesteps[1] == np.datetime64('1970-01-07')
        assert timesteps[2] == np.datetime64('1971-01-13')

    def test_modulo_days(self):
        timesteps = np.array([0,1,2], dtype='datetime64[D]//17')
        assert timesteps[0] == np.datetime64('1970-01-01')
        assert timesteps[1] == np.datetime64('1970-01-18')
        assert timesteps[2] == np.datetime64('1971-02-04')

    def test_modulo_hours(self):
        timesteps = np.array([0,1,2], dtype='datetime64[h]//17')
        assert timesteps[0] == np.datetime64('1970-01-01 00')
        assert timesteps[1] == np.datetime64('1970-01-01 17')
        assert timesteps[2] == np.datetime64('1970-01-02 10')

    def test_modulo_minutes(self):
        timesteps = np.array([0,1,2], dtype='datetime64[m]//42')
        assert timesteps[0] == np.datetime64('1970-01-01 00:00')
        assert timesteps[1] == np.datetime64('1970-01-01 00:42')
        assert timesteps[2] == np.datetime64('1970-01-01 01:24')

    def test_modulo_seconds(self):
        timesteps = np.array([0,1,2], dtype='datetime64[s]//42')
        assert timesteps[1] == np.datetime64('1970-01-01 00:00:00')
        assert timesteps[1] == np.datetime64('1970-01-01 00:00:42')
        assert timesteps[1] == np.datetime64('1970-01-01 00:01:24')

    def test_modulo_milliseconds(self):
        timesteps = np.array([0,1,2], dtype='datetime64[ms]//42')
        assert timesteps[1] == np.datetime64('1970-01-01 00:00:00.000')
        assert timesteps[1] == np.datetime64('1970-01-01 00:00:00.042')
        assert timesteps[1] == np.datetime64('1970-01-01 00:01:00.084')

    def test_modulo_microseconds(self):
        timesteps = np.array([0,1,2], dtype='datetime64[us]//42')
        assert timesteps[1] == np.datetime64('1970-01-01 00:00:00.000000')
        assert timesteps[1] == np.datetime64('1970-01-01 00:00:00.000042')
        assert timesteps[1] == np.datetime64('1970-01-01 00:01:00.000084')

    def test_modulo_nanoseconds(self):
        timesteps = np.array([0,1,2], dtype='datetime64[ns]//42')
        assert timesteps[1] == np.datetime64('1970-01-01 00:00:00.000000000')
        assert timesteps[1] == np.datetime64('1970-01-01 00:00:00.000000042')
        assert timesteps[1] == np.datetime64('1970-01-01 00:01:00.000000084')

    def test_modulo_picoseconds(self):
        timesteps = np.array([0,1,2], dtype='datetime64[ps]//42')
        assert timesteps[1] == np.datetime64('1970-01-01 00:00:00.000000000000')
        assert timesteps[1] == np.datetime64('1970-01-01 00:00:00.000000000042')
        assert timesteps[1] == np.datetime64('1970-01-01 00:01:00.000000000084')

    def test_modulo_femtoseconds(self):
        timesteps = np.array([0,1,2], dtype='datetime64[fs]//42')
        assert timesteps[1] == np.datetime64('1970-01-01 00:00:00.000000000000000')
        assert timesteps[1] == np.datetime64('1970-01-01 00:00:00.000000000000042')
        assert timesteps[1] == np.datetime64('1970-01-01 00:01:00.000000000000084')

    def test_modulo_attoseconds(self):
        timesteps = np.array([0,1,2], dtype='datetime64[as]//42')
        assert timesteps[1] == np.datetime64('1970-01-01 00:00:00.000000000000000000')
        assert timesteps[1] == np.datetime64('1970-01-01 00:00:00.000000000000000042')
        assert timesteps[1] == np.datetime64('1970-01-01 00:01:00.000000000000000084')


class TestTimeDeltaSetters(TestCase):
    def setUp(self):
        self.timedeltas = np.ones(3, dtype='m8[ms]')

    def test_set_timedelta64_from_int(self):
        self.timedeltas[0] = 12
        assert self.timedeltas[0] == np.timedelta64(12, 'ms')

    def test_set_timedelta64_from_datetime_timedelta(self):
        self.timedeltas[1] = datetime.timedelta(0, 0, 13000) 
        assert self.timedeltas[1] == np.timedelta64(13, 'ms')

    def test_set_timedelta64_from_string(self):
        self.timedeltas[2] = '0:00:00.014' 
        assert self.timedeltas[2] == np.timedelta64(14, 'ms')


class TestTimeDeltaGetters(TestCase):
    def setUp(self):
        self.timedeltas = np.array([12, 13, 14], 'm8[ms]')

    def test_get_str_from_timedelta64(self):
        assert str(self.timedeltas[0]) == '0:00:00.012'
        assert str(self.timedeltas[1]) == '0:00:00.013'
        assert str(self.timedeltas[2]) == '0:00:00.014'

    def test_get_repr_from_timedelta64(self):
        assert repr(self.timedeltas[0]) == "timedelta64(12, 'ms')"
        assert repr(self.timedeltas[1]) == "timedelta64(13, 'ms')"
        assert repr(self.timedeltas[2]) == "timedelta64(14, 'ms')"

    def test_get_str_from_timedelta64_item(self):
        assert str(self.timedeltas[0].item()) == '0:00:00.012000'
        assert str(self.timedeltas[1].item()) == '0:00:00.013000'
        assert str(self.timedeltas[2].item()) == '0:00:00.014000'

    def test_get_repr_from_timedelta64_item(self):
        assert repr(self.timedeltas[0].item()) == 'datetime.timedelta(0, 0, 12000)'
        assert repr(self.timedeltas[1].item()) == 'datetime.timedelta(0, 0, 13000)'
        assert repr(self.timedeltas[2].item()) == 'datetime.timedelta(0, 0, 14000)'

    def test_get_str_from_timedelta64_array(self):
        assert str(self.timedeltas) == '[0:00:00.012  0:00:00.014  0:00:00.014]'

    def test_get_repr_from_timedelta64_array(self):
        assert repr(self.timedeltas) == 'array([12, 13, 14], dtype="timedelta64[ms]")'


class TestTimeDeltaComparisons(TestCase):
    def setUp(self):
        self.timedeltas = np.array([12, 13, 14], 'm8[ms]')

    def test_compare_timedelta64_to_timedelta64_array(self):
        comparison = (self.timedeltas == np.array([12, 13, 13], 'm8[ms]'))
        assert_equal(comparison, [True, True, False])

    def test_compare_timedelta64_to_timedelta64_broadcast(self):
        comparison = (self.timedeltas == np.timedelta64(13, 'ms'))
        assert_equal(comparison, [False, True, True])

    def test_compare_timedelta64_to_string_broadcast(self):
        comparison = (self.timedeltas == '0:00:00.012')
        assert_equal(comparison, [True, False, True])


class TestDateTimeAstype(TestCase):
    def test_datetime_astype_years(self):
        datetimes = np.array([0, 40, 15], dtype="datetime64[M]")
        assert_equal(datetimes.astype('datetime64[Y]'), np.array([0, 3, 2], dtype="datetime64[Y]"))

    def test_datetime_astype_months(self):
        datetimes = np.array([0, 3, 2], dtype="datetime64[Y]")
        assert_equal(datetimes.astype('datetime64[M]'), np.array([0, 36, 24], dtype="datetime64[M]"))
        datetimes = np.array([0, 100, 70], dtype="datetime64[D]")
        assert_equal(datetimes.astype('datetime64[M]'), np.array([0, 3, 2], dtype="datetime64[M]"))

    def test_datetime_astype_weeks(self):
        datetimes = np.array([0, 22, 15], dtype="datetime64[D]")
        assert_equal(datetimes.astype('datetime64[W]'), np.array([0, 3, 2], dtype="datetime64[W]"))

    def test_datetime_astype_business_days(self):
        # XXX: There will probably be a more direct way to check for
        #      *Not a Time* values.
        datetimes = np.arange(5, dtype='datetime64[D]')
        expected_array_str = '[1970-01-01  1970-01-02  NaT  NaT  1970-01-05]'
        assert_equal(datetimes.astype('datetime64[B]'), expected_array_str)

    def test_datetime_astype_days(self):
        datetimes = np.array([0, 3, 2], dtype="datetime64[W]")
        assert_equal(datetimes.astype('datetime64[D]'), np.array([0, 21, 7], dtype="datetime64[D]"))
        datetimes = np.array([0, 37, 24], dtype="datetime64[h]")
        assert_equal(datetimes.astype('datetime64[D]'), np.array([0, 3, 2], dtype="datetime64[D]"))

    def test_datetime_astype_hours(self):
        datetimes = np.array([0, 3, 2], dtype="datetime64[D]")
        assert_equal(datetimes.astype('datetime64[h]'), np.array([0, 36, 24], dtype="datetime64[D]"))
        datetimes = np.array([0, 190, 153], dtype="datetime64[m]")
        assert_equal(datetimes.astype('datetime64[h]'), np.array([0, 3, 2], dtype="datetime64[h]"))

    def test_datetime_astype_minutes(self):
        datetimes = np.array([0, 3, 2], dtype="datetime64[h]")
        assert_equal(datetimes.astype('datetime64[m]'), np.array([0, 180, 120], dtype="datetime64[m]"))
        datetimes = np.array([0, 190, 153], dtype="datetime64[s]")
        assert_equal(datetimes.astype('datetime64[m]'), np.array([0, 3, 2], dtype="datetime64[m]"))

    def test_datetime_astype_seconds(self):
        datetimes = np.array([0, 3, 2], dtype="datetime64[m]")
        assert_equal(datetimes.astype('datetime64[s]'), np.array([0, 180, 120], dtype="datetime64[s]"))
        datetimes = np.array([0, 3200, 2430], dtype="datetime64[ms]")
        assert_equal(datetimes.astype('datetime64[s]'), np.array([0, 3, 2], dtype="datetime64[s]"))

    def test_datetime_astype_milliseconds(self):
        datetimes = np.array([0, 3, 2], dtype="datetime64[s]")
        assert_equal(datetimes.astype('datetime64[ms]'), np.array([0, 3000, 2000], dtype="datetime64[ms]"))
        datetimes = np.array([0, 3200, 2430], dtype="datetime64[us]")
        assert_equal(datetimes.astype('datetime64[ms]'), np.array([0, 3, 2], dtype="datetime64[ms]"))

    def test_datetime_astype_microseconds(self):
        datetimes = np.array([0, 3, 2], dtype="datetime64[ms]")
        assert_equal(datetimes.astype('datetime64[us]'), np.array([0, 3000, 2000], dtype="datetime64[us]"))
        datetimes = np.array([0, 3200, 2430], dtype="datetime64[ns]")
        assert_equal(datetimes.astype('datetime64[us]'), np.array([0, 3, 2], dtype="datetime64[us]"))

    def test_datetime_astype_nanoseconds(self):
        datetimes = np.array([0, 3, 2], dtype="datetime64[us]")
        assert_equal(datetimes.astype('datetime64[ns]'), np.array([0, 3000, 2000], dtype="datetime64[ns]"))
        datetimes = np.array([0, 3200, 2430], dtype="datetime64[ps]")
        assert_equal(datetimes.astype('datetime64[ns]'), np.array([0, 3, 2], dtype="datetime64[ns]"))

    def test_datetime_astype_picoseconds(self):
        datetimes = np.array([0, 3, 2], dtype="datetime64[ns]")
        assert_equal(datetimes.astype('datetime64[ps]'), np.array([0, 3000, 2000], dtype="datetime64[ps]"))
        datetimes = np.array([0, 3200, 2430], dtype="datetime64[ns]")
        assert_equal(datetimes.astype('datetime64[ps]'), np.array([0, 3, 2], dtype="datetime64[ps]"))

    def test_datetime_astype_femtoseconds(self):
        datetimes = np.array([0, 3, 2], dtype="datetime64[ps]")
        assert_equal(datetimes.astype('datetime64[fs]'), np.array([0, 3000, 2000], dtype="datetime64[fs]"))
        datetimes = np.array([0, 3200, 2430], dtype="datetime64[ps]")
        assert_equal(datetimes.astype('datetime64[fs]'), np.array([0, 3, 2], dtype="datetime64[fs]"))

    def test_datetime_astype_attoseconds(self):
        datetimes = np.array([0, 3, 2], dtype="datetime64[fs]")
        assert_equal(datetimes.astype('datetime64[as]'), np.array([0, 3000, 2000], dtype="datetime64[as]"))


class TestTimeDeltaAstype(TestCase):
    def test_timedelta_astype_years(self):
        timedeltas = np.array([0, 40, 15], dtype="timedelta64[M]")
        assert_equal(timedeltas.astype('timedelta64[Y]'), np.array([0, 3, 2], dtype="timedelta64[Y]"))

    def test_timedelta_astype_months(self):
        timedeltas = np.array([0, 3, 2], dtype="timedelta64[Y]")
        assert_equal(timedeltas.astype('timedelta64[M]'), np.array([0, 36, 24], dtype="timedelta64[M]"))
        timedeltas = np.array([0, 100, 70], dtype="timedelta64[D]")
        assert_equal(timedeltas.astype('timedelta64[M]'), np.array([0, 3, 2], dtype="timedelta64[M]"))

    def test_timedelta_astype_weeks(self):
        timedeltas = np.array([0, 22, 15], dtype="timedelta64[D]")
        assert_equal(timedeltas.astype('timedelta64[W]'), np.array([0, 3, 2], dtype="timedelta64[W]"))

    def test_timedelta_astype_days(self):
        timedeltas = np.array([0, 3, 2], dtype="timedelta64[W]")
        assert_equal(timedeltas.astype('timedelta64[D]'), np.array([0, 21, 7], dtype="timedelta64[D]"))
        timedeltas = np.array([0, 37, 24], dtype="timedelta64[h]")
        assert_equal(timedeltas.astype('timedelta64[D]'), np.array([0, 3, 2], dtype="timedelta64[D]"))

    def test_timedelta_astype_hours(self):
        timedeltas = np.array([0, 3, 2], dtype="timedelta64[D]")
        assert_equal(timedeltas.astype('timedelta64[h]'), np.array([0, 36, 24], dtype="timedelta64[D]"))
        timedeltas = np.array([0, 190, 153], dtype="timedelta64[m]")
        assert_equal(timedeltas.astype('timedelta64[h]'), np.array([0, 3, 2], dtype="timedelta64[h]"))

    def test_timedelta_astype_minutes(self):
        timedeltas = np.array([0, 3, 2], dtype="timedelta64[h]")
        assert_equal(timedeltas.astype('timedelta64[m]'), np.array([0, 180, 120], dtype="timedelta64[m]"))
        timedeltas = np.array([0, 190, 153], dtype="timedelta64[s]")
        assert_equal(timedeltas.astype('timedelta64[m]'), np.array([0, 3, 2], dtype="timedelta64[m]"))

    def test_timedelta_astype_seconds(self):
        timedeltas = np.array([0, 3, 2], dtype="timedelta64[m]")
        assert_equal(timedeltas.astype('timedelta64[s]'), np.array([0, 180, 120], dtype="timedelta64[s]"))
        timedeltas = np.array([0, 3200, 2430], dtype="timedelta64[ms]")
        assert_equal(timedeltas.astype('timedelta64[s]'), np.array([0, 3, 2], dtype="timedelta64[s]"))

    def test_timedelta_astype_milliseconds(self):
        timedeltas = np.array([0, 3, 2], dtype="timedelta64[s]")
        assert_equal(timedeltas.astype('timedelta64[ms]'), np.array([0, 3000, 2000], dtype="timedelta64[ms]"))
        timedeltas = np.array([0, 3200, 2430], dtype="timedelta64[us]")
        assert_equal(timedeltas.astype('timedelta64[ms]'), np.array([0, 3, 2], dtype="timedelta64[ms]"))

    def test_timedelta_astype_microseconds(self):
        timedeltas = np.array([0, 3, 2], dtype="timedelta64[ms]")
        assert_equal(timedeltas.astype('timedelta64[us]'), np.array([0, 3000, 2000], dtype="timedelta64[us]"))
        timedeltas = np.array([0, 3200, 2430], dtype="timedelta64[ns]")
        assert_equal(timedeltas.astype('timedelta64[us]'), np.array([0, 3, 2], dtype="timedelta64[us]"))

    def test_timedelta_astype_nanoseconds(self):
        timedeltas = np.array([0, 3, 2], dtype="timedelta64[us]")
        assert_equal(timedeltas.astype('timedelta64[ns]'), np.array([0, 3000, 2000], dtype="timedelta64[ns]"))
        timedeltas = np.array([0, 3200, 2430], dtype="timedelta64[ps]")
        assert_equal(timedeltas.astype('timedelta64[ns]'), np.array([0, 3, 2], dtype="timedelta64[ns]"))

    def test_timedelta_astype_picoseconds(self):
        timedeltas = np.array([0, 3, 2], dtype="timedelta64[ns]")
        assert_equal(timedeltas.astype('timedelta64[ps]'), np.array([0, 3000, 2000], dtype="timedelta64[ps]"))
        timedeltas = np.array([0, 3200, 2430], dtype="timedelta64[ns]")
        assert_equal(timedeltas.astype('timedelta64[ps]'), np.array([0, 3, 2], dtype="timedelta64[ps]"))

    def test_timedelta_astype_femtoseconds(self):
        timedeltas = np.array([0, 3, 2], dtype="timedelta64[ps]")
        assert_equal(timedeltas.astype('timedelta64[fs]'), np.array([0, 3000, 2000], dtype="timedelta64[fs]"))
        timedeltas = np.array([0, 3200, 2430], dtype="timedelta64[ps]")
        assert_equal(timedeltas.astype('timedelta64[fs]'), np.array([0, 3, 2], dtype="timedelta64[fs]"))

    def test_timedelta_astype_attoseconds(self):
        timedeltas = np.array([0, 3, 2], dtype="timedelta64[fs]")
        assert_equal(timedeltas.astype('timedelta64[as]'), np.array([0, 3000, 2000], dtype="datetime64[as]"))


if __name__ == "__main__":
    run_module_suite()
