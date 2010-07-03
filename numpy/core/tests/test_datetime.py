from os import path
import numpy as np
from numpy.testing import *
import datetime

class TestDateTime(TestCase):
    def test_creation(self):
        for unit in ['Y', 'M', 'W', 'B', 'D',
                     'h', 'm', 's', 'ms', 'us',
                     'ns', 'ps', 'fs', 'as']:
            dt1 = np.dtype('M8[750%s]'%unit)
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
    # TODO: Add checks for B, ms, us, ns, ps, fs, as
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


if __name__ == "__main__":
    run_module_suite()
