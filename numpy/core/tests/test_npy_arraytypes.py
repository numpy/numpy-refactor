import sys
from numpy.testing import *
import numpy as np
#from datetime import timedelta

types = [np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc,
         np.int_, np.uint, np.longlong, np.ulonglong,
         np.single, np.double, np.longdouble, np.csingle,
         np.cdouble, np.clongdouble]

alltypes = list( types )
alltypes.append( np.datetime64 )
alltypes.append( np.timedelta64 )

class TestArrayTypes(TestCase):

    def test_argmax( self ):

        x = np.array( [False, False, True, False], dtype=np.bool )

        assert x.argmax() == 2, "Broken array.argmax on np.bool"

        # Experience shows that this sequence seems to produce a random
        # result with numpy circa July 2010.  Seems to be something with the
        # argmax function itself, as unicode string comparison seems to work
        # correctly in Python.
        a = np.array( [u'aaa', u'aa', u'bbb'] )

        # u'aaa' > u'aa' and u'bbb' > u'aaa'  Hence, argmax == 2.
        assert a.argmax() == 2, "Broken array.argmax on unicode data."

    def test_argmax_numeric( self ):

        for k,t in enumerate( alltypes ):

            # No fill function for numpy.bool_, can't use arange().  I guess
            # this means bool.argmax() isn't gonna be tested by this code...
            if k == 0: continue

            a = np.arange( 5, dtype=t )
            assert a.argmax() == 4, "Broken array.argmax on type: " + t

    def test_nonzero_numeric_types( self ):

        for k,t in enumerate(types):

            a = np.array( [ t(1) ] )

            assert a, "Broken array.nonzero on type: " + t

    def test_nonzero_string_types( self ):

        a = np.array( [ 'aaa' ] )
        assert a, "Broken array.nonzero on string elements."

        a = np.array( [ u'aaa' ] )
        assert a, "Broken array.nonzero on Unicode elements."

    def test_nonzero_b( self ):

        td = np.timedelta64( 22 )
        atd = np.array( [td] )
        assert atd, "Broken array.nonzero on numpy.timedelta64 elements."

    def test_compare( self ):
        # Light bulb!  argmax doesn't call compare() for numeric/logical
        # types.  It only does that for string types.  Duh.

        pass

    def test_copyswap( self ):

        for k,t in enumerate(types):

            # np.bool_ causes some trouble with this test.
            #if k == 0 or k == 10: continue
            if k == 0: continue

            x = np.arange( 10, dtype=t )
            # This should exeercise <typoe>_copyswap
            x[::2].fill( t(2) )

            assert_equal( x, [2,1,2,3,2,5,2,7,2,9] )

    def test_copyswap_misc( self ):

        x = np.array( [ u'a', u'b', u'c' ] )
        x[::2].fill( u'd' )
        assert_equal( x, [u'd', u'b', u'd'] )

    def test_compare( self ):

        for k,t in enumerate(types):

            if k == 0: continue

            try:
                a = np.arange( 10, dtype=t )
                keys = a[::2]
                b = a.searchsorted( keys )
                c = a.copy()
                np.insert( c, b, b.astype( t ) )
                c.sort()
                assert_equal( c, a )

            except TypeError, e:
                print "Trouble with type %d:" % k, e

    def test_copyswapn( self ):

        for k,t in enumerate(types):

            # Skip troublesome types.
            if k == 0 or k == 10: continue

            x = np.arange( 10, dtype=t )
            y = x.byteswap()
            z = y.byteswap()

            assert_equal( z, x )

    def test_copyswapn_misctypes( self ):

        x = np.arange( 10, dtype=np.timedelta64 )
        y = x.byteswap()
        z = y.byteswap()

        assert_equal( z, x )

        x = np.array( [ u'aaa', u'bbb' ] )
        y = x.byteswap()
        z = y.byteswap()

        assert_equal( z, x )

    def test_fill( self ):

        z = np.byte(0)
        a = np.array( [z,z,z,z], dtype=np.byte )
        b2 = np.byte(2)
        # Why doesn't this execute BYTE_fill?
        a.fill( b2 )

        for i in range(4):
            assert a[i] == 2, "Problem with array.fill on byte elements."

    def test_dot( self ):

        a = np.array( [1,2,3], dtype=np.byte )
        assert a.dot(a) == 14, "Problem with dot product on byte array."

        a = np.array( [False, True], np.bool )
        assert_equal( a.clip(False,False), [False, False] )

    def test_array_casting( self ):

        for k,t in enumerate( alltypes ):

            a = np.array( [ t(1) ] )

            for k2, t2 in enumerate( alltypes ):

                b = a.astype( t2 )

                if k2 < len(types):
                    assert b[0] == 1, \
                           "Busted array type casting: k=%d k2=%d" % (k,k2)

                else:
                    # Casting to datetime64 yields a 1/1/1970+... result,
                    # which isn't so hot for checking against "1".  So, in
                    # these cases, just cast back to the starting time, and
                    # make sure we got back what we started with.
                    c = b.astype( t )
                    assert_equal( c, a )

    def xtest_array_casting_special( self ):

        a = np.array( [ np.datetime64(1) ] )
        for k,t in enumerate( types ):
            b = a.astype( t )

            assert b[0] == 1, "Busted casting from datetime to %d" % k

        a = np.array( [ np.timedelta64(1) ] )
        for k,t in enumerate( types ):
            b = a.astype( t )

            assert b[0] == 1, "Busted casting from datetime to %d" % k

    def test_take( self ):
        a = np.arange( 10, dtype=np.timedelta64 )
        idx = np.arange(5) * 2
        c = np.take( a, idx )
        assert_equal( c, a[::2] )

if __name__ == "__main__":
    run_module_suite()
