import sys
from numpy.testing import *
import numpy as np
#from datetime import timedelta

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

    def test_argmax_float( self ):
        # Moved to a separate function since test_argmax() (above) asserts
        # due to the broken argmas on unicode elements.
        a = np.arange( 5, dtype=np.float32 )
        assert a.argmax() == 4, "Broken array.argmax on float"

    def test_nonzero( self ):

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

    def test_copy( self ):

        a = np.arange( 5, dtype=np.cdouble )
        b = a.copy()
        #assert b, "xxx"

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

        a = np.arange( 5, dtype=np.cfloat )
        b = a.astype( bool )
        c = a.astype( np.bool )

if __name__ == "__main__":
    run_module_suite()
