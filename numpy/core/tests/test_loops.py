# This is to test things in numpy/core/src/umath/loops.c

import sys
from numpy.testing import *
import numpy as np

types = [np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc,
         np.int_, np.uint, np.longlong, np.ulonglong,
         np.single, np.double, np.longdouble, np.csingle,
         np.cdouble, np.clongdouble]

alltypes = types + [ np.datetime64, np.timedelta64 ]

int_types = [ np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc,
              np.int_, np.uint, np.longlong, np.ulonglong ]

rc_types = [ np.single,  np.double,  np.longdouble,
             np.csingle, np.cdouble, np.clongdouble ]

class TestLoops(TestCase):

    def test_conjugate_reals( self ):

        rtypes = [ np.byte, np.ubyte,
                   np.short, np.ushort,
                   np.intc, np.uintc,
                   np.int_, np.uint,
                   np.longlong, np.ulonglong,
                   np.single, np.double, np.longdouble ]

        for t in rtypes:
            self.exercise_conjugate_real_t( t )

    def exercise_conjugate_real_t( self, tp ):

        a = np.arange( 5, dtype=tp )
        b = np.conjugate( a )
        # conjugate has no effect on a pure real number.
        assert_equal( b, a )

    # Still need to test conjugate on cfloat, cdouble and clongdouble

    def test_logical_ops( self ):

        for t in alltypes[1:]:
            self.exercise_greater_t( t )
            self.exercise_greater_equal_t( t )
            self.exercise_less_t( t )
            self.exercise_less_equal_t( t )
            self.exercise_logical_and_t( t )
            self.exercise_logical_or_t( t )
            self.exercise_logical_not_t( t )

    def exercise_greater_t( self, tp ):

        a = np.arange( 5, dtype=tp )
        b = a > 2
        assert_equal( b, [False, False, False, True, True] )

    def exercise_greater_equal_t( self, tp ):

        a = np.arange( 5, dtype=tp )
        b = a >= 2
        assert_equal( b, [False, False, True, True, True] )

    def exercise_less_t( self, tp ):

        a = np.arange( 5, dtype=tp )
        b = a < 2
        assert_equal( b, [True, True, False, False, False] )

    def exercise_less_equal_t( self, tp ):

        a = np.arange( 5, dtype=tp )
        b = a <= 2
        assert_equal( b, [True, True, True, False, False] )

    def exercise_logical_and_t( self, tp ):

        a = np.arange( 5, dtype=tp )
        b = np.ones( 5, dtype=tp )
        assert_equal( np.logical_and(a,b),
                      [False, True, True, True, True] )

    def exercise_logical_or_t( self, tp ):

        a = np.arange( 5, dtype=tp )
        b = np.ones( 5, dtype=tp )
        assert_equal( np.logical_or(a,b),
                      [True, True, True, True, True] )

    def exercise_logical_not_t( self, tp ):

        a = np.arange( 5, dtype=tp )
        b = np.logical_not( a )
        assert_equal( b, [True, False, False, False, False] )

    def test_reciprocal( self ):

        for t in int_types:
            self.exercise_reciprocal_int_t( t )

        for t in rc_types:
            self.exercise_reciprocal_rc_t( t )

    def exercise_reciprocal_int_t( self, tp ):

        a = np.arange( 1, 5, dtype=tp )
        b = np.reciprocal( a )
        assert_equal( b, [1, 0, 0, 0] )

    def exercise_reciprocal_rc_t( self, tp ):

        a = np.arange( 1, 5, dtype=tp )
        b = np.reciprocal( a )
        c = np.reciprocal( b )
        assert_equal( c, a )

    def test_BOOL_ops( self ):

        a = np.array( [False, True] )

        b = a >= True
        assert_equal( b, [False, True] )

        b = a > False
        assert_equal( b, [False, True] )

        b = a <= False
        assert_equal( b, [True, False] )

        b = a < True
        assert_equal( b, [True, False] )

        # This runs the BOOL_to_UNICODE function, but we can't convert back
        # to check that the transformation is reversible.
        a.astype( np.unicode )

        b = a != False
        assert_equal( b, [False, True] )

        x = np.array( [True, False] )
        c = np.maximum( a, x )
        d = np.minimum( a, x )

        assert_equal( c, [True, True] )
        assert_equal( d, [False, False] )

        z = np.ones_like( x )
        assert z.shape == x.shape, "BOOL_ones_like botched the shape."

        # Casting a bool array to np.void throws an exception.  Is that
        # expected?

        # BOOL_scan requires writing a file I/O test.  Deferred for now.

    def test_invert( self ):

        for t in int_types:
            a = np.arange( 5, dtype=t )
            assert_equal( np.invert( np.invert( a ) ), a )

    def test_math_ops( self ):

        for t in types[1:]:
            self.exercise_floor_divide_t( t )

        for t in alltypes[1:]:
            self.exercise_maximum_t( t )

    def exercise_floor_divide_t( self, tp ):

        a = np.arange( 5, dtype=tp )
        assert_equal( np.floor_divide( a, 2 ), [0, 0, 1, 1, 2] )

    def exercise_maximum_t( self, tp ):

        a = np.arange( 5, dtype=tp )
        b = np.empty( 5, dtype=tp )
        b.fill( tp(2) )
        c = np.maximum( a, b )

        assert_equal( c, np.array( [2,2,2,3,4], dtype=tp ) )

if __name__ == "__main__":
    run_module_suite()
