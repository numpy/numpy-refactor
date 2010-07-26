import sys
from numpy.testing import *
import numpy as np

types = [np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc,
         np.int_, np.uint, np.longlong, np.ulonglong,
         np.single, np.double, np.longdouble, np.csingle,
         np.cdouble, np.clongdouble]

# This compares scalarmath against ufuncs.

class TestTypes(TestCase):
    def test_types(self, level=1):
        for atype in types:
            a = atype(1)
            assert a == 1, "error with %r: got %r" % (atype,a)

    def test_type_add(self, level=1):
        # list of types
        for k, atype in enumerate(types):
            vala = atype(3)
            val1 = np.array([3],dtype=atype)
            for l, btype in enumerate(types):
                valb = btype(1)
                val2 = np.array([1],dtype=btype)
                val = vala + valb
                valo = val1 + val2
                assert val.dtype.num == valo.dtype.num and \
                       val.dtype.char == valo.dtype.char, \
                       "error with (%d,%d)" % (k,l)

    def test_type_subtract(self, level=1):
        # list of types
        for k, atype in enumerate(types):
            vala = atype(3)
            val1 = np.array([3],dtype=atype)
            for l, btype in enumerate(types):
                valb = btype(1)
                val2 = np.array([1],dtype=btype)
                val = vala - valb
                valo = val1 - val2
                assert val.dtype.num == valo.dtype.num and \
                       val.dtype.char == valo.dtype.char, \
                       "error with (%d,%d)" % (k,l)

    def test_type_multiply(self, level=1):
        # list of types
        for k, atype in enumerate(types):
            vala = atype(3)
            val1 = np.array([3],dtype=atype)
            for l, btype in enumerate(types):
                valb = btype(1)
                val2 = np.array([1],dtype=btype)
                val = vala * valb
                valo = val1 * val2
                assert val.dtype.num == valo.dtype.num and \
                       val.dtype.char == valo.dtype.char, \
                       "error with (%d,%d)" % (k,l)

    def test_type_divide(self, level=1):
        # Choose more interesting operands for this operation.
        # list of types
        for k, atype in enumerate(types):
            vala = atype(6)
            val1 = np.array([6],dtype=atype)
            for l, btype in enumerate(types):
                valb = btype(2)
                val2 = np.array([2],dtype=btype)
                val = vala / valb
                valo = val1 / val2
                assert val.dtype.num == valo.dtype.num and \
                       val.dtype.char == valo.dtype.char, \
                       "error with (%d,%d)" % (k,l)

    def test_type_remainder(self, level=1):
        # Choose more interesting operands for this operation.
        # list of types
        for k, atype in enumerate(types):
            vala = atype(6)
            val1 = np.array([6],dtype=atype)
            for l, btype in enumerate(types):
                valb = btype(2)
                val2 = np.array([2],dtype=btype)

                try:
                    val = vala % valb
                    valo = val1 % val2
                except TypeError, e:
                    # Some combos just don't work, like byte % complex.  We
                    # just don't worry about classifying the cases here, and
                    # instead just ignore these types of problems.  <grin>
                    pass

                assert val.dtype.num == valo.dtype.num and \
                       val.dtype.char == valo.dtype.char, \
                       "error with (%d,%d)" % (k,l)

    def test_type_negative(self, level=1):
        # Uhh, "negate" ???  Or maybe "unary minus".

        # But shouldn't this fail for unsigned types?  Hmmm...

        # list of types
        # NOTE: unary operators don't require the double loop over types,
        # since there's only one operand.
        for k, atype in enumerate(types):
            vala = atype(3)
            val1 = np.array([3],dtype=atype)

            val = -vala
            valo = -val1

            assert val.dtype.num == valo.dtype.num and \
                   val.dtype.char == valo.dtype.char, \
                   "error with (%d)" % (k)

    def test_type_positive(self, level=1):
        # Otherwise known as "unary plus".
        # list of types
        # NOTE: unary operators don't require the double loop over types,
        # since there's only one operand.
        for k, atype in enumerate(types):
            vala = atype(3)
            val1 = np.array([3],dtype=atype)

            val = +vala
            valo = +val1

            assert val.dtype.num == valo.dtype.num and \
                   val.dtype.char == valo.dtype.char, \
                   "error with (%d)" % (k)

    def test_type_power(self, level=1):
        # Choose more interesting operands for this operation.
        # list of types
        for k, atype in enumerate(types):
            vala = atype(2)
            val1 = np.array([2],dtype=atype)

            # Skip the boolean types
            if vala.dtype.char == '?': continue

            for l, btype in enumerate(types):
                valb = btype(3)
                val2 = np.array([3],dtype=btype)

                # Skip the boolean types
                if valb.dtype.char == '?': continue

                val = vala ** valb
                valo = val1 ** val2
                assert val.dtype.num == valo.dtype.num and \
                       val.dtype.char == valo.dtype.char, \
                       "error with (%d,%d)" % (k,l)

    def test_type_absolute(self, level=1):
        # list of types
        for k, atype in enumerate(types):
            vala = atype(-3)
            val1 = np.array([-3],dtype=atype)

            val = abs(vala)
            valo = abs(val1)

            assert val.dtype.num == valo.dtype.num and \
                   val.dtype.char == valo.dtype.char, \
                   "error with (%d)" % (k)

            # I guess we can't really test for the right result here, unless
            # we can figure out how to exclude the unsigned types.
            #assert val == atype(3) and valo == atype(3), \
            #       "error taking absolute value (%d)." % k

    def test_type_hex(self, level=1):
        # list of types
        for k, atype in enumerate(types):
            vala = atype(3)
            val1 = np.array([3],dtype=atype)

            try:
                val = hex(vala)
                valo = hex(val1)

            except:
                #print "Can't hexify ", k
                pass

            #assert val.dtype.num == valo.dtype.num and \
            #       val.dtype.char == valo.dtype.char, \
            #       "error with (%d)" % (k)

            # We can't demand equivalent repr's either.
            #assert val == valo, "Trouble with hex (%d)" % k

            # So there's not really so much we can check here, beyond simply
            # that the code executes without throwing exceptions.

    def test_type_float(self, level=1):
        # list of types
        for k, atype in enumerate(types):
            vala = atype(3)
            val1 = np.array([3],dtype=atype)

            try:
                val = float(vala)
                valo = float(val1)

            except TypeError, e:
                # The complex type, for example, can't be cast to float, so
                # just skip it.
                continue

            assert val == valo, "Trouble with float (%d)" % k

            # Skip over bool.
            if vala.dtype.char == '?': continue

            assert val == 3 and valo == 3, "Trouble with float (%d)" % k

    def test_misc_niggles(self, level=1):

        # Verify the nonzero method on longdouble and clongdouble types.
        # This is done essentially by evaluating an apprpriately typed
        # (number) object as a condition.  I guess that's probably done
        # elsewhere in the test suite for the other interesting data types.

        x = np.longdouble( 4.4 )
        y = np.nonzero( x )

        assert x, "Trouble with longdouble_nonzero"

        z = np.clongdouble( 4 + 5j )

        assert z, "Trouble with clongdouble_nonzero"

        from operator import itruediv
        itruediv( z, x )

        q = int(z)

        divmod( x, 1.1 )

        r = np.nonzero( z )

        s = np.longlong( 99 )
        t = int(s)

    def xtest_scalarmath_module_methods( self, level=1 ):
        # Rename to test_* when ready.

        # The purpose of this method is to actually exercise the scalarmath
        # mdoule's module-methods, whose names are:
        #    use_scalarmath
        #    use_pythonmath
        #    alter_pyscalars
        #    restore_pyscalars
        # Those module methods need to be exercised.

        pass

    def test_type_create(self, level=1):
        for k, atype in enumerate(types):
            a = np.array([1,2,3],atype)
            b = atype([1,2,3])
            assert_equal(a,b)


class TestPower(TestCase):
    def test_small_types(self):
        for t in [np.int8, np.int16]:
            a = t(3)
            b = a ** 4
            assert b == 81, "error with %r: got %r" % (t,b)

    def test_large_types(self):
        for t in [np.int32, np.int64, np.float32, np.float64, np.longdouble]:
            a = t(51)
            b = a ** 4
            msg = "error with %r: got %r" % (t,b)
            if np.issubdtype(t, np.integer):
                assert b == 6765201, msg
            else:
                assert_almost_equal(b, 6765201, err_msg=msg)


class TestConversion(TestCase):
    def test_int_from_long(self):
        l = [1e6, 1e12, 1e18, -1e6, -1e12, -1e18]
        li = [10**6, 10**12, 10**18, -10**6, -10**12, -10**18]
        for T in [None, np.float64, np.int64]:
            a = np.array(l,dtype=T)
            assert_equal(map(int,a), li)

        a = np.array(l[:3], dtype=np.uint64)
        assert_equal(map(int,a), li[:3])


#class TestRepr(TestCase):
#    def test_repr(self):
#        for t in types:
#            val = t(1197346475.0137341)
#            val_repr = repr(val)
#            val2 = eval(val_repr)
#            assert_equal( val, val2 )


class TestRepr(TestCase):
    def _test_type_repr(self, t):
        finfo=np.finfo(t)
        last_fraction_bit_idx = finfo.nexp + finfo.nmant
        last_exponent_bit_idx = finfo.nexp
        storage_bytes = np.dtype(t).itemsize*8
        # could add some more types to the list below
        for which in ['small denorm','small norm']:
            # Values from http://en.wikipedia.org/wiki/IEEE_754
            constr = np.array([0x00]*storage_bytes,dtype=np.uint8)
            if which == 'small denorm':
                byte = last_fraction_bit_idx // 8
                bytebit = 7-(last_fraction_bit_idx % 8)
                constr[byte] = 1<<bytebit
            elif which == 'small norm':
                byte = last_exponent_bit_idx // 8
                bytebit = 7-(last_exponent_bit_idx % 8)
                constr[byte] = 1<<bytebit
            else:
                raise ValueError('hmm')
            val = constr.view(t)[0]
            val_repr = repr(val)
            val2 = t(eval(val_repr))
            if not (val2 == 0 and val < 1e-100):
                assert_equal(val, val2)

    def test_float_repr(self):
        # long double test cannot work, because eval goes through a python
        # float
        for t in [np.float32, np.float64]:
            yield test_float_repr, t

if __name__ == "__main__":
    run_module_suite()
