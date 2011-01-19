using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Reflection;
using System.Numerics;
using IronPython.Modules;
using IronPython.Runtime;
using IronPython.Runtime.Operations;
using IronPython.Runtime.Types;
using IronPython.Runtime.Exceptions;
using Microsoft.Scripting;
using Microsoft.Scripting.Runtime;


namespace NumpyDotNet
{
    /// <summary>
    /// Implements the Numpy python 'ndarray' object and acts as an interface to
    /// the core NpyArray data structure.  Npy_INTERFACE(NpyArray *) points an
    /// instance of this class.
    /// </summary>
    [PythonType]
    public partial class ndarray : Wrapper, IEnumerable<object>, IBufferProvider, NumpyDotNet.IArray
    {
        public const string __module__ = "numpy";

        public ndarray() {
        }

        public static ndarray __new__(CodeContext cntx, PythonType cls,
                              object shape, object dtype = null,
                              object buffer = null, object offset = null,
                              object strides = null, object order = null) {
            ndarray result = (ndarray)ObjectOps.__new__(cntx, cls);
            result.Construct(cntx, shape, dtype, buffer, offset, strides, order);
            return result;
        }

        internal void Construct(CodeContext cntx, object shape, object dtype = null,
                              object buffer = null, object offset = null,
                              object strides = null, object order = null) {
            dtype type = null;

            core = IntPtr.Zero;

            long[] aShape = NpyUtil_ArgProcessing.IntArrConverter(shape);
            if (dtype != null) {
                type = NpyDescr.DescrConverter(cntx, dtype);
            }

            if (buffer != null)
                throw new NotImplementedException("Buffer support is not implemented.");
            long loffset = NpyUtil_ArgProcessing.IntConverter(offset);
            long[] aStrides = NpyUtil_ArgProcessing.IntArrConverter(strides);
            NpyDefs.NPY_ORDER eOrder = NpyUtil_ArgProcessing.OrderConverter(order);

            if (type == null)
                type = NpyCoreApi.DescrFromType(NpyDefs.DefaultType);

            int itemsize = type.ElementSize;
            if (itemsize == 0) {
                throw new ArgumentException("data-type with unspecified variable length");
            }

            if (aStrides != null) {
                if (aStrides.Length != aShape.Length) {
                    throw new ArgumentException("strides, if given, must be the same length as shape");
                }

                if (!NpyArray.CheckStrides(itemsize, aShape, aStrides)) {
                    throw new ArgumentException("strides is compatible with shape of requested array and size of buffer");
                }
            }

            // Creates a new array object.  By passing 'this' in the current instance
            // becomes the wrapper object for the new array.
            ndarray wrap = NpyCoreApi.NewFromDescr(type, aShape, aStrides, 0,
                new NpyCoreApi.UseExistingWrapper { Wrapper = this });
            if (wrap != this) {
                throw new InvalidOperationException("Internal error: returned array wrapper is different than current instance.");
            }
            // NOTE: CPython fills object arrays with Py_None here.  We don't
            // need to do this since None is null and the arrays are zero filled.
        }

        protected override void Dispose(bool disposing) {
            if (core != IntPtr.Zero) {
                lock (this) {
                    DecreaseMemoryPressure(this);
                    base.Dispose(disposing);
                }
            }
        }

        /// <summary>
        /// Danger!  This method is only intended to be used indirectly during construction
        /// when the new instance is passed into the core as the 'interfaceData' field so
        /// ArrayNewWrapper can pair up this instance with a core object.  If this pointer
        /// is changed after pairing, bad things can happen.
        /// </summary>
        /// <param name="a">Core object to be paired with this wrapper</param>
        internal void SetArray(IntPtr a) {
            if (core == null) {
                throw new InvalidOperationException("Attempt to change core array object for already-constructed wrapper.");
            }
            core = a;
        }


        #region Public interfaces (must match CPython)

        private static Func<ndarray, string> reprFunction;
        private static Func<ndarray, string> strFunction;

        /// <summary>
        /// Sets a function to be triggered for the repr() operator or null to default to the
        /// built-in version.
        /// </summary>
        public static Func<ndarray, string> ReprFunction {
            get { return reprFunction; }
            internal set { reprFunction = (value != null) ? value : x => x.BuildStringRepr(true); }
        }

        /// <summary>
        /// Sets a function to be triggered on the str() operator or ToString() method. Null defaults to
        /// the built-in version.
        /// </summary>
        public static Func<ndarray, string> StrFunction {
            get { return strFunction; }
            internal set { strFunction = (value != null) ? value : x => x.BuildStringRepr(false); }
        }

        static ndarray() {
            ReprFunction = null;
            StrFunction = null;
        }

        #region Python methods

        public virtual string __repr__(CodeContext cntx) {
            return ReprFunction(this);
        }

        public virtual string __str__(CodeContext cntx) {
            return StrFunction(this);
        }

        public virtual object __reduce__(CodeContext cntx, object notused=null) {
            const int version = 1;

            // Result is a tuple of (callable object, arguments, object's state).
            object[] ret = new object[3];
            ret[0] = NpyUtil_Python.GetModuleAttr(cntx, "numpy.core.multiarray", "_reconstruct");
            if (ret[0] == null) return null;

            ret[1] = PythonOps.MakeTuple(DynamicHelpers.GetPythonType(this), PythonOps.MakeTuple(0), "b");

            // Fill in the object's state.  This is a tuple with 5 argumentS:
            //    1) an integer with the pickle version
            //    2) a Tuple giving the shape
            //    3) a dtype object with the correct byteorder set
            //    4) a Bool stating if Fortran or not
            //    5) a Python object representing the data (a string or list or something)
            object[] state = new object[5];
            state[0] = version;
            state[1] = this.shape;
            state[2] = this.Dtype;
            state[3] = this.IsFortran;
            state[4] = Dtype.ChkFlags(NpyDefs.NPY_LIST_PICKLE) ? GetPickleList() : ToBytes();

            ret[2] = new PythonTuple(state);
            return new PythonTuple(ret);
        }


        /// <summary>
        /// Generates a string containing the byte representation of the array.  This is quite
        /// inefficient as the string (being 16-bit unicode) is twice the size needed, but this
        /// is what the pickler uses. Ugh.
        /// </summary>
        /// <param name="order">Desired output order, default is array's current order</param>
        /// <returns>String containing data bytes</returns>
        private String ToBytes(NpyDefs.NPY_ORDER order = NpyDefs.NPY_ORDER.NPY_ANYORDER) {
            if (order == NpyDefs.NPY_ORDER.NPY_ANYORDER) {
                order = IsFortran ? NpyDefs.NPY_ORDER.NPY_FORTRANORDER : NpyDefs.NPY_ORDER.NPY_CORDER;
            }

            long size = itemsize * Size;
            if (size >= Int32.MaxValue) {
                throw new NotImplementedException("Total array size exceeds 2GB limit imposed by .NET string size, unable to pickle array.");
            }

            string result;
            if (IsContiguous && order == NpyDefs.NPY_ORDER.NPY_CORDER ||
                IsFortran && order == NpyDefs.NPY_ORDER.NPY_FORTRANORDER) {
                unsafe {
                    result = new string((sbyte*)UnsafeAddress, 0, (int)size);
                }
            } else {
                // TODO: Implementation requires some thought to implement to try to avoid making multiple copies of
                // the data.  The issue is that we have to return a string.  We can allocate a string of the appropriate
                // size, but it is immutable.  StringBuilder works, but we end up copying. Can do it in C, but end up
                // copying in C, then copy into String. Ugh.
                throw new NotImplementedException("Pickling of non-contiguous arrays or transposing arrays is not supported");
            }
            return result;
        }

        private object GetPickleList() {
            List list = new List();
            for (flatiter iter = this.Flat; iter.MoveNext(); list.append(iter.Current)) ;
            return list;
        }

        public virtual object __setstate__(PythonTuple t) {
            if (t.Count == 4) {
                return __setstate__(0, (PythonTuple)t[0], (dtype)t[1], t[2], t[3]);
            } else if (t.Count == 5) {
                return __setstate__((int)t[0], (PythonTuple)t[1], (dtype)t[2], t[3], t[4]);
            } else {
                throw new NotImplementedException(
                    String.Format("Unhandled pickle format with {0} arguments.", t.Count));
            }
        }

        public virtual object __setstate__(PythonTuple shape, dtype typecode, object fortran, object rawdata) {
            return __setstate__(0, shape, typecode, fortran, rawdata);
        }

        public virtual object __setstate__(int version, PythonTuple shape, dtype typecode, object fortran, object rawData) {
            bool fortranFlag = NpyUtil_ArgProcessing.BoolConverter(fortran);

            if (version != 1 && version != 0) {
                throw new ArgumentException(
                    String.Format("can't handle version {0} of numpy.ndarray pickle.", version));
            }

            IntPtr[] dimensions = NpyUtil_ArgProcessing.IntpArrConverter(shape);
            int nd = dimensions.Length;
            long size = dimensions.Aggregate(1L, (x, y) => x * (long)y);

            if (nd < 1) {
                return null;
            }
            if (typecode.ElementSize == 0) {
                throw new ArgumentException("Invalid data-type size");
            }
            if (size < 0 || size > Int64.MaxValue / typecode.ElementSize) {
                throw new InsufficientMemoryException();
            }

            if (typecode.ChkFlags(NpyDefs.NPY_LIST_PICKLE)) {
                if (!(rawData is List)) {
                    throw new ArgumentTypeException("object pickle not returning list");
                }
            } else {
                if (!(rawData is string)) {
                    throw new ArgumentTypeException("pickle not returning string");
                }
                if (((string)rawData).Length != typecode.itemsize * size) {
                    throw new ArgumentException("buffer size does not match array size");
                }
            }

            // Set the state of this array using the passed in data.  Everything in this array goes away.
            // The .SetState method resizes/reallocated the data memory.
            this.Dtype = typecode;
            NpyCoreApi.SetState(this, dimensions, fortranFlag ? NpyDefs.NPY_ORDER.NPY_FORTRANORDER : NpyDefs.NPY_ORDER.NPY_CORDER,
                rawData as string);

            if (rawData is List) {
                flatiter iter = NpyCoreApi.IterNew(this);
                foreach (object o in (List)rawData) {
                    if (!iter.MoveNext()) {
                        break;
                    }
                    iter.Current = o;
                }
            }
            return null;
        }


        /// <summary>
        /// Returns the length of dimension zero of the array
        /// </summary>
        /// <returns>Length of the first dimension</returns>
        public virtual object __len__() {
            if (ndim == 0) {
                throw new ArgumentTypeException("len() of unsized object");
            }
            return PythonOps.ToPython((IntPtr)Dims[0]);
        }

        public object __abs__(CodeContext cntx) {
            return UnaryOp(cntx,  this, NpyDefs.NpyArray_Ops.npy_op_absolute);
        }

        public ndarray __array__(CodeContext cntx, object descr = null) {
            dtype newtype = null;
            ndarray result;

            if (descr != null) {
                newtype = NpyDescr.DescrConverter(cntx, descr);
            }
            if (GetType() != typeof(ndarray)) {
                result = NpyCoreApi.FromArray(this, Dtype, NpyDefs.NPY_ENSUREARRAY);
            } else {
                result = this;
            }
            if (newtype == null || newtype == result.Dtype) {
                return result;
            } else {
                return NpyCoreApi.CastToType(result, newtype, false);
            }
        }

        public ndarray __array_prepare__(ndarray a, params object[] args) {
            return NpyCoreApi.ViewLike(a, this);
        }

        public ndarray __array_wrap__(ndarray a) {
            if (GetType() == a.GetType()) {
                return a;
            } else {
                return NpyCoreApi.ViewLike(a, this);
            }
        }

        public object __divmod__(CodeContext cntx, Object b) {
            return PythonOps.MakeTuple(
                BinaryOp(cntx, this, b, NpyDefs.NpyArray_Ops.npy_op_floor_divide),
                BinaryOp(cntx, this, b, NpyDefs.NpyArray_Ops.npy_op_remainder));
        }

        public object __rdivmod__(CodeContext cntx, Object a) {
            return PythonOps.MakeTuple(
                BinaryOp(cntx, a, this, NpyDefs.NpyArray_Ops.npy_op_floor_divide),
                BinaryOp(cntx, a, this, NpyDefs.NpyArray_Ops.npy_op_remainder));
        }

        public object __lshift__(CodeContext cntx, Object b) {
            return BinaryOp(cntx, this, b, NpyDefs.NpyArray_Ops.npy_op_left_shift);
        }

        public object __rlshift__(CodeContext cntx, Object a) {
            return BinaryOp(cntx, a, this, NpyDefs.NpyArray_Ops.npy_op_left_shift);
        }

        public object __rshift__(CodeContext cntx, Object b) {
            return BinaryOp(cntx, this, b, NpyDefs.NpyArray_Ops.npy_op_right_shift);
        }

        public object __rrshift__(CodeContext cntx, Object a) {
            return BinaryOp(cntx, a, this, NpyDefs.NpyArray_Ops.npy_op_right_shift);
        }

        public object __sqrt__(CodeContext cntx) {
            return UnaryOp(cntx,  this, NpyDefs.NpyArray_Ops.npy_op_sqrt);
        }

        public object __mod__(CodeContext cntx, Object b) {
            return BinaryOp(cntx, this, b, "remainder");
        }

        public object __rmod__(CodeContext cntx, Object a) {
            return BinaryOp(cntx, a, this, "remainder");
        }

        #endregion

        #region Operators

        internal static object BinaryOp(CodeContext cntx, object a, object b, ufunc f, ndarray ret = null) {
            if (cntx == null) {
                cntx = NpyUtil_Python.DefaultContext;
            }
            try {
                object result;
                if (ret == null) {
                    result = f.Call(cntx, null, a, b);
                } else {
                    result = f.Call(cntx, null, a, b, ret);
                }
                if (result.GetType() == typeof(ndarray)) {
                    return ArrayReturn((ndarray)result);
                } else {
                    return result;
                }
            } catch (NotImplementedException) {
                return cntx.LanguageContext.BuiltinModuleDict["NotImplemented"];
            }
        }

        internal static object BinaryOp(CodeContext cntx, object a, object b,
                                        NpyDefs.NpyArray_Ops op, ndarray ret = null) {
            ufunc f = NpyCoreApi.GetNumericOp(op);
            return BinaryOp(cntx, a, b, f, ret);
        }

        internal static object BinaryOp(CodeContext cntx, object a, object b,
                                        string fname, ndarray ret = null) {
            ufunc f = ufunc.GetFunction(fname);
            return BinaryOp(cntx, a, b, f, ret);
        }


        internal static object UnaryOp(CodeContext cntx, object a, NpyDefs.NpyArray_Ops op,
                                       ndarray ret = null) {
            if (cntx == null) {
                cntx = NpyUtil_Python.DefaultContext;
            }
            ufunc f = NpyCoreApi.GetNumericOp(op);
            object result;
            if (ret == null) {
                result = f.Call(cntx, null, a);
            } else {
                result = f.Call(cntx, null, a, ret);
            }
            if (result is ndarray) {
                return ArrayReturn((ndarray)result);
            } else {
                return result;
            }
        }

        public static object operator +(ndarray a, Object b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_add);
        }

        public static object operator +(object a, ndarray b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_add);
        }

        public static object operator +(ndarray a, ndarray b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_add);
        }

        [SpecialName]
        public object InPlaceAdd(object b) {
            return BinaryOp(null, this, b, NpyDefs.NpyArray_Ops.npy_op_add, this);
        }

        [SpecialName]
        public object InPlaceAdd(ndarray b) {
            return BinaryOp(null, this, b, NpyDefs.NpyArray_Ops.npy_op_add, this);
        }

        public static object operator -(ndarray a, Object b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_subtract);
        }

        public static object operator -(object a, ndarray b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_subtract);
        }

        public static object operator -(ndarray a, ndarray b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_subtract);
        }

        [SpecialName]
        public object InPlaceSubtract(object b) {
            return BinaryOp(null, this, b, NpyDefs.NpyArray_Ops.npy_op_subtract, this);
        }

        [SpecialName]
        public object InPlaceSubtract(ndarray b) {
            return BinaryOp(null, this, b, NpyDefs.NpyArray_Ops.npy_op_subtract, this);
        }

        public static object operator -(ndarray a) {
            return UnaryOp(null, a, NpyDefs.NpyArray_Ops.npy_op_negative);
        }

        public static object operator *(ndarray a, Object b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_multiply);
        }

        public static object operator *(object a, ndarray b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_multiply);
        }

        public static object operator *(ndarray a, ndarray b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_multiply);
        }

        [SpecialName]
        public object InPlaceMultiply(object b) {
            return BinaryOp(null, this, b, NpyDefs.NpyArray_Ops.npy_op_multiply, this);
        }

        [SpecialName]
        public object InPlaceMultiply(ndarray b) {
            return BinaryOp(null, this, b, NpyDefs.NpyArray_Ops.npy_op_multiply, this);
        }

        public static object operator /(ndarray a, Object b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_divide);
        }

        public static object operator /(object a, ndarray b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_divide);
        }

        public static object operator /(ndarray a, ndarray b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_divide);
        }

        [SpecialName]
        public object InPlaceDivide(object b) {
            return BinaryOp(null, this, b, NpyDefs.NpyArray_Ops.npy_op_divide, this);
        }

        [SpecialName]
        public object InPlaceDivide(ndarray b) {
            return BinaryOp(null, this, b, NpyDefs.NpyArray_Ops.npy_op_divide, this);
        }

        public object __pow__(object a) {
            // TODO: Add optimizations for scalar powers
            return BinaryOp(null, this, a, NpyDefs.NpyArray_Ops.npy_op_power);
        }

        // TODO: Add inplace operators.

        public static object operator &(ndarray a, Object b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_and);
        }

        public static object operator &(object a, ndarray b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_and);
        }

        public static object operator &(ndarray a, ndarray b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_and);
        }

        [SpecialName]
        public object InPlaceBitwiseAnd(object b) {
            return BinaryOp(null, this, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_and, this);
        }

        [SpecialName]
        public object InPlaceBitwiseAnd(ndarray b) {
            return BinaryOp(null, this, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_and, this);
        }

        public static object operator |(ndarray a, Object b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_or);
        }

        public static object operator |(object a, ndarray b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_or);
        }

        public static object operator |(ndarray a, ndarray b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_or);
        }

        [SpecialName]
        public object InPlaceBitwiseOr(object b) {
            return BinaryOp(null, this, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_or, this);
        }

        [SpecialName]
        public object InPlaceBitwiseOr(ndarray b) {
            return BinaryOp(null, this, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_or, this);
        }

        public static object operator ^(ndarray a, Object b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_xor);
        }

        public static object operator ^(object a, ndarray b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_xor);
        }

        public static object operator ^(ndarray a, ndarray b) {
            return BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_xor);
        }

        [SpecialName]
        public object InPlaceExclusiveOr(object b) {
            return BinaryOp(null, this, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_xor, this);
        }

        [SpecialName]
        public object InPlaceExclusiveOr(ndarray b) {
            return BinaryOp(null, this, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_xor, this);
        }

        public static object operator ~(ndarray a) {
            return UnaryOp(null, a, NpyDefs.NpyArray_Ops.npy_op_invert);
        }

        public static implicit operator String(ndarray a) {
            return StrFunction(a);
        }

        // NOTE: For comparison operators we use the Python names
        // since these operators usually return boolean arrays and
        // .NET seems to expect them to return bool

        public object __eq__(CodeContext cntx, object o) {
            if (o == null) {
                return false;
            }
            NpyDefs.NPY_TYPES type = Dtype.TypeNum;
            ndarray arrayother = o as ndarray;
            if (arrayother == null) {
                // Try to convert to an array. Return not equal on failure
                try {
                    if (type != NpyDefs.NPY_TYPES.NPY_OBJECT) {
                        type = NpyDefs.NPY_TYPES.NPY_NOTYPE;
                    }
                    arrayother = NpyArray.FromAny(o, NpyCoreApi.DescrFromType(type), flags: NpyDefs.NPY_BEHAVED | NpyDefs.NPY_ENSUREARRAY);
                    if (arrayother == null) {
                        return false;
                    }
                } catch {
                    return false;
                }
            }

            object result = BinaryOp(cntx, this, arrayother, NpyDefs.NpyArray_Ops.npy_op_equal);
            if (result == Builtin.NotImplemented) {
                if (type == NpyDefs.NPY_TYPES.NPY_VOID) {
                    if (Dtype != arrayother.Dtype) {
                        return false;
                    }
                    if (Dtype.HasNames) {
                        object res = null;
                        foreach (string name in Dtype.Names) {
                            ndarray a1 = NpyArray.EnsureAnyArray(this[name]);
                            ndarray a2 = NpyArray.EnsureAnyArray(arrayother[name]);
                            object eq = a1.__eq__(cntx, a2);
                            if (res == null) {
                                res = eq;
                            } else {
                                res = BinaryOp(cntx, res, eq, NpyDefs.NpyArray_Ops.npy_op_logical_and);
                            }
                        }
                        if (res == null) {
                            throw new ArgumentException("No fields found");
                        }
                        return res;
                    }
                    result = NpyCoreApi.CompareStringArrays(this, arrayother, NpyDefs.NPY_COMPARE_OP.NPY_EQ);
                } else {
                    result = strings_compare(o, NpyDefs.NPY_COMPARE_OP.NPY_EQ);
                }
            }
            return result;
        }

        public object __req__(CodeContext cntx, object o) {
            return __eq__(cntx, o);
        }

        public object __ne__(CodeContext cntx, object o) {
            if (o == null) {
                return true;
            }
            NpyDefs.NPY_TYPES type = Dtype.TypeNum;
            ndarray arrayother = o as ndarray;
            if (arrayother == null) {
                // Try to convert to an array. Return not equal on failure
                try {
                    if (type == NpyDefs.NPY_TYPES.NPY_OBJECT) {
                        type = NpyDefs.NPY_TYPES.NPY_NOTYPE;
                    }
                    arrayother = NpyArray.FromAny(o, NpyCoreApi.DescrFromType(type), flags: NpyDefs.NPY_BEHAVED | NpyDefs.NPY_ENSUREARRAY);
                    if (arrayother == null) {
                        return true;
                    }
                } catch {
                    return true;
                }
            }

            object result = BinaryOp(cntx, this, arrayother, NpyDefs.NpyArray_Ops.npy_op_not_equal);
            if (result == Builtin.NotImplemented) {
                if (type == NpyDefs.NPY_TYPES.NPY_VOID) {
                    if (Dtype != arrayother.Dtype) {
                        return false;
                    }
                    if (Dtype.HasNames) {
                        object res = null;
                        foreach (string name in Dtype.Names) {
                            ndarray a1 = NpyArray.EnsureAnyArray(this[name]);
                            ndarray a2 = NpyArray.EnsureAnyArray(arrayother[name]);
                            object eq = a1.__ne__(cntx, a2);
                            if (res == null) {
                                res = eq;
                            } else {
                                res = BinaryOp(cntx, res, eq, NpyDefs.NpyArray_Ops.npy_op_logical_or);
                            }
                        }
                        if (res == null) {
                            throw new ArgumentException("No fields found");
                        }
                        return res;
                    }
                    result = NpyCoreApi.CompareStringArrays(this, arrayother, NpyDefs.NPY_COMPARE_OP.NPY_NE);
                } else {
                    result = strings_compare(o, NpyDefs.NPY_COMPARE_OP.NPY_NE);
                }
            }
            return result;
        }

        public object __rne__(CodeContext cntx, object o) {
            return __ne__(cntx, o);
        }

        public object __lt__(CodeContext cntx, object o) {
            object result = BinaryOp(cntx, this, o, NpyDefs.NpyArray_Ops.npy_op_less);
            if (result == Builtin.NotImplemented) {
                result = strings_compare(o, NpyDefs.NPY_COMPARE_OP.NPY_LT);
            }
            return result;
        }

        public object __rlt__(CodeContext cntx, object o) {
            return __ge__(cntx, o);
        }

        public object __le__(CodeContext cntx, object o) {
            object result = BinaryOp(cntx, this, o, NpyDefs.NpyArray_Ops.npy_op_less_equal);
            if (result == Builtin.NotImplemented) {
                result = strings_compare(o, NpyDefs.NPY_COMPARE_OP.NPY_LE);
            }
            return result;
        }

        public object __rle__(CodeContext cntx, object o) {
            return __gt__(cntx, o);
        }

        public object __gt__(CodeContext cntx, object o) {
            object result = BinaryOp(cntx, this, o, NpyDefs.NpyArray_Ops.npy_op_greater);
            if (result == Builtin.NotImplemented) {
                result = strings_compare(o, NpyDefs.NPY_COMPARE_OP.NPY_GT);
            }
            return result;
        }

        public object __rgt__(CodeContext cntx, object o) {
            return __le__(cntx, o);
        }

        public object __ge__(CodeContext cntx, object o) {
            object result = BinaryOp(cntx, this, o, NpyDefs.NpyArray_Ops.npy_op_greater_equal);
            if (result == Builtin.NotImplemented) {
                result = strings_compare(o, NpyDefs.NPY_COMPARE_OP.NPY_GE);
            }
            return result;
        }

        public object __rge__(CodeContext cntx, object o) {
            return __lt__(cntx, o);
        }

        private object strings_compare(object o, NpyDefs.NPY_COMPARE_OP op) {
            if (NpyDefs.IsString(Dtype.TypeNum)) {
                ndarray self = this;
                ndarray array_other = NpyArray.FromAny(o, flags: NpyDefs.NPY_BEHAVED | NpyDefs.NPY_ENSUREARRAY);
                if (self.Dtype.TypeNum == NpyDefs.NPY_TYPES.NPY_UNICODE &&
                    array_other.Dtype.TypeNum == NpyDefs.NPY_TYPES.NPY_STRING) {
                    dtype dt = new dtype(self.Dtype);
                    dt.ElementSize = array_other.Dtype.ElementSize*4;
                    array_other = NpyCoreApi.FromArray(array_other, dt, 0);
                } else if (self.Dtype.TypeNum == NpyDefs.NPY_TYPES.NPY_STRING &&
                           array_other.Dtype.TypeNum == NpyDefs.NPY_TYPES.NPY_UNICODE) {
                    dtype dt = new dtype(array_other.Dtype);
                    dt.ElementSize = self.Dtype.ElementSize * 4;
                    self = NpyCoreApi.FromArray(self, dt, 0);
                }
                return ArrayReturn(NpyCoreApi.CompareStringArrays(self, array_other, op));
            }
            return Builtin.NotImplemented;
        }

        public object __int__(CodeContext cntx) {
            if (Size != 1) {
                throw new ArgumentException("only length 1 arrays can be converted to scalars");
            }
            return NpyUtil_Python.CallBuiltin(cntx, "int", GetItem(0));
        }

        public object __long__(CodeContext cntx) {
            if (Size != 1) {
                throw new ArgumentException("only length 1 arrays can be converted to scalars");
            }
            return NpyUtil_Python.CallBuiltin(cntx, "long", GetItem(0));
        }

        public object __float__(CodeContext cntx) {
            if (Size != 1) {
                throw new ArgumentException("only length 1 arrays can be converted to scalars");
            }
            return NpyUtil_Python.CallBuiltin(cntx, "float", GetItem(0));
        }

        public object __floordiv__(CodeContext cntx, object o) {
            return BinaryOp(null, this, o, NpyDefs.NpyArray_Ops.npy_op_floor_divide);
        }

        public object __complex__(CodeContext cntx) {
            if (Size != 1) {
                throw new ArgumentException("only length 1 arrays can be converted to scalars");
            }
            return NpyUtil_Python.CallBuiltin(cntx, "complex", GetItem(0));
        }

        public bool __nonzero__() {
            return (bool)this;
        }

        public static explicit operator bool(ndarray arr) {
            int val = NpyCoreApi.ArrayBool(arr);
            if (val < 0) {
                NpyCoreApi.CheckError();
                return false;
            } else {
                return val != 0;
            }
        }

        public static explicit operator int(ndarray arr) {
            object val = arr.__int__(null);
            if (val is int) {
                return (int)val;
            } else {
                throw new OverflowException();
            }
        }

        public static explicit operator BigInteger(ndarray arr) {
            return (BigInteger)arr.__long__(null);
        }

        public static explicit operator double(ndarray arr) {
            return (double)arr.__float__(null);
        }

        public static explicit operator Complex(ndarray arr) {
            return (Complex)arr.__complex__(null);
        }

        #endregion

        #region indexing

        public object this[int index] {
            get {
                return ArrayItem((long)index);
            }
        }

        public object this[long index] {
            get {
                return ArrayItem(index);
            }
        }

        public object this[IntPtr index] {
            get {
                return ArrayItem(index.ToInt64());
            }
        }

        public object this[BigInteger index] {
            get {
                long lIndex = (long)index;
                return ArrayItem(lIndex);
            }
        }

        public Object this[params object[] args] {
            get {
                if (args == null) {
                    args = new object[] { null };
                } else {
                    if (args.Length == 1 && args[0] is PythonTuple) {
                        PythonTuple pt = (PythonTuple)args[0];
                        args = pt.ToArray();
                    }

                    if (args.Length == 1 && args[0] is string) {
                        string field = (string)args[0];
                        return ArrayReturn(NpyCoreApi.GetField(this, field));
                    }
                }
                using (NpyIndexes indexes = new NpyIndexes())
                {
                    NpyUtil_IndexProcessing.IndexConverter(args, indexes);
                    if (indexes.IsSingleItem(ndim))
                    {
                        // Optimization for single item index.
                        long offset = 0;
                        Int64[] dims = Dims;
                        Int64[] s = Strides;
                        for (int i = 0; i < ndim; i++)
                        {
                            long d = dims[i];
                            long val = indexes.GetIntPtr(i).ToInt64();
                            if (val < 0)
                            {
                                val += d;
                            }
                            if (val < 0 || val >= d)
                            {
                                throw new IndexOutOfRangeException();
                            }
                            offset += val * s[i];
                        }
                        return Dtype.ToScalar(this, offset);
                    }

                    // General subscript case.
                    NpyCoreApi.Incref(Array);
                    ndarray result = NpyCoreApi.DecrefToInterface<ndarray>(
                            NpyCoreApi.ArraySubscript(this, indexes));
                    NpyCoreApi.Decref(Array);

                    if (result.ndim == 0) {
                        // We only want to return a scalar if there are not elipses
                        bool noelipses = true;
                        int n = indexes.NumIndexes;
                        for (int i = 0; i < n; i++) {
                            NpyIndexes.NpyIndexTypes t = indexes.IndexType(i);
                            if (t == NpyIndexes.NpyIndexTypes.ELLIPSIS ||
                                t == NpyIndexes.NpyIndexTypes.STRING ||
                                t == NpyIndexes.NpyIndexTypes.BOOL) {
                                noelipses = false;
                                break;
                            }
                        }
                        if (noelipses) {
                            return result.Dtype.ToScalar(this);
                        }
                    }
                    return result;
                }
            }
            set {
                if (!ChkFlags(NpyDefs.NPY_WRITEABLE)) {
                    throw new RuntimeException("array is not writeable.");
                }

                if (args == null) {
                    args = new object[] { null };
                } else {
                    if (args.Length == 1 && args[0] is PythonTuple) {
                        PythonTuple pt = (PythonTuple)args[0];
                        args = pt.ToArray();
                    }

                    if (args.Length == 1 && args[0] is string) {
                        string field = (string)args[0];
                        if (!ChkFlags(NpyDefs.NPY_WRITEABLE)) {
                            throw new RuntimeException("array is not writeable.");
                        }
                        IntPtr descr;
                        int offset = NpyCoreApi.GetFieldOffset(Dtype, field, out descr);
                        if (offset < 0) {
                            throw new ArgumentException(String.Format("field name '{0}' not found.", field));
                        }
                        NpyArray.SetField(this, descr, offset, value);
                        return;
                    }
                }


                using (NpyIndexes indexes = new NpyIndexes())
                {
                    NpyUtil_IndexProcessing.IndexConverter(args, indexes);

                    // Special case for boolean on 0-d arrays.
                    if (ndim == 0 && indexes.NumIndexes == 1 && indexes.IndexType(0) == NpyIndexes.NpyIndexTypes.BOOL)
                    {
                        if (indexes.GetBool(0))
                        {
                            SetItem(value, 0);
                        }
                        return;
                    }

                    // Special case for single assignment.
                    long single_offset = indexes.SingleAssignOffset(this);
                    if (single_offset >= 0)
                    {
                        // This is a single item assignment. Use SetItem.
                        SetItem(value, single_offset);
                        return;
                    }

                    if (indexes.IsSimple)
                    {
                        ndarray view = null;
                        try {
                            if (GetType() == typeof(ndarray)) {
                                view = NpyCoreApi.IndexSimple(this, indexes);
                            } else {
                                // Call through python to let the subtype returns the correct view
                                // TODO: Do we really need this? Why only for set with simple indexing?
                                CodeContext cntx = PythonOps.GetPythonTypeContext(DynamicHelpers.GetPythonType(this));
                                object item = PythonOps.GetIndex(cntx, this, new PythonTuple(args));
                                view = (item as ndarray);
                                if (view == null) {
                                    throw new RuntimeException("Getitem not returning array");
                                }
                            }

                            NpyArray.CopyObject(view, value);
                        } finally {
                            if (view != null) {
                                view.Dispose();
                            }
                        }
                    }
                    else
                    {
                        ndarray array_value = NpyArray.FromAny(value, Dtype, 0, 0, NpyDefs.NPY_FORCECAST, null);
                        try {
                            NpyCoreApi.Incref(array_value.Array);
                            if (NpyCoreApi.IndexFancyAssign(this, indexes, array_value) < 0) {
                                NpyCoreApi.CheckError();
                            }
                        } finally {
                            NpyCoreApi.Decref(array_value.Array);
                        }
                    }
                }
            }
        }

        #endregion

        #region properties

        /// <summary>
        /// Number of dimensions in the array
        /// </summary>
        public int ndim {
            get { return Marshal.ReadInt32(core, NpyCoreApi.ArrayOffsets.off_nd); }
        }

        /// <summary>
        /// Returns the size of each dimension as a tuple.
        /// </summary>
        public object shape {
            get { return NpyUtil_Python.ToPythonTuple(this.Dims); }
            set {
                IntPtr[] shape = NpyUtil_ArgProcessing.IntpArrConverter(value);
                NpyCoreApi.SetShape(this, shape);
            }
        }


        /// <summary>
        /// Total number of elements in the array.
        /// </summary>
        public object size {
            get { return NpyCoreApi.ArraySize(this).ToPython(); }
        }

        public PythonBuffer data {
            get {
                throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Returns the reference count of the core array object.  Used for debugging only.
        /// </summary>
        public int __coreRefCount__ { get { return Marshal.ReadInt32(Array, NpyCoreApi.Offset_RefCount); } }


        /// <summary>
        /// The type descriptor object for this array
        /// </summary>
        public dtype Dtype {
            get {
                if (core == IntPtr.Zero) return null;
                IntPtr descr = Marshal.ReadIntPtr(core, NpyCoreApi.ArrayOffsets.off_descr);
                return NpyCoreApi.ToInterface<dtype>(descr);
            }
            set {
                NpyCoreApi.ArraySetDescr(this, value);
            }
        }


        /// <summary>
        /// The type descriptor object for this array
        /// </summary>
        public object dtype {
            get {
                return this.Dtype;
            }
            set {
                dtype descr = value as dtype;
                if (descr == null) {
                    descr = NpyDescr.DescrConverter(NpyUtil_Python.DefaultContext, value);
                }
                this.Dtype = descr;
            }
        }

        /// <summary>
        /// Flags for this array
        /// </summary>
        public flagsobj flags {
            get {
                return new flagsobj(this);
            }
        }

        /// <summary>
        /// Returns an array of the stride of each dimension.
        /// </summary>
        public Int64[] Strides {
            get { return NpyCoreApi.GetArrayDimsOrStrides(this, false); }
        }

        public PythonTuple strides {
            get { return NpyUtil_Python.ToPythonTuple(Strides); }
        }

        public object real {
            get {
                return NpyCoreApi.GetReal(this);
            }
            set {
                ndarray val = NpyArray.FromAny(value, null, 0, 0, 0, null);
                NpyCoreApi.MoveInto(NpyCoreApi.GetReal(this), val);
            }
        }

        public object imag {
            get {
                if (IsComplex) {
                    return NpyCoreApi.GetImag(this);
                } else {
                    // TODO: np.zeros_like when we have it.
                    ndarray result = copy();
                    result.flat = 0;
                    return result;
                }
            }
            set {
                if (IsComplex) {
                    ndarray val = NpyArray.FromAny(value, null, 0, 0, 0, null);
                    NpyCoreApi.MoveInto(NpyCoreApi.GetImag(this), val);
                } else {
                    throw new ArgumentTypeException("array does not have an imaginary part to set.");
                }
            }
        }

        public object flat {
            get {
                return NpyCoreApi.IterNew(this);
            }
            set {
                // Assing like a.flat[:] = value
                flatiter it = NpyCoreApi.IterNew(this);
                it[new Slice(null)] = value;
            }
        }

        public object @base {
            get {
                // TODO: Handle non-array bases
                return BaseArray;
            }
        }

        public int itemsize {
            get {
                return Dtype.ElementSize;
            }
        }

        public object nbytes {
            get {
                return NpyUtil_Python.ToPython(itemsize*Size);
            }
        }

        public ndarray T {
            get {
                return Transpose();
            }
        }

        #endregion

        #region methods

        public int dump(CodeContext cntx, object file) {
            if (file is string) {
                file = NpyUtil_Python.CallBuiltin(cntx, "open", file, "wb");
            }
            NpyUtil_Python.CallFunction(cntx, "cPickle", "dump", this, file, 2);
            return 0;
        }

        public object dumps(CodeContext cntx) {
            return NpyUtil_Python.CallFunction(cntx, "cPickle", "dumps", this, 2);
        }

        public object all(object axis = null, ndarray @out = null) {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            return ArrayReturn(All(iAxis, @out));
        }

        public object any(object axis = null, ndarray @out = null) {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            return ArrayReturn(Any(iAxis, @out));
        }

        public object argmax(object axis = null, ndarray @out = null) {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            return ArrayReturn(ArgMax(iAxis, @out));
        }

        public object argmin(object axis = null, ndarray @out = null) {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            return ArrayReturn(ArgMin(iAxis, @out));
        }

        public object argsort(object axis = null, string kind = null, object order = null) {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis, -1);
            NpyDefs.NPY_SORTKIND sortkind = NpyUtil_ArgProcessing.SortkindConverter(kind);

            if (order != null) {
                throw new NotImplementedException("Sort field order not yet implemented.");
            }

            return ArrayReturn(ArgSort(iAxis, sortkind));
        }

        public ndarray astype(CodeContext cntx, object dtype = null) {
            dtype d = NpyDescr.DescrConverter(cntx, dtype);
            if (d == this.Dtype) {
                return this;
            }
            if (this.Dtype.HasNames) {
                // CastToType doesn't work properly for
                // record arrays, so we use FromArray.
                int flags = NpyDefs.NPY_FORCECAST;
                if (IsFortran) {
                    flags |= NpyDefs.NPY_FORTRAN;
                }
                return NpyCoreApi.FromArray(this, d, flags);
            }
            return NpyCoreApi.CastToType(this, d, this.IsFortran);
        }

        public ndarray byteswap(bool inplace = false) {
            return NpyCoreApi.Byteswap(this, inplace);
        }

        private static string[] chooseArgNames = { "out", "mode" };

        public object choose([ParamDictionary] IDictionary<object,object> kwargs,
                             params object[] args){
            IEnumerable<object> choices;
            if (args == null) {
                choices = new object[0];
            }
            else if (args.Length == 1 && args[0] is IEnumerable<object>) {
                choices = (IEnumerable<object>)args[0];
            } else {
                choices = args;
            }
            object[] kargs = NpyUtil_ArgProcessing.BuildArgsArray(new object[0], chooseArgNames, kwargs);
            ndarray aout = kargs[0] as ndarray;
            NpyDefs.NPY_CLIPMODE clipMode = NpyUtil_ArgProcessing.ClipmodeConverter(kargs[1]);
            return ArrayReturn(Choose(choices, aout, clipMode));
        }

        public object clip(object min = null, object max = null, ndarray @out = null) {
            return Clip(min, max, @out);
        }

        public ndarray compress(object condition, object axis = null, ndarray @out = null) {
            ndarray aCondition = NpyArray.FromAny(condition, null, 0, 0, 0, null);
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);

            if (aCondition.ndim != 1) {
                throw new ArgumentException("condition must be 1-d array");
            }

            ndarray indexes = aCondition.NonZero()[0];
            return TakeFrom(indexes, iAxis, @out, NpyDefs.NPY_CLIPMODE.NPY_RAISE);
        }

        public ndarray conj(ndarray @out = null) {
            return conjugate(@out);
        }

        public ndarray conjugate(ndarray @out = null) {
            return Conjugate(@out);
        }

        public ndarray copy(object order = null) {
            NpyDefs.NPY_ORDER eOrder = NpyUtil_ArgProcessing.OrderConverter(order);
            return NpyCoreApi.NewCopy(this, eOrder);
        }

        public object cumprod(CodeContext cntx, object axis = null, object dtype = null,
                              ndarray @out = null) {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            dtype rtype = null;
            if (dtype != null) {
                rtype = NpyDescr.DescrConverter(cntx, dtype);
            }
            return CumProd(iAxis, rtype, @out);
        }

        public object cumsum(CodeContext cntx, object axis = null, object dtype = null,
                             ndarray @out = null) {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            dtype rtype = null;
            if (dtype != null) {
                rtype = NpyDescr.DescrConverter(cntx, dtype);
            }
            return CumSum(iAxis, rtype, @out);
        }


        public ndarray diagonal(int offset = 0, int axis1 = 0, int axis2 = 1) {
            return Diagonal(offset, axis1, axis2);
        }

        public object dot(object other) {
            return ModuleMethods.dot(this, other);
        }

        public void fill(object scalar) {
            FillWithScalar(scalar);
        }

        public ndarray flatten(object order = null) {
            NpyDefs.NPY_ORDER eOrder =
                NpyUtil_ArgProcessing.OrderConverter(order);
            return Flatten(eOrder);
        }

        public ndarray getfield(CodeContext cntx, object dtype, int offset = 0) {
            NumpyDotNet.dtype dt = NpyDescr.DescrConverter(cntx, dtype);
            return NpyCoreApi.GetField(this, dt, offset);
        }

        public object item(params object[] args) {
            if (args != null && args.Length == 1 && args[0] is PythonTuple) {
                PythonTuple t = (PythonTuple)args[0];
                args = t.ToArray();
            }
            if (args == null || args.Length == 0) {
                if (ndim == 0 || Size == 1) {
                    return GetItem(0);
                } else {
                    throw new ArgumentException("can only convert an array of size 1 to a Python scalar");
                }
            } else {
                using (NpyIndexes indexes = new NpyIndexes()) {
                    NpyUtil_IndexProcessing.IndexConverter(args, indexes);
                    if (args.Length == 1) {
                        if (indexes.IndexType(0) != NpyIndexes.NpyIndexTypes.INTP) {
                            throw new ArgumentException("invalid integer");
                        }
                        // Do flat indexing
                        return Flat.Get(indexes.GetIntPtr(0));
                    } else {
                        if (indexes.IsSingleItem(ndim)) {
                            long offset = indexes.SingleAssignOffset(this);
                            return GetItem(offset);
                        } else {
                            throw new ArgumentException("Incorrect number of indices for the array");
                        }
                    }
                }
            }
        }

        public void itemset(params object[] args) {
            // Convert args to value and args
            if (args == null || args.Length == 0) {
                throw new ArgumentException("itemset must have at least one argument");
            }
            object value = args.Last();
            args = args.Take(args.Length - 1).ToArray();

            if (args.Length == 1 && args[0] is PythonTuple) {
                PythonTuple t = (PythonTuple)args[0];
                args = t.ToArray();
            }
            if (args.Length == 0) {
                if (ndim == 0 || Size == 1) {
                    SetItem(value, 0);
                } else {
                    throw new ArgumentException("can only convert an array of size 1 to a Python scalar");
                }
            } else {
                using (NpyIndexes indexes = new NpyIndexes()) {
                    NpyUtil_IndexProcessing.IndexConverter(args, indexes);
                    if (args.Length == 1) {
                        if (indexes.IndexType(0) != NpyIndexes.NpyIndexTypes.INTP) {
                            throw new ArgumentException("invalid integer");
                        }
                        // Do flat indexing
                        Flat.SingleAssign(indexes.GetIntPtr(0), value);
                    } else {
                        if (indexes.IsSingleItem(ndim)) {
                            long offset = indexes.SingleAssignOffset(this);
                            SetItem(value, offset);
                        } else {
                            throw new ArgumentException("Incorrect number of indices for the array");
                        }
                    }
                }
            }
        }

        public object max(object axis = null, ndarray @out = null) {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            return ArrayReturn(Max(iAxis, @out));
        }

        public object mean(CodeContext cntx, object axis = null, object dtype = null,
                           ndarray @out = null) {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            dtype rtype = null;
            if (dtype != null) {
                rtype = NpyDescr.DescrConverter(cntx, dtype);
            }
            return Mean(iAxis, GetTypeDouble(this.Dtype, rtype), @out);
        }

        public object min(object axis = null, ndarray @out = null) {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            return ArrayReturn(Min(iAxis, @out));
        }

        public ndarray newbyteorder(string endian = null) {
            dtype newtype = NpyCoreApi.DescrNewByteorder(Dtype, NpyUtil_ArgProcessing.ByteorderConverter(endian));
            return NpyCoreApi.View(this, newtype, null);
        }

        public PythonTuple nonzero() {
            return new PythonTuple(NonZero());
        }

        public object prod(CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null) {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            dtype rtype = null;
            if (dtype != null) {
                rtype = NpyDescr.DescrConverter(cntx, dtype);
            }
            return ArrayReturn(Prod(iAxis, rtype, @out));
        }

        public object ptp(object axis = null, ndarray @out = null) {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            return Ptp(iAxis, @out);
        }

        public void put(object indices, object values, object mode = null) {
            ndarray aIndices;
            ndarray aValues;
            NpyDefs.NPY_CLIPMODE eMode;

            aIndices = (indices as ndarray);
            if (aIndices == null) {
                aIndices = NpyArray.FromAny(indices, NpyCoreApi.DescrFromType(NpyDefs.NPY_INTP),
                    0, 0, NpyDefs.NPY_CARRAY, null);
            }
            aValues = (values as ndarray);
            if (aValues == null) {
                aValues = NpyArray.FromAny(values, Dtype, 0, 0, NpyDefs.NPY_CARRAY, null);
            }
            eMode = NpyUtil_ArgProcessing.ClipmodeConverter(mode);
            PutTo(aValues, aIndices, eMode);
        }

        public ndarray ravel(object order = null) {
            NpyDefs.NPY_ORDER eOrder = NpyUtil_ArgProcessing.OrderConverter(order);
            return Ravel(eOrder);
        }

        public object repeat(object repeats, object axis = null) {
            ndarray aRepeats = (repeats as ndarray);
            if (aRepeats == null) {
                aRepeats = NpyArray.FromAny(repeats, NpyCoreApi.DescrFromType(NpyDefs.NPY_INTP),
                    0, 0, NpyDefs.NPY_CARRAY, null);
            }
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            return ArrayReturn(Repeat(aRepeats, iAxis));
        }

        private static string[] reshapeKeywords = { "order" };

        public ndarray reshape([ParamDictionary] IDictionary<object,object> kwds, params object[] args) {
            object[] keywordArgs = NpyUtil_ArgProcessing.BuildArgsArray(new object[0], reshapeKeywords, kwds);
            NpyDefs.NPY_ORDER order = NpyUtil_ArgProcessing.OrderConverter(keywordArgs[0]);
            IntPtr[] newshape;
            // TODO: Add NpyArray_View call for (None) case. (Why?)
            if (args == null) {
                newshape = new IntPtr[0];
            } else if (args.Length == 1 && args[0] is IList<object>) {
                newshape = NpyUtil_ArgProcessing.IntpListConverter((IList<object>)args[0]);
            } else {
                newshape = NpyUtil_ArgProcessing.IntpListConverter(args);
            }
            return NpyCoreApi.Newshape(this, newshape, order);
        }

        public ndarray Reshape(IEnumerable<long> shape, NpyDefs.NPY_ORDER order = NpyDefs.NPY_ORDER.NPY_ANYORDER) {
            return NpyCoreApi.Newshape(this, shape.Select(x => (IntPtr)x).ToArray(), order);
        }

        private static string[] resizeKeywords = { "refcheck" };

        public void resize([ParamDictionary] IDictionary<object,object> kwds, params object[] args) {
            object[] keywordArgs = NpyUtil_ArgProcessing.BuildArgsArray(new object[0], resizeKeywords, kwds);
            bool refcheck = NpyUtil_ArgProcessing.BoolConverter(keywordArgs[0]);
            IntPtr[] newshape;

            if (args == null || args.Length == 0 || args.Length == 1 && args[0] == null) {
                return;
            }
            if (args.Length == 1 && args[0] is IList<object>) {
                newshape = NpyUtil_ArgProcessing.IntpListConverter((IList<object>)args[0]);
            } else {
                newshape = NpyUtil_ArgProcessing.IntpListConverter(args);
            }
            Resize(newshape, refcheck, NpyDefs.NPY_ORDER.NPY_CORDER);
        }

        public object round(int decimals = 0, ndarray @out = null) {
            return Round(decimals, @out);
        }

        public object searchsorted(object keys, string side = null) {
            NpyDefs.NPY_SEARCHSIDE eSide = NpyUtil_ArgProcessing.SearchsideConverter(side);
            ndarray aKeys = (keys as ndarray);
            if (aKeys == null) {
                aKeys = NpyArray.FromAny(keys, NpyArray.FindArrayType(keys, Dtype, NpyDefs.NPY_MAXDIMS),
                    0, 0, NpyDefs.NPY_CARRAY, null);
            }
            return ArrayReturn(SearchSorted(aKeys, eSide));
        }

        public void setfield(CodeContext cntx, object value, object dtype, int offset = 0) {
            dtype d = NpyDescr.DescrConverter(cntx, dtype);
            NpyArray.SetField(this, d.Descr, offset, value);
        }

        public void setflags(object write = null, object align = null, object uic = null) {
            int flags = RawFlags;
            if (align != null) {
                bool bAlign = NpyUtil_ArgProcessing.BoolConverter(align);
                if (bAlign) {
                    flags |= NpyDefs.NPY_ALIGNED;
                } else {
                    if (!NpyCoreApi.IsAligned(this)) {
                        throw new ArgumentException("cannot set aligned flag of mis-aligned array to True");
                    }
                    flags &= ~NpyDefs.NPY_ALIGNED;
                }
            }
            if (uic != null) {
                bool bUic = NpyUtil_ArgProcessing.BoolConverter(uic);
                if (bUic) {
                    throw new ArgumentException("cannot set UPDATEIFCOPY flag to True");
                } else {
                    NpyCoreApi.ClearUPDATEIFCOPY(this);
                }
            }
            if (write != null) {
                bool bWrite = NpyUtil_ArgProcessing.BoolConverter(write);
                if (bWrite) {
                    if (!NpyCoreApi.IsWriteable(this)) {
                        throw new ArgumentException("cannot set WRITEABLE flag to true on this array");
                    }
                    flags |= NpyDefs.NPY_WRITEABLE;
                } else {
                    flags &= ~NpyDefs.NPY_WRITEABLE;
                }
            }
            RawFlags = flags;
        }

        public void sort(int axis = -1, string kind = null, object order = null) {
            NpyDefs.NPY_SORTKIND sortkind = NpyUtil_ArgProcessing.SortkindConverter(kind);
            if (order != null) {
                throw new NotImplementedException("Field sort order not yet implemented.");
            }
            Sort(axis, sortkind);
        }

        public object squeeze() {
            return Squeeze();
        }

        public object std(CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null, int ddof = 0) {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            dtype rtype = null;
            if (dtype != null) {
                rtype = NpyDescr.DescrConverter(cntx, dtype);
            }
            return Std(iAxis, GetTypeDouble(this.Dtype, rtype), @out, false, ddof);
        }

        public object sum(CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null) {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            dtype rtype = null;
            if (dtype != null) {
                rtype = NpyDescr.DescrConverter(cntx, dtype);
            }
            return ArrayReturn(Sum(iAxis, rtype, @out));
        }


        public ndarray swapaxes(int a1, int a2) {
            return SwapAxes(a1, a2);
        }

        public ndarray swapaxes(object a1, object a2) {
            int iA1 = NpyUtil_ArgProcessing.IntConverter(a1);
            int iA2 = NpyUtil_ArgProcessing.IntConverter(a2);
            return SwapAxes(iA1, iA2);
        }


        public object take(object indices,
                           object axis = null,
                           ndarray @out = null,
                           object mode = null) {
            ndarray aIndices;
            int iAxis;
            NpyDefs.NPY_CLIPMODE cMode;

            aIndices = (indices as ndarray);
            if (aIndices == null) {
                aIndices = NpyArray.FromAny(indices, NpyCoreApi.DescrFromType(NpyDefs.NPY_INTP),
                    1, 0, NpyDefs.NPY_CONTIGUOUS, null);
            }
            iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            cMode = NpyUtil_ArgProcessing.ClipmodeConverter(mode);
            return ArrayReturn(TakeFrom(aIndices, iAxis, @out, cMode));
        }

        public void tofile(CodeContext cntx, PythonFile file, string sep = null, string format = null) {
            ToFile(cntx, file, sep, format);
        }

        public void tofile(CodeContext cntx, string filename, string sep = null, string format = null) {
            PythonFile f = (PythonFile)NpyUtil_Python.CallBuiltin(cntx, "open", filename, "wb");
            try {
                tofile(cntx, f, sep, format);
            } finally {
                f.close();
            }
        }

        public object tolist() {
            if (ndim == 0) {
                return GetItem(0);
            } else {
                List result = new List();
                long size = Dims[0];
                for (long i = 0; i < size; i++) {
                    result.append(NpyCoreApi.ArrayItem(this, i).tolist());
                }
                return result;
            }
        }

        public Bytes tostring(object order = null) {
            NpyDefs.NPY_ORDER eOrder = NpyUtil_ArgProcessing.OrderConverter(order);
            return ToString(eOrder);
        }

        public object trace(CodeContext cntx, int offset = 0, int axis1 = 0, int axis2 = 1,
            object dtype = null, ndarray @out = null) {
            ndarray diag = Diagonal(offset, axis1, axis2);
            return diag.sum(cntx, dtype:dtype, @out:@out);
        }

        public ndarray transpose(params object[] args) {
            if (args == null || args.Length == 0 || args.Length == 1 && args[0] == null) {
                return Transpose();
            } else if (args.Length == 1 && args[0] is IList<object>) {
                return Transpose(NpyUtil_ArgProcessing.IntpListConverter((IList<object>)args[0]));
            } else {
                return Transpose(NpyUtil_ArgProcessing.IntpListConverter(args));
            }
        }

        public object var(CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null, int ddof = 0) {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            dtype rtype = null;
            if (dtype != null) {
                rtype = NpyDescr.DescrConverter(cntx, dtype);
            }
            return Std(iAxis, GetTypeDouble(this.Dtype, rtype), @out, true, ddof);
        }


        public ndarray view(CodeContext cntx, object dtype = null, object type = null) {
            if (dtype != null && type == null) {
                if (IsNdarraySubtype(dtype)) {
                    type = dtype;
                    dtype = null;
                }
            }

            if (type != null && !IsNdarraySubtype(type)) {
                throw new ArgumentException("Type must be a subtype of ndarray.");
            }
            dtype rtype = null;
            if (dtype != null) {
                rtype = NpyDescr.DescrConverter(cntx, dtype);
            }
            return NpyCoreApi.View(this, rtype, type);
        }

        #endregion

        #endregion


        public long Size {
            get { return NpyCoreApi.ArraySize(this).ToInt64(); }
        }

        internal ndarray Real {
            get { return NpyCoreApi.GetReal(this); }
        }

        internal ndarray Imag {
            get { return NpyCoreApi.GetImag(this); }
        }

        public override string ToString() {
            return StrFunction(this);
        }

        internal flatiter Flat {
            get {
                return NpyCoreApi.IterNew(this);
            }
        }

        public ndarray NewCopy(NpyDefs.NPY_ORDER order = NpyDefs.NPY_ORDER.NPY_CORDER) {
            return NpyCoreApi.NewCopy(this, order);
        }


        /// <summary>
        /// Directly accesses the array memory and returns the object at that
        /// offset.  No checks are made, caller can easily crash the program
        /// or retrieve garbage data.
        /// </summary>
        /// <param name="offset">Offset into data array in bytes</param>
        /// <returns>Contents of the location</returns>
        internal object GetItem(long offset) {
            return Dtype.f.GetItem(offset, this);
        }


        /// <summary>
        /// Directly sets a given location in the data array.  No checks are
        /// made to make sure the offset is sensible or the data is valid in
        /// anyway -- caller beware.
        /// 'internal' because this is a security vulnerability.
        /// </summary>
        /// <param name="src">Value to write</param>
        /// <param name="offset">Offset into array in bytes</param>
        internal void SetItem(object src, long offset) {
            Dtype.f.SetItem(src, offset, this);
        }


        /// <summary>
        /// Handle to the core representation.
        /// </summary>
        public IntPtr Array {
            get { return core; }
        }


        /// <summary>
        /// Base address of the array data memory. Use with caution.
        /// </summary>
        internal IntPtr DataAddress {
            get { return Marshal.ReadIntPtr(core, NpyCoreApi.ArrayOffsets.off_data); }
        }

        /// <summary>
        /// Returns an array of the sizes of each dimension. This property allocates
        /// a new array with each call and must make a managed-to-native call so it's
        /// worth caching the results if used in a loop.
        /// </summary>
        public Int64[] Dims {
            get { return NpyCoreApi.GetArrayDimsOrStrides(this, true); }
        }


        /// <summary>
        /// Returns the stride of a given dimension. For looping over all dimensions,
        /// use 'strides'.  This is more efficient if only one dimension is of interest.
        /// </summary>
        /// <param name="dimension">Dimension to query</param>
        /// <returns>Data stride in bytes</returns>
        public long Stride(int dimension) {
            return NpyCoreApi.GetArrayStride(this, dimension);
        }


        /// <summary>
        /// True if memory layout of array is contiguous
        /// </summary>
        public bool IsContiguous {
            get { return ChkFlags(NpyDefs.NPY_CONTIGUOUS); }
        }

        public bool IsOneSegment {
            get { return ndim == 0 || ChkFlags(NpyDefs.NPY_FORTRAN) || ChkFlags(NpyDefs.NPY_CARRAY); }
        }

        /// <summary>
        /// True if memory layout is Fortran order, false implies C order
        /// </summary>
        public bool IsFortran {
            get { return ChkFlags(NpyDefs.NPY_FORTRAN) && ndim > 1; }
        }

        public bool IsNotSwapped {
            get { return Dtype.IsNativeByteOrder; }
        }

        public bool IsByteSwapped {
            get { return !IsNotSwapped; }
        }

        public bool IsCArray {
            get { return ChkFlags(NpyDefs.NPY_CARRAY) && IsNotSwapped; }
        }

        public bool IsCArray_RO {
            get { return ChkFlags(NpyDefs.NPY_CARRAY_RO) && IsNotSwapped; }
        }

        public bool IsFArray {
            get { return ChkFlags(NpyDefs.NPY_FARRAY) && IsNotSwapped; }
        }

        public bool IsFArray_RO {
            get { return ChkFlags(NpyDefs.NPY_FARRAY_RO) && IsNotSwapped; }
        }

        public bool IsBehaved {
            get { return ChkFlags(NpyDefs.NPY_BEHAVED) && IsNotSwapped; }
        }

        public bool IsBehaved_RO {
            get { return ChkFlags(NpyDefs.NPY_ALIGNED) && IsNotSwapped; }
        }

        internal bool IsComplex {
            get { return NpyDefs.IsComplex(Dtype.TypeNum); }
        }

        internal bool IsInteger {
            get { return NpyDefs.IsInteger(Dtype.TypeNum); }
        }

        public bool IsFlexible {
            get { return NpyDefs.IsFlexible(Dtype.TypeNum); }
        }

        public bool IsWriteable {
            get { return ChkFlags(NpyDefs.NPY_WRITEABLE); }
        }

        public bool IsString {
            get { return Dtype.TypeNum == NpyDefs.NPY_TYPES.NPY_STRING; }
        }


        /// <summary>
        /// TODO: What does this return?
        /// </summary>
        public int ElementStrides {
            get { return NpyCoreApi.ElementStrides(this); }
        }

        public bool StridingOk(NpyDefs.NPY_ORDER order) {
            return order == NpyDefs.NPY_ORDER.NPY_ANYORDER ||
                order == NpyDefs.NPY_ORDER.NPY_CORDER && IsContiguous ||
                order == NpyDefs.NPY_ORDER.NPY_FORTRANORDER && IsFortran;
        }

        private bool ChkFlags(int flag) {
            return ((RawFlags & flag) == flag);
        }

        // These operators are useful from other C# code and also turn into the
        // appropriate Python functions (+ goes to __add__, etc).

        #region IEnumerable<object> interface

        public IEnumerator<object> GetEnumerator() {
            return new ndarray_Enumerator(this);
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() {
            return new ndarray_Enumerator(this);
        }

        #endregion

        #region Internal methods

        internal long Length {
            get {
                return Dims[0];
            }
        }

        internal static object ArrayReturn(ndarray a) {
            if (a.ndim == 0) {
                return a.Dtype.ToScalar(a);
            } else {
                return a;
            }
        }
        private string BuildStringRepr(bool repr) {
            // Equivalent to array_repr_builtin (arrayobject.c)
            StringBuilder sb = new StringBuilder();
            if (repr) sb.Append("array(");
            DumpData(sb, this.Dims, this.Strides, 0, 0);

            if (repr) {
                if (NpyDefs.IsExtended(this.Dtype.TypeNum)) {
                    sb.AppendFormat(", '{0}{1}')", (char)Dtype.Type, this.Dtype.ElementSize);
                } else {
                    sb.AppendFormat(", '{0}')", (char)Dtype.Type);
                }
            }
            return sb.ToString();
        }

        /// <summary>
        /// Recursively walks the array and appends a representation of each element
        /// to the passed string builder.  Square brackets delimit each array dimension.
        /// </summary>
        /// <param name="sb">StringBuilder instance to append to</param>
        /// <param name="dimensions">Array of size of each dimension</param>
        /// <param name="strides">Offset in bytes to reach next element in each dimension</param>
        /// <param name="dimIdx">Index of the current dimension (starts at 0, recursively counts up)</param>
        /// <param name="offset">Byte offset into data array, starts at 0</param>
        private void DumpData(StringBuilder sb, long[] dimensions, long[] strides,
            int dimIdx, long offset) {

            if (dimIdx == ndim) {
                Object value = Dtype.f.GetItem(offset, this);
                if (value == null) {
                    sb.Append("None");
                } else {
                    sb.Append((string)PythonOps.Repr(NpyUtil_Python.DefaultContext, value));
                }
            } else {
                sb.Append('[');
                for (int i = 0; i < dimensions[dimIdx]; i++) {
                    DumpData(sb, dimensions, strides, dimIdx + 1,
                                  offset + strides[dimIdx] * i);
                    if (i < dimensions[dimIdx] - 1) {
                        sb.Append(", ");
                    }
                }
                sb.Append(']');
            }
        }

        /// <summary>
        /// Indexes an array by a single long and returns either an item or a sub-array.
        /// </summary>
        /// <param name="index">The index into the array</param>
        object ArrayItem(long index) {
            if (ndim == 1) {
                long dim0 = Dims[0];
                if (index < 0) {
                    index += dim0;
                }
                if (index < 0 || index >= dim0) {
                    throw new IndexOutOfRangeException("Index out of range");
                }
                long offset = index * Strides[0];
                return Dtype.ToScalar(this, offset);
            } else {
                return NpyCoreApi.ArrayItem(this, index);
            }
        }

        internal Int32 RawFlags {
            get {
                return Marshal.ReadInt32(Array + NpyCoreApi.ArrayOffsets.off_flags);
            }
            set {
                Marshal.WriteInt32(Array + NpyCoreApi.ArrayOffsets.off_flags, value);
            }
        }

        internal static dtype GetTypeDouble(dtype dtype1, dtype dtype2) {
            if (dtype2 != null) {
                return dtype2;
            }
            if (dtype1.TypeNum < NpyDefs.NPY_TYPES.NPY_FLOAT) {
                return NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_DOUBLE);
            } else {
                return dtype1;
            }
        }

        private static bool IsNdarraySubtype(object type) {
            if (type == null) {
                return false;
            }
            PythonType pt = type as PythonType;
            if (pt == null) {
                return false;
            }
            return PythonOps.IsSubClass(pt, DynamicHelpers.GetPythonTypeFromType(typeof(ndarray)));
        }

        /// <summary>
        /// Pointer to the internal memory. Should be used with great caution - memory
        /// is native memory, not managed memory.
        /// </summary>
        internal IntPtr UnsafeAddress {
            get { return Marshal.ReadIntPtr(core, NpyCoreApi.ArrayOffsets.off_data); }
        }

        internal ndarray BaseArray {
            get {
                IntPtr p = Marshal.ReadIntPtr(core, NpyCoreApi.ArrayOffsets.off_base_array);
                if (p == IntPtr.Zero) {
                    return null;
                } else {
                    return NpyCoreApi.ToInterface<ndarray>(p);
                }
            }
            set {
                lock (this) {
                    IntPtr p = Marshal.ReadIntPtr(core, NpyCoreApi.ArrayOffsets.off_base_array);
                    if (p != IntPtr.Zero) {
                        NpyCoreApi.Decref(p);
                    }
                    NpyCoreApi.Incref(value.core);
                    Marshal.WriteIntPtr(core, NpyCoreApi.ArrayOffsets.off_base_array, value.core);
                }
            }
        }

        internal unsafe void CopySwapIn(long offset, void* data, bool swap) {
            NpyCoreApi.CopySwapIn(this, offset, data, swap);
        }

        internal unsafe void CopySwapOut(long offset, void* data, bool swap) {
            NpyCoreApi.CopySwapOut(this, offset, data, swap);
        }

        #endregion

        #region Memory pressure handling

        // The GC only knows about the managed memory that has been allocated,
        // not the large pool of native array data.  This means that the GC
        // may not run even if we are about to run out of memory.  Adding
        // memory pressure tells the GC how much native memory is associated
        // with managed objects.

        /// <summary>
        /// Track the total pressure allocated by numpy.  This is just for
        /// error checking and to make sure it goes back to 0 in the end.
        /// </summary>
        private static long TotalMemPressure = 0;

        internal static void IncreaseMemoryPressure(ndarray arr) {
            if (arr.flags.owndata) {
                int newBytes = (int)(arr.Size * arr.Dtype.ElementSize);
                if (newBytes == 0) {
                    return;
                }

                // Stupid annoying hack.  What happens is the finalizer queue
                // is processed by a low-priority background thread and can fall
                // behind, allowing memory to be filled if the primary thread is
                // creating garbage faster than the finalizer thread is cleaning
                // it up.  This is a heuristic to cause the main thread to pause
                // when needed.  All of this is necessary because the ndarray
                // object defines a finalizer, which most .NET objects don't have
                // and .NET doesn't appear well optimized for cases with huge
                // numbers of finalizable objects.
                // TODO: What do we do for a collection heuristic for 64-bit? Don't
                // want to collect too often but don't want to page either.
                if (IntPtr.Size == 4 &&
                    (TotalMemPressure > 1500000000 || TotalMemPressure + newBytes > 1700000000)) {
                    System.GC.Collect();
                    System.GC.WaitForPendingFinalizers();
                }

                System.Threading.Interlocked.Add(ref TotalMemPressure, newBytes);
                System.GC.AddMemoryPressure(newBytes);
                //Console.WriteLine("Added {0} bytes of pressure, now {1}",
                //    newBytes, TotalMemPressure);
            }
        }

        internal static void DecreaseMemoryPressure(ndarray arr) {
            if (arr.flags.owndata) {
                int newBytes = (int)(arr.Size * arr.Dtype.ElementSize);
                System.Threading.Interlocked.Add(ref TotalMemPressure, -newBytes);
                if (newBytes > 0) {
                    System.GC.RemoveMemoryPressure(newBytes);
                }
                //Console.WriteLine("Removed {0} bytes of pressure, now {1}",
                //    newBytes, TotalMemPressure);
            }
        }

        #endregion

        #region Buffer protocol

        public IExtBufferProtocol GetBuffer(NpyBuffer.PyBuf flags) {
            return new ndarrayBufferAdapter(this, flags);
        }

        public IExtBufferProtocol GetPyBuffer(int flags) {
            return GetBuffer((NpyBuffer.PyBuf)flags);
        }

        /// <summary>
        /// Adapts an instance that implements IBufferProtocol and IPythonBufferable
        /// to the IExtBufferProtocol.
        /// </summary>
        private class ndarrayBufferAdapter : IExtBufferProtocol
        {
            internal ndarrayBufferAdapter(ndarray a, NpyBuffer.PyBuf flags) {
                arr = a;

                if ((flags & NpyBuffer.PyBuf.C_CONTIGUOUS) == NpyBuffer.PyBuf.C_CONTIGUOUS &&
                    !arr.ChkFlags(NpyDefs.NPY_C_CONTIGUOUS)) {
                    throw new ArgumentException("ndarray is not C-continuous");
                }
                if ((flags & NpyBuffer.PyBuf.F_CONTIGUOUS) == NpyBuffer.PyBuf.F_CONTIGUOUS &&
                    !arr.ChkFlags(NpyDefs.NPY_F_CONTIGUOUS)) {
                    throw new ArgumentException("ndarray is not F-continuous");
                }
                if ((flags & NpyBuffer.PyBuf.ANY_CONTIGUOUS) == NpyBuffer.PyBuf.ANY_CONTIGUOUS &&
                    !arr.IsOneSegment) {
                    throw new ArgumentException("ndarray is not contiguous");
                }
                if ((flags & NpyBuffer.PyBuf.STRIDES) != NpyBuffer.PyBuf.STRIDES &&
                    (flags & NpyBuffer.PyBuf.ND) == NpyBuffer.PyBuf.ND &&
                    !arr.ChkFlags(NpyDefs.NPY_C_CONTIGUOUS)) {
                    throw new ArgumentException("ndarray is not c-contiguous");
                }
                if ((flags & NpyBuffer.PyBuf.WRITABLE) == NpyBuffer.PyBuf.WRITABLE &&
                    !arr.IsWriteable) {
                    throw new ArgumentException("ndarray is not writable");
                }

                readOnly = ((flags & NpyBuffer.PyBuf.WRITABLE) == 0);
                ndim = ((flags & NpyBuffer.PyBuf.ND) == 0) ? 0 : arr.ndim;
                shape = ((flags & NpyBuffer.PyBuf.ND) == 0) ? null : arr.Dims;
                strides = ((flags & NpyBuffer.PyBuf.STRIDES) == 0) ? null : arr.Strides;

                if ((flags & NpyBuffer.PyBuf.FORMAT) == 0) {
                    // Force an array of unsigned bytes.
                    itemCount = arr.Size * arr.Dtype.ElementSize;
                    itemSize = sizeof(byte);
                    format = null;
                } else {
                    itemCount = arr.Length;
                    itemSize = arr.Dtype.ElementSize;
                    format = NpyCoreApi.GetBufferFormatString(arr);
                }
            }

            #region IExtBufferProtocol

            long IExtBufferProtocol.ItemCount {
                get { return itemCount; }
            }

            string IExtBufferProtocol.Format {
                get { return format; }
            }

            int IExtBufferProtocol.ItemSize {
                get { return itemSize; }
            }

            int IExtBufferProtocol.NumberDimensions {
                get { return ndim; }
            }

            bool IExtBufferProtocol.ReadOnly {
                get { return readOnly; }
            }

            IList<long> IExtBufferProtocol.Shape {
                get { return shape; }
            }

            long[] IExtBufferProtocol.Strides {
                get { return strides; }
            }

            long[] IExtBufferProtocol.SubOffsets {
                get { return null; }
            }

            IntPtr IExtBufferProtocol.UnsafeAddress {
                get { return arr.DataAddress; }
            }

            /// <summary>
            /// Total number of bytes in the array
            /// </summary>
            long IExtBufferProtocol.Size {
                get { return arr.Size; }
            }

            #endregion

            private readonly ndarray arr;
            private readonly bool readOnly;
            private readonly long itemCount;
            private readonly string format;
            private readonly int ndim;
            private readonly int itemSize;
            private readonly IList<long> shape;
            private readonly long[] strides;

        }

        #endregion
    }

    internal class ndarray_Enumerator : IEnumerator<object>
    {
        public ndarray_Enumerator(ndarray a) {
            arr = a;
            index = (IntPtr)(-1);
        }

        public object Current {
            get { return arr[index.ToPython()]; }
        }

        public void Dispose() {
            arr = null;
        }


        public bool MoveNext() {
            index += 1;
            return (index.ToInt64() < arr.Dims[0]);
        }

        public void Reset() {
            index = (IntPtr)(-1);
        }

        private ndarray arr;
        private IntPtr index;
    }
}
