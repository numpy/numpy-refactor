using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using IronPython.Runtime;
using IronPython.Runtime.Types;
using IronPython.Runtime.Operations;
using System.Runtime.InteropServices;
using System.Numerics;
using Microsoft.Scripting;

namespace NumpyDotNet
{
    [PythonType("numpy.generic")]
    public class ScalarGeneric : IArray, IConvertible
    {
        internal virtual ndarray ToArray() {
            return null;
        }

        public virtual object Value {
            get {
                throw new NotImplementedException(
                    String.Format("Internal error: Value has not been overridden for scalar type '{0}'", GetType().Name));
            }
        }

        /// <summary>
        /// Indicates whether the scalars have been "initialized" or not.  This is an
        /// unpleasant hack that mimicks that CPython behavior whereby the tp_new field for
        /// the scalar types is modified in the middle of initialization.
        /// </summary>
        static internal bool Initialized { get; set; }


        /// <summary>
        /// Fill the value with the value from the 0-d array
        /// </summary>
        /// <param name="arr"></param>
        internal virtual ScalarGeneric FillData(ndarray arr, long offset, bool nativeByteOrder) {
            return FillData((IntPtr)(arr.UnsafeAddress.ToInt64() + offset), 0, nativeByteOrder);
        }

        internal virtual ScalarGeneric FillData(IntPtr dataPtr, int size, bool nativeByteOrder) {
            throw new NotImplementedException();
        }

        public object __reduce__(CodeContext cntx) {
            object[] tupleValues = new object[2];

            PythonModule ma = (PythonModule)IronPython.Runtime.Operations.PythonOps.ImportBottom(cntx, "numpy.core.multiarray", 0);
            tupleValues[0] = ma.__getattribute__(cntx, "scalar");

            if (((dtype)dtype).isbuiltin == 0) { // TODO: Should be is scalar
                tupleValues[1] = new PythonTuple(new object[] { dtype, Value });
            } else {
                tupleValues[1] = null;
            }

            return new PythonTuple(tupleValues);
         }

        #region IArray interface

        public object __abs__(CodeContext cntx) {
            return ToArray().__abs__(cntx);
        }

        public object __len__() {
            return ToArray().__len__();
        }

        public virtual object __divmod__(CodeContext cntx, object b) {
            return ToArray().__divmod__(cntx, b);
        }

        public virtual object __rdivmod__(CodeContext cntx, object a) {
            return ToArray().__rdivmod__(cntx, a);
        }

        public object __floordiv__(CodeContext cntx, object o) {
            return ToArray().__floordiv__(cntx, o);
        }

        public object __truediv__(CodeContext cntx, object o) {
            return ToArray().__truediv__(cntx, o);
        }

        public object __lshift__(CodeContext cntx, object b) {
            return ToArray().__lshift__(cntx, b);
        }

        public virtual object __mod__(CodeContext cntx, object b) {
            return ToArray().__mod__(cntx, b);
        }

        public virtual object __repr__(CodeContext context) {
            return ToArray().__str__(context);
        }

        public object __rshift__(CodeContext cntx, object b) {
            return ToArray().__rshift__(cntx, b);
        }

        public object __sqrt__(CodeContext cntx) {
            return ToArray().__sqrt__(cntx);
        }

        public virtual object __str__(CodeContext context) {
            return ToArray().__str__(context);
        }

        public object all(object axis = null, ndarray @out = null) {
            return ToArray().all(axis, @out);
        }

        public object any(object axis = null, ndarray @out = null) {
            return ToArray().any(axis, @out);
        }

        public object argmax(object axis = null, ndarray @out = null) {
            return ToArray().argmax(axis, @out);
        }

        public object argmin(object axis = null, ndarray @out = null) {
            return ToArray().argmin(axis, @out);
        }

        public object argsort(object axis = null, string kind = null, object order = null) {
            return ToArray().argsort(axis, kind, order);
        }

        public ndarray astype(CodeContext cntx, object dtype = null) {
            return ToArray().astype(cntx, dtype);
        }

        public object @base {
            get { return null; }
        }

        public ndarray byteswap(bool inplace = false) {
            if (inplace) {
                throw new ArgumentException("cannot byteswap a scalar inplace");
            } else {
                // TODO: Fix to return a scalar
                return ToArray().byteswap(false);
            }
        }

        public object choose([ParamDictionary] IDictionary<object,object> kwargs, params object[] args) {
            return ToArray().choose(kwargs, args:args);
        }

        public object clip(object min = null, object max = null, ndarray @out = null) {
            return ToArray().clip(min, max, @out);
        }

        public ndarray compress(object condition, object axis = null, ndarray @out = null) {
            return ToArray().compress(condition, axis, @out);
        }

        public ndarray conj(ndarray @out = null) {
            return ToArray().conj(@out);
        }

        public ndarray conjugate(ndarray @out = null) {
            return ToArray().conjugate(@out);
        }

        public object copy(object order = null) {
            return ToArray().copy(order);
        }

        public object cumprod(CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null) {
            return ToArray().cumprod(cntx, axis, dtype, @out);
        }

        public object cumsum(CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null) {
            return ToArray().cumsum(cntx, axis, dtype, @out);
        }

        public PythonBuffer data {
            get {
                throw new NotImplementedException();
            }
        }

        public ndarray diagonal(int offset = 0, int axis1 = 0, int axis2 = 1) {
            return ToArray().diagonal(offset, axis1, axis2);
        }

        public virtual object dtype {
            get {
                return NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_VOID);
            }
            set {
                throw new ArgumentTypeException("array-scalars are immutable");
            }
        }

        public void fill(object scalar) {
            // TODO: This doesn't make any sense but is the same for CPython
            ToArray().fill(scalar);
        }

        public flagsobj flags {
            get { return new flagsobj(null); }
        }

        public object flat {
            get {
                return ToArray().flat;
            }
            set {
                throw new ArgumentTypeException("array-scalars are immutable");
            }
        }

        public ndarray flatten(object order = null) {
            return ToArray().flatten(order);
        }

        public ndarray getfield(CodeContext cntx, object dtype, int offset = 0) {
            return ToArray().getfield(cntx, dtype, offset);
        }

        public virtual object imag {
            get {
                return ndarray.ArrayReturn((ndarray)ToArray().imag);
            }
            set {
                throw new ArgumentTypeException("array-scalars are immutable");
            }
        }

        public object item(params object[] args) {
            return ToArray().item(args:args);
        }

        public void itemset(params object[] args) {
            throw new ArgumentTypeException("array-scalars are immutable");
        }

        public int itemsize {
            get { return ((dtype)dtype).itemsize; }
        }

        public object max(object axis = null, ndarray @out = null) {
            return ToArray().max(axis, @out);
        }

        public object mean(CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null) {
            return ToArray().mean(cntx, axis, dtype, @out);
        }

        public object min(object axis = null, ndarray @out = null) {
            return ToArray().min(axis, @out);
        }


        /// <summary>
        /// Size of the object in bytes
        /// </summary>
        public object nbytes {
            get { return this.itemsize; }
        }

        public int ndim {
            get {
                return 0;
            }
        }

        public ndarray newbyteorder(string endian = null) {
            return ToArray().newbyteorder(endian);
        }

        public PythonTuple nonzero() {
            return ToArray().nonzero();
        }

        public object prod(CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null) {
            return ToArray().prod(cntx, axis, dtype, @out);
        }

        public object ptp(object axis = null, ndarray @out = null) {
            return ToArray().ptp(axis, @out);
        }

        public void put(object indices, object values, object mode = null) {
            // TODO: This doesn't make any sense, but the CPython is the same.
            ToArray().put(indices, values, mode);
        }

        public ndarray ravel(object order = null) {
            return ToArray().ravel(order);
        }

        public virtual object real {
            get {
                return ndarray.ArrayReturn((ndarray)ToArray().real);
            }
            set {
                throw new ArgumentTypeException("array-scalars are immutable");
            }
        }

        public object repeat(object repeats, object axis = null) {
            return ToArray().repeat(repeats, axis);
        }

        public ndarray reshape([ParamDictionary]IDictionary<object,object> kwds, params object[] args) {
            return ToArray().reshape(args:args, kwds:kwds);
        }

        public void resize([ParamDictionary]IDictionary<object,object> kwds, params object[] args) {
            // TODO: This doesn't make any sense, but CPython does the same
            ToArray().resize(args:args, kwds:kwds);
        }

        public object round(int decimals = 0, ndarray @out = null) {
            return ToArray().round(decimals, @out);
        }

        public object searchsorted(object keys, string side = null) {
            return ToArray().searchsorted(keys, side);
        }

        public virtual void setfield(CodeContext cntx, object value, object dtype, int offset = 0) {
            throw new ArgumentTypeException("array-scalars are immutable");
        }

        public void setflags(object write = null, object align = null, object uic = null) {
            // CPython implementation simply does nothing, so we will too.
        }

        public object shape {
            get { return new PythonTuple(); }
        }

        public object size {
            get { return 1; }
        }

        public void sort(int axis = -1, string kind = null, object order = null) {
            // TODO: This doesn't make any sense, but CPython does the same.
            ToArray().sort(axis, kind, order);
        }

        public object squeeze() {
            return this;
        }

        public object std(CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null, int ddof = 0) {
            return ToArray().std(cntx, axis, dtype, @out, ddof);
        }

        public long[] Strides {
            get { return new long[0]; }
        }

        public PythonTuple strides {
            get { return NpyUtil_Python.ToPythonTuple(Strides); }
        }

        public object sum(CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null) {
            return ToArray().sum(cntx, axis, dtype, @out);
        }

        public ndarray swapaxes(int a1, int a2) {
            return ToArray().swapaxes(a1, a2);
        }

        public ndarray swapaxes(object a1, object a2) {
            return ToArray().swapaxes(a1, a2);
        }

        public static object Power(Object a, Object b) {
            if (a is double && b is double) {
                return Math.Pow((double)a, (double)b);
            } else if (a is double && b is ScalarFloat64) {
                return Math.Pow((double)a, (double)((ScalarFloat64)b).Value);
            } else {
                return NpyArray.FromAny(a).__pow__(NpyArray.FromAny(b));
            }
        }

        /// <summary>
        /// Returns the transpose of this object, for scalars there is no change.
        /// </summary>
        public object T {
            get { return this; }
        }


        public object take(object indices, object axis = null, ndarray @out = null, object mode = null) {
            return ToArray().take(indices, axis, @out, mode);
        }

        public object this[params object[] args] {
            get {
                return ToArray()[args: args];
            }
            set {
                throw new ArgumentTypeException("array-scalars are immutable");
            }
        }

        public virtual object this[int index] {
            get {
                return ToArray()[index];
            }
        }

        public virtual object this[long index] {
            get {
                return ToArray()[index];
            }
        }

        public virtual object this[IntPtr index] {
            get {
                return ToArray()[index];
            }
        }

        public virtual object this[System.Numerics.BigInteger index] {
            get {
                return ToArray()[index];
            }
        }

        public void tofile(CodeContext cntx, PythonFile file, string sep = null, string format = null) {
            ToArray().tofile(cntx, file, sep, format);
        }

        public void tofile(CodeContext cntx, string filename, string sep = null, string format = null) {
            ToArray().tofile(cntx, filename, sep, format);
        }

        public object tolist() {
            return ToArray().tolist();
        }

        public Bytes tostring(object order = null) {
            return ToArray().tostring(order);
        }

        public object trace(CodeContext cntx, int offset = 0, int axis1 = 0, int axis2 = 1, object dtype = null, ndarray @out = null) {
            return ToArray().trace(cntx, offset, axis1, axis2, dtype, @out);
        }

        public ndarray transpose(params object[] args) {
            return ToArray().transpose(args);
        }

        public object var(CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null, int ddof = 0) {
            return ToArray().var(cntx, axis, dtype, @out, ddof);
        }

        public ndarray view(CodeContext cntx, object dtype = null, object type = null) {
            return ToArray().view(cntx, dtype, type);
        }

        #endregion

        #region operators

        public static object operator +(ScalarGeneric a, object b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_add);
        }

        public static object operator +(ScalarGeneric a, ScalarGeneric b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_add);
        }

        public static object operator +(object a, ScalarGeneric b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_add);
        }

        public static object operator +(ScalarGeneric a) {
            return a;
        }

        public static object operator -(ScalarGeneric a, object b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_subtract);
        }

        public static object operator -(ScalarGeneric a, ScalarGeneric b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_subtract);
        }

        public static object operator -(object a, ScalarGeneric b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_subtract);
        }

        public static object operator -(ScalarGeneric a) {
            return ndarray.UnaryOp(null, a, NpyDefs.NpyArray_Ops.npy_op_negative);
        }

        public static object operator *(ScalarGeneric a, object b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_multiply);
        }

        public static object operator *(ScalarGeneric a, ScalarGeneric b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_multiply);
        }

        public static object operator *(object a, ScalarGeneric b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_multiply);
        }

        public static object operator /(ScalarGeneric a, object b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_divide);
        }

        public static object operator /(ScalarGeneric a, ScalarGeneric b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_divide);
        }

        public static object operator /(object a, ScalarGeneric b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_divide);
        }

        public object __pow__(object a) {
            return ndarray.BinaryOp(null, this, a, NpyDefs.NpyArray_Ops.npy_op_power);
        }

        public static object operator &(ScalarGeneric a, object b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_and);
        }

        public static object operator &(ScalarGeneric a, ScalarGeneric b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_and);
        }

        public static object operator &(object a, ScalarGeneric b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_and);
        }

        public static object operator |(ScalarGeneric a, object b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_or);
        }

        public static object operator |(ScalarGeneric a, ScalarGeneric b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_or);
        }

        public static object operator |(object a, ScalarGeneric b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_or);
        }

        public static object operator ^(ScalarGeneric a, object b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_xor);
        }

        public static object operator ^(ScalarGeneric a, ScalarGeneric b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_xor);
        }

        public static object operator ^(object a, ScalarGeneric b) {
            return ndarray.BinaryOp(null, a, b, NpyDefs.NpyArray_Ops.npy_op_bitwise_xor);
        }

        public static object operator ~(ScalarGeneric a) {
            return ndarray.UnaryOp(null, a, NpyDefs.NpyArray_Ops.npy_op_invert);
        }

        // NOTE: For comparison operators we use the Python names
        // since these operators usually return boolean arrays and
        // .NET seems to expect them to return bool

        public virtual object __eq__(CodeContext cntx, object o) {
            return ToArray().__eq__(cntx, o);
        }

        public object __req__(CodeContext cntx, object o) {
            return ToArray().__req__(cntx, o);
        }

        public object __ne__(CodeContext cntx, object o) {
            return ToArray().__ne__(cntx, o);
        }

        public object __rne__(CodeContext cntx, object o) {
            return ToArray().__rne__(cntx, o);
        }

        public object __lt__(CodeContext cntx, object o) {
            return ToArray().__lt__(cntx, o);
        }

        public object __rlt__(CodeContext cntx, object o) {
            return ToArray().__rlt__(cntx, o);
        }

        public object __le__(CodeContext cntx, object o) {
            return ToArray().__le__(cntx, o);
        }

        public object __rle__(CodeContext cntx, object o) {
            return ToArray().__rle__(cntx, o);
        }

        public object __gt__(CodeContext cntx, object o) {
            return ToArray().__gt__(cntx, o);
        }

        public object __rgt__(CodeContext cntx, object o) {
            return ToArray().__rgt__(cntx, o);
        }

        public object __ge__(CodeContext cntx, object o) {
            return ToArray().__ge__(cntx, o);
        }

        public object __rge__(CodeContext cntx, object o) {
            return ToArray().__rge__(cntx, o);
        }

        public virtual object __int__(CodeContext cntx) {
            return ToArray().__int__(cntx);
        }

        public virtual object __long__(CodeContext cntx) {
            return ToArray().__long__(cntx);
        }

        public virtual object __float__(CodeContext cntx) {
            return ToArray().__float__(cntx);
        }

        public virtual object __complex__(CodeContext cntx) {
            return ToArray().__complex__(cntx);
        }

        public bool __nonzero__() {
            return (bool)ToArray();
        }

        public static explicit operator bool(ScalarGeneric s) {
            return (bool)s.ToArray();
        }

        #endregion

        internal static dtype GetDtype(int size, char typechar) {
            if (typechar == 'U') {
                dtype d = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_UNICODE);
                d = NpyCoreApi.DescrNew(d);
                d.ElementSize = size * 4;
                return d;
            } else if (typechar == 'S') {
                dtype d = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_STRING);
                d = NpyCoreApi.DescrNew(d);
                d.ElementSize = size;
                return d;
            } else {
                NpyDefs.NPY_TYPES t = NpyCoreApi.TypestrConvert(size, (byte)typechar);
                return NpyCoreApi.DescrFromType(t);
            }
        }

        internal static object ScalarFromData(dtype type, IntPtr data, int size) {
            return type.ToScalar(data, size);
        }

        #region IConvertible

        public virtual bool ToBoolean(IFormatProvider fp=null) {
            throw new NotImplementedException();
        }

        public virtual byte ToByte(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual char ToChar(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual DateTime ToDateTime(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual Decimal ToDecimal(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual Double ToDouble(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual Int16 ToInt16(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual Int32 ToInt32(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual Int64 ToInt64(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual SByte ToSByte(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual Single ToSingle(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual String ToString(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual Object ToType(Type t, IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual UInt16 ToUInt16(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual UInt32 ToUInt32(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual UInt64 ToUInt64(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual TypeCode GetTypeCode() {
            throw new NotImplementedException();
        }

        #endregion
    }

    [PythonType("numpy.bool_")]
    public class ScalarBool : ScalarGeneric
    {
        public static object __new__(PythonType cls) {
            return FALSE;
        }

        public static object __new__(PythonType cls, bool val) {
            return val ? TRUE : FALSE;
        }

        public static object __new__(PythonType cls, object val) {
            ndarray arr = NpyArray.FromAny(val, descr: NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_BOOL),
                                            flags: NpyDefs.NPY_FORCECAST);
            if (arr.ndim == 0) {
                byte b = Marshal.ReadByte(arr.UnsafeAddress);
                return __new__(cls, b != 0);
            } else {
                // TODO: I don't know why we do this here. It means that
                // np.bool_([True, False]) returns an array, not a scalar.
                // This matches the behavior in CPython.
                return ndarray.ArrayReturn(arr);
            }
        }

        public ScalarBool() {
            value = false;
        }

        public ScalarBool(bool val) {
            value = val;
        }

        public override object Value { get { return value; } }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_BOOL);
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            Marshal.WriteByte(result.UnsafeAddress, (byte)(value ? 1 : 0));
            return result;
        }

        internal override ScalarGeneric FillData(IntPtr p, int size, bool isNativeByteOrder) {
            value = (Marshal.ReadByte(p) != 0);
            return (value ? TRUE : FALSE);
        }

        public new bool __nonzero__() {
            return value;
        }

        public static implicit operator bool(ScalarBool s) {
            return s.value;
        }

        public override object __eq__(CodeContext cntx, object o) {
            if (o is Boolean) {
                return value == (Boolean)o;
            } else if (o is ScalarBool) {
                return value == ((ScalarBool)o).value;
            } else if (o is IConvertible) {
                try {
                    bool other = ((IConvertible)o).ToBoolean(null);
                    return value == other;
                } catch { }
            }
            return ToArray().__eq__(cntx, o);
        }


        #region IConvertible

        public override bool ToBoolean(IFormatProvider fp=null) {
            return value;
        }

        public override Int16 ToInt16(IFormatProvider fp = null) {
            return value ? (short)1 : (short)0;
        }

        public override Int32 ToInt32(IFormatProvider fp = null) {
            return value ? 1 : 0;
        }

        public override Int64 ToInt64(IFormatProvider fp = null) {
            return value ? 1 : 0;
        }

        public override UInt16 ToUInt16(IFormatProvider fp = null) {
            return value ? (UInt16)1 : (UInt16)0;
        }

        public override UInt32 ToUInt32(IFormatProvider fp = null) {
            return value ? 1u : 0u;
        }

        public override UInt64 ToUInt64(IFormatProvider fp = null) {
            return value ? 1U : 0U;
        }


        public override String ToString(IFormatProvider fp = null) {
            return value.ToString();
        }

        #endregion

        private bool value;
        static private dtype dtype_;

        static private readonly ScalarBool FALSE = new ScalarBool(false);
        static private readonly ScalarBool TRUE = new ScalarBool(true);
    }

    [PythonType("numpy.number")]
    public class ScalarNumber : ScalarGeneric
    {
        public override object dtype {
            get {
                return NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_DOUBLE);
            }
        }
    }

    [PythonType("numpy.integer")]
    public class ScalarInteger : ScalarNumber
    {
        public override object dtype {
            get {
                return NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_LONG);
            }
        }
    }

    [PythonType("numpy.signedinteger")]
    public class ScalarSignedInteger : ScalarInteger {  }

    public class ScalarIntegerImpl<T> : ScalarInteger where T : IConvertible
    {
        protected T value;

        public override object Value { get { return value; } }

        #region IConvertible

        public override bool ToBoolean(IFormatProvider fp = null) {
            return value.ToBoolean(fp);
        }

        public override byte ToByte(IFormatProvider fp = null) {
            return value.ToByte(fp);
        }

        public override char ToChar(IFormatProvider fp = null) {
            return value.ToChar(fp);
        }

        public override Decimal ToDecimal(IFormatProvider fp = null) {
            return value.ToDecimal(fp);
        }

        public override Double ToDouble(IFormatProvider fp = null) {
            return value.ToDouble(fp);
        }

        public override Int16 ToInt16(IFormatProvider fp = null) {
            return value.ToInt16(fp);
        }

        public override Int32 ToInt32(IFormatProvider fp = null) {
            return value.ToInt32(fp);
        }

        public override Int64 ToInt64(IFormatProvider fp = null) {
            return value.ToInt64(fp);
        }

        public override SByte ToSByte(IFormatProvider fp = null) {
            return value.ToSByte(fp);
        }

        public override Single ToSingle(IFormatProvider fp = null) {
            return value.ToSingle(fp);
        }

        public override UInt16 ToUInt16(IFormatProvider fp = null) {
            return value.ToUInt16(fp);
        }

        public override UInt32 ToUInt32(IFormatProvider fp = null) {
            return value.ToUInt32(fp);
        }

        public override UInt64 ToUInt64(IFormatProvider fp = null) {
            return value.ToUInt64(fp);
        }

        #endregion

    }

    [PythonType("numpy.int8")]
    public class ScalarInt8 : ScalarIntegerImpl<sbyte>
    {
        public ScalarInt8() {
            value = 0;
        }

        public ScalarInt8(sbyte value) {
            this.value = value;
        }

        public ScalarInt8(IConvertible value) {
            this.value = Convert.ToSByte(value);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_BYTE);
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            Marshal.WriteByte(result.UnsafeAddress, (byte)value);
            return result;
        }

        internal override ScalarGeneric FillData(IntPtr p, int size, bool isNativeByteOrder) {
            value = (sbyte)Marshal.ReadByte(p);
            return this;
        }

        public static implicit operator int(ScalarInt8 i) {
            return i.value;
        }

        public static implicit operator BigInteger(ScalarInt8 i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarInt8 i) {
            return i.value;
        }

        public new bool __nonzero__() {
            return value != 0;
        }

        public static explicit operator bool(ScalarInt8 s) {
            return s.value != 0;
        }


        static private dtype dtype_;

        internal static readonly int MinValue = sbyte.MinValue;
        internal static readonly int MaxValue = sbyte.MaxValue;
    }

    [PythonType("numpy.int16")]
    public class ScalarInt16 : ScalarIntegerImpl<Int16>
    {
        public ScalarInt16() {
            value = 0;
        }

        public ScalarInt16(Int16 value) {
            this.value = value;
        }

        public ScalarInt16(string value, int @base = 10) {
            this.value = Convert.ToInt16(value, @base);
        }


        public ScalarInt16(IConvertible value) {
            this.value = Convert.ToInt16(value);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = GetDtype(2, 'i');
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            Marshal.WriteInt16(result.UnsafeAddress, value);
            return result;
        }

        internal override ScalarGeneric FillData(IntPtr p, int size, bool isNativeByteOrder) {
            unsafe {
                fixed (void* data = &value) {
                    NpyCoreApi.CopySwapScalar((dtype)dtype, data, (void *)p, !isNativeByteOrder);
                }
            }
            return this;
        }

        public static implicit operator int(ScalarInt16 i) {
            return i.value;
        }

        public static implicit operator BigInteger(ScalarInt16 i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarInt16 i) {
            return i.value;
        }

        public new bool __nonzero__() {
            return value != 0;
        }

        public static explicit operator bool(ScalarInt16 s) {
            return s.value != 0;
        }

        public object __index__() {
            return value;
        }

        static private dtype dtype_;

        internal static readonly int MinValue = Int16.MinValue;
        internal static readonly int MaxValue = Int16.MaxValue;
    }

    [PythonType("numpy.int32")]
    public class ScalarInt32 : ScalarIntegerImpl<Int32>
    {
        public ScalarInt32() {
            value = 0;
        }

        public ScalarInt32(Int32 value) {
            this.value = value;
        }

        public ScalarInt32(string value, int @base = 10) {
            this.value = Convert.ToInt32(value, @base);
        }

        public ScalarInt32(IConvertible value) {
            this.value = Convert.ToInt32(value);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NpyCoreApi.TypeOf_Int32);
                        }
                    }
                }
                return dtype_;
            }
        }

        public override object Value { get { return value; } }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            Marshal.WriteInt32(result.UnsafeAddress, value);
            return result;
        }

        internal override ScalarGeneric FillData(IntPtr dataPtr, int size, bool isNativeByteOrder) {
            unsafe {
                fixed (void* data = &value) {
                    NpyCoreApi.CopySwapScalar((dtype)dtype, data, (void*)dataPtr, !isNativeByteOrder);
                }
            }
            return this;
        }

        public static implicit operator int(ScalarInt32 i) {
            return i.value;
        }

        public static implicit operator BigInteger(ScalarInt32 i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarInt32 i) {
            return i.value;
        }

        public new bool __nonzero__() {
            return value != 0;
        }

        public static explicit operator bool(ScalarInt32 s) {
            return s.value != 0;
        }

        public object __index__() {
            return value;
        }

        static private dtype dtype_;

        internal static readonly int MinValue = Int32.MinValue;
        internal static readonly int MaxValue = Int32.MaxValue;
    }


    /// <summary>
    /// This is a fairly ugly workaround to an issue with scalars on IronPython.  Each int scalar
    /// represents a specific size integer (8, 16, 32, or 6 bits). However, in the core there are
    /// five types - byte, short, int, long, and longlong with two being the same size based on
    /// platform (32-bit int == long, 64-bit long == longlong). This lets us represent an int were
    /// int and long (int32) are the same size.
    /// </summary>
    [PythonType("numpy.intc")]
    public class ScalarIntC : ScalarInt32
    {
        public ScalarIntC() {
            value = 0;
        }

        public ScalarIntC(Int32 value) : base(value) {
        }

        public ScalarIntC(string value, int @base = 10) : base(value, @base) {
        }

        public ScalarIntC(IConvertible value) : base(value) {
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_INT);
                        }
                    }
                }
                return dtype_;
            }
        }

        static private dtype dtype_;

        internal static readonly int MinValue = Int32.MinValue;
        internal static readonly int MaxValue = Int32.MaxValue;
    }
    
    [PythonType("numpy.int64")]
    public class ScalarInt64 : ScalarIntegerImpl<Int64>
    {
        public ScalarInt64() {
            value = 0;
        }

        public ScalarInt64(Int64 value) {
            this.value = value;
        }

        public ScalarInt64(string value, int @base = 10) {
            this.value = Convert.ToInt64(value, @base);
        }


        public ScalarInt64(IConvertible value) {
            this.value = Convert.ToInt64(value);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = GetDtype(8, 'i');
                        }
                    }
                }
                return dtype_;
            }
        }

        public override object Value { get { return value; } }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            Marshal.WriteInt64(result.UnsafeAddress, value);
            return result;
        }

        internal override ScalarGeneric FillData(IntPtr dataPtr, int size, bool isNativeByteOrder) {
            unsafe {
                fixed (void* data = &value) {
                    NpyCoreApi.CopySwapScalar((dtype)dtype, data, (void*)dataPtr, !isNativeByteOrder);
                }
            }
            return this;
        }

        public static explicit operator int(ScalarInt64 i) {
            if (i < int.MinValue || i > int.MaxValue) {
                throw new OverflowException();
            }
            return (int)i.value;
        }

        public static implicit operator BigInteger(ScalarInt64 i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarInt64 i) {
            return i.value;
        }

        public new bool __nonzero__() {
            return value != 0;
        }

        public static explicit operator bool(ScalarInt64 s) {
            return s.value != 0;
        }

        public object __index__() {
            return value;
        }

        static private dtype dtype_;

        internal static readonly BigInteger MinValue = new BigInteger(Int64.MinValue);
        internal static readonly BigInteger MaxValue = new BigInteger(Int64.MaxValue);
    }


    /// <summary>
    /// This is a fairly ugly workaround to an issue with scalars on IronPython.  Each int scalar
    /// represents a specific size integer (8, 16, 32, or 6 bits). However, in the core there are
    /// five types - byte, short, int, long, and longlong with two being the same size based on
    /// platform (32-bit int == long, 64-bit long == longlong). This lets us represent an int were
    /// int and long (int32) are the same size.
    /// </summary>
    [PythonType("numpy.longlong")]
    public class ScalarLongLong : ScalarInt64
    {
        public ScalarLongLong() {
            value = 0;
        }

        public ScalarLongLong(Int64 value)
            : base(value) {
        }

        public ScalarLongLong(IConvertible value)
            : base(value) {
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_LONGLONG);
                        }
                    }
                }
                return dtype_;
            }
        }

        static private dtype dtype_;

        internal static new readonly long MinValue = Int32.MinValue;
        internal static new readonly long MaxValue = Int32.MaxValue;
    }


    [PythonType("numpy.unsignedinteger")]
    public class ScalarUnsignedInteger : ScalarInteger
    {
        public override object dtype {
            get {
                return NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_ULONG);
            }
        }
    }

    public class ScalarUnsignedImpl<T> : ScalarUnsignedInteger where T : IConvertible
    {
        protected T value;

        public override object Value { get { return value; } }

        #region IConvertible

        public override bool ToBoolean(IFormatProvider fp = null) {
            return value.ToBoolean(fp);
        }

        public override byte ToByte(IFormatProvider fp = null) {
            return value.ToByte(fp);
        }

        public override char ToChar(IFormatProvider fp = null) {
            return value.ToChar(fp);
        }

        public override Decimal ToDecimal(IFormatProvider fp = null) {
            return value.ToDecimal(fp);
        }

        public override Double ToDouble(IFormatProvider fp = null) {
            return value.ToDouble(fp);
        }

        public override Int16 ToInt16(IFormatProvider fp = null) {
            return value.ToInt16(fp);
        }

        public override Int32 ToInt32(IFormatProvider fp = null) {
            return value.ToInt32(fp);
        }

        public override Int64 ToInt64(IFormatProvider fp = null) {
            return value.ToInt64(fp);
        }

        public override SByte ToSByte(IFormatProvider fp = null) {
            return value.ToSByte(fp);
        }

        public override Single ToSingle(IFormatProvider fp = null) {
            return value.ToSingle(fp);
        }

        public override UInt16 ToUInt16(IFormatProvider fp = null) {
            return value.ToUInt16(fp);
        }

        public override UInt32 ToUInt32(IFormatProvider fp = null) {
            return value.ToUInt32(fp);
        }

        public override UInt64 ToUInt64(IFormatProvider fp = null) {
            return value.ToUInt64(fp);
        }

        #endregion

    }

    [PythonType("numpy.uint8")]
    public class ScalarUInt8 : ScalarUnsignedImpl<byte>
    {
        public ScalarUInt8() {
            value = 0;
        }

        public ScalarUInt8(byte value) {
            this.value = value;
        }

        public ScalarUInt8(IConvertible value) {
            try {
                this.value = Convert.ToByte(value);
            } catch (OverflowException) {
                this.value = Byte.MaxValue;
            }
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = GetDtype(1, 'u');
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            Marshal.WriteByte(result.UnsafeAddress, value);
            return result;
        }

        internal override ScalarGeneric FillData(ndarray arr, long offset, bool isNativeByteOrder) {
            IntPtr p = (IntPtr)(arr.UnsafeAddress.ToInt64() + offset);
            value = Marshal.ReadByte(p);
            return this;
        }

        internal override ScalarGeneric FillData(IntPtr dataPtr, int size, bool isNativeByteOrder) {
            value = Marshal.ReadByte(dataPtr);
            return this;
        }

        public static implicit operator int(ScalarUInt8 i) {
            return i.value;
        }

        public static implicit operator BigInteger(ScalarUInt8 i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarUInt8 i) {
            return i.value;
        }

        public new bool __nonzero__() {
            return value != 0;
        }

        public static explicit operator bool(ScalarUInt8 s) {
            return s.value != 0;
        }

        static private dtype dtype_;

        internal static readonly int MinValue = 0;
        internal static readonly int MaxValue = byte.MaxValue;
    }

    [PythonType("numpy.uint16")]
    public class ScalarUInt16 : ScalarUnsignedImpl<UInt16>
    {
        public ScalarUInt16() {
            value = 0;
        }

        public ScalarUInt16(UInt16 value) {
            this.value = value;
        }

        public ScalarUInt16(int value) {
            this.value = (ushort)(short)value;
        }

        public ScalarUInt16(IConvertible value) {
            this.value = Convert.ToUInt16(value);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = GetDtype(2, 'u');
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            Marshal.WriteInt16(result.UnsafeAddress, (Int16)value);
            return result;
        }

        internal override ScalarGeneric FillData(IntPtr dataPtr, int size, bool isNativeByteOrder) {
            unsafe {
                fixed (void* data = &value) {
                    NpyCoreApi.CopySwapScalar((dtype)dtype, data, (void*)dataPtr, !isNativeByteOrder);
                }
            }
            return this;
        }

        public static implicit operator int(ScalarUInt16 i) {
            return i.value;
        }

        public static implicit operator BigInteger(ScalarUInt16 i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarUInt16 i) {
            return i.value;
        }

        public new bool __nonzero__() {
            return value != 0;
        }

        public static explicit operator bool(ScalarUInt16 s) {
            return s.value != 0;
        }

        static private dtype dtype_;

        internal static readonly int MinValue = 0;
        internal static readonly int MaxValue = UInt16.MaxValue;
    }

    [PythonType("numpy.uint32")]
    public class ScalarUInt32 : ScalarUnsignedImpl<UInt32>
    {
        public ScalarUInt32() {
            value = 0;
        }

        public ScalarUInt32(UInt32 value) {
            this.value = value;
        }

        public ScalarUInt32(int value) {
            this.value = (uint)value;
        }

        public ScalarUInt32(IConvertible value) {
            this.value = Convert.ToUInt32(value);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NpyCoreApi.TypeOf_UInt32);
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            Marshal.WriteInt32(result.UnsafeAddress, (Int32)value);
            return result;
        }

        internal override ScalarGeneric FillData(IntPtr dataPtr, int size, bool isNativeByteOrder) {
            unsafe {
                fixed (void* data = &value) {
                    NpyCoreApi.CopySwapScalar((dtype)dtype, data, (void*)dataPtr, !isNativeByteOrder);
                }
            }
            return this;
        }

        public static explicit operator int(ScalarUInt32 i) {
            if (i.value > int.MaxValue) {
                throw new OverflowException();
            }
            return (int)i.value;
        }

        public static implicit operator BigInteger(ScalarUInt32 i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarUInt32 i) {
            return i.value;
        }

        public new bool __nonzero__() {
            return value != 0;
        }

        public static explicit operator bool(ScalarUInt32 s) {
            return s.value != 0;
        }

        static private dtype dtype_;

        internal static readonly int MinValue = 0;
        internal static readonly BigInteger MaxValue = new BigInteger(UInt32.MaxValue);
    }



    /// <summary>
    /// This is a fairly ugly workaround to an issue with scalars on IronPython.  Each int scalar
    /// represents a specific size integer (8, 16, 32, or 6 bits). However, in the core there are
    /// five types - byte, short, int, long, and longlong with two being the same size based on
    /// platform (32-bit int == long, 64-bit long == longlong). This lets us represent an int were
    /// int and long (int32) are the same size.
    /// </summary>
    [PythonType("numpy.uintc")]
    public class ScalarUIntC : ScalarUInt32
    {
        public ScalarUIntC() {
            value = 0;
        }

        public ScalarUIntC(UInt32 value)
            : base(value) {
        }

        public ScalarUIntC(IConvertible value)
            : base(value) {
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_UINT);
                        }
                    }
                }
                return dtype_;
            }
        }

        static private dtype dtype_;

        internal static new readonly uint MinValue = UInt32.MinValue;
        internal static new readonly uint MaxValue = UInt32.MaxValue;
    }

    [PythonType("numpy.uint64")]
    public class ScalarUInt64 : ScalarUnsignedImpl<UInt64>
    {
        public ScalarUInt64() {
            value = 0;
        }

        public ScalarUInt64(UInt64 value) {
            this.value = value;
        }

        public ScalarUInt64(int value) {
            this.value = (ulong)(long)value;    // Cast to signed long then reinterpret bits into ulong so -2 converts to correct (big) value.
        }

        public ScalarUInt64(long value) {
            this.value = (ulong)value;
        }

        public ScalarUInt64(BigInteger value) {
            this.value = (ulong)value;
        }

        public ScalarUInt64(IConvertible value) {
            this.value = Convert.ToUInt64(value);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NpyCoreApi.TypeOf_UInt64);
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            Marshal.WriteInt64(result.UnsafeAddress, (Int64)value);
            return result;
        }

        internal override ScalarGeneric FillData(IntPtr dataPtr, int size, bool isNativeByteOrder) {
            unsafe {
                fixed (void* data = &value) {
                    NpyCoreApi.CopySwapScalar((dtype)dtype, data, (void*)dataPtr, !isNativeByteOrder);
                }
            }
            return this;
        }

        public static explicit operator int(ScalarUInt64 i) {
            if (i.value > int.MaxValue) {
                throw new OverflowException();
            }
            return (int)i.value;
        }

        public static implicit operator BigInteger(ScalarUInt64 i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarUInt64 i) {
            return i.value;
        }

        public new bool __nonzero__() {
            return value != 0;
        }

        public static explicit operator bool(ScalarUInt64 s) {
            return s.value != 0;
        }

        static private dtype dtype_;

        internal static readonly int MinValue = 0;
        internal static readonly BigInteger MaxValue = new BigInteger(UInt64.MaxValue);
    }

    /// <summary>
    /// This is a fairly ugly workaround to an issue with scalars on IronPython.  Each int scalar
    /// represents a specific size integer (8, 16, 32, or 6 bits). However, in the core there are
    /// five types - byte, short, int, long, and longlong with two being the same size based on
    /// platform (32-bit int == long, 64-bit long == longlong). This lets us represent an int were
    /// int and long (int32) are the same size.
    /// </summary>
    [PythonType("numpy.ulonglong")]
    public class ScalarULongLong : ScalarUInt64
    {
        public ScalarULongLong() {
            value = 0;
        }

        public ScalarULongLong(UInt64 value)
            : base(value) {
        }

        public ScalarULongLong(IConvertible value)
            : base(value) {
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_ULONGLONG);
                        }
                    }
                }
                return dtype_;
            }
        }

        static private dtype dtype_;

        internal static new readonly ulong MinValue = UInt64.MinValue;
        internal static new readonly ulong MaxValue = UInt64.MaxValue;
    }



    [PythonType("numpy.timeinteger")]
    public class ScalarTimeInteger : ScalarInt64 { }


    [PythonType("numpy.inexact")]
    public class ScalarInexact : ScalarNumber { }

    [PythonType("numpy.floating")]
    public class ScalarFloating : ScalarInexact { }

    public class ScalarFloatingImpl<T> : ScalarFloating where T : IConvertible
    {
        protected T value;

        public override object __repr__(CodeContext context) {
            // IronPython has its own float formatter that is slightly different than .NET
            // and causes test failures if we don't call it.
            return NpyUtil_Python.CallBuiltin(context, "str", value);
        }

        public override object __str__(CodeContext context) {
            // IronPython has its own float formatter that is slightly different than .NET
            // and causes test failures if we don't call it.
            return NpyUtil_Python.CallBuiltin(context, "str", value);
        }


        public override object Value { get { return value; } }

        #region IConvertible

        public override bool ToBoolean(IFormatProvider fp = null) {
            return value.ToBoolean(fp);
        }

        public override byte ToByte(IFormatProvider fp = null) {
            return value.ToByte(fp);
        }

        public override char ToChar(IFormatProvider fp = null) {
            return value.ToChar(fp);
        }

        public override Decimal ToDecimal(IFormatProvider fp = null) {
            return value.ToDecimal(fp);
        }

        public override Double ToDouble(IFormatProvider fp = null) {
            return value.ToDouble(fp);
        }

        public override Int16 ToInt16(IFormatProvider fp = null) {
            return value.ToInt16(fp);
        }

        public override Int32 ToInt32(IFormatProvider fp = null) {
            return value.ToInt32(fp);
        }

        public override Int64 ToInt64(IFormatProvider fp = null) {
            return value.ToInt64(fp);
        }

        public override SByte ToSByte(IFormatProvider fp = null) {
            return value.ToSByte(fp);
        }

        public override Single ToSingle(IFormatProvider fp = null) {
            return value.ToSingle(fp);
        }

        public override UInt16 ToUInt16(IFormatProvider fp = null) {
            return value.ToUInt16(fp);
        }

        public override UInt32 ToUInt32(IFormatProvider fp = null) {
            return value.ToUInt32(fp);
        }

        public override UInt64 ToUInt64(IFormatProvider fp = null) {
            return value.ToUInt64(fp);
        }

        #endregion
    }

    [PythonType("numpy.float32")]
    public class ScalarFloat32 : ScalarFloatingImpl<Single>
    {
        public ScalarFloat32() {
            value = 0;
        }

        public ScalarFloat32(Single value) {
            this.value = value;
        }

        public ScalarFloat32(IConvertible value) {
            this.value = Convert.ToSingle(value);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = GetDtype(4, 'f');
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            unsafe {
                Single* p = (Single*)result.UnsafeAddress.ToPointer();
                *p = value;
            }
            return result;
        }

        internal override ScalarGeneric FillData(IntPtr dataPtr, int size, bool isNativeByteOrder) {
            unsafe {
                fixed (void* data = &value) {
                    NpyCoreApi.CopySwapScalar((dtype)dtype, data, (void*)dataPtr, !isNativeByteOrder);
                }
            }
            return this;
        }

        public static explicit operator int(ScalarFloat32 i) {
            if (i.value < int.MinValue || i.value > int.MaxValue) {
                throw new OverflowException();
            }
            return (int)i.value;
        }

        public static implicit operator BigInteger(ScalarFloat32 i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarFloat32 i) {
            return i.value;
        }

        public static implicit operator Complex(ScalarFloat32 x) {
            return new Complex(x.value, 0.0);
        }

        public new bool __nonzero__() {
            return value != 0;
        }

        public static explicit operator bool(ScalarFloat32 s) {
            return s.value != 0;
        }

        public virtual object __eq__(CodeContext cntx, ScalarFloat32 o) {
            return value == o.value;
        }

        public virtual object __eq__(CodeContext cntx, float o) {
            return value == o;
        }

        public virtual object __ne__(CodeContext cntx, ScalarFloat32 o) {
            return value != o.value;
        }

        public virtual object __ne__(CodeContext cntx, float o) {
            return value != o;
        }

        public virtual object __lt__(CodeContext cntx, ScalarFloat32 o) {
            return value < o.value;
        }

        public virtual object __lt__(CodeContext cntx, float o) {
            return value < o;
        }

        public virtual object __le__(CodeContext cntx, ScalarFloat32 o) {
            return value <= o.value;
        }

        public virtual object __le__(CodeContext cntx, float o) {
            return value <= o;
        }

        public virtual object __gt__(CodeContext cntx, ScalarFloat32 o) {
            return value > o.value;
        }

        public virtual object __gt__(CodeContext cntx, float o) {
            return value > o;
        }

        public virtual object __ge__(CodeContext cntx, ScalarFloat32 o) {
            return value >= o.value;
        }

        public virtual object __ge__(CodeContext cntx, float o) {
            return value >= o;
        }

        public override object __repr__(CodeContext context) {
            // Format code 'R' is important because it cause all digits to be generated.
            // Primarily important for double type.
            return value.ToString("R");
        }

        static private dtype dtype_;
    }

    [PythonType("numpy.float64")]
    public class ScalarFloat64 : ScalarFloatingImpl<Double>
    {
        public ScalarFloat64() {
            value = 0;
        }

        public ScalarFloat64(Double value) {
            this.value = value;
        }

        public ScalarFloat64(IConvertible value) {
            this.value = Convert.ToSingle(value);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = GetDtype(8, 'f');
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            unsafe {
                Double* p = (Double*)result.UnsafeAddress.ToPointer();
                *p = value;
            }
            return result;
        }

        internal override ScalarGeneric FillData(IntPtr dataPtr, int size, bool isNativeByteOrder) {
            unsafe {
                fixed (void* data = &value) {
                    NpyCoreApi.CopySwapScalar((dtype)dtype, data, (void*)dataPtr, !isNativeByteOrder);
                }
            }
            return this;
        }

        public static explicit operator int(ScalarFloat64 i) {
            if (i.value < int.MinValue || i.value > int.MaxValue) {
                throw new OverflowException();
            }
            return (int)i.value;
        }

        public static implicit operator BigInteger(ScalarFloat64 i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarFloat64 i) {
            return i.value;
        }

        public static implicit operator Complex(ScalarFloat64 x) {
            return new Complex(x.value, 0.0);
        }

        public new bool __nonzero__() {
            return value != 0;
        }

        public static explicit operator bool(ScalarFloat64 s) {
            return s.value != 0;
        }


        public virtual object __eq__(CodeContext cntx, ScalarFloat64 o) {
            return value == o.value;
        }

        public virtual object __eq__(CodeContext cntx, double o) {
            return value == o;
        }

        public virtual object __ne__(CodeContext cntx, ScalarFloat64 o) {
            return value != o.value;
        }

        public virtual object __ne__(CodeContext cntx, double o) {
            return value != o;
        }

        public virtual object __lt__(CodeContext cntx, ScalarFloat64 o) {
            return value < o.value;
        }

        public virtual object __lt__(CodeContext cntx, double o) {
            return value < o;
        }

        public virtual object __le__(CodeContext cntx, ScalarFloat64 o) {
            return value <= o.value;
        }

        public virtual object __le__(CodeContext cntx, double o) {
            return value <= o;
        }

        public virtual object __gt__(CodeContext cntx, ScalarFloat64 o) {
            return value > o.value;
        }

        public virtual object __gt__(CodeContext cntx, double o) {
            return value > o;
        }

        public virtual object __ge__(CodeContext cntx, ScalarFloat64 o) {
            return value >= o.value;
        }

        public virtual object __ge__(CodeContext cntx, double o) {
            return value >= o;
        }

        public override object __repr__(CodeContext context) {
            // Format code 'R' is important because it cause all digits to be generated;
            // by default the format only goes up to 15 digits but double can represent
            // 17 digits.
            return value.ToString("R");
        }


        static private dtype dtype_;
    }

    [PythonType("numpy.complexfloating")]
    public class ScalarComplexFloating : ScalarInexact
    {
        public override object dtype {
            get {
                return NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_CDOUBLE);
            }
        }

        protected static void EmitComplexWarning(CodeContext cntx) {
            object warn = NpyUtil_Python.GetModuleAttr(cntx, "numpy.core", "ComplexWarning");
            if (warn == null) {
                throw new IronPython.Runtime.Exceptions.ImportException("Error importing numpy.core.ComplexWarning");
            }
            NpyUtil_Python.Warn((PythonType)warn, "Casting complex values to real discards the imaginary part", 1);
        }
    }


    [PythonType("numpy.complex64")]
    public class ScalarComplex64 : ScalarComplexFloating
    {
        public ScalarComplex64() {
            value.Real = 0.0f;
            value.Imag = 0.0f;
        }

        public ScalarComplex64(object o) {
            SetFromObj(o);
        }

        public ScalarComplex64(float real, float imag) {
            value.Real = real;
            value.Imag = imag;
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = GetDtype(8, 'c');
                        }
                    }
                }
                return dtype_;
            }
        }

        public override object Value { get { return new Complex(value.Real, value.Imag); } }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            unsafe {
                float* p = (float*)result.UnsafeAddress.ToPointer();
                *p++ = value.Real;
                *p = value.Imag;
            }
            return result;
        }

        internal override ScalarGeneric FillData(IntPtr dataPtr, int size, bool isNativeByteOrder) {
            unsafe {
                fixed (void* data = &value) {
                    NpyCoreApi.CopySwapScalar((dtype)dtype, data, (void*)dataPtr, !isNativeByteOrder);
                }
            }
            return this;
        }

        public override object __int__(CodeContext cntx) {
            EmitComplexWarning(cntx);
            return (int)value.Real;
        }

        public override object __long__(CodeContext cntx) {
            EmitComplexWarning(cntx);
            return (BigInteger)value.Real;
        }

        public override object __float__(CodeContext cntx) {
            EmitComplexWarning(cntx);
            return (double)value.Real;
        }

        public object __complex__() {
            return new Complex(value.Real, value.Imag);
        }

        public override object imag {
            get {
                return new ScalarFloat32(value.Imag);
            }
        }

        public override object real {
            get {
                return new ScalarFloat32(value.Real);
            }
        }

        public static implicit operator Complex(ScalarComplex64 x) {
            return new Complex(x.value.Real, x.value.Imag);
        }

        public static implicit operator string(ScalarComplex64 x) {
            return x.ToString();
        }

        public override object __str__(CodeContext context) {
            return ToString();
        }

        public override object __repr__(CodeContext context) {
            return ToString();
        }

        public override object __mod__(CodeContext cntx, object b) {
            throw new ArgumentTypeException(
                String.Format("unsupported operand type(s) for %: '{0}' and '{1}'",
                    this.GetType().Name, b.GetType().Name));
        }

        public override object __divmod__(CodeContext cntx, object b) {
            throw new ArgumentTypeException(
                String.Format("unsupported operand type(s) for divmod: '{0}' and '{1}'",
                    this.GetType().Name, b.GetType().Name));
        }

        public override object __rdivmod__(CodeContext cntx, object a) {
            throw new ArgumentTypeException(
                String.Format("unsupported operand type(s) for divmod: '{0}' and '{1}'",
                    this.GetType().Name, a.GetType().Name));
        }


        public override string ToString(IFormatProvider fp = null) {
            // Use the Python str() function instead of .NET formatting because the
            // formats are slightly different and cause regression failures.
            if (value.Real == 0.0) {
                return String.Format("{0}j", 
                    NpyUtil_Python.CallBuiltin(NpyUtil_Python.DefaultContext, "str", value.Imag.ToString("R")));
            } else {
                return String.Format("({0}+{1}j)",
                    NpyUtil_Python.CallBuiltin(NpyUtil_Python.DefaultContext, "str", value.Real.ToString("R")),
                    NpyUtil_Python.CallBuiltin(NpyUtil_Python.DefaultContext, "str", value.Imag.ToString("R")));
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        struct Data
        {
            internal float Real;
            internal float Imag;
        }


        /// <summary>
        /// Sets the object value from an unknown object type.  If imagOnly is false, then the real
        /// or real and imaginary parts are set.  If imagOnly is set, then only the imaginary part
        /// is set and arguments of complex type are rejected.
        /// </summary>
        /// <param name="o">Value to set</param>
        /// <param name="imagOnly">True only sets imaginary part, false sets real or both</param>
        protected void SetFromObj(object o) {
            if (o == null) real = imag = 0.0f;
            else if (o is int) {
                value.Real = (int)o;
                value.Imag = 0.0f;
            } else if (o is long) {
                value.Real = (long)o;
                value.Imag = 0.0f;
            } else if (o is float) {
                value.Real = (float)o;
                value.Imag = 0.0f;
            } else if (o is double) {
                value.Real = (float)(double)o;
                value.Imag = 0.0f;
            } else if (o is Complex) {
                value.Real = (float)((Complex)o).Real;
                value.Imag = (float)((Complex)o).Imaginary;
            } else if (o is ScalarComplex64) {
                value = ((ScalarComplex64)o).value;
            } else if (o is ScalarComplex128) {
                value.Real = (float)(double)((ScalarComplex128)o).real;
                value.Imag = (float)(double)((ScalarComplex128)o).imag;
            } else if (o is ScalarGeneric) {
                value.Real = (float)(double)((ScalarGeneric)o).__float__(NpyUtil_Python.DefaultContext);
                value.Imag = 0.0f;
            } else throw new ArgumentTypeException(
                  String.Format("Unable to construct complex value from type '{0}'.", o.GetType().Name));
        }

        public virtual object __eq__(CodeContext cntx, ScalarComplex64 o) {
            return value.Real == o.value.Real && value.Imag == o.value.Imag;
        }

        public virtual object __ne__(CodeContext cntx, ScalarComplex64 o) {
            return value.Real != o.value.Real || value.Imag != o.value.Imag;
        }

        private Data value;
        static private dtype dtype_;
    }

    [PythonType("numpy.complex128")]
    public class ScalarComplex128 : ScalarComplexFloating
    {
        public ScalarComplex128() {
            value = 0;
        }

        public ScalarComplex128(object o) {
            SetFromObj(o);
        }

        public ScalarComplex128(double real, double imag) {
            value = new Complex(real, imag);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = GetDtype(16, 'c');
                        }
                    }
                }
                return dtype_;
            }
        }

        public override object Value { get { return value; } }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            unsafe {
                Complex* p = (Complex*)result.UnsafeAddress.ToPointer();
                *p = value;
            }
            return result;
        }

        internal override ScalarGeneric FillData(IntPtr dataPtr, int size, bool isNativeByteOrder) {
            unsafe {
                fixed (void* data = &value) {
                    NpyCoreApi.CopySwapScalar((dtype)dtype, data, (void*)dataPtr, !isNativeByteOrder);
                }
            }
            return this;
        }

        public override object __int__(CodeContext cntx) {
            EmitComplexWarning(cntx);
            return (int)value.Real;
        }

        public override object __long__(CodeContext cntx) {
            EmitComplexWarning(cntx);
            return (BigInteger)value.Real;
        }

        public override object __float__(CodeContext cntx) {
            EmitComplexWarning(cntx);
            return value.Real;
        }

        public override object __mod__(CodeContext cntx, object b) {
            throw new ArgumentTypeException(
                String.Format("unsupported operand type(s) for %: '{0}' and '{1}'",
                    this.GetType().Name, b.GetType().Name));
        }

        public override object __divmod__(CodeContext cntx, object b) {
            throw new ArgumentTypeException(
                String.Format("unsupported operand type(s) for divmod: '{0}' and '{1}'",
                    this.GetType().Name, b.GetType().Name));
        }

        public override object __rdivmod__(CodeContext cntx, object a) {
            throw new ArgumentTypeException(
                String.Format("unsupported operand type(s) for divmod: '{0}' and '{1}'",
                    this.GetType().Name, a.GetType().Name));
        }

        public object __complex__() {
            return new Complex(value.Real, value.Imaginary);
        }

        public override object imag {
            get {
                return new ScalarFloat64(value.Imaginary);
            }
        }

        public override object real {
            get {
                return new ScalarFloat64(value.Real);
            }
        }

        public override object __str__(CodeContext context) {
            return ToString();
        }

        public override object __repr__(CodeContext context) {
            return ToString();
        }

        public override string ToString(IFormatProvider fp = null) {
            if (value.Real == 0.0) {
                return String.Format("{0}j",
                    NpyUtil_Python.CallBuiltin(NpyUtil_Python.DefaultContext, "str", value.Imaginary.ToString("R")));
            } else {
                return String.Format("({0}+{1}j)",
                    NpyUtil_Python.CallBuiltin(NpyUtil_Python.DefaultContext, "str", value.Real.ToString("R")),
                    NpyUtil_Python.CallBuiltin(NpyUtil_Python.DefaultContext, "str", value.Imaginary.ToString("R")));
            }
        }

        public static implicit operator string(ScalarComplex128 x) {
            return x.ToString();
        }

        public static implicit operator Complex(ScalarComplex128 x) {
            return x.value;
        }

        /// <summary>
        /// Sets the object value from an unknown object type.  If imagOnly is false, then the real
        /// or real and imaginary parts are set.  If imagOnly is set, then only the imaginary part
        /// is set and arguments of complex type are rejected.
        /// </summary>
        /// <param name="o">Value to set</param>
        /// <param name="imagOnly">True only sets imaginary part, false sets real or both</param>
        protected void SetFromObj(object o) {
            if (o == null) real = imag = 0.0f;
            else if (o is int) {
                value = new Complex((int)o, 0.0);
            } else if (o is long) {
                value = new Complex((long)o, 0.0);
            } else if (o is float) {
                value = new Complex((float)o, 0.0);
            } else if (o is double) {
                value = new Complex((double)o, 0.0);
            } else if (o is ScalarInt16) {
                value = new Complex((int)(ScalarInt16)o, 0.0);
            } else if (o is ScalarInt16) {
                value = new Complex((int)(ScalarInt32)o, 0.0);
            } else if (o is ScalarInt16) {
                value = new Complex((long)(ScalarInt64)o, 0.0);
            } else if (o is Complex) {
                value = (Complex)o;
            } else if (o is ScalarComplex64) {
                value = new Complex((float)((ScalarComplex64)o).real, (float)((ScalarComplex64)o).imag);
            } else if (o is ScalarComplex128) {
                value = ((ScalarComplex128)o).value;
            } else if (o is ScalarGeneric) {
                value = new Complex((double)((ScalarGeneric)o).__float__(NpyUtil_Python.DefaultContext), 0.0);
            } else throw new ArgumentTypeException(
                  String.Format("Unable to construct complex value from type '{0}'.", o.GetType().Name));
        }

        public virtual object __eq__(CodeContext cntx, ScalarComplex128 o) {
            return value == o.value;
        }

        public virtual object __ne__(CodeContext cntx, ScalarComplex128 o) {
            return value != o.value;
        }

        private Complex value;
        static private dtype dtype_;
    }

    [PythonType("numpy.flexible")]
    public class ScalarFlexible : ScalarGeneric { }

    [PythonType("numpy.void")]
    public class ScalarVoid : ScalarFlexible, IDisposable
    {
        public static object __new__(CodeContext cntx, PythonType cls) {
            ScalarVoid result = (ScalarVoid)ObjectOps.__new__(cntx, cls);
            return result;
        }

        public static object __new__(CodeContext cntx, PythonType cls, int size) {
            ScalarVoid result = (ScalarVoid)ObjectOps.__new__(cntx, cls);
            result.dtype_ = new dtype(result.dtype_);
            result.dtype_.ElementSize = size;
            return result;
        }

        public static object __new__(CodeContext cntx, PythonType cls, BigInteger size) {
            if (size > int.MaxValue) {
                throw new OverflowException(String.Format("Size must be smaller than {0}", int.MaxValue));
            }
            return __new__(cntx, cls, (int)size);
        }

        public static object __new__(CodeContext cntx, PythonType cls, ndarray arr) {
            if (arr.ndim == 0 && arr.IsInteger) {
                object iVal = arr.__int__(cntx);
                if (iVal is int) {
                    return __new__(cntx, cls, (int)iVal);
                } else {
                    throw new ArgumentException("Size of void is too large");
                }
            } else {
                return FromObject(arr);
            }
        }

        public static object __new__(CodeContext cntx, PythonType cls, ScalarInteger size) {
            object ival = size.__int__(cntx);
            if (ival is int) {
                return __new__(cntx, cls, (int)ival);
            } else {
                throw new ArgumentException("Size of void is too large");
            }
        }

        public static object __new__(PythonType cls, object val) {
            return FromObject(val);
        }

        private static object FromObject(object val) {
            ndarray arr = NpyArray.FromAny(val, NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_VOID), flags: NpyDefs.NPY_FORCECAST);
            return ndarray.ArrayReturn(arr);
        }

        public ScalarVoid() {
            dtype_ = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_VOID);
            dataptr = IntPtr.Zero;
        }

        internal ScalarVoid(int size) {
            AllocData(size);
            dtype_ = new dtype(NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_VOID));
            dtype_.ElementSize = size;
        }

        private void AllocData(int size) {
            dataptr = Marshal.AllocCoTaskMem(size);
            unsafe {
                // TODO: We should be using memset, or something like it.
                byte* p = (byte*)dataptr.ToPointer();
                byte* end = p + size;
                while (p < end) {
                    *p++ = 0;
                }
            }
        }

        ~ScalarVoid() {
            Dispose(false);
        }

        public void Dispose() {
            Dispose(true);
        }

        private void Dispose(bool disposing) {
            if (dataptr != IntPtr.Zero && base_arr == null) {
                lock (this) {
                    if (dataptr != IntPtr.Zero && base_arr == null) {
                        Marshal.FreeCoTaskMem(dataptr);
                        dataptr = IntPtr.Zero;
                        if (disposing) {
                            GC.SuppressFinalize(this);
                        }
                    }
                }
            }
        }

        public override object dtype {
            get {
                return dtype_;
            }
        }

        public override object Value { get { return this[0]; } }

        internal override ndarray ToArray() {
            ndarray a = NpyCoreApi.NewFromDescr(dtype_, new long[0], null, dataptr, 0, null);
            //a.BaseObj = this;
            return a;
        }

        internal override ScalarGeneric FillData(ndarray arr, long offset, bool isNativeByteOrder) {
            int elsize = arr.Dtype.ElementSize;

            if (dataptr != IntPtr.Zero) {
                throw new IronPython.Runtime.Exceptions.RuntimeException("Unexpected modification to existing scalar object.");
            }

/*            if (dtype_.ElementSize != elsize) {
                dtype_ = new dtype(dtype_);
                dtype_.ElementSize = elsize;
                //if (dataptr != IntPtr.Zero) {
                //    Marshal.FreeCoTaskMem(dataptr);
                //}
            } else {
                dtype_ = arr.Dtype;
            } */
            dtype_ = arr.Dtype;

            if (arr.Dtype.HasNames) {
                base_arr = arr;
                dataptr = (IntPtr)((long)arr.UnsafeAddress + offset);
            } else {
                base_arr = null;
                AllocData(elsize);
                unsafe {
                    arr.CopySwapOut(offset, dataptr.ToPointer(), !arr.IsNotSwapped);
                }
            } 
            return this;
        }

        /// <summary>
        /// Set field for void-types is allowed.  This sets some portion of the data
        /// memory based on the passed value.
        /// </summary>
        /// <param name="cntx">IronPython interpreter context</param>
        /// <param name="value">Value to set</param>
        /// <param name="dtype">Type of value</param>
        /// <param name="offset">Offset into scalar's data in bytes</param>
        public override void setfield(CodeContext cntx, object value, object dtype, int offset = 0) {
            dtype valueType = NpyDescr.DescrConverter(cntx, dtype);
            if (offset < 0 || offset + valueType.itemsize > dtype_.itemsize) {
                throw new ArgumentException(
                    String.Format("Need 0 <= offset <= {0} for requested type but received offset of {1}.",
                                  dtype_.itemsize, offset));
            }

            // If we are storing an object, allocate a new GC handle and release any existing
            // GC handle.  Otherwise we just copy the data over.
            if (valueType.TypeNum == NpyDefs.NPY_TYPES.NPY_OBJECT) {
                IntPtr tmp = Marshal.ReadIntPtr(dataptr, offset);
                Marshal.WriteIntPtr(dataptr, offset, GCHandle.ToIntPtr(NpyCoreApi.AllocGCHandle(value)));
                NpyCoreApi.FreeGCHandle(NpyCoreApi.GCHandleFromIntPtr(tmp));
            } else {
                using (ndarray src = NpyArray.FromAny(value, valueType, 0, 0, NpyDefs.NPY_CARRAY)) {
                    unsafe {
                        src.CopySwapOut(0, (dataptr + offset).ToPointer(), false);
                    }
                }
            }
        }

        internal override ScalarGeneric FillData(IntPtr dataPtr, int size, bool isNativeByteOrder) {
            throw new NotImplementedException("Scalar fill operations are not supported for flexible (variable-size) types.");
        }

        public override object this[int index] {
            get {
                return Index(index);
            }
        }

        public override object this[long index] {
            get {
                return Index((int)index);
            }
        }

        public override object this[BigInteger index] {
            get {
                return Index((int)index);
            }
        }

        public object this[string index] {
            get {
                return Index(index);
            }
        }

        private object Index(int index) {
            if (!dtype_.HasNames) {
                throw new IndexOutOfRangeException("cant' index void scalar without fields");
            }
            return Index(dtype_.Names[index]);
        }

        private object Index(string index) {
            return ToArray()[index];
        }

        private dtype dtype_;
        private IntPtr dataptr;

        /// <summary>
        /// When set this object is sharing memroy with the array below.  This occurs when accessing
        /// elements of record type array.
        /// </summary>
        private ndarray base_arr;
    }

    [PythonType("numpy.character")]
    public class ScalarCharacter : ScalarFlexible
    {
        public override object dtype {
            get {
                return NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_STRING);
            }
        }
    }

    [PythonType("numpy.string_")]
    public class ScalarString : ScalarCharacter
    {
        public ScalarString() {
            value = new Bytes();
        }

        public ScalarString(Bytes s) {
            value = s;
        }

        public ScalarString(CodeContext cntx, string s) {
            value = new Bytes(cntx, s, "UTF-8");
        }

        public ScalarString(dynamic s) {
            value = s;
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    dtype_ = GetDtype(value.Count, 'S');
                }
                return dtype_;
            }
        }

        public override object Value { get { return value; } }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            result.SetItem(value, 0);
            return result;
        }

        internal override ScalarGeneric FillData(ndarray arr, long offset, bool isNativeByteOrder) {
            value = (Bytes)arr.GetItem(offset);
            return this;
        }

        internal override ScalarGeneric FillData(IntPtr dataPtr, int size, bool isNativeByteOrder) {
            value = NumericOps.getitemString(dataPtr, size);
            return this;
        }

        public string rstrip() {
            return value.ToString().rstrip();
        }

        public string strip() {
            return value.ToString().strip();
        }

        private Bytes value;
        private dtype dtype_;
    }

    [PythonType("numpy.unicode_")]
    public class ScalarUnicode : ScalarCharacter
    {
        public ScalarUnicode() {
            value = "";
        }

        public ScalarUnicode(string s) {
            value = s;
        }

        public ScalarUnicode(dynamic s) {
            value = s;
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    dtype_ = GetDtype(value.Length, 'U');
                }
                return dtype_;
            }
        }

        public static implicit operator string(ScalarUnicode s) {
            return s.value;
        }

        public override object Value { get { return value; } }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            result.SetItem(value, 0);
            return result;
        }

        internal override ScalarGeneric FillData(ndarray arr, long offset, bool isNativeByteOrder) {
            value = (string)arr.GetItem(offset);
            return this;
        }

        public string rstrip() {
            return value.rstrip();
        }

        public string strip() {
            return value.strip();
        }

        internal override ScalarGeneric FillData(IntPtr dataPtr, int size, bool isNativeByteOrder) {
            value = (string)NumericOps.getitemUnicode(dataPtr, size, false);
            // TODO: Unpickling unicode strings requires a double-copy of the data. We really need a better implementation.
/*            byte[] b = new byte[size];
            for (int i = 0; i < size; i++) b[i] = Marshal.ReadByte(dataPtr, i);
            b = Encoding.Convert(Encoding.UTF32, Encoding.Unicode, b);
            value = Encoding.Unicode.GetString(b); */
            return this;
        }

        private string value;
        private dtype dtype_;
    }

    [PythonType("numpy.object_")]
    public class ScalarObject : ScalarGeneric
    {
        /// <summary>
        /// Constructs a new instance of whatever the type of value is or returns null.  That is,
        /// numpy.object_(arg) behaves like a function returning the argument itself instead of an
        /// instance of this class.
        /// </summary>
        /// <param name="cntx">Code context</param>
        /// <param name="value">Default value or null</param>
        /// <returns>Instance of default value or null</returns>
        public static object __new__(CodeContext cntx, PythonType type, object value=null) {
            object result;

            // This is unpleasant. In CPython the tp_new field of some scalar types is changed during
            // initialization, so some code constructs instances of object_ and other code gets whatever
            // this function creates.
            if (!ScalarGeneric.Initialized) {
                return new ScalarObject(value);
            }

            if (value == null) {
                result = null;
            } else {
                dtype typecode = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_OBJECT);
                ndarray arr = NpyArray.FromAny(value, typecode, 0, 0, NpyDefs.NPY_FORCECAST);
                if (arr == null || arr.ndim > 0) {
                    result = arr;
                } else {
                    result = typecode.ToScalar(arr);
                }
            }
            return result;
        }


        public ScalarObject() {
            value = null;
        }

        public ScalarObject(object o) {
            value = o;
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_OBJECT);
                        }
                    }
                }
                return dtype_;
            }
        }

        public override object Value { get { return value; } }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            result.SetItem(value, 0);
            return result;
        }

        internal override ScalarGeneric FillData(ndarray arr, long offset, bool isNativeByteOrder) {
            value = arr.GetItem(offset);
            return this;
        }


        internal override ScalarGeneric FillData(IntPtr dataPtr, int size, bool isNativeByteOrder) {
            throw new NotImplementedException("Scalar fill operations are not supported for flexible (variable-size) types.");
        }

        private object value;
        private static dtype dtype_;
    }
}
