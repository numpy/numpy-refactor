using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using IronPython.Runtime;
using IronPython.Runtime.Types;
using System.Runtime.InteropServices;
using System.Numerics;
using Microsoft.Scripting;

namespace NumpyDotNet
{
    [PythonType("numpy.generic")]
    public class ScalarGeneric : IArray
    {
        internal virtual ndarray ToArray() {
            return null;
        }

        /// <summary>
        /// Fill the value with the value from the 0-d array
        /// </summary>
        /// <param name="arr"></param>
        internal virtual void FillData(ndarray arr, long offset = 0) {
        }

        #region IArray interface

        public object __abs__() {
            return ToArray().__abs__();
        }

        public object __len__() {
            return ToArray().__len__();
        }

        public object __lshift__(object b) {
            return ToArray().__lshift__(b);
        }

        public object __mod__(object b) {
            return ToArray().__mod__(b);
        }

        public string __repr__(CodeContext context) {
            return ToArray().__str__(context);
        }

        public object __rshift__(object b) {
            return ToArray().__rshift__(b);
        }

        public object __sqrt__() {
            return ToArray().__sqrt__();
        }

        public string __str__(CodeContext context) {
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

        public ndarray byteswap(bool inplace = false) {
            if (inplace) {
                throw new ArgumentException("cannot byteswap a scalar inplace");
            } else {
                // TODO: Fix to return a scalar
                return ToArray().byteswap(false);
            }
        }

        public object choose(IEnumerable<object> choices, ndarray @out = null, object mode = null) {
            return ToArray().choose(choices, @out, mode);
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

        public ndarray copy(object order = null) {
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

        public virtual dtype dtype {
            get {
                return ToArray().dtype;
            }
            set {
                throw new ArgumentException("array-scalars are immutable");
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
                throw new ArgumentException("array-scalars are immutable");
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
                throw new ArgumentException("array-scalars are immutable");
            }
        }

        public object item(params object[] args) {
            return ToArray().item(args:args);
        }

        public void itemset(params object[] args) {
            throw new ArgumentException("array-scalars are immutable");
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
                throw new ArgumentException("array-scalars are immutable");
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

        public void setfield(CodeContext cntx, object value, object dtype, int offset = 0) {
            throw new ArgumentException("array-scalars are immutable");
        }

        public void setflags(object write = null, object align = null, object uic = null) {
            // CPython implementation simply does nothing, so we will too.
        }

        public PythonTuple shape {
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

        public long[] strides {
            get { return new long[0]; }
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

        public object take(object indices, object axis = null, ndarray @out = null, object mode = null) {
            return ToArray().take(indices, axis, @out, mode);
        }

        public object this[params object[] args] {
            get {
                return ToArray()[args: args];
            }
            set {
                throw new ArgumentException("array-scalars are immutable");
            }
        }

        public object this[int index] {
            get {
                return ToArray()[index];
            }
        }

        public object this[long index] {
            get {
                return ToArray()[index];
            }
        }

        public object this[IntPtr index] {
            get {
                return ToArray()[index];
            }
        }

        public object this[System.Numerics.BigInteger index] {
            get {
                return ToArray()[index];
            }
        }

        public object this[string field] {
            get {
                return ToArray()[field];
            }
            set {
                throw new ArgumentException("array-scalars are immutable");
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
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_add);
        }

        public static object operator +(ScalarGeneric a, ScalarGeneric b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_add);
        }

        public static object operator +(object a, ScalarGeneric b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_add);
        }

        public static object operator -(ScalarGeneric a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_subtract);
        }

        public static object operator -(ScalarGeneric a, ScalarGeneric b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_subtract);
        }

        public static object operator -(object a, ScalarGeneric b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_subtract);
        }

        public static object operator *(ScalarGeneric a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_multiply);
        }

        public static object operator *(ScalarGeneric a, ScalarGeneric b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_multiply);
        }

        public static object operator *(object a, ScalarGeneric b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_multiply);
        }

        public static object operator /(ScalarGeneric a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_divide);
        }

        public static object operator /(ScalarGeneric a, ScalarGeneric b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_divide);
        }

        public static object operator /(object a, ScalarGeneric b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_divide);
        }

        public static object operator &(ScalarGeneric a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_bitwise_and);
        }

        public static object operator &(ScalarGeneric a, ScalarGeneric b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_bitwise_and);
        }

        public static object operator &(object a, ScalarGeneric b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_bitwise_and);
        }

        public static object operator |(ScalarGeneric a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_bitwise_or);
        }

        public static object operator |(ScalarGeneric a, ScalarGeneric b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_bitwise_or);
        }

        public static object operator |(object a, ScalarGeneric b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_bitwise_or);
        }

        public static object operator ^(ScalarGeneric a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_bitwise_xor);
        }

        public static object operator ^(ScalarGeneric a, ScalarGeneric b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_bitwise_xor);
        }

        public static object operator ^(object a, ScalarGeneric b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_bitwise_xor);
        }

        public static explicit operator int(ScalarGeneric a) {
            return (int)a.ToArray();
        }

        public static explicit operator BigInteger(ScalarGeneric a) {
            return (BigInteger)a.ToArray();
        }

        public static explicit operator double(ScalarGeneric a) {
            return (double)a.ToArray();
        }

        public static explicit operator Complex(ScalarGeneric a) {
            return (Complex)a.ToArray();
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
    }

    [PythonType("numpy.bool_")]
    public class ScalarBool : ScalarGeneric
    {
        public static object __new__(PythonType cls) {
            return new ScalarBool();
        }

        public static object __new__(PythonType cls, bool val) {
            return new ScalarBool(val);
        }

        public static object __new__(PythonType cls, object val) {
            ndarray arr = NpyArray.FromAny(val, descr: NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_BOOL),
                                            flags: NpyDefs.NPY_FORCECAST);
            if (arr.ndim == 0) {
                byte b = Marshal.ReadByte(arr.UnsafeAddress);
                return new ScalarBool(b != 0);
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

        public override dtype dtype {
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
            ndarray result = NpyCoreApi.AllocArray(dtype, 0, null, false);
            Marshal.WriteByte(result.UnsafeAddress, (byte)(value ? 1 : 0));
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.UnsafeAddress.ToInt64() + offset);
            value = (Marshal.ReadByte(p) != 0);
        }

        private bool value;
        static private dtype dtype_;
    }

    [PythonType("numpy.number")]
    public class ScalarNumber : ScalarGeneric { }

    [PythonType("numpy.integer")]
    public class ScalarInteger : ScalarNumber { }

    [PythonType("numpy.signedinteger")]
    public class ScalarSignedInteger : ScalarInteger { }

    [PythonType("numpy.int8")]
    public class ScalarInt8 : ScalarSignedInteger
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

        public override dtype dtype {
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
            ndarray result = NpyCoreApi.AllocArray(dtype, 0, null, false);
            Marshal.WriteByte(result.UnsafeAddress, (byte)value);
            return result;
        }
  
        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.UnsafeAddress.ToInt64() + offset);
            value = (sbyte)Marshal.ReadByte(p);
        }

        private sbyte value;
        static private dtype dtype_;

        internal static readonly int MinValue = sbyte.MinValue;
        internal static readonly int MaxValue = sbyte.MaxValue;
    }

    [PythonType("numpy.int16")]
    public class ScalarInt16 : ScalarSignedInteger
    {
        public ScalarInt16() {
            value = 0;
        }

        public ScalarInt16(Int16 value) {
            this.value = value;
        }

        public ScalarInt16(IConvertible value) {
            this.value = Convert.ToInt16(value);
        }

        public override dtype dtype {
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
            ndarray result = NpyCoreApi.AllocArray(dtype, 0, null, false);
            Marshal.WriteInt16(result.UnsafeAddress, value);
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.UnsafeAddress.ToInt64() + offset);
            value = Marshal.ReadInt16(p);
        }

        private Int16 value;
        static private dtype dtype_;

        internal static readonly int MinValue = Int16.MinValue;
        internal static readonly int MaxValue = Int16.MaxValue;
    }

    [PythonType("numpy.int32")]
    public class ScalarInt32 : ScalarSignedInteger
    {
        public ScalarInt32() {
            value = 0;
        }

        public ScalarInt32(Int32 value) {
            this.value = value;
        }

        public ScalarInt32(IConvertible value) {
            this.value = Convert.ToInt32(value);
        }

        public override dtype dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = GetDtype(4, 'i');
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray(dtype, 0, null, false);
            Marshal.WriteInt32(result.UnsafeAddress, value);
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.UnsafeAddress.ToInt64() + offset);
            value = Marshal.ReadInt32(p);
        }

        private Int32 value;
        static private dtype dtype_;

        internal static readonly int MinValue = Int32.MinValue;
        internal static readonly int MaxValue = Int32.MaxValue;
    }

    [PythonType("numpy.int64")]
    public class ScalarInt64 : ScalarSignedInteger
    {
        public ScalarInt64() {
            value = 0;
        }

        public ScalarInt64(Int64 value) {
            this.value = value;
        }

        public ScalarInt64(IConvertible value) {
            this.value = Convert.ToInt64(value);
        }

        public override dtype dtype {
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

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray(dtype, 0, null, false);
            Marshal.WriteInt64(result.UnsafeAddress, value);
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.UnsafeAddress.ToInt64() + offset);
            value = Marshal.ReadInt64(p);
        }

        private Int64 value;
        static private dtype dtype_;

        internal static readonly BigInteger MinValue = new BigInteger(Int64.MinValue);
        internal static readonly BigInteger MaxValue = new BigInteger(Int64.MaxValue);
    }

    [PythonType("numpy.unsignedinteger")]
    public class ScalarUnsignedInteger : ScalarInteger { }

    [PythonType("numpy.uint8")]
    public class ScalarUInt8 : ScalarUnsignedInteger
    {
        public ScalarUInt8() {
            value = 0;
        }

        public ScalarUInt8(byte value) {
            this.value = value;
        }

        public ScalarUInt8(IConvertible value) {
            this.value = Convert.ToByte(value);
        }

        public override dtype dtype {
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
            ndarray result = NpyCoreApi.AllocArray(dtype, 0, null, false);
            Marshal.WriteByte(result.UnsafeAddress, value);
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.UnsafeAddress.ToInt64() + offset);
            value = Marshal.ReadByte(p);
        }

        private byte value;
        static private dtype dtype_;

        internal static readonly int MinValue = 0;
        internal static readonly int MaxValue = byte.MaxValue;
    }

    [PythonType("numpy.uint16")]
    public class ScalarUInt16 : ScalarUnsignedInteger
    {
        public ScalarUInt16() {
            value = 0;
        }

        public ScalarUInt16(UInt16 value) {
            this.value = value;
        }

        public ScalarUInt16(IConvertible value) {
            this.value = Convert.ToUInt16(value);
        }

        public override dtype dtype {
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
            ndarray result = NpyCoreApi.AllocArray(dtype, 0, null, false);
            Marshal.WriteInt16(result.UnsafeAddress, (Int16)value);
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.UnsafeAddress.ToInt64() + offset);
            value = (UInt16)Marshal.ReadInt16(p);
        }

        private UInt16 value;
        static private dtype dtype_;

        internal static readonly int MinValue = 0;
        internal static readonly int MaxValue = UInt16.MaxValue;
    }

    [PythonType("numpy.uint32")]
    public class ScalarUInt32 : ScalarUnsignedInteger
    {
        public ScalarUInt32() {
            value = 0;
        }

        public ScalarUInt32(UInt32 value) {
            this.value = value;
        }

        public ScalarUInt32(IConvertible value) {
            this.value = Convert.ToUInt32(value);
        }

        public override dtype dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = GetDtype(4, 'u');
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray(dtype, 0, null, false);
            Marshal.WriteInt32(result.UnsafeAddress, (Int32)value);
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.UnsafeAddress.ToInt64() + offset);
            value = (UInt32)Marshal.ReadInt32(p);
        }

        private UInt32 value;
        static private dtype dtype_;

        internal static readonly int MinValue = 0;
        internal static readonly BigInteger MaxValue = new BigInteger(UInt32.MaxValue);
    }

    [PythonType("numpy.uint64")]
    public class ScalarUInt64 : ScalarUnsignedInteger
    {
        public ScalarUInt64() {
            value = 0;
        }

        public ScalarUInt64(UInt64 value) {
            this.value = value;
        }

        public ScalarUInt64(IConvertible value) {
            this.value = Convert.ToUInt64(value);
        }

        public override dtype dtype {
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

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray(dtype, 0, null, false);
            Marshal.WriteInt64(result.UnsafeAddress, (Int64)value);
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.UnsafeAddress.ToInt64() + offset);
            value = (UInt64)Marshal.ReadInt64(p);
        }

        private UInt64 value;
        static private dtype dtype_;

        internal static readonly int MinValue = 0;
        internal static readonly BigInteger MaxValue = new BigInteger(UInt64.MaxValue);
    }

    [PythonType("numpy.inexact")]
    public class ScalarInexact : ScalarNumber { }

    [PythonType("numpy.floating")]
    public class ScalarFloating : ScalarInexact { }

    [PythonType("numpy.float32")]
    public class ScalarFloat32 : ScalarFloating
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

        public override dtype dtype {
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
            ndarray result = NpyCoreApi.AllocArray(dtype, 0, null, false);
            unsafe {
                Single* p = (Single*) result.UnsafeAddress.ToPointer();
                *p = value;
            }
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.UnsafeAddress.ToInt64() + offset);
            unsafe {
                Single* ptr = (Single*)p.ToPointer();
                value = *ptr;
            }
        }

        private Single value;
        static private dtype dtype_;
    }

    [PythonType("numpy.float64")]
    public class ScalarFloat64 : ScalarFloating
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

        public override dtype dtype {
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
            ndarray result = NpyCoreApi.AllocArray(dtype, 0, null, false);
            unsafe {
                Double* p = (Double*)result.UnsafeAddress.ToPointer();
                *p = value;
            }
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.UnsafeAddress.ToInt64() + offset);
            unsafe {
                Double* ptr = (Double*)p.ToPointer();
                value = *ptr;
            }
        }

        private Double value;
        static private dtype dtype_;
    }

    [PythonType("numpy.complexfloating")]
    public class ScalarComplexFloating : ScalarInexact { }

    [PythonType("numpy.complex64")]
    public class ScalarComplex64 : ScalarComplexFloating
    {
        public ScalarComplex64() {
            Real = 0.0f;
            Imag = 0.0f;
        }

        public ScalarComplex64(Single value) {
            this.Real = value;
            this.Imag = 0.0f;
        }

        public ScalarComplex64(Single real, Single imag) {
            Real = real;
            Imag = imag;
        }

        public ScalarComplex64(dynamic value) {
            Complex c = (Complex)value;
            Real = (float)c.Real;
            Imag = (float)c.Imaginary;
        }

        public override dtype dtype {
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

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray(dtype, 0, null, false);
            unsafe {
                float* p = (float*)result.UnsafeAddress.ToPointer();
                *p++ = Real;
                *p = Imag;
            }
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.UnsafeAddress.ToInt64() + offset);
            unsafe {
                float* ptr = (float*)p.ToPointer();
                Real = *ptr++;
                Imag = *ptr;
            }
        }

        public override object imag {
            get {
                return new ScalarFloat32(Imag);
            }
        }

        public override object real {
            get {
                return new ScalarFloat32(Real);
            }
        }

        public override string ToString() {
            if (Real == 0.0) {
                return String.Format("{0}j", Imag);
            } else {
                return String.Format("({0}+{1}j)", Real, Imag);
            }
        }

        private float Real;
        private float Imag;
        static private dtype dtype_;
    }

    [PythonType("numpy.complex128")]
    public class ScalarComplex128 : ScalarComplexFloating
    {
        public ScalarComplex128() {
            value = 0;
        }

        public ScalarComplex128(double value) {
            this.value = value;
        }

        public ScalarComplex128(dynamic value) {
            this.value = (Complex)value;
        }

        public override dtype dtype {
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

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray(dtype, 0, null, false);
            unsafe {
                Complex* p = (Complex*)result.UnsafeAddress.ToPointer();
                *p = value;
            }
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.UnsafeAddress.ToInt64() + offset);
            unsafe {
                Complex* ptr = (Complex*)p.ToPointer();
                value = *ptr;
            }
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

        private Complex value;
        static private dtype dtype_;
    }

    [PythonType("numpy.flexible")]
    public class ScalarFlexible : ScalarGeneric { }

    [PythonType("numpy.character")]
    public class ScalarCharacter : ScalarFlexible { }

    [PythonType("numpy.string_")]
    public class ScalarString : ScalarCharacter
    {
        public ScalarString() {
            value = new Bytes();
        }

        public ScalarString(Bytes s) {
            value = s;
        }

        public ScalarString(dynamic s) {
            value = s;
        }

        public override dtype dtype {
            get {
                if (dtype_ == null) {
                    dtype_ = GetDtype(value.Count, 'S');
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray(dtype, 0, null, false);
            result.SetItem(value, 0);
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            value = (Bytes)arr.GetItem(offset);
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

        public override dtype dtype {
            get {
                if (dtype_ == null) {
                    dtype_ = GetDtype(value.Length, 'U');
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray(dtype, 0, null, false);
            result.SetItem(value, 0);
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            value = (string)arr.GetItem(offset);
        }

        private string value;
        private dtype dtype_;
    }
}
