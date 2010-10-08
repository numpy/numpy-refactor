using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using IronPython.Runtime;
using System.Runtime.InteropServices;
using System.Numerics;

namespace NumpyDotNet
{
    public class generic : IArray
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

        public IntPtr data {
            get { throw new NotImplementedException(); }
        }

        public ndarray diagonal(int offset = 0, int axis1 = 0, int axis2 = 1) {
            return ToArray().diagonal(offset, axis1, axis2);
        }

        public virtual dtype dtype {
            get {
                return ToArray().dtype;
            }
            set {
                throw new NotImplementedException();
            }
        }

        public void fill(object scalar) {
            // TODO: This doesn't make any sense but is the same for CPython
            ToArray().fill(scalar);
        }

        public flagsobj flags {
            get { throw new NotImplementedException(); }
        }

        public object flat {
            get {
                throw new NotImplementedException();
            }
            set {
                throw new NotImplementedException();
            }
        }

        public ndarray flatten(object order = null) {
            return ToArray().flatten(order);
        }

        public ndarray getfield(CodeContext cntx, object dtype, int offset = 0) {
            return ToArray().getfield(cntx, dtype, offset);
        }

        public object imag {
            get {
                throw new NotImplementedException();
            }
            set {
                throw new NotImplementedException();
            }
        }

        public object item(params object[] args) {
            return ToArray().item(args:args);
        }

        public void itemset(params object[] args) {
            throw new NotImplementedException();
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
            get { throw new NotImplementedException(); }
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

        public object real {
            get {
                throw new NotImplementedException();
            }
            set {
                throw new NotImplementedException();
            }
        }

        public object repeat(object repeats, object axis = null) {
            return ToArray().repeat(repeats, axis);
        }

        public ndarray reshape(Microsoft.Scripting.IAttributesCollection kwds, params object[] args) {
            throw new NotImplementedException();
        }

        public void resize(Microsoft.Scripting.IAttributesCollection kwds, params object[] args) {
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
            throw new NotImplementedException();
        }

        public void setflags(object write = null, object align = null, object uic = null) {
            throw new NotImplementedException();
        }

        public PythonTuple shape {
            get { throw new NotImplementedException(); }
        }

        public object size {
            get { throw new NotImplementedException(); }
        }

        public void sort(int axis = -1, string kind = null, object order = null) {
            // TODO: This doesn't make any sense, but CPython does the same.
            ToArray().sort(axis, kind, order);
        }

        public ndarray squeeze() {
            throw new NotImplementedException();
        }

        public object std(CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null, int ddof = 0) {
            return ToArray().std(cntx, axis, dtype, @out, ddof);
        }

        public long[] strides {
            get { throw new NotImplementedException(); }
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
                throw new NotImplementedException();
            }
            set {
                throw new NotImplementedException();
            }
        }

        public object this[int index] {
            get { throw new NotImplementedException(); }
        }

        public object this[long index] {
            get { throw new NotImplementedException(); }
        }

        public object this[IntPtr index] {
            get { throw new NotImplementedException(); }
        }

        public object this[System.Numerics.BigInteger index] {
            get { throw new NotImplementedException(); }
        }

        public object this[string field] {
            get {
                throw new NotImplementedException();
            }
            set {
                throw new NotImplementedException();
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

        public static object operator +(generic a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_add);
        }

        public static object operator +(generic a, generic b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_add);
        }

        public static object operator +(object a, generic b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_add);
        }

        public static object operator -(generic a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_subtract);
        }

        public static object operator -(generic a, generic b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_subtract);
        }

        public static object operator -(object a, generic b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_subtract);
        }

        public static object operator *(generic a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_multiply);
        }

        public static object operator *(generic a, generic b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_multiply);
        }

        public static object operator *(object a, generic b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_multiply);
        }

        public static object operator /(generic a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_divide);
        }

        public static object operator /(generic a, generic b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_divide);
        }

        public static object operator /(object a, generic b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_divide);
        }

        public static object operator &(generic a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_bitwise_and);
        }

        public static object operator &(generic a, generic b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_bitwise_and);
        }

        public static object operator &(object a, generic b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_bitwise_and);
        }

        public static object operator |(generic a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_bitwise_or);
        }

        public static object operator |(generic a, generic b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_bitwise_or);
        }

        public static object operator |(object a, generic b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_bitwise_or);
        }

        public static object operator ^(generic a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_bitwise_xor);
        }

        public static object operator ^(generic a, generic b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_bitwise_xor);
        }

        public static object operator ^(object a, generic b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_bitwise_xor);
        }

        public static explicit operator int(generic a) {
            return (int)a.ToArray();
        }

        public static explicit operator BigInteger(generic a) {
            return (BigInteger)a.ToArray();
        }

        public static explicit operator double(generic a) {
            return (double)a.ToArray();
        }

        public static explicit operator Complex(generic a) {
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

    public class number : generic { }

    public class integer : number { }

    public class signedinteger : integer { }

    public class int8 : signedinteger
    {
        public int8() {
            value = 0;
        }

        public int8(sbyte value) {
            this.value = value;
        }

        public int8(IConvertible value) {
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
            Marshal.WriteByte(result.data, (byte)value);
            return result;
        }
  
        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.data.ToInt64() + offset);
            value = (sbyte)Marshal.ReadByte(p);
        }

        private sbyte value;
        static private dtype dtype_;
    }

    public class int16 : signedinteger
    {
        public int16() {
            value = 0;
        }

        public int16(Int16 value) {
            this.value = value;
        }

        public int16(IConvertible value) {
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
            Marshal.WriteInt16(result.data, value);
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.data.ToInt64() + offset);
            value = Marshal.ReadInt16(p);
        }

        private Int16 value;
        static private dtype dtype_;
    }

    public class int32 : signedinteger
    {
        public int32() {
            value = 0;
        }

        public int32(Int32 value) {
            this.value = value;
        }

        public int32(IConvertible value) {
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
            Marshal.WriteInt32(result.data, value);
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.data.ToInt64() + offset);
            value = Marshal.ReadInt32(p);
        }

        private Int32 value;
        static private dtype dtype_;
    }

    public class int64 : signedinteger
    {
        public int64() {
            value = 0;
        }

        public int64(Int64 value) {
            this.value = value;
        }

        public int64(IConvertible value) {
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
            Marshal.WriteInt64(result.data, value);
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.data.ToInt64() + offset);
            value = Marshal.ReadInt64(p);
        }

        private Int64 value;
        static private dtype dtype_;
    }


    public class unsignedinteger : integer { }

    public class uint8 : unsignedinteger
    {
        public uint8() {
            value = 0;
        }

        public uint8(byte value) {
            this.value = value;
        }

        public uint8(IConvertible value) {
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
            Marshal.WriteByte(result.data, value);
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.data.ToInt64() + offset);
            value = Marshal.ReadByte(p);
        }

        private byte value;
        static private dtype dtype_;
    }

    public class uint16 : unsignedinteger
    {
        public uint16() {
            value = 0;
        }

        public uint16(UInt16 value) {
            this.value = value;
        }

        public uint16(IConvertible value) {
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
            Marshal.WriteInt16(result.data, (Int16)value);
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.data.ToInt64() + offset);
            value = (UInt16)Marshal.ReadInt16(p);
        }

        private UInt16 value;
        static private dtype dtype_;
    }

    public class uint32 : unsignedinteger
    {
        public uint32() {
            value = 0;
        }

        public uint32(UInt32 value) {
            this.value = value;
        }

        public uint32(IConvertible value) {
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
            Marshal.WriteInt32(result.data, (Int32)value);
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.data.ToInt64() + offset);
            value = (UInt32)Marshal.ReadInt32(p);
        }

        private UInt32 value;
        static private dtype dtype_;
    }

    public class uint64 : unsignedinteger
    {
        public uint64() {
            value = 0;
        }

        public uint64(UInt64 value) {
            this.value = value;
        }

        public uint64(IConvertible value) {
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
            Marshal.WriteInt64(result.data, (Int64)value);
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.data.ToInt64() + offset);
            value = (UInt64)Marshal.ReadInt64(p);
        }

        private UInt64 value;
        static private dtype dtype_;
    }

    public class inexact : generic { }

    public class floating : inexact { }

    public class float32 : floating
    {
        public float32() {
            value = 0;
        }

        public float32(Single value) {
            this.value = value;
        }

        public float32(IConvertible value) {
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
                Single* p = (Single*) result.data.ToPointer();
                *p = value;
            }
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.data.ToInt64() + offset);
            unsafe {
                Single* ptr = (Single*)p.ToPointer();
                value = *ptr;
            }
        }

        private Single value;
        static private dtype dtype_;
    }

    public class float64 : floating
    {
        public float64() {
            value = 0;
        }

        public float64(Double value) {
            this.value = value;
        }

        public float64(IConvertible value) {
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
                Double* p = (Double*)result.data.ToPointer();
                *p = value;
            }
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.data.ToInt64() + offset);
            unsafe {
                Double* ptr = (Double*)p.ToPointer();
                value = *ptr;
            }
        }

        private Double value;
        static private dtype dtype_;
    }

    public class complexfloating : inexact { }

    public class complex128 : complexfloating
    {
        public complex128() {
            value = 0;
        }

        public complex128(Single value) {
            this.value = value;
        }

        public complex128(dynamic value) {
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
                Complex* p = (Complex*)result.data.ToPointer();
                *p = value;
            }
            return result;
        }

        internal override void FillData(ndarray arr, long offset = 0) {
            IntPtr p = (IntPtr)(arr.data.ToInt64() + offset);
            unsafe {
                Complex* ptr = (Complex*)p.ToPointer();
                value = *ptr;
            }
        }

        private Complex value;
        static private dtype dtype_;
    }

    public class flexible : generic { }

    public class character : flexible { }

    public class string_ : character
    {
        public string_() {
            value = new Bytes();
        }

        public string_(Bytes s) {
            value = s;
        }

        public string_(dynamic s) {
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

    public class unicode : character
    {
        public unicode() {
            value = "";
        }

        public unicode(string s) {
            value = s;
        }

        public unicode(dynamic s) {
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
