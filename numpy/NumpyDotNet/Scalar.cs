using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using IronPython.Runtime;

namespace NumpyDotNet
{
    [PythonType("generic")]
    public class Scalar : IArray
    {
        internal virtual ndarray ToArray() {
            return null;
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

        public dtype dtype {
            get {
                throw new NotImplementedException();
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

        public static object operator +(Scalar a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_add);
        }

        public static object operator +(Scalar a, Scalar b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_add);
        }

        public static object operator +(object a, Scalar b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_add);
        }

        public static object operator -(Scalar a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_subtract);
        }

        public static object operator -(Scalar a, Scalar b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_subtract);
        }

        public static object operator -(object a, Scalar b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_subtract);
        }

        public static object operator *(Scalar a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_multiply);
        }

        public static object operator *(Scalar a, Scalar b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_multiply);
        }

        public static object operator *(object a, Scalar b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_multiply);
        }

        public static object operator /(Scalar a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_divide);
        }

        public static object operator /(Scalar a, Scalar b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_divide);
        }

        public static object operator /(object a, Scalar b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_divide);
        }

        public static object operator &(Scalar a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_bitwise_and);
        }

        public static object operator &(Scalar a, Scalar b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_bitwise_and);
        }

        public static object operator &(object a, Scalar b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_bitwise_and);
        }

        public static object operator |(Scalar a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_bitwise_or);
        }

        public static object operator |(Scalar a, Scalar b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_bitwise_or);
        }

        public static object operator |(object a, Scalar b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_bitwise_or);
        }

        public static object operator ^(Scalar a, object b) {
            return ndarray.BinaryOp(a.ToArray(), NpyArray.FromAny(b), NpyDefs.NpyArray_Ops.npy_op_bitwise_xor);
        }

        public static object operator ^(Scalar a, Scalar b) {
            return ndarray.BinaryOp(a.ToArray(), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_bitwise_xor);
        }

        public static object operator ^(object a, Scalar b) {
            return ndarray.BinaryOp(NpyArray.FromAny(a), b.ToArray(), NpyDefs.NpyArray_Ops.npy_op_bitwise_xor);
        }

        #endregion

    }

    [PythonType("int_")]
    public class ScalarInt : Scalar {

        public ScalarInt(int value) {
            this.value = value;
        }

        public ScalarInt(IConvertible value) {
            this.value = Convert.ToInt32(value);
        }

        internal override ndarray ToArray() {
            return NpyArray.FromAny(value);
        }

        private int value;
    }

    [PythonType("float_")]
    public class ScalarFloat : Scalar
    {

        public ScalarFloat(float value) {
            this.value = value;
        }

        public ScalarFloat(IConvertible value) {
            this.value = Convert.ToSingle(value);
        }

        internal override ndarray ToArray() {
            return NpyArray.FromAny(value);
        }

        private float value;
    }

}
