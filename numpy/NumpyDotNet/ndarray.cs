using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Reflection;
using System.Numerics;
using IronPython.Runtime;
using IronPython.Runtime.Operations;
using IronPython.Modules;
using Microsoft.Scripting;
using NumpyDotNet;
using System.Collections;

namespace NumpyDotNet
{
    /// <summary>
    /// Implements the Numpy python 'ndarray' object and acts as an interface to
    /// the core NpyArray data structure.  Npy_INTERFACE(NpyArray *) points an 
    /// instance of this class.
    /// </summary>
    public partial class ndarray : Wrapper, IEnumerable<object>
    {
        private static String[] ndarryArgNames = { "shape", "dtype", "buffer",
                                                   "offset", "strides", "order" };

        public ndarray(CodeContext cntx, [ParamDictionary] IAttributesCollection kwargs)
        {
            Object[] posArgs = { };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx.LanguageContext, args);
        }

        public ndarray(CodeContext cntx, Object a1, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx.LanguageContext, args);
        }

        public ndarray(CodeContext cntx, Object a1, Object a2, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1, a2 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx.LanguageContext, args);
        }

        public ndarray(CodeContext cntx, Object a1, Object a2, Object a3, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1, a2, a3 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx.LanguageContext, args);
        }

        public ndarray(CodeContext cntx, Object a1, Object a2, Object a3, Object a4, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1, a2, a3, a4 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx.LanguageContext, args);
        }

        public ndarray(CodeContext cntx, Object a1, Object a2, Object a3, Object a4, 
            Object a5, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1, a2, a3, a4, a5 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx.LanguageContext, args);
        }

        public ndarray(CodeContext cntx, Object a1, Object a2, Object a3, Object a4, 
            Object a5, Object a6, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1, a2, a3, a4, a5, a6 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx.LanguageContext, args);
        }


        /// <summary>
        /// Arguments are: object, dtype, copy, order, subok
        /// </summary>
        /// <param name="cntx"></param>
        /// <param name="args"></param>
        private void Construct(PythonContext cntx, Object[] args) {
            dtype type = null;

            // Ensures that the numeric operations are initialized once at startup.
            // TODO: This is unpleasant, there must be a better way to do this.
            NumericOps.InitUFuncOps(cntx);

            core = IntPtr.Zero;

            long[] shape = NpyUtil_ArgProcessing.IntArrConverter(args[0]);
            if (shape == null) 
                throw new ArgumentException("Array constructor requires a shape to be specified.");

            if (args[1] != null) type = NpyDescr.DescrConverter(cntx, args[1]);
            if (args[2] != null)
                throw new NotImplementedException("Buffer support is not implemented.");
            long offset = NpyUtil_ArgProcessing.IntConverter(args[3]);
            long[] strides = NpyUtil_ArgProcessing.IntArrConverter(args[4]);
            NpyDefs.NPY_ORDER order = NpyUtil_ArgProcessing.OrderConverter(args[5]);

            if (type == null)
                type = NpyCoreApi.DescrFromType(NpyDefs.DefaultType);

            int itemsize = type.ElementSize;
            if (itemsize == 0) {
                throw new ArgumentException("data-type with unspecified variable length");
            }

            if (strides != null) {
                if (strides.Length != shape.Length) {
                    throw new ArgumentException("strides, if given, must be the same length as shape");
                }

                if (!NpyArray.CheckStrides(itemsize, shape, strides)) {
                    throw new ArgumentException("strides is compatible with shape of requested array and size of buffer");
                }
            }

            // Creates a new array object.  By passing 'this' in the current instance
            // becomes the wrapper object for the new array.
            ndarray wrap = NpyCoreApi.NewFromDescr(type, shape, strides, 0, 
                new NpyCoreApi.UseExistingWrapper { Wrapper = this });
            if (wrap != this) {
                throw new InvalidOperationException("Internal error: returned array wrapper is different than current instance.");
            }
            if ((type.Flags & NpyDefs.NPY__ITEM_HASOBJECT) != 0) {
                throw new NotImplementedException("PyArray_FillObject not implemented yet");
            }
        }


        // Creates a wrapper for an array created on the native side, such as the result of a slice operation.
        internal ndarray(IntPtr a)
        {
            core = a;
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

        #region Python methods

        public virtual string __repr__(CodeContext context) {
            // TODO: No support for user-set repr function.
            return BuildStringRepr(true);
        }

        public virtual string __str__(CodeContext context) {
            // TODO: No support for user-set string function
            return BuildStringRepr(false);
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

        internal long Length {
            get {
                return Dims[0];
            }
        }

        public ndarray __abs__() {
            ufunc f = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_absolute);
            return NpyCoreApi.GenericUnaryOp(this, f);
        }

        public ndarray __lshift__(Object b) {
            ufunc f = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_left_shift);
            return NpyCoreApi.GenericBinaryOp(this, NpyArray.FromAny(b, null, 0, 0, 0, null), f);
        }

        public ndarray __rshift__(Object b) {
            ufunc f = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_right_shift);
            return NpyCoreApi.GenericBinaryOp(this, NpyArray.FromAny(b, null, 0, 0, 0, null), f);
        }

        public ndarray __sqrt__() {
            ufunc f = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_sqrt);
            return NpyCoreApi.GenericUnaryOp(this, f);
        }

        public ndarray __mod__(Object b) {
            ufunc f = ufunc.GetFunction("fmod");
            return NpyCoreApi.GenericBinaryOp(this, NpyArray.FromAny(b), f);
        }

        #endregion

        #region Public interfaces (must match CPython)

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

        public object this[string field] {
            set {
                if (!ChkFlags(NpyDefs.NPY_WRITEABLE)) {
                    throw new ArgumentException("array is not writeable.");
                } 
                IntPtr descr;
                int offset = NpyCoreApi.GetFieldOffset(dtype.Descr, field, out descr);
                if (offset < 0) {
                    throw new ArgumentException(String.Format("field name '{0}' not found.", field));
                }
                NpyArray.SetField(this, descr, offset, value);
            }
        }

        public Object this[params object[] args] {
            get {
                using (NpyIndexes indexes = new NpyIndexes())
                {
                    NpyUtil_IndexProcessing.IndexConverter(args, indexes);
                    if (indexes.IsSingleItem(ndim))
                    {
                        // Optimization for single item index.
                        long offset = 0;
                        Int64[] dims = Dims;
                        Int64[] s = strides;
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
                        return GetItem(offset);
                    }

                    // General subscript case.
                    ndarray result = NpyCoreApi.DecrefToInterface<ndarray>(
                            NpyCoreApi.NpyArray_Subscript(Array, indexes.Indexes, indexes.NumIndexes));
                    if (result.ndim == 0) {
                        // TODO: This should return a numpy scalar.
                        return result.dtype.f.GetItem(0, result);
                    } else {
                        return result;
                    }
                }
            }
            set {
                if (args.Length == 1 && args[0] == null)
                {
                    throw new ArgumentException("cannot delete array elements.");
                }
                if (!ChkFlags(NpyDefs.NPY_WRITEABLE))
                {
                    throw new ArgumentException("array is not writeable.");
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
                        // TODO: Handle array subclasses.
                        ndarray view = NpyCoreApi.DecrefToInterface<ndarray>(
                            NpyCoreApi.NpyArray_IndexSimple(core, indexes.Indexes, indexes.NumIndexes)
                            );

                        NpyArray.CopyObject(view, value);
                    }
                    else
                    {
                        using (ndarray array_value = NpyArray.FromAny(value, dtype, 0, 0, NpyDefs.NPY_FORCECAST, null))
                        {
                            if (NpyCoreApi.NpyArray_IndexFancyAssign(Array, indexes.Indexes, indexes.NumIndexes, array_value.Array) < 0)
                            {
                                NpyCoreApi.CheckError();
                            }
                        }
                    }
                }
            }
        }


        /// <summary>
        /// Number of dimensions in the array
        /// </summary>
        public int ndim {
            get { return Marshal.ReadInt32(core, NpyCoreApi.ArrayOffsets.off_nd); }
        }

        /// <summary>
        /// Returns the size of each dimension as a tuple.
        /// </summary>
        public IronPython.Runtime.PythonTuple shape {
            get { return new PythonTuple(this.Dims); }
        }


        /// <summary>
        /// Total number of elements in the array.
        /// </summary>
        public object size {
            get { return NpyCoreApi.NpyArray_Size(core).ToPython(); }
        }

        public long Size {
            get { return NpyCoreApi.NpyArray_Size(core).ToInt64(); }
        }

        /// <summary>
        /// Pointer to the internal memory. Should be used with great caution - memory
        /// is native memory, not managed memory.
        /// </summary>
        public IntPtr data {
            get { return Marshal.ReadIntPtr(core, NpyCoreApi.ArrayOffsets.off_data); }
        }


        /// <summary>
        /// The type descriptor object for this array
        /// </summary>
        public dtype dtype {
            get {
                if (core == IntPtr.Zero) return null;
                IntPtr descr = Marshal.ReadIntPtr(core, NpyCoreApi.ArrayOffsets.off_descr);
                return NpyCoreApi.ToInterface<dtype>(descr);
            }
            set {
                NpyCoreApi.ArraySetDescr(core, value.Descr);
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
        public Int64[] strides {
            get { return NpyCoreApi.GetArrayDimsOrStrides(this, false); }
        }

        public override string ToString() {
            return BuildStringRepr(false);
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

        internal flatiter Flat {
            get {
                return NpyCoreApi.IterNew(this);
            }
        }

        #endregion


        public ndarray NewCopy(NpyDefs.NPY_ORDER order = NpyDefs.NPY_ORDER.NPY_CORDER) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_NewCopy(core, (byte)order));
        }


        /// <summary>
        /// Directly accesses the array memory and returns the object at that
        /// offset.  No checks are made, caller can easily crash the program
        /// or retrieve garbage data.
        /// </summary>
        /// <param name="offset">Offset into data array in bytes</param>
        /// <returns>Contents of the location</returns>
        internal object GetItem(long offset) {
            return dtype.f.GetItem(offset, this);
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
            dtype.f.SetItem(src, offset, this);
        }


        /// <summary>
        /// Handle to the core representation.
        /// </summary>
        public IntPtr Array {
            get { return core; }
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
            return NpyCoreApi.GetArrayStride(Array, dimension);
        }


        /// <summary>
        /// True if memory layout of array is contiguous
        /// </summary>
        public bool IsContiguous {
            get { return ChkFlags(NpyDefs.NPY_CONTIGUOUS); }
        }

        /// <summary>
        /// True if memory layout is Fortran order, false implies C order
        /// </summary>
        public bool IsFortran {
            get { return ChkFlags(NpyDefs.NPY_FORTRAN) && ndim > 1; }
        }

        public bool IsNotSwapped {
            get { return dtype.IsNativeByteOrder; }
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

        /// <summary>
        /// TODO: What does this return?
        /// </summary>
        public int ElementStrides {
            get { return NpyCoreApi.NpyArray_ElementStrides(core); }
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
        #region Operators

        public static ndarray operator +(ndarray a, Object b) {
            ufunc f = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_add);
            return NpyCoreApi.GenericBinaryOp(a, NpyArray.FromAny(b), f);
        }

        public static ndarray operator -(ndarray a, ndarray b) {
            ufunc f = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_subtract);
            return NpyCoreApi.GenericBinaryOp(a, NpyArray.FromAny(b), f);
        }

        public static ndarray operator *(ndarray a, Object b) {
            ufunc f = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_multiply);
            return NpyCoreApi.GenericBinaryOp(a, NpyArray.FromAny(b), f);
        }

        public static ndarray operator /(ndarray a, Object b) {
            ufunc f = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_divide);
            return NpyCoreApi.GenericBinaryOp(a, NpyArray.FromAny(b), f);
        }

        public static ndarray operator&(ndarray a, Object b) {
            ufunc f = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_bitwise_and);
            return NpyCoreApi.GenericBinaryOp(a, NpyArray.FromAny(b), f);
        }

        public static ndarray operator |(ndarray a, Object b) {
            ufunc f = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_bitwise_or);
            return NpyCoreApi.GenericBinaryOp(a, NpyArray.FromAny(b), f);
        }

        public static ndarray operator ^(ndarray a, Object b) {
            ufunc f = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_bitwise_xor);
            return NpyCoreApi.GenericBinaryOp(a, NpyArray.FromAny(b), f);
        }



        // TODO: Temporary test function
        public static ndarray Compare(ndarray a, ndarray b) {
            ufunc f = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_equal);
            return NpyCoreApi.GenericBinaryOp(a, b, f);
        }

        // TODO: end of test functions

        #endregion


        #region python methods from methods.c

        public object all(object axis = null, ndarray @out = null) {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            return ArrayReturn(NpyCoreApi.All(this, iAxis, @out));
        }

        public object any(object axis = null, ndarray @out = null) {
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            return ArrayReturn(NpyCoreApi.Any(this, iAxis, @out));
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
            int iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            NpyDefs.NPY_SORTKIND sortkind = NpyUtil_ArgProcessing.SortkindConverter(kind);

            if (order != null) {
                throw new NotImplementedException("Sort field order not yet implemented.");
            }

            return ArrayReturn(ArgSort(iAxis, sortkind));
        }

        public ndarray astype(CodeContext cntx, object dtype = null) {
            dtype d = NpyDescr.DescrConverter(cntx.LanguageContext, dtype);
            return NpyCoreApi.CastToType(this, d, this.IsFortran);
        }

        public ndarray byteswap(bool inplace = false) {
            return NpyCoreApi.Byteswap(this, inplace);
        }

        public object choose(IEnumerable<object> choices, ndarray @out=null, object mode=null) {
            NpyDefs.NPY_CLIPMODE clipMode = NpyUtil_ArgProcessing.ClipmodeConverter(mode);
            return ArrayReturn(Choose(choices, @out, clipMode));
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

        public ndarray copy(object order = null) {
            NpyDefs.NPY_ORDER eOrder = NpyUtil_ArgProcessing.OrderConverter(order);
            return NpyCoreApi.NewCopy(this, eOrder);
        }

        public ndarray diagonal(int offset = 0, int axis1 = 0, int axis2 = 1) {
            return Diagonal(offset, axis1, axis2);
        }

        public ndarray flatten(object order = null) {
            NpyDefs.NPY_ORDER eOrder = NpyUtil_ArgProcessing.OrderConverter(order);
            return Flatten(eOrder);
        }

        public ndarray getfield(CodeContext cntx, object dtype, int offset = 0) {
            NumpyDotNet.dtype dt = NpyDescr.DescrConverter(cntx.LanguageContext, dtype);
            return NpyCoreApi.GetField(this, dt, offset);
        }
            
        public PythonTuple nonzero() {
            return new PythonTuple(NonZero());
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
                aValues = NpyArray.FromAny(values, dtype, 0, 0, NpyDefs.NPY_CARRAY, null);
            }
            eMode = NpyUtil_ArgProcessing.ClipmodeConverter(mode);
            PutTo(aValues, aIndices, eMode);
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

        public ndarray reshape([ParamDictionary] IAttributesCollection kwds, params object[] args) {
            object[] keywordArgs = NpyUtil_ArgProcessing.BuildArgsArray(new object[0], reshapeKeywords, kwds);
            NpyDefs.NPY_ORDER order = NpyUtil_ArgProcessing.OrderConverter(keywordArgs[0]);
            IntPtr[] newshape;
            // TODO: Add NpyArray_View call for (None) case. (Why?)
            if (args.Length == 1 && args[0] is IList<object>) {
                newshape = NpyUtil_ArgProcessing.IntpListConverter((IList<object>)args[0]);
            } else {
                newshape = NpyUtil_ArgProcessing.IntpListConverter(args);
            }
            return NpyCoreApi.Newshape(this, newshape, order);
        }

        public ndarray ravel(object order = null) {
            NpyDefs.NPY_ORDER eOrder = NpyUtil_ArgProcessing.OrderConverter(order);
            return Ravel(eOrder);
        }

        private static string[] resizeKeywords = { "refcheck" };

        public void resize([ParamDictionary] IAttributesCollection kwds, params object[] args) {
            object[] keywordArgs = NpyUtil_ArgProcessing.BuildArgsArray(new object[0], resizeKeywords, kwds);
            bool refcheck = NpyUtil_ArgProcessing.BoolConverter(keywordArgs[0]);
            IntPtr[] newshape;

            if (args.Length == 0) {
                return;
            }
            if (args.Length == 1 && args[0] is IList<object>) {
                newshape = NpyUtil_ArgProcessing.IntpListConverter((IList<object>)args[0]);
            } else {
                newshape = NpyUtil_ArgProcessing.IntpListConverter(args);
            }
            Resize(newshape, refcheck, NpyDefs.NPY_ORDER.NPY_CORDER);
        }

        public object searchsorted(object keys, string side = null) {
            NpyDefs.NPY_SEARCHSIDE eSide = NpyUtil_ArgProcessing.SearchsideConverter(side);
            ndarray aKeys = (keys as ndarray);
            if (aKeys == null) {
                aKeys = NpyArray.FromAny(keys, NpyArray.FindArrayType(keys, dtype, NpyDefs.NPY_MAXDIMS),
                    0, 0, NpyDefs.NPY_CARRAY, null);
            }
            return ArrayReturn(SearchSorted(aKeys, eSide));
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
                    NpyCoreApi.ClearUPDATEIFCOPY(Array);
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

            
            public ndarray squeeze() {
            return Squeeze();
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

        public ndarray transpose(params object[] args) {
            if (args.Length == 0 || args.Length == 1 && args[0] == null) {
                return Transpose();
            } else if (args.Length == 1 && args[0] is IList<object>) {
                return Transpose(NpyUtil_ArgProcessing.IntpListConverter((IList<object>)args[0]));
            } else {
                return Transpose(NpyUtil_ArgProcessing.IntpListConverter(args));
            }
        }

        #endregion

        #region IEnumerable<object> interface

        public IEnumerator<object> GetEnumerator() {
            return new ndarray_Enumerator(this);
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() {
            return new ndarray_Enumerator(this);
        }

        #endregion

        #region Internal methods

        internal static object ArrayReturn(ndarray a) {
            if (a.ndim == 0) {
                // TODO: This should return a scalar
                return a.GetItem(0);
            } else {
                return a;
            }
        }
        private string BuildStringRepr(bool repr) {
            // Equivalent to array_repr_builtin (arrayobject.c)
            StringBuilder sb = new StringBuilder();
            if (repr) sb.Append("array(");
            if (!DumpData(sb, this.Dims, this.strides, 0, 0)) {
                return null;
            }

            if (repr) {
                if (NpyDefs.IsExtended(this.dtype.TypeNum)) {
                    sb.AppendFormat(", '{0}{1}')", (char)dtype.Type, this.dtype.ElementSize);
                } else {
                    sb.AppendFormat(", '{0}')", (char)dtype.Type);
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
        /// <returns>True on success, false on failure</returns>
        private bool DumpData(StringBuilder sb, long[] dimensions, long[] strides,
            int dimIdx, long offset) {

            if (dimIdx == ndim) {
                Object value = dtype.f.GetItem(offset, this);
                if (value == null) return false;

                // TODO: Calling repr method failed for Python objects. Is ToString() sufficient?
                //MethodInfo repr = value.GetType().GetMethod("__repr__");
                //sb.Append(repr != null ? repr.Invoke(repr, null) : value.ToString());
                sb.Append(value.ToString());
            } else {
                sb.Append('[');
                for (int i = 0; i < dimensions[dimIdx]; i++) {
                    if (!DumpData(sb, dimensions, strides, dimIdx + 1,
                                  offset + strides[dimIdx] * i)) {
                        return false;
                    }
                    if (i < dimensions[dimIdx] - 1) {
                        sb.Append(", ");
                    }
                }
                sb.Append(']');
            }
            return true;
        }

        /// <summary>
        /// Indexes an array by a single long and returns either an item or a sub-array.
        /// </summary>
        /// <param name="index">The index into the array</param>
        object ArrayItem(long index) {
            if (ndim == 1) {
                // TODO: This should really returns a Numpy scalar.
                long dim0 = Dims[0];
                if (index < 0) {
                    index += dim0;
                }
                if (index < 0 || index >= dim0) {
                    throw new IndexOutOfRangeException("Index out of range");
                }
                long offset = index * strides[0];
                return GetItem(offset);
            } else {
                return ArrayBigItem(index);
            }
        }

        /// <summary>
        /// Indexes an array by a single long and returns the sub-array.
        /// </summary>
        /// <param name="index">The index into the array.</param>
        /// <returns>The sub-array.</returns>
        internal ndarray ArrayBigItem(long index)
        {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                    NpyCoreApi.NpyArray_ArrayItem(Array, (IntPtr)index)
                   );
        }

        internal Int32 RawFlags {
            get {
                return Marshal.ReadInt32(Array + NpyCoreApi.ArrayOffsets.off_flags);
            }
            set {
                Marshal.WriteInt32(Array + NpyCoreApi.ArrayOffsets.off_flags, value);
            }
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
