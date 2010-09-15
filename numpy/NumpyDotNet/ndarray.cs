using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Reflection;
using System.Numerics;
using IronPython.Runtime;
using IronPython.Modules;
using Microsoft.Scripting;
using NumpyDotNet;

namespace NumpyDotNet
{
    /// <summary>
    /// Implements the Numpy python 'ndarray' object and acts as an interface to
    /// the core NpyArray data structure.  Npy_INTERFACE(NpyArray *) points an 
    /// instance of this class.
    /// </summary>
    public class ndarray : IDisposable
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

            array = IntPtr.Zero;

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
            array = a;
        }


        /// <summary>
        /// Danger!  This method is only intended to be used indirectly during construction
        /// when the new instance is passed into the core as the 'interfaceData' field so
        /// ArrayNewWrapper can pair up this instance with a core object.  If this pointer
        /// is changed after pairing, bad things can happen.
        /// </summary>
        /// <param name="a">Core object to be paired with this wrapper</param>
        internal void SetArray(IntPtr a) {
            if (array == null) {
                throw new InvalidOperationException("Attempt to change core array object for already-constructed wrapper.");
            }
            array = a;
        }

        ~ndarray()
        {
            Dispose(false);
        }

        protected void Dispose(bool disposing)
        {
            if (array != IntPtr.Zero)
            {
                lock (this) {
                    IntPtr a = array;
                    array = IntPtr.Zero;
                    NpyCoreApi.NpyArray_dealloc(a);
                }
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
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
        public virtual int __len__() {
            if (ndim == 0) {
                throw new ArgumentTypeException("len() of unsized object");
            }
            return (int)Dims[0];
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
                            NpyCoreApi.NpyArray_IndexSimple(array, indexes.Indexes, indexes.NumIndexes)
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
            get { return Marshal.ReadInt32(array, NpyCoreApi.ArrayOffsets.off_nd); }
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
        public long size {
            get { return (long)NpyCoreApi.NpyArray_Size(array); }
        }

        /// <summary>
        /// Pointer to the internal memory. Should be used with great caution - memory
        /// is native memory, not managed memory.
        /// </summary>
        public IntPtr data {
            get { return Marshal.ReadIntPtr(array, NpyCoreApi.ArrayOffsets.off_data); }
        }


        /// <summary>
        /// The type descriptor object for this array
        /// </summary>
        public dtype dtype {
            get {
                if (array == IntPtr.Zero) return null;
                IntPtr descr = Marshal.ReadIntPtr(array, NpyCoreApi.ArrayOffsets.off_descr);
                return NpyCoreApi.ToInterface<dtype>(descr);
            }
            set {
                NpyCoreApi.ArraySetDescr(array, value.Descr);
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


        // TODO: Temporary for testing CopyObject method
        public ndarray AssignTo(object src) {
            NpyArray.CopyObject(this, src);
            return this;
        }

        public flatiter flat
        {
            get
            {
                return NpyCoreApi.IterNew(this);
            }
        }


        #endregion


        public ndarray NewCopy(NpyDefs.NPY_ORDER order) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_NewCopy(array, (byte)order));
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
            get { return array; }
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
            get { return NpyCoreApi.NpyArray_ElementStrides(array); }
        }

        public bool StridingOk(NpyDefs.NPY_ORDER order) {
            return order == NpyDefs.NPY_ORDER.NPY_ANYORDER ||
                order == NpyDefs.NPY_ORDER.NPY_CORDER && IsContiguous ||
                order == NpyDefs.NPY_ORDER.NPY_FORTRANORDER && IsFortran;
        }

        private bool ChkFlags(int flag) {
            int curFlags = Marshal.ReadInt32(array, NpyCoreApi.ArrayOffsets.off_flags);
            return ((curFlags & flag) == flag);
        }


        #region Internal methods

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

                MethodInfo repr = value.GetType().GetMethod("__repr__");
                sb.Append(repr != null ? repr.Invoke(repr, null) : value.ToString());
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
        ndarray ArrayBigItem(long index)
        {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                    NpyCoreApi.NpyArray_ArrayItem(Array, (IntPtr)index)
                   );
        }

        #endregion


        private static PythonContext pyContext = null;

        /// <summary>
        ///  Pointer to the native object 
        /// </summary>
        private IntPtr array;

    }
}
