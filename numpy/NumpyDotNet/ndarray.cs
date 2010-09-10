using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Reflection;
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
    public class ndarray
    {
        private static String[] ndarryArgNames = { "object", "dtype", "copy",
                                                     "order", "subok", "ndwin" };

        public ndarray(CodeContext cntx, [ParamDictionary] IAttributesCollection kwargs)
        {
            Object[] posArgs = { };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx, args);
        }

        public ndarray(CodeContext cntx, Object a1, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx, args);
        }

        public ndarray(CodeContext cntx, Object a1, Object a2, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1, a2 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx, args);
        }

        public ndarray(CodeContext cntx, Object a1, Object a2, Object a3, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1, a2, a3 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx, args);
        }

        public ndarray(CodeContext cntx, Object a1, Object a2, Object a3, Object a4, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1, a2, a3, a4 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx, args);
        }

        public ndarray(CodeContext cntx, Object a1, Object a2, Object a3, Object a4, 
            Object a5, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1, a2, a3, a4, a5 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx, args);
        }

        public ndarray(CodeContext cntx, Object a1, Object a2, Object a3, Object a4, 
            Object a5, Object a6, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1, a2, a3, a4, a5, a6 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx, args);
        }


        /// <summary>
        /// Arguments are: object, dtype, copy, order, subok
        /// </summary>
        /// <param name="cntx"></param>
        /// <param name="args"></param>
        private void Construct(CodeContext cntx, Object[] args) {
            if (pyContext == null && cntx != null) pyContext = (PythonContext)cntx.LanguageContext;

            Object dims = args[0];
            dtype type = (dtype)args[1];

            array = IntPtr.Zero;
        }


        // Creates a wrapper for an array created on the native side, such as the result of a slice operation.
        internal ndarray(IntPtr a)
        {
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

        // TODO: Assumes contiguous, C-array for now
        public Object this[params object[] args] {
            get {
                return dtype.f.GetItem(ComputeOffset(args), this);
            }
            set {
                long offset = ComputeOffset(args);
                dtype.f.SetItem(value, offset, this);
                Console.WriteLine(String.Format("{0} = {1} vs {2}", offset, value,
                    dtype.f.GetItem(offset, this)));
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
        /// Pointer to the internal memory. Should be used with great caution - memory
        /// is native memory, not managed memory.
        /// </summary>
        internal IntPtr Data {
            get { return Marshal.ReadIntPtr(array, NpyCoreApi.ArrayOffsets.off_data); }
        }


        /// <summary>
        /// The type descriptor object for this array
        /// </summary>
        public dtype dtype {
            get {
                IntPtr descr = Marshal.ReadIntPtr(array, NpyCoreApi.ArrayOffsets.off_descr);
                return NpyCoreApi.ToInterface<dtype>(descr);
            }
            set {
                NpyCoreApi.ArraySetDescr(array, value.Descr);
            }
        }


        public override string ToString() {
            return BuildStringRepr(false);
        }


        public long Stride(int dimension) {
            return NpyCoreApi.GetArrayStride(Array, dimension);
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
            get { return NpyCoreApi.GetArrayDims(this); }
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

        /// <summary>
        /// Computes an offset into the byte array based on the provided index. The
        /// index may be a sequence of integers or could be a list or tuple of integers.
        /// </summary>
        /// <param name="index">Index - array of ints or tuple/list of ints</param>
        /// <returns>Byte offset into data array</returns>
        private long ComputeOffset(object[] index) {
            long offset = 0;

            if (index.Length == this.ndim) {
                // Since index elements is the same as the number of dimensions we
                // assume that the elements are integers. Anything else is invalid.
                for (int i = 0; i < this.ndim; i++) {
                    long idx = 0;

                    if (index[i] is long) idx = (long)index[i];
                    else if (index[i] is int) idx = (long)(int)index[i];
                    else {
                        throw new IndexOutOfRangeException(
                            String.Format("Index '{0}' at position {1} is not an integer.", index[i].ToString(), i));
                    }
                    offset += this.Stride(i) * idx;
                }
            } else if (index.Length == 1 && index[0] is IEnumerable<Object>) {
                // Index is a sequence such as a tuple or list.
                // TODO: Probably eed a more efficient implementation
                offset = ComputeOffset(((IEnumerable<Object>)index[0]).ToArray());
            } else {
                throw new NotImplementedException("Invalid/unimplemented index type.");
            }
            return offset;
        }

        private string BuildStringRepr(bool repr) {
            // Equivalent to array_repr_builtin (arrayobject.c)
            StringBuilder sb = new StringBuilder();
            if (repr) sb.Append("array(");
            if (!DumpData(sb, this.Dims, 0, 0)) {
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
        private bool DumpData(StringBuilder sb, long[] dimensions,
            int dimIdx, long offset) {

            if (dimIdx == ndim) {
                Object value = dtype.f.GetItem(offset, this);
                if (value == null) return false;

                MethodInfo repr = value.GetType().GetMethod("__repr__");
                sb.Append(repr != null ? repr.Invoke(repr, null) : value.ToString());
            } else {
                sb.Append('[');
                for (int i = 0; i < dimensions[dimIdx]; i++) {
                    if (!DumpData(sb, dimensions, dimIdx + 1,
                                  offset + this.Stride(dimIdx) * i)) {
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

        #endregion


        private static PythonContext pyContext = null;

        /// <summary>
        ///  Pointer to the native object 
        /// </summary>
        private IntPtr array;

    }
}
