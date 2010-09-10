using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
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


        #region Public interfaces (must match CPython)

        // TODO: Assumes contiguous, C-array for now
        public Object this[params object[] args] {
            get {
                using (NpyIndexes indexes = new NpyIndexes())
                {
                    NpyUtil_IndexProcessing.IndexConverter(args, indexes);
                    ndarray result = NpyCoreApi.DecrefToInterface<ndarray>(
                            NpyCoreApi.NpyArray_Subscript(Array, indexes.Indexes, indexes.NumIndexes));
                    if (result.ndim == 0) {
                        return result.dtype.f.GetItem(0, result);
                    } else {
                        return result;
                    }
                }
            }
            set {
                using (NpyIndexes indexes = new NpyIndexes())
                {
                    NpyUtil_IndexProcessing.IndexConverter(args, indexes);
                    using (ndarray array_value = NpyArray.FromAny(value, null, 0, 0, 0, null))
                    {
                        NpyCoreApi.NpyArray_SubscriptAssign(Array, indexes.Indexes, indexes.NumIndexes, array_value.Array);
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
            // TODO: Temporary implementation
            return String.Format("Array[{0}], type = {1}",
                String.Join(", ", this.Dims.Select(x => x.ToString())), this.dtype.Type);
        }


        public long Stride(int dimension) {
            return NpyCoreApi.GetArrayStride(Array, dimension);
        }
        #endregion


        public ndarray NewCopy(NpyCoreApi.NPY_ORDER order) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_NewCopy(array, (byte)order));
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
            get { return ChkFlags(NpyCoreApi.NPY_CONTIGUOUS); }
        }

        /// <summary>
        /// True if memory layout is Fortran order, false implies C order
        /// </summary>
        public bool IsFortran {
            get { return ChkFlags(NpyCoreApi.NPY_FORTRAN) && ndim > 1; }
        }


        /// <summary>
        /// TODO: What does this return?
        /// </summary>
        public int ElementStrides {
            get { return NpyCoreApi.NpyArray_ElementStrides(array); }
        }

        public bool StridingOk(NpyCoreApi.NPY_ORDER order) {
            return order == NpyCoreApi.NPY_ORDER.NPY_ANYORDER ||
                order == NpyCoreApi.NPY_ORDER.NPY_CORDER && IsContiguous ||
                order == NpyCoreApi.NPY_ORDER.NPY_FORTRANORDER && IsFortran;
        }

        private bool ChkFlags(int flag) {
            int curFlags = Marshal.ReadInt32(array, NpyCoreApi.ArrayOffsets.off_flags);
            return ((curFlags & flag) == flag);
        }


        #region Internal methods

        #endregion


        private static PythonContext pyContext = null;

        /// <summary>
        ///  Pointer to the native object 
        /// </summary>
        private IntPtr array;

    }
}
