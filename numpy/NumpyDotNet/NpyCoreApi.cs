using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security;
using System.Text;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Threading;
using IronPython.Runtime;
using IronPython.Runtime.Types;
using IronPython.Runtime.Operations;
using IronPython.Modules;
using Microsoft.Scripting.Runtime;
using Microsoft.Scripting.Utils;

namespace NumpyDotNet {
    /// <summary>
    /// NpyCoreApi class wraps the interactions with the libndarray core library. It
    /// also makes use of NpyAccessLib.dll for a few functions that must be
    /// implemented in native code.
    ///
    /// TODO: This class is going to get very large.  Not sure if it's better to
    /// try to break it up or just use partial classes and split it across
    /// multiple files.
    /// </summary>
    [SuppressUnmanagedCodeSecurity]
    public static class NpyCoreApi {

        /// <summary>
        /// Stupid hack to allow us to pass an already-allocated wrapper instance
        /// through the interfaceData argument and tell the wrapper creation functions
        /// like ArrayNewWrapper to use an existing instance instead of creating a new
        /// one.  This is necessary because CPython does construction as an allocator
        /// but .NET only triggers code after allocation.
        /// </summary>
        internal struct UseExistingWrapper
        {
            internal object Wrapper;
        }

        #region API Wrappers

        /// <summary>
        /// Returns a new descriptor object for internal types or user defined
        /// types.
        /// </summary>
        internal static dtype DescrFromType(NpyDefs.NPY_TYPES type) {
            // NOTE: No GIL wrapping here, function is re-entrant and includes locking.
            IntPtr descr = NpyArray_DescrFromType((int)type);
            CheckError();
            return DecrefToInterface<dtype>(descr);
        }

        internal static bool IsAligned(ndarray arr) {
            lock (GlobalIterpLock) {
                return Npy_IsAligned(arr.Array) != 0;
            }
        }

        internal static bool IsWriteable(ndarray arr) {
            lock (GlobalIterpLock) {
                return Npy_IsWriteable(arr.Array) != 0;
            }
        }

        internal static byte OppositeByteOrder {
            get { return oppositeByteOrder; }
        }

        internal static byte NativeByteOrder {
            get { return (oppositeByteOrder == '<') ? (byte)'>' : (byte)'<'; }
        }

        internal static dtype SmallType(dtype t1, dtype t2) {
            lock (GlobalIterpLock) {
                return ToInterface<dtype>(
                    NpyArray_SmallType(t1.Descr, t2.Descr));
            }
        }


        /// <summary>
        /// Moves the contents of src into dest.  Arrays are assumed to have the
        /// same number of elements, but can be different sizes and different types.
        /// </summary>
        /// <param name="dest">Destination array</param>
        /// <param name="src">Source array</param>
        internal static void MoveInto(ndarray dest, ndarray src) {
            lock (GlobalIterpLock) {
                if (NpyArray_MoveInto(dest.Array, src.Array) == -1) {
                    CheckError();
                }
            }
        }


        /// <summary>
        /// Allocates a new array and returns the ndarray wrapper
        /// </summary>
        /// <param name="descr">Type descriptor</param>
        /// <param name="numdim">Num of dimensions</param>
        /// <param name="dimensions">Size of each dimension</param>
        /// <param name="fortran">True if Fortran layout, false for C layout</param>
        /// <returns>Newly allocated array</returns>
        internal static ndarray AllocArray(dtype descr, int numdim, long[] dimensions,
            bool fortran) {
            IntPtr nativeDims = IntPtr.Zero;

            Incref(descr.Descr);
            lock (GlobalIterpLock) {
                return DecrefToInterface<ndarray>(
                    NpyArrayAccess_AllocArray(descr.Descr, numdim, dimensions, fortran));
            }
        }

        /// <summary>
        /// Constructs a new array from an input array and descriptor type.  The
        /// Underlying array may or may not be copied depending on the requirements.
        /// </summary>
        /// <param name="src">Source array</param>
        /// <param name="descr">Desired type</param>
        /// <param name="flags">New array flags</param>
        /// <returns>New array (may be source array)</returns>
        internal static ndarray FromArray(ndarray src, dtype descr, int flags) {
            if (descr == null && flags == 0) return src;
            if (descr == null) descr = src.Dtype;
            if (descr != null) NpyCoreApi.Incref(descr.Descr);
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(
                    NpyCoreApi.NpyArray_FromArray(src.Array, descr.Descr, flags));
            }
        }


        /// <summary>
        /// Returns an array with the size or stride of each dimension in the given array.
        /// </summary>
        /// <param name="arr">The array</param>
        /// <param name="getDims">True returns size of each dimension, false returns stride of each dimension</param>
        /// <returns>Array w/ an array size or stride for each dimension</returns>
        internal static Int64[] GetArrayDimsOrStrides(ndarray arr, bool getDims) {
            Int64[] retArr;

            IntPtr srcPtr = Marshal.ReadIntPtr(arr.Array, getDims ? ArrayOffsets.off_dimensions : ArrayOffsets.off_strides);
            retArr = new Int64[arr.ndim];
            unsafe {
                fixed (Int64* dimMem = retArr) {
                    lock (GlobalIterpLock) {
                        if (!GetIntpArray(srcPtr, arr.ndim, dimMem)) {
                            throw new IronPython.Runtime.Exceptions.RuntimeException("Error getting array dimensions.");
                        }
                    }
                }
            }
            return retArr;
        }

        internal static Int64[] GetArrayDims(broadcast iter, bool getDims) {
            Int64[] retArr;

            // off_dimensions is to start of array, not pointer to array!
            IntPtr srcPtr = iter.Iter + MultiIterOffsets.off_dimensions;
            retArr = new Int64[iter.nd];
            unsafe {
                fixed (Int64* dimMem = retArr) {
                    lock (GlobalIterpLock) {
                        if (!GetIntpArray(srcPtr, iter.nd, dimMem)) {
                            throw new IronPython.Runtime.Exceptions.RuntimeException("Error getting iterator dimensions.");
                        }
                    }
                }
            }
            return retArr;
        }

        internal static ndarray NewFromDescr(dtype descr, long[] dims, long[] strides,
            int flags, object interfaceData) {
            if (interfaceData == null) {
                Incref(descr.Descr);
                lock (GlobalIterpLock) {
                    return DecrefToInterface<ndarray>(
                        NewFromDescrThunk(descr.Descr, dims.Length, flags, dims, strides, IntPtr.Zero, IntPtr.Zero));
                }
            } else {
                GCHandle h = AllocGCHandle(interfaceData);
                try {
                    Incref(descr.Descr);
                    Monitor.Enter(GlobalIterpLock);
                    return DecrefToInterface<ndarray>(NewFromDescrThunk(descr.Descr, dims.Length,
                        flags, dims, strides, IntPtr.Zero, GCHandle.ToIntPtr(h)));
                } finally {
                    Monitor.Exit(GlobalIterpLock);
                    FreeGCHandle(h);
                }
            }
        }

        internal static ndarray NewFromDescr(dtype descr, long[] dims, long[] strides, IntPtr data,
            int flags, object interfaceData) {
            if (interfaceData == null) {
                Incref(descr.Descr);
                lock (GlobalIterpLock) {
                    return DecrefToInterface<ndarray>(
                        NewFromDescrThunk(descr.Descr, dims.Length, flags, dims, strides, data, IntPtr.Zero));
                }
            } else {
                GCHandle h = AllocGCHandle(interfaceData);
                try {
                    Incref(descr.Descr);
                    Monitor.Enter(GlobalIterpLock);
                    return DecrefToInterface<ndarray>(NewFromDescrThunk(descr.Descr, dims.Length,
                        flags, dims, strides, IntPtr.Zero, GCHandle.ToIntPtr(h)));
                } finally {
                    Monitor.Exit(GlobalIterpLock);
                    FreeGCHandle(h);
                }
            }
        }

        internal static flatiter IterNew(ndarray ao) {
            lock (GlobalIterpLock) {
                return DecrefToInterface<flatiter>(
                    NpyArray_IterNew(ao.Array));
            }
        }

        internal static ndarray IterSubscript(flatiter iter, NpyIndexes indexes) {
            lock (GlobalIterpLock) {
                return DecrefToInterface<ndarray>(
                    NpyArray_IterSubscript(iter.Iter, indexes.Indexes, indexes.NumIndexes));
            }
        }

        internal static void IterSubscriptAssign(flatiter iter, NpyIndexes indexes, ndarray val) {
            lock (GlobalIterpLock) {
                if (NpyArray_IterSubscriptAssign(iter.Iter, indexes.Indexes, indexes.NumIndexes, val.Array) < 0) {
                    CheckError();
                }
            }
        }

        internal static ndarray FlatView(ndarray a)
        {
            lock (GlobalIterpLock) {
                return DecrefToInterface<ndarray>(
                    NpyArray_FlatView(a.Array)
                    );
            }
        }


        /// <summary>
        /// Creates a multiterator
        /// </summary>
        /// <param name="objs">Sequence of objects to iterate over</param>
        /// <returns>Pointer to core multi-iterator structure</returns>
        internal static IntPtr MultiIterFromObjects(IEnumerable<object> objs) {
            return MultiIterFromArrays(objs.Select(x => NpyArray.FromAny(x)));
        }

        internal static IntPtr MultiIterFromArrays(IEnumerable<ndarray> arrays) {
            IntPtr[] coreArrays = arrays.Select(x => { Incref(x.Array); return x.Array; }).ToArray();
            IntPtr result;

            lock (GlobalIterpLock) {
                result = NpyArrayAccess_MultiIterFromArrays(coreArrays, coreArrays.Length);
            }
            CheckError();
            return result;
        }

        internal static ufunc GetNumericOp(NpyDefs.NpyArray_Ops op) {
            IntPtr ufuncPtr;

            lock (GlobalIterpLock) {
                ufuncPtr = NpyArray_GetNumericOp((int)op);
            }
            return ToInterface<ufunc>(ufuncPtr);
        }

#if NOTDEF
        internal static object GenericUnaryOp(ndarray a1, ufunc f, ndarray ret = null) {
            // TODO: We need to do the error handling and wrapping of outputs.
            Incref(a1.Array);
            Incref(f.UFunc);
            if (ret != null) {
                Incref(ret.Array);
            }
            IntPtr result = NpyArray_GenericUnaryFunction(a1.Array, f.UFunc,
                (ret == null ? IntPtr.Zero : ret.Array));
            ndarray rval = DecrefToInterface<ndarray>(result);
            Decref(a1.Array);
            Decref(f.UFunc);
            if (ret == null) {
                return ndarray.ArrayReturn(rval);
            } else {
                Decref(ret.Array);
                return rval;
            }
        }

        internal static object GenericBinaryOp(ndarray a1, ndarray a2, ufunc f, ndarray ret = null) {
            //ndarray arr = new ndarray[] { a1, a2, ret };
            //return GenericFunction(f, arr, null);
            // TODO: We need to do the error handling and wrapping of outputs.
            Incref(f.UFunc);

            IntPtr result = NpyArray_GenericBinaryFunction(a1.Array, a2.Array, f.UFunc,
                (ret == null ? IntPtr.Zero : ret.Array));
            ndarray rval = DecrefToInterface<ndarray>(result);
            Decref(f.UFunc);

            if (ret == null) {
                return ndarray.ArrayReturn(rval);
            } else {
                return rval;
            }
        }

#endif

        internal static object GenericReduction(ufunc f, ndarray arr,
            ndarray indices, ndarray ret, int axis, dtype otype, ufunc.ReduceOp op) {
            if (indices != null) {
                Incref(indices.Array);
            }

            ndarray rval;
            lock (GlobalIterpLock) {
                rval = DecrefToInterface<ndarray>(
                    NpyUFunc_GenericReduction(f.UFunc, arr.Array,
                        (indices != null) ? indices.Array : IntPtr.Zero,
                        (ret != null) ? ret.Array : IntPtr.Zero,
                        axis, (otype != null) ? otype.Descr : IntPtr.Zero, (int)op));
            }
            if (rval != null) {
                // TODO: Call array wrap processing: ufunc_object.c:1011
            }
            return ndarray.ArrayReturn(rval);
        }

        internal class PrepareArgs
        {
            internal CodeContext cntx;
            internal Action<CodeContext, ufunc, ndarray[], object[]> prepare;
            internal object[] args;
            internal Exception ex;
        }

        internal static int PrepareCallback(IntPtr ufunc, IntPtr arrays, IntPtr prepare_args) {
            PrepareArgs args = (PrepareArgs)GCHandleFromIntPtr(prepare_args).Target;
            ufunc f = ToInterface<ufunc>(ufunc);
            ndarray[] arrs = new ndarray[f.nargs];
            // Copy the data into the array
            for (int i = 0; i < arrs.Length; i++) {
                arrs[i] = DecrefToInterface<ndarray>(Marshal.ReadIntPtr(arrays, IntPtr.Size * i));
            }
            try {
                args.prepare(args.cntx, f, arrs, args.args);
            } catch (Exception ex) {
                args.ex = ex;
                return -1;
            } finally {
                // Copy the arrays back
                for (int i = 0; i < arrs.Length; i++) {
                    IntPtr coreArray = arrs[i].Array;
                    Incref(coreArray);
                    Marshal.WriteIntPtr(arrays, IntPtr.Size * i, arrs[i].Array);
                }
            }
            return 0;
        }

        internal static void GenericFunction(CodeContext cntx, ufunc f, ndarray[] arrays, NpyDefs.NPY_TYPES[] sig,
            Action<CodeContext, ufunc, ndarray[],object[]> prepare_outputs, object[] args) {
            // Convert the typenums
            int[] rtypenums = null;
            int ntypenums = 0;
            if (sig != null) {
                rtypenums = sig.Cast<int>().ToArray();
                ntypenums = rtypenums.Length;
            }
            unsafe {
                // Convert and INCREF the arrays
                IntPtr* mps = stackalloc IntPtr[arrays.Length];
                for (int i = 0; i < arrays.Length; i++) {
                    ndarray a = arrays[i];
                    if (a == null) {
                        mps[i] = IntPtr.Zero;
                    } else {
                        IntPtr p = a.Array;
                        NpyCoreApi.Incref(p);
                        mps[i] = p;
                    }
                }

                if (prepare_outputs != null) {
                    PrepareArgs pargs = new PrepareArgs { cntx = cntx, prepare = prepare_outputs, args = args, ex = null };
                    GCHandle h = AllocGCHandle(pargs);
                    try {
                        int val;
                        Incref(f.UFunc);
                        lock (GlobalIterpLock) {
                            val = NpyUFunc_GenericFunction(f.UFunc, f.nargs, mps, ntypenums, rtypenums, 0,
                                PrepareCallback, GCHandle.ToIntPtr(h));
                        }
                        if (val < 0) {
                            CheckError();
                            if (pargs.ex != null) {
                                throw pargs.ex;
                            }
                        }
                    } finally {
                        // Release the handle
                        FreeGCHandle(h);
                        // Convert the args back.
                        for (int i = 0; i < arrays.Length; i++) {
                            if (mps[i] != IntPtr.Zero) {
                                arrays[i] = DecrefToInterface<ndarray>(mps[i]);
                            } else {
                                arrays[i] = null;
                            }
                        }
                        Decref(f.UFunc);
                    }
                } else {
                    try {
                        Incref(f.UFunc);
                        Monitor.Enter(GlobalIterpLock);
                        if (NpyUFunc_GenericFunction(f.UFunc, f.nargs, mps, ntypenums, rtypenums, 0,
                                null, IntPtr.Zero) < 0) {
                            CheckError();
                        }
                    } finally {
                        Monitor.Exit(GlobalIterpLock);
                        // Convert the args back.
                        for (int i = 0; i < arrays.Length; i++) {
                            if (mps[i] != IntPtr.Zero) {
                                arrays[i] = DecrefToInterface<ndarray>(mps[i]);
                            } else {
                                arrays[i] = null;
                            }
                        }
                        Decref(f.UFunc);
                    }
                }
            }
        }

        internal static ndarray Byteswap(ndarray arr, bool inplace) {
            lock (GlobalIterpLock) {
                return DecrefToInterface<ndarray>(
                    NpyArray_Byteswap(arr.Array, inplace ? (byte)1 : (byte)0));
            }
        }

        public static ndarray CastToType(ndarray arr, dtype d, bool fortran) {
            Incref(d.Descr);
            lock (GlobalIterpLock) {
                return DecrefToInterface<ndarray>(
                    NpyArray_CastToType(arr.Array, d.Descr, (fortran ? 1 : 0)));
            }
        }

        internal static ndarray CheckAxis(ndarray arr, ref int axis, int flags) {
            lock (GlobalIterpLock) {
                return DecrefToInterface<ndarray>(
                    NpyArray_CheckAxis(arr.Array, ref axis, flags));
            }
        }

        internal static void CopyAnyInto(ndarray dest, ndarray src) {
            lock (GlobalIterpLock) {
                if (NpyArray_CopyAnyInto(dest.Array, src.Array) < 0) {
                    CheckError();
                }
            }
        }

        internal static void DescrDestroyFields(IntPtr fields) {
            lock (GlobalIterpLock) {
                NpyDict_Destroy(fields);
            }
        }


        internal static ndarray GetField(ndarray arr, dtype d, int offset) {
            Incref(d.Descr);
            lock (GlobalIterpLock) {
                return DecrefToInterface<ndarray>(
                    NpyArray_GetField(arr.Array, d.Descr, offset));
            }
        }

        internal static ndarray GetImag(ndarray arr) {
            lock (GlobalIterpLock) {
                return DecrefToInterface<ndarray>(
                    NpyArray_GetImag(arr.Array));
            }
        }

        internal static ndarray GetReal(ndarray arr) {
            lock (GlobalIterpLock) {
                return DecrefToInterface<ndarray>(
                    NpyArray_GetReal(arr.Array));
            }
        }
        internal static ndarray GetField(ndarray arr, string name) {
            NpyArray_DescrField field = GetDescrField(arr.Dtype, name);
            dtype field_dtype = ToInterface<dtype>(field.descr);
            return GetField(arr, field_dtype, field.offset);
        }

        internal static ndarray Newshape(ndarray arr, IntPtr[] dims, NpyDefs.NPY_ORDER order) {
            lock (GlobalIterpLock) {
                return DecrefToInterface<ndarray>(
                    NpyArrayAccess_Newshape(arr.Array, dims.Length, dims, (int)order));
            }
        }

        internal static void SetShape(ndarray arr, IntPtr[] dims) {
            lock (GlobalIterpLock) {
                if (NpyArrayAccess_SetShape(arr.Array, dims.Length, dims) < 0) {
                    CheckError();
                }
            }
        }

        internal static void SetState(ndarray arr, IntPtr[] dims, NpyDefs.NPY_ORDER order, string rawdata) {
            lock (GlobalIterpLock) {
                NpyArrayAccess_SetState(arr.Array, dims.Length, dims, (int)order, rawdata, (rawdata != null) ? rawdata.Length : 0);
            }
            CheckError();
        }


        internal static ndarray NewView(dtype d, int nd, IntPtr[] dims, IntPtr[] strides,
            ndarray arr, IntPtr offset, bool ensure_array) {
            Incref(d.Descr);
            lock (GlobalIterpLock) {
                return DecrefToInterface<ndarray>(
                    NpyArray_NewView(d.Descr, nd, dims, strides, arr.Array, offset, ensure_array ? 1 : 0));
            }
        }

        /// <summary>
        /// Returns a copy of the passed array in the specified order (C, Fortran)
        /// </summary>
        /// <param name="arr">Array to copy</param>
        /// <param name="order">Desired order</param>
        /// <returns>New array</returns>
        internal static ndarray NewCopy(ndarray arr, NpyDefs.NPY_ORDER order) {
            lock (GlobalIterpLock) {
                return DecrefToInterface<ndarray>(
                    NpyArray_NewCopy(arr.Array, (int)order));
            }
        }

        internal static NpyDefs.NPY_TYPES TypestrConvert(int elsize, byte letter) {
            lock (GlobalIterpLock) {
                return (NpyDefs.NPY_TYPES)NpyArray_TypestrConvert(elsize, (int)letter);
            }
        }

        internal static void AddField(IntPtr fields, IntPtr names, int i,
            string name, dtype fieldType, int offset, string title) {
            Incref(fieldType.Descr);
            lock (GlobalIterpLock) {
                if (NpyArrayAccess_AddField(fields, names, i, name, fieldType.Descr, offset, title) < 0) {
                    CheckError();
                }
            }
        }

        internal static NpyArray_DescrField GetDescrField(dtype d, string name) {
            NpyArray_DescrField result;
            lock (GlobalIterpLock) {
                if (NpyArrayAccess_GetDescrField(d.Descr, name, out result) < 0) {
                    throw new ArgumentException(String.Format("Field {0} does not exist", name));
                }
            }
            return result;
        }

        internal static dtype DescrNewVoid(IntPtr fields, IntPtr names, int elsize, int flags, int alignment) {
            lock (GlobalIterpLock) {
                return DecrefToInterface<dtype>(
                    NpyArrayAccess_DescrNewVoid(fields, names, elsize, flags, alignment));
            }
        }

        internal static dtype DescrNewSubarray(dtype basetype, IntPtr[] shape) {
            lock (GlobalIterpLock) {
                return DecrefToInterface<dtype>(
                    NpyArrayAccess_DescrNewSubarray(basetype.Descr, shape.Length, shape));
            }
        }

        internal static dtype DescrNew(dtype d) {
            lock (GlobalIterpLock) {
                return DecrefToInterface<dtype>(
                    NpyArray_DescrNew(d.Descr));
            }
        }

        internal static void GetBytes(ndarray arr, byte[] bytes, NpyDefs.NPY_ORDER order) {
            lock (GlobalIterpLock) {
                if (NpyArrayAccess_GetBytes(arr.Array, bytes, bytes.LongLength, (int)order) < 0) {
                    CheckError();
                }
            }
        }

        internal static void FillWithObject(ndarray arr, object obj) {
            GCHandle h = AllocGCHandle(obj);
            try {
                Monitor.Enter(GlobalIterpLock);
                if (NpyArray_FillWithObject(arr.Array, GCHandle.ToIntPtr(h)) < 0) {
                    CheckError();
                }
            } finally {
                Monitor.Exit(GlobalIterpLock);
                FreeGCHandle(h);
            }
        }

        internal static void FillWithScalar(ndarray arr, ndarray zero_d_array) {
            lock (GlobalIterpLock) {
                if (NpyArray_FillWithScalar(arr.Array, zero_d_array.Array) < 0) {
                    CheckError();
                }
            }
        }

        internal static ndarray View(ndarray arr, dtype d, object subtype) {
            IntPtr descr = (d == null ? IntPtr.Zero : d.Descr);
            if (descr != IntPtr.Zero) {
                Incref(descr);
            }
            if (subtype != null) {
                GCHandle h = AllocGCHandle(subtype);
                try {
                    Monitor.Enter(GlobalIterpLock);
                    return DecrefToInterface<ndarray>(
                        NpyArray_View(arr.Array, descr, GCHandle.ToIntPtr(h)));
                } finally {
                    Monitor.Exit(GlobalIterpLock);
                    FreeGCHandle(h);
                }
            }
            else {
                lock (GlobalIterpLock) {
                    return DecrefToInterface<ndarray>(
                        NpyArray_View(arr.Array, descr, IntPtr.Zero));
                }
            }
        }

        internal static ndarray ViewLike(ndarray arr, ndarray proto) {
            lock (GlobalIterpLock) {
                return DecrefToInterface<ndarray>(
                    NpyArrayAccess_ViewLike(arr.Array, proto.Array));
            }
        }

        internal static ndarray Subarray(ndarray self, IntPtr dataptr) {
            lock (GlobalIterpLock) {
                return DecrefToInterface<ndarray>(NpyArray_Subarray(self.Array, dataptr));
            }
        }

        internal static dtype DescrNewByteorder(dtype d, char order) {
            lock (GlobalIterpLock) {
                return DecrefToInterface<dtype>(
                    NpyArray_DescrNewByteorder(d.Descr, (byte)order));
            }
        }

        internal static void UpdateFlags(ndarray arr, int flagmask) {
            lock (GlobalIterpLock) {
                NpyArray_UpdateFlags(arr.Array, flagmask);
            }
        }

        /// <summary>
        /// Calls the fill function on the array dtype.  This takes the first 2 values in the array and fills the array
        /// so the difference between each pair of elements is the same.
        /// </summary>
        /// <param name="arr"></param>
        internal static void Fill(ndarray arr) {
            lock (GlobalIterpLock) {
                if (NpyArrayAccess_Fill(arr.Array) < 0) {
                    CheckError();
                }
            }
        }

        internal static void SetDateTimeInfo(dtype d, string units, int num, int den, int events) {
            lock (GlobalIterpLock) {
                if (NpyArrayAccess_SetDateTimeInfo(d.Descr, units, num, den, events) < 0) {
                    CheckError();
                }
            }
        }

        internal static dtype InheritDescriptor(dtype t1, dtype other) {
            lock (GlobalIterpLock) {
                return DecrefToInterface<dtype>(NpyArrayAccess_InheritDescriptor(t1.Descr, other.Descr));
            }
        }

        internal static bool EquivTypes(dtype d1, dtype d2) {
            lock (GlobalIterpLock) {
                return NpyArray_EquivTypes(d1.Descr, d2.Descr) != 0;
            }
        }

        internal static bool CanCastTo(dtype d1, dtype d2) {
            lock (GlobalIterpLock) {
                return NpyArray_CanCastTo(d1.Descr, d2.Descr);
            }
        }

        /// <summary>
        /// Returns the PEP 3118 format encoding for the type of an array.
        /// </summary>
        /// <param name="arr">Array to get the format string for</param>
        /// <returns>Format string</returns>
        internal static string GetBufferFormatString(ndarray arr) {
            IntPtr ptr;
            lock (GlobalIterpLock) {
                ptr = NpyArrayAccess_GetBufferFormatString(arr.Array);
            }

            String s = Marshal.PtrToStringAnsi(ptr);
            lock (GlobalIterpLock) {
                NpyArrayAccess_Free(ptr); // ptr was allocated with malloc, not SysStringAlloc - don't use automatic marshalling
            }
            return s;
        }


        /// <summary>
        /// Reads the specified text or binary file and produces an array from the content.  Currently only
        /// the file name is allowed and not a PythonFile or Stream type due to limitations in the core
        /// (assumes FILE *).
        /// </summary>
        /// <param name="fileName">File to read</param>
        /// <param name="type">Type descriptor for the resulting array</param>
        /// <param name="count">Number of elements to read, less than zero reads all available</param>
        /// <param name="sep">Element separator string for text files, null for binary files</param>
        /// <returns>Array of file contents</returns>
        internal static ndarray ArrayFromFile(string fileName, dtype type, int count, string sep) {
            lock (GlobalIterpLock) {
                return DecrefToInterface<ndarray>(NpyArrayAccess_FromFile(fileName, (type != null) ? type.Descr : IntPtr.Zero, count, sep));
            }
        }


        internal static ndarray ArrayFromString(string data, dtype type, int count, string sep) {
            if (type != null) Incref(type.Descr);
            lock (GlobalIterpLock) {
                return DecrefToInterface<ndarray>(NpyArray_FromString(data, (IntPtr)data.Length, (type != null) ? type.Descr : IntPtr.Zero, count, sep));
            }
        }

        internal static ndarray ArrayFromBytes(byte[] data, dtype type, int count, string sep) {
            if (type != null) Incref(type.Descr);
            lock (GlobalIterpLock) {
                return DecrefToInterface<ndarray>(NpyArray_FromBytes(data, (IntPtr)data.Length, (type != null) ? type.Descr : IntPtr.Zero, count, sep));
            }
        }

        internal static ndarray CompareStringArrays(ndarray a1, ndarray a2, NpyDefs.NPY_COMPARE_OP op,
                                                    bool rstrip = false) {
            lock (GlobalIterpLock) {
                return DecrefToInterface<ndarray>(
                    NpyArray_CompareStringArrays(a1.Array, a2.Array, (int)op, rstrip ? 1 : 0));
            }
        }

        // API Defintions: every native call is private and must currently be wrapped by a function
        // that at least holds the global interpreter lock (GlobalInterpLock).
        internal static int ElementStrides(ndarray arr) { lock (GlobalIterpLock) { return NpyArray_ElementStrides(arr.Array); } }

        internal static IntPtr ArraySubscript(ndarray arr, NpyIndexes indexes) {
            lock (GlobalIterpLock) { return NpyArray_Subscript(arr.Array, indexes.Indexes, indexes.NumIndexes); }
        }

        internal static void IndexDealloc(NpyIndexes indexes) {
            lock (GlobalIterpLock) { NpyArray_IndexDealloc(indexes.Indexes, indexes.NumIndexes); }
        }

        internal static IntPtr ArraySize(ndarray arr) {
            lock (GlobalIterpLock) { return NpyArray_Size(arr.Array); }
        }

        /// <summary>
        /// Indexes an array by a single long and returns the sub-array.
        /// </summary>
        /// <param name="index">The index into the array.</param>
        /// <returns>The sub-array.</returns>
        internal static ndarray ArrayItem(ndarray arr, long index) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(NpyCoreApi.NpyArray_ArrayItem(arr.Array, (IntPtr)index));
            }
        }

        internal static ndarray IndexSimple(ndarray arr, NpyIndexes indexes) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(NpyCoreApi.NpyArray_IndexSimple(arr.Array, indexes.Indexes, indexes.NumIndexes));
            }
        }

        internal static int IndexFancyAssign(ndarray dest, NpyIndexes indexes, ndarray values) {
            lock (GlobalIterpLock) { return NpyArray_IndexFancyAssign(dest.Array, indexes.Indexes, indexes.NumIndexes, values.Array); }
        }

        internal static int SetField(ndarray arr, IntPtr dtype, int offset, ndarray srcArray) {
            lock (GlobalIterpLock) { return NpyArray_SetField(arr.Array, dtype, offset, srcArray.Array); }
        }

        internal static void SetNumericOp(int op, ufunc ufunc) {
            lock (GlobalIterpLock) { NpyArray_SetNumericOp(op, ufunc.UFunc); }
        }

        internal static ndarray ArrayAll(ndarray arr, int axis, ndarray ret = null) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(NpyCoreApi.NpyArray_All(arr.Array, axis, (ret == null ? IntPtr.Zero : ret.Array)));
            }
        }

        internal static ndarray ArrayAny(ndarray arr, int axis, ndarray ret = null) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(NpyCoreApi.NpyArray_Any(arr.Array, axis, (ret == null ? IntPtr.Zero : ret.Array)));
            }
        }

        internal static ndarray ArrayArgMax(ndarray self, int axis, ndarray ret) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(NpyCoreApi.NpyArray_ArgMax(self.Array, axis, (ret == null ? IntPtr.Zero : ret.Array)));
            }
        }

        internal static ndarray ArgSort(ndarray arr, int axis, NpyDefs.NPY_SORTKIND sortkind) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(
                    NpyCoreApi.NpyArray_ArgSort(arr.Array, axis, (int)sortkind));
            }
        }

        internal static int ArrayBool(ndarray arr) {
            lock (GlobalIterpLock) { return NpyArray_Bool(arr.Array); }
        }

        internal static int ScalarKind(int typenum, ndarray arr) {
            lock (GlobalIterpLock) { return NpyArray_ScalarKind(typenum, arr.Array); }
        }

        internal static ndarray Choose(ndarray sel, ndarray[] arrays, ndarray ret = null, NpyDefs.NPY_CLIPMODE clipMode = NpyDefs.NPY_CLIPMODE.NPY_RAISE) {
            lock (GlobalIterpLock) {
                IntPtr[] coreArrays = arrays.Select(x => x.Array).ToArray();
                return NpyCoreApi.DecrefToInterface<ndarray>(
                    NpyCoreApi.NpyArray_Choose(sel.Array, coreArrays, coreArrays.Length,
                    ret == null ? IntPtr.Zero : ret.Array, (int)clipMode));
            }
        }

        internal static ndarray Conjugate(ndarray arr, ndarray ret = null) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(NpyArray_Conjugate(arr.Array, (ret == null ? IntPtr.Zero : ret.Array)));
            }
        }

        internal static ndarray Correlate(ndarray arr1, ndarray arr2, NpyDefs.NPY_TYPES typenum, int mode) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(NpyArray_Correlate(arr1.Array, arr2.Array, (int)typenum, mode));
            }
        }

        internal static ndarray Correlate2(ndarray arr1, ndarray arr2, NpyDefs.NPY_TYPES typenum, int mode) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(NpyArray_Correlate2(arr1.Array, arr2.Array, (int)typenum, mode));
            }
        }

        internal static ndarray CopyAndTranspose(ndarray arr) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(NpyArray_CopyAndTranspose(arr.Array));
            }
        }

        internal static ndarray CumProd(ndarray arr, int axis, dtype rtype, ndarray ret = null) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(
                    NpyCoreApi.NpyArray_CumProd(arr.Array, axis,
                        (int)(rtype == null ? NpyDefs.NPY_TYPES.NPY_NOTYPE : rtype.TypeNum),
                        (ret == null ? IntPtr.Zero : ret.Array)));
            }
        }

        internal static ndarray CumSum(ndarray arr, int axis, dtype rtype, ndarray ret = null) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(
                    NpyCoreApi.NpyArray_CumSum(arr.Array, axis,
                        (int)(rtype == null ? NpyDefs.NPY_TYPES.NPY_NOTYPE : rtype.TypeNum),
                        (ret == null ? IntPtr.Zero : ret.Array)));
            }
        }

        internal static void DestroySubarray(IntPtr subarrayPtr) {
            lock (GlobalIterpLock) { NpyArray_DestroySubarray(subarrayPtr); }
        }

        internal static int DescrFindObjectFlag(dtype type) {
            lock (GlobalIterpLock) { return NpyArray_DescrFindObjectFlag(type.Descr); }
        }

        internal static ndarray Flatten(ndarray arr, NpyDefs.NPY_ORDER order) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Flatten(arr.Array, (int)order));
        }

        internal static ndarray InnerProduct(ndarray arr1, ndarray arr2, NpyDefs.NPY_TYPES type) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(NpyCoreApi.NpyArray_InnerProduct(arr1.Array, arr2.Array, (int)type));
            }
        }

        internal static ndarray LexSort(ndarray[] arrays, int axis) {
            int n = arrays.Length;
            IntPtr[] coreArrays = arrays.Select(x => x.Array).ToArray();
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(
                    NpyCoreApi.NpyArray_LexSort(coreArrays, n, axis));
            }
        }

        internal static ndarray MatrixProduct(ndarray arr1, ndarray arr2, NpyDefs.NPY_TYPES type) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(NpyArray_MatrixProduct(arr1.Array, arr2.Array, (int)type));
            }
        }

        internal static ndarray ArrayMax(ndarray arr, int axis, ndarray ret = null) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(NpyCoreApi.NpyArray_Max(arr.Array, axis, (ret == null ? IntPtr.Zero : ret.Array)));
            }
        }

        internal static ndarray ArrayMin(ndarray arr, int axis, ndarray ret = null) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(NpyCoreApi.NpyArray_Min(arr.Array, axis, (ret == null ? IntPtr.Zero : ret.Array)));
            }
        }

        internal static ndarray[] NonZero(ndarray arr) {
            int nd = arr.ndim;
            IntPtr[] coreArrays = new IntPtr[nd];
            GCHandle h = NpyCoreApi.AllocGCHandle(arr);
            try {
                Monitor.Enter(GlobalIterpLock);
                if (NpyCoreApi.NpyArray_NonZero(arr.Array, coreArrays, GCHandle.ToIntPtr(h)) < 0) {
                    NpyCoreApi.CheckError();
                }
            } finally {
                Monitor.Exit(GlobalIterpLock);
                NpyCoreApi.FreeGCHandle(h);
            }
            return coreArrays.Select(x => NpyCoreApi.DecrefToInterface<ndarray>(x)).ToArray();
        }

        internal static ndarray Prod(ndarray arr, int axis, dtype rtype, ndarray ret = null) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(
                    NpyCoreApi.NpyArray_Prod(arr.Array, axis,
                        (int)(rtype == null ? NpyDefs.NPY_TYPES.NPY_NOTYPE : rtype.TypeNum),
                        (ret == null ? IntPtr.Zero : ret.Array)));
            }
        }

        internal static int PutMask(ndarray arr, ndarray values, ndarray mask) {
            lock (GlobalIterpLock) {
                return NpyArray_PutMask(arr.Array, values.Array, mask.Array);
            }
        }

        internal static int PutTo(ndarray arr, ndarray values, ndarray indices, NpyDefs.NPY_CLIPMODE clipmode) {
            lock (GlobalIterpLock) {
                return NpyArray_PutTo(arr.Array, values.Array, indices.Array, (int)clipmode);
            }
        }


        internal static ndarray Ravel(ndarray arr, NpyDefs.NPY_ORDER order) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(NpyCoreApi.NpyArray_Ravel(arr.Array, (int)order));
            }
        }

        internal static ndarray Repeat(ndarray arr, ndarray repeats, int axis) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(NpyCoreApi.NpyArray_Repeat(arr.Array, repeats.Array, axis));
            }
        }

        internal static ndarray Searchsorted(ndarray arr, ndarray keys, NpyDefs.NPY_SEARCHSIDE side) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(NpyCoreApi.NpyArray_SearchSorted(arr.Array, keys.Array, (int)side));
            }
        }

        internal static void Sort(ndarray arr, int axis, NpyDefs.NPY_SORTKIND sortkind) {
            lock (GlobalIterpLock) {
                if (NpyCoreApi.NpyArray_Sort(arr.Array, axis, (int)sortkind) < 0) {
                    NpyCoreApi.CheckError();
                }
            }
        }

        internal static ndarray Squeeze(ndarray arr) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(NpyCoreApi.NpyArray_Squeeze(arr.Array));
            }
        }

        internal static ndarray Sum(ndarray arr, int axis, dtype rtype, ndarray ret = null) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Sum(arr.Array, axis,
                    (int)(rtype == null ? NpyDefs.NPY_TYPES.NPY_NOTYPE : rtype.TypeNum),
                    (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal static ndarray SwapAxis(ndarray arr, int a1, int a2) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(NpyCoreApi.NpyArray_SwapAxes(arr.Array, a1, a2));
            }
        }

        internal static ndarray TakeFrom(ndarray arr, ndarray indices, int axis, ndarray ret, NpyDefs.NPY_CLIPMODE clipMode) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(
                    NpyCoreApi.NpyArray_TakeFrom(arr.Array, indices.Array, axis, (ret != null ? ret.Array : IntPtr.Zero), (int)clipMode)
                    );
            }
        }

        internal static bool DescrIsNative(dtype type) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DescrIsNative(type.Descr) != 0;
            }
        }

        #endregion


        #region C API Definitions

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_DescrNew(IntPtr descr);
        internal static IntPtr DescrNewRaw(dtype d) {
            lock (GlobalIterpLock) { return NpyArray_DescrNew(d.Descr); }
        }

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_DescrFromType(Int32 type);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_SmallType(IntPtr descr1, IntPtr descr2);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern byte NpyArray_EquivTypes(IntPtr t1, IntPtr typ2);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArray_ElementStrides(IntPtr arr);
              
        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArray_MoveInto(IntPtr dest, IntPtr src);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_FromArray(IntPtr arr, IntPtr descr, int flags);

        //[DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        //private static extern void NpyArray_dealloc(IntPtr arr);

        //[DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        //private static extern void NpyArray_DescrDestroy(IntPtr arr);

        //[DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        //internal static extern void NpyArray_DescrDeallocNamesAndFields(IntPtr dtype);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_Subarray(IntPtr arr, IntPtr dataptr);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_Subscript(IntPtr arr, IntPtr indexes, int n);

        //[DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        //private static extern int NpyArray_SubscriptAssign(IntPtr self, IntPtr indexes, int n, IntPtr value);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NpyArray_IndexDealloc(IntPtr indexes, int n);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_Size(IntPtr arr);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_ArrayItem(IntPtr array, IntPtr index);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_IndexSimple(IntPtr arr, IntPtr indexes, int n);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArray_IndexFancyAssign(IntPtr dest, IntPtr indexes, int n, IntPtr value_array);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArray_SetField(IntPtr arr, IntPtr descr, int offset, IntPtr val);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern int Npy_IsAligned(IntPtr arr);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern int Npy_IsWriteable(IntPtr arr);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_IterNew(IntPtr ao);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_IterSubscript(IntPtr iter, IntPtr indexes, int n);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArray_IterSubscriptAssign(IntPtr iter, IntPtr indexes, int n, IntPtr array_val);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArray_FillWithObject(IntPtr arr, IntPtr obj);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArray_FillWithScalar(IntPtr arr, IntPtr zero_d_array);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_FlatView(IntPtr arr);

        //[DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        //private static extern void npy_ufunc_dealloc(IntPtr arr);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_GetNumericOp(int op);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NpyArray_SetNumericOp(int op, IntPtr ufunc);
        
        //[DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        //internal static extern IntPtr NpyArray_GenericUnaryFunction(IntPtr arr1, IntPtr ufunc, IntPtr ret);

        //[DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        //internal static extern IntPtr NpyArray_GenericBinaryFunction(IntPtr arr1, IntPtr arr2, IntPtr ufunc, IntPtr ret);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_All(IntPtr self, int axis, IntPtr ret);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_Any(IntPtr self, int axis, IntPtr ret);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_ArgMax(IntPtr self, int axis, IntPtr ret);
        
        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_ArgSort(IntPtr arr, int axis, int sortkind);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArray_Bool(IntPtr arr);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArray_ScalarKind(int typenum, IntPtr arr);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_Byteswap(IntPtr arr, byte inplace);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern bool NpyArray_CanCastTo(IntPtr fromDtype, IntPtr toDtype);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_CastToType(IntPtr array, IntPtr descr, int fortran);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_CheckAxis(IntPtr arr, ref int axis,
                                                         int flags);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_Choose(IntPtr array,
            [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 2)]IntPtr[] mps, int n, IntPtr ret, int clipMode);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_CompareStringArrays(IntPtr a1, IntPtr a2,
                                                                   int op, int rstrip);
        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_Conjugate(IntPtr arr, IntPtr ret);
        
        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_Correlate(IntPtr arr1, IntPtr arr2, int typenum, int mode);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_Correlate2(IntPtr arr1, IntPtr arr2, int typenum, int mode);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_CopyAndTranspose(IntPtr arr);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArray_CopyAnyInto(IntPtr dest, IntPtr src);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_CumProd(IntPtr arr, int axis, int
                                                       rtype, IntPtr ret);


        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_CumSum(IntPtr arr, int axis, int
                                                      rtype, IntPtr ret);
        
        // Reentrant - does not need to be wrapped.
        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl, EntryPoint="NpyArray_DescrAllocNames")]
        internal static extern IntPtr DescrAllocNames(int n);

        // Reentrant - does not need to be wrapped.
        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl, EntryPoint="NpyArray_DescrAllocFields")]
        internal static extern IntPtr DescrAllocFields();

        /// <summary>
        /// Deallocates a subarray block.  The pointer passed in is descr->subarray, not
        /// a descriptor object itself.
        /// </summary>
        /// <param name="subarrayPtr">Subarray structure</param>
        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NpyArray_DestroySubarray(IntPtr subarrayPtr);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl,
            EntryPoint="npy_descr_find_object_flag")]
        private static extern int NpyArray_DescrFindObjectFlag(IntPtr subarrayPtr);

        // Reentrant -- does not need to be wrapped.
        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl, EntryPoint="NpyArray_DateTimeInfoNew")]
        internal static extern IntPtr DateTimeInfoNew(string units, int num, int den, int events);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_DescrNewByteorder(IntPtr descr, byte order);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_Flatten(IntPtr arr, int order);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_GetField(IntPtr arr, IntPtr dtype, int offset);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_GetImag(IntPtr arr);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_GetReal(IntPtr arr);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_InnerProduct(IntPtr arr, IntPtr arr2, int type);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_LexSort(
            [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] IntPtr[] mps, int n, int axis);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_MatrixProduct(IntPtr arr, IntPtr arr2, int type);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_Max(IntPtr arr, int axis, IntPtr ret);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_Min(IntPtr arr, int axis, IntPtr ret);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_NewCopy(IntPtr arr, int order);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_NewView(IntPtr descr, int nd,
            [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)]IntPtr[] dims,
            [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)]IntPtr[] strides,
            IntPtr arr, IntPtr offset, int ensureArray);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArray_NonZero(IntPtr self,
            [MarshalAs(UnmanagedType.LPArray,SizeConst=NpyDefs.NPY_MAXDIMS)] IntPtr[] index_arrays,
            IntPtr obj);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_Prod(IntPtr arr, int axis, int
                                                    rtype, IntPtr ret);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArray_PutMask(IntPtr arr, IntPtr values, IntPtr mask);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArray_PutTo(IntPtr arr, IntPtr values, IntPtr indices, int clipmode);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_Ravel(IntPtr arr, int fortran);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_Repeat(IntPtr arr, IntPtr repeats, int axis);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_SearchSorted(IntPtr op1, IntPtr op2, int side);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArray_Sort(IntPtr arr, int axis, int sortkind);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_Squeeze(IntPtr self);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_Sum(IntPtr arr, int axis, int rtype, IntPtr ret);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_SwapAxes(IntPtr arr, int a1, int a2);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_TakeFrom(IntPtr self, IntPtr indices, int axis, IntPtr ret, int clipMode);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArray_TypestrConvert(int itemsize, int gentype);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NpyArray_UpdateFlags(IntPtr arr, int flagmask);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_View(IntPtr arr, IntPtr descr, IntPtr subtype);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NpyDict_Destroy(IntPtr dict);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate int del_PrepareOutputs(IntPtr ufunc, IntPtr arrays, IntPtr args);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static unsafe extern int NpyUFunc_GenericFunction(IntPtr func, int nargs, IntPtr* mps,
            int ntypenums, [In][MarshalAs(UnmanagedType.LPArray)] int[] rtypenums,
            int originalObjectWasArray, del_PrepareOutputs npy_prepare_outputs_func, IntPtr prepare_out_args);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyUFunc_GenericReduction(IntPtr ufunc,
            IntPtr arr, IntPtr indices, IntPtr arrOut, int axis, IntPtr descr,
            int operation);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal unsafe delegate void del_GetErrorState(int* bufsizep, int* maskp, IntPtr* objp);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal unsafe delegate void del_ErrorHandler(sbyte* name, int errormask, IntPtr errobj, int retstatus, int* first);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NpyUFunc_SetFpErrFuncs(del_GetErrorState errorState, del_ErrorHandler handler);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_FromString(string data, IntPtr len, IntPtr dtype, int num, string sep);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl, EntryPoint = "NpyArray_FromString")]
        private static extern IntPtr NpyArray_FromBytes(byte[] data, IntPtr len, IntPtr dtype, int num, string sep);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl, EntryPoint="npy_arraydescr_isnative")]
        private static extern int DescrIsNative(IntPtr descr);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern void npy_initlib(IntPtr functionDefs, IntPtr wrapperFuncs,
            IntPtr error_set, IntPtr error_occured, IntPtr error_clear,
            IntPtr cmp_priority, IntPtr incref, IntPtr decref,
            IntPtr enable_thread, IntPtr disable_thread);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl, EntryPoint = "npy_add_sortfuncs")]
        private static extern IntPtr NpyArray_InitSortModule();

        #endregion

        #region NpyAccessLib functions

        internal static void ArraySetDescr(ndarray arr, dtype newDescr) {
            lock (GlobalIterpLock) { NpyArrayAccess_ArraySetDescr(arr.Array, newDescr.Descr); }
        }

        internal static long GetArrayStride(ndarray arr, int dims) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.NpyArrayAccess_GetArrayStride(arr.Array, dims);
            }
        }

        internal static int BindIndex(ndarray arr, NpyIndexes indexes, NpyIndexes result) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.NpyArrayAccess_BindIndex(arr.Array, indexes.Indexes, indexes.NumIndexes, result.Indexes);
            }
        }

        internal static int GetFieldOffset(dtype descr, string fieldName, out IntPtr descrPtr) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.NpyArrayAccess_GetFieldOffset(descr.Descr, fieldName, out descrPtr);
            }
        }

        internal static void Resize(ndarray arr, IntPtr[] newshape, bool refcheck, NpyDefs.NPY_ORDER order) {
            lock (GlobalIterpLock) {
                if (NpyCoreApi.NpyArrayAccess_Resize(arr.Array, newshape.Length, newshape, (refcheck ? 1 : 0), (int)order) < 0) {
                    NpyCoreApi.CheckError();
                }
            }
        }

        internal static ndarray Transpose(ndarray arr, IntPtr[] permute) {
            lock (GlobalIterpLock) {
                return NpyCoreApi.DecrefToInterface<ndarray>(
                    NpyCoreApi.NpyArrayAccess_Transpose(arr.Array, (permute != null) ? permute.Length : 0, permute));
            }
        }

        internal static void ClearUPDATEIFCOPY(ndarray arr) {
            lock (GlobalIterpLock) {
                NpyArrayAccess_ClearUPDATEIFCOPY(arr.Array);
            }
        }


        internal static IntPtr IterNext(IntPtr corePtr) {
            lock (GlobalIterpLock) {
                return NpyArrayAccess_IterNext(corePtr);
            }
        }

        internal static void IterReset(IntPtr iter) {
            lock (GlobalIterpLock) {
                NpyArrayAccess_IterReset(iter);
            }
        }

        internal static IntPtr IterGoto1D(flatiter iter, IntPtr index) {
            lock (GlobalIterpLock) {
                return NpyArrayAccess_IterGoto1D(iter.Iter, index);
            }
        }

        internal static IntPtr IterCoords(flatiter iter) {
            lock (GlobalIterpLock) {
                return NpyArrayAccess_IterCoords(iter.Iter);
            }
        }

        internal static void DescrReplaceSubarray(dtype descr, dtype baseDescr, IntPtr[] dims) {
            lock (GlobalIterpLock) {
                NpyArrayAccess_DescrReplaceSubarray(descr.Descr, baseDescr.Descr, dims.Length, dims);
            }
        }

        internal static void DescrReplaceFields(dtype descr, IntPtr namesPtr, IntPtr fieldsDict) {
            lock (GlobalIterpLock) {
                NpyArrayAccess_DescrReplaceFields(descr.Descr, namesPtr, fieldsDict);
            }
        }

        internal static void ZeroFill(ndarray arr, IntPtr offset) {
            lock (GlobalIterpLock) {
                NpyArrayAccess_ZeroFill(arr.Array, offset);
            }
        }

        /// <summary>
        /// Allocates a block of memory using NpyDataMem_NEW that is the same size as a single
        /// array element and zeros the bytes.  This is usually good enough, but is not a correct
        /// zero for object arrays.  The caller must free the memory with NpyDataMem_FREE().
        /// </summary>
        /// <param name="arr">Array to take the element size from</param>
        /// <returns>Pointer to zero'd memory</returns>
        internal static IntPtr DupZeroElem(ndarray arr) {
            lock (GlobalIterpLock) {
                return NpyArrayAccess_DupZeroElem(arr.Array);
            }
        }

        internal unsafe static void CopySwapIn(ndarray arr, long offset, void* data, bool swap) {
            lock (GlobalIterpLock) {
                NpyArrayAccess_CopySwapIn(arr.Array, offset, data, swap ? 1 : 0);
            }
        }

        internal unsafe static void CopySwapOut(ndarray arr, long offset, void* data, bool swap) {
            lock (GlobalIterpLock) {
                NpyArrayAccess_CopySwapOut(arr.Array, offset, data, swap ? 1 : 0);
            }
        }

        internal unsafe static void CopySwapScalar(dtype dtype, void* dest, void* src, bool swap) {
            lock (GlobalIterpLock) {
                NpyArrayAccess_CopySwapScalar(dtype.Descr, dest, src, swap);
            }
        }

        internal static void SetNamesList(dtype descr, string[] nameslist) {
            lock (GlobalIterpLock) {
                NpyArrayAccess_SetNamesList(descr.Descr, nameslist, nameslist.Length);
            }
        }

        /// <summary>
        /// Deallocates the core data structure.  The obj IntRef is no longer valid after this
        /// point and there must not be any existing internal core references to this object
        /// either.
        /// </summary>
        /// <param name="obj">Core NpyObject instance to deallocate</param>
        internal static void Dealloc(IntPtr obj) {
            lock(GlobalIterpLock) { 
                NpyArrayAccess_Dealloc(obj); 
            }
        }


        /// <summary>
        /// Constructs a native ufunc object from a Python function.  The inputs define the
        /// number of arguments taken, number of outputs, and function name.  The pyLoopFunc
        /// function implements the iteration over a given array and should always by PyUFunc_Om_On.
        /// pyFunc is the actual function object to call.
        /// </summary>
        /// <param name="nin">Number of input arguments</param>
        /// <param name="nout">Number of result values (a PythonTuple if > 1)</param>
        /// <param name="funcName">Name of the function</param>
        /// <param name="pyWrapperFunc">PyUFunc_Om_On, implements looping over the array</param>
        /// <param name="pyFunc">Function to call</param>
        internal static IntPtr UFuncFromPyFunc(int nin, int nout, String funcName,
            IntPtr pyWrapperFunc, IntPtr pyFunc) {
            lock (GlobalIterpLock) {
                return NpyUFuncAccess_UFuncFromPyFunc(nin, nout, funcName, pyWrapperFunc, pyFunc);
            }
        }


        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void NpyUFuncAccess_Init(IntPtr funcDict,
            IntPtr funcDefs, IntPtr callMethodFunc, IntPtr addToDictFunc);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NpyArrayAccess_ArraySetDescr(IntPtr array, IntPtr newDescr);

        /// <summary>
        /// Increments the reference count of the core object.  This routine is re-entrant and
        /// locking is handled at the bottom layer.
        /// </summary>
        /// <param name="obj">Pointer to the core object to increment reference count to</param>
        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint="NpyArrayAccess_Incref")]
        internal static extern void Incref(IntPtr obj);

        /// <summary>
        /// Decrements the reference count of the core object.  This can trigger the release of
        /// the reference to the managed wrapper and eventually trigger a garbage collection of
        /// the object.  If the core object does not have a managed wrapper, this can trigger the
        /// immediate destruction of the core object. 
        /// 
        /// This function is re-entrant/thread-safe.
        /// </summary>
        /// <param name="obj">Pointer to the core object</param>
        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint="NpyArrayAccess_Decref")]
        internal static extern void Decref(IntPtr obj);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_GetNativeTypeInfo")]
        private static extern byte GetNativeTypeInfo(out int intSize,
            out int longsize, out int longLongSize, out int longDoubleSize);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_GetIntpArray")]
        unsafe private static extern bool GetIntpArray(IntPtr srcPtr, int len, Int64 *dimMem);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArrayAccess_AllocArray(IntPtr descr, int nd,
            [In][MarshalAs(UnmanagedType.LPArray,SizeParamIndex=1)] long[] dims, bool fortran);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern long NpyArrayAccess_GetArrayStride(IntPtr arr, int dims);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArrayAccess_BindIndex(IntPtr arr, IntPtr indexes, int n, IntPtr bound_indexes);

        [StructLayout(LayoutKind.Sequential)]
        internal struct NpyArray_DescrField
        {
            internal IntPtr descr;
            internal int offset;
            internal IntPtr title;
        }

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArrayAccess_GetDescrField(IntPtr descr,
            [In][MarshalAs(UnmanagedType.LPStr)]string name, out NpyArray_DescrField field);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArrayAccess_GetFieldOffset(IntPtr descr, [MarshalAs(UnmanagedType.LPStr)] string fieldName, out IntPtr out_descr);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArrayAccess_MultiIterFromArrays([MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)]IntPtr[] arrays, int n);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArrayAccess_Newshape(IntPtr arr, int ndim,
            [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)]IntPtr[] dims,
            int order);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArrayAccess_SetShape(IntPtr arr, int ndim,
            [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)]IntPtr[] dims);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NpyArrayAccess_SetState(IntPtr arr, int ndim,
            [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)]IntPtr[] dims, int order,
            // Note string is marshalled as LPWStr (16-bit unicode) to avoid making a copy of it
            [MarshalAsAttribute(UnmanagedType.LPWStr)]string rawdata, int rawLength);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArrayAccess_Resize(IntPtr arr, int ndim,
            [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] IntPtr[] newshape, int resize, int fortran);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArrayAccess_Transpose(IntPtr arr, int ndim,
            [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] IntPtr[] permute);

        /// <summary>
        /// Returns the current ABI version.  Re-entrant, does not need locking.
        /// </summary>
        /// <returns>current version #</returns>
        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl, EntryPoint="NpyArrayAccess_GetAbiVersion")]
        internal static extern float GetAbiVersion();

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NpyArrayAccess_ClearUPDATEIFCOPY(IntPtr arr);


        /// <summary>
        /// Deallocates an NpyObject. Thread-safe.
        /// </summary>
        /// <param name="obj">The object to deallocate</param>
        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NpyArrayAccess_Dealloc(IntPtr obj);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_IterNext")]
        private static extern IntPtr NpyArrayAccess_IterNext(IntPtr iter);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_IterReset")]
        private static extern void NpyArrayAccess_IterReset(IntPtr iter);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_IterGoto1D")]
        private static extern IntPtr NpyArrayAccess_IterGoto1D(IntPtr iter, IntPtr index);

        // Re-entrant
        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_IterArray")]
        internal static extern IntPtr IterArray(IntPtr iter);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArrayAccess_IterCoords(IntPtr iter);

        //
        // Offset functions - these return the offsets to fields in native structures
        // as a workaround for not being able to include the C header file.
        //

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_ArrayGetOffsets")]
        private static extern void ArrayGetOffsets(out int magicNumOffset,
            out int descrOffset, out int ndOffset, out int dimensionsOffset,
            out int stridesOffset, out int flagsOffset, out int dataOffset,
            out int baseObjOffset, out int baseArrayOffset);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_DescrGetOffsets")]
        private static extern void DescrGetOffsets(out int magicNumOffset,
            out int kindOffset, out int typeOffset, out int byteorderOffset,
            out int flagsOffset, out int typenumOffset, out int elsizeOffset,
            out int alignmentOffset, out int namesOFfset, out int subarrayOffset,
            out int fieldsOffset, out int dtinfoOffset, out int fieldsOffsetOffset,
            out int fieldsDescrOffset, out int fieldsTitleOffset);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_IterGetOffsets")]
        private static extern void IterGetOffsets(out int sizeOffset, out int indexOffset);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl, EntryPoint = "NpyArrayAccess_MultiIterGetOffsets")]
        private static extern void MultiIterGetOffsets(out int numiterOffset, out int sizeOffset,
            out int indexOffset, out int ndOffset, out int dimensionsOffset, out int itersOffset);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_UFuncGetOffsets")]
        private static extern void UFuncGetOffsets(out int ninOffset,
            out int noutOffset, out int nargsOffset, out int coreEnabledOffset,
            out int identifyOffset, out int ntypesOffset, out int checkRetOffset,
            out int nameOffset, out int typesOffset, out int coreSigOffset);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_GetIndexInfo")]
        private static extern void GetIndexInfo(out int unionOffset, out int indexSize, out int maxDims);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_NewFromDescrThunk")]
        private static extern IntPtr NewFromDescrThunk(IntPtr descr, int nd, int flags,
            [In][MarshalAs(UnmanagedType.LPArray,SizeParamIndex=1)] long[] dims,
            [In][MarshalAs(UnmanagedType.LPArray,SizeParamIndex=1)] long[] strides, IntPtr data, IntPtr interfaceData);

        // Thread-safe.
        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl, EntryPoint = "NpyArrayAccess_DescrDestroyNames")]
        internal static extern void DescrDestroyNames(IntPtr p, int n);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArrayAccess_AddField(IntPtr fields, IntPtr names, int i,
            [MarshalAs(UnmanagedType.LPStr)]string name, IntPtr descr, int offset,
            [MarshalAs(UnmanagedType.LPStr)]string title);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArrayAccess_DescrNewVoid(IntPtr fields, IntPtr names, int elsize, int flags, int alignment);

        /// <summary>
        /// Allocates a new VOID descriptor and sets the subarray field as specified.
        /// </summary>
        /// <param name="baseDescr">Base descriptor for the subarray</param>
        /// <param name="ndim">Number of dimensions</param>
        /// <param name="dims">Array of size of each dimension</param>
        /// <returns>New descriptor object</returns>
        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArrayAccess_DescrNewSubarray(IntPtr baseDescr,
            int ndim, [In][MarshalAs(UnmanagedType.LPArray)]IntPtr[] dims);

        /// <summary>
        /// Replaces / sets the subarray field of an existing object.
        /// </summary>
        /// <param name="descr">Descriptor object to be modified</param>
        /// <param name="baseDescr">Base descriptor for the subaray</param>
        /// <param name="ndim">Number of dimensions</param>
        /// <param name="dims">Array of size of each dimension</param>
        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NpyArrayAccess_DescrReplaceSubarray(IntPtr descr, IntPtr baseDescr,
            int ndim, [In][MarshalAs(UnmanagedType.LPArray)]IntPtr[] dims);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NpyArrayAccess_DescrReplaceFields(IntPtr descr, IntPtr namesArr, IntPtr fieldsDict);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArrayAccess_GetBytes(IntPtr arr,
            [Out][MarshalAs(UnmanagedType.LPArray,SizeParamIndex=2)] byte[] bytes, long len, int order);

        // Thread-safe
        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr NpyArrayAccess_ToInterface(IntPtr arr);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NpyArrayAccess_ZeroFill(IntPtr arr, IntPtr offset);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArrayAccess_DupZeroElem(IntPtr arr);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArrayAccess_Fill(IntPtr arr);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static unsafe extern void NpyArrayAccess_CopySwapIn(IntPtr arr, long offset, void* data, int swap);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArrayAccess_ViewLike(IntPtr arr, IntPtr proto);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static unsafe extern void NpyArrayAccess_CopySwapOut(IntPtr arr, long offset, void* data, int swap);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static unsafe extern void NpyArrayAccess_CopySwapScalar(IntPtr dtype, void *dest, void* src, bool swap);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern int NpyArrayAccess_SetDateTimeInfo(IntPtr descr,
            [MarshalAs(UnmanagedType.LPStr)]string units, int num, int den, int events);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArrayAccess_InheritDescriptor(IntPtr type, IntPtr conv);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArrayAccess_GetBufferFormatString(IntPtr arr);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NpyArrayAccess_Free(IntPtr ptr);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArrayAccess_FromFile(string fileName, IntPtr dtype, int count, string sep);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern void NpyArrayAccess_SetNamesList(IntPtr dtype, string[] nameslist, int len);

        // Thread-safe
        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl, EntryPoint = "NpyArrayAccess_DictAllocIter")]
        internal static extern IntPtr NpyDict_AllocIter();

        // Thread-safe
        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl, EntryPoint="NpyArrayAccess_DictFreeIter")]
        internal static extern void NpyDict_FreeIter(IntPtr iter);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyUFuncAccess_UFuncFromPyFunc(int nin, int nout, String funcName, IntPtr pyThunk, IntPtr func);

        /// <summary>
        /// Accesses the next dictionary item, returning the key and value.  Thread-safe when operating across
        /// separate iterators; caller must ensure that one iterator is not access simultaneously from two
        /// different threads.
        /// </summary>
        /// <param name="dict">Pointer to the dictionary object</param>
        /// <param name="iter">Iterator structure</param>
        /// <param name="key">Next key</param>
        /// <param name="value">Next value</param>
        /// <returns>True if an element was returned, false at the end of the sequence</returns>
        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl, EntryPoint="NpyArrayAccess_DictNext")]
        internal static extern bool NpyDict_Next(IntPtr dict, IntPtr iter, out IntPtr key, out IntPtr value);

        // Thread-safe
        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl, EntryPoint = "NpyArrayAccess_FormatLongFloat")]
        internal static extern string FormatLongFloat(double v, int precision);

        #endregion


        #region Callbacks and native access

        /* This structure must match the NpyObject_HEAD structure in npy_object.h
         * exactly as it is used to determine the platform-specific offsets. The
         * offsets allow the C# code to access these fields directly. */
        [StructLayout(LayoutKind.Sequential)]
        internal struct NpyObject_HEAD {
            internal IntPtr nob_refcnt;
            internal IntPtr nob_type;
            internal IntPtr nob_interface;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct NpyInterface_WrapperFuncs {
            internal IntPtr array_new_wrapper;
            internal IntPtr iter_new_wrapper;
            internal IntPtr multi_iter_new_wrapper;
            internal IntPtr neighbor_iter_new_wrapper;
            internal IntPtr descr_new_from_type;
            internal IntPtr descr_new_from_wrapper;
            internal IntPtr ufunc_new_wrapper;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct NpyArrayOffsets {
            internal int off_magic_number;
            internal int off_descr;
            internal int off_nd;
            internal int off_dimensions;
            internal int off_strides;
            internal int off_flags;
            internal int off_data;
            internal int off_base_obj;
            internal int off_base_array;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct NpyArrayDescrOffsets
        {
            internal int off_magic_number;
            internal int off_kind;
            internal int off_type;
            internal int off_byteorder;
            internal int off_flags;
            internal int off_type_num;
            internal int off_elsize;
            internal int off_alignment;
            internal int off_names;
            internal int off_subarray;
            internal int off_fields;
            internal int off_dtinfo;

            /// <summary>
            /// Offset to the 'offset' field of the NpyArray_DescrField structure.
            /// </summary>
            internal int off_fields_offset;

            /// <summary>
            /// Offset to the 'descr' field of the NpyArray_DescrField structure.
            /// </summary>
            internal int off_fields_descr;

            /// <summary>
            /// Offset to the 'title' field of the NpyArray_DescrField structure.
            /// </summary>
            internal int off_fields_title;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct NpyArrayIterOffsets
        {
            internal int off_size;
            internal int off_index;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct NpyArrayMultiIterOffsets
        {
            internal int off_numiter;
            internal int off_size;
            internal int off_index;
            internal int off_nd;
            internal int off_dimensions;
            internal int off_iters;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct NpyArrayIndexInfo {
            internal int off_union;
            internal int sizeof_index;
            internal int max_dims;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct NpyUFuncOffsets
        {
            internal int off_nin;
            internal int off_nout;
            internal int off_nargs;
            internal int off_identify;
            internal int off_ntypes;
            internal int off_check_return;
            internal int off_name;
            internal int off_types;
            internal int off_core_signature;
            internal int off_core_enabled;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal class DateTimeInfo {
            internal NpyDefs.NPY_DATETIMEUNIT @base;
            internal int num;
            internal int den;
            internal int events;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal unsafe struct NpyArray_ArrayDescr {
            internal IntPtr @base;
            internal IntPtr shape_num_dims;
            internal IntPtr* shape_dims;
        }

        internal static readonly NpyArrayOffsets ArrayOffsets;
        internal static readonly NpyArrayDescrOffsets DescrOffsets;
        internal static readonly NpyArrayIterOffsets IterOffsets;
        internal static readonly NpyArrayMultiIterOffsets MultiIterOffsets;
        internal static readonly NpyArrayIndexInfo IndexInfo;
        internal static readonly NpyUFuncOffsets UFuncOffsets;

        internal static byte oppositeByteOrder;

        /// <summary>
        /// Used for synchronizing modifications to interface pointer.
        /// </summary>
        private static object interfaceSyncRoot = new Object();

        /// <summary>
        /// Offset to the interface pointer.
        /// </summary>
        internal static int Offset_InterfacePtr = (int)Marshal.OffsetOf(typeof(NpyObject_HEAD), "nob_interface");

        /// <summary>
        /// Offset to the reference count in the header structure.
        /// </summary>
        internal static int Offset_RefCount = (int)Marshal.OffsetOf(typeof(NpyObject_HEAD), "nob_refcnt");

        private static IntPtr lastArrayHandle = IntPtr.Zero;

        /// <summary>
        /// Given a pointer to a core (native) object, returns the managed wrapper.
        /// </summary>
        /// <param name="ptr">Address of native object</param>
        /// <returns>Managed wrapper object</returns>
        internal static TResult ToInterface<TResult>(IntPtr ptr) {
            if (ptr == IntPtr.Zero) {
                return default(TResult);
            }

            IntPtr wrapper = Marshal.ReadIntPtr(ptr, (int)Offset_InterfacePtr);
            if (wrapper == IntPtr.Zero) {
                // The wrapper object is dynamically created for some instances
                // so this call into native land triggers that magic.
                wrapper = NpyArrayAccess_ToInterface(ptr);
                if (wrapper == IntPtr.Zero) {
                    throw new IronPython.Runtime.Exceptions.RuntimeException(
                        String.Format("Managed wrapper for type '{0}' is NULL.", typeof(TResult).Name));
                }
            }
            return (TResult)GCHandleFromIntPtr(wrapper).Target;
        }


        /// <summary>
        /// Same as ToInterface but releases the core reference.
        /// </summary>
        /// <typeparam name="TResult">Type of the expected object</typeparam>
        /// <param name="ptr">Pointer to the core object</param>
        /// <returns>Wrapper instance corresponding to ptr</returns>
        internal static TResult DecrefToInterface<TResult>(IntPtr ptr) {
            CheckError();
            if (ptr == IntPtr.Zero) {
                return default(TResult);
            }
            TResult result = ToInterface<TResult>(ptr);
            Decref(ptr);
            return result;
        }


        /// <summary>
        /// Allocates a managed wrapper for the passed array object.
        /// </summary>
        /// <param name="coreArray">Pointer to the native array object</param>
        /// <param name="ensureArray">If true forces base array type, not subtype</param>
        /// <param name="customStrides">Not sure how this is used</param>
        /// <param name="interfaceData">Not used</param>
        /// <param name="interfaceRet">void ** for us to store the allocated wrapper</param>
        /// <returns>True on success, false on failure</returns>
        private static int ArrayNewWrapper(IntPtr coreArray, int ensureArray,
            int customStrides, IntPtr subtypePtr, IntPtr interfaceData,
            IntPtr interfaceRet) {
            int success = 1;     // Success

            try {
                PythonType subtype = null;
                object interfaceObj = null;
                ndarray wrapArray = null;

                if (interfaceData != IntPtr.Zero) {
                    interfaceObj = GCHandleFromIntPtr(interfaceData, true).Target;
                }

                if (interfaceObj is UseExistingWrapper) {
                    // Special case for UseExistingWrapper
                    UseExistingWrapper w = (UseExistingWrapper)interfaceObj;
                    wrapArray = (ndarray)w.Wrapper;
                    wrapArray.SetArray(coreArray);
                    subtype = DynamicHelpers.GetPythonType(wrapArray);
                } else {
                    // Determine the subtype. null means ndarray
                    if (ensureArray == 0) {
                        if (subtypePtr != IntPtr.Zero) {
                            subtype = (PythonType)GCHandleFromIntPtr(subtypePtr).Target;
                        } else if (interfaceObj != null) {
                            subtype = DynamicHelpers.GetPythonType(interfaceObj);
                        }
                    }
                    // Create the wrapper
                    if (subtype != null) {
                        CodeContext cntx = NpyUtil_Python.DefaultContext;
                        wrapArray = (ndarray)ObjectOps.__new__(cntx, subtype);
                        wrapArray.SetArray(coreArray);
                    } else {
                        wrapArray = new ndarray();
                        wrapArray.SetArray(coreArray);
                    }
                }

                // Call __array_finalize__ for subtypes
                if (subtype != null) {
                    CodeContext cntx = NpyUtil_Python.DefaultContext;
                    if (PythonOps.HasAttr(cntx, wrapArray, "__array_finalize__")) {
                        object func = PythonOps.ObjectGetAttribute(cntx, wrapArray, "__array_finalize__");
                        if (func != null) {
                            if (customStrides != 0) {
                                UpdateFlags(wrapArray, NpyDefs.NPY_UPDATE_ALL);
                            }
                            // TODO: Check for a Capsule
                            PythonCalls.Call(cntx, func, interfaceObj);
                        }
                    }
                }

                // Write the result
                IntPtr ret = GCHandle.ToIntPtr(AllocGCHandle(wrapArray));
                lastArrayHandle = ret;
                Marshal.WriteIntPtr(interfaceRet, ret);
                ndarray.IncreaseMemoryPressure(wrapArray);

                // TODO: Skipping subtype-specific initialization (ctors.c:718)
            } catch (InsufficientMemoryException) {
                Console.WriteLine("Insufficient memory while allocating array wrapper.");
                success = 0;
            } catch (Exception e) {
                Console.WriteLine("Exception while allocating array wrapper: {0}", e.Message);
                success = 0;
            }
            return success;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int del_ArrayNewWrapper(IntPtr coreArray, int ensureArray,
            int customStrides, IntPtr subtypePtr, IntPtr interfaceData,
            IntPtr interfaceRet);


        /// <summary>
        /// Constructs a new managed wrapper for an interator object. This function
        /// is thread-safe.
        /// </summary>
        /// <param name="coreIter">Pointer to the native instance</param>
        /// <param name="interfaceRet">Location to store GCHandle to the wrapper</param>
        /// <returns>1 on success, 0 on error</returns>
        private static int IterNewWrapper(IntPtr coreIter, ref IntPtr interfaceRet) {
            int success = 1;

            try {
                lock (interfaceSyncRoot) {
                    // Check interfaceRet inside the lock because some interface
                    // wrappers are dynamically created and two threads could
                    // trigger these event at the same time.
                    if (interfaceRet == IntPtr.Zero) {
                        flatiter wrapIter = new flatiter(coreIter);
                        interfaceRet = GCHandle.ToIntPtr(AllocGCHandle(wrapIter));
                    }
                }
            } catch (InsufficientMemoryException) {
                Console.WriteLine("Insufficient memory while allocating iterator wrapper.");
                success = 0;
            } catch (Exception) {
                Console.WriteLine("Exception while allocating iterator wrapper.");
                success = 0;
            }
            return success;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int del_IterNewWrapper(IntPtr coreIter, ref IntPtr interfaceRet);



        /// <summary>
        /// Constructs a new managed wrapper for a multi-iterator.  This funtion
        /// is thread safe.
        /// </summary>
        /// <param name="coreIter">Pointer to the native instance</param>
        /// <param name="interfaceRet">Location to store the wrapper handle</param>
        /// <returns></returns>
        private static int MultiIterNewWrapper(IntPtr coreIter, ref IntPtr interfaceRet) {
            int success = 1;
            try {
                lock (interfaceSyncRoot) {
                    // Check interfaceRet inside the lock because some interface
                    // wrappers are dynamically created and two threads could
                    // trigger these event at the same time.
                    if (interfaceRet == IntPtr.Zero) {
                        broadcast wrapIter = broadcast.BeingCreated;
                        interfaceRet = GCHandle.ToIntPtr(AllocGCHandle(wrapIter));
                    }
                }
            } catch (InsufficientMemoryException) {
                Console.WriteLine("Insufficient memory while allocating iterator wrapper.");
                success = 0;
            } catch (Exception) {
                Console.WriteLine("Exception while allocating iterator wrapper.");
                success = 0;
            }
            return success;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int del_MultiIterNewWrapper(IntPtr coreIter, ref IntPtr interfaceRet);


        /// <summary>
        /// Allocated a managed wrapper for one of the core, native types
        /// </summary>
        /// <param name="type">Type code (not used)</param>
        /// <param name="descr">Pointer to the native descriptor object</param>
        /// <param name="interfaceRet">void** for returning allocated wrapper</param>
        /// <returns>1 on success, 0 on error</returns>
        private static int DescrNewFromType(int type, IntPtr descr, IntPtr interfaceRet) {
            int success = 1;

            try {
                // TODO: Descriptor typeobj not handled. Do we need to?

                dtype wrap = new dtype(descr, type);
                Marshal.WriteIntPtr(interfaceRet,
                    GCHandle.ToIntPtr(AllocGCHandle(wrap)));
            } catch (InsufficientMemoryException) {
                Console.WriteLine("Insufficient memory while allocating descriptor wrapper.");
                success = 0;
            } catch (Exception) {
                Console.WriteLine("Exception while allocating descriptor wrapper.");
                success = 0;
            }
            return success;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int del_DescrNewFromType(int type, IntPtr descr, IntPtr interfaceRet);




        /// <summary>
        /// Allocated a managed wrapper for a user defined type
        /// </summary>
        /// <param name="baseTmp">Pointer to the base descriptor (not used)</param>
        /// <param name="descr">Pointer to the native descriptor object</param>
        /// <param name="interfaceRet">void** for returning allocated wrapper</param>
        /// <returns>1 on success, 0 on error</returns>
        private static int DescrNewFromWrapper(IntPtr baseTmp, IntPtr descr, IntPtr interfaceRet) {
            int success = 1;

            try {
                // TODO: Descriptor typeobj not handled. Do we need to?

                dtype wrap = new dtype(descr);
                Marshal.WriteIntPtr(interfaceRet,
                    GCHandle.ToIntPtr(AllocGCHandle(wrap)));
            } catch (InsufficientMemoryException) {
                Console.WriteLine("Insufficient memory while allocating descriptor wrapper.");
                success = 0;
            } catch (Exception) {
                Console.WriteLine("Exception while allocating descriptor wrapper.");
                success = 0;
            }
            return success;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int del_DescrNewFromWrapper(IntPtr baseTmp, IntPtr descr, IntPtr interfaceRet);



        /// <summary>
        /// Allocated a managed wrapper for a UFunc object.
        /// </summary>
        /// <param name="baseTmp">Pointer to the base object</param>
        /// <param name="interfaceRet">void** for returning allocated wrapper</param>
        /// <returns>1 on success, 0 on error</returns>
        private static void UFuncNewWrapper(IntPtr basePtr, IntPtr interfaceRet) {
            try {
                ufunc wrap = new ufunc(basePtr);
                Marshal.WriteIntPtr(interfaceRet,
                    GCHandle.ToIntPtr(AllocGCHandle(wrap)));
            } catch (InsufficientMemoryException) {
                Console.WriteLine("Insufficient memory while allocating ufunc wrapper.");
            } catch (Exception) {
                Console.WriteLine("Exception while allocating ufunc wrapper.");
            }
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void del_UFuncNewWrapper(IntPtr basePtr, IntPtr interfaceRet);


        /// <summary>
        /// Accepts a pointer to an existing GCHandle object and allocates
        /// an additional GCHandle to the same object.  This effectively
        /// does an "incref" on the object.  Used in cases where an array
        /// of objects is being copied.
        ///
        /// Usually wrapPtr is NULL meaning that we just allocate a new
        /// handle and return it.  If wrapPtr != NULL then we assign the
        /// new handle to it as well.  Must be done atomically.
        /// </summary>
        /// <param name="ptr">Pointer to GCHandle of object to reference</param>
        /// <param name="nobInterfacePtr">Address of the nob_interface field (not value of it)</param>
        /// <returns>New handle to the input object</returns>
        private static IntPtr IncrefCallback(IntPtr ptr, IntPtr nobInterfacePtr) {
            if (ptr == IntPtr.Zero) {
                return IntPtr.Zero;
            }

            IntPtr newWrapRef = IntPtr.Zero;
            lock (interfaceSyncRoot) {
                GCHandle oldWrapRef = GCHandleFromIntPtr(ptr, true);
                object wrapperObj = oldWrapRef.Target;
                newWrapRef = GCHandle.ToIntPtr(AllocGCHandle(wrapperObj));
                if (nobInterfacePtr != IntPtr.Zero) {
                    // Replace the contents of nobInterfacePtr with the new reference.
                    Marshal.WriteIntPtr(nobInterfacePtr, newWrapRef);
                    FreeGCHandle(oldWrapRef);
                }
            }
            return newWrapRef;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr del_Incref(IntPtr ptr, IntPtr wrapPtr);

        /// <summary>
        /// Releases the reference to the given interface object.  Note that
        /// this is not a decref but actual freeingo of this handle, it can
        /// not be used again.
        /// </summary>
        /// <param name="ptr">Interface object to 'decref'</param>
        private static void DecrefCallback(IntPtr ptr, IntPtr nobInterfacePtr) {
            lock (interfaceSyncRoot) {
                if (nobInterfacePtr != IntPtr.Zero) {
                    // Deferencing the interface wrapper.  We can't just null the
                    // wrapPtr because we have to have maintain the link so we
                    // allocate a weak reference instead.
                    GCHandle oldWrapRef = GCHandleFromIntPtr(ptr);
                    Object wrapperObj = oldWrapRef.Target;
                    Marshal.WriteIntPtr(nobInterfacePtr,
                        GCHandle.ToIntPtr(AllocGCHandle(wrapperObj, GCHandleType.Weak)));
                    FreeGCHandle(oldWrapRef);
                } else {
                    if (ptr != IntPtr.Zero) {
                        FreeGCHandle(GCHandleFromIntPtr(ptr));
                    }
                }
            }
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void del_Decref(IntPtr ptr, IntPtr wrapPtr);


        internal static IntPtr GetRefcnt(IntPtr obj) {
            // NOTE: I'm relying on the refcnt being first.
            return Marshal.ReadIntPtr(obj);
        }



        #region Error handling

        /// <summary>
        /// Error type, determines which type of exception to throw.
        /// DANGER! Must be kept in sync with npy_api.h
        /// </summary>
        private enum NpyExc_Type {
            MemoryError = 0,
            IOError,
            ValueError,
            TypeError,
            IndexError,
            RuntimeError,
            AttributeError,
            ComplexWarning,
            NotImplementedError,
            FloatingPointError,
            NoError
        }


        /// <summary>
        /// Indicates the most recent error code or NpyExc_NoError if nothing pending
        /// </summary>
        [ThreadStatic]
        private static NpyExc_Type ErrorCode = NpyExc_Type.NoError;

        /// <summary>
        /// Stores the most recent error message per-thread
        /// </summary>
        [ThreadStatic]
        private static string ErrorMessage = null;

        public static void CheckError() {
            if (ErrorCode != NpyExc_Type.NoError) {
                NpyExc_Type errTmp = ErrorCode;
                String msgTmp = ErrorMessage;

                ErrorCode = NpyExc_Type.NoError;
                ErrorMessage = null;

                switch (errTmp) {
                    case NpyExc_Type.MemoryError:
                        throw new InsufficientMemoryException(msgTmp);
                    case NpyExc_Type.IOError:
                        throw new System.IO.IOException(msgTmp);
                    case NpyExc_Type.ValueError:
                        throw new ArgumentException(msgTmp);
                    case NpyExc_Type.IndexError:
                        throw new IndexOutOfRangeException(msgTmp);
                    case NpyExc_Type.RuntimeError:
                        throw new IronPython.Runtime.Exceptions.RuntimeException(msgTmp);
                    case NpyExc_Type.AttributeError:
                        throw new MissingMemberException(msgTmp);
                    case NpyExc_Type.ComplexWarning:
                        PythonOps.Warn(NpyUtil_Python.DefaultContext, ComplexWarning, msgTmp);
                        break;
                    case NpyExc_Type.TypeError:
                        throw new IronPython.Runtime.Exceptions.TypeErrorException(msgTmp);
                    case NpyExc_Type.NotImplementedError:
                        throw new NotImplementedException(msgTmp);
                    case NpyExc_Type.FloatingPointError:
                        throw new IronPython.Runtime.Exceptions.FloatingPointException(msgTmp);
                    default:
                        Console.WriteLine("Unhandled exception type {0} in CheckError.", errTmp);
                        throw new IronPython.Runtime.Exceptions.RuntimeException(msgTmp);
                }
            }
        }

        private static PythonType complexWarning;

        internal static PythonType ComplexWarning {
            get {
                if (complexWarning == null) {
                    CodeContext cntx = NpyUtil_Python.DefaultContext;
                    PythonModule core = (PythonModule)PythonOps.ImportBottom(cntx, "numpy.core", 0);
                    object tmp;
                    if (PythonOps.ModuleTryGetMember(cntx, core, "ComplexWarning", out tmp)) {
                        complexWarning = (PythonType)tmp;
                    }
                }
                return complexWarning;
            }
        }

        private static void SetError(NpyExc_Type exceptType, string msg) {
            if (exceptType == NpyExc_Type.ComplexWarning) {
                Console.WriteLine("Warning: {0}", msg);
            } else {
                ErrorCode = exceptType;
                ErrorMessage = msg;
            }
        }


        /// <summary>
        /// Called by NpyErr_SetMessage in the native world when something bad happens
        /// </summary>
        /// <param name="exceptType">Type of exception to be thrown</param>
        /// <param name="bStr">Message string</param>
        unsafe private static void SetErrorCallback(int exceptType, sbyte* bStr) {
            if (exceptType < 0 || exceptType >= (int)NpyExc_Type.NoError) {
                Console.WriteLine("Internal error: invalid exception type {0}, likely ErrorType and npyexc_type (npy_api.h) are out of sync.",
                    exceptType);
            }
            SetError((NpyExc_Type)exceptType, new string(bStr));
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        unsafe public delegate void del_SetErrorCallback(int exceptType, sbyte* msg);


        /// <summary>
        /// Called by native side to check to see if an error occurred
        /// </summary>
        /// <returns>1 if an error is pending, 0 if not</returns>
        private static int ErrorOccurredCallback() {
            return (ErrorCode != NpyExc_Type.NoError) ? 1 : 0;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int del_ErrorOccurredCallback();


        private static void ClearErrorCallback() {
            ErrorCode = NpyExc_Type.NoError;
            ErrorMessage = null;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void del_ClearErrorCallback();

        private static unsafe void GetErrorState(int* bufsizep, int* errmaskp, IntPtr* errobjp) {
            // deref any existing obj
            if (*errobjp != IntPtr.Zero) {
                FreeGCHandle(GCHandleFromIntPtr(*errobjp));
                *errobjp = IntPtr.Zero;
            }
            var info = umath.errorInfo;
            if (info == null) {
                *bufsizep = NpyDefs.NPY_BUFSIZE;
                *errmaskp = NpyDefs.NPY_UFUNC_ERR_DEFAULT;
                *errobjp = IntPtr.Zero;
            } else {
                umath.ErrorInfo vInfo = (umath.ErrorInfo)info;
                *bufsizep = vInfo.bufsize;
                *errmaskp = vInfo.errmask;
                if (vInfo.errobj != null) {
                    GCHandle h = AllocGCHandle(vInfo.errobj);
                    *errobjp = GCHandle.ToIntPtr(h);
                }
            }
        }

        private static unsafe void ErrorHandler(sbyte* name, int errormask, IntPtr errobj, int retstatus, int* first) {
            try {
                object obj;
                if (errobj != IntPtr.Zero) {
                    obj = GCHandleFromIntPtr(errobj).Target;
                } else {
                    obj = null;
                }
                string sName = new string(name);
                NpyDefs.NPY_UFUNC_ERR method;
                if ((retstatus & (int)NpyDefs.NPY_UFUNC_FPE.DIVIDEBYZERO) != 0) {
                    bool bfirst = (*first != 0);
                    int handle = (errormask & (int)NpyDefs.NPY_UFUNC_MASK.DIVIDEBYZERO);
                    method = (NpyDefs.NPY_UFUNC_ERR)(handle >> (int)NpyDefs.NPY_UFUNC_SHIFT.DIVIDEBYZERO);
                    umath.ErrorHandler(sName, method, obj, "divide by zero", retstatus, ref bfirst);
                    *first = bfirst ? 1 : 0;
                }
                if ((retstatus & (int)NpyDefs.NPY_UFUNC_FPE.OVERFLOW) != 0) {
                    bool bfirst = (*first != 0);
                    int handle = (errormask & (int)NpyDefs.NPY_UFUNC_MASK.OVERFLOW);
                    method = (NpyDefs.NPY_UFUNC_ERR)(handle >> (int)NpyDefs.NPY_UFUNC_SHIFT.OVERFLOW);
                    umath.ErrorHandler(sName, method, obj, "overflow", retstatus, ref bfirst);
                    *first = bfirst ? 1 : 0;
                }
                if ((retstatus & (int)NpyDefs.NPY_UFUNC_FPE.UNDERFLOW) != 0) {
                    bool bfirst = (*first != 0);
                    int handle = (errormask & (int)NpyDefs.NPY_UFUNC_MASK.UNDERFLOW);
                    method = (NpyDefs.NPY_UFUNC_ERR)(handle >> (int)NpyDefs.NPY_UFUNC_SHIFT.UNDERFLOW);
                    umath.ErrorHandler(sName, method, obj, "underflow", retstatus, ref bfirst);
                    *first = bfirst ? 1 : 0;
                }
                if ((retstatus & (int)NpyDefs.NPY_UFUNC_FPE.INVALID) != 0) {
                    bool bfirst = (*first != 0);
                    int handle = (errormask & (int)NpyDefs.NPY_UFUNC_MASK.INVALID);
                    method = (NpyDefs.NPY_UFUNC_ERR)(handle >> (int)NpyDefs.NPY_UFUNC_SHIFT.INVALID);
                    umath.ErrorHandler(sName, method, obj, "invalid", retstatus, ref bfirst);
                    *first = bfirst ? 1 : 0;
                }
            } catch (Exception ex) {
                SetError(NpyExc_Type.FloatingPointError, ex.Message);
            }
        }

        #endregion

        #region Thread handling
        // CPython uses a threading model that is single threaded unless the global interpreter lock
        // is explicitly released. While .NET supports true threading, the ndarray core has not been
        // completely checked to makes sure that it is re-entrant  much less modify each function to
        // perform fine-grained locking on individual objects.  Thus we artificially lock IronPython
        // down and force ndarray accesses to be single threaded for now.

        /// <summary>
        /// Equivalent to the CPython GIL.
        /// </summary>
        private static readonly object GlobalIterpLock = new object();

        /// <summary>
        /// Releases the GIL so other threads can run.
        /// </summary>
        /// <returns>Return value is unused</returns>
        private static IntPtr EnableThreads() {
            Monitor.Exit(GlobalIterpLock);
            return IntPtr.Zero;
        }
        private delegate IntPtr del_EnableThreads();

        /// <summary>
        /// Re-acquires the GIL forcing code to stop until other threads have exited the ndarray core.
        /// </summary>
        /// <param name="unused">Unused</param>
        private static void DisableThreads(IntPtr unused) {
            Monitor.Enter(GlobalIterpLock);
        }
        private delegate void del_DisableThreads(IntPtr unused);

        #endregion

        //
        // These variables hold a reference to the delegates passed into the core.
        // Failure to hold these references causes the callback function to disappear
        // at some point when the GC runs.
        //
        private static readonly NpyInterface_WrapperFuncs wrapFuncs;

        private static readonly del_ArrayNewWrapper ArrayNewWrapDelegate =
            new del_ArrayNewWrapper(ArrayNewWrapper);
        private static readonly del_IterNewWrapper IterNewWrapperDelegate =
            new del_IterNewWrapper(IterNewWrapper);
        private static readonly del_MultiIterNewWrapper MultiIterNewWrapperDelegate =
            new del_MultiIterNewWrapper(MultiIterNewWrapper);
        private static readonly del_DescrNewFromType DescrNewFromTypeDelegate =
            new del_DescrNewFromType(DescrNewFromType);
        private static readonly del_DescrNewFromWrapper DescrNewFromWrapperDelegate =
            new del_DescrNewFromWrapper(DescrNewFromWrapper);
        private static readonly del_UFuncNewWrapper UFuncNewWrapperDelegate =
            new del_UFuncNewWrapper(UFuncNewWrapper);

        private static readonly del_Incref IncrefCallbackDelegate =
            new del_Incref(IncrefCallback);
        private static readonly del_Decref DecrefCallbackDelegate =
            new del_Decref(DecrefCallback);
        unsafe private static readonly del_SetErrorCallback SetErrorCallbackDelegate =
            new del_SetErrorCallback(SetErrorCallback);
        private static readonly del_ErrorOccurredCallback ErrorOccurredCallbackDelegate =
            new del_ErrorOccurredCallback(ErrorOccurredCallback);
        private static readonly del_ClearErrorCallback ClearErrorCallbackDelegate =
            new del_ClearErrorCallback(ClearErrorCallback);
        private static readonly del_EnableThreads EnableThreadsDelegate =
            new del_EnableThreads(EnableThreads);
        private static readonly del_DisableThreads DisableThreadsDelegate =
            new del_DisableThreads(DisableThreads);

        private static unsafe readonly del_GetErrorState GetErrorStateDelegate = new del_GetErrorState(GetErrorState);
        private static unsafe readonly del_ErrorHandler ErrorHandlerDelegate = new del_ErrorHandler(ErrorHandler);


        /// <summary>
        /// The native type code that matches up to a 32-bit int.
        /// </summary>
        internal static readonly NpyDefs.NPY_TYPES TypeOf_Int32;

        /// <summary>
        /// Native type code that matches up to a 64-bit int.
        /// </summary>
        internal static readonly NpyDefs.NPY_TYPES TypeOf_Int64;

        /// <summary>
        /// Native type code that matches up to a 32-bit unsigned int.
        /// </summary>
        internal static readonly NpyDefs.NPY_TYPES TypeOf_UInt32;

        /// <summary>
        /// Native type code that matches up to a 64-bit unsigned int.
        /// </summary>
        internal static readonly NpyDefs.NPY_TYPES TypeOf_UInt64;

        /// <summary>
        /// Size of element in integer arrays, in bytes.
        /// </summary>
        internal static readonly int Native_SizeOfInt;

        /// <summary>
        /// Size of element in long arrays, in bytes.
        /// </summary>
        internal static readonly int Native_SizeOfLong;

        /// <summary>
        /// Size of element in long long arrays, in bytes.
        /// </summary>
        internal static readonly int Native_SizeOfLongLong;

        /// <summary>
        /// Size fo element in long double arrays, in bytes.
        /// </summary>
        internal static readonly int Native_SizeOfLongDouble;


        /// <summary>
        /// Initializes the core library with necessary callbacks on load.
        /// </summary>
        static NpyCoreApi() {
            try {
                // Check the native byte ordering (make sure it matches what .NET uses) and
                // figure out the mapping between types that vary in size in the core and
                // fixed-size .NET types.
                int intSize, longSize, longLongSize, longDoubleSize;
                oppositeByteOrder = GetNativeTypeInfo(out intSize, out longSize, out longLongSize,
                                                      out longDoubleSize);

                Native_SizeOfInt = intSize;
                Native_SizeOfLong = longSize;
                Native_SizeOfLongLong = longLongSize;
                Native_SizeOfLongDouble = longDoubleSize;

                // Important: keep this consistent with NpyArray_TypestrConvert in npy_conversion_utils.c
                if (intSize == 4 && longSize == 4 && longLongSize == 8) {
                    TypeOf_Int32 = NpyDefs.NPY_TYPES.NPY_LONG;
                    TypeOf_Int64 = NpyDefs.NPY_TYPES.NPY_LONGLONG;
                    TypeOf_UInt32 = NpyDefs.NPY_TYPES.NPY_ULONG;
                    TypeOf_UInt64 = NpyDefs.NPY_TYPES.NPY_ULONGLONG;
                } else if (intSize == 4 && longSize == 8 && longLongSize == 8) {
                    TypeOf_Int32 = NpyDefs.NPY_TYPES.NPY_INT;
                    TypeOf_Int64 = NpyDefs.NPY_TYPES.NPY_LONG;
                    TypeOf_UInt32 = NpyDefs.NPY_TYPES.NPY_UINT;
                    TypeOf_UInt64 = NpyDefs.NPY_TYPES.NPY_ULONG;
                } else {
                    throw new NotImplementedException(
                        String.Format("Unimplemented combination of native type sizes: int = {0}b, long = {1}b, longlong = {2}b",
                                      intSize, longSize, longLongSize));
                }


                wrapFuncs = new NpyInterface_WrapperFuncs();

                wrapFuncs.array_new_wrapper =
                    Marshal.GetFunctionPointerForDelegate(ArrayNewWrapDelegate);
                wrapFuncs.iter_new_wrapper =
                    Marshal.GetFunctionPointerForDelegate(IterNewWrapperDelegate);
                wrapFuncs.multi_iter_new_wrapper =
                    Marshal.GetFunctionPointerForDelegate(MultiIterNewWrapperDelegate);
                wrapFuncs.neighbor_iter_new_wrapper = IntPtr.Zero;
                wrapFuncs.descr_new_from_type =
                    Marshal.GetFunctionPointerForDelegate(DescrNewFromTypeDelegate);
                wrapFuncs.descr_new_from_wrapper =
                    Marshal.GetFunctionPointerForDelegate(DescrNewFromWrapperDelegate);
                wrapFuncs.ufunc_new_wrapper =
                    Marshal.GetFunctionPointerForDelegate(UFuncNewWrapperDelegate);

                int s = Marshal.SizeOf(wrapFuncs.descr_new_from_type);

                NumericOps.NpyArray_FunctionDefs funcDefs = NumericOps.GetFunctionDefs();
                IntPtr funcDefsHandle = IntPtr.Zero;
                IntPtr wrapHandle = IntPtr.Zero;
                try {
                    funcDefsHandle = Marshal.AllocHGlobal(Marshal.SizeOf(funcDefs));
                    Marshal.StructureToPtr(funcDefs, funcDefsHandle, true);
                    wrapHandle = Marshal.AllocHGlobal(Marshal.SizeOf(wrapFuncs));
                    Marshal.StructureToPtr(wrapFuncs, wrapHandle, true);

                    npy_initlib(funcDefsHandle, wrapHandle,
                        Marshal.GetFunctionPointerForDelegate(SetErrorCallbackDelegate),
                        Marshal.GetFunctionPointerForDelegate(ErrorOccurredCallbackDelegate),
                        Marshal.GetFunctionPointerForDelegate(ClearErrorCallbackDelegate),
                        Marshal.GetFunctionPointerForDelegate(NumericOps.ComparePriorityDelegate),
                        Marshal.GetFunctionPointerForDelegate(IncrefCallbackDelegate),
                        Marshal.GetFunctionPointerForDelegate(DecrefCallbackDelegate),
                        IntPtr.Zero, IntPtr.Zero);
                    // for now we run full threaded, no safety net.
                    //Marshal.GetFunctionPointerForDelegate(EnableThreadsDelegate),
                    //Marshal.GetFunctionPointerForDelegate(DisableThreadsDelegate));
                } catch (Exception e) {
                    Console.WriteLine("Failed during initialization: {0}", e);
                } finally {
                    Marshal.FreeHGlobal(funcDefsHandle);
                    Marshal.FreeHGlobal(wrapHandle);
                }

                // Initialize the offsets to each structure type for fast access
                // TODO: Not a great way to do this, but for now it's
                // a convenient way to get hard field offsets from the core.
                ArrayGetOffsets(out ArrayOffsets.off_magic_number,
                                out ArrayOffsets.off_descr,
                                out ArrayOffsets.off_nd,
                                out ArrayOffsets.off_dimensions,
                                out ArrayOffsets.off_strides,
                                out ArrayOffsets.off_flags,
                                out ArrayOffsets.off_data,
                                out ArrayOffsets.off_base_obj,
                                out ArrayOffsets.off_base_array);

                DescrGetOffsets(out DescrOffsets.off_magic_number,
                                out DescrOffsets.off_kind,
                                out DescrOffsets.off_type,
                                out DescrOffsets.off_byteorder,
                                out DescrOffsets.off_flags,
                                out DescrOffsets.off_type_num,
                                out DescrOffsets.off_elsize,
                                out DescrOffsets.off_alignment,
                                out DescrOffsets.off_names,
                                out DescrOffsets.off_subarray,
                                out DescrOffsets.off_fields,
                                out DescrOffsets.off_dtinfo,
                                out DescrOffsets.off_fields_offset,
                                out DescrOffsets.off_fields_descr,
                                out DescrOffsets.off_fields_title);

                IterGetOffsets(out IterOffsets.off_size,
                               out IterOffsets.off_index);

                MultiIterGetOffsets(out MultiIterOffsets.off_numiter,
                                    out MultiIterOffsets.off_size,
                                    out MultiIterOffsets.off_index,
                                    out MultiIterOffsets.off_nd,
                                    out MultiIterOffsets.off_dimensions,
                                    out MultiIterOffsets.off_iters);

                GetIndexInfo(out IndexInfo.off_union, out IndexInfo.sizeof_index, out IndexInfo.max_dims);

                UFuncGetOffsets(out UFuncOffsets.off_nin, out UFuncOffsets.off_nout,
                    out UFuncOffsets.off_nargs, out UFuncOffsets.off_core_enabled,
                    out UFuncOffsets.off_identify, out UFuncOffsets.off_ntypes,
                    out UFuncOffsets.off_check_return, out UFuncOffsets.off_name,
                    out UFuncOffsets.off_types, out UFuncOffsets.off_core_signature);

                NpyUFunc_SetFpErrFuncs(GetErrorStateDelegate, ErrorHandlerDelegate);

                // Causes the sort functions to be registered with the type descriptor objects.
                NpyArray_InitSortModule();
            } catch (Exception e) {
                // Report any details that we can here because IronPython only reports 
                // that the static type initializer failed.
                Console.WriteLine("Failed while initializing NpyCoreApi: {0}:{1}", e.GetType().Name, e.Message);
                Console.WriteLine("NumpyDotNet stack trace:\n{0}", e.StackTrace);
                throw e;
            }
        }
        #endregion


        #region Memory verification

        private const bool CheckMemoryAccesses = true;

        /// <summary>
        /// Set of all currently allocated GCHandles and the type of handle.
        /// </summary>
        private static readonly Dictionary<IntPtr, GCHandleType> AllocatedHandles = new Dictionary<IntPtr, GCHandleType>();

        /// <summary>
        /// Set of freed GC handles that we should not be accessing.
        /// </summary>
        private static readonly HashSet<IntPtr> FreedHandles = new HashSet<IntPtr>();

        /// <summary>
        /// Allocates a GCHandle for a given object. If CheckMemoryAccesses is false,
        /// this is inlined into the normal GCHandle call.  If not, it performs the
        /// access checking.
        /// </summary>
        /// <param name="o">Object to get a handle to</param>
        /// <param name="type">Handle type, default is normal</param>
        /// <returns>GCHandle instance</returns>
        internal static GCHandle AllocGCHandle(Object o, GCHandleType type=GCHandleType.Normal) {
            GCHandle h = GCHandle.Alloc(o, type);
            if (CheckMemoryAccesses) {
                lock (AllocatedHandles) {
                    IntPtr p = GCHandle.ToIntPtr(h);
                    if (AllocatedHandles.ContainsKey(p)) {
                        throw new AccessViolationException(
                            String.Format("Internal error: detected duplicate allocation of GCHandle. Probably a bookkeeping error. Handle is {0}.",
                            p));
                    }
                    if (FreedHandles.Contains(p)) {
                        FreedHandles.Remove(p);
                    }
                    AllocatedHandles.Add(p, type);
                }
            }
            return h;
        }

        /// <summary>
        /// Verifies that a GCHandle is known and good prior to using it.  If
        /// CheckMemoryAccesses is false, this is a no-op and goes away.
        /// </summary>
        /// <param name="h">Handle to verify</param>
        internal static GCHandle GCHandleFromIntPtr(IntPtr p, bool weakOk=false) {
            if (CheckMemoryAccesses) {
                lock (AllocatedHandles) {
                    GCHandleType handleType;
                    if (FreedHandles.Contains(p)) {
                        throw new AccessViolationException(
                            String.Format("Internal error: accessing already freed GCHandle {0}.", p));
                    }
                    if (!AllocatedHandles.TryGetValue(p, out handleType)) {
                        throw new AccessViolationException(
                            String.Format("Internal error: attempt to access unknown GCHandle {0}.", p));
                    } // else if (handleType == GCHandleType.Weak && !weakOk) {
                    //    throw new AccessViolationException(
                    //        String.Format("Internal error: invalid attempt to access weak reference {0}.", p));
                    //}
                }
            }
            return GCHandle.FromIntPtr(p);
        }

        /// <summary>
        /// Releases a GCHandle instance for an object.  If CheckMemoryAccesses is
        /// false this is inlined to the GCHandle.Free() method.  Otherwise it verifies
        /// that the handle is legit.
        /// </summary>
        /// <param name="h">GCHandle to release</param>
        internal static void FreeGCHandle(GCHandle h) {
            if (CheckMemoryAccesses) {
                lock (AllocatedHandles) {
                    IntPtr p = GCHandle.ToIntPtr(h);
                    if (FreedHandles.Contains(p)) {
                        throw new AccessViolationException(
                            String.Format("Internal error: freeing already freed GCHandle {0}.", p));
                    }
                    if (!AllocatedHandles.ContainsKey(p)) {
                        throw new AccessViolationException(
                            String.Format("Internal error: freeing unknown GCHandle {0}.", p));
                    }
                    AllocatedHandles.Remove(p);
                    FreedHandles.Add(p);
                }
            }
            h.Free();
        }

        #endregion
    }
}
