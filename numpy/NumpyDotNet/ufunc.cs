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

namespace NumpyDotNet {
    public class ufunc : Wrapper
    {
         private static String[] ufuncArgNames = { "extobj", "sig" };

         internal ufunc(IntPtr corePtr) {
            core = corePtr;

            // The core object comes with a reference so we need to set the interface
            // pointer and then discard the core reference, leaving just this instance
            // as the sole reference to it.
            IntPtr offset = Marshal.OffsetOf(typeof(NpyCoreApi.NpyObject_HEAD), "nob_interface");
            Marshal.WriteIntPtr(corePtr, (int)offset,
                GCHandle.ToIntPtr(GCHandle.Alloc(this, GCHandleType.Weak)));
            NpyCoreApi.Decref(corePtr);
        }


        ~ufunc()
        {
            Dispose(false);
        }

        internal IntPtr UFunc {
            get { return core; }
        }

        public object Call(Object a) {
            if (nin == 1) {
                return NpyCoreApi.GenericUnaryOp(NpyArray.FromAny(a), this);
            }
            throw new ArgumentException("Insufficient number of arguments.");
        }

        public object Call(Object a, Object b) {
            if (nin == 1) {
                return NpyCoreApi.GenericUnaryOp(NpyArray.FromAny(a), this, (ndarray)b);
            } else if (nin == 2) {
                return NpyCoreApi.GenericBinaryOp(NpyArray.FromAny(a), NpyArray.FromAny(b), this);
            }
            throw new ArgumentException("Insufficient number of arguments.");
        }

        public object Call(Object a, Object b, Object c) {
            if (nin == 2) {
                return NpyCoreApi.GenericBinaryOp(NpyArray.FromAny(a),
                    NpyArray.FromAny(b), this, (ndarray)c);
            }
            throw new ArgumentException("Insufficient number of arguments.");
        }


        /// <summary>
        /// Named arguments for reduce & accumulate.
        /// </summary>
        private static string[] ReduceArgNames = new String[] { 
            "array", "axis", "dtype", "out" };


        public object reduce(CodeContext cntx, [ParamDictionary] IAttributesCollection kwargs, params Object[] posArgs) {
            object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ReduceArgNames, kwargs);

            if (args[0] == null) {
                throw new ArgumentException("Insufficient number of arguments.");
            }

            // TODO: Not passing context to FromAny - see ufunc_object.c:960
            ndarray arr = NpyArray.FromAny(args[0]);
            int axis = NpyUtil_ArgProcessing.IntConverter(args[1]);
            dtype type = NpyDescr.DescrConverter(cntx.LanguageContext, args[2]);
            ndarray arrOut = (args[3] != null) ? NpyArray.FromAny(args[3]) : null;

            return GenericReduce(arr, null, axis, type, arrOut, ReduceOp.NPY_UFUNC_REDUCE);
        }

        public object accumulate(CodeContext cntx, [ParamDictionary] IAttributesCollection kwargs, params Object[] posArgs) {
            object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ReduceArgNames, kwargs);

            if (args[0] == null) {
                throw new ArgumentException("Insufficient number of arguments.");
            }

            // TODO: Not passing context to FromAny - see ufunc_object.c:960
            ndarray arr = NpyArray.FromAny(args[0]);
            int axis = NpyUtil_ArgProcessing.IntConverter(args[1]);
            dtype type = NpyDescr.DescrConverter(cntx.LanguageContext, args[2]);
            ndarray arrOut = (args[3] != null) ? NpyArray.FromAny(args[3]) : null;

            return GenericReduce(arr, null, axis, type, arrOut, ReduceOp.NPY_UFUNC_ACCUMULATE);
        }



        private static string[] ReduceAtArgNames = new String[] { 
            "array", "indices", "axis", "dtype", "out" };

        public object reduceat(CodeContext cntx, [ParamDictionary] IAttributesCollection kwargs, params Object[] posArgs) {
            object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ReduceAtArgNames, kwargs);

            if (args[0] == null || args[1] == null) {
                throw new ArgumentException("Insufficient number of arguments.");
            }

            // TODO: Not passing context to FromAny - see ufunc_object.c:960
            ndarray arr = NpyArray.FromAny(args[0]);
            ndarray indices = NpyArray.FromAny(args[1],
                NpyCoreApi.DescrFromType(NpyDefs.NPY_INTP),
                1, 1, NpyDefs.NPY_CARRAY, null);
            int axis = NpyUtil_ArgProcessing.IntConverter(args[2]);
            dtype type = NpyDescr.DescrConverter(cntx.LanguageContext, args[3]);
            ndarray arrOut = (args[4] != null) ? NpyArray.FromAny(args[4]) : null;

            return GenericReduce(arr, null, axis, type, arrOut, ReduceOp.NPY_UFUNC_REDUCEAT);
        }


        #region Python interface

        public string __repr__() {
            return String.Format("<ufunc '{0}'>", __name__());
        }

        public string __str__() {
            // TODO: Unimplemented
            return "str";
        }

        public int nin {
            get {
                CheckValid();
                return Marshal.ReadInt32(core, NpyCoreApi.UFuncOffsets.off_nin);
            }
        }

        public int nout {
            get {
                CheckValid();
                return Marshal.ReadInt32(core, NpyCoreApi.UFuncOffsets.off_nout); 
            }
        }

        public int nargs {
            get {
                CheckValid();
                return Marshal.ReadInt32(core, NpyCoreApi.UFuncOffsets.off_nargs);
            }
        }

        public int ntypes {
            get {
                CheckValid();
                return Marshal.ReadInt32(core, NpyCoreApi.UFuncOffsets.off_ntypes);
            }
        }

        // TODO: Implement 'types'
        public override string ToString() {
            return __name__();
        }

        public string __name__() {
            CheckValid();
            IntPtr strPtr = Marshal.ReadIntPtr(core, NpyCoreApi.UFuncOffsets.off_name);
            return (strPtr != IntPtr.Zero) ? Marshal.PtrToStringAnsi(strPtr) : null;
        }

        // TODO: Implement 'identity'

        public string signature() {
            CheckValid();
            IntPtr strPtr = Marshal.ReadIntPtr(core, NpyCoreApi.UFuncOffsets.off_core_signature);
            return (strPtr != IntPtr.Zero) ? Marshal.PtrToStringAnsi(strPtr) : null;
        }


        #endregion

        public static ufunc GetFunction(string name) {
            return umath.GetUFunc(name);
        }


        /// <summary>
        /// Simply checks to verify that the object was correctly initialized and hasn't
        /// already been disposed before we go accessing native memory.
        /// </summary>
        private void CheckValid() {
            if (core == IntPtr.Zero)
                throw new InvalidOperationException("UFunc object is invalid or already disposed.");
        }


        /// <summary>
        /// Reduce/accumulate operations.  These values must stay in sync with
        /// the values NPY_UFUNC_REDUCE, NPY_UFUNC_ACCUMULATE, etc defined in
        /// npy_ufunc_object.h in the core.
        /// </summary>
        internal enum ReduceOp { 
            NPY_UFUNC_REDUCE=0, 
            NPY_UFUNC_ACCUMULATE=1, 
            NPY_UFUNC_REDUCEAT=2,
            NPY_UFUNC_OUTER=3
        };


        /// <summary>
        /// Performs a generic reduce or accumulate operation on an input array.
        /// A reduce operation reduces the number of dimensions of the input array
        /// by one where accumulate does not.  Accumulate stores in incremental
        /// accumulated values in the extra dimension.
        /// </summary>
        /// <param name="arr">Input array</param>
        /// <param name="indices">Used only for reduceat</param>
        /// <param name="axis">Axis to reduce</param>
        /// <param name="otype">Output type of the array</param>
        /// <param name="outArr">Optional output array</param>
        /// <param name="operation">Reduce/accumulate operation to perform</param>
        /// <returns>Resulting array, either outArr or a new array</returns>
        private Object GenericReduce(ndarray arr, ndarray indices, int axis,
            dtype otype, ndarray outArr, ReduceOp operation) {

            if (signature() != null) {
                throw new IronPython.Runtime.Exceptions.RuntimeException(
                    "Reduction is not defined on ufunc's with signatures");
            }
            if (nin != 2) {
                throw new ArgumentException("Reduce/accumulate only supported for binary functions");
            }
            if (nout != 1) {
                throw new ArgumentException("Reduce/accumulate only supported for functions returning a single value");
            }

            if (arr.ndim == 0) {
                throw new ArgumentTypeException("Cannot reduce/accumulate a scalar");
            }

            if (arr.IsFlexible || (otype != null && NpyDefs.IsFlexible(otype.TypeNum))) {
                throw new ArgumentTypeException("Cannot perform reduce/accumulate with flexible type");
            }

            return NpyCoreApi.GenericReduction(this, arr, indices,
                outArr, axis, otype, operation);
        }
    }
}
