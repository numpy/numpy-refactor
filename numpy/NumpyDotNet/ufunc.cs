using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using IronPython.Runtime;
using IronPython.Runtime.Operations;
using IronPython.Runtime.Types;
using IronPython.Modules;
using Microsoft.Scripting;
using NumpyDotNet;

namespace NumpyDotNet
{
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


        ~ufunc() {
            Dispose(false);
        }

        internal IntPtr UFunc {
            get { return core; }
        }

        public object Call(CodeContext cntx, [ParamDictionary] IAttributesCollection kwargs, params object[] args) {
            object extobj = null;
            NpyDefs.NPY_TYPES[] sig = null;
            if (kwargs != null) {
                foreach (var pair in kwargs) {
                    string skey = (string)pair.Key;
                    if (skey.Length >= 6 && skey.Substring(0, 6) == "extobj") {
                        extobj = pair.Value;
                    } else if (skey.Length >= 3 && skey.Substring(0, 3) == "sig") {
                        sig = ConvertSig(cntx, pair.Value);
                    } else {
                        throw new ArgumentTypeException(String.Format("'{0}' is an invalid keywork argument to {1}", skey, this));
                    }
                }
            }

            if (extobj != null) {
                throw new NotImplementedException("extobj not supported yet.");
            }

            ndarray[] arrays = ConvertArgs(args);
            NpyCoreApi.GenericFunction(cntx, this, arrays, sig, (ctx, ufunc, ars, ag) => ufunc.PrepareOutputs(ctx, ars, ag), args);

            if (nout == 1) {
                return arrays[nin];
            } else {
                return new PythonTuple(arrays.Skip(nin).ToArray());
            }
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
        internal enum ReduceOp
        {
            NPY_UFUNC_REDUCE = 0,
            NPY_UFUNC_ACCUMULATE = 1,
            NPY_UFUNC_REDUCEAT = 2,
            NPY_UFUNC_OUTER = 3
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

        class WithPrepare
        {
            public object arg;
            public object prepare;
        }

        internal void PrepareOutputs(CodeContext cntx, ndarray[] arrays, object[] args) {
            object[] wraparr = FindArrayPrepare(cntx, args);
            object[] wraparg = null;
            for (int i = 0; i < nout; i++) {
                int j = nin + i;
                object wrap = wraparr[i];
                if (wrap != null) {
                    if (wraparg == null) {
                        wraparg = new object[] { this, new PythonTuple(args), i };
                    } else {
                        wraparg[2] = i;
                    }
                    ndarray wrapped = (PythonOps.CallWithContext(cntx, wrap, arrays[j], new PythonTuple(wraparg)) as ndarray);
                    if (wrapped == null) {
                        throw new ArgumentTypeException("__array_prepare__ must returns an ndarray or subclass thereof.");
                    }
                    arrays[j] = wrapped;
                }
            }
        }

        internal object[] FindArrayPrepare(CodeContext cntx, object[] args) {


            var with_prepare = args.Take(nin)
                .Where(x => x is ndarray && x.GetType() != typeof(ndarray) && PythonOps.HasAttr(cntx, x, "__array_prepare__"))
                .Select(x => new WithPrepare { arg = x, prepare = PythonOps.ObjectGetAttribute(cntx, x, "__array_prepare__") })
                .Where(x=>PythonOps.IsCallable(cntx, x.prepare)).ToList();

            // Find the one with the highest priority
            object wrap = null;
            if (with_prepare.Count == 1) {
                wrap = with_prepare.First().prepare;
            } else if (with_prepare.Count > 1) {
                wrap = with_prepare.OrderByDescending(x => NumericOps.GetPriority(cntx, x.arg, 1.0)).First().prepare;
            }

            // Use the output __array_prepare__ if it has one, otherwise wrap
            object[] result = Enumerable.Repeat(wrap, nout).ToArray();
            int i = 0;
            foreach (var output in args.Skip(nin).Take(nout)) {
                result[i] = wrap;
                if (PythonOps.HasAttr(cntx, output, "__array__prepare__")) {
                    object prepare = PythonOps.ObjectGetAttribute(cntx, output, "__array_prepare__");
                    if (PythonOps.IsCallable(cntx, prepare)) {
                        result[i] = prepare;
                    }
                }
                i++;
            }
            return result;
        }

        /// <summary>
        /// Converts a sig argument into an array of types.
        /// </summary>
        /// <param name="sig"></param>
        /// <returns></returns>
        private NpyDefs.NPY_TYPES[] ConvertSig(CodeContext cntx, object sig) {
            string ssig = (sig as string);
            if (sig is PythonTuple) {
                PythonTuple s = (PythonTuple)sig;
                int n = s.Count;
                if (n != 1 && n != nargs) {
                    throw new ArgumentException(
                        String.Format("a type-tuple must be specified of length 1 or {0} for {1}", nargs, this));
                }
                return s.Select(x => NpyDescr.DescrConverter(cntx.LanguageContext, x).TypeNum).ToArray();
            } else if (ssig != null && IsStringType(ssig)) {
                return ssig.Where(x => (x != '-' && x != '>'))
                    .Select(x => NpyCoreApi.DescrFromType((NpyDefs.NPY_TYPES)x).TypeNum).ToArray();
            } else {
                return new NpyDefs.NPY_TYPES[] { NpyDescr.DescrConverter(cntx.LanguageContext, sig).TypeNum };
            }
        }

        private bool IsStringType(string sig) {
            int pos = sig.IndexOf("->");
            if (pos == -1) {
                return false;
            } else {
                int n = sig.Length - 2;
                if (pos != nin || n - 2 != nout) {
                    throw new ArgumentException(
                        String.Format("a type-string for {0}, requires {1} typecode(s) before and {2} after the -> sign",
                                      this, nin, nout));
                }
                return true;
            }
        }

        /// <summary>
        /// Converts args to arrays and return an array nargs long containing the arrays and
        /// nulls.
        /// </summary>
        /// <param name="args"></param>
        /// <returns></returns>
        private ndarray[] ConvertArgs(object[] args) {
            if (args.Length < nin || args.Length > nargs) {
                throw new ArgumentException("invalid number of arguments");
            }
            ndarray[] result = new ndarray[nargs];
            for (int i = 0; i < args.Length; i++) {
                // TODO: Add check for scalars
                object arg = args[i];
                object context = null;
                object[] contextArray = null;
                if (!(arg is ndarray)) {
                    if (contextArray == null) {
                        contextArray = new object[] { this, new PythonTuple(args), i };
                    } else {
                        contextArray[2] = i;
                    }
                    context = new PythonTuple(contextArray);
                }
                result[i] = NpyArray.FromAny(arg, context: context);
            }
            return result;
        }
    }
}

