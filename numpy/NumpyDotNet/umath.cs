using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Text;
using IronPython.Runtime;
using IronPython.Modules;
using IronPython.Runtime.Exceptions;
using IronPython.Runtime.Types;
using IronPython.Runtime.Operations;
using Microsoft.Scripting;

namespace NumpyDotNet
{
    public static class umath
    {

        public const double PINF = double.PositiveInfinity;
        public const double NINF = double.NegativeInfinity;
        public const double PZERO = 0.0;
        public const double NZERO = -0.0;
        public const double NAN = double.NaN;
        
        public const int FLOATING_POINT_SUPPORT = 1;

        public const int ERR_IGNORE = (int)NpyDefs.NPY_UFUNC_ERR.IGNORE;
        public const int ERR_WARN = (int)NpyDefs.NPY_UFUNC_ERR.WARN;
        public const int ERR_CALL = (int)NpyDefs.NPY_UFUNC_ERR.CALL;
        public const int ERR_RAISE = (int)NpyDefs.NPY_UFUNC_ERR.RAISE;
        public const int ERR_PRINT = (int)NpyDefs.NPY_UFUNC_ERR.PRINT;
        public const int ERR_LOG = (int)NpyDefs.NPY_UFUNC_ERR.LOG;
        public const int ERR_DEFAULT = NpyDefs.NPY_UFUNC_ERR_DEFAULT;
        public const int ERR_DEFAULT2 = NpyDefs.NPY_UFUNC_ERR_DEFAULT2;

        public const int SHIFT_DIVIDEBYZERO = (int)NpyDefs.NPY_UFUNC_SHIFT.DIVIDEBYZERO;
        public const int SHIFT_OVERFLOW = (int)NpyDefs.NPY_UFUNC_SHIFT.OVERFLOW;
        public const int SHIFT_UNDERFLOW = (int)NpyDefs.NPY_UFUNC_SHIFT.UNDERFLOW;
        public const int SHIFT_INVALID = (int)NpyDefs.NPY_UFUNC_SHIFT.INVALID;

        public const int FPE_DIVIDEBYZERO = (int)NpyDefs.NPY_UFUNC_FPE.DIVIDEBYZERO;
        public const int FPE_OVERFLOW = (int)NpyDefs.NPY_UFUNC_FPE.OVERFLOW;
        public const int FPE_UNDERFLOW = (int)NpyDefs.NPY_UFUNC_FPE.UNDERFLOW;
        public const int FPE_INVALID = (int)NpyDefs.NPY_UFUNC_FPE.INVALID;

        public const int UFUNC_BUFSIZE_DEFAULT = NpyDefs.NPY_BUFSIZE;

        internal struct ErrorInfo
        {
            internal int bufsize;
            internal int errmask;
            internal object errobj;
        }

        [ThreadStatic]
        internal static ErrorInfo? errorInfo;

        public static List geterrobj() {
            if (errorInfo == null) {
                List result = new List();
                result.append(NpyDefs.NPY_BUFSIZE);
                result.append(NpyDefs.NPY_UFUNC_ERR_DEFAULT);
                result.append(null);
                return result;
            } else {
                List result = new List();
                ErrorInfo info = (ErrorInfo)errorInfo;
                result.append(info.bufsize);
                result.append(info.errmask);
                result.append(info.errobj);
                return result;
            }
        }

        public static void seterrobj(List obj) {
            if (obj.Count != 3) {
                throw new ArgumentException("Error object must be a list of length 3");
            }
            int bufsize = NpyUtil_Python.ConvertToInt(obj[0]);
            if (bufsize < NpyDefs.NPY_MIN_BUFSIZE ||
                bufsize > NpyDefs.NPY_MAX_BUFSIZE ||
                (bufsize % 16 != 0)) {
                    throw new ArgumentException(String.Format("buffer size ({0}) is not in range ({1} - {2}) or not a multiple of 16",
                        bufsize, NpyDefs.NPY_MIN_BUFSIZE, NpyDefs.NPY_MAX_BUFSIZE));
            }
            int errmask = NpyUtil_Python.ConvertToInt(obj[1]);
            object errobj = obj[2];
            if (errobj != null && !NpyUtil_Python.IsCallable(errobj)) {
                object write = PythonOps.ObjectGetAttribute(NpyUtil_Python.DefaultContext, errobj, "write");
                if (write == null || !NpyUtil_Python.IsCallable(write)) {
                    throw new ArgumentException("python object must be callable or have a callable write method");
                }
            }
            errorInfo = new ErrorInfo() { bufsize = bufsize, errmask = errmask, errobj = errobj };
        }



        /// <summary>
        /// Implements a loop over n input arrays and calls a user-provided Python function for each
        /// element.  func must be a pointer to a GCHandle referencing the Python function.
        /// </summary>
        /// <param name="args">Array of pointers to object arrays. args[n] corresponds to each 
        /// input array followed by each output array pointer</param>
        /// <param name="dimensions">Dimension[0] is the number of elements to loop over</param>
        /// <param name="steps">Offset to next element, each steps[n] is offset for args[n] array</param>
        /// <param name="nin">Number of input arguments to function</param>
        /// <param name="nout">Number of return values from function</param>
        /// <param name="func">GCHandle referencing Python function to call</param>
        private static unsafe void PyUFunc_Om_On(IntPtr* args, IntPtr* dimensions, IntPtr* steps, int nin, int nout, IntPtr func) {
            long n = (long)dimensions[0];
            object f = NpyCoreApi.GCHandleFromIntPtr(func).Target;
            int ntot = nin + nout;
            IntPtr[] ptrs = new IntPtr[ntot];

            // Initialize the pointers, we will advance these by 'step' each iteration.
            for (int j = 0; j < ntot; j++) {
                ptrs[j] = args[j];
            }

            object[] arglist = new object[nin];
            for (long i = 0; i < n; i++) {
                // First nin values are inputs to the function.
                for (int j = 0; j < nin; j++) {
                    IntPtr* v = (IntPtr*)ptrs[j];
                    arglist[j] = (*v != IntPtr.Zero) ? 
                                    NpyCoreApi.GCHandleFromIntPtr(*v).Target : null;
                }
                object result = PythonCalls.Call(f, arglist);
                if (result == null) return;     // Failed.

                // Result of the function is either a tuple of nout values or a single value.
                PythonTuple tup = result as PythonTuple;
                if (tup != null) {
                    if (nout != tup.Count) {
                        throw new ArgumentException(String.Format("Invalid return tuple size {0}, expected {1}",
                            tup.Count, nout));
                    }
                    for (int j = 0; j < nout; j++) {
                        // If we have a GCHandle already in this slot, overwrite the target. Otherwise
                        // we alloc a new GCHandle
                        IntPtr* elem = (IntPtr*)(ptrs[j + nin]);
                        if (*elem != IntPtr.Zero) {
                            GCHandle h = NpyCoreApi.GCHandleFromIntPtr(*elem);
                            h.Target = tup[j];
                        } else {
                            *elem = GCHandle.ToIntPtr(NpyCoreApi.AllocGCHandle(tup[j]));
                        }
                    }
                } else {
                    IntPtr *elem = (IntPtr *)ptrs[nin];
                    if (*elem != IntPtr.Zero) {
                        GCHandle h = NpyCoreApi.GCHandleFromIntPtr(*elem);
                        h.Target = result;
                    } else {
                        *elem = GCHandle.ToIntPtr(NpyCoreApi.AllocGCHandle(result));
                    }
                }

                for (int j = 0; j < ntot; j++) {
                    ptrs[j] = IntPtr.Add(ptrs[j], steps[j].ToInt32());
                }
            }
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate void del_PyUFunc_Om_On(IntPtr* args, IntPtr* dimensions, IntPtr* steps, int nin, int nout, IntPtr func);
        private static unsafe readonly del_PyUFunc_Om_On PyUFunc_Om_On_Delegate = new del_PyUFunc_Om_On(PyUFunc_Om_On);

        public static object frompyfunc(CodeContext cntx, object func, int nin, int nout) {
            if (!IronPython.Runtime.Operations.PythonOps.IsCallable(cntx, func)) {
                throw new ArgumentTypeException("function must be callback");
            }

            string funcName = PythonOps.HasAttr(cntx, func, "__name__") ?
                (string)PythonOps.GetBoundAttr(cntx, func, "__name__") : "?";
            GCHandle funcHandle = NpyCoreApi.AllocGCHandle(func);
            IntPtr ufunc = NpyCoreApi.UFuncFromPyFunc(nin, nout, funcName, 
                Marshal.GetFunctionPointerForDelegate(PyUFunc_Om_On_Delegate), GCHandle.ToIntPtr(funcHandle));

            // Allocate a managed wrapper for the ufunc, then set the native object's interface pointer
            // to this object.  Once that is done, we can decref the native object and it won't go away
            // because it's owned by the managed object.
            ufunc self = new ufunc(ufunc);
            Marshal.WriteIntPtr(ufunc, NpyCoreApi.Offset_InterfacePtr,
                GCHandle.ToIntPtr(NpyCoreApi.AllocGCHandle(self)));
            NpyCoreApi.Decref(ufunc);
            return self;
        }


        /// <summary>
        /// Map of function names to all defined ufunc objects.
        /// </summary>
        private static PythonDictionary ModuleDict;


        /// <summary>
        /// Returns the ufunc matching a named function or null if not defined.
        /// </summary>
        /// <param name="name">Function name</param>
        /// <returns>ufunc implementing the function or null</returns>
        internal static ufunc GetUFunc(string name) {
            object result = null;
            if (!ModuleDict.TryGetValue(name, out result)) {
                result = null;
            }
            return (ufunc)result;
        }

        public static void Init() {
            __init__(NpyUtil_Python.DefaultContext);
        }

        public static void __init__(CodeContext cntx) {
            // Initialize the ufunc instances.
            NumericOps.InitUFuncOps(cntx);

            ExternFuncs funcs = new ExternFuncs();

            // External loop functions - these are provided by the native code,
            // not passed in.
            funcs.cmp_equal = Marshal.GetFunctionPointerForDelegate(NumericOps.Compare_Equal);
            funcs.cmp_not_equal = Marshal.GetFunctionPointerForDelegate(NumericOps.Compare_NotEqual);
            funcs.cmp_greater = Marshal.GetFunctionPointerForDelegate(NumericOps.Compare_Greater);
            funcs.cmp_greater_equal = Marshal.GetFunctionPointerForDelegate(NumericOps.Compare_GreaterEqual);
            funcs.cmp_less = Marshal.GetFunctionPointerForDelegate(NumericOps.Compare_Less);
            funcs.cmp_less_equal = Marshal.GetFunctionPointerForDelegate(NumericOps.Compare_LessEqual);
            funcs.op_sign = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_Sign);

            // Generic arithmatic functions that operate on objects and are provided by
            // the managed layer.  These typically call into IronPython to perform the
            // operation.
            funcs.absolute = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_Absolute);
            funcs.add = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_Add);
            funcs.subtract = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_Subtract);
            funcs.multiply = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_Multiply);
            funcs.divide = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_Divide);
            funcs.trueDivide = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_TrueDivide);
            funcs.floorDivide = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_FloorDivide);
            funcs.invert = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_Invert);
            funcs.negative = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_Negate);
            funcs.remainder = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_Remainder);
            funcs.square = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_Square);
            funcs.power = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_Power);
            funcs.min = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_Min);
            funcs.max = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_Max);
            funcs.reciprocal = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_Reciprocal);
            funcs.and = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_And);
            funcs.or = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_Or);
            funcs.xor = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_Xor);
            funcs.lshift = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_LShift);
            funcs.rshift = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_RShift);
            funcs.get_one = Marshal.GetFunctionPointerForDelegate(NumericOps.Op_GetOne);

            funcs.sentinel = NpyDefs.NPY_VALID_MAGIC;

            ModuleDict = cntx.ModuleContext.Module.Get__dict__();

            GCHandle dictHandle = NpyCoreApi.AllocGCHandle(ModuleDict);
            IntPtr funcsHandle = IntPtr.Zero;
            try {
                funcsHandle = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(ExternFuncs)));
                Marshal.StructureToPtr(funcs, funcsHandle, true);
                NpyCoreApi.NpyUFuncAccess_Init(GCHandle.ToIntPtr(dictHandle),
                    funcsHandle,
                    Marshal.GetFunctionPointerForDelegate(MethodCallDelegate),
                    Marshal.GetFunctionPointerForDelegate(AddToDictDelegate));
                RegisterCoreUFuncs(ModuleDict);
            } finally {
                NpyCoreApi.FreeGCHandle(dictHandle);
                Marshal.FreeHGlobal(funcsHandle);
            }

        }


        private static void RegisterCoreUFuncs(PythonDictionary funcDict) {
            Action<NpyDefs.NpyArray_Ops, string> set = (op, opStr) => {
                object f;
                if (funcDict.TryGetValue(opStr, out f)) {
                    NpyCoreApi.SetNumericOp((int)op, (ufunc)f);
                }
            };

            set(NpyDefs.NpyArray_Ops.npy_op_add, "add");
            set(NpyDefs.NpyArray_Ops.npy_op_subtract, "subtract");
            set(NpyDefs.NpyArray_Ops.npy_op_multiply, "multiply");
            set(NpyDefs.NpyArray_Ops.npy_op_divide, "divide");
            set(NpyDefs.NpyArray_Ops.npy_op_remainder, "remainder");
            set(NpyDefs.NpyArray_Ops.npy_op_power, "power");
            set(NpyDefs.NpyArray_Ops.npy_op_square, "square");
            set(NpyDefs.NpyArray_Ops.npy_op_reciprocal, "reciprocal");
            set(NpyDefs.NpyArray_Ops.npy_op_ones_like, "ones_like");
            set(NpyDefs.NpyArray_Ops.npy_op_sqrt, "sqrt");
            set(NpyDefs.NpyArray_Ops.npy_op_negative, "negative");
            set(NpyDefs.NpyArray_Ops.npy_op_absolute, "absolute");
            set(NpyDefs.NpyArray_Ops.npy_op_invert, "invert");
            set(NpyDefs.NpyArray_Ops.npy_op_left_shift, "left_shift");
            set(NpyDefs.NpyArray_Ops.npy_op_right_shift, "right_shift");
            set(NpyDefs.NpyArray_Ops.npy_op_bitwise_and, "bitwise_and");
            set(NpyDefs.NpyArray_Ops.npy_op_bitwise_or, "bitwise_or");
            set(NpyDefs.NpyArray_Ops.npy_op_bitwise_xor, "bitwise_xor");
            set(NpyDefs.NpyArray_Ops.npy_op_less, "less");
            set(NpyDefs.NpyArray_Ops.npy_op_less_equal, "less_equal");
            set(NpyDefs.NpyArray_Ops.npy_op_equal, "equal");
            set(NpyDefs.NpyArray_Ops.npy_op_not_equal, "not_equal");
            set(NpyDefs.NpyArray_Ops.npy_op_greater, "greater");
            set(NpyDefs.NpyArray_Ops.npy_op_greater_equal, "greater_equal");
            set(NpyDefs.NpyArray_Ops.npy_op_floor_divide, "floor_divide");
            set(NpyDefs.NpyArray_Ops.npy_op_true_divide, "true_divide");
            set(NpyDefs.NpyArray_Ops.npy_op_logical_or, "logical_or");
            set(NpyDefs.NpyArray_Ops.npy_op_logical_and, "logical_and");
            set(NpyDefs.NpyArray_Ops.npy_op_floor, "floor");
            set(NpyDefs.NpyArray_Ops.npy_op_ceil, "ceil");
            set(NpyDefs.NpyArray_Ops.npy_op_maximum, "maximum");
            set(NpyDefs.NpyArray_Ops.npy_op_minimum, "minimum");
            set(NpyDefs.NpyArray_Ops.npy_op_rint, "rint");
            set(NpyDefs.NpyArray_Ops.npy_op_conjugate, "conjugate");

            object s;
            if (funcDict.TryGetValue("conjugate", out s)) {
                funcDict["conj"] = s;
            }
            if (funcDict.TryGetValue("remainder", out s)) {
                funcDict["mod"] = s;
            }
        }

        /// <summary>
        /// Structure for passing loop and arithmetic functions into the native
        /// world.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct ExternFuncs
        {
            // Comparison funtions, returns 1 on true, 0 on false, -1 on error.
            internal IntPtr cmp_equal;
            internal IntPtr cmp_not_equal;
            internal IntPtr cmp_greater;
            internal IntPtr cmp_greater_equal;
            internal IntPtr cmp_less;
            internal IntPtr cmp_less_equal;
            internal IntPtr op_sign;

            // Generic arithmatic functions that operate on objects and are provided by
            // the managed layer.  These typically call into IronPython to perform the
            // operation.
            internal IntPtr absolute;
            internal IntPtr add, subtract, multiply, divide;
            internal IntPtr trueDivide, floorDivide;
            internal IntPtr invert, negative;
            internal IntPtr remainder, square, power;
            internal IntPtr min, max, reciprocal;
            internal IntPtr and, or, xor;
            internal IntPtr lshift, rshift, get_one;

            internal int sentinel;   // Used to verify matching structure sizes, must be last.
        }



        /// <summary>
        /// Callback function to allow native code to add ufuncs to the function
        /// dictionary
        /// </summary>
        /// <param name="dictHandle">GCHandle for a dictionary</param>
        /// <param name="bStr">IntPtr to a null-terminated char string</param>
        /// <param name="ufuncHandle">GCHandle to ufunc</param>
        private unsafe static void AddToDict(IntPtr dictHandle, sbyte* bStr, IntPtr ufuncHandle) {
            PythonDictionary dict =
                (PythonDictionary)NpyCoreApi.GCHandleFromIntPtr(dictHandle).Target;
            String funcStr = new String(bStr);
            ufunc f = (ufunc)NpyCoreApi.GCHandleFromIntPtr(ufuncHandle).Target;
            dict.Add(funcStr, f);
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public unsafe delegate void del_AddToDict(IntPtr dict, sbyte* str, IntPtr ufunc);

        private unsafe static readonly del_AddToDict AddToDictDelegate =
            new del_AddToDict(AddToDict);
        private unsafe static readonly NumericOps.del_MethodCall MethodCallDelegate =
            new NumericOps.del_MethodCall(NumericOps.MethodCall);

        #region error handling

        private static PythonType PyExc_RuntimeWarning = PythonExceptions.RuntimeWarning;

        internal static void ErrorHandler(string name, NpyDefs.NPY_UFUNC_ERR method, object errobj, string errtype, int retstatus, ref bool first) {
            string msg;
            object func;

            switch (method) {
                case NpyDefs.NPY_UFUNC_ERR.WARN:
                    msg = String.Format("{0} encountered in {1}", errtype, name);
                    NpyUtil_Python.Warn(PyExc_RuntimeWarning, msg);
                    break;
                case NpyDefs.NPY_UFUNC_ERR.RAISE:
                    msg = String.Format("{0} encountered in {1}", errtype, name);
                    throw new FloatingPointException(msg);
                case NpyDefs.NPY_UFUNC_ERR.CALL:
                    func = errobj;
                    if (func == null) {
                        msg = String.Format("python callback specified for {0} (in {1}) but no function found", errtype, name);
                        throw new ArgumentException(msg);
                    }
                    PythonCalls.Call(NpyUtil_Python.DefaultContext, func, errtype, retstatus);
                    break;
                case NpyDefs.NPY_UFUNC_ERR.PRINT:
                    if (first) {
                        Console.Error.WriteLine("Warning: {0} encountered in {1}", errtype, name);
                        first = false;
                    }
                    break;
                case NpyDefs.NPY_UFUNC_ERR.LOG:
                    func = errobj;
                    if (func == null) {
                        msg = String.Format("log specified for {0} (in {1}) but no function found", errtype, name);
                        throw new ArgumentException(msg);
                    }
                    msg = String.Format("Warning: {0} encountered in {1}\n", errtype, name);
                    PythonCalls.Call(NpyUtil_Python.DefaultContext, func, "write", "s", msg);
                    break;
            }
        }

        #endregion
    }
}
