using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Text;
using IronPython.Runtime;
using IronPython.Modules;
using Microsoft.Scripting;

namespace NumpyDotNet
{
    public static class umath
    {

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


        public static void __init__(CodeContext cntx) {
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
            funcs.trueDivide = IntPtr.Zero; // TODO: True divide not implemented
            funcs.floorDivide = IntPtr.Zero;  // TODO: floor divide not implemented
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

            GCHandle dictHandle = GCHandle.Alloc(ModuleDict);
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
                dictHandle.Free();
                Marshal.FreeHGlobal(funcsHandle);
            }



        }


        private static void RegisterCoreUFuncs(PythonDictionary funcDict) {
            Action<NpyDefs.NpyArray_Ops, string> set = (op, opStr) => {
                object f;
                if (funcDict.TryGetValue(opStr, out f)) {
                    NpyCoreApi.NpyArray_SetNumericOp((int)op, ((ufunc)f).UFunc);
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
                (PythonDictionary)GCHandle.FromIntPtr(dictHandle).Target;
            String funcStr = new String(bStr);
            ufunc f = (ufunc)GCHandle.FromIntPtr(ufuncHandle).Target;
            dict.Add(funcStr, f);
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public unsafe delegate void del_AddToDict(IntPtr dict, sbyte* str, IntPtr ufunc);

        private unsafe static readonly del_AddToDict AddToDictDelegate =
            new del_AddToDict(AddToDict);
        private unsafe static readonly NumericOps.del_MethodCall MethodCallDelegate =
            new NumericOps.del_MethodCall(NumericOps.MethodCall);

    }
}
