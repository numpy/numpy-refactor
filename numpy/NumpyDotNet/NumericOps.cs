using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Text;
using IronPython.Runtime;
using IronPython.Modules;
using Microsoft.Scripting.Runtime;
using Microsoft.Scripting.Utils;

namespace NumpyDotNet {


    /// <summary>
    /// Records the type-specific get/set items for each descriptor type.
    /// </summary>
    public class ArrFuncs {
        internal Func<long, ndarray, Object> GetItem { get; set; }
        internal Action<Object, long, ndarray> SetItem { get; set; }
    }



    /// <summary>
    /// Collection of getitem/setitem functions and operations on object types.
    /// These are mostly used as callbacks from the core and operate on native
    /// memory.
    /// </summary>
    internal static class NumericOps {
        internal static ArrFuncs[] arrFuncs = new ArrFuncs[(int)NpyDefs.NPY_TYPES.NPY_NTYPES];


        /// <summary>
        /// Initializes the type-specific functions for each native type.
        /// </summary>
        static NumericOps() {
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_BOOL] =
                new ArrFuncs() { GetItem = NumericOps.getitemBool, SetItem = NumericOps.setitemBool };
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_BYTE] =
                new ArrFuncs() { GetItem = NumericOps.getitemByte, SetItem = NumericOps.setitemByte };
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_UBYTE] =
                new ArrFuncs() { GetItem = NumericOps.getitemByte, SetItem = NumericOps.setitemByte };
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_SHORT] =
                new ArrFuncs() { GetItem = NumericOps.getitemShort, SetItem = NumericOps.setitemShort };
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_USHORT] =
                new ArrFuncs() { GetItem = NumericOps.getitemUShort, SetItem = NumericOps.setitemUShort };
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_INT] =
                new ArrFuncs() { GetItem = NumericOps.getitemInt32, SetItem = NumericOps.setitemInt32 };
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_UINT] =
                new ArrFuncs() { GetItem = NumericOps.getitemUInt32, SetItem = NumericOps.setitemUInt32 };
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_LONG] =
                new ArrFuncs() { GetItem = NumericOps.getitemLong, SetItem = NumericOps.setitemLong };
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_ULONG] =
                new ArrFuncs() { GetItem = NumericOps.getitemULong, SetItem = NumericOps.setitemULong };
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_LONGLONG] = null;
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_ULONGLONG] = null;
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_FLOAT] =
                new ArrFuncs() { GetItem = NumericOps.getitemFloat, SetItem = NumericOps.setitemFloat };
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_DOUBLE] =
                new ArrFuncs() { GetItem = NumericOps.getitemDouble, SetItem = NumericOps.setitemDouble };
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_LONGDOUBLE] = null;
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_CFLOAT] = null;
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_CDOUBLE] =
                new ArrFuncs() { GetItem = NumericOps.getitemCDouble, SetItem = NumericOps.setitemCDouble };
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_CLONGDOUBLE] = null;
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_DATETIME] = null;
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_TIMEDELTA] = null;
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_OBJECT] = null;
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_STRING] = null;
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_UNICODE] = null;
            arrFuncs[(int)NpyDefs.NPY_TYPES.NPY_VOID] = null;
        }


        #region GetItem functions
        internal static Object getitemBool(long offset, ndarray arr) {
            bool f;

            unsafe {
                bool* p = (bool*)((byte *)arr.data.ToPointer() + offset);
                f = *p;
            }
            return f;
        }

        // Both Byte and UByte
        internal static Object getitemByte(long offset, ndarray arr) {
            byte f;

            unsafe {
                byte* p = (byte*)arr.data.ToPointer() + offset;
                f = *p;
            }
            return f;
        }


        internal static Object getitemShort(long offset, ndarray arr) {
            short f;

            unsafe {
                byte* p = (byte *)arr.data.ToPointer() + offset;
                if (arr.IsBehaved) {
                    f = *(short*)p;
                } else {
                    CopySwap2((byte*)&f, p, !arr.IsNotSwapped);
                }
            }
            return f;
        }

        internal static Object getitemUShort(long offset, ndarray arr) {
            ushort f;

            unsafe {
                byte* p = (byte*)arr.data.ToPointer() + offset;
                if (arr.IsBehaved) {
                    f = *(ushort*)p;
                } else {
                    CopySwap2((byte*)&f, p, !arr.IsNotSwapped);
                }
            }
            return f;
        }

        internal static Object getitemInt32(long offset, ndarray arr) {
            int f;

            unsafe {
                byte* p = (byte*)arr.data.ToPointer() + offset;
                if (arr.IsBehaved) {
                    f = *(int*)p;
                } else {
                    CopySwap4((byte*)&f, p, !arr.IsNotSwapped);
                }
            }
            return f;
        }

        internal static Object getitemUInt32(long offset, ndarray arr) {
            uint f;

            unsafe {
                byte* p = (byte*)arr.data.ToPointer() + offset;
                if (arr.IsBehaved) {
                    f = *(uint*)p;
                } else {
                    CopySwap4((byte*)&f, p, !arr.IsNotSwapped);
                }
            }
            return f;
        }

        internal static Object getitemLong(long offset, ndarray arr) {
            long f;

            unsafe {
                byte* p = (byte*)arr.data.ToPointer() + offset;
                if (arr.IsBehaved) {
                    f = *(long*)p;
                } else {
                    CopySwap8((byte*)&f, p, !arr.IsNotSwapped);
                }
            }
            return f;
        }

        internal static Object getitemULong(long offset, ndarray arr) {
            ulong f;

            unsafe {
                byte* p = (byte*)arr.data.ToPointer() + offset;
                if (arr.IsBehaved) {
                    f = *(ulong*)p;
                } else {
                    CopySwap8((byte*)&f, p, !arr.IsNotSwapped);
                }
            }
            return f;
        }

        internal static Object getitemFloat(long offset, ndarray arr) {
            float f;

            unsafe {
                byte* p = (byte*)arr.data.ToPointer() + offset;
                if (arr.IsBehaved) {
                    f = *(float*)p;
                } else {
                    CopySwap4((byte*)&f, p, !arr.IsNotSwapped);
                }
            }
            return f;
        }

        internal static Object getitemDouble(long offset, ndarray arr) {
            double f;

            unsafe {
                byte* p = (byte*)arr.data.ToPointer() + offset;
                if (arr.IsBehaved) {
                    f = *(double*)p;
                } else {
                    CopySwap8((byte*)&f, p, !arr.IsNotSwapped);
                }
            }
            return f;
        }

        internal static Object getitemCDouble(long offset, ndarray arr) {
            Complex f;

            unsafe {
                byte* p = (byte*)arr.data.ToPointer() + offset;
                if (arr.IsBehaved) {
                    f = new Complex(*(double*)p, *((double*)p + 1));
                } else {
                    double r, i;
                    CopySwap8((byte*)&r, p, !arr.IsNotSwapped);
                    CopySwap8((byte*)&i, (byte*)((double*)p + 1), !arr.IsNotSwapped);
                    f = new Complex(r, i);
                }
            }
            return f;
        }

        #endregion


        #region SetItem methods

        internal static void setitemBool(Object o, long offset, ndarray arr) {
            bool f;

            if (o is Boolean) f = (bool)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToBoolean(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                bool* p = (bool*)((byte *)arr.data.ToPointer() + offset);
                *p = f;
            }
        }

        internal static void setitemByte(Object o, long offset, ndarray arr) {
            byte f;

            if (o is Byte) f = (byte)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToByte(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                byte* p = (byte*)arr.data.ToPointer() + offset;
                *p = f;
            }
        }

        internal static void setitemShort(Object o, long offset, ndarray arr) {
            short f;

            if (o is Int16) f = (short)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToInt16(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                byte* p = (byte *)arr.data.ToPointer() + offset;
                if (arr.IsBehaved) {
                    *(short*)p = f;
                } else {
                    CopySwap2(p, (byte*)&f, !arr.IsNotSwapped);
                }
            }
        }

        internal static void setitemUShort(Object o, long offset, ndarray arr) {
            ushort f;

            if (o is UInt16) f = (ushort)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToUInt16(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                byte* p = (byte *)arr.data.ToPointer() + offset;
                if (arr.IsBehaved) {
                    *(ushort*)p = f;
                } else {
                    CopySwap2(p, (byte*)&f, !arr.IsNotSwapped);
                }
            }
        }

        internal static void setitemInt32(Object o, long offset, ndarray arr) {
            int f;

            if (o is Int32) f = (int)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToInt32(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                byte* p = (byte*)arr.data.ToPointer() + offset;
                if (arr.IsBehaved) {
                    *(int*)p = f;
                } else {
                    CopySwap4(p, (byte*)&f, !arr.IsNotSwapped);
                }
            }
        }

        internal static void setitemUInt32(Object o, long offset, ndarray arr) {
            uint f;

            if (o is UInt32) f = (uint)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToUInt32(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                byte* p = (byte*)arr.data.ToPointer() + offset;
                if (arr.IsBehaved) {
                    *(uint*)p = f;
                } else {
                    CopySwap4(p, (byte*)&f, !arr.IsNotSwapped);
                }
            }
        }

        internal static void setitemLong(Object o, long offset, ndarray arr) {
            long f;

            if (o is Int64) f = (long)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToInt64(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                byte* p = (byte*)arr.data.ToPointer() + offset;
                if (arr.IsBehaved) {
                    *(long*)p = f;
                } else {
                    CopySwap8(p, (byte*)&f, !arr.IsNotSwapped);
                }
            }
        }

        internal static void setitemULong(Object o, long offset, ndarray arr) {
            ulong f;

            if (o is UInt64) f = (ulong)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToUInt64(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                byte* p = (byte*)arr.data.ToPointer() + offset;
                if (arr.IsBehaved) {
                    *(ulong*)p = f;
                } else {
                    CopySwap8(p, (byte*)&f, !arr.IsNotSwapped);
                }
            }
        }

        internal static void setitemFloat(Object o, long offset, ndarray arr) {
            float f;

            if (o is Single) f = (float)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToSingle(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                byte* p = (byte*)arr.data.ToPointer() + offset;
                if (arr.IsBehaved) {
                    *(float*)p = f;
                } else {
                    CopySwap4(p, (byte*)&f, !arr.IsNotSwapped);
                }
            }
        }

        internal static void setitemDouble(Object o, long offset, ndarray arr) {
            double f;

            if (o is Double) f = (double)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToDouble(null);
            else throw new NotImplementedException(
                String.Format("Elvis has just left Wichita (type {0}).", o.GetType().Name));

            unsafe {
                byte* p = (byte*)arr.data.ToPointer() + offset;
                if (arr.IsBehaved) {
                    *(double*)p = f;
                } else {
                    CopySwap8(p, (byte*)&f, !arr.IsNotSwapped);
                }
            }
        }


        internal static void setitemCDouble(Object o, long offset, ndarray arr) {
            Complex f;

            if (o is Complex) f = (Complex)o;
            else if (o is IConvertible) {
                double d = ((IConvertible)o).ToDouble(null);
                f = new Complex(d, 0.0);
            } else throw new NotImplementedException(
                String.Format("Elvis has just left Wichita (type {0}).", o.GetType().Name));

            unsafe {
                byte* p = (byte*)arr.data.ToPointer() + offset;
                if (arr.IsBehaved) {
                    *(double*)p = f.Real;
                    *((double*)p + 1) = f.Imaginary;
                } else {
                    double r = f.Real;
                    double i = f.Imaginary;
                    CopySwap8(p, (byte*)&r, !arr.IsNotSwapped);
                    CopySwap8(p, (byte*)&i, !arr.IsNotSwapped);
                }
            }
        }

        #endregion

        #region Copy ops for swapping and unaligned access
        /// <summary>
        /// Copies two bytes from src to dest, optionally swapping the order
        /// for a change of endianess.  Either way, unaligned access is handled correctly.
        /// </summary>
        /// <param name="dest">Destination pointer</param>
        /// <param name="src">Source pointer</param>
        /// <param name="swap">True swaps byte order, false preserves the byte ordering</param>
        private unsafe static void CopySwap2(byte* dest, byte* src, bool swap) {
            if (!swap) {
                dest[0] = src[0];
                dest[1] = src[1];
            } else {
                dest[0] = src[1];
                dest[1] = src[0];
            }
        }

        private unsafe static void CopySwap4(byte* dest, byte* src, bool swap) {
            if (!swap) {
                dest[0] = src[0];
                dest[1] = src[1];
                dest[2] = src[2];
                dest[3] = src[3];
            } else {
                dest[0] = src[3];
                dest[1] = src[2];
                dest[2] = src[1];
                dest[3] = src[0];
            }
        }

        private unsafe static void CopySwap8(byte* dest, byte* src, bool swap) {
            if (!swap) {
                dest[0] = src[0];
                dest[1] = src[1];
                dest[2] = src[2];
                dest[3] = src[3];
                dest[4] = src[4];
                dest[5] = src[5];
                dest[6] = src[6];
                dest[7] = src[7];
            } else {
                dest[0] = src[7];
                dest[1] = src[6];
                dest[2] = src[5];
                dest[3] = src[4];
                dest[4] = src[3];
                dest[5] = src[2];
                dest[6] = src[1];
                dest[7] = src[0];
            }
        }
        #endregion

        #region Comparison Functions

        private static Object SyncRoot = new Object();
        private static LanguageContext PyContext = null;
        private static CallSite<Func<CallSite, Object, Object, int>> Site_Equal;
        private static CallSite<Func<CallSite, Object, Object, int>> Site_NotEqual;
        private static CallSite<Func<CallSite, Object, Object, int>> Site_Greater;
        private static CallSite<Func<CallSite, Object, Object, int>> Site_GreaterEqual;
        private static CallSite<Func<CallSite, Object, Object, int>> Site_Less;
        private static CallSite<Func<CallSite, Object, Object, int>> Site_LessEqual;
        private static CallSite<Func<CallSite, Object, int>> Site_Sign;

        private static CallSite<Func<CallSite, Object, Object, Object>> Site_Add;
        private static CallSite<Func<CallSite, Object, Object, Object>> Site_Subtract;
        private static CallSite<Func<CallSite, Object, Object, Object>> Site_Multiply;
        private static CallSite<Func<CallSite, Object, Object, Object>> Site_Divide;

        internal static void InitUFuncOps(LanguageContext cntx) {
            // Fast escape which will occur all except the first time.
            if (PyContext != null) {
                if (PyContext != cntx) {
                    // I don't think this can happen, but just in case...
                    throw new NotImplementedException("Internal error: multiply IronPython contexts are not supported.");
                }
                return;
            }

            lock (SyncRoot) {
                if (PyContext == null) {
                    // Construct the call sites for each operation we will need. This is much
                    // faster than constructing/destroying them with each loop.
                    Site_Equal = CallSite<Func<CallSite, Object, Object, int>>.Create(
                        cntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.Equal));
                    Site_NotEqual = CallSite<Func<CallSite, Object, Object, int>>.Create(
                        cntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.NotEqual));
                    Site_Greater = CallSite<Func<CallSite, Object, Object, int>>.Create(
                        cntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.GreaterThan));
                    Site_GreaterEqual = CallSite<Func<CallSite, Object, Object, int>>.Create(
                        cntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.GreaterThanOrEqual));
                    Site_Less = CallSite<Func<CallSite, Object, Object, int>>.Create(
                        cntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.LessThan));
                    Site_LessEqual = CallSite<Func<CallSite, Object, Object, int>>.Create(
                        cntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.LessThanOrEqual));

                    Site_Add = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        cntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.Add));
                    Site_Subtract = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        cntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.Subtract));
                    Site_Multiply = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        cntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.Multiply));
                    Site_Divide = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        cntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.Divide));

                    // Set this last so any other accesses will block while we create
                    // the sites.
                    PyContext = cntx;
                }
            }
        }

        
        /// <summary>
        /// Generic comparison function.  First argument should be bound to one of
        /// the callsite operations.
        /// </summary>
        /// <param name="site">CallSite operation to be performed</param>
        /// <param name="aPtr">First argument</param>
        /// <param name="bPtr">Second argument</param>
        /// <returns>1 if true, 0 if false, -1 on error</returns>
        private static int GenericCmp(CallSite<Func<CallSite, Object, Object, int>> site,
            IntPtr aPtr, IntPtr bPtr) {
            Object a = GCHandle.FromIntPtr(aPtr).Target;
            Object b = GCHandle.FromIntPtr(bPtr).Target;
            return site.Target(Site_Equal, a, b);
        }
        internal delegate int del_GenericCmp(IntPtr a, IntPtr b);


        /// <summary>
        /// Generic binary operation.  First argument should be bound to a binary
        /// callsite function.
        /// </summary>
        /// <param name="site">Callsite of some binary operation to perform</param>
        /// <param name="aPtr">First argument</param>
        /// <param name="bPtr">Second argument</param>
        /// <returns>IntPtr to GCHandle referencing the result</returns>
        private static IntPtr GenericBinOp(CallSite<Func<CallSite, Object, Object, Object>> site,
            IntPtr aPtr, IntPtr bPtr) {
            Object a = GCHandle.FromIntPtr(aPtr).Target;
            Object b = GCHandle.FromIntPtr(bPtr).Target;
            Object r = site.Target(Site_Equal, a, b);
            return GCHandle.ToIntPtr(GCHandle.Alloc(r));
        }
        internal delegate IntPtr del_GenericBinOp(IntPtr a, IntPtr b);

        //static internal Func<IntPtr, IntPtr, int> Compare_Equal =
        //    (a, b) => GenericCmp(Site_Equal, a, b);
        static internal del_GenericCmp Compare_Equal =
            (a, b) => GenericCmp(Site_Equal, a, b);
        static internal del_GenericCmp  Compare_NotEqual =
            (a, b) => GenericCmp(Site_NotEqual, a, b);
        static internal del_GenericCmp Compare_Greater =
            (a, b) => GenericCmp(Site_Greater, a, b);
        static internal del_GenericCmp Compare_GreaterEqual =
            (a, b) => GenericCmp(Site_GreaterEqual, a, b);
        static internal del_GenericCmp Compare_Less =
            (a, b) => GenericCmp(Site_Less, a, b);
        static internal del_GenericCmp Compare_LessEqual =
            (a, b) => GenericCmp(Site_LessEqual, a, b);

        static internal del_GenericBinOp Compare_Add =
            (a, b) => GenericBinOp(Site_Add, a, b);
        static internal del_GenericBinOp Compare_Subtract =
            (a, b) => GenericBinOp(Site_Subtract, a, b);
        static internal del_GenericBinOp Compare_Multiply =
            (a, b) => GenericBinOp(Site_Multiply, a, b);
        static internal del_GenericBinOp Compare_Divide =
            (a, b) => GenericBinOp(Site_Divide, a, b);

        #endregion

    }
}
