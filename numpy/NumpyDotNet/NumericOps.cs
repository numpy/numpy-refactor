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
        internal static ArrFuncs[] arrFuncs = new ArrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_NTYPES];


        /// <summary>
        /// Initializes the type-specific functions for each native type.
        /// </summary>
        static NumericOps() {
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_BOOL] =
                new ArrFuncs() { GetItem = NumericOps.getitemBool, SetItem = NumericOps.setitemBool };
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_BYTE] =
                new ArrFuncs() { GetItem = NumericOps.getitemByte, SetItem = NumericOps.setitemByte };
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_UBYTE] =
                new ArrFuncs() { GetItem = NumericOps.getitemByte, SetItem = NumericOps.setitemByte };
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_SHORT] =
                new ArrFuncs() { GetItem = NumericOps.getitemShort, SetItem = NumericOps.setitemShort };
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_USHORT] =
                new ArrFuncs() { GetItem = NumericOps.getitemUShort, SetItem = NumericOps.setitemUShort };
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_INT] =
                new ArrFuncs() { GetItem = NumericOps.getitemInt32, SetItem = NumericOps.setitemInt32 };
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_UINT] =
                new ArrFuncs() { GetItem = NumericOps.getitemUInt32, SetItem = NumericOps.setitemUInt32 };
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_LONG] =
                new ArrFuncs() { GetItem = NumericOps.getitemLong, SetItem = NumericOps.setitemLong };
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_ULONG] =
                new ArrFuncs() { GetItem = NumericOps.getitemULong, SetItem = NumericOps.setitemULong };
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_LONGLONG] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_ULONGLONG] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_FLOAT] =
                new ArrFuncs() { GetItem = NumericOps.getitemFloat, SetItem = NumericOps.setitemFloat };
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_DOUBLE] =
                new ArrFuncs() { GetItem = NumericOps.getitemDouble, SetItem = NumericOps.setitemDouble };
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_LONGDOUBLE] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_CFLOAT] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_CDOUBLE] =
                new ArrFuncs() { GetItem = NumericOps.getitemCDouble, SetItem = NumericOps.setitemCDouble };
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_CLONGDOUBLE] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_DATETIME] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_TIMEDELTA] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_OBJECT] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_STRING] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_UNICODE] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_VOID] = null;
        }


        #region GetItem functions
        internal static Object getitemBool(long offset, ndarray arr) {
            bool f;

            unsafe {
                bool* p = (bool*)((byte *)arr.Data.ToPointer() + offset);
                f = *p;
            }
            return f;
        }

        // Both Byte and UByte
        internal static Object getitemByte(long offset, ndarray arr) {
            byte f;

            unsafe {
                byte* p = (byte*)arr.Data.ToPointer() + offset;
                f = *p;
            }
            return f;
        }


        internal static Object getitemShort(long offset, ndarray arr) {
            short f;

            unsafe {
                short* p = (short*)((byte *)arr.Data.ToPointer() + offset);
                f = *p;
            }
            return f;
        }

        internal static Object getitemUShort(long offset, ndarray arr) {
            ushort f;

            unsafe {
                ushort* p = (ushort*)((byte *)arr.Data.ToPointer() + offset);
                f = *p;
            }
            return f;
        }

        internal static Object getitemInt32(long offset, ndarray arr) {
            int f;

            unsafe {
                int* p = (int*)((byte *)arr.Data.ToPointer() + offset);
                f = *p;
            }
            return f;
        }

        internal static Object getitemUInt32(long offset, ndarray arr) {
            uint f;

            unsafe {
                uint* p = (uint*)((byte *)arr.Data.ToPointer() + offset);
                f = *p;
            }
            return f;
        }

        internal static Object getitemLong(long offset, ndarray arr) {
            long f;

            unsafe {
                long* p = (long*)((byte *)arr.Data.ToPointer() + offset);
                f = *p;
            }
            return f;
        }

        internal static Object getitemULong(long offset, ndarray arr) {
            ulong f;

            unsafe {
                ulong* p = (ulong*)((byte *)arr.Data.ToPointer() + offset);
                f = *p;
            }
            return f;
        }

        internal static Object getitemFloat(long offset, ndarray arr) {
            float f;

            unsafe {
                float* p = (float*)((byte *)arr.Data.ToPointer() + offset);
                f = *p;
            }
            return f;
        }

        internal static Object getitemDouble(long offset, ndarray arr) {
            double f;

            unsafe {
                double* p = (double*)((byte *)arr.Data.ToPointer() + offset);
                f = *p;
            }
            return f;
        }

        internal static Object getitemCDouble(long offset, ndarray arr) {
            Complex f;

            unsafe {
                double* r = (double*)((byte *)arr.Data.ToPointer() + offset);
                double* i = (double*)((byte *)arr.Data.ToPointer() + offset) + 1;   // Outside parens so +1 double size, not +1 byte
                f = new Complex(*r, *i);
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
                bool* p = (bool*)((byte *)arr.Data.ToPointer() + offset);
                *p = f;
            }
        }

        internal static void setitemByte(Object o, long offset, ndarray arr) {
            byte f;

            if (o is Byte) f = (byte)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToByte(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                byte* p = (byte*)arr.Data.ToPointer() + offset;
                *p = f;
            }
        }

        internal static void setitemShort(Object o, long offset, ndarray arr) {
            short f;

            if (o is Int16) f = (short)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToInt16(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                short* p = (short*)((byte *)arr.Data.ToPointer() + offset);
                *p = f;
            }
        }

        internal static void setitemUShort(Object o, long offset, ndarray arr) {
            ushort f;

            if (o is UInt16) f = (ushort)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToUInt16(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                ushort* p = (ushort*)((byte *)arr.Data.ToPointer() + offset);
                *p = f;
            }
        }

        internal static void setitemInt32(Object o, long offset, ndarray arr) {
            int f;

            if (o is Int32) f = (int)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToInt32(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                int* p = (int*)((byte *)arr.Data.ToPointer() + offset);
                *p = f;
            }
        }

        internal static void setitemUInt32(Object o, long offset, ndarray arr) {
            uint f;

            if (o is UInt32) f = (uint)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToUInt32(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                uint* p = (uint*)((byte *)arr.Data.ToPointer() + offset);
                *p = f;
            }
        }

        internal static void setitemLong(Object o, long offset, ndarray arr) {
            long f;

            if (o is Int64) f = (long)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToInt64(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                long* p = (long*)((byte *)arr.Data.ToPointer() + offset);
                *p = f;
            }
        }

        internal static void setitemULong(Object o, long offset, ndarray arr) {
            ulong f;

            if (o is UInt64) f = (ulong)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToUInt64(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                ulong* p = (ulong*)((byte *)arr.Data.ToPointer() + offset);
                *p = f;
            }
        }

        internal static void setitemFloat(Object o, long offset, ndarray arr) {
            float f;

            if (o is Single) f = (float)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToSingle(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                float* p = (float*)((byte *)arr.Data.ToPointer() + offset);
                *p = f;
            }
        }

        internal static void setitemDouble(Object o, long offset, ndarray arr) {
            double f;

            if (o is Double) f = (double)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToDouble(null);
            else throw new NotImplementedException(
                String.Format("Elvis has just left Wichita (type {0}).", o.GetType().Name));

            unsafe {
                double* p = (double*)((byte *)arr.Data.ToPointer() + offset);
                *p = f;
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
                double* p = (double*)((byte *)arr.Data.ToPointer() + offset);
                *p = f.Real;
                *p = f.Imaginary;
            }
        }

        #endregion
    }
}
