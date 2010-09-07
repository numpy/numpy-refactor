using System;
using System.Collections.Generic;
using System.Linq;
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
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_BOOL] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_BYTE] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_UBYTE] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_SHORT] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_USHORT] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_INT] =
                new ArrFuncs() { GetItem = NumericOps.getitemInt32, SetItem = NumericOps.setitemInt32 };
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_UINT] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_LONG] =
                new ArrFuncs() { GetItem = NumericOps.getitemInt32, SetItem = NumericOps.setitemInt32 };
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_ULONG] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_LONGLONG] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_ULONGLONG] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_FLOAT] =
                new ArrFuncs() { GetItem = NumericOps.getitemFloat, SetItem = NumericOps.setitemFloat };
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_DOUBLE] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_LONGDOUBLE] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_CFLOAT] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_CDOUBLE] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_CLONGDOUBLE] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_DATETIME] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_TIMEDELTA] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_OBJECT] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_STRING] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_UNICODE] = null;
            arrFuncs[(int)NpyCoreApi.NPY_TYPES.NPY_VOID] = null;
        }


        #region GetItem functions
        internal static Object getitemInt32(long offset, ndarray arr) {
            int f;

            unsafe {
                int* p = (int*)arr.Data.ToPointer() + offset;
                f = *p;
            }
            return f;
        }

        internal static Object getitemLong(long offset, ndarray arr) {
            long f;

            unsafe {
                long* p = (long*)arr.Data.ToPointer() + offset;
                f = *p;
            }
            return f;
        }

        internal static Object getitemFloat(long offset, ndarray arr) {
            float f;

            unsafe {
                float* p = (float*)arr.Data.ToPointer() + offset;
                f = *p;
            }
            return f;
        }

        #endregion


        #region SetItem methods

        internal static void setitemInt32(Object o, long offset, ndarray arr) {
            int f;

            if (o is Int32) f = (int)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToInt32(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                int* p = (int*)arr.Data.ToPointer() + offset;
                *p = f;
            }
        }

        internal static void setitemLong(Object o, long offset, ndarray arr) {
            long f;

            if (o is Int64) f = (long)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToInt64(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                long* p = (long*)arr.Data.ToPointer() + offset;
                *p = f;
            }
        }

        internal static void setitemFloat(Object o, long offset, ndarray arr) {
            float f;

            if (o is Single) f = (float)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToSingle(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                float* p = (float*)arr.Data.ToPointer() + offset;
                *p = f;
            }
        }

        #endregion
    }
}
