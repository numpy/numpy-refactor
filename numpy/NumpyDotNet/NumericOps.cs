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
using Microsoft.CSharp.RuntimeBinder;
using System.Dynamic;

namespace NumpyDotNet {


    /// <summary>
    /// Records the type-specific get/set items for each descriptor type.
    /// </summary>
    public class ArrFuncs {
        internal Object GetItem(long offset, ndarray arr) {
            return GetFunc((IntPtr)(arr.data.ToInt64() + offset), arr);
        }

        internal void SetItem(Object value, long offset, ndarray arr) {
            SetFunc(value, (IntPtr)(arr.data.ToInt64() + offset), arr);
        }

        internal Func<IntPtr, ndarray, Object> GetFunc { get; set; }
        internal Action<Object, IntPtr, ndarray> SetFunc { get; set; }
    }



    /// <summary>
    /// Collection of getitem/setitem functions and operations on object types.
    /// These are mostly used as callbacks from the core and operate on native
    /// memory.
    /// </summary>
    internal static class NumericOps {
        private static ArrFuncs[] ArrFuncs = null;
        private static Object ArrFuncsSyncRoot = new Object();

        /// <summary>
        /// Returns the array of functions appropriate to a given type.  The actual
        /// functions in the array will vary with the type sizes in the native code.
        /// </summary>
        /// <param name="t">Native array type</param>
        /// <returns>Functions matching that type</returns>
        internal static ArrFuncs FuncsForType(NpyDefs.NPY_TYPES t) {
            if (ArrFuncs == null) {
                InitArrFuncs();
            }
            return ArrFuncs[(int)t];
        }

        private static void GetGetSetItems(int numBytes, 
            out Func<IntPtr, ndarray, Object> getter,
            out Action<Object, IntPtr, ndarray> setter,
            out Func<IntPtr, ndarray, Object> ugetter,
            out Action<Object, IntPtr, ndarray> usetter) {
            switch (numBytes) {
                case 4:
                    getter = NumericOps.getitemInt32;
                    setter = NumericOps.setitemInt32;
                    ugetter = NumericOps.getitemUInt32;
                    usetter = NumericOps.setitemUInt32;
                    break;

                case 8:
                    getter = NumericOps.getitemInt64;
                    setter = NumericOps.setitemInt64;
                    ugetter = NumericOps.getitemUInt64;
                    usetter = NumericOps.setitemUInt64;
                    break;
                    
                default:
                    throw new NotImplementedException(
                        String.Format("Numeric size of {0} is not yet implemented.", numBytes));
            }
        }


        /// <summary>
        /// Initializes the type-specific functions for each native type.
        /// </summary>
        private static void InitArrFuncs() {
            lock (ArrFuncsSyncRoot) {
                if (ArrFuncs == null) {
                    ArrFuncs[] arr = new ArrFuncs[(int)NpyDefs.NPY_TYPES.NPY_NTYPES];

                    Func<IntPtr, ndarray, Object> intGet, longGet, longLongGet;
                    Func<IntPtr, ndarray, Object> uintGet, ulongGet, ulongLongGet;
                    Action<Object, IntPtr, ndarray> intSet, longSet, longLongSet;
                    Action<Object, IntPtr, ndarray> uintSet, ulongSet, ulongLongSet;

                    GetGetSetItems(NpyCoreApi.Native_SizeOfInt, out intGet, out intSet,
                        out uintGet, out uintSet);
                    GetGetSetItems(NpyCoreApi.Native_SizeOfLong, out longGet, out longSet,
                        out ulongGet, out ulongSet);
                    GetGetSetItems(NpyCoreApi.Native_SizeOfLongLong, out longLongGet,
                        out longLongSet, out ulongLongGet, out ulongLongSet);

                    arr[(int)NpyDefs.NPY_TYPES.NPY_BOOL] =
                        new ArrFuncs() { GetFunc = NumericOps.getitemBool, SetFunc = NumericOps.setitemBool };
                    arr[(int)NpyDefs.NPY_TYPES.NPY_BYTE] =
                        new ArrFuncs() { GetFunc = NumericOps.getitemByte, SetFunc = NumericOps.setitemByte };
                    arr[(int)NpyDefs.NPY_TYPES.NPY_UBYTE] =
                        new ArrFuncs() { GetFunc = NumericOps.getitemByte, SetFunc = NumericOps.setitemByte };
                    arr[(int)NpyDefs.NPY_TYPES.NPY_SHORT] =
                        new ArrFuncs() { GetFunc = NumericOps.getitemShort, SetFunc = NumericOps.setitemShort };
                    arr[(int)NpyDefs.NPY_TYPES.NPY_USHORT] =
                        new ArrFuncs() { GetFunc = NumericOps.getitemUShort, SetFunc = NumericOps.setitemUShort };
                    arr[(int)NpyDefs.NPY_TYPES.NPY_INT] =
                        new ArrFuncs() { GetFunc = intGet, SetFunc = intSet };
                    arr[(int)NpyDefs.NPY_TYPES.NPY_UINT] =
                        new ArrFuncs() { GetFunc = uintGet, SetFunc = uintSet };
                    arr[(int)NpyDefs.NPY_TYPES.NPY_LONG] =
                        new ArrFuncs() { GetFunc = longGet, SetFunc = longSet };
                    arr[(int)NpyDefs.NPY_TYPES.NPY_ULONG] =
                        new ArrFuncs() { GetFunc = ulongGet, SetFunc = ulongSet };
                    arr[(int)NpyDefs.NPY_TYPES.NPY_LONGLONG] =
                        new ArrFuncs() { GetFunc = longLongGet, SetFunc = longLongSet };
                    arr[(int)NpyDefs.NPY_TYPES.NPY_ULONGLONG] =
                        new ArrFuncs() { GetFunc = ulongLongGet, SetFunc = ulongLongSet };
                    arr[(int)NpyDefs.NPY_TYPES.NPY_FLOAT] =
                        new ArrFuncs() { GetFunc = NumericOps.getitemFloat, SetFunc = NumericOps.setitemFloat };
                    arr[(int)NpyDefs.NPY_TYPES.NPY_DOUBLE] =
                        new ArrFuncs() { GetFunc = NumericOps.getitemDouble, SetFunc = NumericOps.setitemDouble };
                    arr[(int)NpyDefs.NPY_TYPES.NPY_LONGDOUBLE] = null;
                    arr[(int)NpyDefs.NPY_TYPES.NPY_CFLOAT] = null;
                    arr[(int)NpyDefs.NPY_TYPES.NPY_CDOUBLE] =
                        new ArrFuncs() { GetFunc = NumericOps.getitemCDouble, SetFunc = NumericOps.setitemCDouble };
                    arr[(int)NpyDefs.NPY_TYPES.NPY_CLONGDOUBLE] = null;
                    arr[(int)NpyDefs.NPY_TYPES.NPY_DATETIME] = null;
                    arr[(int)NpyDefs.NPY_TYPES.NPY_TIMEDELTA] = null;
                    arr[(int)NpyDefs.NPY_TYPES.NPY_OBJECT] =
                        new ArrFuncs() { GetFunc = NumericOps.getitemObject, SetFunc = NumericOps.setitemObject };
                    arr[(int)NpyDefs.NPY_TYPES.NPY_STRING] =
                        new ArrFuncs() { GetFunc = NumericOps.getitemString, SetFunc = NumericOps.setitemString };
                    arr[(int)NpyDefs.NPY_TYPES.NPY_UNICODE] = null;
                    arr[(int)NpyDefs.NPY_TYPES.NPY_VOID] =
                        new ArrFuncs() { GetFunc = NumericOps.getitemVOID, SetFunc = NumericOps.setitemVOID };

                    ArrFuncs = arr;
                }
            }
        }


        #region GetItem functions

        /// <summary>
        /// Delegate type for getitem* functions given to the core.  These take a
        /// pointer to the raw memory location and a pointer to the core NpyArray
        /// structure and return a boxed result.
        /// </summary>
        /// <param name="ptr">Pointer into some memory array data, may be unaligned</param>
        /// <param name="arr">Pointer to NpyArray core data structure</param>
        /// <returns>IntPtr to a GCHandle to the result object (boxed)</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate IntPtr GetitemDelegate(IntPtr ptr, IntPtr arr);

        /// <summary>
        /// Allocating the GCHandle and getting the interface pointer to the array
        /// object is slow and unnecessary for callers from the managed layer so
        /// this generic function takes care of the unwrapping / wrapping of arguments
        /// and return values.
        /// </summary>
        /// <param name="f">Specific getitem function to use</param>
        /// <param name="ptr">Array memory pointer (may be unaligned)</param>
        /// <param name="arrPtr">Point to the NpyArray core data structure</param>
        /// <returns>GCHandle to the result object</returns>
        private static IntPtr GetItemWrapper(Func<IntPtr, ndarray, Object> f, IntPtr ptr, IntPtr arrPtr) {
            Object result = f(ptr, NpyCoreApi.ToInterface<ndarray>(arrPtr));
            return GCHandle.ToIntPtr(GCHandle.Alloc(result));
        }


        internal static Object getitemBool(IntPtr ptr, ndarray arr) {
            bool f;

            unsafe {
                bool* p = (bool*)ptr.ToPointer();
                f = *p;
            }
            return f;
        }
        internal static GetitemDelegate getitemBoolDelegate =
            (ptr, arrPtr) => GetItemWrapper(getitemBool, ptr, arrPtr);


        // Both Byte and UByte
        internal static Object getitemByte(IntPtr ptr, ndarray arr) {
            byte f;

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
                f = *p;
            }
            return f;
        }
        internal static GetitemDelegate getitemByteDelegate =
            (ptr, arrPtr) => GetItemWrapper(getitemByte, ptr, arrPtr);


        internal static Object getitemShort(IntPtr ptr, ndarray arr) {
            short f;

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
                if (arr.IsBehaved) {
                    f = *(short*)p;
                } else {
                    CopySwap2((byte*)&f, p, !arr.IsNotSwapped);
                }
            }
            return f;
        }
        internal static GetitemDelegate getitemShortDelegate =
            (ptr, arrPtr) => GetItemWrapper(getitemShort, ptr, arrPtr);

        internal static Object getitemUShort(IntPtr ptr, ndarray arr) {
            ushort f;

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
                if (arr.IsBehaved) {
                    f = *(ushort*)p;
                } else {
                    CopySwap2((byte*)&f, p, !arr.IsNotSwapped);
                }
            }
            return f;
        }
        internal static GetitemDelegate getitemUShortDelegate =
            (ptr, arrPtr) => GetItemWrapper(getitemUShort, ptr, arrPtr);

        internal static Object getitemInt32(IntPtr ptr, ndarray arr) {
            int f;

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
                if (arr.IsBehaved) {
                    f = *(int*)p;
                } else {
                    CopySwap4((byte*)&f, p, !arr.IsNotSwapped);
                }
            }
            return f;
        }
        internal static GetitemDelegate getitemInt32Delegate =
            (ptr, arrPtr) => GetItemWrapper(getitemInt32, ptr, arrPtr);

        internal static Object getitemUInt32(IntPtr ptr, ndarray arr) {
            uint f;

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
                if (arr.IsBehaved) {
                    f = *(uint*)p;
                } else {
                    CopySwap4((byte*)&f, p, !arr.IsNotSwapped);
                }
            }
            return f;
        }
        internal static GetitemDelegate getitemUInt32Delegate =
            (ptr, arrPtr) => GetItemWrapper(getitemUInt32, ptr, arrPtr);

        internal static Object getitemInt64(IntPtr ptr, ndarray arr) {
            long f;

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
                if (arr.IsBehaved) {
                    f = *(long*)p;
                } else {
                    CopySwap8((byte*)&f, p, !arr.IsNotSwapped);
                }
            }
            return f;
        }
        internal static GetitemDelegate getitemInt64Delegate =
            (ptr, arrPtr) => GetItemWrapper(getitemInt64, ptr, arrPtr);

        internal static Object getitemUInt64(IntPtr ptr, ndarray arr) {
            ulong f;

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
                if (arr.IsBehaved) {
                    f = *(ulong*)p;
                } else {
                    CopySwap8((byte*)&f, p, !arr.IsNotSwapped);
                }
            }
            return f;
        }
        internal static GetitemDelegate getitemUInt64Delegate =
            (ptr, arrPtr) => GetItemWrapper(getitemUInt64, ptr, arrPtr);

        internal static Object getitemFloat(IntPtr ptr, ndarray arr) {
            float f;

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
                if (arr.IsBehaved) {
                    f = *(float*)p;
                } else {
                    CopySwap4((byte*)&f, p, !arr.IsNotSwapped);
                }
            }
            return f;
        }
        internal static GetitemDelegate getitemFloatDelegate =
            (ptr, arrPtr) => GetItemWrapper(getitemFloat, ptr, arrPtr);

        internal static Object getitemDouble(IntPtr ptr, ndarray arr) {
            double f;

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
                if (arr.IsBehaved) {
                    f = *(double*)p;
                } else {
                    CopySwap8((byte*)&f, p, !arr.IsNotSwapped);
                }
            }
            return f;
        }
        internal static GetitemDelegate getitemDoubleDelegate =
            (ptr, arrPtr) => GetItemWrapper(getitemDouble, ptr, arrPtr);

        internal static Object getitemCDouble(IntPtr ptr, ndarray arr) {
            Complex f;

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
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
        internal static GetitemDelegate getitemCDoubleDelegate =
            (ptr, arrPtr) => GetItemWrapper(getitemCDouble, ptr, arrPtr);

        internal static Object getitemObject(IntPtr ptr, ndarray arr) {
            IntPtr f;

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
                if (arr.IsBehaved) {
                    switch (IntPtr.Size) {
                        case 4:
                            f = new IntPtr(*(int*)p);
                            break;
                        case 8:
                            f = new IntPtr(*(long*)p);
                            break;
                        default:
                            throw new NotImplementedException(
                                String.Format("IntPtr of size {0} is not supported.", IntPtr.Size));
                    }
                } else if (IntPtr.Size == 4) {
                    int r;
                    CopySwap4((byte*)&r, p, !arr.IsNotSwapped);
                    f = new IntPtr(*(int*)p);
                } else if (IntPtr.Size == 8) {
                    long r;
                    CopySwap8((byte*)&r, p, !arr.IsNotSwapped);
                    f = new IntPtr(*(long*)p);
                } else {
                    throw new NotImplementedException(
                        String.Format("IntPtr of size {0} is not implemented.", IntPtr.Size));
                }
            }
            return GCHandle.FromIntPtr(f).Target;
        }
        internal static GetitemDelegate getitemObjectDelegate =
            (ptr, arrPtr) => GetItemWrapper(getitemObject, ptr, arrPtr);

        internal static Object getitemString(IntPtr ptr, ndarray arr) {
            String s = Marshal.PtrToStringAnsi(ptr, arr.dtype.ElementSize);
            return s.TrimEnd((char)0);
        }
        internal static GetitemDelegate getitemStringDelegate =
            (ptr, arrPtr) => GetItemWrapper(getitemString, ptr, arrPtr);


        internal static Object getitemVOID(IntPtr ptr, ndarray arr) {
            dtype d = arr.dtype;
            if (d.HasNames) {
                List<string> names = d.Names;
                object[] result = new object[names.Count];
                Int32 savedflags = arr.RawFlags;
                try {
                    int i = 0;
                    foreach (string name in names) {
                        NpyCoreApi.NpyArray_DescrField field = NpyCoreApi.GetDescrField(d, name);
                        dtype field_dtype = NpyCoreApi.ToInterface<dtype>(field.descr);
                        Marshal.WriteIntPtr(arr.Array, NpyCoreApi.ArrayOffsets.off_descr, field.descr);
                        int alignment = arr.dtype.Alignment;

                        long fieldPtr = ptr.ToInt64() + field.offset;
                        if (alignment > 1 &&
                            fieldPtr % alignment != 0) {
                            arr.RawFlags = savedflags & ~NpyDefs.NPY_ALIGNED;
                        } else {
                            arr.RawFlags = savedflags | NpyDefs.NPY_ALIGNED;
                        }
                        result[i++] = field_dtype.f.GetFunc(new IntPtr(fieldPtr), arr);
                    }
                    return new PythonTuple(result);
                } finally {
                    arr.RawFlags = savedflags;
                    Marshal.WriteIntPtr(arr.Array, NpyCoreApi.ArrayOffsets.off_descr, d.Descr);
                }
            } else {
                throw new NotImplementedException("VOID type only implemented for fields");
            }
        }
        internal static GetitemDelegate getitemVOIDDelegate =
            (ptr, arrPtr) => GetItemWrapper(getitemVOID, ptr, arrPtr);

        #endregion


        #region SetItem methods

        /// <summary>
        /// Delegate type for setitem* functions given to the core.  These take an
        /// IntPtr to a value (GCHandle), a pointer to the raw memory location and 
        /// a pointer to the core NpyArray structure. The value is written into the
        /// array in the appropriate native type.
        /// </summary>
        /// <param name="value">GCHandle to the value to be written</param>
        /// <param name="ptr">Pointer into some memory array data, may be unaligned</param>
        /// <param name="arr">Pointer to NpyArray core data structure</param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void SetitemDelegate(IntPtr value, IntPtr ptr, IntPtr arr);

        /// <summary>
        /// Allocating the GCHandle and getting the interface pointer to the array
        /// object is slow and unnecessary for callers from the managed layer so
        /// this generic function takes care of the unwrapping the arguments.
        /// </summary>
        /// <param name="f">Specific getitem function to use</param>
        /// <param name="value">GCHandle to the value to be written</param>
        /// <param name="ptr">Array memory pointer (may be unaligned)</param>
        /// <param name="arrPtr">Point to the NpyArray core data structure</param>
        private static void SetItemWrapper(Action<Object, IntPtr, ndarray> f, IntPtr value, IntPtr ptr, IntPtr arrPtr) {
            Object v = GCHandle.FromIntPtr(value);
            f(v, ptr, NpyCoreApi.ToInterface<ndarray>(arrPtr));
        }

        internal static void setitemBool(Object o, IntPtr ptr, ndarray arr) {
            bool f;

            if (o is Boolean) f = (bool)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToBoolean(null);
            else if (o is BigInteger) {
                int fTmp;
                ((BigInteger)o).AsInt32(out fTmp);
                f = (fTmp != 0);
            }  else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                bool* p = (bool*)ptr.ToPointer();
                *p = f;
            }
        }
        internal static SetitemDelegate setitemBoolDelegate =
            (value, ptr, arrPtr) => SetItemWrapper(setitemBool, value, ptr, arrPtr);

        internal static void setitemByte(Object o, IntPtr ptr, ndarray arr) {
            byte f;

            if (o is Byte) f = (byte)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToByte(null);
            else if (o is BigInteger) {
                int fTmp;
                ((BigInteger)o).AsInt32(out fTmp);
                f = (byte)fTmp;
            } else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
                *p = f;
            }
        }
        internal static SetitemDelegate setitemByteDelegate =
            (value, ptr, arrPtr) => SetItemWrapper(setitemByte, value, ptr, arrPtr);

        internal static void setitemShort(Object o, IntPtr ptr, ndarray arr) {
            short f;

            if (o is Int16) f = (short)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToInt16(null);
            else if (o is BigInteger) {
                int fTmp;
                ((BigInteger)o).AsInt32(out fTmp);
                f = (short)fTmp;
            }  else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                byte* p = (byte *)ptr.ToPointer();
                if (arr.IsBehaved) {
                    *(short*)p = f;
                } else {
                    CopySwap2(p, (byte*)&f, !arr.IsNotSwapped);
                }
            }
        }
        internal static SetitemDelegate setitemShortDelegate =
            (value, ptr, arrPtr) => SetItemWrapper(setitemShort, value, ptr, arrPtr);

        internal static void setitemUShort(Object o, IntPtr ptr, ndarray arr) {
            ushort f;

            if (o is UInt16) f = (ushort)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToUInt16(null);
            else if (o is BigInteger) {
                int fTmp;
                ((BigInteger)o).AsInt32(out fTmp);
                f = (ushort)fTmp;
            } else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                byte* p = (byte *)ptr.ToPointer();
                if (arr.IsBehaved) {
                    *(ushort*)p = f;
                } else {
                    CopySwap2(p, (byte*)&f, !arr.IsNotSwapped);
                }
            }
        }
        internal static SetitemDelegate setitemUShortDelegate =
            (value, ptr, arrPtr) => SetItemWrapper(setitemUShort, value, ptr, arrPtr);

        internal static void setitemInt32(Object o, IntPtr ptr, ndarray arr) {
            int f;

            if (o is Int32) f = (int)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToInt32(null);
            else if (o is BigInteger) ((BigInteger)o).AsInt32(out f);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
                if (arr.IsBehaved) {
                    *(int*)p = f;
                } else {
                    CopySwap4(p, (byte*)&f, !arr.IsNotSwapped);
                }
            }
        }
        internal static SetitemDelegate setitemInt32Delegate =
            (value, ptr, arrPtr) => SetItemWrapper(setitemInt32, value, ptr, arrPtr);

        internal static void setitemUInt32(Object o, IntPtr ptr, ndarray arr) {
            uint f;

            if (o is UInt32) f = (uint)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToUInt32(null);
            else if (o is BigInteger) ((BigInteger)o).AsUInt32(out f);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
                if (arr.IsBehaved) {
                    *(uint*)p = f;
                } else {
                    CopySwap4(p, (byte*)&f, !arr.IsNotSwapped);
                }
            }
        }
        internal static SetitemDelegate setitemUInt32Delegate =
            (value, ptr, arrPtr) => SetItemWrapper(setitemUInt32, value, ptr, arrPtr);

        internal static void setitemInt64(Object o, IntPtr ptr, ndarray arr) {
            long f;

            if (o is Int64) f = (long)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToInt64(null);
            else if (o is BigInteger) ((BigInteger)o).AsInt64(out f);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
                if (arr.IsBehaved) {
                    *(long*)p = f;
                } else {
                    CopySwap8(p, (byte*)&f, !arr.IsNotSwapped);
                }
            }
        }
        internal static SetitemDelegate setitemInt64Delegate =
            (value, ptr, arrPtr) => SetItemWrapper(setitemInt64, value, ptr, arrPtr);

        internal static void setitemUInt64(Object o, IntPtr ptr, ndarray arr) {
            ulong f;

            if (o is UInt64) f = (ulong)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToUInt64(null);
            else if (o is BigInteger) ((BigInteger)o).AsUInt64(out f);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
                if (arr.IsBehaved) {
                    *(ulong*)p = f;
                } else {
                    CopySwap8(p, (byte*)&f, !arr.IsNotSwapped);
                }
            }
        }
        internal static SetitemDelegate setitemUInt64Delegate =
            (value, ptr, arrPtr) => SetItemWrapper(setitemUInt64, value, ptr, arrPtr);

        internal static void setitemFloat(Object o, IntPtr ptr, ndarray arr) {
            float f;

            if (o is Single) f = (float)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToSingle(null);
            else throw new NotImplementedException("Elvis has just left Wichita.");

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
                if (arr.IsBehaved) {
                    *(float*)p = f;
                } else {
                    CopySwap4(p, (byte*)&f, !arr.IsNotSwapped);
                }
            }
        }
        internal static SetitemDelegate setitemFloatDelegate =
            (value, ptr, arrPtr) => SetItemWrapper(setitemFloat, value, ptr, arrPtr);

        internal static void setitemDouble(Object o, IntPtr ptr, ndarray arr) {
            double f;

            if (o is Double) f = (double)o;
            else if (o is IConvertible) f = ((IConvertible)o).ToDouble(null);
            else throw new NotImplementedException(
                String.Format("Elvis has just left Wichita (type {0}).", o.GetType().Name));

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
                if (arr.IsBehaved) {
                    *(double*)p = f;
                } else {
                    CopySwap8(p, (byte*)&f, !arr.IsNotSwapped);
                }
            }
        }
        internal static SetitemDelegate setitemDoubleDelegate =
            (value, ptr, arrPtr) => SetItemWrapper(setitemDouble, value, ptr, arrPtr);


        internal static void setitemCDouble(Object o, IntPtr ptr, ndarray arr) {
            Complex f;

            if (o is Complex) f = (Complex)o;
            else if (o is IConvertible) {
                double d = ((IConvertible)o).ToDouble(null);
                f = new Complex(d, 0.0);
            } else throw new NotImplementedException(
                String.Format("Elvis has just left Wichita (type {0}).", o.GetType().Name));

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
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
        internal static SetitemDelegate setitemCDoubleDelegate =
            (value, ptr, arrPtr) => SetItemWrapper(setitemCDouble, value, ptr, arrPtr);



        internal static void setitemObject(Object o, IntPtr ptr, ndarray arr) {
            IntPtr f = GCHandle.ToIntPtr(GCHandle.Alloc(o));
            IntPtr prev = IntPtr.Zero;

            unsafe {
                byte* p = (byte*)ptr.ToPointer();
                if (arr.IsBehaved) {
                    switch (IntPtr.Size) {
                        case 4:
                            prev = (IntPtr)(* (int*)p);
                            *(int*)p = (int)f;
                            break;
                        case 8:
                            prev = (IntPtr)(* (int*)p);
                            *(long*)p = (long)f;
                            break;
                        default:
                            throw new NotImplementedException(
                                String.Format("IntPtr size of {0} is not supported.", IntPtr.Size));
                    }
                } else if (IntPtr.Size == 4) {
                    int r;
                    CopySwap4((byte *)&r, p, !arr.IsNotSwapped);
                    prev = new IntPtr(r);
                    r = (int)f;
                    CopySwap4(p, (byte*)&r, !arr.IsNotSwapped);
                } else if (IntPtr.Size == 4) {
                    long r;
                    CopySwap8((byte *)&r, p, !arr.IsNotSwapped);
                    prev = new IntPtr(r);
                    r = (long)f;
                    CopySwap8(p, (byte*)&r, !arr.IsNotSwapped);
                } else {
                    throw new NotImplementedException(
                        String.Format("IntPtr size of {0} is not supported.", IntPtr.Size));
                }                    
            }

            // Release our handle to any previous object.
            if (prev != IntPtr.Zero) {
                GCHandle.FromIntPtr(prev).Free();
            }
        }
        internal static SetitemDelegate setitemObjectDelegate =
            (value, ptr, arrPtr) => SetItemWrapper(setitemObject, value, ptr, arrPtr);


        internal static void setitemString(Object o, IntPtr ptr, ndarray arr) {
            string s = o.ToString();
            byte[] bytes = Encoding.UTF8.GetBytes(s);
            int elsize = arr.dtype.ElementSize;
            int copySize = Math.Min(bytes.Length, elsize);
            int i;
            for (i = 0; i < copySize; i++) {
                Marshal.WriteByte(ptr, i, bytes[i]);
            }
            for (; i < elsize; i++) {
                Marshal.WriteByte(ptr, i, (byte)0);
            }
        }
        internal static SetitemDelegate setitemStringDelegate =
            (value, ptr, arrPtr) => SetItemWrapper(setitemString, value, ptr, arrPtr);


        internal static void setitemVOID(object value, IntPtr ptr, ndarray arr) {
            dtype d = arr.dtype;
            PythonTuple t = (value as PythonTuple);
            if (d.HasNames && t != null) {
                List<string> names = d.Names;
                if (names.Count != t.Count) {
                    throw new ArgumentException("size of tuple must match number of fields");
                }
                Int32 savedflags = arr.RawFlags;
                try {
                    int i = 0;
                    foreach (string name in names) {
                        NpyCoreApi.NpyArray_DescrField field = NpyCoreApi.GetDescrField(d, name);
                        dtype field_dtype = NpyCoreApi.ToInterface<dtype>(field.descr);
                        Marshal.WriteIntPtr(arr.Array, NpyCoreApi.ArrayOffsets.off_descr, field.descr);
                        int alignment = arr.dtype.Alignment;
                        long fieldPtr = ptr.ToInt64() + field.offset;
                        if (alignment > 1 &&
                            fieldPtr % alignment != 0) {
                            arr.RawFlags = savedflags & ~NpyDefs.NPY_ALIGNED;
                        } else {
                            arr.RawFlags = savedflags | NpyDefs.NPY_ALIGNED;
                        }
                        field_dtype.f.SetFunc(t[i++], new IntPtr(fieldPtr), arr);
                    }
                } finally {
                    arr.RawFlags = savedflags;
                    Marshal.WriteIntPtr(arr.Array, NpyCoreApi.ArrayOffsets.off_descr, d.Descr);
                }
            } else {
                throw new NotImplementedException("VOID type only implemented for fields");
            }
        }
        internal static SetitemDelegate setitemVOIDDelegate =
            (value, ptr, arrPtr) => SetItemWrapper(setitemVOID, value, ptr, arrPtr);

        #endregion

        #region Copy ops for swapping and unaligned access

        /// <summary>
        /// Delegate for single copy/swap operation.  Copies potentially unaligned
        /// n-byte value from src to dest. If 'swap' is set, byte order is reversed.
        /// </summary>
        /// <param name="dest">Destination pointer</param>
        /// <param name="src">Source pointer</param>
        /// <param name="swap">Unused</param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal unsafe delegate void CopySwapDelegate(byte *dest, byte *src, bool swap);

        /// <summary>
        /// Same as copyswap, but copies or swaps n values.
        /// </summary>
        /// <param name="dest">Destination pointer</param>
        /// <param name="dstride">Destination stride (bytes per iteration)</param>
        /// <param name="src">Source pointer</param>
        /// <param name="sstride">Source stride (bytes per iteration</param>
        /// <param name="n">Number of elements to copy</param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal unsafe delegate void CopySwapNDelegate(byte* dest, IntPtr dstride,
            byte* src, IntPtr sstride, IntPtr n, bool swap);


        /// <summary>
        /// Copies two bytes from src to dest, optionally swapping the order
        /// for a change of endianess.  Either way, unaligned access is handled correctly.
        /// </summary>
        /// <param name="dest">Destination pointer</param>
        /// <param name="src">Source pointer</param>
        /// <param name="swap">True swaps byte order, false preserves the byte ordering</param>
        private unsafe static void CopySwap1(byte* dest, byte* src, bool swap) {
            dest[0] = src[0];
        }
        unsafe internal static CopySwapDelegate CopySwap1Delegate =
            new CopySwapDelegate(CopySwap1);

        private unsafe static void CopySwap2(byte* dest, byte* src, bool swap) {
            if (!swap) {
                Marshal.WriteInt16((IntPtr)dest, Marshal.ReadInt16((IntPtr)src));
            } else {
                dest[0] = src[1];
                dest[1] = src[0];
            }
        }
        unsafe internal static CopySwapDelegate CopySwap2Delegate =
            new CopySwapDelegate(CopySwap2);

        private unsafe static void CopySwap4(byte* dest, byte* src, bool swap) {
            if (!swap) {
                Marshal.WriteInt32((IntPtr)dest, Marshal.ReadInt32((IntPtr)src));
            } else {
                dest[0] = src[3];
                dest[1] = src[2];
                dest[2] = src[1];
                dest[3] = src[0];
            }
        }
        unsafe internal static CopySwapDelegate CopySwap4Delegate =
            new CopySwapDelegate(CopySwap4);

        private unsafe static void CopySwap8(byte* dest, byte* src, bool swap) {
            if (!swap) {
                Marshal.WriteInt64((IntPtr)dest, Marshal.ReadInt64((IntPtr)src));
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
        unsafe internal static CopySwapDelegate CopySwap8Delegate =
            new CopySwapDelegate(CopySwap8);

        private unsafe static void CopySwapObject(byte* dest, byte* src, bool notused) {
            IntPtr tmp = IntPtr.Zero;
            tmp = Marshal.ReadIntPtr((IntPtr)dest);
            if (tmp != IntPtr.Zero) {
                GCHandle.FromIntPtr(tmp).Free();
            }
            tmp = Marshal.ReadIntPtr((IntPtr)src);
            if (tmp != IntPtr.Zero) {
                tmp = GCHandle.ToIntPtr(GCHandle.Alloc(GCHandle.FromIntPtr(tmp).Target));
            }
            Marshal.WriteIntPtr((IntPtr)dest, tmp);
        }
        unsafe internal static CopySwapDelegate CopySwapObjectDelegate =
            new CopySwapDelegate(CopySwapObject);



        private unsafe static void CopySwapNObject(byte* dest, IntPtr dstride, 
            byte* src, IntPtr sstride, IntPtr n, bool notused) {
            if ((IntPtr)src == IntPtr.Zero) return;

            for (long i = 0; i < (long)n; i++) {
                CopySwapObject(dest, src, false);
                dest += (int)dstride;
                src += (int)sstride;
            }
        }
        unsafe internal static CopySwapNDelegate CopySwapNObjectDelegate =
            new CopySwapNDelegate(CopySwapNObject);
        #endregion

        #region Object Operation Functions

        private static Object SyncRoot = new Object();
        private static CodeContext CodeCntx = null;
        private static LanguageContext PyContext = null;
        private static CallSite<Func<CallSite, Object, Object, Object>> Site_Equal;
        private static CallSite<Func<CallSite, Object, Object, Object>> Site_NotEqual;
        private static CallSite<Func<CallSite, Object, Object, Object>> Site_Greater;
        private static CallSite<Func<CallSite, Object, Object, Object>> Site_GreaterEqual;
        private static CallSite<Func<CallSite, Object, Object, Object>> Site_Less;
        private static CallSite<Func<CallSite, Object, Object, Object>> Site_LessEqual;
        private static CallSite<Func<CallSite, Object, int>> Site_Sign;

        private static CallSite<Func<CallSite, Object, Object, Object>> Site_Add;
        private static CallSite<Func<CallSite, Object, Object, Object>> Site_Subtract;
        private static CallSite<Func<CallSite, Object, Object, Object>> Site_Multiply;
        private static CallSite<Func<CallSite, Object, Object, Object>> Site_Divide;
        private static CallSite<Func<CallSite, Object, Object>> Site_Negative;
        private static CallSite<Func<CallSite, Object, Object>> Site_Abs;

        private static CallSite<Func<CallSite, Object, Object, Object>> Site_Power;
        private static CallSite<Func<CallSite, Object, Object, Object>> Site_Remainder;
        private static CallSite<Func<CallSite, Object, Object>> Site_Not;
        private static CallSite<Func<CallSite, Object, Object, Object>> Site_And;
        private static CallSite<Func<CallSite, Object, Object, Object>> Site_Or;
        private static CallSite<Func<CallSite, Object, Object, Object>> Site_Xor;
        private static CallSite<Func<CallSite, Object, Object, Object>> Site_LShift;
        private static CallSite<Func<CallSite, Object, Object, Object>> Site_RShift;

        internal static void InitUFuncOps(CodeContext cntx) {
            // Fast escape which will occur all except the first time.
            if (CodeCntx != null) {
                if (CodeCntx != cntx) {
                    // I don't think this can happen, but just in case...
                    throw new NotImplementedException("Internal error: multiply IronPython contexts are not supported.");
                }
                return;
            }

            lock (SyncRoot) {
                if (PyContext == null) {
                    LanguageContext pyCntx = cntx.LanguageContext;
                    PyContext = pyCntx;
                    CodeCntx = cntx;

                    // Construct the call sites for each operation we will need. This is much
                    // faster than constructing/destroying them with each loop.
                    Site_Equal = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        pyCntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.Equal));
                    Site_NotEqual = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        pyCntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.NotEqual));
                    Site_Greater = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        pyCntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.GreaterThan));
                    Site_GreaterEqual = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        pyCntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.GreaterThanOrEqual));
                    Site_Less = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        pyCntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.LessThan));
                    Site_LessEqual = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        pyCntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.LessThanOrEqual));

                    Site_Add = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        pyCntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.Add));
                    Site_Subtract = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        pyCntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.Subtract));
                    Site_Multiply = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        pyCntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.Multiply));
                    Site_Divide = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        pyCntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.Divide));
                    Site_Negative = CallSite<Func<CallSite, Object, Object>>.Create(
                        pyCntx.CreateUnaryOperationBinder(System.Linq.Expressions.ExpressionType.Negate));

                    Site_Abs = CallSite<Func<CallSite, Object, Object>>.Create(
                        Binder.InvokeMember(CSharpBinderFlags.None, "__abs__",
                        null, typeof(NumericOps),
                        new CSharpArgumentInfo[] { 
                            CSharpArgumentInfo.Create(CSharpArgumentInfoFlags.None, null),
                        }));

                    Site_Power = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        pyCntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.Power));
                    Site_Remainder = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        pyCntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.Modulo));
                    Site_Not = CallSite<Func<CallSite, Object, Object>>.Create(
                        pyCntx.CreateUnaryOperationBinder(System.Linq.Expressions.ExpressionType.Not));
                    Site_And = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        pyCntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.And));
                    Site_Or = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        pyCntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.Or));
                    Site_Xor = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        pyCntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.ExclusiveOr));
                    Site_LShift = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        pyCntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.LeftShift));
                    Site_RShift = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                        pyCntx.CreateBinaryOperationBinder(System.Linq.Expressions.ExpressionType.RightShift));

                    
                    
                    // Set this last so any other accesses will block while we create
                    // the sites.
                    PyContext = pyCntx;
                }
            }
        }


        /// <summary>
        /// Cache of method call sites taking zero arguments.
        /// </summary>
        private static Dictionary<string, CallSite<Func<CallSite, Object, Object>>> ZeroArgMethodSites =
            new Dictionary<string, CallSite<Func<CallSite, Object, Object>>>();

        /// <summary>
        /// Cache of method call sites taking one argument.
        /// </summary>
        private static Dictionary<string, CallSite<Func<CallSite, Object, Object, Object>>> OneArgMethodSites =
            new Dictionary<string, CallSite<Func<CallSite, Object, Object, Object>>>();

        /// <summary>
        /// Executes a specified method taking one argument on an object. In order to
        /// be efficient, each method name is cached with the call site instance so
        /// future calls (this will likely be called in a loop) execute faster.
        /// 
        /// Passing IntPtr.Zero for argPtr causes it to execute a zero-argument method,
        /// otherwise it executes a one-argument method.  No facility is in place for
        /// passing null to a one-argument method.
        /// </summary>
        /// <param name="objPtr">Object method should be invoked on</param>
        /// <param name="methodName">Method name</param>
        /// <param name="argPtr">Optional argument, pass IntPtr.Zero if not needed</param>
        /// <returns>IntPtr to GCHandle of result object</returns>
        unsafe internal static IntPtr MethodCall(IntPtr objPtr, sbyte *methodName, IntPtr argPtr) {
            Object obj = GCHandle.FromIntPtr(objPtr).Target;
            Object result = null;
            String method = new String(methodName);

            if (argPtr != IntPtr.Zero) {
                Object arg = GCHandle.FromIntPtr(argPtr).Target;
                CallSite<Func<CallSite, Object, Object, Object>> site;

                // Cache the call site object based on method name.
                lock (OneArgMethodSites) {
                    if (!OneArgMethodSites.TryGetValue(method, out site)) {
                        site = CallSite<Func<CallSite, Object, Object, Object>>.Create(
                            Binder.InvokeMember(CSharpBinderFlags.None, method,
                            null, typeof(NumericOps),
                            new CSharpArgumentInfo[] { 
                                CSharpArgumentInfo.Create(CSharpArgumentInfoFlags.None, null),
                                CSharpArgumentInfo.Create(CSharpArgumentInfoFlags.None, null)
                            }));
                        OneArgMethodSites.Add(method, site);
                    }
                }
                result = site.Target(site, obj, arg);
            } else {
                CallSite<Func<CallSite, Object, Object>> site;

                lock (ZeroArgMethodSites) {
                    if (!ZeroArgMethodSites.TryGetValue(method, out site)) {
                        site = CallSite<Func<CallSite, Object, Object>>.Create(
                            Binder.InvokeMember(CSharpBinderFlags.None, method,
                            null, typeof(NumericOps),
                            new CSharpArgumentInfo[] { 
                                CSharpArgumentInfo.Create(CSharpArgumentInfoFlags.None, null) 
                            }));
                        ZeroArgMethodSites.Add(method, site);
                    }
                }
                result = site.Target(site, obj);
            }
            return (result != null) ? GCHandle.ToIntPtr(GCHandle.Alloc(result)) : IntPtr.Zero;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        unsafe internal delegate IntPtr del_MethodCall(IntPtr a, sbyte *b, IntPtr arg);
        

        /// <summary>
        /// Generic comparison function.  First argument should be bound to one of
        /// the callsite operations.
        /// </summary>
        /// <param name="site">CallSite operation to be performed</param>
        /// <param name="aPtr">First argument</param>
        /// <param name="bPtr">Second argument</param>
        /// <returns>1 if true, 0 if false, -1 on error</returns>
        private static int GenericCmp(CallSite<Func<CallSite, Object, Object, Object>> site,
            IntPtr aPtr, IntPtr bPtr) {
            Object a = GCHandle.FromIntPtr(aPtr).Target;
            Object b = GCHandle.FromIntPtr(bPtr).Target;
            return (bool)site.Target(Site_Equal, a, b) ? 1 : 0;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate int del_GenericCmp(IntPtr a, IntPtr b);


        /// <summary>
        /// Generic unary operation.  First argument should be bound to a binary
        /// callsite function.
        /// </summary>
        /// <param name="site">Callsite of some binary operation to perform</param>
        /// <param name="aPtr">Function argument</param>
        /// <returns>IntPtr to GCHandle referencing the result</returns>
        private static IntPtr GenericUnaryOp(CallSite<Func<CallSite, Object, Object>> site,
            IntPtr aPtr) {
            Object a = GCHandle.FromIntPtr(aPtr).Target;
            Object r = site.Target(site, a);
            return GCHandle.ToIntPtr(GCHandle.Alloc(r));
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate IntPtr del_GenericUnaryOp(IntPtr a);


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
            Object r = site.Target(site, a, b);
            return GCHandle.ToIntPtr(GCHandle.Alloc(r));
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
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
        
        static internal del_GenericBinOp Op_Add =
            (a, b) => GenericBinOp(Site_Add, a, b);
        static internal del_GenericBinOp Op_Subtract =
            (a, b) => GenericBinOp(Site_Subtract, a, b);
        static internal del_GenericBinOp Op_Multiply =
            (a, b) => GenericBinOp(Site_Multiply, a, b);
        static internal del_GenericBinOp Op_Divide =
            (a, b) => GenericBinOp(Site_Divide, a, b);
        static internal del_GenericUnaryOp Op_Negate = 
            a => GenericUnaryOp(Site_Negative, a);
        static internal del_GenericUnaryOp Op_Sign = aPtr => {
            Object a = GCHandle.FromIntPtr(aPtr).Target;
            int result;

            if ((bool)Site_Less.Target(Site_Less, a, 0.0)) result = -1;
            else if ((bool)Site_Greater.Target(Site_Greater, a, 0.0)) result = 1;
            else result = 0;
            return GCHandle.ToIntPtr(GCHandle.Alloc(result));
        };
        static internal del_GenericUnaryOp Op_Absolute =
            a => GenericUnaryOp(Site_Abs, a);

        // TODO: trueDivide
        // TODO: floorDivide

        static internal del_GenericBinOp Op_Remainder =
            (a, b) => GenericBinOp(Site_Remainder, a, b);

        static internal del_GenericUnaryOp Op_Square = aPtr => {
            Object a = GCHandle.FromIntPtr(aPtr).Target;
            Object result = Site_Divide.Target(Site_Multiply, a, a);
            return GCHandle.ToIntPtr(GCHandle.Alloc(result));
        };

        static internal del_GenericBinOp Op_Power =
            (a, b) => GenericBinOp(Site_Power, a, b);

        static internal del_GenericUnaryOp Op_Reciprocal = aPtr => {
            Object a = GCHandle.FromIntPtr(aPtr).Target;
            Object result = Site_Divide.Target(Site_Divide, 1.0, a);
            return GCHandle.ToIntPtr(GCHandle.Alloc(result));
        };

        static internal del_GenericBinOp Op_Min = (aPtr, bPtr) => {
            Object a = GCHandle.FromIntPtr(aPtr).Target;
            Object b = GCHandle.FromIntPtr(bPtr).Target;
            Object result = (bool)Site_LessEqual.Target(Site_LessEqual, a, b) ? a : b;
            return GCHandle.ToIntPtr(GCHandle.Alloc(result));
        };

        static internal del_GenericBinOp Op_Max = (aPtr, bPtr) => {
            Object a = GCHandle.FromIntPtr(aPtr).Target;
            Object b = GCHandle.FromIntPtr(bPtr).Target;
            Object result = (bool)Site_GreaterEqual.Target(Site_GreaterEqual, a, b) ? a : b;
            return GCHandle.ToIntPtr(GCHandle.Alloc(result));
        };


        // Logical NOT - not reciprocal
        static internal del_GenericUnaryOp Op_Invert = aPtr => {
            Object a = GCHandle.FromIntPtr(aPtr).Target;
            Object result = Site_Not.Target(Site_Not, a);
            return GCHandle.ToIntPtr(GCHandle.Alloc(result));
        };

        static internal del_GenericBinOp Op_And =
            (a, b) => GenericBinOp(Site_And, a, b);
        static internal del_GenericBinOp Op_Or =
            (a, b) => GenericBinOp(Site_Or, a, b);
        static internal del_GenericBinOp Op_Xor =
            (a, b) => GenericBinOp(Site_Xor, a, b);
        static internal del_GenericBinOp Op_LShift =
            (a, b) => GenericBinOp(Site_LShift, a, b);
        static internal del_GenericBinOp Op_RShift =
            (a, b) => GenericBinOp(Site_RShift, a, b);

        // Just returns the number 1.
        static internal del_GenericUnaryOp Op_GetOne = aPtr => {
            return GCHandle.ToIntPtr(GCHandle.Alloc(1));
        };

        static internal int OBJECT_compare(IntPtr objPtrPtr1, IntPtr objPtrPtr2, IntPtr unused) {
            Object obj1 = DerefObjPtr(objPtrPtr1, 0);
            Object obj2 = DerefObjPtr(objPtrPtr2, 0);
            return IronPython.Runtime.Operations.PythonOps.Compare(obj1, obj2);
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate int del_OBJECT_compare(IntPtr ObjPtrPtr1, IntPtr objPtrPtr2,
            IntPtr unused);
        static internal del_OBJECT_compare OBJECT_compare_delegate = new del_OBJECT_compare(OBJECT_compare);


        static internal int OBJECT_argmax(IntPtr objArr, IntPtr nTmp, out IntPtr maxIndx, IntPtr unused) {
            long i;
            long n = (long)nTmp;
            IntPtr maxPtr = IntPtr.Zero;

            maxIndx = (IntPtr)0;
            for (i = 0; i < n && maxPtr == IntPtr.Zero; i++) {
                // Not using offset argument to ReadIntPtr because it's only 'int' size.
                maxPtr = Marshal.ReadIntPtr((IntPtr)((long)objArr + i * IntPtr.Size));
            }
            Object maxObj = (maxPtr != IntPtr.Zero) ? GCHandle.FromIntPtr(maxPtr).Target : null;
            for (; i < n; i++) {
                IntPtr curPtr = Marshal.ReadIntPtr((IntPtr)((long)objArr + i * IntPtr.Size));
                Object curObj = GCHandle.FromIntPtr(curPtr).Target;
                if (IronPython.Runtime.Operations.PythonOps.Compare(curObj, maxObj) > 0) {
                    maxPtr = curPtr;
                    maxObj = curObj;
                    maxIndx = (IntPtr)i;
                }
            }
            return 0;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate int del_OBJECT_argmax(IntPtr objArr, IntPtr n, out IntPtr maxIndex, IntPtr unused);
        static internal del_OBJECT_argmax OBJECT_argmax_delegate = new del_OBJECT_argmax(OBJECT_argmax);


        /// <summary>
        /// Computes the dot product of two array of object pointers (GCHandles).
        /// </summary>
        /// <param name="inPtr1"></param>
        /// <param name="stride1"></param>
        /// <param name="inPtr2"></param>
        /// <param name="stride2"></param>
        /// <param name="outPtr"></param>
        /// <param name="n"></param>
        /// <param name="unused"></param>
        static internal void OBJECT_dot(IntPtr inPtrPtr1, IntPtr stride1Tmp, IntPtr inPtrPtr2,
            IntPtr stride2Tmp, ref IntPtr outPtr, IntPtr nTmp, IntPtr unused) {
            Object cumsum = null;
            Object prod = null;
            long stride1 = (long)stride1Tmp;
            long stride2 = (long)stride2Tmp;
            long i;
            long n = (long)nTmp;

            for (i = 0; i < n; i++) {
                Object in1 = DerefObjPtr(inPtrPtr1, stride1 * i);
                Object in2 = DerefObjPtr(inPtrPtr2, stride2 * i);

                if (in1 == null || in2 == null) {
                    prod = (Object)false;
                } else {
                    prod = Site_Multiply.Target(Site_Multiply, in1, in2);
                    if (prod == null) {
                        cumsum = null;
                        break;
                    }
                }

                if (i == 0) {
                    cumsum = prod;
                } else {
                    cumsum = Site_Add.Target(Site_Add, cumsum, prod);
                    if (cumsum == null) {
                        break;
                    }
                }
            }

            if (outPtr != IntPtr.Zero) {
                GCHandle.FromIntPtr(outPtr).Free();
            }
            outPtr = GCHandle.ToIntPtr(GCHandle.Alloc(cumsum));
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void del_OBJECT_dot(IntPtr inPtrPtr1, IntPtr stride1Tmp, IntPtr inPtrPtr2,
            IntPtr stride2Tmp, ref IntPtr outPtr, IntPtr nTmp, IntPtr unused);
        static internal del_OBJECT_dot OBJECT_dot_delegate = new del_OBJECT_dot(OBJECT_dot);


        static internal bool OBJECT_nonzero(IntPtr inPtrPtr, IntPtr arrUnused) {
            Object obj = DerefObjPtr(inPtrPtr, 0);
            return (obj == null) ? false :
                IronPython.Runtime.Operations.PythonOps.IsTrue(obj);
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate bool del_OBJECT_nonzero(IntPtr inPtrPtr1, IntPtr unused);
        static internal del_OBJECT_nonzero OBJECT_nonzero_delegate = new del_OBJECT_nonzero(OBJECT_nonzero);



        /// <summary>
        /// Reads an IntPtr from a memory location and converts the GCHandle back
        /// to an object.  If the value is 0 then null is returned.  Unaligned
        /// addresses are handled correctly.
        /// </summary>
        /// <param name="ptr">Base memory location from which to read an IntPtr</param>
        /// <param name="offset">Offset past ptr in bytes</param>
        /// <returns>Object or null</returns>
        static private Object DerefObjPtr(IntPtr ptr, long offset) {
            IntPtr oPtr = Marshal.ReadIntPtr((IntPtr)((long)ptr + offset));
            return (oPtr != IntPtr.Zero) ? GCHandle.FromIntPtr(oPtr).Target : null;
        }


        private static double GetPriority(Object o, double defaultValue) {
            double priority = 0.0;

            if (o.GetType() != typeof(ndarray) &&
                IronPython.Runtime.Operations.PythonOps.HasAttr(CodeCntx, o, "__array_priority__")) {
                try {
                    Object a = IronPython.Runtime.Operations.PythonOps.GetBoundAttr(CodeCntx, o, "__array_priority__");
                    if (a != null) {
                        priority = (double)a;
                    }
                } catch (Exception) {
                    priority = defaultValue;
                }
            }
            return priority;
        }


        private static int ComparePriorityCallback(IntPtr obj1Ptr, IntPtr obj2Ptr) {
            Object obj1 = GCHandle.FromIntPtr(obj1Ptr).Target;
            Object obj2 = GCHandle.FromIntPtr(obj2Ptr).Target;
            int result = 0;

            if (IronPython.Runtime.Operations.PythonOps.CompareTypesNotEqual(CodeCntx, obj1, obj2) &&
                GetPriority(obj1, 0.0) > GetPriority(obj2, 0.0)) {
                // PyArray_GetPriority
            } else {
                result = 0;
            }
            return result;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate int del_ComparePriorityCallback(IntPtr objPtr1, IntPtr objPtr2);
        static internal del_ComparePriorityCallback ComparePriorityDelegate = 
            new del_ComparePriorityCallback(ComparePriorityCallback);


        #endregion


        #region Core registration

        static internal NpyArray_FunctionDefs GetFunctionDefs() {
            NpyArray_FunctionDefs defs = new NpyArray_FunctionDefs();

            defs.cast_from_obj = new IntPtr[(int)NpyDefs.NPY_TYPES.NPY_NTYPES];
            defs.cast_from_string = new IntPtr[(int)NpyDefs.NPY_TYPES.NPY_NTYPES];
            defs.cast_from_unicode = new IntPtr[(int)NpyDefs.NPY_TYPES.NPY_NTYPES];
            defs.cast_from_void = new IntPtr[(int)NpyDefs.NPY_TYPES.NPY_NTYPES];
            defs.cast_to_obj = new IntPtr[(int)NpyDefs.NPY_TYPES.NPY_NTYPES];
            defs.cast_to_string = new IntPtr[(int)NpyDefs.NPY_TYPES.NPY_NTYPES];
            defs.cast_to_unicode = new IntPtr[(int)NpyDefs.NPY_TYPES.NPY_NTYPES];
            defs.cast_to_void = new IntPtr[(int)NpyDefs.NPY_TYPES.NPY_NTYPES];

            defs.BOOL_getitem = Marshal.GetFunctionPointerForDelegate(getitemBoolDelegate);
            defs.BOOL_setitem = Marshal.GetFunctionPointerForDelegate(setitemBoolDelegate);
            defs.BYTE_getitem = Marshal.GetFunctionPointerForDelegate(getitemByteDelegate);
            defs.BYTE_setitem = Marshal.GetFunctionPointerForDelegate(setitemByteDelegate);
            defs.SHORT_getitem = Marshal.GetFunctionPointerForDelegate(getitemShortDelegate);
            defs.SHORT_setitem = Marshal.GetFunctionPointerForDelegate(setitemShortDelegate);
            defs.USHORT_getitem = Marshal.GetFunctionPointerForDelegate(getitemUShortDelegate);
            defs.USHORT_setitem = Marshal.GetFunctionPointerForDelegate(setitemUShortDelegate);

            switch (NpyCoreApi.Native_SizeOfInt) {
                case 4:
                    defs.INT_getitem = Marshal.GetFunctionPointerForDelegate(getitemInt32Delegate);
                    defs.INT_setitem = Marshal.GetFunctionPointerForDelegate(setitemInt32Delegate);
                    defs.UINT_getitem = Marshal.GetFunctionPointerForDelegate(getitemUInt32Delegate);
                    defs.UINT_setitem = Marshal.GetFunctionPointerForDelegate(setitemUInt32Delegate);
                    break;
                case 8:
                    defs.INT_getitem = Marshal.GetFunctionPointerForDelegate(getitemInt64Delegate);
                    defs.INT_setitem = Marshal.GetFunctionPointerForDelegate(setitemInt64Delegate);
                    defs.UINT_getitem = Marshal.GetFunctionPointerForDelegate(getitemUInt64Delegate);
                    defs.UINT_setitem = Marshal.GetFunctionPointerForDelegate(setitemUInt64Delegate);
                    break;
                default:
                    throw new NotImplementedException(
                        String.Format("Int size {0} is not yet supported.", NpyCoreApi.Native_SizeOfInt));
            }

            switch (NpyCoreApi.Native_SizeOfLong) {
                case 4:
                    defs.LONG_getitem = Marshal.GetFunctionPointerForDelegate(getitemInt32Delegate);
                    defs.LONG_setitem = Marshal.GetFunctionPointerForDelegate(setitemInt32Delegate);
                    defs.ULONG_getitem = Marshal.GetFunctionPointerForDelegate(getitemUInt32Delegate);
                    defs.ULONG_setitem = Marshal.GetFunctionPointerForDelegate(setitemUInt32Delegate);
                    break;
                case 8:
                    defs.LONG_getitem = Marshal.GetFunctionPointerForDelegate(getitemInt64Delegate);
                    defs.LONG_setitem = Marshal.GetFunctionPointerForDelegate(setitemInt64Delegate);
                    defs.ULONG_getitem = Marshal.GetFunctionPointerForDelegate(getitemUInt64Delegate);
                    defs.ULONG_setitem = Marshal.GetFunctionPointerForDelegate(setitemUInt64Delegate);
                    break;
                default:
                    throw new NotImplementedException(
                        String.Format("Int size {0} is not yet supported.", NpyCoreApi.Native_SizeOfInt));
            }

            defs.FLOAT_getitem = Marshal.GetFunctionPointerForDelegate(getitemFloatDelegate);
            defs.FLOAT_setitem = Marshal.GetFunctionPointerForDelegate(setitemFloatDelegate);
            defs.DOUBLE_getitem = Marshal.GetFunctionPointerForDelegate(getitemDoubleDelegate);
            defs.DOUBLE_setitem = Marshal.GetFunctionPointerForDelegate(setitemDoubleDelegate);
            defs.CDOUBLE_getitem = Marshal.GetFunctionPointerForDelegate(getitemCDoubleDelegate);
            defs.CDOUBLE_setitem = Marshal.GetFunctionPointerForDelegate(setitemCDoubleDelegate);
            defs.OBJECT_getitem = Marshal.GetFunctionPointerForDelegate(getitemObjectDelegate);
            defs.OBJECT_setitem = Marshal.GetFunctionPointerForDelegate(setitemObjectDelegate);
            defs.STRING_getitem = Marshal.GetFunctionPointerForDelegate(getitemStringDelegate);
            defs.STRING_setitem = Marshal.GetFunctionPointerForDelegate(setitemStringDelegate);
            defs.VOID_getitem = Marshal.GetFunctionPointerForDelegate(getitemVOIDDelegate);
            defs.VOID_setitem = Marshal.GetFunctionPointerForDelegate(setitemVOIDDelegate);

            defs.OBJECT_copyswapn = Marshal.GetFunctionPointerForDelegate(CopySwapNObjectDelegate);
            defs.OBJECT_copyswap = Marshal.GetFunctionPointerForDelegate(CopySwapObjectDelegate);
            defs.OBJECT_argmax = Marshal.GetFunctionPointerForDelegate(OBJECT_argmax_delegate);
            defs.OBJECT_compare = Marshal.GetFunctionPointerForDelegate(OBJECT_compare_delegate);
            defs.OBJECT_dotfunc = Marshal.GetFunctionPointerForDelegate(OBJECT_dot_delegate);
            defs.OBJECT_nonzero = Marshal.GetFunctionPointerForDelegate(OBJECT_nonzero_delegate);

            for (int i = 0; i < (int)NpyDefs.NPY_TYPES.NPY_NTYPES; i++) {
                defs.cast_to_obj[i] = (IntPtr)i;
            }
            defs.sentinel = NpyDefs.NPY_VALID_MAGIC;

            return defs;
        }


        /// <summary>
        /// MUST MATCH THE LAYOUT OF NpyArray_FunctionDefs IN THE CORE!!!
        /// The memory layout must be exact.  In C# the arrays are allocated
        /// separately but the marshal attribute 'ByValArray' causes them
        /// to be placed in-line like the C structure.  The sentinel value
        /// at the end must be set to NPY_VALID_MAGIC so the core can verify
        /// that the structure was correctly initialized and that the memory
        /// layouts are correct.
        /// </summary>
        [StructLayout(LayoutKind.Sequential), Serializable]
        internal struct NpyArray_FunctionDefs
        {
            // Get-set methods per type. 
            internal IntPtr BOOL_getitem;
            internal IntPtr BYTE_getitem;
            internal IntPtr UBYTE_getitem;
            internal IntPtr SHORT_getitem;
            internal IntPtr USHORT_getitem;
            internal IntPtr INT_getitem;
            internal IntPtr LONG_getitem;
            internal IntPtr UINT_getitem;
            internal IntPtr ULONG_getitem;
            internal IntPtr LONGLONG_getitem;
            internal IntPtr ULONGLONG_getitem;
            internal IntPtr FLOAT_getitem;
            internal IntPtr DOUBLE_getitem;
            internal IntPtr LONGDOUBLE_getitem;
            internal IntPtr CFLOAT_getitem;
            internal IntPtr CDOUBLE_getitem;
            internal IntPtr CLONGDOUBLE_getitem;
            internal IntPtr UNICODE_getitem;
            internal IntPtr STRING_getitem;
            internal IntPtr OBJECT_getitem;
            internal IntPtr VOID_getitem;
            internal IntPtr DATETIME_getitem;
            internal IntPtr TIMEDELTA_getitem;

            internal IntPtr BOOL_setitem;
            internal IntPtr BYTE_setitem;
            internal IntPtr UBYTE_setitem;
            internal IntPtr SHORT_setitem;
            internal IntPtr USHORT_setitem;
            internal IntPtr INT_setitem;
            internal IntPtr LONG_setitem;
            internal IntPtr UINT_setitem;
            internal IntPtr ULONG_setitem;
            internal IntPtr LONGLONG_setitem;
            internal IntPtr ULONGLONG_setitem;
            internal IntPtr FLOAT_setitem;
            internal IntPtr DOUBLE_setitem;
            internal IntPtr LONGDOUBLE_setitem;
            internal IntPtr CFLOAT_setitem;
            internal IntPtr CDOUBLE_setitem;
            internal IntPtr CLONGDOUBLE_setitem;
            internal IntPtr UNICODE_setitem;
            internal IntPtr STRING_setitem;
            internal IntPtr OBJECT_setitem;
            internal IntPtr VOID_setitem;
            internal IntPtr DATETIME_setitem;
            internal IntPtr TIMEDELTA_setitem;

            /* Object type methods. */
            internal IntPtr OBJECT_copyswapn;
            internal IntPtr OBJECT_copyswap;
            internal IntPtr OBJECT_compare;
            internal IntPtr OBJECT_argmax;
            internal IntPtr OBJECT_dotfunc;
            internal IntPtr OBJECT_scanfunc;
            internal IntPtr OBJECT_fromstr;
            internal IntPtr OBJECT_nonzero;
            internal IntPtr OBJECT_fill;
            internal IntPtr OBJECT_fillwithscalar;
            internal IntPtr OBJECT_scalarkind;
            internal IntPtr OBJECT_fastclip;
            internal IntPtr OBJECT_fastputmask;
            internal IntPtr OBJECT_fasttake;

            // Unboxing (object-to-type) 
            [MarshalAsAttribute(UnmanagedType.ByValArray, SizeConst = 23)]
            internal IntPtr[] cast_from_obj;
            // String-to-type 
            [MarshalAsAttribute(UnmanagedType.ByValArray, SizeConst = 23)]
            internal IntPtr[] cast_from_string;
            // Unicode-to-type 
            [MarshalAsAttribute(UnmanagedType.ByValArray, SizeConst = 23)]
            internal IntPtr[] cast_from_unicode;
            // Void-to-type
            [MarshalAsAttribute(UnmanagedType.ByValArray, SizeConst = 23)]
            internal IntPtr[] cast_from_void;

            // Boxing (type-to-object)
            [MarshalAsAttribute(UnmanagedType.ByValArray, SizeConst = 23)]
            internal IntPtr[] cast_to_obj;
            // Type-to-string 
            [MarshalAsAttribute(UnmanagedType.ByValArray, SizeConst = 23)]
            internal IntPtr[] cast_to_string;
            // Type-to-unicode 
            [MarshalAsAttribute(UnmanagedType.ByValArray, SizeConst = 23)]
            internal IntPtr[] cast_to_unicode;
            // Type-to-void
            [MarshalAsAttribute(UnmanagedType.ByValArray, SizeConst = 23)]
            internal IntPtr[] cast_to_void;

            internal int sentinel;
        }

        #endregion
    }
}
