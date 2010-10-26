using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using IronPython.Runtime;
using IronPython.Runtime.Types;
using IronPython.Modules;
using Microsoft.Scripting;
using NumpyDotNet;

namespace NumpyDotNet {

    [PythonType]
    public class dtype : Wrapper {

        public static object __new__(CodeContext cntx, PythonType cls, object dtype) {
            return NpyDescr.DescrConverter(cntx, dtype);
        }

        /// <summary>
        /// Constructs a new NpyArray_Descr objet matching the passed one.
        /// Equivalent to NpyAray_DescrNew.
        /// </summary>
        /// <param name="d">Descriptor to duplicate</param>
        internal dtype(dtype d) {
            core = NpyCoreApi.NpyArray_DescrNew(d.core);
            funcs = NumericOps.FuncsForType(this.TypeNum);
        }
        
        /// <summary>
        /// Creates a wrapper for an array created on the native side, such as 
        /// the result of a slice operation.
        /// </summary>
        /// <param name="d">Pointer to core NpyArray_Descr structure</param>
        internal dtype(IntPtr d) {
            core = d;
            funcs = NumericOps.FuncsForType(this.TypeNum);
        }


        /// <summary>
        /// Creates a wrapper for an array created on the native side, such as 
        /// the result of a slice operation.
        /// </summary>
        /// <param name="d">Pointer to core NpyArray_Descr structure</param>
        internal dtype(IntPtr d, int type) {
            core = d;
            funcs = NumericOps.FuncsForType((NpyDefs.NPY_TYPES)type);
        }

        #region Properties

        public IntPtr Descr {
            get { return core; }
        }

        public bool IsNativeByteOrder {
            get { return ByteOrder != NpyCoreApi.OppositeByteOrder; }
        }

        public byte Kind {
            get {
                return Marshal.ReadByte(core, NpyCoreApi.DescrOffsets.off_kind);
            }
        }

        public NpyDefs.NPY_TYPECHAR Type {
            get { return (NpyDefs.NPY_TYPECHAR)Marshal.ReadByte(core, NpyCoreApi.DescrOffsets.off_type); }
        }

        public string @char {
            get {
                StringBuilder s = new StringBuilder(2);
                s.Append((char)Type);
                return s.ToString();
            }
        }

        public byte ByteOrder {
            get { return Marshal.ReadByte(core, NpyCoreApi.DescrOffsets.off_byteorder); }
            set { Marshal.WriteByte(core, NpyCoreApi.DescrOffsets.off_byteorder, value); }
        }

        public int Flags {
            get { return Marshal.ReadInt32(core, NpyCoreApi.DescrOffsets.off_flags); }
        }

        internal bool ChkFlags(int flags) {
            return (Flags & flags) == flags;
        }

        internal bool IsObject {
            get { return ChkFlags(NpyDefs.NPY_ITEM_REFCOUNT); }
        }

        public NpyDefs.NPY_TYPES TypeNum {
            get { return (NpyDefs.NPY_TYPES)Marshal.ReadInt32(core, NpyCoreApi.DescrOffsets.off_type_num); }
        }

        public int ElementSize {
            get { return Marshal.ReadInt32(core, NpyCoreApi.DescrOffsets.off_elsize); }
            internal set { Marshal.WriteInt32(core, NpyCoreApi.DescrOffsets.off_elsize, value); }
        }

        public int itemsize {
            get { return ElementSize; }
        }

        public int Alignment {
            get { return Marshal.ReadInt32(core, NpyCoreApi.DescrOffsets.off_alignment); }
        }

        public bool HasNames {
            get { return Marshal.ReadIntPtr(core, NpyCoreApi.DescrOffsets.off_names) != IntPtr.Zero; }
        }

        public List<string> Names {
            get {
                IntPtr names = Marshal.ReadIntPtr(core, NpyCoreApi.DescrOffsets.off_names);
                List<string> result = null;
                if (names != IntPtr.Zero) {
                    result = new List<string>();
                    int offset = 0;
                    while (true) {
                        IntPtr namePtr = Marshal.ReadIntPtr(names, offset);
                        if (namePtr == IntPtr.Zero) {
                            break;
                        }
                        offset += IntPtr.Size;
                        result.Add(Marshal.PtrToStringAnsi(namePtr));
                    }
                }
                return result;
            }
        }

        public PythonTuple names {
            get { return new PythonTuple(Names); }
        }


        public bool HasSubarray {
            get { return Marshal.ReadIntPtr(core, NpyCoreApi.DescrOffsets.off_subarray) != IntPtr.Zero; }
        }

        public ArrFuncs f {
            get { return funcs; }
        }

        public string str {
            get {
                byte endian = ByteOrder;
                int size = ElementSize;
                if (endian == (byte)'=') {
                    endian = NpyCoreApi.NativeByteOrder;
                }
                if (TypeNum == NpyDefs.NPY_TYPES.NPY_UNICODE) {
                    size /= 4;
                }
                StringBuilder result = new StringBuilder();
                result.Append((char)endian);
                result.Append((char)Kind);
                result.Append(size);
                return result.ToString();
            }
        }

        #endregion


        #region Comparison
        public override bool Equals(object obj) {
            if (obj != null && obj is dtype) return Equals((dtype)obj);
            return false;
        }

        public bool Equals(dtype other) {
            if (other == null) return false;
            return (this.core == other.core ||
                    NpyCoreApi.NpyArray_EquivTypes(core, other.core) != 0);
        }

        /// <summary>
        /// Compares two types and returns true if they are equivalent,
        /// including complex types, even if represented by two different
        /// underlying descriptor objects.
        /// </summary>
        /// <param name="t1">Type 1</param>
        /// <param name="t2">Type 2</param>
        /// <returns>True if types are equivalent</returns>
        public static bool operator ==(dtype t1, dtype t2) {
            return System.Object.ReferenceEquals(t1, t2) ||
                (object)t1 != null && (object)t2 != null && t1.Equals(t2);
        }

        public static bool operator !=(dtype t1, dtype t2) {
            return !System.Object.ReferenceEquals(t1, t2) ||
                (object)t1 != null && (object)t2 != null && !t1.Equals(t2);
        }

        public override int GetHashCode() {
            return (int)core;
        }
        #endregion


        #region Internal data

        /// <summary>
        /// Type-specific functions
        /// </summary>
        private readonly ArrFuncs funcs;

        #endregion

        #region Scalar type support

        class ScalarInfo {
            internal Type ScalarType;
            internal Func<ScalarGeneric> ScalarConstructor;

            internal static ScalarInfo Make<T>() where T: ScalarGeneric, new() {
                return new ScalarInfo { ScalarType = typeof(T), ScalarConstructor = (() => new T()) };
            }
        };

        private ScalarInfo scalarInfo = null;

        public Type ScalarType {
            get {
                if (scalarInfo == null) {
                    FindScalarInfo();
                }
                return scalarInfo.ScalarType;
            }
        }

        private void FindScalarInfo() {
            ScalarInfo info = null;
            NpyDefs.NPY_TYPES type = TypeNum;
            if (NpyDefs.IsSigned(type)) {
                switch (ElementSize) {
                    case 1:
                        info = ScalarInfo.Make<ScalarInt8>();
                        break;
                    case 2:
                        info = ScalarInfo.Make<ScalarInt16>();
                        break;
                    case 4:
                        info = ScalarInfo.Make<ScalarInt32>();
                        break;
                    case 8:
                        info = ScalarInfo.Make<ScalarInt64>();
                        break;
                }
            } else if (NpyDefs.IsUnsigned(type)) {
                switch (ElementSize) {
                    case 1:
                        info = ScalarInfo.Make<ScalarUInt8>();
                        break;
                    case 2:
                        info = ScalarInfo.Make<ScalarUInt16>();
                        break;
                    case 4:
                        info = ScalarInfo.Make<ScalarUInt32>();
                        break;
                    case 8:
                        info = ScalarInfo.Make<ScalarUInt64>();
                        break;
                }
            } else if (NpyDefs.IsFloat(type)) {
                switch (ElementSize) {
                    case 4:
                        info = ScalarInfo.Make<ScalarFloat32>();
                        break;
                    case 8:
                        info = ScalarInfo.Make<ScalarFloat64>();
                        break;
                }
            } else if (NpyDefs.IsComplex(type)) {
                switch (ElementSize) {
                    case 8:
                        info = ScalarInfo.Make<ScalarComplex64>();
                        break;
                    case 16:
                        info = ScalarInfo.Make<ScalarComplex128>();
                        break;
                }
            } else if (type == NpyDefs.NPY_TYPES.NPY_UNICODE) {
                info = ScalarInfo.Make<ScalarUnicode>();
            } else if (type == NpyDefs.NPY_TYPES.NPY_STRING) {
                info = ScalarInfo.Make<ScalarString>();
            } else if (type == NpyDefs.NPY_TYPES.NPY_BOOL) {
                info = ScalarInfo.Make<ScalarBool>();
            } else if (type == NpyDefs.NPY_TYPES.NPY_VOID) {
                info = ScalarInfo.Make<ScalarVoid>();
            } else if (type == NpyDefs.NPY_TYPES.NPY_OBJECT) {
                info = ScalarInfo.Make<ScalarObject>();
            }

            if (info == null) {
                info = new ScalarInfo();
            }

            scalarInfo = info;
        }

        public PythonType type {
            get {
                if (ScalarType != null) {
                    return DynamicHelpers.GetPythonTypeFromType(ScalarType);
                } else {
                    return null;
                }
            }
        }

        /// <summary>
        /// Converts a 0-d array to a scalar
        /// </summary>
        /// <param name="arr"></param>
        /// <returns></returns>
        internal object ToScalar(ndarray arr, long offset = 0) {
            if (ScalarType == null) {
                return arr.GetItem(offset);
            } else {
                ScalarGeneric result = scalarInfo.ScalarConstructor();
                result.FillData(arr, offset);
                return result;
            }
        }

        #endregion
    }
}
