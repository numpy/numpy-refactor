using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using IronPython.Runtime;
using IronPython.Runtime.Operations;
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

        #region Python interface

        public object subdtype {
            get {
                PythonTuple t = new PythonTuple(this.shape);
                return new PythonTuple(new Object[] { @base, t });
            }
        }


        /// <summary>
        /// Returns the name of the underlying data type such as 'int32' or 'object'.
        /// </summary>
        public string name {
            get {
                string typeName = (string)this.type.__getattribute__(NpyUtil_Python.DefaultContext, "__name__");
                if (NpyDefs.IsUserDefined(this.TypeNum)) {
                    int i = typeName.LastIndexOf('.');
                    if (i != -1) {
                        typeName = typeName.Substring(i + 1);
                    }
                } else {
                    int prefixLen = "numpy.".Length;
                    int len = typeName.Length;
                    if (typeName[len - 1] == '_') {
                        len--;
                    }
                    len -= prefixLen;
                    typeName = typeName.Substring(prefixLen, len);
                }

                if (NpyDefs.IsFlexible(this.TypeNum) && this.ElementSize != 0) {
                    typeName += this.ElementSize.ToString();
                }
                if (NpyDefs.IsDatetime(this.TypeNum)) {
                    typeName = AppendDateTimeTypestr(typeName);
                }
                return typeName;
            }
        }

        public string str {
            get {
                byte endian = this.ByteOrder;
                int size = this.ElementSize;

                if (endian == '=') {
                    endian = this.IsNativeByteOrder ? (byte)'<' : (byte)'>';
                }
                if (this.TypeNum == NpyDefs.NPY_TYPES.NPY_UNICODE) {
                    size >>= 2;
                }

                string ret = String.Format("{0}{1}{2}", endian, this.Kind, size);
                if (this.Type == NpyDefs.NPY_TYPECHAR.NPY_DATETIMELTR) {
                    ret = AppendDateTimeTypestr(ret);
                }
                return ret;
            }
        }

        public object descr {
            get {
                if (!this.HasNames) {
                    List<PythonTuple> res = new List<PythonTuple>();
                    res.Add(new PythonTuple(new Object[] { "", this.str }));
                    return res;
                }

                return NpyUtil_Python.CallInternal(NpyUtil_Python.DefaultContext, "_array_descr", this);
            }
        }

        public object @base {
            get {
                if (!this.HasSubarray) {
                    return this;
                } else {
                    return this.Subarray;
                }
            }
        }


        /// <summary>
        /// A tuple describing the size of each dimension of the array.
        /// </summary>
        public object shape {
            get { return this.HasSubarray ? this.Subarray.shape : new PythonTuple(); }
        }


        /// <summary>
        /// Returns 0 for built=-in types, 1 for a composite type, 2 for user-defined types.
        /// </summary>
        public int isbuiltin {
            get {
                int val = 0;

                if (this.fields != null) {
                    val = 1;
                }
                if (NpyDefs.IsUserDefined(this.TypeNum)) {
                    val = 2;
                }
                return val;
            }
        }

        public bool isnative {
            get {
                return NpyCoreApi.DescrIsNative(this.Descr) != 0;
            }
        }


        public object fields { get; set; }           // arraydescr_fields_get

        public object dtinfo { get; set; }           // arraydescr_dtinfo_get

        public PythonTuple names {
            get { return new PythonTuple(Names); }
            set { /* TODO */ }                  // arraydescr_names_set
        }

        public bool hasobject { get; set; }          // arraydescr_hasobject_get

        public PythonType type {
            get {
                return DynamicHelpers.GetPythonTypeFromType(ScalarType);
            }
        }


        public byte kind { get { return this.Kind; } }

        public string @char {
            get {
                StringBuilder s = new StringBuilder(2);
                s.Append((char)Type);
                return s.ToString();
            }
        }

        public int num { get; set; }                 // arraydescr_num_get

        public byte byteorder { get { return this.ByteOrder; } }

        public int itemsize { get; set; }            // arraydescr_itemsize_get

        public int alignment { get; set; }           // arraydescr_alignment_get

        public int flags { get; set; }               // arraydescr_flags_get

        #endregion

        #region .NET Properties

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


        public bool HasSubarray {
            get { return Marshal.ReadIntPtr(core, NpyCoreApi.DescrOffsets.off_subarray) != IntPtr.Zero; }
        }

        public ndarray Subarray {
            get {
                IntPtr arr = Marshal.ReadIntPtr(core, NpyCoreApi.DescrOffsets.off_subarray);
                return (arr != IntPtr.Zero) ? NpyCoreApi.ToInterface<ndarray>(arr) : null;
            }
        }

        public ArrFuncs f {
            get { return funcs; }
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


        #region Internal data & methods

        private string AppendDateTimeTypestr(string str) {
            // TODO: Fix date time type string. See descriptor.c: _append_to_datetime_typestr
            throw new NotImplementedException("to do ");
        }


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
            }

            if (info == null) {
                info = new ScalarInfo();
            }

            scalarInfo = info;
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
