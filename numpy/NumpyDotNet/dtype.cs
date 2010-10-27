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

        public static object __new__(CodeContext cntx, PythonType cls, object dtype,
                                     object align = null) {
            return NpyDescr.DescrConverter(cntx, dtype, NpyUtil_ArgProcessing.BoolConverter(align));
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

        public override string ToString() {
            return this.__str__();
        }


        #region Python interface

        public string __str__() {
            string ret;

            if (this.HasNames) {
                Object lst = this.descr;
                ret = (lst != null) ? PythonOps.ToString(lst) : "<err>";
                if (TypeNum != NpyDefs.NPY_TYPES.NPY_VOID) {
                    ret = String.Format("('{0}', {1})", this.str, this.descr);
                }
            } else if (this.HasSubarray) {
                // TODO: Subarrays are not implemented yet.  See descriptor.c:255 for code.
                throw new NotImplementedException("Subarrays are not yet implemented.");
            } else if (NpyDefs.IsFlexible(this.TypeNum) || !this.IsNativeByteOrder) {
                ret = this.str;
            } else {
                ret = this.name;
            }
            return ret;
        }

        public string __repr__() {
            string ret = __str__();

            if (ret != null) {
                ret = String.Format((!HasNames && !HasSubarray) ? "dtype('{0}')" : "dtype({0})", ret);
            }
            return ret;
        }

        public object subdtype {
            get {
                return HasSubarray ? new PythonTuple(new Object[] { @base, this.shape }) : null;
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

                string ret = String.Format("{0}{1}{2}", (char)endian, (char)this.Kind, size);
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
                    // TODO: This is wrong.
                    throw new NotImplementedException("Subarrays are not yet implemented.");
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


        public object fields { get { return this.GetFieldsDict(); } }

        public object dtinfo {
            get {
                IntPtr dtinfoPtr = Marshal.ReadIntPtr(core, NpyCoreApi.DescrOffsets.off_dtinfo);
                if (dtinfoPtr == IntPtr.Zero) {
                    return null;
                }

                NpyCoreApi.DateTimeInfo dtinfo = new NpyCoreApi.DateTimeInfo();
                Marshal.PtrToStructure(dtinfoPtr, dtinfo);

                return new PythonTuple(new object[] {
                    dtinfo.@base.ToString(), dtinfo.num, dtinfo.den, dtinfo.events });
            }
        }

        public int itemsize {
            get { return ElementSize; }
        }

        public PythonTuple names {
            get { return new PythonTuple(Names); }
            set {
                int n = this.Names.Count();
                if (!(value is IEnumerable<Object>)) {
                    throw new ArgumentException(String.Format("Value must be a sequence of {0} strings.", n));
                }
                if (value.Any(x => !(x is string))) {
                    throw new ArgumentException("All items must be strings.");
                }
                NpyCoreApi.NpyArrayAccess_SetNamesList(core, value.Cast<String>().ToArray(), n);
            }
        }

        public bool hasobject { get { return this.ChkFlags(NpyDefs.NPY__ITEM_HASOBJECT); } }

        public PythonType type {
            get {
                if (ScalarType != null) {
                    return DynamicHelpers.GetPythonTypeFromType(ScalarType);
                } else {
                    return null;
                }
            }
        }

        public string kind { get { return new string((char)this.Kind, 1); } }

        public string @char {
            get {
                return ((char)this.Type).ToString(); 
            }
        }

        public int num { get { return (int)this.TypeNum; } }

        public string byteorder { get { return new string((char)this.ByteOrder, 1); } }

        public int alignment { get { return this.Alignment; } }

        public int flags { get { return this.Flags; } }

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

        /// <summary>
        /// Returns the number of names.
        /// </summary>
        /// <returns></returns>
        public int __len__() {
            IntPtr names = Marshal.ReadIntPtr(core, NpyCoreApi.DescrOffsets.off_names);
            int result = 0;
            if (names != IntPtr.Zero) {
                int offset = 0;
                while (true) {
                    IntPtr namePtr = Marshal.ReadIntPtr(names, offset);
                    if (namePtr == IntPtr.Zero) {
                        break;
                    }
                    offset += IntPtr.Size;
                    result++;
                }
            }
            return result;
        }

        public object this[string field] {
            get {
                var d = GetFieldsDict();
                if (d == null) {
                    throw new KeyNotFoundException("There are no fields in the dtype");
                }
                PythonTuple val;
                if (!d.TryGetValue(field, out val)) {
                    throw new KeyNotFoundException(String.Format("Field named '{0}' not found.", field));
                }
                return val[0];
            }
        }

        public object this[int i] {
            get {
                var names = Names;
                if (i < 0 || i >= names.Count) {
                    throw new IndexOutOfRangeException("Field index out of range");
                }
                return this[names[i]];
            }
        }

        public bool HasSubarray {
            get { return Marshal.ReadIntPtr(core, NpyCoreApi.DescrOffsets.off_subarray) != IntPtr.Zero; }
        }

        public ndarray Subarray {
            get {
                IntPtr arr = Marshal.ReadIntPtr(core, NpyCoreApi.DescrOffsets.off_subarray);
                if (arr == IntPtr.Zero) return null;
                throw new NotImplementedException("Subarrays are not implemented yet");
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


        private Dictionary<string, PythonTuple> GetFieldsDict() {
            Dictionary<string, PythonTuple> ret;

            if (!HasNames) {
                ret = null;
            } else {
                IntPtr iter = IntPtr.Zero;
                IntPtr dict = Marshal.ReadIntPtr(this.core, NpyCoreApi.DescrOffsets.off_fields);
                ret = new Dictionary<string, PythonTuple>();
                try {
                    IntPtr keyPtr;
                    IntPtr value;
                    iter = NpyCoreApi.NpyDict_AllocIter();
                    while (NpyCoreApi.NpyDict_Next(dict, iter, out keyPtr, out value)) {
                        PythonTuple t;
                        string key = Marshal.PtrToStringAnsi(keyPtr);

                        IntPtr title = Marshal.ReadIntPtr(value, NpyCoreApi.DescrOffsets.off_fields_title);
                        dtype d = NpyCoreApi.ToInterface<dtype>(Marshal.ReadIntPtr(value, NpyCoreApi.DescrOffsets.off_fields_descr));
                        int offset = Marshal.ReadInt32(value, NpyCoreApi.DescrOffsets.off_fields_offset);
                        if (title == IntPtr.Zero) {
                            t = new PythonTuple(new Object[] { d, offset });
                        } else {
                            t = new PythonTuple(new Object[] { d, offset, Marshal.PtrToStringAnsi(title) });
                        }
                        ret.Add(key, t);
                    }
                } finally {
                    NpyCoreApi.NpyDict_FreeIter(iter);
                }
            }
            return ret;
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
            } else if (type == NpyDefs.NPY_TYPES.NPY_OBJECT) {
                info = ScalarInfo.Make<ScalarObject>();
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
            if (ScalarType == null ||
                ChkFlags(NpyDefs.NPY_USE_GETITEM)) {
                return arr.GetItem(offset);
            } else {
                ScalarGeneric result = scalarInfo.ScalarConstructor();
                return result.FillData(arr, offset);
            }
        }

        #endregion
    }
}
