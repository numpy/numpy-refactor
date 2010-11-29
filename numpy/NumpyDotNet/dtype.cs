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
    public class dtype : Wrapper, IEnumerable<Object> {

        public const string __module__ = "numpy";

        public static object __new__(CodeContext cntx, PythonType cls, object dtype,
                                     object align = null, object copy = null) {
            bool c = NpyUtil_ArgProcessing.BoolConverter(copy);
            dtype ret = NpyDescr.DescrConverter(cntx, dtype, NpyUtil_ArgProcessing.BoolConverter(align));
            if (c) {
                ret = NpyCoreApi.DescrNew(ret);
            }
            return ret;
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

        #region IEnumerable<object> interface

        public IEnumerator<object> GetEnumerator() {
            return new dtype_Enumerator(this);
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() {
            return new dtype_Enumerator(this);
        }

        #endregion

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
                dtype b = @base;
                if (!b.HasNames && !b.HasSubarray) {
                    ret = String.Format("('{0}',{1})", b.__str__(), shape);
                } else {
                    ret = String.Format("({0},{1})", b.__str__(), shape);
                }
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


        /// <summary>
        /// Called during pickling to store the state of the dtype instance. Output is
        /// sent to __setstate_ to restore the object.
        /// </summary>
        /// <param name="cntx"></param>
        /// <returns>Encoding of the object's state (Python tuple)</returns>
        public object __reduce__(CodeContext cntx, object notused=null) {
            const int version = 4;

            object[] ret = new object[3];

            object mod = PythonOps.ImportBottom(cntx, "numpy", 0);
            if (mod == null) return null;
            ret[0] = PythonOps.GetBoundAttr(cntx, mod, "dtype");
            if (ret[0] == null) return null;

            if (NpyDefs.IsUserDefined(TypeNum) ||
                TypeNum == NpyDefs.NPY_TYPES.NPY_VOID && false) {
                // TODO: The original code (descriptor.c:2072) checks the object type
                // against PyVoidArrType_Type, but we don't have an equivalent yet. Not
                // sure of the purpose of the check.
                throw new NotImplementedException("Pickling VOID arrays is unhandled.");
            } else {
                ret[1] = new PythonTuple(new object[] { String.Format("{0}{1}", (char)Kind, ElementSize), 0, 1 });
            }

            // Build up the state, which is at least the byte order.
            object[] state;

            byte endian = ByteOrder;
            if (endian == '=') {
                endian = NpyDefs.IsNativeByteOrder((byte)'<') ? (byte)'<' : (byte)'>';
            }

            if (NpyDefs.IsDatetime(TypeNum)) {
                state = new object[9];
                state[0] = version;
                state[8] = dtinfo;
            } else {
                state = new object[8];
                state[0] = 3;
            }
            state[1] = ((char)endian).ToString();
            state[2] = subdtype;
            state[3] = HasNames ? names : null;
            state[4] = HasNames ? fields : null;
            state[5] = NpyDefs.IsExtended(TypeNum) ? ElementSize : -1;
            state[6] = NpyDefs.IsExtended(TypeNum) ? alignment : -1;
            state[7] = flags;

            ret[2] = new PythonTuple(state);
            return new PythonTuple(ret);
        }


        /// <summary>
        /// Restores the state of an object using a pickling tuple such as generated by
        /// __reduce__() above.
        /// </summary>
        /// <param name="args"></param>
        /// <returns>Always returns null</returns>
        public object __setstate__(object argTmp) {
            PythonTuple args = argTmp as PythonTuple;

            if (args == null || !(args is PythonTuple)) {
                throw new ArgumentException("Invalid arguments to __setstate__().");
            }

            PythonTuple state = (PythonTuple)args;
            int elsize = -1;
            int alignment = -1;
            int version = 4;
            int dtypeFlags = 0;
            char endian = 'x';
            string[] names = null;
            PythonTuple namesTup = null;
            PythonDictionary fields = null;
            object dtinfo = null;
            object subarray = null;

            switch (state.Count()) {
                case 9:
                    dtinfo = state[8];
                    goto case 8;
                case 8:
                    dtypeFlags = (int)state[7];
                    goto case 7;
                case 7:
                    alignment = (int)state[6];
                    elsize = (int)state[5];
                    fields = (PythonDictionary)state[4];
                    namesTup = (PythonTuple)state[3];
                    subarray = state[2];
                    endian = ((string)state[1])[0];
                    version = (int)state[0];
                    break;

                case 6:
                    alignment = (int)state[5];
                    elsize = (int)state[4];
                    fields = (PythonDictionary)state[3];
                    subarray = state[2];
                    endian = ((string)state[1])[0];
                    version = (int)state[0];
                    break;

                case 5:
                    version = 0;
                    endian = ((string)state[0])[0];
                    subarray = state[1];
                    fields = (PythonDictionary)state[2];
                    elsize = (int)state[3];
                    alignment = (int)state[4];
                    break;

                default:
                    version = (state.Count() > 5) ? (int)state[0] : -1;
                    break;
            }

            // Reconstruct the names array for earlier format versions.
            if (version == 0 || version == 1) {
                names = (fields != null) ? ((PythonDictionary)fields).Keys.Cast<String>().ToArray() : null;
            } else {
                names = (namesTup != null) ? namesTup.Cast<String>().ToArray() : null;
            }

            if (fields == null && names != null || fields != null && names == null) {
                throw new ArgumentException("inconsistent fields and names during set state");
            }

            if (endian != '|' && NpyDefs.IsNativeByteOrder((byte)endian)) {
                endian = '=';
            }
            this.ByteOrder = (byte)endian;

            // Setup the subarray data.
            IntPtr subarrayPtr = Marshal.ReadIntPtr(this.core, NpyCoreApi.DescrOffsets.off_subarray);
            if (subarrayPtr != IntPtr.Zero) {
                NpyCoreApi.NpyArray_DestroySubarray(subarrayPtr);
            }
            if (subarray != null) {
                // subarray is tuple of descriptor, shape (tuple of ints).
                dtype baseDescr = (dtype)((PythonTuple)subarray)[0];
                PythonTuple shape = (PythonTuple)((PythonTuple)subarray)[1];
                NpyCoreApi.NpyArrayAccess_DescrReplaceSubarray(this.Descr, baseDescr.Descr, shape.Count(), shape.Cast<long>().Cast<IntPtr>().ToArray());
            }

            // Copy in the fields & names.
            if (names != null) {
                IntPtr namesPtr = NpyCoreApi.NpyArray_DescrAllocNames(names.Length);
                IntPtr fieldsPtr = NpyCoreApi.NpyArray_DescrAllocFields();

                names.Iteri((n, i) => {
                    PythonTuple fieldInfo = (PythonTuple)fields[n];
                    dtype descr = (dtype)fieldInfo[0];
                    int offset = (int)fieldInfo[1];
                    string title = (fieldInfo.__len__() == 3) ? (string)fieldInfo[2] : null;
                    NpyCoreApi.AddField(fieldsPtr, namesPtr, i, n, descr, offset, title);
                });
                NpyCoreApi.DescrReplaceFields(this.Descr, namesPtr, fieldsPtr);
            }

            if (NpyDefs.IsExtended(this.TypeNum)) {
                this.ElementSize = elsize;
                this.Alignment = alignment;
            }
            this.Flags = dtypeFlags;
            if (version < 3) {
                this.Flags = NpyCoreApi.NpyArray_DescrFindObjectFlag(this.Descr);
            }

            // Reset the date/time info if present.
            if (NpyDefs.IsDatetime(this.TypeNum) && dtinfo != null) {
                this.dtinfo = dtinfo;
            }

            return null;
        }


        public object __repeat__(object length) {
            if ((int)length < 0) {
                throw new ArgumentException(
                    String.Format("Array length must be >= 0, not {0}", length));
            }

            return NpyDescr.DescrConverter(IronPython.Runtime.DefaultContext.Default, new PythonTuple(new object[] { this, length }));
        }


        /// <summary>
        /// IronPython doesn't appear to automatically trigger the repeat function so we do it manually here.
        /// </summary>
        /// <param name="a">Integer repeat index</param>
        /// <param name="b">Input dtype</param>
        /// <returns></returns>
        public static object operator *(object a, dtype b) {
            try {
                long len = NpyUtil_ArgProcessing.IntConverter(a);
                return b.__repeat__(a);
            } catch {
                throw new ArgumentTypeException(
                    String.Format("can't multiply sequence by non-int of type '{0}'", a.GetType().Name));
            }
        }

        /// <summary>
        /// IronPython doesn't appear to automatically trigger the repeat function so we do it manually here.
        /// </summary>
        /// <param name="a">Input dtype</param>
        /// <param name="b">Integer repeat index</param>
        /// <returns></returns>
        public static object operator *(dtype a, object b) {
            try {
                long len = NpyUtil_ArgProcessing.IntConverter(b);
                return a.__repeat__(b);
            } catch {
                throw new ArgumentTypeException(
                    String.Format("can't multiply sequence by non-int of type '{0}'", b.GetType().Name));
            }
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
                    endian = NpyDefs.IsNativeByteOrder((byte)'<') ? (byte)'<' : (byte)'>';
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

        public dtype @base {
            get {
                unsafe {
                    var subarray = Subarray;
                    if (subarray == null) {
                        return this;
                    } else {
                        return NpyCoreApi.ToInterface<dtype>(subarray->@base);
                    }
                }
            }
        }


        /// <summary>
        /// A tuple describing the size of each dimension of the array.
        /// </summary>
        public PythonTuple shape {
            get {
                unsafe {
                    var subarray = Subarray;
                    if (subarray == null) {
                        return new PythonTuple();
                    } else {
                        long n = subarray->shape_num_dims.ToInt64();
                        object[] dims = new object[n];
                        for (long i=0; i<n; i++) {
                            dims[i] = subarray->shape_dims[i].ToPython();
                        }
                        return new PythonTuple(dims);
                    }
                }
            }
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

            internal set {
                PythonTuple dtTup = (PythonTuple)value;
                Marshal.WriteIntPtr(core, NpyCoreApi.DescrOffsets.off_dtinfo,
                    NpyCoreApi.NpyArray_DateTimeInfoNew((string)dtTup[0], (int)dtTup[1], (int)dtTup[2], (int)dtTup[3]));
            }
        }

        public int itemsize {
            get { return ElementSize; }
        }

        public object names {
            get {
                var n = Names;
                if (n != null) {
                    return new PythonTuple(n);
                } else {
                    return null;
                }
            }
            set {
                int n = this.Names.Count();
                IEnumerable<object> ival = value as IEnumerable<object>;
                if (ival == null) {
                    throw new ArgumentException(String.Format("Value must be a sequence of {0} strings.", n));
                }
                if (ival.Any(x => !(x is string))) {
                    throw new ArgumentException("All items must be strings.");
                }
                NpyCoreApi.NpyArrayAccess_SetNamesList(core, ival.Cast<String>().ToArray(), n);
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

        public dtype newbyteorder(string endian = null) {
            return NpyCoreApi.DescrNewByteorder(this, NpyUtil_ArgProcessing.ByteorderConverter(endian));
        }

        public object this[Object idx] {
            get {
                object result;

                if (!this.HasNames) {
                    result = null;
                } else if (idx is string) {
                    if (!GetFieldsDict().TryGetValue((string)idx, out result)) {
                        throw new System.Collections.Generic.KeyNotFoundException(
                            String.Format("Field named \"{0}\" not found.", (string)idx));
                    }
                } else {
                    try {
                        int i = (int)NpyUtil_Python.ConvertToLong(idx);
                        result = GetFieldsDict()[Names[i]]; // Names list checks index out of range, throws correct exception.
                    } catch {
                        throw new ArgumentException("Field key must be an integer, string, or unicode");
                    }
                }

                // If result is set, it is a PythonTuple of the dtype and offset. We just want to return
                // the dtype itself.
                if (result != null) {
                    result = ((PythonTuple)result)[0];
                }
                return result;
            }
        }

        #endregion

        #region .NET Properties

        public IntPtr Descr {
            get { return core; }
        }

        public bool IsNativeByteOrder {
            get { return NpyDefs.IsNativeByteOrder(ByteOrder); }
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
            internal set { Marshal.WriteInt32(core, NpyCoreApi.DescrOffsets.off_flags, value); }
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
            internal set { Marshal.WriteInt32(core, NpyCoreApi.DescrOffsets.off_alignment, value); }
        }

        public bool HasNames {
            get { return Marshal.ReadIntPtr(core, NpyCoreApi.DescrOffsets.off_names) != IntPtr.Zero; }
        }

        public int Length {
            get {
                IntPtr names = Marshal.ReadIntPtr(core, NpyCoreApi.DescrOffsets.off_names);
                int result = 0;
                if (names != IntPtr.Zero) {
                    int offset = 0;
                    while (Marshal.ReadIntPtr(names, offset) != IntPtr.Zero) {
                        offset += IntPtr.Size;
                        result++;
                    }
                }
                return result;
            }
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
            return this.Length;
        }

        public bool HasSubarray {
            get { return Marshal.ReadIntPtr(core, NpyCoreApi.DescrOffsets.off_subarray) != IntPtr.Zero; }
        }

        internal unsafe NpyCoreApi.NpyArray_ArrayDescr* Subarray {
            get {
                IntPtr arr = Marshal.ReadIntPtr(core, NpyCoreApi.DescrOffsets.off_subarray);
                return (NpyCoreApi.NpyArray_ArrayDescr*)arr.ToPointer();
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
            return (this.core == other.core || NpyCoreApi.EquivTypes(this, other));
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
            return !System.Object.ReferenceEquals(t1, t2) &&
                ((object)t1 == null || (object)t2 == null || !t1.Equals(t2));
        }

        public override int GetHashCode() {
            int hash = 17;
            foreach (object item in HashItems()) {
                hash = hash * 31 + item.GetHashCode();
            }
            return hash;
        }

        private IEnumerable<object> HashItems() {
            if (!HasNames && !HasSubarray) {
                yield return Kind;
                yield return ByteOrder;
                yield return TypeNum;
                yield return ElementSize;
                yield return Alignment;
            } else {
                if (HasNames) {
                    foreach (object item in FieldsHashItems()) {
                        yield return item;
                    }
                }
                if (HasSubarray) {
                    foreach (object item in SubarrayHashItems()) {
                        yield return item;
                    }
                }
            }
        }

        private IEnumerable<object> FieldsHashItems() {
            if (HasNames) {
                IntPtr iter = IntPtr.Zero;
                IntPtr dict = Marshal.ReadIntPtr(this.core, NpyCoreApi.DescrOffsets.off_fields);
                try {
                    IntPtr keyPtr;
                    IntPtr value;
                    iter = NpyCoreApi.NpyDict_AllocIter();
                    while (NpyCoreApi.NpyDict_Next(dict, iter, out keyPtr, out value)) {
                        yield return Marshal.PtrToStringAnsi(keyPtr);
                        IntPtr title = Marshal.ReadIntPtr(value, NpyCoreApi.DescrOffsets.off_fields_title);
                        dtype d = NpyCoreApi.ToInterface<dtype>(Marshal.ReadIntPtr(value, NpyCoreApi.DescrOffsets.off_fields_descr));
                        int offset = Marshal.ReadInt32(value, NpyCoreApi.DescrOffsets.off_fields_offset);
                        foreach (object item in d.HashItems()) {
                            yield return item;
                        }
                        yield return offset;
                        if (title != IntPtr.Zero) {
                            yield return Marshal.PtrToStringAnsi(title);
                        }
                    }
                } finally {
                    NpyCoreApi.NpyDict_FreeIter(iter);
                }
            }
        }

        private IEnumerable<object> SubarrayHashItems() {
            if (HasSubarray) {
                foreach (object item in @base.HashItems()) {
                    yield return item;
                }
                foreach (object item in shape) {
                    yield return item;
                }
            }
        }

        #region Python richcompare operators

        public object __lt__(CodeContext cntx, object other) {
            dtype d = (other is dtype) ? (dtype)other : NpyDescr.DescrConverter(cntx, other);
            return NpyCoreApi.EquivTypes(this, d) && NpyCoreApi.CanCastTo(this, d);
        }

        public object __le__(CodeContext cntx, object other) {
            dtype d = (other is dtype) ? (dtype)other : NpyDescr.DescrConverter(cntx, other);
            return NpyCoreApi.CanCastTo(this, d);
        }

        public object __eq__(CodeContext cntx, object other) {
            dtype d = (other is dtype) ? (dtype)other : NpyDescr.DescrConverter(cntx, other);
            return NpyCoreApi.EquivTypes(this, d);
        }

        public object __ne__(CodeContext cntx, object other) {
            dtype d = (other is dtype) ? (dtype)other : NpyDescr.DescrConverter(cntx, other);
            return !NpyCoreApi.EquivTypes(this, d);
        }

        public object __ge__(CodeContext cntx, object other) {
            dtype d = (other is dtype) ? (dtype)other : NpyDescr.DescrConverter(cntx, other);
            return NpyCoreApi.CanCastTo(d, this);
        }

        public object __gt__(CodeContext cntx, object other) {
            dtype d = (other is dtype) ? (dtype)other : NpyDescr.DescrConverter(cntx, other);
            return NpyCoreApi.EquivTypes(this, d) && NpyCoreApi.CanCastTo(d, this);
        }


        #endregion

        #endregion


        #region Internal data & methods

        private string AppendDateTimeTypestr(string str) {
            // TODO: Fix date time type string. See descriptor.c: _append_to_datetime_typestr
            throw new NotImplementedException("to do ");
        }


        private PythonDictionary GetFieldsDict() {
            PythonDictionary ret;

            if (!HasNames) {
                ret = null;
            } else {
                IntPtr iter = IntPtr.Zero;
                IntPtr dict = Marshal.ReadIntPtr(this.core, NpyCoreApi.DescrOffsets.off_fields);
                ret = new PythonDictionary();
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
        [NonSerialized]
        private readonly ArrFuncs funcs;

        #endregion

        #region Scalar type support

        [Serializable]
        internal class ScalarInfo {
            internal Type ScalarType;
            [NonSerialized]
            internal Func<ScalarGeneric> ScalarConstructor;

            internal static ScalarInfo Make<T>() where T: ScalarGeneric, new() {
                return new ScalarInfo { ScalarType = typeof(T), ScalarConstructor = (() => new T()) };
            }
        };

        internal ScalarInfo scalarInfo = null;

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

        internal object ToScalar(IntPtr dataPtr, int size) {
            if (ScalarType == null) {
                throw new ArgumentException("Attempt to construct scalar from non-scalar type");
            }

            ScalarGeneric result = scalarInfo.ScalarConstructor();
            return result.FillData(dataPtr, size);
        }

        #endregion
    }

    internal class dtype_Enumerator : IEnumerator<object>
    {
        public dtype_Enumerator(dtype a) {
            descr = a;
            index = -1;
        }

        public object Current {
            get { return descr[index]; }
        }

        public void Dispose() {
            descr = null;
        }


        public bool MoveNext() {
            index += 1;
            return (index < descr.Length);
        }

        public void Reset() {
            index = -1;
        }

        private dtype descr;
        private int index;
    }

}
