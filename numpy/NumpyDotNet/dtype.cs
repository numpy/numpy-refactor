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
    public class dtype : Wrapper {
        public dtype(CodeContext cntx, [ParamDictionary] IAttributesCollection kwargs) {
            String[] unsupportedArgs = { };

            if (pyContext == null && cntx != null) pyContext = (PythonContext)cntx.LanguageContext;

            Dictionary<String, Object> y = kwargs
                .Select((k, v) => new KeyValuePair<String, Object>(k.ToString(), v))
                .ToDictionary((kvPair => kvPair.Key), (kvPair => kvPair.Value));

            foreach (String bad in unsupportedArgs) {
                if (y.ContainsKey(bad))
                    throw new NotImplementedException(String.Format("ndarray argument '%s' is not yet implemented.", bad));
            }
            funcs = NumericOps.FuncsForType(this.TypeNum);
        }


        /// <summary>
        /// Constructs a new NpyArray_Descr objet matching the passed one.
        /// Equivalent to NpyAray_DescrNew.
        /// </summary>
        /// <param name="d">Descriptor to duplicate</param>
        public dtype(dtype d) {
            core = NpyCoreApi.NpyArray_DescrNew(d.core);
            Console.WriteLine("Arg = {0}, {1}", this.Type, this.TypeNum);
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

        public PythonTuple names {
            get { return new PythonTuple(Names); }
        }


        public bool HasSubarray {
            get { return Marshal.ReadIntPtr(core, NpyCoreApi.DescrOffsets.off_subarray) == IntPtr.Zero; }
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


        #region Internal data
        private static PythonContext pyContext = null;

        /// <summary>
        /// Type-specific functions
        /// </summary>
        private readonly ArrFuncs funcs;

        #endregion

        #region Scalar type support

        private static bool ScalarTypesInited = false;
        private Type ScalarType_ = null;
        private Func<generic> ScalarConstructor = null;

        public Type ScalarType {
            get {
                if (!ScalarTypesInited) {
                    lock (typeof(dtype)) {
                        if (!ScalarTypesInited) {
                            InitScalarTypes();
                        }
                    }
                }
                return ScalarType_;
            }
        }

        public PythonType type {
            get {
                return DynamicHelpers.GetPythonTypeFromType(ScalarType);
            }
        }

        private static void InitScalarTypes() {
            RegisterScalarType<int8>();
            RegisterScalarType<int16>();
            RegisterScalarType<int32>();
            RegisterScalarType<int64>();
            RegisterScalarType<uint8>();
            RegisterScalarType<uint16>();
            RegisterScalarType<uint32>();
            RegisterScalarType<uint64>();
            RegisterScalarType<float32>();
            RegisterScalarType<float64>();
        }

        private static void RegisterScalarType<T>() 
            where T : generic, new()
        {
            T scalar = new T();
            dtype d = scalar.dtype;
            d.ScalarType_ = typeof(T);
            d.ScalarConstructor = (() => new T());
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
                generic result = ScalarConstructor();
                result.FillData(arr, offset);
                return result;
            }
        }

        #endregion
    }
}
