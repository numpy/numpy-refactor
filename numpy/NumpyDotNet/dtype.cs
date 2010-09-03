using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using IronPython.Runtime;
using IronPython.Modules;
using Microsoft.Scripting;
using NumpyDotNet;

namespace NumpyDotNet {
    public class dtype {
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
        }


        /// <summary>
        /// Constructs a new NpyArray_Descr objet matching the passed one.
        /// Equivalent to NpyAray_DescrNew.
        /// </summary>
        /// <param name="d">Descriptor to duplicate</param>
        public dtype(dtype d) {
            descr = NpyCoreApi.NpyArray_DescrNew(this.descr);
        }
        
        
        /// <summary>
        /// Creates a wrapper for an array created on the native side, such as 
        /// the result of a slice operation.
        /// </summary>
        /// <param name="d">Pointer to core NpyArray_Descr structure</param>
        internal dtype(IntPtr d) {
            descr = d;
        }

        ~dtype() {
            Dispose(false);
        }

        protected void Dispose(bool disposing) {
            if (descr != IntPtr.Zero) {
                IntPtr a = descr;
                descr = IntPtr.Zero;
                //SimpleArray_delete(a);
                //PythonStub.CheckError();
            }
        }

        public void Dispose() {
            Dispose(true);
            GC.SuppressFinalize(this);
        }


        #region Properties

        public IntPtr Descr {
            get { return descr; }
        }

        public int Byteorder {
            get { return 0; }
            set { }
        }

        public bool IsNativeByteOrder {
            get { return Byteorder != NpyCoreApi.NativeByteOrder; }
        }

        public byte Kind {
            get {
                return Marshal.ReadByte(descr, NpyCoreApi.DescrOffsets.off_kind);
            }
        }

        public NpyCoreApi.NPY_TYPECHAR Type {
            get { return (NpyCoreApi.NPY_TYPECHAR)Marshal.ReadByte(descr, NpyCoreApi.DescrOffsets.off_type); }
        }

        public byte ByteOrder {
            get { return Marshal.ReadByte(descr, NpyCoreApi.DescrOffsets.off_byteorder); }
        }

        public int Flags {
            get { return Marshal.ReadInt32(descr, NpyCoreApi.DescrOffsets.off_flags); }
        }

        public NpyCoreApi.NPY_TYPES TypeNum {
            get { return (NpyCoreApi.NPY_TYPES)Marshal.ReadInt32(descr, NpyCoreApi.DescrOffsets.off_type_num); }
        }

        public int ElementSize {
            get { return Marshal.ReadInt32(descr, NpyCoreApi.DescrOffsets.off_elsize); }
        }

        public int Alignment {
            get { return Marshal.ReadInt32(descr, NpyCoreApi.DescrOffsets.off_alignment); }
        }

        public bool HasNames {
            get { return Marshal.ReadIntPtr(descr, NpyCoreApi.DescrOffsets.off_names) == IntPtr.Zero; }
        }

        public bool HasSubarray {
            get { return Marshal.ReadIntPtr(descr, NpyCoreApi.DescrOffsets.off_subarray) == IntPtr.Zero; }
        }

        #endregion


        #region Comparison
        public override bool Equals(object obj) {
            if (obj != null && obj is dtype) return Equals((dtype)obj);
            return false;
        }

        public bool Equals(dtype other) {
            if (other == null) return false;
            return (this.descr == other.descr ||
                    NpyCoreApi.NpyArray_EquivTypes(descr, other.descr) != 0);
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
                (object)t1 != null && t1.Equals(t2);
        }

        public static bool operator !=(dtype t1, dtype t2) {
            return !System.Object.ReferenceEquals(t1, t2) ||
                (object)t1 != null && !t1.Equals(t2);
        }

        public override int GetHashCode() {
            return (int)descr;
        }
        #endregion


        #region Internal data
        private static PythonContext pyContext = null;

        /// <summary>
        ///  Pointer to the native object 
        /// </summary>
        private IntPtr descr;

        #endregion
    }
}
