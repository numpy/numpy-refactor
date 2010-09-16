using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using IronPython.Runtime;
using IronPython.Modules;
using Microsoft.Scripting;
using NumpyDotNet;

namespace NumpyDotNet {
    class ufunc : IDisposable
    {
        /// <summary>
        /// Initializes the umath module
        /// </summary>
        static ufunc() {
            
        }


        private static String[] ufuncArgNames = { "extobj", "sig" };




/*        public ufunc(CodeContext cntx, [ParamDictionary] IAttributesCollection kwargs)
        {
            Object[] posArgs = { };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ufuncArgNames, kwargs);

            Construct(cntx, args);
        }

        public ufunc(CodeContext cntx, Object a1, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ufuncArgNames, kwargs);

            Construct(cntx, args);
        }

        public ufunc(CodeContext cntx, Object a1, Object a2, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1, a2 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ufuncArgNames, kwargs);

            Construct(cntx, args);
        } */


        internal ufunc(IntPtr corePtr) {
            core = corePtr;

            // The core object comes with a reference so we need to set the interface
            // pointer and then discard the core reference, leaving just this instance
            // as the sole reference to it.
            IntPtr offset = Marshal.OffsetOf(typeof(NpyCoreApi.NpyObject_HEAD), "nob_interface");
            Marshal.WriteIntPtr(corePtr, (int)offset,
                GCHandle.ToIntPtr(GCHandle.Alloc(this, GCHandleType.Weak)));
            NpyCoreApi.Decref(corePtr);
        }

        /// <summary>
        /// Arguments are: object, dtype, copy, order, subok
        /// </summary>
        /// <param name="cntx"></param>
        /// <param name="args"></param>
        private void Construct(CodeContext cntx, Object[] args) {
            Object dims = args[0];
            dtype type = (dtype)args[1];

            core = IntPtr.Zero;
        }


        ~ufunc()
        {
            Dispose(false);
        }

        protected void Dispose(bool disposing)
        {
            if (core != IntPtr.Zero)
            {
                lock (this) {
                    IntPtr a = core;
                    core = IntPtr.Zero;
                    NpyCoreApi.npy_ufunc_dealloc(a);
                }
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }


        #region Python interface

        public string __repr__() {
            return String.Format("<ufunc '{0}'>", __name__());
        }

        public string __str__() {
            // TODO: Unimplemented
            return "str";
        }

        public int nin {
            get {
                CheckValid();
                return Marshal.ReadInt32(core, NpyCoreApi.UFuncOffsets.off_nin);
            }
        }

        public int nout {
            get {
                CheckValid();
                return Marshal.ReadInt32(core, NpyCoreApi.UFuncOffsets.off_nout); 
            }
        }

        public int nargs {
            get {
                CheckValid();
                return Marshal.ReadInt32(core, NpyCoreApi.UFuncOffsets.off_nargs);
            }
        }

        public int ntypes {
            get {
                CheckValid();
                return Marshal.ReadInt32(core, NpyCoreApi.UFuncOffsets.off_ntypes);
            }
        }

        // TODO: Implement 'types'

        public string __name__() {
            CheckValid();
            IntPtr strPtr = Marshal.ReadIntPtr(core, NpyCoreApi.UFuncOffsets.off_name);
            return Marshal.PtrToStringAuto(strPtr);
        }

        // TODO: Implement 'identity'
        public string signature() {
            CheckValid();
            IntPtr strPtr = Marshal.ReadIntPtr(core, NpyCoreApi.UFuncOffsets.off_core_signature);
            return Marshal.PtrToStringAuto(strPtr);
        }


        #endregion

        /// <summary>
        /// Simply checks to verify that the object was correctly initialized and hasn't
        /// already been disposed before we go accessing native memory.
        /// </summary>
        private void CheckValid() {
            if (core == IntPtr.Zero)
                throw new InvalidOperationException("UFunc object is invalid or already disposed.");
        }

        private IntPtr core;
    }
}
