using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using IronPython.Runtime;

namespace NumpyDotNet
{
    /// <summary>
    /// A multi-array iterator.
    /// </summary>
    public class broadcast : Wrapper
    {
        public broadcast(params object[] args) {
            // Convert args to arrays.
            ndarray[] arrays = new ndarray[args.Length];
            for (int i = 0; i < args.Length; i++) {
                arrays[i] = NpyArray.FromAny(args[i], null, 0, 0, 0, null);
            }
            try {
                BeingCreated = this;
                core = NpyCoreApi.MultiIterFromArrays(arrays);
            } finally {
                BeingCreated = null;
            }
        }

        [ThreadStatic]
        internal static broadcast BeingCreated;

        internal IntPtr Iter {
            get {
                return core;
            }
        }

        public int numiter {
            get {
                return Marshal.ReadInt32(core + NpyCoreApi.MultiIterOffsets.off_numiter);
            }
        }

        public long size {
            get {
                return Marshal.ReadIntPtr(core + NpyCoreApi.MultiIterOffsets.off_size).ToInt64();
            }
        }

        public long index {
            get {
                return Marshal.ReadIntPtr(core + NpyCoreApi.MultiIterOffsets.off_index).ToInt64();
            }
        }

        public int nd {
            get {
                return Marshal.ReadInt32(core + NpyCoreApi.MultiIterOffsets.off_nd);
            }
        }

        public PythonTuple shape {
            get {
                int ndim = nd;
                long[] result = new long[ndim];
                IntPtr dims = core + NpyCoreApi.MultiIterOffsets.off_dimensions;
                for (int i = 0; i < ndim; i++) {
                    result[i] = Marshal.ReadIntPtr(dims).ToInt64();
                    dims += IntPtr.Size;
                }
                return new PythonTuple(result);
            }
        }

        public PythonTuple iters {
            get {
                int n = numiter;
                flatiter[] result = new flatiter[n];
                IntPtr iters = core + NpyCoreApi.MultiIterOffsets.off_iters;
                for (int i=0; i<n; i++) {
                    result[i] = NpyCoreApi.ToInterface<flatiter>(Marshal.ReadIntPtr(iters));
                    iters += IntPtr.Size;
                }
                return new PythonTuple(result);
            }
        }
            
    }
}
