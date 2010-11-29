using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using IronPython.Runtime;
using IronPython.Runtime.Operations;

namespace NumpyDotNet
{
    /// <summary>
    /// A multi-array iterator.
    /// </summary>
    public class broadcast : Wrapper, IEnumerator<PythonTuple>
    {
        public broadcast(params object[] args) {
            try {
                BeingCreated = this;
                core = NpyCoreApi.MultiIterFromObjects(args);
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

        public long[] dims {
            get {
                return NpyCoreApi.GetArrayDims(this, true);
            }
        }

        public object size {
            get {
                return Marshal.ReadIntPtr(core + NpyCoreApi.MultiIterOffsets.off_size).ToPython();
            }
        }

        public object index {
            get {
                return Marshal.ReadIntPtr(core + NpyCoreApi.MultiIterOffsets.off_index).ToPython();
            }
        }

        private long getIndex() {
            return Marshal.ReadIntPtr(core + NpyCoreApi.MultiIterOffsets.off_index).ToInt64();
        }

        private void setIndex(long value) {
            Marshal.WriteIntPtr(core + NpyCoreApi.MultiIterOffsets.off_index, (IntPtr)value);
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



        private flatiter iter(int i) {
            IntPtr iter = core + NpyCoreApi.MultiIterOffsets.off_iters + i * IntPtr.Size;
            return NpyCoreApi.ToInterface<flatiter>(Marshal.ReadIntPtr(iter));
        }

        public void reset() {
            Reset();
        }

        public PythonTuple Current {
            get {
                int n = numiter;
                object[] result = new object[n];
                for (int i = 0; i < n; i++) {
                    result[i] = iter(i).Current;
                }
                return new PythonTuple(result);
            }
        }

        object System.Collections.IEnumerator.Current {
            get {
                int n = numiter;
                object[] result = new object[n];
                for (int i = 0; i < n; i++) {
                    result[i] = iter(i).Current;
                }
                return new PythonTuple(result);
            }
        }

        public bool MoveNext() {
            bool result = false;
            int n = numiter;
            setIndex(getIndex() + 1);
            for (int i = 0; i < n; i++) {
                result = iter(i).MoveNext();
            }
            return result;
        }

        public void Reset() {
            int n = numiter;
            setIndex(-1);
            for (int i = 0; i < n; i++) {
                iter(i).Reset();
            }
        }
    }
}
