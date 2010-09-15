using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumpyDotNet
{
    /// <summary>
    /// A multi-array iterator.
    /// </summary>
    public class broadcast : IDisposable
    {
        public broadcast(params object[] args) {
            // Convert args to arrays.
            ndarray[] arrays = new ndarray[args.Length];
            for (int i = 0; i < args.Length; i++) {
                arrays[i] = NpyArray.FromAny(args[i], null, 0, 0, 0, null);
            }
            try {
                BeingCreated = this;
                iter = NpyCoreApi.MultiIterFromArrays(arrays);
            } finally {
                BeingCreated = null;
            }
        }

        [ThreadStatic]
        internal static broadcast BeingCreated;

        ~broadcast() {
            Dispose(false);
        }

        public void Dispose() {
            Dispose(true);
        }

        protected void Dispose(bool disposing) {
            lock (this) {
                IntPtr a = iter;
                iter = IntPtr.Zero;
                NpyCoreApi.Dealloc(a);
            }
        }

        internal IntPtr Iter {
            get {
                return iter;
            }
        }

        private IntPtr iter;


    }
}
