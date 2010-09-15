using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

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

        private IntPtr core;


    }
}
