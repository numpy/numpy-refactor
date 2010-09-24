using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using IronPython.Runtime;

namespace NumpyDotNet
{
    public partial class ndarray
    {
        internal Bytes ToString(NpyDefs.NPY_ORDER order = NpyDefs.NPY_ORDER.NPY_ANYORDER) {
            long nbytes = Size * dtype.ElementSize;
            byte[] data = new byte[nbytes];
            NpyCoreApi.GetBytes(this, data, order);
            return new Bytes(data);
        }
    }
}
