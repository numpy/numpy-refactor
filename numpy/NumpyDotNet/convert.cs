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

        internal void FillWithScalar(object scalar) {
            if (dtype.IsObject) {
                // TODO: Fix this.
                throw new NotImplementedException("Can't fill object arrays.");
            }
            ndarray zero_d_array = NpyArray.FromAny(scalar, dtype, flags: NpyDefs.NPY_ALIGNED);
            NpyCoreApi.FillWithScalar(this, zero_d_array);
        }
    }
}
