using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using IronPython.Runtime;
using IronPython.Runtime.Operations;

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
                NpyCoreApi.FillWithObject(this, scalar);
            } else {
                ndarray zero_d_array = NpyArray.FromAny(scalar, dtype, flags: NpyDefs.NPY_ALIGNED);
                NpyCoreApi.FillWithScalar(this, zero_d_array);
            }
        }

        internal void ToFile(CodeContext cntx, PythonFile file, string sep = null, string format = null) {
            if (sep == null || sep.Length == 0) {
                // Write as a binary file
                // TODO: Avoid converting to string.e
                Bytes data = ToString();
                file.write(data.ToString());
            } else {
                bool hasFormat = (format != null && format.Length > 0);
                flatiter it = Flat;
                long n = it.Length;
                while (it.MoveNext()) {
                    string s;
                    if (hasFormat) {
                        s = PythonOps.FormatString(cntx, format, it.Current);
                    } else {
                        s = PythonOps.ToString(it.Current);
                    }
                    file.write(s);
                    if (--n > 0) {
                        file.write(sep);
                    }
                }
            }
        }
    }
}
