using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Scripting;
using IronPython.Runtime.Operations;

namespace NumpyDotNet
{
    public partial class ndarray 
    {
        internal ndarray ArgMax(int axis, ndarray ret) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_ArgMax(Array, axis, (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal ndarray ArgMin(int axis, ndarray ret) {
            object obj;
            if (NpyDefs.IsFlexible(dtype.TypeNum)) {
                throw new ArgumentTypeException("argmin is unuspporeted for this type.");
            } else if (NpyDefs.IsUnsigned(dtype.TypeNum)) {
                obj = -1;
            } else if (NpyDefs.IsBool(dtype.TypeNum)) {
                obj = 1;
            } else {
                obj = 0;
            }
            ndarray tmp = NpyArray.FromAny(obj, dtype, 0, 0, 0, null);
            tmp = tmp - this;
            return tmp.ArgMax(axis, ret);
        }
    }
}
