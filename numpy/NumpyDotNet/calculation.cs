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

        internal ndarray Max(int axis, ndarray ret) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Max(Array, axis, (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal ndarray Min(int axis, ndarray ret) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Min(Array, axis, (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal ndarray Sum(int axis, dtype rtype, ndarray ret) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Sum(Array, axis, 
                    (int)(rtype == null ? NpyDefs.NPY_TYPES.NPY_NOTYPE : rtype.TypeNum), 
                    (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal ndarray Prod(int axis, dtype rtype, ndarray ret) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Prod(Array, axis, 
                    (int)(rtype == null ? NpyDefs.NPY_TYPES.NPY_NOTYPE : rtype.TypeNum), 
                    (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal ndarray CumSum(int axis, dtype rtype, ndarray ret) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_CumSum(Array, axis, 
                    (int)(rtype == null ? NpyDefs.NPY_TYPES.NPY_NOTYPE : rtype.TypeNum), 
                    (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal ndarray CumProd(int axis, dtype rtype, ndarray ret) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_CumProd(Array, axis, 
                    (int)(rtype == null ? NpyDefs.NPY_TYPES.NPY_NOTYPE : rtype.TypeNum), 
                    (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal ndarray Mean(int axis, dtype rtype, ndarray ret) {
            ndarray newArray = NpyCoreApi.CheckAxis(this, ref axis, 0);
            ndarray sum = newArray.Sum(axis, rtype, ret);
            ndarray denom = NpyArray.FromAny(newArray.Dims[axis], NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_DOUBLE),
                0, 0, 0, null);
            ufunc divide = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_divide);
            return NpyCoreApi.GenericBinaryOp(sum, denom, divide, ret);
        }


        internal ndarray All(int axis, ndarray ret) {
             return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_All(Array, axis, (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal ndarray Any(int axis, ndarray ret) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Any(Array, axis, (ret == null ? IntPtr.Zero : ret.Array)));
        }
    }
}
