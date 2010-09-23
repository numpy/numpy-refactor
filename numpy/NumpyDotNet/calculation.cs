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
            tmp = NpyArray.FromAny(tmp - this, null, 0, 0, 0, null);
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

        internal object Ptp(int axis, ndarray ret) {
            ndarray arr = NpyCoreApi.CheckAxis(this, ref axis, 0);
            ndarray a1 = arr.Max(axis, ret);
            ndarray a2 = arr.Min(axis, null);
            ufunc subtract = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_subtract);
            return NpyCoreApi.GenericBinaryOp(a1, a2, subtract, ret);
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

        internal object Mean(int axis, dtype rtype, ndarray ret) {
            ndarray newArray = NpyCoreApi.CheckAxis(this, ref axis, 0);
            ndarray sum = newArray.Sum(axis, rtype, ret);
            ndarray denom = NpyArray.FromAny(newArray.Dims[axis], NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_DOUBLE),
                0, 0, 0, null);
            ufunc divide = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_divide);
            return NpyCoreApi.GenericBinaryOp(sum, denom, divide, ret);
        }

        private static readonly double[] p10 = new double[] { 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9 };

        private static double PowerOfTen(int n) {
            double ret;
            if (n < p10.Length) {
                ret = p10[n];
            } else {
                int start = p10.Length - 1;
                ret = p10[start];
                while (n-- > start) {
                    ret *= 10;
                }
            }
            return ret;
        }

        internal object Round(int decimals, ndarray ret) {
            // For complex just round both parts.
            if (IsComplex) {
                if (ret == null) {
                    ret = copy();
                }
                Real.Round(decimals, ret.Real);
                Imag.Round(decimals, ret.Imag);
                return ret;
            }

            if (decimals >= 0 && IsInteger) {
                // There is nothing to do for integers.
                if (ret != null) {
                    NpyCoreApi.CopyAnyInto(ret, this);
                    return ret;
                } else {
                    return this;
                }
            }
            
            ufunc round_op = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_rint);
            if (decimals == 0) {
                // This is just a ufunc
                return NpyCoreApi.GenericUnaryOp(this, round_op, ret);
            }

            // Set up to do a multiply, round, divide, or the other way around.
            ufunc pre;
            ufunc post;
            if (decimals >= 0) {
                pre = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_multiply);
                post = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_divide);
            } else {
                pre = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_divide);
                post = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_multiply);
                decimals = -decimals;
            }
            ndarray factor = NpyArray.FromAny(PowerOfTen(decimals));

            // Make a temporary array, if we need it.
            NpyDefs.NPY_TYPES tmpType = NpyDefs.NPY_TYPES.NPY_DOUBLE;
            if (!IsInteger) {
                tmpType = dtype.TypeNum;
            }
            ndarray tmp;
            if (ret != null && ret.dtype.TypeNum == tmpType) {
                tmp = ret;
            } else {
                tmp = NpyCoreApi.NewFromDescr(NpyCoreApi.DescrFromType(tmpType), Dims, null, 0, null);
            }

            // Do the work
            NpyCoreApi.GenericBinaryOp(this, factor, pre, tmp);
            NpyCoreApi.GenericUnaryOp(tmp, round_op, tmp);
            if (!IsInteger || ret != null) {
                return NpyCoreApi.GenericBinaryOp(tmp, factor, post, ret);
            } else {
                // We need to convert to the integer type
                ret = NpyCoreApi.NewFromDescr(dtype, Dims, null, 0, null);
                return NpyCoreApi.GenericBinaryOp(tmp, factor, post, ret);
            }
        }

        internal ndarray All(int axis, ndarray ret) {
             return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_All(Array, axis, (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal ndarray Any(int axis, ndarray ret) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Any(Array, axis, (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal ndarray Conjugate(ndarray ret) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Conjugate(Array, (ret == null ? IntPtr.Zero : ret.Array)));
        }
    }
}
