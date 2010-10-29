using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Scripting;
using IronPython.Runtime.Operations;
using IronPython.Runtime.Types;

namespace NumpyDotNet
{
    public partial class ndarray 
    {
        internal ndarray ArgMax(int axis, ndarray ret = null) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_ArgMax(Array, axis, (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal ndarray ArgMin(int axis, ndarray ret = null) {
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

        internal ndarray Max(int axis, ndarray ret = null) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Max(Array, axis, (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal ndarray Min(int axis, ndarray ret = null) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Min(Array, axis, (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal object Ptp(int axis, ndarray ret = null) {
            ndarray arr = NpyCoreApi.CheckAxis(this, ref axis, 0);
            ndarray a1 = arr.Max(axis, ret);
            ndarray a2 = arr.Min(axis, null);
            ufunc subtract = NpyCoreApi.GetNumericOp(NpyDefs.NpyArray_Ops.npy_op_subtract);
            return BinaryOp(null, a1, a2, NpyDefs.NpyArray_Ops.npy_op_subtract, ret);
        }

        internal object Std(int axis, dtype rtype, ndarray ret, bool variance, int ddof) {
            ndarray x, mean, tmp;
            object result;
            long n;
            IntPtr[] newshape;

            // Reshape and get axis
            x = NpyCoreApi.CheckAxis(this, ref axis, 0);
            // Compute the mean
            mean = NpyArray.FromAny(x.Mean(axis, rtype, null), rtype);
            // Add an axis back to the mean so it will broadcast correctly
            newshape = x.Dims.Select(y => (IntPtr)y).ToArray();
            newshape[axis] = (IntPtr)1;
            tmp = NpyCoreApi.Newshape(mean, newshape, NpyDefs.NPY_ORDER.NPY_CORDER);
            // Compute x - mean
            tmp = NpyArray.FromAny(x - tmp);
            // Square the difference
            if (tmp.IsComplex) {
                tmp = NpyArray.FromAny(tmp * tmp.Conjugate());
            } else {
                tmp = NpyArray.FromAny(tmp * tmp);
            }

            // Sum the square
            tmp = tmp.Sum(axis, rtype);

            // Divide by n (or n-ddof) and maybe take the sqrt
            n = x.Dims[axis] - ddof;
            if (n == 0) n = 1;
            if (!variance) {
                result = NpyArray.FromAny(tmp * (1.0 / n)).__sqrt__(null);
            } else {
                result = tmp * (1.0 / n);
            }

            // Deal with subclasses
            if (result is ndarray && result.GetType() != GetType()) {
                ndarray aresult;
                if (result.GetType() != typeof(ndarray)) {
                    aresult = NpyArray.FromAny(result, flags: NpyDefs.NPY_ENSUREARRAY);
                } else {
                    aresult = (ndarray)result;
                }
                if (GetType() != typeof(ndarray)) {
                    PythonType t = DynamicHelpers.GetPythonType(this);
                    aresult = NpyCoreApi.View(aresult, null, t);
                }
                result = aresult;
            }

            // Copy into ret, if necessary
            if (ret != null) {
                NpyCoreApi.CopyAnyInto(ret, NpyArray.FromAny(result));
                return ret;
            }
            return result;
        }

        internal ndarray Sum(int axis, dtype rtype, ndarray ret = null) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Sum(Array, axis, 
                    (int)(rtype == null ? NpyDefs.NPY_TYPES.NPY_NOTYPE : rtype.TypeNum), 
                    (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal ndarray Prod(int axis, dtype rtype, ndarray ret = null) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Prod(Array, axis, 
                    (int)(rtype == null ? NpyDefs.NPY_TYPES.NPY_NOTYPE : rtype.TypeNum), 
                    (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal ndarray CumSum(int axis, dtype rtype, ndarray ret = null) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_CumSum(Array, axis, 
                    (int)(rtype == null ? NpyDefs.NPY_TYPES.NPY_NOTYPE : rtype.TypeNum), 
                    (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal ndarray CumProd(int axis, dtype rtype, ndarray ret = null) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_CumProd(Array, axis, 
                    (int)(rtype == null ? NpyDefs.NPY_TYPES.NPY_NOTYPE : rtype.TypeNum), 
                    (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal object Mean(int axis, dtype rtype, ndarray ret = null) {
            ndarray newArray = NpyCoreApi.CheckAxis(this, ref axis, 0);
            ndarray sum = newArray.Sum(axis, rtype, ret);
            ndarray denom = NpyArray.FromAny(newArray.Dims[axis], NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_DOUBLE),
                0, 0, 0, null);
            return BinaryOp(null, sum, denom, NpyDefs.NpyArray_Ops.npy_op_divide, ret);
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

        internal object Round(int decimals, ndarray ret = null) {
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
            

            if (decimals == 0) {
                // This is just a ufunc
                return UnaryOp(null, this, NpyDefs.NpyArray_Ops.npy_op_rint, ret);
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
            BinaryOp(null, this, factor, pre, tmp);
            UnaryOp(null, tmp, NpyDefs.NpyArray_Ops.npy_op_rint, tmp);
            BinaryOp(null, tmp, factor, post, tmp);

            if (ret != null && tmp != ret) {
                NpyCoreApi.CopyAnyInto(ret, tmp);
                return ret;
            }
            return tmp;
        }

        internal ndarray All(int axis, ndarray ret = null) {
             return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_All(Array, axis, (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal ndarray Any(int axis, ndarray ret = null) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Any(Array, axis, (ret == null ? IntPtr.Zero : ret.Array)));
        }

        internal object Clip(object min, object max, ndarray ret = null) {
            // TODO: Add fast clipping
            if (min == null && max == null) {
                throw new ArgumentException("must set either max or min");
            }
            if (min == null) {
                return BinaryOp(null, this, max, NpyDefs.NpyArray_Ops.npy_op_minimum, ret);
            } else if (max == null) {
                return BinaryOp(null, this, min, NpyDefs.NpyArray_Ops.npy_op_maximum, ret);
            } else {
                object tmp = BinaryOp(null, this, max, NpyDefs.NpyArray_Ops.npy_op_minimum);
                return BinaryOp(null, tmp, min, NpyDefs.NpyArray_Ops.npy_op_maximum, ret);
            }
        }

        internal ndarray Conjugate(ndarray ret = null) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Conjugate(Array, (ret == null ? IntPtr.Zero : ret.Array)));
        }
    }
}
