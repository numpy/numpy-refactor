using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Reflection;
using IronPython.Runtime;
using IronPython.Runtime.Operations;
using IronPython.Runtime.Types;
using IronPython.Modules;
using Microsoft.Scripting;

[assembly: PythonModule("_compiled_base", typeof(NumpyDotNet.CompiledBase))]
namespace NumpyDotNet {
    /// <summary>
    /// ModuleMethods implements the module-level numpy functions.
    /// </summary>
    public static class CompiledBase {
        /// <summary>
        /// Checks to see if 'mask' is either the same shape as input or a 1-d array
        /// with the same number of elements as input.
        /// </summary>
        /// <param name="mask">Array to check vs input</param>
        /// <param name="input">Base array to test against</param>
        /// <returns>True if mask is same shape or 1-d</returns>
        private static bool SameShape(ndarray mask, ndarray input) {
            if (mask.ndim == input.ndim) {
                return mask.Dims.SequenceEqual(input.Dims);
            } else {
                return mask.ndim == 1 && mask.Size == input.Size;
            }
        }


        /// <summary>
        /// Inserts values from sequene 'vals' into 'input' wherever 'mask' is true.
        /// </summary>
        /// <param name="cntx">Current code context (IronPython)</param>
        /// <param name="input">Input array to modify</param>
        /// <param name="mask">Mask value (same size or shape as input)</param>
        /// <param name="vals">Values to be inserted</param>
        public static object _insert(CodeContext cntx, object input, object mask, object vals) {
            ndarray arrInp = NpyArray.FromAny(input, null, 0, 0, NpyDefs.NPY_CARRAY);

            ndarray arrMask = NpyArray.CheckFromAny(mask, null, 0, 0, NpyDefs.NPY_CARRAY, cntx);
            if (arrMask.Dtype.IsObject)
                arrMask = NpyCoreApi.CastToType(arrMask, NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_INT), false);

            if (!SameShape(arrMask, arrInp)) {
                throw new ArgumentTypeException("mask array must be 1-d or same shape as input array");
            }

            ndarray arrVals = NpyArray.FromAny(vals, arrInp.Dtype, 0, 1);

            flatiter inpIter = arrInp.Flat;
            flatiter maskIter = arrMask.Flat;
            flatiter valsIter = arrVals.Flat;
            while (inpIter.MoveNext() && maskIter.MoveNext()) {
                // TODO: Ugly way of testing for 0. No, comparing maskIter.Current == true doesn't work.
                if (NpyUtil_Python.ConvertToInt(maskIter.Current, cntx) != 0) {
                    if (!valsIter.MoveNext()) {
                        // If we go past the end, reset to the set.
                        valsIter = arrVals.Flat;
                        valsIter.MoveNext();
                    }
                    inpIter.Current = valsIter.Current;
                }
            }
            return arrInp;
        }


        /// <summary>
        /// bincount accepts one or two arguments. The first is an array of non-negative integers
        /// and the second, if present, is an array of weights, which must be promotable to double.
        /// Call these arguments list and weight.  Both must be one-dimensional with len(weight) ==
        /// len(list).  If weight is not present then bincount(list)[i] is the number of occurances
        /// of i in list.  If weight is present then bincount(self, list, weight)[i] is the sum of
        /// all weigh[j] where list[j] == i. 
        /// </summary>
        /// <param name="list">List of integers</param>
        /// <param name="wrights">Optional list of type promotable to double</param>
        /// <returns>Array of binds</returns>
        public static object bincount(object list, object weights=null) {
            ndarray listArr = NpyArray.FromAny(list, NpyCoreApi.DescrFromType(NpyCoreApi.TypeOf_Int64), 1, 1, NpyDefs.NPY_DEFAULT);
            int len =(int) listArr.Size;
            if (len < 1) throw new ArgumentException("The first argument cannot be empty.");

            int minIdx, maxIdx;
            findMinMax(listArr, out minIdx, out maxIdx);
            if (listArr.ReadAsInt64(minIdx) < 0) {
                throw new ArgumentException("The values in the first argument to bincount must be non-negative");
            }

            ndarray answer;
            long answerSize = listArr.ReadAsInt64(maxIdx)+1;
            if (weights == null) {
                answer = NpyArray.Zeros(new long[] { answerSize }, NpyCoreApi.DescrFromType(NpyCoreApi.TypeOf_Int64));
                for (long i=0; i < len; i++) {
                    long idx = (long)listArr.ReadAsInt64(i);
                    answer.WriteAsInt64(idx, answer.ReadAsInt64(idx) + 1);
                }
            } else {
                dtype type = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_DOUBLE);
                ndarray weightsArr = NpyArray.FromAny(weights, type, 1, 1, NpyDefs.NPY_DEFAULT);
                if (weightsArr.Size != len) {
                    throw new ArgumentException("The weights and list arguments don't have the same length.");
                }
                answer = NpyArray.Zeros(new long[] { answerSize }, NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_DOUBLE));
                for (long i=0; i < len; i++) {
                    long idx = (long)listArr.ReadAsIntPtr(i);
                    answer.WriteAsDouble(idx, answer.ReadAsDouble(idx) + weightsArr.ReadAsDouble(i));
                }
            }
            return answer;
        }

        private static void findMinMax(ndarray intArr, out int minIdx, out int maxIdx) {
            if (intArr.Dtype.TypeNum != NpyCoreApi.TypeOf_Int64) {
                throw new ArgumentTypeException(String.Format("Expected array of {0} type.", NpyCoreApi.TypeOf_Int64));
            }

            minIdx = maxIdx = 0;
            long minVal, maxVal;
            minVal = maxVal = intArr.ReadAsInt64(0);

            // Walk the array using direct memory access so we don't have to box/unbox every value
            // in the array.
            for (int i = 0; i < intArr.Size; i++) {
                long v = intArr.ReadAsInt64(i);
                if (v < minVal) {
                    minIdx = i;
                    minVal = v;
                } else if (v > maxVal) {
                    maxIdx = i;
                    maxVal = v;
                }
            }
        }
    }
}
