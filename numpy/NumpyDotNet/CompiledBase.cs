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

            ndarray arrMask = NpyArray.CheckFromArray(mask, null, 0, 0, NpyDefs.NPY_CARRAY, cntx);
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
    }
}
