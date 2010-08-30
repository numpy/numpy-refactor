using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using IronPython.Runtime;
using IronPython.Modules;
using Microsoft.Scripting;

namespace NumpyDotNet {
    public static class ModuleMethods {
        private static String[] arrayKwds = { "object", "dtype", "copy", "order", "subok", "ndmin" };

        /// <summary>
        /// Module method 'array': constructs a new array from an input object and
        /// optional type and other arguments.
        /// </summary>
        /// <param name="cntx"></param>
        /// <param name="o">Source object</param>
        /// <param name="kwargs">Optional named args</param>
        /// <returns>New array object</returns>
        public static ndarray array(CodeContext cntx, Object o, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] args = { o };
            return array_fromobject(NpyUtil_ArgProcessing.BuildArgsArray(args, arrayKwds, kwargs));
        }

        /// <summary>
        /// Module method 'array': constructs a new array from an input object and
        /// optional type and other arguments.
        /// </summary>
        /// <param name="cntx"></param>
        /// <param name="o">Source object</param>
        /// <param name="descr">The type descriptor object</param>
        /// <param name="kwargs">Optional named args</param>
        /// <returns>New array object</returns>
        public static ndarray array(CodeContext cntx, Object o, Object descr, 
            [ParamDictionary] IAttributesCollection kwargs) {
            Object[] args = { o, descr };
            return array_fromobject(NpyUtil_ArgProcessing.BuildArgsArray(args, arrayKwds, kwargs));
        }



        internal static ndarray array_fromobject(Object[] args) {
            Object src = args[0];
            dtype type = null;
            bool copy = true;
            NpyArray.NPY_ORDER order = NpyArray.NPY_ORDER.NPY_ANYORDER;
            bool subok = false;
            int ndmin = 0;
            ndarray result = null;

            if (src == null) {
                throw new IronPython.Runtime.Exceptions.RuntimeException(
                    "Object can not be null/none.");
            }

            if (args[1] != null) type = (dtype)args[1];
            if (args[2] != null) copy = NpyUtil_ArgProcessing.BoolConverter(args[2]);
            if (args[3] != null) order = (NpyArray.NPY_ORDER)args[3];   // TODO: Order converter here
            if (args[4] != null) subok = NpyUtil_ArgProcessing.BoolConverter(args[4]);
            if (args[5] != null) ndmin = NpyUtil_ArgProcessing.IntConverter(args[5]);

            if (ndmin >= NpyArray.NPY_MAXDIMS) {
                throw new IronPython.Runtime.Exceptions.RuntimeException(
                    String.Format("ndmin ({0} bigger than allowable number of dimension ({1}).", 
                    ndmin, NpyArray.NPY_MAXDIMS-1));
            }

            // TODO: Check that the first is equiv to PyArray_Check() and the
            // second is equiv to PyArray_CheckExact().
            if (subok && src is ndarray ||
                !subok && src.GetType() == typeof(ndarray)) {
                ndarray arr = (ndarray)src;
                if (type == null) {
                    if (!copy && arr.StridingOk(order)) {
                        result = arr;
                    } else {
                        result = NpyArray.NewCopy(arr, order);
                    }
                } else {
                    result = null;
                }
            }
            return null;
        }

    }
}
