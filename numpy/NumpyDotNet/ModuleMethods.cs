using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using IronPython.Runtime;
using IronPython.Modules;
using Microsoft.Scripting;

namespace NumpyDotNet {
    /// <summary>
    /// ModuleMethods implements the module-level numpy functions.
    /// </summary>
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
            return arrayFromObject(cntx.LanguageContext,
                NpyUtil_ArgProcessing.BuildArgsArray(args, arrayKwds, kwargs));
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
            return arrayFromObject(cntx.LanguageContext,
                NpyUtil_ArgProcessing.BuildArgsArray(args, arrayKwds, kwargs));
        }



        internal static ndarray arrayFromObject(PythonContext cntx, Object[] args) {
            Object src = args[0];
            dtype type = null;
            bool copy = true;
            NpyDefs.NPY_ORDER order = NpyDefs.NPY_ORDER.NPY_ANYORDER;
            bool subok = false;
            int ndmin = 0;
            ndarray result = null;

            try {
                if (src == null) {
                    throw new IronPython.Runtime.Exceptions.RuntimeException(
                        "Object can not be null/none.");
                }

                if (args[1] != null) type = NpyDescr.DescrConverter(cntx, args[1]);
                if (args[2] != null) copy = NpyUtil_ArgProcessing.BoolConverter(args[2]);

                if (args[3] != null && args[3] is string &&
                    String.Compare((String)args[3], "Fortran", true) == 0)
                    order = NpyDefs.NPY_ORDER.NPY_FORTRANORDER;
                else order = NpyDefs.NPY_ORDER.NPY_CORDER;

                if (args[4] != null) subok = NpyUtil_ArgProcessing.BoolConverter(args[4]);
                if (args[5] != null) ndmin = NpyUtil_ArgProcessing.IntConverter(args[5]);

                if (ndmin >= NpyDefs.NPY_MAXDIMS) {
                    throw new IronPython.Runtime.Exceptions.RuntimeException(
                        String.Format("ndmin ({0} bigger than allowable number of dimension ({1}).",
                        ndmin, NpyDefs.NPY_MAXDIMS - 1));
                }

                Console.WriteLine("copy = {0}, order = {1}, subok = {2}, ndmin = {3}",
                    copy, order, subok, ndmin);
                Console.WriteLine("magic_offset = {0}, descr offset = {1}",
                    NpyCoreApi.ArrayOffsets.off_magic_number, NpyCoreApi.ArrayOffsets.off_descr);

                // TODO: Check that the first is equiv to PyArray_Check() and the
                // second is equiv to PyArray_CheckExact().
                if (subok && src is ndarray ||
                    !subok && src.GetType() == typeof(ndarray)) {
                    ndarray arr = (ndarray)src;
                    if (type == null) {
                        if (!copy && arr.StridingOk(order)) {
                            result = arr;
                        } else {
                            result = NpyCoreApi.NewCopy(arr, order);
                        }
                    } else {
                        dtype oldType = arr.dtype;
                        if (oldType == type) {
                            result = arr;
                        } else {
                            result = NpyCoreApi.NewCopy(arr, order);
                            if (oldType != type) {
                                arr.dtype = oldType;
                            }
                        }
                    }
                }

                // If no result has been determined...
                if (result == null) {
                    int flags = 0;

                    if (copy) flags = NpyDefs.NPY_ENSURECOPY;
                    if (order == NpyDefs.NPY_ORDER.NPY_CORDER) {
                        flags |= NpyDefs.NPY_CONTIGUOUS;
                    } else if (order == NpyDefs.NPY_ORDER.NPY_FORTRANORDER ||
                             src is ndarray && ((ndarray)src).IsFortran) {
                                 flags |= NpyDefs.NPY_FORTRAN;
                    }

                    if (!subok) flags |= NpyDefs.NPY_ENSUREARRAY;

                    flags |= NpyDefs.NPY_FORCECAST;
                    result = NpyArray.CheckFromArray(src, type, 0, 0, flags, null);
                }

                if (result != null || result.ndim < ndmin) {
                    result = NpyArray.PrependOnes(result, result.ndim, ndmin);
                }
            } catch (Exception e) {
                Console.WriteLine("Stack trace: {0}\n{1}", e.Message, e.StackTrace);
                throw e;
            }
            return result;
        }
    }
}
