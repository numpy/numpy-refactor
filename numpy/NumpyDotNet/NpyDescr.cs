using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using IronPython.Runtime;
using IronPython.Runtime.Types;
using IronPython.Modules;
using Microsoft.Scripting;

namespace NumpyDotNet {
    /// <summary>
    /// Implements the descriptor (NpyArray_Descr) functionality.  This is not the
    /// public wrapper but a collection of funtionality to support the dtype class.
    /// </summary>
    internal class NpyDescr {

        internal static dtype DescrConverter(Object obj) {
            dtype result;

            if (obj is dtype) result = (dtype)obj;
            else if (obj is IronPython.Runtime.Types.PythonType) {
                result = ConvertFromPythonType((IronPython.Runtime.Types.PythonType)obj);
            } else if (obj is string) {
                string s = (string)obj;
                if (CheckForDatetime(s)) {
                    result = ConvertFromDatetime(s);
                } else if (CheckForCommaString(s)) {
                    result = ConvertFromCommaString(s);
                } else {
                    // Try decoding a type string.
                    throw new NotImplementedException(
                        String.Format("Conversion of strings ({0}) to dtype is not yet implemented.", s));
                }
            } else {
                throw new NotImplementedException(
                    String.Format("Convertion of type '{0}' to type descriptor is not supported.",
                    obj.GetType().Name));
            }
            return result;
        }


        /// <summary>
        /// Converts a Python type into a descriptor object
        /// </summary>
        /// <param name="t">Python type object</param>
        /// <returns>Corresponding descriptor object</returns>
        private static dtype ConvertFromPythonType(IronPython.Runtime.Types.PythonType t) {
            Console.WriteLine("Type name = {0}", 
                IronPython.Runtime.Types.PythonType.Get__name__(t));

            NpyCoreApi.NPY_TYPES type;
            if (t == PyInt_Type) type = NpyCoreApi.NPY_TYPES.NPY_INT;
            else if (t == PyLong_Type) type = NpyCoreApi.NPY_TYPES.NPY_LONG;
            else if (t == PyFloat_Type) type = NpyCoreApi.NPY_TYPES.NPY_FLOAT;
            else if (t == PyDouble_Type) type = NpyCoreApi.NPY_TYPES.NPY_DOUBLE;
            else if (t == PyBool_Type) type = NpyCoreApi.NPY_TYPES.NPY_BOOL;
            else if (t == PyComplex_Type) type = NpyCoreApi.NPY_TYPES.NPY_CDOUBLE;
            else if (t == PyUnicode_Type) type = NpyCoreApi.NPY_TYPES.NPY_UNICODE;
            else type = NpyCoreApi.NPY_TYPES.NPY_NOTYPE;

            return (type != NpyCoreApi.NPY_TYPES.NPY_NOTYPE) ? NpyCoreApi.DescrFromType(type) : null;
        }


        private static bool CheckForDatetime(String s) {
            // TODO: Conversion from datetime strings is not implemented.
            return false;
        }

        private static dtype ConvertFromDatetime(String s) {
            throw new NotImplementedException();
        }

        private static bool CheckForCommaString(String s) {
            // TODO: Conversion from comma strings is not implemented.
            return false;
        }

        private static dtype ConvertFromCommaString(String s) {
            throw new NotImplementedException();
        }

        
        private static readonly PythonType PyInt_Type = DynamicHelpers.GetPythonTypeFromType(typeof(int));
        private static readonly PythonType PyLong_Type = DynamicHelpers.GetPythonTypeFromType(typeof(long));
        private static readonly PythonType PyFloat_Type = DynamicHelpers.GetPythonTypeFromType(typeof(float));
        private static readonly PythonType PyDouble_Type = DynamicHelpers.GetPythonTypeFromType(typeof(double));
        private static readonly PythonType PyBool_Type = DynamicHelpers.GetPythonTypeFromType(typeof(bool));
        private static readonly PythonType PyUnicode_Type = DynamicHelpers.GetPythonTypeFromType(typeof(string));
        private static readonly PythonType PyComplex_Type = DynamicHelpers.GetPythonTypeFromType(typeof(System.Numerics.Complex));
    }
}
