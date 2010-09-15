using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
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

        internal static dtype DescrConverter(PythonContext cntx, Object obj) {
            dtype result;

            if (obj is dtype) result = (dtype)obj;
            else if (obj is IronPython.Runtime.Types.PythonType) {
                result = ConvertFromPythonType((IronPython.Runtime.Types.PythonType)obj);
            } else if (obj is string) {
                string s = (string)obj;
                if (!String.IsNullOrEmpty(s) && CheckForDatetime(s)) {
                    result = ConvertFromDatetime(cntx, s);
                } else if (CheckForCommaString(s)) {
                    result = ConvertFromCommaString(s);
                } else {
                    result = ConvertBracketString(s);
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

            NpyDefs.NPY_TYPES type;
            if (t == PyInt_Type) type = NpyDefs.NPY_TYPES.NPY_INT;
            else if (t == PyLong_Type) type = NpyDefs.NPY_TYPES.NPY_LONG;
            else if (t == PyFloat_Type) type = NpyDefs.NPY_TYPES.NPY_FLOAT;
            else if (t == PyDouble_Type) type = NpyDefs.NPY_TYPES.NPY_DOUBLE;
            else if (t == PyBool_Type) type = NpyDefs.NPY_TYPES.NPY_BOOL;
            else if (t == PyComplex_Type) type = NpyDefs.NPY_TYPES.NPY_CDOUBLE;
            else if (t == PyUnicode_Type) type = NpyDefs.NPY_TYPES.NPY_UNICODE;
            else type = NpyDefs.NPY_TYPES.NPY_NOTYPE;

            return (type != NpyDefs.NPY_TYPES.NPY_NOTYPE) ? NpyCoreApi.DescrFromType(type) : null;
        }


        /// <summary>
        /// Checks to see if a given string matches any of date/time types.
        /// </summary>
        /// <param name="s">Type string</param>
        /// <returns>True if it's a date/time format, false if not</returns>
        private static bool CheckForDatetime(String s) {
            if (s.Length < 2) return false;
            if (s[1] == '8' && (s[0] == 'M' || s[0] == 'm')) return true;
            return s.StartsWith("datetime64") || s.StartsWith("timedelta64");
        }

        private static dtype ConvertFromDatetime(PythonContext cntx, String s) {
            IEnumerable<Object> arf = ParseDatetimeString(cntx, s);
            Console.WriteLine("Result is {0}, {1}", arf.First(), arf.Skip(1).First());

            throw new NotImplementedException();
        }


        /// <summary>
        /// Comma strings are ones that start with an integer, are empty tuples,
        /// or contain commas.  
        /// </summary>
        /// <param name="s">Datetime format string</param>
        /// <returns>True if a comma string</returns>
        private static bool CheckForCommaString(String s) {
            Func<char, bool> checkByteOrder = 
                b => b == '>' || b == '<' || b == '|' || b == '=';

            // Check for ints at the start of a string.
            if (s[0] >= '0' && s[0] <= '9' ||
                s.Length > 1 && checkByteOrder(s[0]) && s[1] >= '0' && s[1] <= '9')
                return true;

            // Empty tuples
            if (s.Length > 1 && s[0] == '(' && s[1] == ')' ||
                s.Length > 3 && checkByteOrder(s[0]) && s[1] == '(' && s[2] == ')')
                return true;

            // Any commas in the string?
            return s.Contains(',');
        }

        private static dtype ConvertFromCommaString(String s) {
            // TODO: Calls Python function, needs integration of numpy + .net interface
            throw new NotImplementedException();
        }


        private static dtype ConvertBracketString(String s) {
            throw new NotImplementedException();
        }

        private static IEnumerable<Object> ParseDatetimeString(PythonContext cntx, String s) {
            CallSite<Func<CallSite, String, Object>> site;

            // Perform the operation based on the type of object we are looking at.
            string[] argNames = {"astr"};
            System.Dynamic.CallInfo call = new System.Dynamic.CallInfo(1, argNames);
            var binder = cntx.CreateCallBinder("numpy._internal._datetimestring", true, call);
            site = CallSite<Func<CallSite, String, Object>>.Create(binder);
            
            return site.Target(site, s) as IEnumerable<Object>;
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
