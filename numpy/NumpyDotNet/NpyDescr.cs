using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Numerics;
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

            if (obj == null) {
                result = NpyCoreApi.DescrFromType(NpyDefs.DefaultType);
            } else if (obj is ScalarGeneric) {
                return ((ScalarGeneric)obj).dtype;
            } else if (obj is dtype) {
                result = (dtype)obj;
            } else if (obj is IronPython.Runtime.Types.PythonType) {
                result = ConvertFromPythonType((IronPython.Runtime.Types.PythonType)obj);
            } else if (obj is string) {
                string s = (string)obj;
                if (!String.IsNullOrEmpty(s) && CheckForDatetime(s)) {
                    result = ConvertFromDatetime(cntx, s);
                } else if (CheckForCommaString(s)) {
                    result = ConvertFromCommaString(s);
                } else {
                    result = ConvertSimpleString(s);
                }
            } else if (obj is List) {
                result = ConvertFromArrayDescr((List)obj, 0);
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


        private static dtype ConvertSimpleString(String s) {
            byte endian = (byte)'=';
            if (s.Length == 0) {
                throw new ArgumentTypeException("data type not understood");
            }
            switch (s[0]) {
                case '<':
                case '>':
                case '|':
                case '=':
                    endian = (byte)s[0];
                    s = s.Substring(1);
                    if (endian == (byte)'|') {
                        endian = (byte)'=';
                    }
                    break;
            }
            return ConvertSimpleString(s, endian);
        }

        private static dtype ConvertSimpleString(string s, byte endian) {
            byte type_char = (byte)' ';
            NpyDefs.NPY_TYPES type = NpyDefs.NPY_TYPES.NPY_NOTYPE+10;
            int elsize = 0;

            if (s.Length == 0) {
                throw new ArgumentTypeException("data type not understood");
            }
            type_char = (byte)s[0];
            if (s.Length > 1) {
                elsize = int.Parse(s.Substring(1));
                if (elsize == 0) {
                } else if (type_char == (byte)NpyDefs.NPY_TYPECHAR.NPY_UNICODELTR) {
                    type = (NpyDefs.NPY_TYPES)type_char;
                    elsize <<= 2;
                } else if (type_char != (byte)NpyDefs.NPY_TYPECHAR.NPY_STRINGLTR &&
                           type_char != (byte)NpyDefs.NPY_TYPECHAR.NPY_VOIDLTR &&
                           type_char != (byte)NpyDefs.NPY_TYPECHAR.NPY_STRINGLTR2) {
                    type = NpyCoreApi.TypestrConvert(elsize, type_char);
                    // The size is encoded in the type, so reset.
                    elsize = 0;
                } else {
                    // For some types the char is the type, even though the type is not defined in NPY_TYPES!
                    type = (NpyDefs.NPY_TYPES)type_char;
                }
            } else {
                type = (NpyDefs.NPY_TYPES)type_char;
            }
            // TODO: Handle typeDict.
            dtype result = null;
            if (type != NpyDefs.NPY_TYPES.NPY_NOTYPE+10) {
                result = NpyCoreApi.DescrFromType(type);
            }
            if (result == null) {
                throw new ArgumentTypeException("data type not understood");
            }
            if (elsize != 0 && result.ElementSize == 0) {
                result = NpyCoreApi.DescrNew(result);
                result.ElementSize = elsize;
            }
            if (endian != (byte)'=' && result.ByteOrder != (byte)'|' &&
                result.ByteOrder != endian) {
                result = NpyCoreApi.DescrNew(result);
                result.ByteOrder = endian;
            }
            return result;
        }

        private static dtype ConvertFromArrayDescr(List l, int align) {
            // TODO: Need to be completed.  Right now only handles pairs
            // of (name, type) as items.
            int n = l.Count;
            int totalSize = 0;
            int maxalign = 0;
            int dtypeflags = 0;
            IntPtr names = NpyCoreApi.NpyArray_DescrAllocNames(n);
            IntPtr fields = NpyCoreApi.NpyArray_DescrAllocFields();
            try {
                for (int i=0; i<n; i++) {
                    object item = l[i];
                    PythonTuple t = (item as PythonTuple);
                    if (t == null || t.Count != 2) {
                        throw new ArgumentTypeException("data type not understood");
                    }
                    string name = (t[0] as string);
                    object type_descr = t[1];
                    if (name == null) {
                        throw new ArgumentTypeException("data type not understood");
                    }
                    if (name.Length == 0) {
                        name = String.Format("f{0}", i);
                    }
                    dtype field_type = DescrConverter(null, type_descr);

                    dtypeflags |= field_type.Flags & NpyDefs.NPY_FROM_FIELDS;

                    if (align != 0) {
                        int field_align = field_type.Alignment;
                        if (field_align > 0) {
                            totalSize = ((totalSize + field_align - 1) / field_align) * field_align;
                        }
                        maxalign = Math.Max(maxalign, field_align);
                    }
                    NpyCoreApi.AddField(fields, names, i, name, field_type, totalSize, null);
                    totalSize += field_type.ElementSize;
                }
            } catch {
                NpyCoreApi.DescrDestroyNames(names, n);
                NpyCoreApi.DescrDestroyFields(fields);
                throw;
            }

            if (maxalign > 1) {
                totalSize = ((totalSize + maxalign - 1) / maxalign) * maxalign;
            }
            int alignment = (align != 0 ? maxalign : 1);
            return NpyCoreApi.DescrNewVoid(fields, names, totalSize, dtypeflags, alignment);
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
        private static readonly PythonType PyLong_Type = DynamicHelpers.GetPythonTypeFromType(typeof(BigInteger));
        private static readonly PythonType PyFloat_Type = DynamicHelpers.GetPythonTypeFromType(typeof(float));
        private static readonly PythonType PyDouble_Type = DynamicHelpers.GetPythonTypeFromType(typeof(double));
        private static readonly PythonType PyBool_Type = DynamicHelpers.GetPythonTypeFromType(typeof(bool));
        private static readonly PythonType PyUnicode_Type = DynamicHelpers.GetPythonTypeFromType(typeof(string));
        private static readonly PythonType PyComplex_Type = DynamicHelpers.GetPythonTypeFromType(typeof(System.Numerics.Complex));
    }
}
