using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Numerics;
using IronPython.Runtime;
using IronPython.Runtime.Types;
using IronPython.Runtime.Operations;
using IronPython.Modules;
using Microsoft.Scripting;

namespace NumpyDotNet {
    /// <summary>
    /// Implements the descriptor (NpyArray_Descr) functionality.  This is not the
    /// public wrapper but a collection of funtionality to support the dtype class.
    /// </summary>
    internal class NpyDescr {

        private static bool IsScalarType(CodeContext cntx, PythonType t) {
            return PythonOps.IsSubClass(t, PyGenericArrType_Type);
        }

        internal static dtype DescrConverter(CodeContext cntx, Object obj, bool align=false) {
            dtype result = null;
            PythonType pt;

            if (obj == null) {
                result = NpyCoreApi.DescrFromType(NpyDefs.DefaultType);
            } else if (obj is dtype) {
                result = (dtype)obj;
            } else if ((pt = obj as PythonType) != null) {
                if (IsScalarType(cntx, pt)) {
                    object scalar = PythonCalls.Call(cntx, pt);
                    result = (dtype)PythonOps.ObjectGetAttribute(cntx, scalar, "dtype");
                } else {
                    result = ConvertFromPythonType(cntx, pt);
                }
            } else if (obj is string) {
                string s = (string)obj;
                if (!String.IsNullOrEmpty(s) && CheckForDatetime(s)) {
                    result = ConvertFromDatetime(cntx, s);
                } else if (CheckForCommaString(s)) {
                    result = ConvertFromCommaString(cntx, s, align);
                } else {
                    result = ConvertSimpleString(s);
                }
            } else if (obj is PythonTuple) {
                result = TryConvertFromTuple(cntx, (PythonTuple)obj);
                if (result == null) {
                    throw new ArgumentException("data type not understood.");
                }
            } else if (obj is List) {
                result = ConvertFromArrayDescr(cntx, (List)obj, align);
            } else if (obj is PythonDictionary) {
                result = ConvertFromDictionary(cntx, (PythonDictionary)obj, align);
            } else if (!(obj is ndarray)) {
                result = DescrFromObject(cntx, obj);
            }
            if (result == null) {
                throw new ArgumentException("data type not understood");
            }
            return result;
        }

        internal static dtype ConvertFromDictionary(CodeContext cntx, PythonDictionary dict, bool align) {
            object oNames = dict.get("names");
            object oDescrs = dict.get("formats");
            // If it doesn't name names and formats then try it as a fields dict
            if (oNames == null || oDescrs == null) {
                return (dtype)NpyUtil_Python.CallInternal(cntx, "_usefields", dict, align ? 1 : 0);
            }
            IList<object> names = (IList<object>)oNames;
            IList<object> descrs = (IList<object>)oDescrs;
            IList<object> offsets = dict.get("offsets") as IList<object>;
            IList<object> titles = dict.get("titles") as IList<object>;

            int n = names.Count;
            if (descrs.Count != n || offsets != null && offsets.Count != n || titles != null && titles.Count != n) {
                throw new ArgumentException("all items in the dictionary must have the same length");
            }
            List<FieldInfo> fields = new List<FieldInfo>(n);
            try {
                for (int i = 0; i < n; i++) {
                    FieldInfo info = new FieldInfo();
                    info.name = NpyUtil_Python.ConvertToString(names[i], cntx);
                    info.dtype = descrs[i];
                    if (offsets != null) {
                        if (offsets[i] != null) {
                            info.offset = NpyUtil_Python.ConvertToInt(offsets[i], cntx);
                        }
                    }
                    if (titles != null) {
                        if (titles[i] != null) {
                            info.title = NpyUtil_Python.ConvertToString(titles[i], cntx);
                        }
                    }
                    fields.Add(info);
                }
            } catch {
                throw new ArgumentException("data type not understood");
            }
            return ConvertFromFields(cntx, fields, align);
        }

        internal static dtype TryConvertFromTuple(CodeContext cntx, PythonTuple tup) {
            if (tup.Count != 2) {
                return null;
            }
            dtype t1 = DescrConverter(cntx, tup[0]);
            object other = tup[1];
            dtype result = UseInherit(cntx, t1, other);
            if (result != null) {
                return result;
            }
            if (t1.ElementSize == 0) {
                // Interpret the next item as a size
                int itemsize;
                try {
                    itemsize = NpyUtil_Python.ConvertToInt(other, cntx);
                } catch {
                    throw new ArgumentException("invalid itemsize in generic type tuple");
                }
                if (t1.TypeNum == NpyDefs.NPY_TYPES.NPY_UNICODE) {
                    itemsize *= 4;
                }
                result = new dtype(t1);
                result.ElementSize = itemsize;
                return result;
            }
            if (other is PythonDictionary) {
                // This is a metadata dictionary.  Just ignore it.
                return t1;
            }
            // Assume other is a shape
            IntPtr[] shape = NpyUtil_ArgProcessing.IntpArrConverter(other);
            if (shape == null) {
                throw new ArgumentException("invalid shape in fixed-type tuple");
            }
            // (type, 1) or (type, ()) should be treated as type
            if (shape.Length == 0 && other is PythonTuple ||
                shape.Length == 1 && shape[0].ToInt64() == 1 && 
                !(other is IEnumerable<object>)) {
                return t1;
            }

            return NpyCoreApi.DescrNewSubarray(t1, shape);
        }

        private static dtype UseInherit(CodeContext cntx, dtype t1, object other) {
            dtype conv;
            // Check to see if other is a type
            if (other is ScalarInteger ||
                NpyUtil_Python.IsTupleOfIntegers(other)) {
                return null;
            }
            try {
                conv = DescrConverter(cntx, other);
            } catch {
                return null;
            }

            return NpyCoreApi.InheritDescriptor(t1, conv);
        }

        /// <summary>
        /// Converts a Python type into a descriptor object
        /// </summary>
        /// <param name="t">Python type object</param>
        /// <returns>Corresponding descriptor object</returns>
        private static dtype ConvertFromPythonType(CodeContext cntx, IronPython.Runtime.Types.PythonType t) {
            NpyDefs.NPY_TYPES type = NpyDefs.NPY_TYPES.NPY_OBJECT;
            if (t == PyInt_Type) type = NpyDefs.NPY_TYPES.NPY_LONG;
            else if (t == PyLong_Type) type = NpyDefs.NPY_TYPES.NPY_LONGLONG;
            else if (t == PyFloat_Type) type = NpyDefs.NPY_TYPES.NPY_DOUBLE;
            else if (t == PyBool_Type) type = NpyDefs.NPY_TYPES.NPY_BOOL;
            else if (t == PyComplex_Type) type = NpyDefs.NPY_TYPES.NPY_CDOUBLE;
            else if (t == PyBytes_Type) type = NpyDefs.NPY_TYPES.NPY_STRING;
            else if (t == PyUnicode_Type) type = NpyDefs.NPY_TYPES.NPY_UNICODE;
            else if (t == PyBuffer_Type) type = NpyDefs.NPY_TYPES.NPY_VOID;
            else if (t == PyMemoryView_Type) type = NpyDefs.NPY_TYPES.NPY_VOID;
            else {
                dtype result = DescrFromObject(cntx, t);
                if (result != null) {
                    return result;
                }
            }

            return NpyCoreApi.DescrFromType(type);
        }

        private static dtype DescrFromObject(CodeContext cntx, object obj) {
            // Try a dtype attribute
            try {
                return DescrConverter(cntx, PythonOps.ObjectGetAttribute(cntx, obj, "dtype"));
            } catch { }
            // Try a ctype type, possibly with a length
            try {
                dtype d = DescrConverter(cntx, PythonOps.ObjectGetAttribute(cntx, obj, "_type_"));
                try {
                    object length = PythonOps.ObjectGetAttribute(cntx, obj, "_length_");
                    PythonTuple tup = new PythonTuple(new object[] { d, length });
                    return DescrConverter(cntx, tup);
                } catch { }
                return d;
            } catch { }
            // Try a ctype fields
            try {
                return DescrConverter(cntx, PythonOps.ObjectGetAttribute(cntx, obj, "_fields_"));
            } catch { }
 
            return null;
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

        private static dtype ConvertFromDatetime(CodeContext cntx, String s) {
            PythonTuple val = NpyUtil_Python.CallInternal(cntx, "_datetimestring", s) as PythonTuple;
            if (val == null || val.Count != 2) {
                throw new IronPython.Runtime.Exceptions.RuntimeException("_datetimestring did not return a pair");
            }
            PythonTuple dt_tuple = val[0] as PythonTuple;

            if (dt_tuple == null || dt_tuple.Count != 4 || !(val[1] is bool)) {
                throw new IronPython.Runtime.Exceptions.RuntimeException("_datetimestring is not returning a length 4 tuple and a boolean.");
            }
            bool datetime = (bool)val[1];

            dtype result;
            if (datetime) {
                result = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_DATETIME);
            } else {
                result = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_TIMEDELTA);
            }

            NpyCoreApi.SetDateTimeInfo(result,
                NpyUtil_Python.ConvertToString(dt_tuple[0], cntx),
                NpyUtil_Python.ConvertToInt(dt_tuple[1], cntx),
                NpyUtil_Python.ConvertToInt(dt_tuple[2], cntx),
                NpyUtil_Python.ConvertToInt(dt_tuple[3], cntx));

            return result;
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

        private static dtype ConvertFromCommaString(CodeContext cntx, String s, bool align) {
            List val = NpyUtil_Python.CallInternal(cntx, "_commastring", s) as List;
            if (val == null || val.Count < 1) {
                throw new IronPython.Runtime.Exceptions.RuntimeException(
                    "_commastring not returning a list with len >= 1");
            }

            if (val.Count == 1) {
                return DescrConverter(cntx, val[0]);
            } else {
                return ConvertFromCommaStringList(cntx, val, align);
            }
        }

        private static dtype ConvertFromCommaStringList(CodeContext cntx, List l, bool align) {
            // This is simply a list of formats
            List<FieldInfo> fieldInfo = l.Where(x => !(x is string && (string)x == "")).Select(x => new FieldInfo { dtype = x }).ToList();
            return ConvertFromFields(cntx, fieldInfo, align);
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

        private class FieldInfo
        {
            public string name;
            public object dtype;
            public string title;
            public int? offset;
        };

        private static dtype ConvertFromFields(CodeContext cntx, IList<FieldInfo> l, bool align) {
            int n = l.Count;
            int totalSize = 0;
            int maxalign = 0;
            int dtypeflags = 0;
            int offset;
            IntPtr names = NpyCoreApi.NpyArray_DescrAllocNames(n);
            IntPtr fields = NpyCoreApi.NpyArray_DescrAllocFields();
            try {
                for (int i=0; i<n; i++) {
                    FieldInfo item = l[i];
                    if (item.name == null) {
                        item.name = String.Format("f{0}", i);
                    }
                    dtype field_type = DescrConverter(cntx, item.dtype);
                    dtypeflags |= field_type.Flags & NpyDefs.NPY_FROM_FIELDS;

                    if (align) {
                        int field_align = field_type.Alignment;
                        if (field_align > 0) {
                            totalSize = ((totalSize + field_align - 1) / field_align) * field_align;
                        }
                        maxalign = Math.Max(maxalign, field_align);
                    }
                    if (item.offset == null) {
                        offset = totalSize;
                        totalSize += field_type.ElementSize;
                    } else {
                        offset = (int)item.offset;
                        if (offset < 0) {
                            throw new ArgumentException("Field offsets can't be negative");
                        }
                        totalSize = Math.Max(totalSize, offset + field_type.ElementSize);
                    }
                    NpyCoreApi.AddField(fields, names, i, item.name, field_type, offset, item.title);
                }
            } catch {
                NpyCoreApi.DescrDestroyNames(names, n);
                NpyCoreApi.DescrDestroyFields(fields);
                throw;
            }

            if (maxalign > 1) {
                totalSize = ((totalSize + maxalign - 1) / maxalign) * maxalign;
            }
            int alignment = (align ? maxalign : 1);
            return NpyCoreApi.DescrNewVoid(fields, names, totalSize, dtypeflags, alignment);
        }


        private static FieldInfo ItemToFieldInfo(object item) {
            FieldInfo result = new FieldInfo();
            PythonTuple tup = item as PythonTuple;
            if (tup == null || tup.Count < 2) {
                throw new ArgumentException("Data type not understood");
            }
            // Deal with the name and title
            object val = tup[0];
            if (val is string) {
                result.name = (string)val;
            } else if (val is PythonTuple) {
                PythonTuple name_tuple = (PythonTuple)val;
                if (name_tuple.Count != 2 || !(name_tuple[0] is string) || !(name_tuple[1] is string)) {
                    throw new ArgumentException("Data type not understood: name and title must both be strings");
                }
                result.name = (string)name_tuple[0];
                result.name = (string)name_tuple[1];
            }
            if (tup.Count == 2) {
                result.dtype = tup[1];
            } else {
                result.dtype = tup[new Slice(1, 3)];
            }
            return result;
        }

        private static dtype ConvertFromArrayDescr(CodeContext cntx, List l, bool align) {
            List<FieldInfo> fields = l.Select(ItemToFieldInfo).ToList();
            return ConvertFromFields(cntx, fields, align);
        }


        private static readonly PythonType PyInt_Type = DynamicHelpers.GetPythonTypeFromType(typeof(int));
        private static readonly PythonType PyLong_Type = DynamicHelpers.GetPythonTypeFromType(typeof(BigInteger));
        private static readonly PythonType PyFloat_Type = DynamicHelpers.GetPythonTypeFromType(typeof(double));
        private static readonly PythonType PyBool_Type = DynamicHelpers.GetPythonTypeFromType(typeof(bool));
        private static readonly PythonType PyBytes_Type = DynamicHelpers.GetPythonTypeFromType(typeof(Bytes));
        private static readonly PythonType PyUnicode_Type = DynamicHelpers.GetPythonTypeFromType(typeof(string));
        private static readonly PythonType PyComplex_Type = DynamicHelpers.GetPythonTypeFromType(typeof(System.Numerics.Complex));
        private static readonly PythonType PyBuffer_Type = DynamicHelpers.GetPythonTypeFromType(typeof(PythonBuffer));
        private static readonly PythonType PyMemoryView_Type = DynamicHelpers.GetPythonTypeFromType(typeof(MemoryView));

        private static readonly PythonType PyGenericArrType_Type = DynamicHelpers.GetPythonTypeFromType(typeof(ScalarGeneric));
    }
}
