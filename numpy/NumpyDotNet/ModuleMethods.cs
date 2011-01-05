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

namespace NumpyDotNet {
    /// <summary>
    /// ModuleMethods implements the module-level numpy functions.
    /// </summary>
    public static class ModuleMethods {
        private static String[] arrayKwds = { "object", "dtype", "copy", "order", "subok", "ndmin" };

        public const string __module__ = "numpy.core.multiarray";

        public static void __init__(CodeContext cntx) {
            PythonDictionary moduleDict = cntx.ModuleContext.Module.Get__dict__();

            moduleDict.Add(new KeyValuePair<object, object>("CLIP", (int)NpyDefs.NPY_CLIPMODE.NPY_CLIP));
            moduleDict.Add(new KeyValuePair<object, object>("WRAP", (int)NpyDefs.NPY_CLIPMODE.NPY_WRAP));
            moduleDict.Add(new KeyValuePair<object, object>("RAISE", (int)NpyDefs.NPY_CLIPMODE.NPY_RAISE));
            moduleDict.Add(new KeyValuePair<object, object>("ALLOW_THREADS", 1));
            moduleDict.Add(new KeyValuePair<object, object>("BUFSIZE", (int)NpyDefs.NPY_BUFSIZE));
            moduleDict.Add(new KeyValuePair<object, object>("ITEM_HASOBJECT", (int)NpyDefs.NPY__ITEM_HASOBJECT));
            moduleDict.Add(new KeyValuePair<object, object>("LIST_PICKLE", (int)NpyDefs.NPY_LIST_PICKLE));
            moduleDict.Add(new KeyValuePair<object, object>("ITEM_IS_POINTER", (int)NpyDefs.NPY_ITEM_IS_POINTER));
            moduleDict.Add(new KeyValuePair<object, object>("NEEDS_INIT", (int)NpyDefs.NPY_NEEDS_INIT));
            moduleDict.Add(new KeyValuePair<object, object>("NEEDS_PYAPI", (int)NpyDefs.NPY_NEEDS_PYAPI));
            moduleDict.Add(new KeyValuePair<object, object>("USE_GETITEM", (int)NpyDefs.NPY_USE_GETITEM));
            moduleDict.Add(new KeyValuePair<object, object>("USE_SETITEM", (int)NpyDefs.NPY_USE_SETITEM));
            moduleDict.Add(new KeyValuePair<object, object>("MAXDIMS", (int)NpyDefs.NPY_MAXDIMS));
        }

        /// <summary>
        /// Ick. See ScalarGeneric.Initialized.
        /// </summary>
        public static void InitializeScalars() {
            ScalarGeneric.Initialized = true;
        }


        /// <summary>
        /// Module method 'array': constructs a new array from an input object and
        /// optional type and other arguments.
        /// </summary>
        /// <param name="cntx"></param>
        /// <param name="o">Source object</param>
        /// <param name="kwargs">Optional named args</param>
        /// <returns>New array object</returns>
        public static ndarray array(CodeContext cntx, Object o, [ParamDictionary] IDictionary<object,object> kwargs) {
            Object[] args = { o };
            return arrayFromObject(cntx,
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
            [ParamDictionary] IDictionary<object,object> kwargs) {
            Object[] args = { o, descr };
            return arrayFromObject(cntx,
                NpyUtil_ArgProcessing.BuildArgsArray(args, arrayKwds, kwargs));
        }



        internal static ndarray arrayFromObject(CodeContext cntx, Object[] args) {
            Object src = args[0];
            dtype type = null;
            bool copy = true;
            NpyDefs.NPY_ORDER order = NpyDefs.NPY_ORDER.NPY_ANYORDER;
            bool subok = false;
            int ndmin = 0;
            ndarray result = null;

            // Ensures that the numeric operations are initialized once at startup.
            // TODO: This is unpleasant, there must be a better way to do this.
            NumericOps.InitUFuncOps(cntx);

            try {
                if (args[1] != null) type = NpyDescr.DescrConverter(cntx, args[1]);
                if (args[2] != null) copy = NpyUtil_ArgProcessing.BoolConverter(args[2]);

                order = NpyUtil_ArgProcessing.OrderConverter(args[3]);
                if (args[4] != null) subok = NpyUtil_ArgProcessing.BoolConverter(args[4]);
                if (args[5] != null) ndmin = NpyUtil_ArgProcessing.IntConverter(args[5]);

                if (ndmin >= NpyDefs.NPY_MAXDIMS) {
                    throw new ArgumentException(
                        String.Format("ndmin ({0}) bigger than allowable number of dimension ({1}).",
                        ndmin, NpyDefs.NPY_MAXDIMS - 1));
                }

                if (subok && src is ndarray ||
                    !subok && src != null && src.GetType() == typeof(ndarray)) {
                    ndarray arr = (ndarray)src;
                    if (type == null) {
                        if (!copy && arr.StridingOk(order)) {
                            result = arr;
                        } else {
                            result = NpyCoreApi.NewCopy(arr, order);
                        }
                    } else {
                        dtype oldtype = arr.Dtype;
                        if (NpyCoreApi.EquivTypes(oldtype, type)) {
                            if (!copy && arr.StridingOk(order)) {
                                result = arr;
                            } else {
                                result = NpyCoreApi.NewCopy(arr, order);
                                if (oldtype != type) {
                                    result.Dtype = oldtype;
                                }
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

                if (result != null && result.ndim < ndmin) {
                    result = NpyArray.PrependOnes(result, result.ndim, ndmin);
                }
            } catch (Exception e) {
                throw e;
            }
            return result;
        }

        public static ndarray _reconstruct(CodeContext cntx, PythonType subtype, object shape, object dtype) {
            return ndarray.__new__(cntx, subtype, shape, dtype);
        }

        public static ndarray arange(CodeContext cntx, object start, object stop = null, object step = null, object dtype = null) {
            dtype rtype = null;
            if (dtype != null) {
                rtype = NpyDescr.DescrConverter(cntx, dtype);
            }
            return NpyArray.Arange(cntx, start, stop, step, rtype);
        }

        public static ndarray empty(CodeContext cntx, object shape, object dtype = null, object order = null) {
            long[] aShape;

            try {
                aShape = NpyUtil_ArgProcessing.IntArrConverter(shape);
            } catch (OverflowException) {
                // Translate error message to match CPython, make regressions happy.
                throw new ArgumentException("Maximum allowed dimension exceeded");
            }

            dtype d = NpyDescr.DescrConverter(cntx, dtype);
            NpyDefs.NPY_ORDER eOrder = NpyUtil_ArgProcessing.OrderConverter(order);

            return NpyArray.Empty(aShape, d, eOrder);
        }

        public static ndarray zeros(CodeContext cntx, object shape, object dtype = null, object order = null) {
            long[] aShape = NpyUtil_ArgProcessing.IntArrConverter(shape);
            dtype d = NpyDescr.DescrConverter(cntx, dtype);
            NpyDefs.NPY_ORDER eOrder = NpyUtil_ArgProcessing.OrderConverter(order);

            return NpyArray.Zeros(aShape, d, eOrder);
        }

        public static void putmask(ndarray arr, object mask, object values) {
            ndarray aMask;
            ndarray aValues;

            aMask = (mask as ndarray);
            if (aMask == null) {
                aMask = NpyArray.FromAny(mask, NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_BOOL),
                    0, 0, NpyDefs.NPY_CARRAY | NpyDefs.NPY_FORCECAST, null);
            }

            aValues = (values as ndarray);
            if (aValues == null) {
                aValues = NpyArray.FromAny(values, arr.Dtype, 0, 0, NpyDefs.NPY_CARRAY, null);
            }

            arr.PutMask(aValues, aMask);
        }


        /// <summary>
        /// Reads the contents of a text or binary file and turns the contents into an array. If
        /// 'sep' is specified the file is assumed to be text, other it is assumed binary.
        /// </summary>
        /// <param name="file">PythonFile, FileStream, or file name string</param>
        /// <param name="dtype">Optional type for the resulting array, default is double</param>
        /// <param name="count">Optional number of array elements to read, default reads all elements</param>
        /// <param name="sep">Optional separator for text elements</param>
        /// <returns></returns>
        public static ndarray fromfile(CodeContext cntx, object file, object dtype = null, object count = null, object sep = null) {
            string fileName;
            dtype rtype;
            int num;

            // Annoying.  PythonFile is not convertable to Stream and neither convert to FILE* needed by the
            // current implementation. We really need to implement a new file reader that take callbacks to
            // the interface to really support multiple platforms.
            if (file is string) fileName = (string)file;
            else if (file is PythonFile || file is Stream) {
                throw new NotImplementedException("File and stream types are not supported pending implementation of a cross-platform reader.");
            } else {
                throw new NotImplementedException(String.Format("Unsupported file type '{0}'.", file.GetType().Name));
            }

            rtype = NpyDescr.DescrConverter(cntx, dtype);
            num = (count != null) ? NpyUtil_ArgProcessing.IntConverter(count) : -1;

            return NpyCoreApi.ArrayFromFile(fileName, rtype, num, (sep != null) ? sep.ToString() : null);
        }


        /// <summary>
        /// Constructs an array from a text input string. Since strings are unicode in .NET, sep must be
        /// specified in this case and only text strings are supported.  Use the version accepting bytes,
        /// below, for binary strings.
        /// </summary>
        /// <param name="cntx">Python code context</param>
        /// <param name="string">Input text string</param>
        /// <param name="dtype">Desired array type or null</param>
        /// <param name="count">Max number of array elements to convert</param>
        /// <param name="sep">Element separator</param>
        /// <returns>Array populated with elements from the string</returns>
        public static ndarray fromstring(CodeContext cntx, string @string, object dtype=null, object count=null, object sep=null) {
            dtype rtype;
            int num;

            rtype = NpyDescr.DescrConverter(cntx, dtype);
            num = (count != null) ? NpyUtil_ArgProcessing.IntConverter(count) : -1;
            return NpyCoreApi.ArrayFromString(@string, rtype, num, (sep == null ? null : sep.ToString()));
        }

        /// <summary>
        /// Constructs an array from a text input string. Since strings are unicode in .NET, sep must be
        /// specified in this case and only text strings are supported.  Use the version accepting bytes,
        /// below, for binary strings.
        /// </summary>
        /// <param name="cntx">Python code context</param>
        /// <param name="string">Input text string</param>
        /// <param name="dtype">Desired array type or null</param>
        /// <param name="count">Max number of array elements to convert</param>
        /// <param name="sep">Element separator</param>
        /// <returns>Array populated with elements from the string</returns>
        public static ndarray fromstring(CodeContext cntx, Bytes @string, object dtype=null, object count=null, object sep=null) {
            dtype rtype;
            int num;

            rtype = NpyDescr.DescrConverter(cntx, dtype);
            num = (count != null) ? NpyUtil_ArgProcessing.IntConverter(count) : -1;
            // Make a copy since there is no way to get
            // at the byte[] in bytes.
            byte[] copy = new byte[@string.Count];
            @string.CopyTo(copy, 0);
            return NpyCoreApi.ArrayFromBytes(copy, rtype, num, (sep == null ? null : sep.ToString()));
        }

        /// <summary>
        /// Constructs an array from from a iterable sequence. This function triggers iter.Count(), thus
        /// requiring two iterations through the complete sequence. This is faster than dynamically
        /// resizing the array for fast sequences but will be slower for complex generator expressions.
        /// </summary>
        /// <param name="cntx">Python code context</param>
        /// <param name="iter">Sequence to build the array from</param>
        /// <param name="dtype">Type of the resulting array</param>
        /// <param name="count">Maximum number of elements to convert</param>
        /// <returns>Array populated with elements from iter</returns>
        public static ndarray fromiter(CodeContext cntx, IEnumerable<object> iter, object dtype = null, object count = null) {
            dtype rtype;
            int num;

            rtype = NpyDescr.DescrConverter(cntx, dtype);
            num = (count != null) ? NpyUtil_ArgProcessing.IntConverter(count) : int.MaxValue;
            if (num == int.MaxValue && rtype.ElementSize == 0) {
                throw new ArgumentException("A length must be specified when using variable-sized data type");
            }

            // Python generators appear as IEnumerables but can only be iterated over once
            // so we need to save the results. 
            if (iter is IEnumerableOfTWrapper<object>) {
                iter = ((IEnumerableOfTWrapper<object>)iter).ToList();
            }

            int tmp = iter.Count();
            if (num != int.MaxValue && num > tmp) {
                throw new ArgumentException("iterator too short");
            } else {
                if (num < tmp) {
                    iter = iter.Take(num);
                } else {
                    num = tmp;
                }
            } 

            if (rtype.ChkFlags(NpyDefs.NPY_ITEM_REFCOUNT)) {
                throw new ArgumentException("cannot create object arrays from iterators");
            }

            ndarray ret = NpyCoreApi.AllocArray(rtype, 1, new long[] { num }, false);
            NpyArray.AssignToArray(iter, ret);
            return ret;
        }

        public static ndarray frombuffer(object buffer, object dtype = null,
                                         object count = null, object offset = null) {
            throw new NotImplementedException();
        }


        public static ndarray lexsort(object keysObj, int axis = -1) {
            IEnumerable<object> keys = keysObj as IEnumerable<object>;
            if (keys == null || keys.Count() == 0) {
                throw new ArgumentTypeException("Need sequence of keys with len > 0 in lexsort");
            }

            int n = keys.Count();
            ndarray[] arrays = new ndarray[n];
            int i = 0;
            foreach (object k in keys) {
                arrays[i++] = NpyArray.FromAny(k);
            }
            return ndarray.LexSort(arrays, axis);
        }

        private static object[] TypeInfoArray(NpyDefs.NPY_TYPES typenum) {
            dtype d = NpyCoreApi.DescrFromType(typenum);
            if (d.ScalarType == null) {
                return null;
            }
            object[] objs;
            PythonType pt = d.type;
            if (d.ScalarType.IsSubclassOf(typeof(ScalarInteger))) {
                object maxValue = d.ScalarType.GetField("MaxValue", BindingFlags.Static | BindingFlags.NonPublic).GetValue(null);
                object minValue = d.ScalarType.GetField("MinValue", BindingFlags.Static | BindingFlags.NonPublic).GetValue(null);
                objs = new object[] { new string((char)d.Type, 1), (int)typenum, 8*d.ElementSize, d.Alignment, maxValue, minValue, pt };
            } else {
                objs = new object[] { new string((char)d.Type, 1), (int)typenum, 8*d.ElementSize, d.Alignment, pt };
            }
            return objs;
        }

        private static void AddTypeType(PythonDictionary dict, NpyDefs.NPY_TYPES typenum) {
            object[] value = TypeInfoArray(typenum);
            if (value != null) {
                string name = typenum.ToString().Substring(4);
                dict[name] = new PythonTuple(value);
            }
        }

        public static PythonDictionary typeinfo {
            get {
                object[] tmp;

                PythonDictionary result = new PythonDictionary();
                // Add types for all the C types
                NpyDefs.NPY_TYPES i;
                for (i=0; i<NpyDefs.NPY_TYPES.NPY_NTYPES; i++) {
                    AddTypeType(result, i);
                }
                // Add intp
                tmp = TypeInfoArray(NpyDefs.NPY_INTP);
                tmp[0] = 'p';
                result["INTP"] = new PythonTuple(tmp);
                tmp = TypeInfoArray(NpyDefs.NPY_UINTP);
                tmp[0] = 'P';
                result["UINTP"] = new PythonTuple(tmp);
                // Add the abstract types
                result["Generic"] = DynamicHelpers.GetPythonTypeFromType(typeof(ScalarGeneric));
                result["Number"] = DynamicHelpers.GetPythonTypeFromType(typeof(ScalarNumber));
                result["Integer"] = DynamicHelpers.GetPythonTypeFromType(typeof(ScalarInteger));
                result["SignedInteger"] = DynamicHelpers.GetPythonTypeFromType(typeof(ScalarSignedInteger));
                result["TimeInteger"] = DynamicHelpers.GetPythonTypeFromType(typeof(ScalarTimeInteger));
                result["UnsignedInteger"] = DynamicHelpers.GetPythonTypeFromType(typeof(ScalarUnsignedInteger));
                result["Inexact"] = DynamicHelpers.GetPythonTypeFromType(typeof(ScalarInexact));
                result["Floating"] = DynamicHelpers.GetPythonTypeFromType(typeof(ScalarFloating));
                result["ComplexFloating"] = DynamicHelpers.GetPythonTypeFromType(typeof(ScalarComplexFloating));
                result["Flexible"] = DynamicHelpers.GetPythonTypeFromType(typeof(ScalarFlexible));
                result["Character"] = DynamicHelpers.GetPythonTypeFromType(typeof(ScalarCharacter));
                return result;
            }
        }

        public static PythonDictionary _flagdict {
            get {
                PythonDictionary result = new PythonDictionary();
                result["OWNDATA"] = NpyDefs.NPY_OWNDATA;
                result["O"] = NpyDefs.NPY_OWNDATA;
                result["FORTRAN"] = NpyDefs.NPY_FORTRAN;
                result["F"] = NpyDefs.NPY_FORTRAN;
                result["CONTIGUOUS"] = NpyDefs.NPY_CONTIGUOUS;
                result["C"] = NpyDefs.NPY_CONTIGUOUS;
                result["UPDATEIFCOPY"] = NpyDefs.NPY_UPDATEIFCOPY;
                result["U"] = NpyDefs.NPY_UPDATEIFCOPY;
                result["WRITEABLE"] = NpyDefs.NPY_WRITEABLE;
                result["W"] = NpyDefs.NPY_WRITEABLE;
                result["C_CONTIGUOUS"] = NpyDefs.NPY_C_CONTIGUOUS;
                result["F_CONTIGUOUS"] = NpyDefs.NPY_F_CONTIGUOUS;
                return result;
            }
        }

        public static ndarray concatenate(IEnumerable<object> seq, int axis = 0) {
            return NpyArray.Concatenate(seq, axis);
        }

        public static object inner(object o1, object o2) {
            return ndarray.ArrayReturn(NpyArray.InnerProduct(o1, o2));
        }

        public static object correlate(object o1, object o2, object mode) {
            return correlateInternal(o1, o2, mode, NpyCoreApi.Correlate);
        }

        public static object correlate2(object o1, object o2, object mode) {
            return correlateInternal(o1, o2, mode, NpyCoreApi.Correlate2);
        }

        private static object correlateInternal(object o1, object o2, object mode, Func<ndarray, ndarray, NpyDefs.NPY_TYPES, int, ndarray> f) {
            // Find a type that works for both inputs. (ie, one is int, the other is float,
            // we want float for the result).
            dtype type = NpyArray.FindArrayType(o1, null);
            type = NpyArray.FindArrayType(o2, type);

            int modeNum = ModeToInt(mode);

            ndarray arr1 = NpyArray.FromAny(o1, type, 1, 1, NpyDefs.NPY_DEFAULT);
            ndarray arr2 = NpyArray.FromAny(o2, type, 1, 1, NpyDefs.NPY_DEFAULT);
            return f(arr1, arr2, type.TypeNum, modeNum);
        }

        private static int ModeToInt(object o) {
            if (o is string) {
                switch (((string)o).ToLower()[0]) {
                    case 'v': return 0;
                    case 's': return 1;
                    case 'f': return 2;
                    default:
                        throw new NotImplementedException(String.Format("Unknown mode '{0}'", o));
                }
            } else if (o is int) {
                return (int)o;
            }
            throw new NotImplementedException(String.Format("Unhandled mode type '{0}'", o.GetType().Name));
        }


        public static object dot(object o1, object o2) {
            return ndarray.ArrayReturn(NpyArray.MatrixProduct(o1, o2));
        }

        public static object where(object o, object x=null, object y=null) {
            ndarray arr = NpyArray.FromAny(o);
            if (x == null && y == null) {
                return arr.nonzero();
            }
            if (x == null || y == null) {
                throw new ArgumentException("either both or neither of x and y should be given");
            }
            ndarray obj = NpyArray.FromAny(arr.__ne__(NpyUtil_Python.DefaultContext, 0), flags: NpyDefs.NPY_ENSUREARRAY);
            return obj.Choose(new object[] { y, x });
        }

        public static object _fastCopyAndTranspose(object a) {
            ndarray arr = NpyArray.FromAny(a, flags: NpyDefs.NPY_CARRAY);
            return NpyCoreApi.CopyAndTranspose(arr);
        }

        public static object _vec_string(CodeContext cntx, object charArrObj, object typeObj, string methodName, object argsSeq = null) {
            ndarray charArray = NpyArray.FromAny(charArrObj, null, 0, 0, NpyDefs.NPY_CARRAY);
            dtype type = NpyDescr.DescrConverter(cntx, typeObj);

            object method;
            if (charArray.Dtype.TypeNum == NpyDefs.NPY_TYPES.NPY_STRING) {
                method = PythonOps.GetBoundAttr(cntx, DynamicHelpers.GetPythonTypeFromType(typeof(Bytes)), methodName);
            } else if (charArray.Dtype.TypeNum == NpyDefs.NPY_TYPES.NPY_UNICODE) {
                method = PythonOps.GetBoundAttr(cntx, DynamicHelpers.GetPythonTypeFromType(typeof(string)), methodName);
            } else {
                throw new ArgumentTypeException("string operation on non-string array");
            }

            object result = null;
            if (method != null) {
                IEnumerable<object> seq = (argsSeq != null) ? (IEnumerable<object>)argsSeq : null;
                if (seq != null && seq.Count() > 0) {
                    result = VecStringWithArgs(cntx, charArray, type, method, seq);
                } else if (argsSeq == null || seq != null) {
                    result = VecStringNoArgs(cntx, charArray, type, method);
                } else {    // Occurs when argsSeq is some different type
                    throw new ArgumentTypeException("'args' must be a sequence of arguments");
                }
            }
            return result;
        }

        private static object VecStringNoArgs(CodeContext cntx, ndarray arr, dtype type, object method) {
            ndarray result = NpyCoreApi.NewFromDescr(type, arr.Dims, arr.Strides, 0, null);
            flatiter inIter = NpyCoreApi.IterNew(arr);
            flatiter outIter = NpyCoreApi.IterNew(result);

            while (inIter.MoveNext()) {
                outIter.MoveNext();

                outIter.Current = PythonOps.CallWithContext(cntx, method, inIter.Current);
            }
            return result;
        }

        private static object VecStringWithArgs(CodeContext cntx, ndarray arr, dtype type, object method, IEnumerable<object> argsSeq) {

            if (argsSeq.Count() >= NpyDefs.NPY_MAXARGS) {
                throw new ArgumentException(
                    String.Format("len(args) must be < {0}", NpyDefs.NPY_MAXARGS));
            }

            // Build an iterator over the array itself plus all arguments.
            object[] a = (new object[] { arr }).Concat(argsSeq).ToArray();
            broadcast inIter = new broadcast(args: a);
            ndarray result = NpyCoreApi.NewFromDescr(type, inIter.dims, null, 0, null);
            flatiter outIter = NpyCoreApi.IterNew(result);

            List<flatiter> iters = inIter.iters.Cast<flatiter>().ToList();
            while (inIter.MoveNext()) {
                outIter.MoveNext();
                outIter.Current = PythonOps.CallWithContext(cntx, method, args: iters.Select(i => i.Current).ToArray());
            }
            return result;
        }


        public static int _get_ndarray_c_version() {
            return (int)NpyCoreApi.GetAbiVersion();
        }


        /// <summary>
        /// Allows callers to replace the to-string function for ndarray class.
        /// </summary>
        /// <param name="f">Function to call</param>
        /// <param name="repr">Set the repr function (true) or the string function (false)</param>
        public static void set_string_function(CodeContext cntx, object f = null, int repr = 1) {
            if (f != null && !PythonOps.IsCallable(cntx, f)) {
                throw new ArgumentTypeException("Argument must be callable");
            }
            Func<ndarray, string> ff = null;
            if (f != null) ff = x => (string)PythonOps.CallWithContext(cntx, f, x);

            if (repr != 0) {
                ndarray.ReprFunction = ff;
            } else {
                ndarray.StrFunction = ff;
            }
        }


        public static void set_typeDict(object o) {
            if (!(o is PythonDictionary)) {
                throw new ArgumentTypeException(
                    String.Format("Attempt to set type dictionary to non-dictionary type '{0}'.", o.GetType().Name));
            }
            NpyDescr.TypeDict = (PythonDictionary)o;
        }


        public static string format_longfloat(object x, int precision) {
            if (x is ScalarFloat64) {
                return NpyCoreApi.FormatLongFloat((double)(ScalarFloat64)x, precision);
            } else if (x is double) {
                return NpyCoreApi.FormatLongFloat((double)x, precision);
            }
            throw new NotImplementedException(
                String.Format("Unhandled long float type '{0}'", x.GetType().Name));
        }


        public static object compare_chararrays(object a1, object a2, string cmp, object rstrip) {
            bool rstripFlag = NpyUtil_ArgProcessing.BoolConverter(rstrip);

            NpyDefs.NPY_COMPARE_OP cmpOp;

            if (cmp == "==") cmpOp = NpyDefs.NPY_COMPARE_OP.NPY_EQ;
            else if (cmp == "!=") cmpOp = NpyDefs.NPY_COMPARE_OP.NPY_NE;
            else if (cmp == "<") cmpOp = NpyDefs.NPY_COMPARE_OP.NPY_LT;
            else if (cmp == "<=") cmpOp = NpyDefs.NPY_COMPARE_OP.NPY_LE;
            else if (cmp == ">=") cmpOp = NpyDefs.NPY_COMPARE_OP.NPY_GE;
            else if (cmp == ">") cmpOp = NpyDefs.NPY_COMPARE_OP.NPY_GT;
            else throw new ArgumentException("comparison must be '==', '!=', '<', '>', '<=', or '>='");

            ndarray arr1 = NpyArray.FromAny(a1);
            ndarray arr2 = NpyArray.FromAny(a2);
            if (arr1 == null || arr2 == null) {
                return null;
            }

            ndarray res;
            if (arr1.IsString && arr2.IsString) {
                if (arr1.Dtype.TypeNum == NpyDefs.NPY_TYPES.NPY_STRING &&
                    arr2.Dtype.TypeNum == NpyDefs.NPY_TYPES.NPY_UNICODE) {
                    arr1 = PromoteToUnicode(arr1, arr2);
                } else if (arr2.Dtype.TypeNum == NpyDefs.NPY_TYPES.NPY_STRING &&
                    arr1.Dtype.TypeNum == NpyDefs.NPY_TYPES.NPY_UNICODE) {
                    arr2 = PromoteToUnicode(arr2, arr1);
                }
                res = NpyCoreApi.CompareStringArrays(arr1, arr2, cmpOp, rstripFlag);
            } else {
                throw new ArgumentTypeException("comparison of non-string arrays");
            }
            return res;
        }

        private static ndarray PromoteToUnicode(ndarray arr, ndarray src) {
            dtype unicode = new dtype(src.Dtype);
            unicode.ElementSize = src.itemsize * 4;
            return NpyArray.FromAny(arr, unicode);
        }

        /// <summary>
        /// Creates a scalar instance representing the (scalar) type in typecode. 'obj' is used to
        /// initialize the value as either an object or a sequence of bytes that are re-interpreted
        /// as the value (not string, use b'foo bar').  That is, the byte string is interpreted as
        /// just a memory buffer that is turned into whatever the desired type is.  This is used for
        /// pickling/unpickling.
        /// </summary>
        /// <param name="typecode">Descriptor for the desired type</param>
        /// <param name="obj">Optional object or Bytes initializer</param>
        /// <returns>ScalarGeneric instance</returns>
        public static object scalar(CodeContext cntx, dtype typecode, object obj = null) {
            if (typecode.ElementSize == 0) {
                throw new ArgumentException("itermsize cannot be zero");
            }

            IntPtr dataPtr = IntPtr.Zero;
            int size;
            object ret = null;
            try {
                // What we are doing here is allocating a block of memory the same size as the typecode's
                // element size to build a scalar from.  This is either zero filled (obj ==null) or, if an
                // object a handle to the object or, as a last resort, using a byte array.  The byte array
                // is mostly used for pickling/unpickling as a way to stuff arbitrary byte data into a type.
                if (typecode.ChkFlags(NpyDefs.NPY_ITEM_IS_POINTER)) {
                    ret = new ScalarObject(obj);
                } else {
                    if (obj == null) {
                        dataPtr = Marshal.AllocHGlobal(typecode.ElementSize);
                        for (int i = 0; i < typecode.ElementSize; i++) Marshal.WriteByte(dataPtr, i, 0);
                        size = typecode.ElementSize;
                    } else {
                        dataPtr = Marshal.AllocHGlobal(typecode.ElementSize);
                        NumericOps.setitemUnicode(obj, dataPtr, typecode);
                        size = typecode.ElementSize;
                    }

                    ret = ScalarGeneric.ScalarFromData(typecode, dataPtr, size);
                }
            } finally {
                if (dataPtr != IntPtr.Zero) Marshal.FreeHGlobal(dataPtr);
            }
            return ret;
        }
    }
}
