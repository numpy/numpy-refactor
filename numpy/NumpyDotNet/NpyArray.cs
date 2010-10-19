using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Runtime.InteropServices;
using IronPython.Runtime;
using IronPython.Runtime.Operations;
using Microsoft.Scripting;

namespace NumpyDotNet {
    /// <summary>
    /// Implements array manipulation and construction functionality.  This
    /// class has functionality corresponding to functions in arrayobject.c, 
    /// ctors.c, and multiarraymodule.c
    /// </summary>
    internal static class NpyArray {


        /// <summary>
        /// Copies the source object into the destination array.  src can be
        /// any type so long as the number of elements matches dest.  In the
        /// case of strings, they will be padded with spaces if needed but
        /// can not be longer than the number of elements in dest.
        /// </summary>
        /// <param name="dest">Destination array</param>
        /// <param name="src">Source object</param>
        internal static void CopyObject(ndarray dest, Object src) {
            // For char arrays pad the input string.
            if (dest.dtype.Type == NpyDefs.NPY_TYPECHAR.NPY_CHARLTR &&
                dest.ndim > 0 && src is String) {
                int ndimNew = (int)dest.Dims[dest.ndim - 1];
                int ndimOld = ((String)src).Length;

                if (ndimNew > ndimOld) {
                    src = ((String)src).PadRight(ndimNew, ' ');
                }
            }

            ndarray srcArray;
            if (src is ndarray) {
                srcArray = (ndarray)src;
            } else if (false) {
                // TODO: Not handling scalars.  See arrayobject.c:111
            } else {
                srcArray = FromAny(src, dest.dtype, 0, dest.ndim, 
                                   dest.dtype.Flags & NpyDefs.NPY_FORTRAN, null);
            }
            NpyCoreApi.MoveInto(dest, srcArray);
        }


        internal static void SetField(ndarray dest, IntPtr descr, int offset, object src)
        {
            // For char arrays pad the input string.
            if (dest.dtype.Type == NpyDefs.NPY_TYPECHAR.NPY_CHARLTR &&
                dest.ndim > 0 && src is String)
            {
                int ndimNew = (int)dest.Dims[dest.ndim - 1];
                int ndimOld = ((String)src).Length;

                if (ndimNew > ndimOld)
                {
                    src = ((String)src).PadRight(ndimNew, ' ');
                }
            }
            ndarray srcArray;
            if (src is ndarray)
            {
                srcArray = (ndarray)src;
            }
            else if (false)
            {
                // TODO: Not handling scalars.  See arrayobject.c:111
            }
            else
            {
                srcArray = FromAny(src, dest.dtype, 0, dest.ndim,
                                   dest.dtype.Flags & NpyDefs.NPY_FORTRAN, null);
            }
            if (NpyCoreApi.NpyArray_SetField(dest.Array, descr, offset, srcArray.Array) < 0)
            {
                NpyCoreApi.CheckError();
            }
        }


        

        /// <summary>
        /// Checks the strides against the shape of the array.  This duplicates 
        /// NpyArray_CheckStrides and is only here because we don't currently support
        /// buffers and can simplify this function plus it's much faster to do here
        /// than to pass the arrays into the native world.
        /// </summary>
        /// <param name="elSize">Size of array element in bytes</param>
        /// <param name="shape">Size of each dimension of the array</param>
        /// <param name="strides">Stride of each dimension</param>
        /// <returns>True if strides are ok, false if not</returns>
        internal static bool CheckStrides(int elSize, long[] shape, long[] strides) {
            // Product of all dimension sizes * element size in bytes.
            long numbytes = shape.Aggregate(1L, (acc, x) => acc * x) * elSize;
            long end = numbytes - elSize;
            for (int i = 0; i < shape.Length; i++) {
                if (strides[i] * (shape[i] - 1) > end) return false;
            }
            return true;
        }


        internal static ndarray CheckFromArray(Object src, dtype descr, int minDepth,
            int maxDepth, int requires, Object context) {

                if ((requires & NpyDefs.NPY_NOTSWAPPED) != 0) {
                if (descr != null && src is ndarray &&
                    ((ndarray)src).dtype.IsNativeByteOrder) {
                    descr = new dtype(((ndarray)src).dtype);
                } else if (descr != null && !descr.IsNativeByteOrder) {
                    // Descr replace
                }
                if (descr != null) {
                    descr.ByteOrder = (byte)'=';
                }
            }

            ndarray arr = FromAny(src, descr, minDepth, maxDepth, requires, context);

            if (arr != null && (requires & NpyDefs.NPY_ELEMENTSTRIDES) != 0 &&
                arr.ElementStrides == 0) {
                    arr = arr.NewCopy(NpyDefs.NPY_ORDER.NPY_ANYORDER);
            }
            return arr;
        }


        private static Exception UpdateIfCopyError() {
            return new ArgumentException("UPDATEIFCOPY used for non-array input.");
        }

        /// <summary>
        /// Constructs a new array from multiple input types, like lists, arrays, etc.
        /// </summary>
        /// <param name="src"></param>
        /// <param name="descr"></param>
        /// <param name="minDepth"></param>
        /// <param name="maxDepth"></param>
        /// <param name="requires"></param>
        /// <param name="context"></param>
        /// <returns></returns>
        internal static ndarray FromAny(Object src, dtype descr=null, int minDepth=0,
            int maxDepth=0, int flags=0, Object context=null) {
            ndarray result = null;

            Type t = src.GetType();

            if (t != typeof(List) && t != typeof(PythonTuple)) { 
                if (src is ndarray) {
                    result = FromArray((ndarray)src, descr, flags);
                } 
            
                if (src is ScalarGeneric) {
                    if ((flags & NpyDefs.NPY_UPDATEIFCOPY)!=0) {
                        throw UpdateIfCopyError();
                    }
                    return FromScalar((ScalarGeneric)src, descr);
                }
            
                dtype newtype = (descr ?? FindScalarType(src));
                if (newtype != null) {
                    if ((flags & NpyDefs.NPY_UPDATEIFCOPY) != 0) {
                        throw UpdateIfCopyError();
                    }
                    return FromPythonScalar(src, newtype);
                } 

                // TODO: Handle buffer protocol
                // TODO: Look at __array_struct__ and __array_interface__
                result = FromArrayAttr(NpyUtil_Python.DefaultContext, src, descr, context);
                if (result != null) {
                    if (descr != null && !NpyCoreApi.EquivTypes(descr, result.dtype) || flags != 0) {
                        return FromArray(result, descr, flags);
                    }
                }
            }

            bool is_object = false;

            if ((flags&NpyDefs.NPY_UPDATEIFCOPY)!=0) {
                throw UpdateIfCopyError();
            }
            if (descr == null) {
                descr = FindArrayType(src, null);
            } else if (descr.TypeNum == NpyDefs.NPY_TYPES.NPY_OBJECT) {
                is_object = true;
            }

            if (src is IEnumerable<object>) {
                try {
                    return FromIEnumerable((IEnumerable<object>)src, descr, (flags & NpyDefs.NPY_FORTRAN) != 0, minDepth, maxDepth);
                } catch (InsufficientMemoryException) {
                    throw;
                } catch {
                    if (is_object) {
                        return FromNestedList(src, descr, (flags & NpyDefs.NPY_FORTRAN) != 0);
                    } else {
                        return FromPythonScalar(src, descr);
                    }
                }
            } else {
                return FromPythonScalar(src, descr);
            }
        }

        internal static ndarray FromNestedList(object src, dtype descr, bool fortran) {
            throw new NotImplementedException();
        }

        internal static ndarray FromArrayAttr(CodeContext cntx, object src, dtype descr, object context) {
            object f = PythonOps.ObjectGetAttribute(cntx, src, "__array__");
            if (f == null) {
                return null;
            }
            object result;
            if (context == null) {
                if (descr == null) {
                    result = PythonCalls.Call(cntx, f);
                } else {
                    result = PythonCalls.Call(cntx, f, descr);
                }
            } else {
                if (descr == null) {
                    try {
                        result = PythonCalls.Call(cntx, f, null, context);
                    } catch (ArgumentTypeException) {
                        result = PythonCalls.Call(cntx, f);
                    }
                } else {
                    try {
                        result = PythonCalls.Call(cntx, f, descr, context);
                    } catch (ArgumentTypeException) {
                        result = PythonCalls.Call(cntx, f, context);
                    }
                }
            }
            if (!(result is ndarray)) {
                throw new ArgumentException("object __array__ method not producing an array");
            }
            return (ndarray)result;
        }

        internal static ndarray FromScalar(ScalarGeneric scalar, dtype descr = null) {
            if (descr == null || NpyCoreApi.EquivTypes(scalar.dtype, descr)) {
                return scalar.ToArray();
            } else {
                ndarray arr = scalar.ToArray();
                return FromArray(arr, descr, 0);
            }
        }

        /// <summary>
        /// Constructs a new array from an input array and descriptor type.  The
        /// Underlying array may or may not be copied depending on the requirements.
        /// </summary>
        /// <param name="src">Source array</param>
        /// <param name="descr">Desired type</param>
        /// <param name="flags">New array flags</param>
        /// <returns>New array (may be source array)</returns>
        internal static ndarray FromArray(ndarray src, dtype descr, int flags) {
            if (descr == null && flags == 0) return src;
            if (descr == null) descr = src.dtype;
            if (descr != null) NpyCoreApi.Incref(descr.Descr);
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_FromArray(src.Array, descr.Descr, flags));
        }


        internal static ndarray FromPythonScalar(object src, dtype descr) {
            int itemsize = descr.ElementSize;
            NpyDefs.NPY_TYPES type = descr.TypeNum;

            if (itemsize == 0 && NpyDefs.IsExtended(type)) {
                int n = PythonOps.Length(src);
                if (type == NpyDefs.NPY_TYPES.NPY_UNICODE) {
                    n *= 4;
                }
                descr = new dtype(descr);
                descr.ElementSize = n;
            }
 
            ndarray result = NpyCoreApi.AllocArray(descr, 0, null, false);
            if (result.ndim > 0) {
                throw new ArgumentException("shape-mismatch on array construction");
            }

            result.dtype.f.SetItem(src, 0, result);
            return result;
        }


        /// <summary>
        /// Builds an array from a sequence of objects.  The elements of the sequence
        /// can also be sequences in which case this function recursively walks the
        /// nested sequences and builds an n dimentional array.
        /// 
        /// IronPython tuples and lists work as sequences.
        /// </summary>
        /// <param name="src">Input sequence</param>
        /// <param name="descr">Desired array element type or null to determine automatically</param>
        /// <param name="fortran">True if array should be Fortran layout, false for C</param>
        /// <param name="minDepth"></param>
        /// <param name="maxDepth"></param>
        /// <returns>New array instance</returns>
        internal static ndarray FromIEnumerable(IEnumerable<Object> src, dtype descr, 
            bool fortran, int minDepth, int maxDepth) {
            ndarray result = null;
            
            if (descr == null) {
                descr = FindArrayType(src, null, NpyDefs.NPY_MAXDIMS);
            }

            int itemsize = descr.ElementSize;

            NpyDefs.NPY_TYPES type = descr.TypeNum;
            bool checkIt = (descr.Type == NpyDefs.NPY_TYPECHAR.NPY_CHARLTR);
            bool stopAtString =
                type != NpyDefs.NPY_TYPES.NPY_STRING ||
                descr.Type == NpyDefs.NPY_TYPECHAR.NPY_STRINGLTR;
            bool stopAtTuple =
                type == NpyDefs.NPY_TYPES.NPY_VOID &&
                (descr.HasNames || descr.HasSubarray);

            int numDim = DiscoverDepth(src, NpyDefs.NPY_MAXDIMS + 1, stopAtString, stopAtTuple);
            if (numDim == 0) {
                return FromPythonScalar(src, descr);
            } else {
                if (maxDepth > 0 && type == NpyDefs.NPY_TYPES.NPY_OBJECT &&
                    numDim > maxDepth) {
                    numDim = maxDepth;
                }   
                if (maxDepth > 0 && numDim > maxDepth ||
                    minDepth > 0 && numDim < minDepth) {
                    throw new ArgumentException("Invalid number of dimensions.");
                }

                long[] dims = new long[numDim];
                DiscoverDimensions(src, numDim, dims, 0, checkIt);
                if (descr.Type == NpyDefs.NPY_TYPECHAR.NPY_CHARLTR &&
                    numDim > 0 && dims[numDim - 1] == 1) {
                    numDim--;
                }

                if (itemsize == 0 && NpyDefs.IsExtended(descr.TypeNum)) {
                    itemsize = DiscoverItemsize(src, numDim, 0);
                    if (descr.TypeNum == NpyDefs.NPY_TYPES.NPY_UNICODE) {
                        itemsize *= 4;
                    }
                    descr = new dtype(descr);
                    descr.ElementSize = itemsize;
                }

                result = NpyCoreApi.AllocArray(descr, numDim, dims, fortran);
                AssignToArray(src, result);
            }
            return result;
        }
        
        internal static ndarray PrependOnes(ndarray arr, int nd, int ndmin) {
            IntPtr[] newdims = new IntPtr[ndmin];
            IntPtr[] newstrides = new IntPtr[ndmin];
            int num = ndmin - nd;
            // Set the first num dims and strides for the 1's
            for (int i=0; i<num; i++) {
                newdims[i] = (IntPtr)1;
                newstrides[i] = (IntPtr)arr.dtype.ElementSize;
            }
            // Copy in the rest of dims and strides
            for (int i=num; i<ndmin; i++) {
                int k = i-num;
                newdims[i] = (IntPtr)arr.Dims[k];
                newstrides[i] = (IntPtr)arr.strides[k];
            }

            return NpyCoreApi.NewView(arr.dtype, ndmin, newdims, newstrides, arr, IntPtr.Zero, false);
        }

        internal static dtype FindArrayType(Object src, dtype minitype, int max = NpyDefs.NPY_MAXDIMS) {
            dtype chktype = null;

            if (src is ndarray) {
                chktype = ((ndarray)src).dtype;
                if (minitype == null) return chktype;
            } else {
                chktype = FindScalarType(src);
                if (chktype != null && minitype == null) return chktype;
            }
            
            // If a minimum type wasn't give, default to bool.
            if (minitype == null)
                minitype = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_BOOL);

            if (max >= 0) {
                chktype = FindScalarType(src);
                if (chktype == null) {
                    // TODO: No handling for PyBytes (common.c:133)
                    // TODO: No handling for Unicode (common.c:139)
                    // TODO: No handling for __array_interface property (common.c:175)
                    // TODO: No handling for __array_struct property (common.c:191)
                    // TODO: No handling for __array__ property (common.c:221)

                    if (src is IEnumerable<Object>) {
                        IEnumerable<Object> seq = (IEnumerable<Object>)src;

                        if (seq.Count() == 0 && minitype.TypeNum == NpyDefs.NPY_TYPES.NPY_BOOL) {
                            minitype = NpyCoreApi.DescrFromType(NpyDefs.DefaultType);
                        }
                        minitype = seq.Aggregate(minitype,
                            (acc, obj) => NpyCoreApi.SmallType(FindArrayType(obj, acc, max - 1), acc));
                        chktype = minitype;
                    }
                }
            }

            // Still nothing? Fall back to the default.
            if (chktype == null)
                chktype = UseDefaultType(src);

            // Final clean up, pick the min of the two types.  Void types
            // should only appear if the input was already void.
            chktype = NpyCoreApi.SmallType(chktype, minitype);
            if (chktype.TypeNum == NpyDefs.NPY_TYPES.NPY_VOID &&
                minitype.TypeNum != NpyDefs.NPY_TYPES.NPY_VOID) {
                    chktype = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_OBJECT);
            }
            return chktype;
        }

        private static dtype UseDefaultType(Object src) {
            // TODO: User-defined types are not implemented yet.
            return NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_OBJECT);
        }


        /// <summary>
        /// Returns the descriptor for a given native type or null if src is
        /// not a scalar type
        /// </summary>
        /// <param name="src">Object to type</param>
        /// <returns>Descriptor for type of 'src' or null if not scalar</returns>
        internal static dtype FindScalarType(Object src) {
            NpyDefs.NPY_TYPES type;

            // TODO: Complex numbers not handled.  
            // TODO: Are int32/64 -> long, longlong correct?
            if (src is Double) type = NpyDefs.NPY_TYPES.NPY_DOUBLE;
            else if (src is Single) type = NpyDefs.NPY_TYPES.NPY_FLOAT;
            else if (src is Boolean) type = NpyDefs.NPY_TYPES.NPY_BOOL;
            else if (src is Byte) type = NpyDefs.NPY_TYPES.NPY_BYTE;
            else if (src is Int16) type = NpyDefs.NPY_TYPES.NPY_SHORT;
            else if (src is Int32) type = NpyCoreApi.TypeOf_Int32;
            else if (src is Int64) type = NpyCoreApi.TypeOf_Int64;
            else if (src is UInt16) type = NpyDefs.NPY_TYPES.NPY_USHORT;
            else if (src is UInt32) type = NpyCoreApi.TypeOf_UInt32;
            else if (src is UInt64) type = NpyCoreApi.TypeOf_UInt64;
            else if (src is BigInteger) type = NpyDefs.NPY_TYPES.NPY_LONG;
            else if (src is Complex) type = NpyDefs.NPY_TYPES.NPY_CDOUBLE;
            else if (src is string) type = NpyDefs.NPY_TYPES.NPY_UNICODE;
            else if (src is Bytes) type = NpyDefs.NPY_TYPES.NPY_STRING;
            else type = NpyDefs.NPY_TYPES.NPY_NOTYPE;

            return (type != NpyDefs.NPY_TYPES.NPY_NOTYPE) ?
                NpyCoreApi.DescrFromType(type) : null;
        }


        /// <summary>
        /// Recursively discovers the nesting depth of a source object.  
        /// </summary>
        /// <param name="src">Input object</param>
        /// <param name="max">Max recursive depth</param>
        /// <param name="stopAtString">Stop discovering if string is encounted</param>
        /// <param name="stopAtTuple">Stop discovering if tuple is encounted</param>
        /// <returns>Nesting depth or -1 on error</returns>
        private static int DiscoverDepth(Object src, int max,
            bool stopAtString, bool stopAtTuple) {
            int d = 0;

            if (max < 1) {
                throw new ArgumentException("invalid input sequence");
            }

            if (src is IEnumerable<Object>) {
                IEnumerable<Object> seq = (IEnumerable<Object>)src;

                if (stopAtTuple && seq is IronPython.Runtime.PythonTuple)
                    d = 0;
                else if (seq.Count() == 0) d = 1;
                else {
                    d = DiscoverDepth(seq.First(), max - 1, stopAtString, stopAtTuple);
                    if (d >= 0) d++;
                }
            } else if (src is ndarray) {
                d = ((ndarray)src).ndim;
            } else if (src is String) {
                d = stopAtString ? 0 : 1;
            }
                // TODO: Not handling __array_struct__ attribute
                // TODO: Not handling __array_interface__ attribute
            else d = 0;
            return d;
        }


        /// <summary>
        /// Recursively discovers the size of each dimension given an input object.
        /// </summary>
        /// <param name="src">Input object</param>
        /// <param name="numDim">Number of dimensions</param>
        /// <param name="dims">Uninitialized array of dimension sizes to be filled in</param>
        /// <param name="dimIdx">Current index into dims, incremented recursively</param>
        /// <param name="checkIt">Verify that src is consistent</param>
        private static void DiscoverDimensions(Object src, int numDim,
            Int64[] dims, int dimIdx, bool checkIt) {

            if (src is ndarray) {
                ndarray arr = (ndarray)src;
                if (arr.ndim == 0) dims[dimIdx] = 0;
                else {
                    Int64[] d = arr.Dims;
                    for (int i = 0; i < numDim; i++) {
                        dims[i + dimIdx] = d[i];
                    }
                }
            } else if (src is IEnumerable<Object>) {
                IEnumerable<Object> seq = (IEnumerable<Object>)src;

                Int64 nLowest = 0;
                dims[dimIdx] = seq.Count();
                if (numDim > 1 && dims[dimIdx] > 1) {
                    foreach (Object o in seq) {
                        DiscoverDimensions(o, numDim - 1, dims, dimIdx + 1, checkIt);
                        if (checkIt && nLowest != 0 && nLowest != dims[dimIdx + 1]) {
                            throw new ArgumentException("Inconsistent shape in sequence");
                        }
                        if (dims[dimIdx + 1] > nLowest) nLowest = dims[dimIdx + 1];
                    }
                    dims[dimIdx + 1] = nLowest;
                }
            } else {
                // Scalar condition.
                dims[dimIdx] = 1;
            }
        }

        private static int DiscoverItemsize(object s, int nd, int min) {
            if (s is ndarray) {
                ndarray a = (ndarray)s;
                return Math.Max(min, a.dtype.ElementSize);
            }
            int n = (int)NpyUtil_Python.CallBuiltin(null, "len", s);
            if (nd == 0 || s is string || s is Bytes || s is MemoryView || s is PythonBuffer) {
                return Math.Max(min, n);
            } else {
                int result = min;
                for (int i = 0; i < n; i++) {
                    object item = PythonOps.GetIndex(NpyUtil_Python.DefaultContext, s, i);
                    result = DiscoverItemsize(item, nd - 1, result);
                }
                return result;
            }
        }

        internal static ndarray Empty(long[] shape, dtype type = null, NpyDefs.NPY_ORDER order = NpyDefs.NPY_ORDER.NPY_CORDER) {
            if (type == null) {
                type = NpyCoreApi.DescrFromType(NpyDefs.DefaultType);
            }
            return NpyCoreApi.NewFromDescr(type, shape, null, (int)order, null);
        }

        internal static ndarray Zeros(long[] shape, dtype type = null, NpyDefs.NPY_ORDER order = NpyDefs.NPY_ORDER.NPY_CORDER) {
            ndarray result = Empty(shape, type, order);
            NpyCoreApi.NpyArrayAccess_ZeroFill(result.Array, IntPtr.Zero);
            if (type.IsObject) {
                FillObjects(result, 0);
            }
            return result;
        }

        internal static ndarray Arange(CodeContext cntx, object start, object stop = null, object step = null, dtype d = null) {
            long[] dims;

            if (d == null) {
                d = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_LONG);
                d = FindArrayType(start, d);
                if (stop != null) {
                    d = FindArrayType(stop, d);
                }
                if (step != null) {
                    d = FindArrayType(step, d);
                }

            }
            if (step == null) {
                step = 1;
            }
            if (stop == null) {
                stop = start;
                start = 0;
            }

            object next;
            IntPtr len = CalcLength(cntx, start, stop, step, out next, NpyDefs.IsComplex(d.TypeNum));
            if (len.ToInt64() < 0) {
                dims = new long[] { 0 };
                return NpyCoreApi.NewFromDescr(d, dims, null, 0, null);
            }

            dtype native;
            bool swap;
            if (!d.IsNativeByteOrder) {
                native = NpyCoreApi.DescrNewByteorder(d, '=');
                swap = true;
            } else {
                native = d;
                swap = false;
            }

            dims = new long[] { len.ToInt64() };
            ndarray result = NpyCoreApi.NewFromDescr(native, dims, null, 0, null);
            result.SetItem(start, 0);
            result.SetItem(next, d.ElementSize);

            if (len.ToInt64() > 2) {
                NpyCoreApi.Fill(result);
            }
            if (swap) {
                NpyCoreApi.Byteswap(result, true);
                result.dtype = d;
            }
            return result;
        }

        internal static IntPtr CeilToIntPtr(double d) {
            d = Math.Ceiling(d);
            if (IntPtr.Size == 4) {
                if (d > int.MaxValue || d < int.MinValue) {
                    throw new OverflowException();
                }
                return (IntPtr)(int)d;
            } else {
                if (d > long.MaxValue || d < long.MinValue) {
                    throw new OverflowException();
                }
                return (IntPtr)(long)d;
            }
        }

        internal static IntPtr CalcLength(CodeContext cntx, object start, object stop, object step, out object next, bool complex) {
            dynamic ops = PythonOps.ImportTop(cntx, "operator", 0);
            object n = ops.sub(stop, start);
            object val = ops.truediv(n, step);
            IntPtr result;
            if (complex && val is Complex) {
                Complex c = (Complex)val;
                result = CeilToIntPtr(Math.Min(c.Real, c.Imaginary));
            } else {
                double d = Convert.ToDouble(val);
                result = CeilToIntPtr(d);
            }
            next = ops.add(start, step);
            return result;
        }

        internal static void FillObjects(ndarray arr, object o) {
            dtype d = arr.dtype;
            if (d.IsObject) {
                if (d.HasNames) {
                    foreach (string name in d.Names) {
                        using (ndarray view = NpyCoreApi.GetField(arr, name)) {
                            FillObjects(view, o);
                        }
                    }
                } else {
                    NpyCoreApi.FillWithObject(arr, o);
                }
            }
        }

        internal static void AssignToArray(Object src, ndarray result) {
            if (src is IEnumerable<Object>) {
                AssignFromSeq((IEnumerable<Object>)src, result, 0, 0);
            } else {
                // TODO: Assign from array and other types is not implemented.
                throw new NotImplementedException(
                    String.Format("Assign to array from type '{0}' is not yet implemented.",
                        src.GetType().Name));
            }
        }

        private static void AssignFromSeq(IEnumerable<Object> seq, ndarray result,
            int dim, long offset) {
            if (dim >= result.ndim) {
                throw new IronPython.Runtime.Exceptions.RuntimeException(
                    String.Format("Source dimensions ({0}) exceeded target array dimensions ({1}).",
                    dim, result.ndim));
            }

            if (seq.Count() != result.Dims[dim]) {
                throw new IronPython.Runtime.Exceptions.RuntimeException(
                    "AssignFromSeq: sequence/array shape mismatch.");
            }

            long stride = result.Stride(dim);
            if (dim < result.ndim - 1) {
                // Sequence elements should be additional sequences
                seq.Iteri((o, i) =>
                    AssignFromSeq((IEnumerable<Object>)o, result, dim + 1, offset + stride * i));
            } else {
                seq.Iteri((o, i) => result.dtype.f.SetItem(o, offset + i*stride, result));
            }
        }
    }
}

