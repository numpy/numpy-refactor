using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Runtime.InteropServices;
using IronPython.Runtime;
using IronPython.Runtime.Operations;
using IronPython.Runtime.Types;
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
                dtype src_dtype = NpyCoreApi.ToInterface<dtype>(descr);
                srcArray = FromAny(src, src_dtype, 0, dest.ndim,
                                   dest.dtype.Flags & NpyDefs.NPY_FORTRAN, null);
            }
            NpyCoreApi.Incref(descr);
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

        private static ndarray FromAnyReturn(ndarray result, int minDepth, int maxDepth) {
            if (minDepth != 0 && result.ndim < minDepth) {
                throw new ArgumentException("object of too small depth for desired array");
            }
            if (maxDepth != 0 && result.ndim > maxDepth) {
                throw new ArgumentException("object too deep for desired array");
            }
            return result;
        }

        internal static ndarray EnsureArray(object o) {
            if (o == null) {
                return null;
            }
            if (o.GetType() == typeof(ndarray)) {
                return (ndarray)o;
            }
            if (o is ndarray) {
                return FromArray((ndarray)o, null, NpyDefs.NPY_ENSUREARRAY);
            }
            return FromAny(o, flags: NpyDefs.NPY_ENSUREARRAY);
        }

        internal static ndarray EnsureAnyArray(object o) {
            if (o == null) {
                return null;
            }
            if (o is ndarray) {
                return (ndarray)o;
            }
            return FromAny(o, flags: NpyDefs.NPY_ENSUREARRAY);
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

            if (src == null) {
                return Empty(new long[0], NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_OBJECT));
            }

            Type t = src.GetType();

            if (t != typeof(List) && t != typeof(PythonTuple)) { 
                if (src is ndarray) {
                    result = FromArray((ndarray)src, descr, flags);
                    return FromAnyReturn(result, minDepth, maxDepth);
                }
                if (src is ScalarGeneric) {
                    if ((flags & NpyDefs.NPY_UPDATEIFCOPY)!=0) {
                        throw UpdateIfCopyError();
                    }
                    result = FromScalar((ScalarGeneric)src, descr);
                    return FromAnyReturn(result, minDepth, maxDepth);
                }

                dtype newtype = (descr ?? FindScalarType(src));
                if (descr == null && newtype != null) {
                    if ((flags & NpyDefs.NPY_UPDATEIFCOPY) != 0) {
                        throw UpdateIfCopyError();
                    }
                    result = FromPythonScalar(src, newtype);
                    return FromAnyReturn(result, minDepth, maxDepth);
                } 

                // TODO: Handle buffer protocol
                // TODO: Look at __array_struct__ and __array_interface__
                result = FromArrayAttr(NpyUtil_Python.DefaultContext, src, descr, context);
                if (result != null) {
                    if (descr != null && !NpyCoreApi.EquivTypes(descr, result.dtype) || flags != 0) {
                        result = FromArray(result, descr, flags);
                        return FromAnyReturn(result, minDepth, maxDepth);
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

            bool seq = false;
            if (src is IEnumerable<object>) {
                try {
                    result = FromIEnumerable((IEnumerable<object>)src, descr, (flags & NpyDefs.NPY_FORTRAN) != 0, minDepth, maxDepth);
                    seq = true;
                } catch (InsufficientMemoryException) {
                    throw;
                } catch {
                    if (is_object) {
                        result = FromNestedList(src, descr, (flags & NpyDefs.NPY_FORTRAN) != 0);
                        seq = true;
                    } 
                }
            }
            if (!seq) {
                result = FromPythonScalar(src, descr);
            }
            return FromAnyReturn(result, minDepth, maxDepth);
        }

        internal static ndarray FromNestedList(object src, dtype descr, bool fortran) {
            long[] dims = new long[NpyDefs.NPY_MAXDIMS];
            int nd = ObjectDepthAndDimension(src, dims, 0, NpyDefs.NPY_MAXDIMS);
            if (nd == 0) {
                return FromPythonScalar(src, descr);
            }
            ndarray result = NpyCoreApi.AllocArray(descr, nd, dims, fortran);
            AssignToArray(src, result);
            return result;
        }

        /// <summary>
        /// Walks a set of nested lists (or tuples) to get the dimensions.  The dimensionality must
        /// be consistent for each nesting level. Thus, if one level is a mix of lsits and scalars,
        /// it is truncated and all are assumed to be scalar objects.
        /// 
        /// That is, [[1, 2], 3, 4] is a 1-d array of 3 elements.  It just happens that element 0 is
        /// an object that is a list of [1, 2].
        /// </summary>
        /// <param name="src">Input object to talk</param>
        /// <param name="dims">Array of dimensions of size 'max' filled in up to the return value</param>
        /// <param name="idx">Current iteration depth, always start with 0</param>
        /// <param name="max">Size of dims array at the start, then becomes depth so far when !firstElem</param>
        /// <param name="firstElem">True if processing the first element of the list (populates dims), false for subsequent (checks dims)</param>
        /// <returns>Number of dimensions (depth of nesting)</returns>
        internal static int ObjectDepthAndDimension(object src, long[] dims, int idx, int max, bool firstElem=true)
        {
            int nd = -1;

            // Recursively walk the tree and get the sizes of each dimension. When processing the
            // first element in each sequence, firstElem is true and we populate dims[]. After that,
            // we just verify that dims[] matches for subsequent elements.
            IList<object> list = src as IList<object>;  // List and PythonTuple both implement IList
            if (max < 1 || list == null) {
                nd = 0;
            } else if (list.Count == 0) {
                nd = 0;
            } else if (max < 2) {
                // On the first pass, populate the dimensions array. One subsequent passes verify
                // that the size is the same or, if not, 
                if (firstElem) {
                    dims[idx] = list.Count;
                    nd = 1;
                } else {
                    nd = (dims[idx] == list.Count) ? 1 : 0;
                }
            } else if (!firstElem && dims[idx] != list.Count) {
                nd = 0;
            } else {
                // First element we traverse up to max depth and fill in the dims array.
                nd = ObjectDepthAndDimension(list.First(), dims, idx + 1, max - 1, firstElem);

                // Subsequent elements we just check that the size of each dimension is the
                // same as clip the max depth to shallowest depth we have seen thus far.
                nd = list.Skip(1).Aggregate(nd, (ndAcc, elem) =>
                    Math.Min(ndAcc, ObjectDepthAndDimension(elem, dims, idx + 1, ndAcc, false))
                );
                nd += 1;
                dims[idx] = list.Count;
            }
            return nd;
        }

        internal static ndarray FromArrayAttr(CodeContext cntx, object src, dtype descr, object context) {
            object f;
            if (src is PythonType ||
                !PythonOps.TryGetBoundAttr(cntx, src, "__array__", out f)) {
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
                // passing scalar.dtype instead of descr in because otherwise we loose information. Not
                // sure if more processing is needed.  Relevant CPython code is PyArray_DescrFromScalarUnwrap
                return FromArray(arr, scalar.dtype, 0);
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
            bool checkIt = (descr.Type != NpyDefs.NPY_TYPECHAR.NPY_CHARLTR);
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
                newstrides[i] = (IntPtr)arr.Strides[k];
            }

            return NpyCoreApi.NewView(arr.dtype, ndmin, newdims, newstrides, arr, IntPtr.Zero, false);
        }

        private static dtype FindArrayReturn(dtype chktype,  dtype minitype) {
            dtype result = NpyCoreApi.SmallType(chktype, minitype);
            if (result.TypeNum == NpyDefs.NPY_TYPES.NPY_VOID &&
                minitype.TypeNum != NpyDefs.NPY_TYPES.NPY_VOID) {
                result = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_OBJECT);
            }
            return result;
        }

        internal static dtype FindArrayType(Object src, dtype minitype, int max = NpyDefs.NPY_MAXDIMS) {
            dtype chktype = null;

            if (src is ndarray) {
                chktype = ((ndarray)src).dtype;
                if (minitype == null) {
                    return chktype;
                } else {
                    return FindArrayReturn(chktype, minitype);
                }
            }

            if (src is ScalarGeneric) {
                chktype = ((ScalarGeneric)src).dtype;
                if (minitype == null) {
                    return chktype;
                } else {
                    return FindArrayReturn(chktype, minitype);
                }
            }

            if (minitype == null) {
                minitype = NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_BOOL);
            }
            if (max < 0) {
                chktype = UseDefaultType(src);
                return FindArrayReturn(chktype, minitype);
            }

            chktype = FindScalarType(src);
            if (chktype != null) {
                return FindArrayReturn(chktype, minitype);
            }

            if (src is Bytes) {
                Bytes b = (Bytes)src;
                chktype = new dtype(NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_STRING));
                chktype.ElementSize = b.Count;
                return FindArrayReturn(chktype, minitype);
            }

            if (src is String) {
                String s = (String)src;
                chktype = new dtype(NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_UNICODE));
                chktype.ElementSize = s.Length*4;
                return FindArrayReturn(chktype, minitype);
            }

            // TODO: Handle buffer protocol
            // TODO: __array_interface__
            // TODO: __array_struct__
            CodeContext cntx = NpyUtil_Python.DefaultContext;
            object arrayAttr;
            if (PythonOps.TryGetBoundAttr(cntx, src, "__array__", out arrayAttr)) {
                try {
                    object ip = PythonCalls.Call(cntx, arrayAttr);
                    if (ip is ndarray) {
                        chktype = ((ndarray)ip).dtype;
                        return FindArrayReturn(chktype, minitype);
                    }
                } catch {
                    // Ignore errors
                }
            }
            // TODO: PyInstance_Check?
            if (src is IEnumerable<object>) {
                // TODO: This does not work for user-defined Python sequences
                int l;
                try {
                    l = PythonOps.Length(src);
                } catch {
                    chktype = UseDefaultType(src);
                    return FindArrayReturn(chktype, minitype);
                }
                if (l == 0 && minitype.TypeNum == NpyDefs.NPY_TYPES.NPY_BOOL) {
                    minitype = NpyCoreApi.DescrFromType(NpyDefs.DefaultType);
                }
                while (--l >= 0) {
                    object item;
                    try {
                        item = PythonOps.GetIndex(cntx, src, l);
                    } catch {
                        chktype = UseDefaultType(src);
                        return FindArrayReturn(chktype, minitype);
                    }
                    chktype = FindArrayType(item, minitype, max-1);
                    minitype = NpyCoreApi.SmallType(chktype, minitype);
                }
                chktype = minitype;
                return chktype;
            }

            chktype = UseDefaultType(src);
            return FindArrayReturn(chktype, minitype);
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
            else if (src is BigInteger) {
                BigInteger bi = (BigInteger)src;
                if (Int64.MinValue <= bi && bi <= Int64.MaxValue) {
                    type = NpyCoreApi.TypeOf_Int64;
                } else {
                    type = NpyDefs.NPY_TYPES.NPY_OBJECT;
                }
            }
            else if (src is Complex) type = NpyDefs.NPY_TYPES.NPY_CDOUBLE;
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

            if (stopAtTuple && src is PythonTuple) {
                return 0;
            }
            if (src is string) {
                return (stopAtString ? 0 : 1);
            }

            if (src is ndarray) {
                return ((ndarray)src).ndim;
            }

            if (src is IList<object>) {
                IList<object> list = (IList<object>)src;
                if (list.Count == 0) {
                    return 1;
                } else {
                    d = DiscoverDepth(list[0], max-1, stopAtString, stopAtTuple);
                    return d+1;
                }
            }
            
            if (src is IEnumerable<object>) {
                IEnumerable<object> seq = (IEnumerable<object>)src;
                object first;
                try {
                    first = seq.First();
                } catch (InvalidOperationException) {
                    // Empty sequence
                    return 1;
                }
                d = DiscoverDepth(first, max-1, stopAtString, stopAtTuple);
                return d+1;
            }

                // TODO: Not handling __array_struct__ attribute
                // TODO: Not handling __array_interface__ attribute
            return 0;
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
            Int64 nLowest;

            if (src is ndarray) {
                ndarray arr = (ndarray)src;
                if (arr.ndim == 0) dims[dimIdx] = 0;
                else {
                    Int64[] d = arr.Dims;
                    for (int i = 0; i < numDim; i++) {
                        dims[i + dimIdx] = d[i];
                    }
                }
            } else if (src is IList<object>) {
                IList<object> seq = (IList<object>)src;

                nLowest = 0;
                dims[dimIdx] = seq.Count();
                if (numDim > 1) {
                    foreach (Object o in seq) {
                        DiscoverDimensions(o, numDim - 1, dims, dimIdx + 1, checkIt);
                        if (checkIt && nLowest != 0 && nLowest != dims[dimIdx + 1]) {
                            throw new ArgumentException("Inconsistent shape in sequence");
                        }
                        if (dims[dimIdx + 1] > nLowest) nLowest = dims[dimIdx + 1];
                    }
                    dims[dimIdx + 1] = nLowest;
                }
            }
            else if (src is IEnumerable<Object>) {
                IEnumerable<Object> seq = (IEnumerable<Object>)src;

                nLowest = 0;
                dims[dimIdx] = seq.Count();
                if (numDim > 1) {
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
                // Object arrays are zero filled when created
                FillObjects(result, 0);
            } else {
                NpyCoreApi.NpyArrayAccess_ZeroFill(result.Array, IntPtr.Zero);
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
            if (len.ToInt64() > 1) {
                result.SetItem(next, d.ElementSize);
            }

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
            IEnumerable<object> seq = src as IEnumerable<object>;
            if (seq == null) {
                throw new ArgumentException("assignment from non-sequence");
            }
            if (result.ndim == 0) {
                throw new ArgumentException("assignment to 0-d array");
            }
            AssignFromSeq(seq, result, 0, 0);
        }

        private static void AssignFromSeq(IEnumerable<Object> seq, ndarray result,
            int dim, long offset) {
            if (dim >= result.ndim) {
                throw new IronPython.Runtime.Exceptions.RuntimeException(
                    String.Format("Source dimensions ({0}) exceeded target array dimensions ({1}).",
                    dim, result.ndim));
            }

            if (seq is ndarray && seq.GetType() != typeof(ndarray)) {
                // Convert to an array to ensure the dimensionality reduction 
                // assumption works.
                ndarray array = FromArray((ndarray)seq, null, NpyDefs.NPY_ENSUREARRAY);
                seq = (IEnumerable<object>)array;
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

        internal static ndarray Concatenate(IEnumerable<object> arrays, int axis) {
            int i;

            try {
                arrays.First();
            } catch (InvalidOperationException) {
                throw new ArgumentException("concatenation of zero-length sequence is impossible");
            }
       
            ndarray[] mps = NpyUtil_ArgProcessing.ConvertToCommonType(arrays);
            int n = mps.Length;
            // TODO: Deal with subtypes
            if (axis >= NpyDefs.NPY_MAXDIMS) {
                // Flatten the arrays
                for (i = 0; i < n; i++) {
                    mps[i] = mps[i].Ravel(NpyDefs.NPY_ORDER.NPY_CORDER);
                }
            } else if (axis != 0) {
                // Swap to make the axis 0
                for (i = 0; i < n; i++) {
                    mps[i] = NpyArray.FromArray(mps[i].SwapAxes(axis, 0), null, NpyDefs.NPY_C_CONTIGUOUS);
                }
            }
            long[] dims = mps[0].Dims;
            if (dims.Length == 0) {
                throw new ArgumentException("0-d arrays can't be concatenated");
            }
            long new_dim = dims[0];
            for (i = 1; i < n; i++) {
                long[] dims2 = mps[i].Dims;
                if (dims.Length != dims2.Length) {
                    throw new ArgumentException("arrays must have same number of dimensions");
                }
                bool eq = Enumerable.Zip(dims.Skip(1), dims2.Skip(1), (a, b) => (a == b)).All(x => x);
                if (!eq) {
                    throw new ArgumentException("array dimensions do not agree");
                }
                new_dim += dims2[0];
            }
            dims[0] = new_dim;
            ndarray result = NpyCoreApi.AllocArray(mps[0].dtype, dims.Length, dims, false);
            // TODO: We really should be doing a memcpy here.
            unsafe {
                byte* dest = (byte*)result.UnsafeAddress.ToPointer();
                foreach (ndarray a in mps) {
                    long s = a.Size*a.dtype.ElementSize;
                    byte* src = (byte*)a.UnsafeAddress;
                    while (s-- > 0) {
                        *dest++ = *src++;
                    }
                }
            }
            if (0 < axis && axis < NpyDefs.NPY_MAXDIMS || axis < 0) {
                return result.SwapAxes(axis, 0);
            } else {
                return result;
            }
        }

        internal static ndarray InnerProduct(object o1, object o2) {
            dtype d = FindArrayType(o1, null);
            d = FindArrayType(o2, d);

            ndarray a1 = FromAny(o1, d, flags: NpyDefs.NPY_ALIGNED);
            ndarray a2 = FromAny(o2, d, flags: NpyDefs.NPY_ALIGNED);
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_InnerProduct(a1.Array, a2.Array, (int)d.TypeNum));
        }

        internal static ndarray MatrixProduct(object o1, object o2) {
            dtype d = FindArrayType(o1, null);
            d = FindArrayType(o2, d);

            ndarray a1 = FromAny(o1, d, flags: NpyDefs.NPY_ALIGNED);
            ndarray a2 = FromAny(o2, d, flags: NpyDefs.NPY_ALIGNED);
            if (a1.ndim == 0) {
                return NpyArray.EnsureAnyArray(a1.item() * a2);
            } else if (a2.ndim == 0) {
                return NpyArray.EnsureAnyArray(a1 * a2.item());
            } else {
                return NpyCoreApi.DecrefToInterface<ndarray>(
                    NpyCoreApi.NpyArray_MatrixProduct(a1.Array, a2.Array, (int)d.TypeNum));
            }
        }
    }
}

