using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

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
                    descr.ByteOrder = NpyCoreApi.NativeByteOrder;
                }
            }

            ndarray arr = FromAny(src, descr, minDepth, maxDepth, requires, context);

            if (arr != null && (requires & NpyDefs.NPY_ELEMENTSTRIDES) != 0 &&
                arr.ElementStrides == 0) {
                    arr = arr.NewCopy(NpyDefs.NPY_ORDER.NPY_ANYORDER);
            }
            return arr;
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
        internal static ndarray FromAny(Object src, dtype descr, int minDepth,
            int maxDepth, int flags, Object context) {
            ndarray result = null;

            if (src is ndarray) {
                result = FromArray((ndarray)src, descr, flags);
            } else {
                dtype type = FindScalarType(src);
                if (type != null) {
                    result = FromScalar(src, (descr != null ? descr : type));
                } else {
                    if ((flags & NpyDefs.NPY_UPDATEIFCOPY) != 0)
                        throw new IronPython.Runtime.Exceptions.RuntimeException("UPDATEIFCOPY used for non-array input");

                    if (src is IEnumerable<Object>) {
                        Console.WriteLine("Enumerable type = {0}", src.GetType().ToString());
                        result = FromIEnumerable((IEnumerable<Object>)src, descr,
                            (flags & NpyDefs.NPY_FORTRAN) != 0, minDepth, maxDepth);
                    } else {
                        throw new NotImplementedException(
                            String.Format("In FromArray, type {0} is not handled yet.", src.GetType().ToString()));
                    }
                }
            }
            return result;
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
            if (descr != null) NpyCoreApi.Incref(descr.Descr);
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_FromArray(src.Array, descr.Descr, flags));
        }


        internal static ndarray FromScalar(object src, dtype descr) {
            int itemsize = descr.ElementSize;
            NpyDefs.NPY_TYPES type = descr.TypeNum;

            if (itemsize == 0 && NpyDefs.IsExtended(type)) {
                if (src is string) itemsize = ((string)src).Length;
                else if (src is Array) itemsize = ((Array)src).Length;
                else itemsize = 1;

                throw new NotImplementedException("Need to figure out storage/handling of strings.");
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
                // TODO: Handle scalar conversion
                throw new NotImplementedException("Scalar-to-array conversion not implemented");
            } else {
                if (maxDepth > 0 && type == NpyDefs.NPY_TYPES.NPY_OBJECT &&
                    numDim > maxDepth) {
                    numDim = maxDepth;
                }   
                if (maxDepth > 0 && numDim > maxDepth ||
                    minDepth > 0 && numDim < minDepth) {
                    throw new IronPython.Runtime.Exceptions.RuntimeException("Invalid number of dimensions.");
                }

                long[] dims = new long[numDim];
                if (DiscoverDimensions(src, numDim, dims, 0, checkIt)) {
                    if (descr.Type == NpyDefs.NPY_TYPECHAR.NPY_CHARLTR &&
                        numDim > 0 && dims[numDim - 1] == 1) {
                        // TODO: Check this. Is this because it stores a string
                        // pointer in each array entry?
                        numDim--;
                    }

                    result = NpyCoreApi.AllocArray(descr, numDim, dims, fortran);
                    AssignToArray(src, result);
                }
            }
            return result;
        }
        
        internal static ndarray PrependOnes(ndarray arr, int nd, int ndmin) {
            // TODO: Unimplemented
            return arr;
        }

        internal static dtype FindArrayType(Object src, dtype minitype, int max) {
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
            throw new NotImplementedException("UseDefaultType (see common.c: _use_default_type) not implemented.");
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

            if (max < 1) return -1;

            if (src is IEnumerable<Object>) {
                IEnumerable<Object> seq = (IEnumerable<Object>)src;

                if (stopAtTuple && seq is IronPython.Runtime.PythonTuple)
                    d = 1;
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
        /// <returns>dims array is filled in; true on success, false on error</returns>
        private static bool DiscoverDimensions(Object src, int numDim,
            Int64[] dims, int dimIdx, bool checkIt) {
            bool error = false;

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
                        if (!DiscoverDimensions(o, numDim - 1, dims, dimIdx + 1, checkIt)) {
                            error = true;
                            break;
                        }
                        if (checkIt && nLowest != 0 && nLowest != dims[dimIdx + 1]) {
                            throw new IronPython.Runtime.Exceptions.RuntimeException("Inconsistent shape in sequence");
                        }
                        if (dims[dimIdx + 1] > nLowest) nLowest = dims[dimIdx + 1];
                    }
                    dims[dimIdx + 1] = nLowest;
                }
            } else {
                // Scalar condition.
                dims[dimIdx] = 1;
            }
            return !error;
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
