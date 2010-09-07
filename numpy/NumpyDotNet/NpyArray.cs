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
        internal static ndarray CheckFromArray(Object src, dtype descr, int minDepth,
            int maxDepth, int requires, Object context) {

            if ((requires & NpyCoreApi.NPY_NOTSWAPPED) != 0) {
                if (descr != null && src is ndarray &&
                    ((ndarray)src).Descr.IsNativeByteOrder) {
                    descr = new dtype(((ndarray)src).Descr);
                } else if (descr != null && !descr.IsNativeByteOrder) {
                    // Descr replace
                }
                if (descr != null) {
                    descr.Byteorder = NpyCoreApi.NativeByteOrder;
                }
            }

            ndarray arr = FromAny(src, descr, minDepth, maxDepth, requires, context);

            if (arr != null && (requires & NpyCoreApi.NPY_ELEMENTSTRIDES) != 0 &&
                arr.ElementStrides == 0) {
                arr = arr.NewCopy(NpyCoreApi.NPY_ORDER.NPY_ANYORDER);
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
                if ((flags & NpyCoreApi.NPY_UPDATEIFCOPY) != 0)
                    throw new IronPython.Runtime.Exceptions.RuntimeException("UPDATEIFCOPY used for non-array input");

                if (src is IEnumerable<Object>) {
                    Console.WriteLine("Enumerable type = {0}", src.GetType().ToString());
                    result = FromIEnumerable((IEnumerable<Object>)src, descr,
                        (flags & NpyCoreApi.NPY_FORTRAN) != 0, minDepth, maxDepth);
                } else {
                    Console.WriteLine("Type name = {0}", src.GetType().ToString());
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


        internal static ndarray FromIEnumerable(IEnumerable<Object> src, dtype descr, 
            bool fortran, int minDepth, int maxDepth) {
            ndarray result = null;

            if (descr == null) {
                descr = FindArrayType(src, null, NpyCoreApi.NPY_MAXDIMS);
            }

            NpyCoreApi.NPY_TYPES type = descr.TypeNum;
            bool checkIt = (descr.Type == NpyCoreApi.NPY_TYPECHAR.NPY_CHARLTR);
            bool stopAtString =
                type != NpyCoreApi.NPY_TYPES.NPY_STRING ||
                descr.Type == NpyCoreApi.NPY_TYPECHAR.NPY_STRINGLTR;
            bool stopAtTuple =
                type == NpyCoreApi.NPY_TYPES.NPY_VOID &&
                (descr.HasNames || descr.HasSubarray);

            int numDim = DiscoverDepth(src, NpyCoreApi.NPY_MAXDIMS + 1, stopAtString, stopAtTuple);
            if (numDim == 0) {
                // TODO: Handle scalar conversion
                throw new NotImplementedException("Scalar-to-array conversion not implemented");
            } else {
                if (maxDepth > 0 && type == NpyCoreApi.NPY_TYPES.NPY_OBJECT &&
                    numDim > maxDepth) {
                    numDim = maxDepth;
                }   
                if (maxDepth > 0 && numDim > maxDepth ||
                    minDepth > 0 && numDim < minDepth) {
                    throw new IronPython.Runtime.Exceptions.RuntimeException("Invalid number of dimensions.");
                }

                long[] dims = new long[numDim];
                if (DiscoverDimensions(src, numDim, dims, 0, checkIt)) {
                    if (descr.Type == NpyCoreApi.NPY_TYPECHAR.NPY_CHARLTR &&
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
            return arr;
        }

        internal static dtype FindArrayType(Object src, dtype minitype, int max) {
            dtype chktype = null;

            if (src is ndarray) {
                chktype = ((ndarray)src).Descr;
                if (minitype == null) return chktype;
            } else {
                chktype = FindScalarType(src);
                if (chktype != null && minitype == null) return chktype;
            }

            // If a minimum type wasn't give, default to bool.
            if (minitype == null) 
                minitype = NpyCoreApi.DescrFromType(NpyCoreApi.NPY_TYPES.NPY_BOOL);

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

                        if (seq.Count() == 0 && minitype.TypeNum == NpyCoreApi.NPY_TYPES.NPY_BOOL) {
                            minitype = NpyCoreApi.DescrFromType(NpyCoreApi.DefaultType);
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
            if (chktype.TypeNum == NpyCoreApi.NPY_TYPES.NPY_VOID &&
                minitype.TypeNum != NpyCoreApi.NPY_TYPES.NPY_VOID) {
                chktype = NpyCoreApi.DescrFromType(NpyCoreApi.NPY_TYPES.NPY_OBJECT);
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
            NpyCoreApi.NPY_TYPES type;

            // TODO: Complex numbers not handled.  
            // TODO: Are int32/64 -> long, longlong correct?
            if (src is Double) type = NpyCoreApi.NPY_TYPES.NPY_DOUBLE;
            else if (src is Single) type = NpyCoreApi.NPY_TYPES.NPY_FLOAT;
            else if (src is Boolean) type = NpyCoreApi.NPY_TYPES.NPY_BOOL;
            else if (src is Byte) type = NpyCoreApi.NPY_TYPES.NPY_BYTE;
            else if (src is Int16 || src is Int32) type = NpyCoreApi.NPY_TYPES.NPY_LONG;
            else if (src is Int64) type = NpyCoreApi.NPY_TYPES.NPY_LONGLONG;
            else type = NpyCoreApi.NPY_TYPES.NPY_NOTYPE;

            return (type != NpyCoreApi.NPY_TYPES.NPY_NOTYPE) ?
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
                d = ((ndarray)src).Ndim;
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
                if (arr.Ndim == 0) dims[dimIdx] = 0;
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
            if (dim >= result.Ndim) {
                throw new IronPython.Runtime.Exceptions.RuntimeException(
                    String.Format("Source dimensions ({0}) exceeded target array dimensions ({1}).",
                    dim, result.Ndim));
            }

            if (seq.Count() != result.Dims[dim]) {
                throw new IronPython.Runtime.Exceptions.RuntimeException(
                    "AssignFromSeq: sequence/array shape mismatch.");
            }

            long stride = result.Stride(dim);
            if (dim < result.Ndim - 1) {
                // Sequence elements should be additional sequences
                seq.Iteri((o, i) =>
                    AssignFromSeq((IEnumerable<Object>)o, result, dim + 1, offset + stride * i));
            } else {
                seq.Iteri((o, i) => result.Descr.f.SetItem(o, offset + i*stride, result));
            }
        }
    }
}
