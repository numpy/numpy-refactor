using System;
using System.Collections.Generic;
using System.Linq;
using System.Security;
using System.Text;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using IronPython.Runtime;
using IronPython.Modules;
using Microsoft.Scripting.Runtime;
using Microsoft.Scripting.Utils;

namespace NumpyDotNet {
    /// <summary>
    /// NpyCoreApi class wraps the interactions with the libndarray core library. It
    /// also makes use of NpyAccessLib.dll for a few functions that must be
    /// implemented in native code.
    /// 
    /// TODO: This class is going to get very large.  Not sure if it's better to
    /// try to break it up or just use partial classes and split it across 
    /// multiple files.
    /// </summary>
    [SuppressUnmanagedCodeSecurity]
    public static class NpyCoreApi {

        #region API Wrappers

        /// <summary>
        /// Returns a new descriptor object for internal types or user defined
        /// types.
        /// </summary>
        internal static dtype DescrFromType(NpyDefs.NPY_TYPES type) {
            IntPtr descr = NpyArray_DescrFromType((int)type);
            CheckError();
            return DecrefToInterface<dtype>(descr);
        }

        internal static byte NativeByteOrder {
            get { return nativeByteOrder; }
        }

        internal static dtype SmallType(dtype t1, dtype t2) {
            return ToInterface<dtype>(
                NpyArray_SmallType(t1.Descr, t2.Descr));
        }

        /// <summary>
        /// Returns a copy of the passed array in the specified order (C, Fortran)
        /// </summary>
        /// <param name="arr">Array to copy</param>
        /// <param name="order">Desired order</param>
        /// <returns>New array</returns>
        internal static ndarray NewCopy(ndarray arr, NpyDefs.NPY_ORDER order) {
            // TODO: NewCopy is not implemented.
            return arr;
        }


        /// <summary>
        /// Moves the contents of src into dest.  Arrays are assumed to have the
        /// same number of elements, but can be different sizes and different types.
        /// </summary>
        /// <param name="dest">Destination array</param>
        /// <param name="src">Source array</param>
        internal static void MoveInto(ndarray dest, ndarray src) {
            if (NpyArray_MoveInto(dest.Array, src.Array) == -1) {
                CheckError();
            }
        }


        private static object AllocArraySyncRoot = new Object();

        /// <summary>
        /// Allocates a new array and returns the ndarray wrapper
        /// </summary>
        /// <param name="descr">Type descriptor</param>
        /// <param name="numdim">Num of dimensions</param>
        /// <param name="dimensions">Size of each dimension</param>
        /// <param name="fortran">True if Fortran layout, false for C layout</param>
        /// <returns>Newly allocated array</returns>
        internal static ndarray AllocArray(dtype descr, int numdim, long[] dimensions,
            bool fortran) {
            IntPtr nativeDims = IntPtr.Zero;

            lock (AllocArraySyncRoot) {
                Incref(descr.Descr);
                return DecrefToInterface<ndarray>(
                    NpyArrayAccess_AllocArray(descr.Descr, numdim, dimensions, fortran));
            }
        }


        /// <summary>
        /// Returns an array with the size or stride of each dimension in the given array.
        /// </summary>
        /// <param name="arr">The array</param>
        /// <param name="getDims">True returns size of each dimension, false returns stride of each dimension</param>
        /// <returns>Array w/ an array size or stride for each dimension</returns>
        internal static Int64[] GetArrayDimsOrStrides(ndarray arr, bool getDims) {
            Int64[] retArr;

            retArr = new Int64[arr.ndim];
            unsafe {
                fixed (Int64* dimMem = retArr) {
                    if (!GetArrayDimsOrStrides(arr.Array, arr.ndim, getDims, dimMem)) {
                        throw new IronPython.Runtime.Exceptions.RuntimeException("Error getting array dimensions.");
                    }
                }
            }
            return retArr;
        }

        internal static ndarray NewFromDescr(dtype descr, long[] dims, long[] strides,
            int flags, object interfaceData) {
            GCHandle h = GCHandle.Alloc(interfaceData);
            try {
                Incref(descr.Descr);
                return DecrefToInterface<ndarray>(NewFromDescrThunk(descr.Descr, dims.Length,
                    flags, dims, strides, IntPtr.Zero, GCHandle.ToIntPtr(h)));
            } finally {
                h.Free();
            }
        }


        internal static flatiter IterNew(ndarray ao) {
            return DecrefToInterface<flatiter>(
                NpyArray_IterNew(ao.Array));
        }

        internal static ndarray IterSubscript(flatiter iter, NpyIndexes indexes) {
            return DecrefToInterface<ndarray>(
                NpyArray_IterSubscript(iter.Iter, indexes.Indexes, indexes.NumIndexes));
        }

        internal static void IterSubscriptAssign(flatiter iter, NpyIndexes indexes, ndarray val) {
            if (NpyArray_IterSubscriptAssign(iter.Iter, indexes.Indexes, indexes.NumIndexes, val.Array) < 0) {
                CheckError();
            }
        }

        internal static ndarray Flatten(ndarray a, NpyDefs.NPY_ORDER order) {
            return DecrefToInterface<ndarray>(
                NpyArray_Flatten(a.Array, order)
                );
        }

        #endregion


        #region C API Definitions

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr NpyArray_DescrNew(IntPtr descr);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr NpyArray_DescrFromType(Int32 type);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr NpyArray_SmallType(IntPtr descr1, IntPtr descr2);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern byte NpyArray_EquivTypes(IntPtr t1, IntPtr typ2);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NpyArray_ElementStrides(IntPtr arr);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr NpyArray_NewCopy(IntPtr arr, byte order);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NpyArray_MoveInto(IntPtr dest, IntPtr src);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr NpyArray_FromArray(IntPtr arr, IntPtr descr,
            int flags);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void NpyArray_dealloc(IntPtr arr);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void NpyArray_DescrDestroy(IntPtr arr);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void npy_initlib(IntPtr functionDefs, IntPtr wrapperFuncs,
            IntPtr error_set, IntPtr error_occured, IntPtr error_clear,
            IntPtr cmp_priority, IntPtr incref, IntPtr decref);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr NpyArray_Subscript(IntPtr arr, IntPtr indexes, int n);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NpyArray_SubscriptAssign(IntPtr self, IntPtr indexes, int n, IntPtr value);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void NpyArray_IndexDealloc(IntPtr indexes, int n);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr NpyArray_Size(IntPtr arr);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr NpyArray_ArrayItem(IntPtr array, IntPtr index);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr NpyArray_IndexSimple(IntPtr arr, IntPtr indexes, int n);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NpyArray_IndexFancyAssign(IntPtr dest, IntPtr indexes, int n, IntPtr value_array);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NpyArray_SetField(IntPtr arr, IntPtr descr, int offset, IntPtr val);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr NpyArray_IterNew(IntPtr ao);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr NpyArray_IterSubscript(IntPtr iter, IntPtr indexes, int n);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NpyArray_IterSubscriptAssign(IntPtr iter, IntPtr indexes, int n, IntPtr array_val);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr NpyArray_Flatten(IntPtr arr, NpyDefs.NPY_ORDER order);

        #endregion

        #region NpyAccessLib functions

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_ArraySetDescr")]
        internal static extern void ArraySetDescr(IntPtr array, IntPtr newDescr);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_Incref")]
        internal static extern void Incref(IntPtr obj);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_Decref")]
        internal static extern void Decref(IntPtr obj);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_GetNativeTypeInfo")]
        private static extern byte GetNativeTypeInfo(out int intSize,
            out int longsize, out int longLongSize);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_GetArrayDimsOrStrides")]
        unsafe private static extern bool GetArrayDimsOrStrides(IntPtr arr, int numDims, bool getDims, Int64* dimMem);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr NpyArrayAccess_AllocArray(IntPtr descr, int nd,
            [MarshalAs(UnmanagedType.LPArray)] long[] dims, bool fortran);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_GetArrayStride")]
        internal static extern long GetArrayStride(IntPtr arr, int dims);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_BindIndex")]
        internal static extern int BindIndex(IntPtr arr, IntPtr indexes, int n, IntPtr bound_indexes);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_GetFieldOffset")]
        internal static extern int GetFieldOffset(IntPtr descr, [MarshalAs(UnmanagedType.LPStr)] string fieldName, out IntPtr out_descr);

        /// <summary>
        /// Deallocates an NpyObject.
        /// </summary>
        /// <param name="obj">The object to deallocate</param>
        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_Dealloc")]
        internal static extern void Dealloc(IntPtr obj);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_IterNext")]
        internal static extern IntPtr IterNext(IntPtr iter);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_IterReset")]
        internal static extern void IterReset(IntPtr iter);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_IterGoto1D")]
        internal static extern IntPtr IterGoto1D(IntPtr iter, IntPtr index);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_IterArray")]
        internal static extern IntPtr IterArray(IntPtr iter);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_IterCoords")]
        internal static extern IntPtr IterCoords(IntPtr iter);

        //
        // Offset functions - these return the offsets to fields in native structures
        // as a workaround for not being able to include the C header file.
        //

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_ArrayGetOffsets")]
        private static extern void ArrayGetOffsets(out int magicNumOffset,
            out int descrOffset, out int ndOffset, out int flagsOffset, out int dataOffset);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_DescrGetOffsets")]
        private static extern void DescrGetOffsets(out int magicNumOffset,
            out int kindOffset, out int typeOffset, out int byteorderOffset,
            out int flagsOffset, out int typenumOffset, out int elsizeOffset,
            out int alignmentOffset, out int namesOFfset, out int subarrayOffset);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_IterGetOffsets")]
        private static extern void IterGetOffsets(out int sizeOffset, out int indexOffset);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_GetIndexInfo")]
        internal static extern void GetIndexInfo(out int unionOffset, out int indexSize, out int maxDims);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_NewFromDescrThunk")]
        internal static extern IntPtr NewFromDescrThunk(IntPtr descr, int nd,
            int flags, long[] dims, long[] strides, IntPtr data, IntPtr interfaceData);

        #endregion


        #region Callbacks and native access

        /* This structure must match the NpyObject_HEAD structure in npy_object.h
         * exactly as it is used to determine the platform-specific offsets. The
         * offsets allow the C# code to access these fields directly. */
        [StructLayout(LayoutKind.Sequential)]
        struct NpyObject_HEAD {
            internal IntPtr nob_refcnt;
            internal IntPtr nob_type;
            internal IntPtr nob_interface;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct NpyInterface_WrapperFuncs {
            internal IntPtr array_new_wrapper;
            internal IntPtr iter_new_wrapper;
            internal IntPtr multi_iter_new_wrapper;
            internal IntPtr neighbor_iter_new_wrapper;
            internal IntPtr descr_new_from_type;
            internal IntPtr descr_new_from_wrapper;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct NpyArrayOffsets {
            internal int off_magic_number;
            internal int off_descr;
            internal int off_nd;
            internal int off_flags;
            internal int off_data;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct NpyArrayDescrOffsets {
            internal int off_magic_number;
            internal int off_kind;
            internal int off_type;
            internal int off_byteorder;
            internal int off_flags;
            internal int off_type_num;
            internal int off_elsize;
            internal int off_alignment;
            internal int off_names;
            internal int off_subarray;
        }

        internal struct NpyArrayIterOffsets {
            internal int off_size;
            internal int off_index;
        }

        internal struct NpyArrayIndexInfo {
            internal int off_union;
            internal int sizeof_index;
            internal int max_dims;
        }


        internal static readonly NpyArrayOffsets ArrayOffsets;
        internal static readonly NpyArrayDescrOffsets DescrOffsets;
        internal static readonly NpyArrayIterOffsets IterOffsets;
        internal static readonly NpyArrayIndexInfo IndexInfo;

        internal static byte nativeByteOrder;

        /// <summary>
        /// Used for synchronizing modifications to interface pointer.
        /// </summary>
        private static object interfaceSyncRoot = new Object();

        /// <summary>
        /// Offset to the interface pointer.
        /// </summary>
        private static int Offset_InterfacePtr = (int)Marshal.OffsetOf(typeof(NpyObject_HEAD), "nob_interface");

        /// <summary>
        /// Given a pointer to a core (native) object, returns the managed wrapper.
        /// </summary>
        /// <param name="ptr">Address of native object</param>
        /// <returns>Managed wrapper object</returns>
        internal static TResult ToInterface<TResult>(IntPtr ptr) {
            if (ptr == IntPtr.Zero) {
                return default(TResult);
            }
            IntPtr wrapper = Marshal.ReadIntPtr(ptr, (int)Offset_InterfacePtr);
            if (wrapper == IntPtr.Zero) {
                throw new IronPython.Runtime.Exceptions.RuntimeException(
                    String.Format("Managed wrapper for type '{0}' is NULL.", typeof(TResult).Name));
            }
            return (TResult)GCHandle.FromIntPtr(wrapper).Target;
        }

        /// <summary>
        /// Same as ToInterface but releases the core reference. 
        /// </summary>
        /// <typeparam name="TResult">Type of the expected object</typeparam>
        /// <param name="ptr">Pointer to the core object</param>
        /// <returns>Wrapper instance corresponding to ptr</returns>
        internal static TResult DecrefToInterface<TResult>(IntPtr ptr) {
            CheckError();
            if (ptr == IntPtr.Zero) {
                return default(TResult);
            }
            TResult result = ToInterface<TResult>(ptr);
            Decref(ptr);
            return result;
        }


        /// <summary>
        /// Allocates a managed wrapper for the passed array object.
        /// </summary>
        /// <param name="coreArray">Pointer to the native array object</param>
        /// <param name="ensureArray">If true forces base array type, not subtype</param>
        /// <param name="customStrides">Not sure how this is used</param>
        /// <param name="interfaceData">Not used</param>
        /// <param name="interfaceRet">void ** for us to store the allocated wrapper</param>
        /// <returns>True on success, false on failure</returns>
        private static int ArrayNewWrapper(IntPtr coreArray, int ensureArray,
            int customStrides, IntPtr subtypePtr, IntPtr interfaceData,
            IntPtr interfaceRet) {
            int success = 1;     // Success

            try {
                // TODO: subtyping is not figured out or implemented yet.

                ndarray wrapArray;
                if (interfaceData != IntPtr.Zero) {
                    wrapArray = (ndarray)GCHandle.FromIntPtr(interfaceData).Target;
                    wrapArray.SetArray(coreArray);
                } else {
                    wrapArray = new ndarray(coreArray);
                }

                IntPtr ret = GCHandle.ToIntPtr(GCHandle.Alloc(wrapArray));
                Marshal.WriteIntPtr(interfaceRet, ret);

                // TODO: Skipping subtype-specific initialization (ctors.c:718)
            } catch (InsufficientMemoryException) {
                Console.WriteLine("Insufficient memory while allocating array wrapper.");
                success = 0;
            } catch (Exception) {
                Console.WriteLine("Exception while allocating array wrapper.");
                success = 0;
            }
            return success;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int del_ArrayNewWrapper(IntPtr coreArray, int ensureArray,
            int customStrides, IntPtr subtypePtr, IntPtr interfaceData,
            IntPtr interfaceRet);

        private static int IterNewWrapper(IntPtr coreIter, IntPtr interfaceRet) {
            int success = 1;

            try {
                flatiter wrapIter = new flatiter(coreIter);
                IntPtr ret = GCHandle.ToIntPtr(GCHandle.Alloc(wrapIter));
                Marshal.WriteIntPtr(interfaceRet, ret);
            } catch (InsufficientMemoryException) {
                Console.WriteLine("Insufficient memory while allocating iterator wrapper.");
                success = 0;
            } catch (Exception) {
                Console.WriteLine("Exception while allocating iterator wrapper.");
                success = 0;
            }
            return success;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int del_IterNewWrapper(IntPtr coreIter, IntPtr interfaceRet);


        /// <summary>
        /// Allocated a managed wrapper for one of the core, native types
        /// </summary>
        /// <param name="type">Type code (not used)</param>
        /// <param name="descr">Pointer to the native descriptor object</param>
        /// <param name="interfaceRet">void** for returning allocated wrapper</param>
        /// <returns>1 on success, 0 on error</returns>
        private static int DescrNewFromType(int type, IntPtr descr, IntPtr interfaceRet) {
            int success = 1;

            try {
                // TODO: Descriptor typeobj not handled. Do we need to?

                dtype wrap = new dtype(descr, type);
                Marshal.WriteIntPtr(interfaceRet,
                    GCHandle.ToIntPtr(GCHandle.Alloc(wrap)));
            } catch (InsufficientMemoryException) {
                Console.WriteLine("Insufficient memory while allocating descriptor wrapper.");
                success = 0;
            } catch (Exception) {
                Console.WriteLine("Exception while allocating descriptor wrapper.");
                success = 0;
            }
            return success;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int del_DescrNewFromType(int type, IntPtr descr, IntPtr interfaceRet);




        /// <summary>
        /// Allocated a managed wrapper for a user defined type
        /// </summary>
        /// <param name="baseTmp">Pointer to the base descriptor (not used)</param>
        /// <param name="descr">Pointer to the native descriptor object</param>
        /// <param name="interfaceRet">void** for returning allocated wrapper</param>
        /// <returns>1 on success, 0 on error</returns>
        private static int DescrNewFromWrapper(IntPtr baseTmp, IntPtr descr, IntPtr interfaceRet) {
            int success = 1;

            try {
                // TODO: Descriptor typeobj not handled. Do we need to?

                dtype wrap = new dtype(descr);
                Marshal.WriteIntPtr(interfaceRet,
                    GCHandle.ToIntPtr(GCHandle.Alloc(wrap)));
            } catch (InsufficientMemoryException) {
                Console.WriteLine("Insufficient memory while allocating descriptor wrapper.");
                success = 0;
            } catch (Exception) {
                Console.WriteLine("Exception while allocating descriptor wrapper.");
                success = 0;
            }
            return success;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int del_DescrNewFromWrapper(IntPtr baseTmp, IntPtr descr, IntPtr interfaceRet);



        /// <summary>
        /// Accepts a pointer to an existing GCHandle object and allocates
        /// an additional GCHandle to the same object.  This effectively
        /// does an "incref" on the object.  Used in cases where an array
        /// of objects is being copied.
        /// 
        /// Usually wrapPtr is NULL meaning that we just allocate a new
        /// handle and return it.  If wrapPtr != NULL then we assign the
        /// new handle to it as well.  Must be done atomically.
        /// </summary>
        /// <param name="ptr">Pointer to GCHandle of object to reference</param>
        /// <param name="nobInterfacePtr">Address of the nob_interface field (not value of it)</param>
        /// <returns>New handle to the input object</returns>
        private static IntPtr IncrefCallback(IntPtr ptr, IntPtr nobInterfacePtr) {
            GCHandle oldWrapRef = GCHandle.FromIntPtr(ptr);
            object wrapperObj = oldWrapRef.Target;
            IntPtr newWrapRef = GCHandle.ToIntPtr(GCHandle.Alloc(wrapperObj));
            if (nobInterfacePtr != IntPtr.Zero) {
                lock (interfaceSyncRoot) {
                    // Replace the contents of nobInterfacePtr with the new reference.
                    Marshal.WriteIntPtr(nobInterfacePtr, newWrapRef);
                    oldWrapRef.Free();
                }
            }
            return newWrapRef;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr del_Incref(IntPtr ptr, IntPtr wrapPtr);

        /// <summary>
        /// Releases the reference to the given interface object.  Note that
        /// this is not a decref but actual freeingo of this handle, it can
        /// not be used again.
        /// </summary>
        /// <param name="ptr">Interface object to 'decref'</param>
        private static void DecrefCallback(IntPtr ptr, IntPtr nobInterfacePtr) {
            if (nobInterfacePtr != IntPtr.Zero) {
                // Deferencing the interface wrapper.  We can't just null the
                // wrapPtr because we have to have maintain the link so we
                // allocate a weak reference instead.
                lock (interfaceSyncRoot) {
                    GCHandle oldWrapRef = GCHandle.FromIntPtr(ptr);
                    Object wrapperObj = oldWrapRef.Target;
                    Marshal.WriteIntPtr(nobInterfacePtr,
                        GCHandle.ToIntPtr(GCHandle.Alloc(wrapperObj, GCHandleType.Weak)));
                    oldWrapRef.Free();
                }
            } else {
                GCHandle.FromIntPtr(ptr).Free();
            }
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void del_Decref(IntPtr ptr, IntPtr wrapPtr);

        #region Error handling

        /// <summary>
        /// Error type, determines which type of exception to throw.  
        /// DANGER! Must be kept in sync with npy_api.h
        /// </summary>
        private enum NpyExc_Type {
            MemoryError = 0,
            IOError,
            ValueError,
            TypeError,
            IndexError,
            RuntimeError,
            AttributeError,
            ComplexWarning,
            NoError
        }


        /// <summary>
        /// Indicates the most recent error code or NpyExc_NoError if nothing pending
        /// </summary>
        [ThreadStatic]
        private static NpyExc_Type ErrorCode = NpyExc_Type.NoError;

        /// <summary>
        /// Stores the most recent error message per-thread
        /// </summary>
        [ThreadStatic]
        private static string ErrorMessage = null;

        internal static void CheckError() {
            if (ErrorCode != NpyExc_Type.NoError) {
                NpyExc_Type errTmp = ErrorCode;
                String msgTmp = ErrorMessage;

                ErrorCode = NpyExc_Type.NoError;
                ErrorMessage = null;

                switch (errTmp) {
                    case NpyExc_Type.MemoryError:
                        throw new InsufficientMemoryException(msgTmp);
                    case NpyExc_Type.IOError:
                        throw new System.IO.IOException(msgTmp);
                    case NpyExc_Type.ValueError:
                        throw new ArgumentException(msgTmp);
                    case NpyExc_Type.IndexError:
                        throw new IndexOutOfRangeException(msgTmp);
                    case NpyExc_Type.RuntimeError:
                        throw new IronPython.Runtime.Exceptions.RuntimeException(msgTmp);
                    case NpyExc_Type.AttributeError:
                        throw new MissingMemberException(msgTmp);
                    case NpyExc_Type.ComplexWarning:
                        throw new IronPython.Runtime.Exceptions.RuntimeException(msgTmp);
                    default:
                        Console.WriteLine("Unhandled exception type {0} in CheckError.", errTmp);
                        throw new IronPython.Runtime.Exceptions.RuntimeException(msgTmp);
                }
            }
        }



        /// <summary>
        /// Called by NpyErr_SetMessage in the native world when something bad happens
        /// </summary>
        /// <param name="exceptType">Type of exception to be thrown</param>
        /// <param name="bStr">Message string</param>
        unsafe private static void SetErrorCallback(int exceptType, sbyte* bStr) {
            if (exceptType < 0 || exceptType >= (int)NpyExc_Type.NoError) {
                Console.WriteLine("Internal error: invalid exception type {0}, likely ErrorType and npyexc_type (npy_api.h) are out of sync.",
                    exceptType);
            }
            ErrorCode = (NpyExc_Type)exceptType;
            ErrorMessage = new string(bStr);
            Console.WriteLine("Set error {0}: {1}", exceptType, ErrorMessage);
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        unsafe public delegate void del_SetErrorCallback(int exceptType, sbyte* msg);


        /// <summary>
        /// Called by native side to check to see if an error occurred
        /// </summary>
        /// <returns>1 if an error is pending, 0 if not</returns>
        private static int ErrorOccurredCallback() {
            return (ErrorCode != NpyExc_Type.NoError) ? 1 : 0;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int del_ErrorOccurredCallback();


        private static void ClearErrorCallback() {
            ErrorCode = NpyExc_Type.NoError;
            ErrorMessage = null;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void del_ClearErrorCallback();

        #endregion

        //
        // These variables hold a reference to the delegates passed into the core.
        // Failure to hold these references causes the callback function to disappear
        // at some point when the GC runs.
        //
        private static readonly NpyInterface_WrapperFuncs wrapFuncs;

        private static readonly del_ArrayNewWrapper ArrayNewWrapDelegate =
            new del_ArrayNewWrapper(ArrayNewWrapper);
        private static readonly del_IterNewWrapper IterNewWrapperDelegate =
            new del_IterNewWrapper(IterNewWrapper);
        private static readonly del_DescrNewFromType DescrNewFromTypeDelegate =
            new del_DescrNewFromType(DescrNewFromType);
        private static readonly del_DescrNewFromWrapper DescrNewFromWrapperDelegate =
            new del_DescrNewFromWrapper(DescrNewFromWrapper);
        private static readonly del_Incref IncrefCallbackDelegate =
            new del_Incref(IncrefCallback);
        private static readonly del_Decref DecrefCallbackDelegate =
            new del_Decref(DecrefCallback);
        unsafe private static readonly del_SetErrorCallback SetErrorCallbackDelegate =
            new del_SetErrorCallback(SetErrorCallback);
        private static readonly del_ErrorOccurredCallback ErrorOccurredCallbackDelegate =
            new del_ErrorOccurredCallback(ErrorOccurredCallback);
        private static readonly del_ClearErrorCallback ClearErrorCallbackDelegate =
            new del_ClearErrorCallback(ClearErrorCallback);

        /// <summary>
        /// The native type code that matches up to a 32-bit int.
        /// </summary>
        internal static readonly NpyDefs.NPY_TYPES TypeOf_Int32;

        /// <summary>
        /// Native type code that matches up to a 64-bit int.
        /// </summary>
        internal static readonly NpyDefs.NPY_TYPES TypeOf_Int64;

        /// <summary>
        /// Native type code that matches up to a 32-bit unsigned int.
        /// </summary>
        internal static readonly NpyDefs.NPY_TYPES TypeOf_UInt32;

        /// <summary>
        /// Native type code that matches up to a 64-bit unsigned int.
        /// </summary>
        internal static readonly NpyDefs.NPY_TYPES TypeOf_UInt64;


        /// <summary>
        /// Initializes the core library with necessary callbacks on load.
        /// </summary>
        static NpyCoreApi() {
            wrapFuncs = new NpyInterface_WrapperFuncs();

            wrapFuncs.array_new_wrapper =
                Marshal.GetFunctionPointerForDelegate(ArrayNewWrapDelegate);
            wrapFuncs.iter_new_wrapper =
                Marshal.GetFunctionPointerForDelegate(IterNewWrapperDelegate);
            wrapFuncs.multi_iter_new_wrapper = IntPtr.Zero;
            wrapFuncs.neighbor_iter_new_wrapper = IntPtr.Zero;
            wrapFuncs.descr_new_from_type =
                Marshal.GetFunctionPointerForDelegate(DescrNewFromTypeDelegate);
            wrapFuncs.descr_new_from_wrapper =
                Marshal.GetFunctionPointerForDelegate(DescrNewFromWrapperDelegate);

            int s = Marshal.SizeOf(wrapFuncs.descr_new_from_type);

            IntPtr wrapHandle = IntPtr.Zero;
            try {
                wrapHandle = Marshal.AllocHGlobal(Marshal.SizeOf(wrapFuncs));
                Marshal.StructureToPtr(wrapFuncs, wrapHandle, true);


                npy_initlib(IntPtr.Zero,
                    wrapHandle,
                    Marshal.GetFunctionPointerForDelegate(SetErrorCallbackDelegate),
                    Marshal.GetFunctionPointerForDelegate(ErrorOccurredCallbackDelegate),
                    Marshal.GetFunctionPointerForDelegate(ClearErrorCallbackDelegate),
                    IntPtr.Zero,
                    Marshal.GetFunctionPointerForDelegate(IncrefCallbackDelegate),
                    Marshal.GetFunctionPointerForDelegate(DecrefCallbackDelegate));
            } finally {
                Marshal.FreeHGlobal(wrapHandle);
            }

            // Initialize the offsets to each structure type for fast access
            // TODO: Not sure if this is a great way to do this, but for now it's
            // a convenient way to get hard field offsets from the core.
            ArrayGetOffsets(out ArrayOffsets.off_magic_number,
                            out ArrayOffsets.off_descr,
                            out ArrayOffsets.off_nd,
                            out ArrayOffsets.off_flags,
                            out ArrayOffsets.off_data);

            DescrGetOffsets(out DescrOffsets.off_magic_number,
                            out DescrOffsets.off_kind,
                            out DescrOffsets.off_type,
                            out DescrOffsets.off_byteorder,
                            out DescrOffsets.off_flags,
                            out DescrOffsets.off_type_num,
                            out DescrOffsets.off_elsize,
                            out DescrOffsets.off_alignment,
                            out DescrOffsets.off_names,
                            out DescrOffsets.off_subarray);

            IterGetOffsets(out IterOffsets.off_size,
                           out IterOffsets.off_index);

            GetIndexInfo(out IndexInfo.off_union, out IndexInfo.sizeof_index, out IndexInfo.max_dims);

            // Check the native byte ordering (make sure it matches what .NET uses) and
            // figure out the mapping between types that vary in size in the core and
            // fixed-size .NET types.
            int intSize, longSize, longLongSize;
            nativeByteOrder = GetNativeTypeInfo(out intSize, out longSize, out longLongSize);

            if (intSize == 4 && longSize == 4 && longLongSize == 8) {
                TypeOf_Int32 = NpyDefs.NPY_TYPES.NPY_INT;
                TypeOf_Int64 = NpyDefs.NPY_TYPES.NPY_LONGLONG;
                TypeOf_UInt32 = NpyDefs.NPY_TYPES.NPY_UINT;
                TypeOf_UInt64 = NpyDefs.NPY_TYPES.NPY_ULONGLONG;
            } else if (intSize == 4 && longSize == 8 && longLongSize == 8) {
                TypeOf_Int32 = NpyDefs.NPY_TYPES.NPY_INT;
                TypeOf_Int64 = NpyDefs.NPY_TYPES.NPY_LONG;
                TypeOf_UInt32 = NpyDefs.NPY_TYPES.NPY_UINT;
                TypeOf_UInt64 = NpyDefs.NPY_TYPES.NPY_ULONG;
            } else {
                throw new NotImplementedException(
                    String.Format("Unimplemented combination of native type sizes: int = {0}b, long = {1}b, longlong = {2}b",
                                  intSize, longSize, longLongSize));
            }
        }

        #endregion
    }
}
