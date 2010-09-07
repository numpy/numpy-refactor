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

namespace NumpyDotNet
{
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
        #region ConstantDefs
        public enum NPY_TYPES {
            NPY_BOOL = 0,
            NPY_BYTE, NPY_UBYTE,
            NPY_SHORT, NPY_USHORT,
            NPY_INT, NPY_UINT,
            NPY_LONG, NPY_ULONG,
            NPY_LONGLONG, NPY_ULONGLONG,
            NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
            NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
            NPY_DATETIME, NPY_TIMEDELTA,
            NPY_OBJECT = 19,
            NPY_STRING, NPY_UNICODE,
            NPY_VOID,
            NPY_NTYPES,
            NPY_NOTYPE,
            NPY_CHAR,      /* special flag */
            NPY_USERDEF = 256  /* leave room for characters */
        };
        internal const NPY_TYPES DefaultType = NPY_TYPES.NPY_DOUBLE;

        public enum NPY_TYPECHAR : byte {
            NPY_BOOLLTR = (byte)'?',
            NPY_BYTELTR = (byte)'b',
            NPY_UBYTELTR = (byte)'B',
            NPY_SHORTLTR = (byte)'h',
            NPY_USHORTLTR = (byte)'H',
            NPY_INTLTR = (byte)'i',
            NPY_UINTLTR = (byte)'I',
            NPY_LONGLTR = (byte)'l',
            NPY_ULONGLTR = (byte)'L',
            NPY_LONGLONGLTR = (byte)'q',
            NPY_ULONGLONGLTR = (byte)'Q',
            NPY_FLOATLTR = (byte)'f',
            NPY_DOUBLELTR = (byte)'d',
            NPY_LONGDOUBLELTR = (byte)'g',
            NPY_CFLOATLTR = (byte)'F',
            NPY_CDOUBLELTR = (byte)'D',
            NPY_CLONGDOUBLELTR = (byte)'G',
            NPY_OBJECTLTR = (byte)'O',
            NPY_STRINGLTR = (byte)'S',
            NPY_STRINGLTR2 = (byte)'a',
            NPY_UNICODELTR = (byte)'U',
            NPY_VOIDLTR = (byte)'V',
            NPY_DATETIMELTR = (byte)'M',
            NPY_TIMEDELTALTR = (byte)'m',
            NPY_CHARLTR = (byte)'c',

            /*
             * No Descriptor, just a define -- this let's
             * Python users specify an array of integers
             * large enough to hold a pointer on the
             * platform
             */
            NPY_INTPLTR = (byte)'p',
            NPY_UINTPLTR = (byte)'P',

            NPY_GENBOOLLTR = (byte)'b',
            NPY_SIGNEDLTR = (byte)'i',
            NPY_UNSIGNEDLTR = (byte)'u',
            NPY_FLOATINGLTR = (byte)'f',
            NPY_COMPLEXLTR = (byte)'c'
        }

        
        public enum NPY_ORDER {
            NPY_ANYORDER = -1,
            NPY_CORDER = 0,
            NPY_FORTRANORDER = 1
        };

        internal const int NPY_MAXDIMS = 32;
        internal const int NPY_MAXARGS = 32;

        /* The item must be reference counted when it is inserted or extracted. */
        internal const int NPY_ITEM_REFCOUNT = 0x01;
        /* Same as needing REFCOUNT */
        internal const int NPY__ITEM_HASOBJECT= 0x01;
        /* Convert to list for pickling */
        internal const int NPY_LIST_PICKLE    = 0x02;
        /* The item is a POINTER  */
        internal const int NPY_ITEM_IS_POINTER = 0x04;
        /* memory needs to be initialized for this data-type */
        internal const int NPY_NEEDS_INIT     = 0x08;
        /* operations need Python C-API so don't give-up thread. */
        internal const int NPY_NEEDS_PYAPI    = 0x10;
        /* Use f.getitem when extracting elements of this data-type */
        internal const int NPY_USE_GETITEM    = 0x20;
        /* Use f.setitem when setting creating 0-d array from this data-type.*/
        internal const int NPY_USE_SETITEM    = 0x40;


        /* Data-type needs extra initialization on creation */
        internal const int NPY_EXTRA_DTYPE_INIT = 0x80;

        /* When creating an array of this type -- call extra function */
        internal const int NPY_UFUNC_OUTPUT_CREATION = 0x100;

        /*
         *These are inherited for global data-type if any data-types in the
         * field have them
         */
        internal const int NPY_FROM_FIELDS = (NPY_NEEDS_INIT | NPY_LIST_PICKLE |
                                  NPY_ITEM_REFCOUNT | NPY_NEEDS_PYAPI);

        internal const int NPY_OBJECT_DTYPE_FLAGS = 
            (NPY_LIST_PICKLE | NPY_USE_GETITEM |
             NPY_ITEM_IS_POINTER | NPY_ITEM_REFCOUNT | 
             NPY_NEEDS_INIT | NPY_NEEDS_PYAPI);


        /*
         * Means c-style contiguous (last index varies the fastest). The data
         * elements right after each other.
         */
        internal const int NPY_CONTIGUOUS   = 0x0001;

        /*
         * set if array is a contiguous Fortran array: the first index varies
         * the fastest in memory (strides array is reverse of C-contiguous
         * array)
         */
        internal const int NPY_FORTRAN      = 0x0002;

        internal const int NPY_C_CONTIGUOUS = NPY_CONTIGUOUS;
        internal const int NPY_F_CONTIGUOUS = NPY_FORTRAN;

        /*
         * Note: all 0-d arrays are CONTIGUOUS and FORTRAN contiguous. If a
         * 1-d array is CONTIGUOUS it is also FORTRAN contiguous
         */

        /*
         * If set, the array owns the data: it will be free'd when the array
         * is deleted.
         */
        internal const int NPY_OWNDATA      = 0x0004;

        /*
         * An array never has the next four set; they're only used as parameter
         * flags to the the various FromAny functions
         */

        /* Cause a cast to occur regardless of whether or not it is safe. */
        internal const int NPY_FORCECAST    = 0x0010;

        /*
         * Always copy the array. Returned arrays are always CONTIGUOUS,
         * ALIGNED, and WRITEABLE.
         */
        internal const int NPY_ENSURECOPY   = 0x0020;

        /* Make sure the returned array is a base-class ndarray */
        internal const int NPY_ENSUREARRAY  = 0x0040;

        /*
         * Make sure that the strides are in units of the element size Needed
         * for some operations with record-arrays.
         */
        internal const int NPY_ELEMENTSTRIDES = 0x0080;

        /*
         * Array data is aligned on the appropiate memory address for the type
         * stored according to how the compiler would align things (e.g., an
         * array of integers (4 bytes each) starts on a memory address that's
         * a multiple of 4)
         */
        internal const int NPY_ALIGNED      = 0x0100;

        /* Array data has the native endianness */
        internal const int NPY_NOTSWAPPED   = 0x0200;

        /* Array data is writeable */
        internal const int NPY_WRITEABLE    = 0x0400;

        /*
         * If this flag is set, then base contains a pointer to an array of
         * the same size that should be updated with the current contents of
         * this array when this array is deallocated
         */
        internal const int NPY_UPDATEIFCOPY  = 0x1000;

        /* This flag is for the array interface */
        internal const int NPY_ARR_HAS_DESCR  = 0x0800;


        internal const int NPY_BEHAVED = (NPY_ALIGNED | NPY_WRITEABLE);
        internal const int NPY_BEHAVED_NS = (NPY_ALIGNED | NPY_WRITEABLE | NPY_NOTSWAPPED);
        internal const int NPY_CARRAY = (NPY_CONTIGUOUS | NPY_BEHAVED);
        internal const int NPY_CARRAY_RO = (NPY_CONTIGUOUS | NPY_ALIGNED);
        internal const int NPY_FARRAY = (NPY_FORTRAN | NPY_BEHAVED);
        internal const int NPY_FARRAY_RO = (NPY_FORTRAN | NPY_ALIGNED);
        internal const int NPY_DEFAULT = NPY_CARRAY;
        internal const int NPY_IN_ARRAY = NPY_CARRAY_RO;
        internal const int NPY_OUT_ARRAY = NPY_CARRAY;
        internal const int NPY_INOUT_ARRAY = (NPY_CARRAY | NPY_UPDATEIFCOPY);
        internal const int NPY_IN_FARRAY = NPY_FARRAY_RO;
        internal const int NPY_OUT_FARRAY = NPY_FARRAY;
        internal const int NPY_INOUT_FARRAY = (NPY_FARRAY | NPY_UPDATEIFCOPY);

        internal const int NPY_UPDATE_ALL = (NPY_CONTIGUOUS | NPY_FORTRAN | NPY_ALIGNED);
        #endregion


        #region API Wrappers

        /// <summary>
        /// Returns a new descriptor object for internal types or user defined
        /// types.
        /// </summary>
        internal static dtype DescrFromType(NPY_TYPES type) {
            IntPtr descr = NpyArray_DescrFromType((int)type);
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
        internal static ndarray NewCopy(ndarray arr, NPY_ORDER order) {
            // TODO: NewCopy is not implemented.
            return arr;
        }


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

            try {
/*                nativeDims = Marshal.AllocHGlobal(sizeof(long) * numdim);
                Marshal.StructureToPtr(dimensions, nativeDims, true); */
                Incref(descr.Descr);
                return DecrefToInterface<ndarray>(
                    NpyArrayAccess_AllocArray(descr.Descr, numdim, dimensions, fortran));
            } finally {
                //Marshal.FreeHGlobal(nativeDims);
            }
        }


        /// <summary>
        /// Returns an array with the size of each dimension in the given array.
        /// </summary>
        /// <param name="arr">The array</param>
        /// <returns>Array w/ an array size for each dimension</returns>
        internal static Int64[] GetArrayDims(ndarray arr) {
            Int64[] dims;
            
            dims = new Int64[arr.Ndim];
            unsafe {
                fixed (Int64* dimMem = dims) {
                    if (!GetArrayDims(arr.Array, arr.Ndim, dimMem)) {
                        throw new IronPython.Runtime.Exceptions.RuntimeException("Error getting array dimensions.");
                    }
                }
            }
            return dims;
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
        internal static extern IntPtr NpyArray_FromArray(IntPtr arr, IntPtr descr, 
            int flags);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void npy_initlib(IntPtr functionDefs, IntPtr wrapperFuncs,
            IntPtr error_set, IntPtr error_occured, IntPtr error_clear,
            IntPtr cmp_priority, IntPtr incref, IntPtr decref);

        #endregion

        #region NpyAccessLib functions

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint="NpyArrayAccess_ArrayGetOffsets")]
        unsafe private static extern void ArrayGetOffsets(int *magicNumOffset,
            int *descrOffset, int *ndOffset, int *flagsOffset, int *dataOffset);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_DescrGetOffsets")]
        unsafe private static extern void DescrGetOffsets(int* magicNumOffset,
            int* kindOffset, int* typeOffset, int* byteorderOffset,
            int* flagsOffset, int* typenumOffset, int* elsizeOffset, 
            int* alignmentOffset, int* namesOFfset, int* subarrayOffset);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint="NpyArrayAccess_ArraySetDescr")]
        internal static extern void ArraySetDescr(IntPtr array, IntPtr newDescr);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint="NpyArrayAccess_Incref")]
        internal static extern void Incref(IntPtr obj);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint="NpyArrayAccess_Decref")]
        internal static extern void Decref(IntPtr obj);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_GetNativeByteOrder")]
        private static extern byte GetNativeByteOrder();

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_GetArrayDims")]
        unsafe private static extern bool GetArrayDims(IntPtr arr, int numDims, Int64 *dimMem);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr NpyArrayAccess_AllocArray(IntPtr descr, int nd,
            [MarshalAs(UnmanagedType.LPArray)] long[] dims, bool fortran);

        [DllImport("NpyAccessLib", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "NpyArrayAccess_GetArrayStride")]
        internal static extern long GetArrayStride(IntPtr arr, int dims);


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


        internal static readonly NpyArrayOffsets ArrayOffsets;
        internal static readonly NpyArrayDescrOffsets DescrOffsets;

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

                ndarray wrapArray = new ndarray(coreArray);
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
        /// <returns>New handle to the input object</returns>
        private static IntPtr IncrefCallback(IntPtr ptr, IntPtr wrapPtr) {
            object obj = GCHandle.FromIntPtr(ptr).Target;
            IntPtr retval = GCHandle.ToIntPtr(GCHandle.Alloc(obj));
            if (wrapPtr != IntPtr.Zero) {
                lock (interfaceSyncRoot) {
                    GCHandle old = GCHandle.FromIntPtr(wrapPtr);
                    wrapPtr = retval;
                    old.Free();
                }
            }
            return retval;
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr del_Incref(IntPtr ptr, IntPtr wrapPtr);

        /// <summary>
        /// Releases the reference to the given interface object.  Note that
        /// this is not a decref but actual freeingo of this handle, it can
        /// not be used again.
        /// </summary>
        /// <param name="ptr">Interface object to 'decref'</param>
        private static void DecrefCallback(IntPtr ptr, IntPtr wrapPtr) {
            if (wrapPtr != IntPtr.Zero) {
                // Deferencing the interface wrapper.  We can't just null the
                // wrapPtr because we have to have maintain the link so we
                // allocate a weak reference instead.
                GCHandle handle = GCHandle.FromIntPtr(ptr);
                Object target = handle.Target;
                lock (interfaceSyncRoot) {
                    if (ptr == Marshal.ReadIntPtr(wrapPtr)) {
                        Marshal.WriteIntPtr(wrapPtr,
                            GCHandle.ToIntPtr(GCHandle.Alloc(target, GCHandleType.Weak)));
                    } else {
                        Console.WriteLine("Unexpected decref where wrapPtr != ptr.");
                    }
                    handle.Free();
                }
            } else {
                GCHandle.FromIntPtr(ptr).Free();
            }
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void del_Decref(IntPtr ptr, IntPtr wrapPtr);

        //
        // These variables hold a reference to the delegates passed into the core.
        // Failure to hold these references causes the callback function to disappear
        // at some point when the GC runs.
        //
        private static readonly NpyInterface_WrapperFuncs wrapFuncs;

        private static readonly del_ArrayNewWrapper ArrayNewWrapDelegate =
            new del_ArrayNewWrapper(ArrayNewWrapper);
        private static readonly del_DescrNewFromType DescrNewFromTypeDelegate =
            new del_DescrNewFromType(DescrNewFromType);
        private static readonly del_DescrNewFromWrapper DescrNewFromWrapperDelegate =
            new del_DescrNewFromWrapper(DescrNewFromWrapper);
        private static readonly del_Incref IncrefCallbackDelegate =
            new del_Incref(IncrefCallback);
        private static readonly del_Decref DecrefCallbackDelegate =
            new del_Decref(DecrefCallback);


        /// <summary>
        /// Initializes the core library with necessary callbacks on load.
        /// </summary>
        static NpyCoreApi() {
            System.Console.WriteLine("Hello world");

            wrapFuncs = new NpyInterface_WrapperFuncs();

            wrapFuncs.array_new_wrapper = 
                Marshal.GetFunctionPointerForDelegate(ArrayNewWrapDelegate);
            wrapFuncs.iter_new_wrapper = IntPtr.Zero;
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
                    IntPtr.Zero,
                    IntPtr.Zero,
                    IntPtr.Zero,
                    IntPtr.Zero,
                    Marshal.GetFunctionPointerForDelegate(IncrefCallbackDelegate),
                    Marshal.GetFunctionPointerForDelegate(DecrefCallbackDelegate));
            } finally {
                Marshal.FreeHGlobal(wrapHandle);
            }

            // Initialize the offsets to each structure type for fast access
            // TODO: Not sure if this is a great way to do this, but for now it's
            // a convenient way to get hard field offsets from the core.
            unsafe {
                fixed (int* magicOffset = &ArrayOffsets.off_magic_number,
                            descrOffset = &ArrayOffsets.off_descr,
                            flagsOffset = &ArrayOffsets.off_flags,
                            ndOffset = &ArrayOffsets.off_nd,
                            dataOffset = &ArrayOffsets.off_data) {
                    ArrayGetOffsets(magicOffset, descrOffset, 
                                    ndOffset, flagsOffset, dataOffset);
                }

                fixed (int* magicOffset = &DescrOffsets.off_magic_number,
                            kindOffset = &DescrOffsets.off_kind,
                            typeOffset = &DescrOffsets.off_type,
                            byteorderOffset = &DescrOffsets.off_byteorder,
                            flagsOffset = &DescrOffsets.off_flags,
                            typenumOffset = &DescrOffsets.off_type_num,
                            elsizeOffset = &DescrOffsets.off_elsize,
                            alignmentOffset = &DescrOffsets.off_alignment,
                            namesOffset = &DescrOffsets.off_names,
                            subarrayOffset = &DescrOffsets.off_subarray) {
                    DescrGetOffsets(magicOffset, kindOffset, typeOffset,
                        byteorderOffset, flagsOffset, typenumOffset, elsizeOffset,
                        alignmentOffset, namesOffset, subarrayOffset);
                }
            }

            nativeByteOrder = GetNativeByteOrder();
        }

        #endregion
    }
}
