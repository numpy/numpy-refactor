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
    [SuppressUnmanagedCodeSecurity]
    internal static class NpyArray {
        #region ConstantDefs
        internal enum NPY_TYPES {
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

        /** Returns a new descriptor object for internal types or user defined
         * types */
        internal static dtype DescrFromType(Int32 type) {
            IntPtr descr = NpyArray_DescrFromType(type);
            return ToInterface<dtype>(descr);
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

        #endregion


        #region C API Definitions

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr NpyArray_DescrFromType(Int32 type);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void SimpleArray_initCallbacks(IntPtr setupBinOp, IntPtr performBinOp, IntPtr tearDownOp, IntPtr releaseMemPressure);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr SimpleArray_create(UInt32 size, int dtype);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int SimpleArray_isValid(IntPtr a);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void SimpleArray_incRef(IntPtr a);

        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr SimpleArray_create(UInt32 size);

        [DllImport("ndarray", CallingConvention=CallingConvention.Cdecl)]
        internal static extern void npy_initlib(IntPtr functionDefs, IntPtr wrapperFuncs,
            IntPtr error_set, IntPtr error_occured, IntPtr error_clear,
            IntPtr cmp_priority, IntPtr incref, IntPtr decref);

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
        };

        [StructLayout(LayoutKind.Sequential)]
        struct NpyInterface_WrapperFuncs {
            internal IntPtr array_new_wrapper;
            internal IntPtr iter_new_wrapper;
            internal IntPtr multi_iter_new_wrapper;
            internal IntPtr neighbor_iter_new_wrapper;
            internal IntPtr descr_new_from_type;
            internal IntPtr descr_new_from_wrapper;
        };



        /// <summary>
        /// Offset to the interface pointer.
        /// </summary>
        private static int Offset_InterfacePtr = (int)Marshal.OffsetOf(typeof(NpyObject_HEAD), "nob_interface");

        /// <summary>
        /// Given a pointer to a core (native) object, returns the managed wrapper.
        /// </summary>
        /// <param name="ptr">Address of native object</param>
        /// <returns>Managed wrapper object</returns>
        private static TResult ToInterface<TResult>(IntPtr ptr) {
            if (ptr == IntPtr.Zero) {
                return default(TResult);
            }
            IntPtr wrapper = Marshal.ReadIntPtr(ptr, (int)Offset_InterfacePtr);
            return (TResult)GCHandle.FromIntPtr(wrapper).Target;
        }


        /// <summary>
        /// Allocates a managed wrapper for the passed array object.
        /// </summary>
        /// <param name="array">Pointer to the native array object</param>
        /// <returns>Handle to the managed wrapper</returns>
        private static IntPtr ArrayNewWrapper(IntPtr array) {
            ndarray wrap = new ndarray(array);
            return GCHandle.ToIntPtr(GCHandle.Alloc(wrap));
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr del_ArrayNewWrapper(IntPtr ptr);


        /// <summary>
        /// Accepts a pointer to an existing GCHandle object and allocates
        /// an additional GCHandle to the same object.  This effectively
        /// does an "incref" on the object.  Used in cases where an array
        /// of objects is being copied.
        /// </summary>
        /// <param name="ptr">Pointer to GCHandle of object to reference</param>
        /// <returns>New handle to the input object</returns>
        private static IntPtr Incref(IntPtr ptr) {
            object obj = GCHandle.FromIntPtr(ptr).Target;
            return GCHandle.ToIntPtr(GCHandle.Alloc(obj));
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr del_Incref(IntPtr ptr);


        /// <summary>
        /// Releases the reference to the given interface object.  Note that
        /// this is not a decref but actual freeingo of this handle, it can
        /// not be used again.
        /// </summary>
        /// <param name="ptr">Interface object to 'decref'</param>
        private static void Decref(IntPtr ptr) {
            GCHandle.FromIntPtr(ptr).Free();
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void del_Decref(IntPtr ptr);


        /// <summary>
        /// Initializes the core library with necessary callbacks on load.
        /// </summary>
        static NpyArray() {
            System.Console.WriteLine("Hello world");

            NpyInterface_WrapperFuncs wrapFuncs = new NpyInterface_WrapperFuncs();

            wrapFuncs.array_new_wrapper = 
                Marshal.GetFunctionPointerForDelegate(new del_ArrayNewWrapper(ArrayNewWrapper));
            wrapFuncs.iter_new_wrapper = IntPtr.Zero;
            wrapFuncs.multi_iter_new_wrapper = IntPtr.Zero;
            wrapFuncs.neighbor_iter_new_wrapper = IntPtr.Zero;
            wrapFuncs.descr_new_from_type = IntPtr.Zero;
            wrapFuncs.descr_new_from_wrapper = IntPtr.Zero;
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
                    Marshal.GetFunctionPointerForDelegate(new del_Incref(Incref)),
                    Marshal.GetFunctionPointerForDelegate(new del_Decref(Decref)));
            } finally {
                Marshal.FreeHGlobal(wrapHandle);
            }
        }

        #endregion
    }
}
