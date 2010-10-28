using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumpyDotNet {
    public class NpyDefs {
        #region ConstantDefs

        public const int NPY_VALID_MAGIC = 1234567;

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
        internal static readonly NPY_TYPES NPY_INTP = (IntPtr.Size == 4 ? NPY_TYPES.NPY_INT : NPY_TYPES.NPY_LONG);
        internal static readonly NPY_TYPES NPY_UINTP = (IntPtr.Size == 4 ? NPY_TYPES.NPY_UINT : NPY_TYPES.NPY_ULONG);
        internal const int NPY_NTYPES = (int)NPY_TYPES.NPY_NTYPES;

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
            NPY_COMPLEXLTR = (byte)'c',

            NPY_NOTYPELTR = 0
        }

        public enum NPY_SORTKIND
        {
            NPY_QUICKSORT = 0,
            NPY_HEAPSORT = 1,
            NPY_MERGESORT = 2
        }
        internal const int NPY_NSORTS = 3;

        public enum NPY_SEARCHSIDE
        {
            NPY_SEARCHLEFT = 0,
            NPY_SEARCHRIGHT = 1
        }
        internal const int NPY_NSEARCHSIDES = 2;

        public enum NPY_ORDER {
            NPY_ANYORDER = -1,
            NPY_CORDER = 0,
            NPY_FORTRANORDER = 1
        };

        public enum NPY_CLIPMODE
        {
            NPY_CLIP = 0,
            NPY_WRAP = 1,
            NPY_RAISE = 2
        }

        internal const int NPY_MAXDIMS = 32;
        internal const int NPY_MAXARGS = 32;

        /* The item must be reference counted when it is inserted or extracted. */
        internal const int NPY_ITEM_REFCOUNT = 0x01;
        /* Same as needing REFCOUNT */
        internal const int NPY__ITEM_HASOBJECT = 0x01;
        /* Convert to list for pickling */
        internal const int NPY_LIST_PICKLE = 0x02;
        /* The item is a POINTER  */
        internal const int NPY_ITEM_IS_POINTER = 0x04;
        /* memory needs to be initialized for this data-type */
        internal const int NPY_NEEDS_INIT = 0x08;
        /* operations need Python C-API so don't give-up thread. */
        internal const int NPY_NEEDS_PYAPI = 0x10;
        /* Use f.getitem when extracting elements of this data-type */
        internal const int NPY_USE_GETITEM = 0x20;
        /* Use f.setitem when setting creating 0-d array from this data-type.*/
        internal const int NPY_USE_SETITEM = 0x40;


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
        public const int NPY_CONTIGUOUS = 0x0001;

        /*
         * set if array is a contiguous Fortran array: the first index varies
         * the fastest in memory (strides array is reverse of C-contiguous
         * array)
         */
        public const int NPY_FORTRAN = 0x0002;

        public const int NPY_C_CONTIGUOUS = NPY_CONTIGUOUS;
        public const int NPY_F_CONTIGUOUS = NPY_FORTRAN;

        /*
         * Note: all 0-d arrays are CONTIGUOUS and FORTRAN contiguous. If a
         * 1-d array is CONTIGUOUS it is also FORTRAN contiguous
         */

        /*
         * If set, the array owns the data: it will be free'd when the array
         * is deleted.
         */
        public const int NPY_OWNDATA = 0x0004;

        /*
         * An array never has the next four set; they're only used as parameter
         * flags to the the various FromAny functions
         */

        /* Cause a cast to occur regardless of whether or not it is safe. */
        public const int NPY_FORCECAST = 0x0010;

        /*
         * Always copy the array. Returned arrays are always CONTIGUOUS,
         * ALIGNED, and WRITEABLE.
         */
        public const int NPY_ENSURECOPY = 0x0020;

        /* Make sure the returned array is a base-class ndarray */
        public const int NPY_ENSUREARRAY = 0x0040;

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
        public const int NPY_ALIGNED = 0x0100;

        /* Array data has the native endianness */
        public const int NPY_NOTSWAPPED = 0x0200;

        /* Array data is writeable */
        public const int NPY_WRITEABLE = 0x0400;

        /*
         * If this flag is set, then base contains a pointer to an array of
         * the same size that should be updated with the current contents of
         * this array when this array is deallocated
         */
        public const int NPY_UPDATEIFCOPY = 0x1000;

        /* This flag is for the array interface */
        public const int NPY_ARR_HAS_DESCR = 0x0800;


        public const int NPY_BEHAVED = (NPY_ALIGNED | NPY_WRITEABLE);
        public const int NPY_BEHAVED_NS = (NPY_ALIGNED | NPY_WRITEABLE | NPY_NOTSWAPPED);
        public const int NPY_CARRAY = (NPY_CONTIGUOUS | NPY_BEHAVED);
        public const int NPY_CARRAY_RO = (NPY_CONTIGUOUS | NPY_ALIGNED);
        public const int NPY_FARRAY = (NPY_FORTRAN | NPY_BEHAVED);
        public const int NPY_FARRAY_RO = (NPY_FORTRAN | NPY_ALIGNED);
        public const int NPY_DEFAULT = NPY_CARRAY;
        public const int NPY_IN_ARRAY = NPY_CARRAY_RO;
        public const int NPY_OUT_ARRAY = NPY_CARRAY;
        public const int NPY_INOUT_ARRAY = (NPY_CARRAY | NPY_UPDATEIFCOPY);
        public const int NPY_IN_FARRAY = NPY_FARRAY_RO;
        public const int NPY_OUT_FARRAY = NPY_FARRAY;
        public const int NPY_INOUT_FARRAY = (NPY_FARRAY | NPY_UPDATEIFCOPY);

        public const int NPY_UPDATE_ALL = (NPY_CONTIGUOUS | NPY_FORTRAN | NPY_ALIGNED);
        #endregion


        #region Array operations

        /// <summary>
        /// Array operators. Warning: this must remain in sync with NpyArray_Ops in
        /// npy_ufunc_object.h
        /// </summary>
        public enum NpyArray_Ops
        {
            npy_op_add=0,
            npy_op_subtract,
            npy_op_multiply,
            npy_op_divide,
            npy_op_remainder,
            npy_op_power,
            npy_op_square,
            npy_op_reciprocal,
            npy_op_ones_like,
            npy_op_sqrt,
            npy_op_negative,
            npy_op_absolute,
            npy_op_invert,
            npy_op_left_shift,
            npy_op_right_shift,
            npy_op_bitwise_and,
            npy_op_bitwise_xor,
            npy_op_bitwise_or,
            npy_op_less,
            npy_op_less_equal,
            npy_op_equal,
            npy_op_not_equal,
            npy_op_greater,
            npy_op_greater_equal,
            npy_op_floor_divide,
            npy_op_true_divide,
            npy_op_logical_or,
            npy_op_logical_and,
            npy_op_floor,
            npy_op_ceil,
            npy_op_maximum,
            npy_op_minimum,
            npy_op_rint,
            npy_op_conjugate,
            npy_op_end_of_ops       // Must be the last entry
        };

        #endregion

        #region umath errors

        public const int NPY_BUFSIZE = 10000;
        public const int NPY_MIN_BUFSIZE = 2 * sizeof(double);
        public const int NPY_MAX_BUFSIZE = NPY_MIN_BUFSIZE * 1000000;

        public enum NPY_UFUNC_FPE
        {
            DIVIDEBYZERO = 1,
            OVERFLOW = 2,
            UNDERFLOW = 4,
            INVALID = 8
        }

        public enum NPY_UFUNC_ERR 
        {
            IGNORE = 0,
            WARN = 1,
            RAISE = 2,
            CALL = 3,
            PRINT = 4,
            LOG = 5
        }

        public enum NPY_UFUNC_MASK
        {
            DIVIDEBYZERO = 0x07,
            OVERFLOW = 0x3f,
            UNDERFLOW = 0x1ff,
            INVALID = 0xfff
        }

        public enum NPY_UFUNC_SHIFT
        {
            DIVIDEBYZERO = 0,
            OVERFLOW = 3,
            UNDERFLOW = 6,
            INVALID = 9
        }

        public enum NPY_DATETIMEUNIT : int
        {
            NPY_FR_Y=0,
            NPY_FR_M,
            NPY_FR_W,
            NPY_FR_B,
            NPY_FR_D,
            NPY_FR_h,
            NPY_FR_m,
            NPY_FR_s,
            NPY_FR_ms,
            NPY_FR_us,
            NPY_FR_ns,
            NPY_FR_ps,
            NPY_FR_fs,
            NPY_FR_as
        }

        public const int NPY_UFUNC_ERR_DEFAULT = 0;
        public const int NPY_UFUNC_ERR_DEFAULT2 =
            ((int)NPY_UFUNC_ERR.PRINT << (int)NPY_UFUNC_SHIFT.DIVIDEBYZERO) +
            ((int)NPY_UFUNC_ERR.PRINT << (int)NPY_UFUNC_SHIFT.OVERFLOW) +
            ((int)NPY_UFUNC_ERR.PRINT << (int)NPY_UFUNC_SHIFT.INVALID);

        #endregion


        #region Type functions

        public static bool IsBool(NPY_TYPES type) {
            return type == NPY_TYPES.NPY_BOOL;
        }

        public static bool IsUnsigned(NPY_TYPES type) {
            return type == NPY_TYPES.NPY_UBYTE || type == NPY_TYPES.NPY_USHORT ||
                type == NPY_TYPES.NPY_UINT || type == NPY_TYPES.NPY_ULONG ||
                type == NPY_TYPES.NPY_ULONGLONG;
        }

        public static bool IsSigned(NPY_TYPES type) {
            return type == NPY_TYPES.NPY_BYTE || type == NPY_TYPES.NPY_SHORT ||
                type == NPY_TYPES.NPY_INT || type == NPY_TYPES.NPY_LONG ||
                type == NPY_TYPES.NPY_LONGLONG;
        }

        public static bool IsInteger(NPY_TYPES type) {
            return NPY_TYPES.NPY_BYTE <= type && type <= NPY_TYPES.NPY_ULONGLONG;
        }

        public static bool IsFloat(NPY_TYPES type) {
            return NPY_TYPES.NPY_FLOAT <= type && type <= NPY_TYPES.NPY_LONGDOUBLE;
        }

        public static bool IsNumber(NPY_TYPES type) {
            return type <= NPY_TYPES.NPY_CLONGDOUBLE;
        }

        public static bool IsString(NPY_TYPES type) {
            return type == NPY_TYPES.NPY_STRING || type == NPY_TYPES.NPY_UNICODE;
        }

        public static bool IsComplex(NPY_TYPES type) {
            return NPY_TYPES.NPY_CFLOAT <= type && type <= NPY_TYPES.NPY_CLONGDOUBLE;
        }

        public static bool IsPython(NPY_TYPES type) {
            return type == NPY_TYPES.NPY_LONG || type == NPY_TYPES.NPY_DOUBLE ||
                type == NPY_TYPES.NPY_CDOUBLE || type == NPY_TYPES.NPY_BOOL ||
                type == NPY_TYPES.NPY_OBJECT;
        }

        public static bool IsFlexible(NPY_TYPES type) {
            return NPY_TYPES.NPY_STRING <= type && type <= NPY_TYPES.NPY_VOID;
        }

        public static bool IsDatetime(NPY_TYPES type) {
            return NPY_TYPES.NPY_DATETIME <= type && type <= NPY_TYPES.NPY_TIMEDELTA;
        }

        public static bool IsUserDefined(NPY_TYPES type) {
            return NPY_TYPES.NPY_USERDEF <= type && 
                (int)type <= (int)NPY_TYPES.NPY_USERDEF + 0; // TODO: Need GetNumUserTypes
        }

        public static bool IsExtended(NPY_TYPES type) {
            return IsFlexible(type) || IsUserDefined(type);
        }

        public static bool IsNativeByteOrder(byte endian) {
            return endian != NpyCoreApi.OppositeByteOrder;
        }

        #endregion

    }
}
