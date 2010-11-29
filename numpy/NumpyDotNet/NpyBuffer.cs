using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using IronPython.Runtime;
using IronPython.Runtime.Operations;

namespace NumpyDotNet
{
    /// <summary>
    /// Extended buffer protocol.  This is very similar to IBufferProtocol but adds the ability
    /// to get an unsafe address for the data.  It does not implement IBufferProtocol for now
    /// because several methods on that interface are unnecessary for the time being.
    /// </summary>
    public interface IExtBufferProtocol
    {
        /// <summary>
        /// Number of items in the buffer
        /// </summary>
        long ItemCount {
            get;
        }

        string Format {
            get;
        }

        /// <summary>
        /// Size of each element in bytes.
        /// </summary>
        int ItemSize {
            get;
        }

        /// <summary>
        /// Number of dimensions in each array.
        /// </summary>
        int NumberDimensions {
            get;
        }

        /// <summary>
        /// True if array can not be written to, false if data (UnsafeAddress) is writable.
        /// </summary>
        bool ReadOnly {
            get;
        }

        /// <summary>
        /// Size of each dimension in array elements.
        /// </summary>
        /// <returns>List of each dimension size</returns>
        IList<long> Shape {
            get;
        }

        /// <summary>
        /// Number of bytes to skip to get to the next element in each dimension.
        /// </summary>
        long[] Strides {
            get;
        }

        long[] SubOffsets {
            get;
        }


        /// <summary>
        /// UNSAFE!  Address of the base of the data array.  Use the strides and
        /// sub offsets to index into the array. UnsafeAddress() + Size is the address
        /// of the top of the buffer memory range.
        /// </summary>
        IntPtr UnsafeAddress {
            get;
        }

        /// <summary>
        /// Total size of the buffer in bytes.
        /// </summary>
        long Size {
            get;
        }
    }


    /// <summary>
    /// Indicates that a given object can provide an adapter that implements the buffer
    /// protocol.
    /// </summary>
    public interface IBufferProvider
    {
        IExtBufferProtocol GetBuffer(NpyBuffer.PyBuf flags);
    }



    // Temporary until real IPythonBufferable is exposed.
    public interface IPythonBufferable
    {
        IntPtr UnsafeAddress {
            get;
        }

        int Size {
            get;
        }
    }


    /// <summary>
    /// Provides utilities and adapters that mimick the CPython PEP 3118 buffer protocol
    /// as far as currently needed.
    /// </summary>
    public static class NpyBuffer
    {
        [Flags]
        public enum PyBuf
        {
            SIMPLE = 0x00,
            WRITABLE = 0x01,
            FORMAT = 0x02,
            ND = 0x04,
            STRIDES = 0x0C,         // Implies ND
            C_CONTIGUOUS = 0x1C,    // Implies STRIDES
            F_CONTIGUOUS = 0x2C,    // Implies STRIDES
            ANY_CONTIGUOUS = 0x4C,  // Implies STRIDES
            INDIRECT = 0x8C,        // Implies STRIDES

            // Composite sets
            CONTIG = 0x41,          // Multidimensional ( ND | WRITABLE )
            CONTIG_RO = 0x40,       // ND
            STRIDED = 0x0D,         // Multidimensional, aligned ( STRIDES | WRITABLE )
            STRIDED_RO = 0x0C,      // STRIDES
            RECORDS = 0x0F,         // Multidimensional, unaligned (STRIDEs | WRITABLE | FORMAT )
            RECORDS_RO = 0x0E,      // STRIDES | FORMAT
            FULL = 0x8F,            // Multidimensional using sub-offsets
            FULL_RO = 0x8E          //
        }


        public static IExtBufferProtocol GetBufferForObject(Object o, PyBuf flags) {
            if (o is IBufferProvider) {
                return ((IBufferProvider)o).GetBuffer(flags);
            } else if (o is IPythonBufferable) {
                if (o is IBufferProvider) {
                    return new BufferProtocolAdapter(o as IBufferProtocol, flags);
                }
                // TODO: Support for objects only implementing IPythonBufferable is
                // not supported because no examples exist and it's unclear whether
                // the Size property is the total number of bytes in the array or
                // the size of an individual item type. (Should be the former, but
                // is NativeType.Size in the array implementation.)
            }
            return null;
        }


        /// <summary>
        /// Adapts an instance that implements IBufferProtocol and IPythonBufferable
        /// to the IExtBufferProtocol.
        /// </summary>
        internal class BufferProtocolAdapter : IExtBufferProtocol
        {
            internal BufferProtocolAdapter(IBufferProtocol o, PyBuf flags) {
                obj = o;

                if (obj.NumberDimensions > 1) {
                    // CTypes.Arrays supports multi-dimensional arrays, but only by nesting arrays as array elements.
                    // Thus we aren't talking about a single memory block.  This might be supportable using the
                    // subOffsets returns except that array always returns null.  So this is unsupported until we
                    // have an example to test against.
                    throw new ArgumentException("Multi-dimensional arrays are not yet supported.");
                }

                forceByteArray = (flags == PyBuf.SIMPLE);

                if ((flags & PyBuf.FORMAT) != 0 && obj.Format == null) {
                    throw new ArgumentException("Object does not provide format information.");
                }
                if ((flags & PyBuf.ND) != 0 && obj.GetShape(0, 0) == null) {
                    throw new ArgumentException("Object does not provide shape information.");
                }

                if ((flags & PyBuf.STRIDES) == 0 && obj.Strides != null) {
                    throw new ArgumentException("Object is not contiguous.");
                }

                if ((flags & PyBuf.WRITABLE) != 0 && obj.ReadOnly) {
                    throw new ArgumentException("Object is read-only.");
                }
            }

            #region IExtBufferProtocol

            long IExtBufferProtocol.ItemCount {
                get { return forceByteArray ? Size : obj.ItemCount; }
            }

            string IExtBufferProtocol.Format {
                get { return obj.Format; }
            }

            int IExtBufferProtocol.ItemSize {
                get { return forceByteArray ? sizeof(Byte) : (int)obj.ItemSize; }
            }

            int IExtBufferProtocol.NumberDimensions {
                get { return (int)obj.NumberDimensions; }
            }

            bool IExtBufferProtocol.ReadOnly {
                get { return obj.ReadOnly; }
            }

            IList<long> IExtBufferProtocol.Shape {
                get { return obj.GetShape(0, 0).Select(x => (long)x).ToList(); }
            }

            long[] IExtBufferProtocol.Strides {
                get { return PythonOps.ConvertTupleToArray<long>(obj.Strides); }
            }

            long[] IExtBufferProtocol.SubOffsets {
                get { return null; }        // TODO: IBufferProtocol returns an object type, what is it?!
            }

            IntPtr IExtBufferProtocol.UnsafeAddress {
                get { return ((IPythonBufferable)obj).UnsafeAddress; }
            }

            /// <summary>
            /// Total number of bytes in the array
            /// </summary>
            public long Size {
                get { return obj.ItemCount * (long)obj.ItemSize; }
            }

            #endregion

            private readonly IBufferProtocol obj;
            private readonly bool forceByteArray;
        }

    }
}
