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
        /// Returns the size of each dimension in array elements.
        /// </summary>
        /// <param name="start">Starting dimension</param>
        /// <param name="end">Optional ending dimension</param>
        /// <returns>List of each dimension size</returns>
        IList<long> GetShape(int start, int? end);

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
        IExtBufferProtocol GetBuffer(int flags);
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
        public IExtBufferProtocol GetBufferForObject(Object o, int flags) {
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
            internal BufferProtocolAdapter(IBufferProtocol o, int flags) {
                obj = o;
            }

            #region IExtBufferProtocol

            public long IExtBufferProtocol.ItemCount {
                get { return obj.ItemCount; }
            }

            public string IExtBufferProtocol.Format {
                get { return obj.Format; } 
            }

            public int IExtBufferProtocol.ItemSize {
                get { return (int)obj.ItemSize; }
            }

            public int IExtBufferProtocol.NumberDimensions {
                get { return (int)obj.NumberDimensions; }
            }

            public bool IExtBufferProtocol.ReadOnly {
                get { return obj.ReadOnly; }
            }

            public IList<long> IExtBufferProtocol.GetShape(int start, int? end) {
                return obj.GetShape(start, end).Select(x => (long)x).ToList();
            }

            public long[] IExtBufferProtocol.Strides {
                get { return PythonOps.ConvertTupleToArray<long>(obj.Strides); }
            }

            public long[] IExtBufferProtocol.SubOffsets {
                get { return null; }        // TODO: IBufferProtocol returns an object type, what is it?!
            }

            public IntPtr IExtBufferProtocol.UnsafeAddress {
                get { return ((IPythonBufferable)obj).UnsafeAddress; }
            }

            /// <summary>
            /// Total number of bytes in the array
            /// </summary>
            public long Size {
                get { return obj.ItemCount * (long)obj.ItemSize; }
            }

            #endregion

            private IBufferProtocol obj;
        }

    }
}
