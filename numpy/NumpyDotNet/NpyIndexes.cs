using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using Microsoft.Scripting.Runtime;

namespace NumpyDotNet
{

    internal class NpyIndexes : IDisposable
    {

        public NpyIndexes()
        {
            indexes = Marshal.AllocCoTaskMem(NpyCoreApi.IndexInfo.sizeof_index * NpyCoreApi.IndexInfo.max_dims);
        }

        ~NpyIndexes()
        {
            FreeIndexes();
        }

        private void FreeIndexes()
        {
            if (indexes != IntPtr.Zero) {
                if (num_indexes > 0)
                {
                    NpyCoreApi.NpyArray_IndexDealloc(indexes, num_indexes);
                    num_indexes = 0;
                }
                Marshal.FreeCoTaskMem(indexes);
                indexes = IntPtr.Zero;
            }
        }

        public void Dispose()
        {
            FreeIndexes();
        }

        public int NumIndexes
        {
            get
            {
                return num_indexes;
            }
        }

        public IntPtr Indexes
        {
            get
            {
                return indexes;
            }
        }

        // The must be kept in sync with NpyIndex.h
        internal enum NpyIndexTypes
        {
            INTP,
            BOOL,
            SLICE_NOSTOP,
            SLICE,
            STRING,
            BOOL_ARRAY,
            INTP_ARRAY,
            ELLIPSIS,
            NEW_AXIS
        }

        /// <summary>
        /// Whether or not this is a simple (not fancy) index.
        /// </summary>
        public bool IsSimple
        {
            get
            {
                for (int i = 0; i < num_indexes; i++)
                {
                    switch (IndexType(i))
                    {
                        case NpyIndexTypes.BOOL_ARRAY:
                        case NpyIndexTypes.INTP_ARRAY:
                        case NpyIndexTypes.STRING:
                            return false;
                    }
                }
                return true;
            }
        }

        /// <summary>
        /// Returns whether or not this index is a single item index for an array on size ndims.
        /// </summary>
        public bool IsSingleItem(int ndims)
        {
            if (num_indexes != ndims)
            {
                return false;
            }
            for (int i = 0; i < num_indexes; i++)
            {
                if (IndexType(i) != NpyIndexTypes.INTP)
                {
                    return false;
                }
            }
            return true;
        }

        /// <summary>
        /// Finds the offset for a single item assignment to the array.
        /// </summary>
        /// <param name="arr">The array we are assigning to.</param>
        /// <returns>The offset or -1 if this is not a single assignment.</returns>
        public Int64 SingleAssignOffset(ndarray arr)
        {
            // Check to see that there are just newaxis, ellipsis, intp or bool indexes
            for (int i = 0; i < num_indexes; i++)
            {
                switch (IndexType(i))
                {
                    case NpyIndexTypes.NEW_AXIS:
                    case NpyIndexTypes.ELLIPSIS:
                    case NpyIndexTypes.INTP:
                    case NpyIndexTypes.BOOL:
                        break;
                    default:
                        return -1;
                }
            }

            // Bind to the array and calculate the offset.
            using (NpyIndexes bound = Bind(arr))
            {
                long offset = 0;
                int nd = 0;

                for (int i = 0; i < bound.num_indexes; i++)
                {
                    switch (bound.IndexType(i))
                    {
                        case NpyIndexTypes.NEW_AXIS:
                            break;
                        case NpyIndexTypes.INTP:
                            offset += arr.Stride(nd++) * bound.GetIntPtr(i).ToInt64();
                            break;
                        case NpyIndexTypes.SLICE:
                            // An ellipsis became a slice on binding. 
                            // This is not a single item assignment.
                            return -1;
                        default:
                            // This should never happen
                            return -1;
                    }
                }
                if (nd != arr.ndim)
                {
                    // Not enough indexes. This is not a single item.
                    return -1;
                }
                return offset;
            }
        }

        public NpyIndexes Bind(ndarray arr)
        {
            NpyIndexes result = new NpyIndexes();
            int n = NpyCoreApi.BindIndex(arr.Array, indexes, num_indexes, result.indexes);
            if (n < 0)
            {
                NpyCoreApi.CheckError();
            }
            else
            {
                result.num_indexes = n;
            }
            return result;
        }


        public void AddIndex(bool value)
        {
            // Write the type
            int offset = num_indexes * NpyCoreApi.IndexInfo.sizeof_index;
            Marshal.WriteInt32(indexes + offset, (Int32)NpyIndexTypes.BOOL);

            // Write the data
            offset += NpyCoreApi.IndexInfo.off_union;
            Byte val = (value ? (Byte)1 : (Byte)0);
            Marshal.WriteByte(indexes + offset, val);

            ++num_indexes;
        }
       
        public void AddIndex(IntPtr value)
        {
            // Write the type
            int offset = num_indexes * NpyCoreApi.IndexInfo.sizeof_index;
            Marshal.WriteInt32(indexes + offset, (Int32)NpyIndexTypes.INTP);

            // Write the data
            offset += NpyCoreApi.IndexInfo.off_union;
            Marshal.WriteIntPtr(indexes + offset, value);

            ++num_indexes;
        }

        public void AddIndex(ISlice slice)
        {
            IntPtr step;
            bool negativeStep;
            IntPtr start;
            IntPtr stop;
            bool hasStop;

            // Find the step
            if (slice.Step == null)
            {
                step = (IntPtr)1;
                negativeStep = false;
            }
            else
            {
                step = (IntPtr) Convert.ToInt64(slice.Step);
                negativeStep = (step.ToInt64() < 0);
            }

            // Find the start
            if (slice.Start == null)
            {
                start = (IntPtr)(negativeStep ? -1 : 0);
            }
            else
            {
                start = (IntPtr) Convert.ToInt64(slice.Start);
            }


            // Find the stop
            if (slice.Stop == null) {
                hasStop = false;
                stop = IntPtr.Zero;
            }
            else {
                hasStop = true;
                stop = (IntPtr) Convert.ToInt64(slice.Stop);
            }

            // Write the type
            int offset = num_indexes * NpyCoreApi.IndexInfo.sizeof_index;
            if (!hasStop)
            {
                Marshal.WriteInt32(indexes + offset, (Int32)NpyIndexTypes.SLICE_NOSTOP);
            }
            else
            {
                Marshal.WriteInt32(indexes + offset, (Int32)NpyIndexTypes.SLICE);
            }


            // Write the start
            offset += NpyCoreApi.IndexInfo.off_union;
            Marshal.WriteIntPtr(indexes + offset, start);

            // Write the step
            offset += IntPtr.Size;
            Marshal.WriteIntPtr(indexes + offset, step);

            // Write the stop
            if (hasStop)
            {
                offset += IntPtr.Size;
                Marshal.WriteIntPtr(indexes + offset, stop);
            }

            ++num_indexes;
        }

        public void AddBoolArray(Object arg)
        {
            // Convert to an intp array
            ndarray arr = NpyArray.FromAny(arg, NpyCoreApi.DescrFromType(NpyDefs.NPY_TYPES.NPY_BOOL), 0, 0, 0, null);
            // Write the type
            int offset = num_indexes * NpyCoreApi.IndexInfo.sizeof_index;
            Marshal.WriteInt32(indexes + offset, (Int32)NpyIndexTypes.BOOL_ARRAY);
            // Write the array
            offset += NpyCoreApi.IndexInfo.off_union;
            Marshal.WriteIntPtr(indexes + offset, arr.Array);

            ++num_indexes;
        }

        public void AddIntpArray(Object arg)
        {
            // Convert to an intp array
            ndarray arr = NpyArray.FromAny(arg, NpyCoreApi.DescrFromType(NpyDefs.NPY_INTP), 0, 0, 0, null);
            // Write the type
            int offset = num_indexes * NpyCoreApi.IndexInfo.sizeof_index;
            Marshal.WriteInt32(indexes + offset, (Int32)NpyIndexTypes.INTP_ARRAY);
            // Write the array
            offset += NpyCoreApi.IndexInfo.off_union;
            Marshal.WriteIntPtr(indexes + offset, arr.Array);

            ++num_indexes;
        }

        public void AddNewAxis()
        {
            int offset = num_indexes * NpyCoreApi.IndexInfo.sizeof_index;
            Marshal.WriteInt32(indexes + offset, (Int32)NpyIndexTypes.NEW_AXIS);
            ++num_indexes;
        }

        public void AddEllipsis()
        {
            int offset = num_indexes * NpyCoreApi.IndexInfo.sizeof_index;
            Marshal.WriteInt32(indexes + offset, (Int32)NpyIndexTypes.ELLIPSIS);
            ++num_indexes;
        }

        internal NpyIndexTypes IndexType(int n)
        {
            int offset = n * NpyCoreApi.IndexInfo.sizeof_index;
            return (NpyIndexTypes) Marshal.ReadInt32(indexes + offset);
        }

        public IntPtr GetIntPtr(int i)
        {
            int offset = i * NpyCoreApi.IndexInfo.sizeof_index + NpyCoreApi.IndexInfo.off_union;
            return Marshal.ReadIntPtr(indexes, offset);
        }

        public bool GetBool(int i)
        {
            int offset = i * NpyCoreApi.IndexInfo.sizeof_index + NpyCoreApi.IndexInfo.off_union;
            Byte val = Marshal.ReadByte(indexes, offset);
            return (val != 0);
        }

        private int num_indexes;
        private IntPtr indexes;
    }
}
