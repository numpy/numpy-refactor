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
        private enum NpyIndexTypes
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

        private int num_indexes;
        private IntPtr indexes;
    }
}
