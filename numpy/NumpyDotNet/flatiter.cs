using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Numerics;
using IronPython.Runtime;
using IronPython.Runtime.Operations;

namespace NumpyDotNet
{
    [PythonType]
    public class flatiter : Wrapper, IEnumerator<object>
    {
        internal flatiter(IntPtr coreIter)
        {
            core = coreIter;
            arr = NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.IterArray(core));
        }

        public Object this[params object[] args]
        {
            get
            {
                ndarray result;

                using (NpyIndexes indexes = new NpyIndexes())
                {
                    NpyUtil_IndexProcessing.IndexConverter(args, indexes);
                    result = NpyCoreApi.IterSubscript(this, indexes);
                }
                if (result.ndim == 0)
                {
                    return result.dtype.ToScalar(result, 0);
                }
                else
                {
                    return result;
                }
            }

            set
            {
                using (NpyIndexes indexes = new NpyIndexes())
                {
                    NpyUtil_IndexProcessing.IndexConverter(args, indexes);

                    if (indexes.NumIndexes == 1)
                    {
                        // Special cases for single assigment.
                        switch (indexes.IndexType(0))
                        {
                            case NpyIndexes.NpyIndexTypes.INTP:
                                SingleAssign(indexes.GetIntPtr(0), value);
                                return;
                            case NpyIndexes.NpyIndexTypes.BOOL:
                                if (indexes.GetBool(0))
                                {
                                    SingleAssign(IntPtr.Zero, value);
                                }
                                return;
                            default:
                                break;
                        }
                    }

                    ndarray array_val = NpyArray.FromAny(value, arr.dtype, 0, 0, 0, null);
                    NpyCoreApi.IterSubscriptAssign(this, indexes, array_val);
                }
            }
        }

        public object this[int index] {
            set {
                SingleAssign((IntPtr)index, value);
            }
        }

        public object this[BigInteger index] {
            set {
                long lIndex = (long)index;
                SingleAssign((IntPtr)lIndex, value);
            }
        }

        public object this[bool index] {
            set {
                if (index) {
                    SingleAssign(IntPtr.Zero, value);
                }
            }
        }

        // TODO: Add comparison operators.

        public object __len__()
        {
            return Marshal.ReadIntPtr(core + NpyCoreApi.IterOffsets.off_size).ToPython();
        }

        internal long Length {
            get { return Marshal.ReadIntPtr(core, NpyCoreApi.IterOffsets.off_size).ToInt64(); }
        }

        public ndarray @base {
            get
            {
                return arr;
            }
        }

        public object index
        {
            get
            {
                return Marshal.ReadIntPtr(core + NpyCoreApi.IterOffsets.off_index).ToPython();
            }
        }

        public PythonTuple coords
        {
            get
            {
                int nd = arr.ndim;
                long[] result = new long[nd];
                IntPtr coords = NpyCoreApi.IterCoords(core);
                for (int i = 0; i < nd; i++)
                {
                    result[i] = Marshal.ReadIntPtr(coords).ToInt64();
                    coords += IntPtr.Size;
                }
                return new PythonTuple(result);
            }
        }

        public ndarray copy()
        {
            return arr.Flatten(NpyDefs.NPY_ORDER.NPY_CORDER);
        }

        /// <summary>
        /// Returns a contiguous, 1-d array that can be used to update the underlying array.  If the array
        /// is contiguous this is a 1-d view of the array.  Otherwise it is a copy with UPDATEIFCOPY set so that
        /// the data will be copied back when the returned array is freed.
        /// </summary>
        /// <returns></returns>
        public ndarray __array__(object ignored=null)
        {
            return NpyCoreApi.FlatView(arr);
        }

        #region Rich compare operators

        public object __lt__(CodeContext cntx, object o) {
            return this.__array__().__lt__(cntx, o);
        }

        public object __rlt__(CodeContext cntx, object o) {
            return this.__array__().__rlt__(cntx, o);
        }

        public object __le__(CodeContext cntx, object o) {
            return this.__array__().__le__(cntx, o);
        }

        public object __rle__(CodeContext cntx, object o) {
            return this.__array__().__rle__(cntx, o);
        }

        public object __eq__(CodeContext cntx, object o) {
            return this.__array__().__eq__(cntx, o);
        }

        public object __req__(CodeContext cntx, object o) {
            return this.__array__().__req__(cntx, o);
        }

        public object __ne__(CodeContext cntx, object o) {
            return this.__array__().__ne__(cntx, o);
        }

        public object __rne__(CodeContext cntx, object o) {
            return this.__array__().__rne__(cntx, o);
        }

        public object __ge__(CodeContext cntx, object o) {
            return this.__array__().__ge__(cntx, o);
        }

        public object __rge__(CodeContext cntx, object o) {
            return this.__array__().__rge__(cntx, o);
        }

        public object __gt__(CodeContext cntx, object o) {
            return this.__array__().__gt__(cntx, o);
        }

        public object __rgt__(CodeContext cntx, object o) {
            return this.__array__().__rgt__(cntx, o);
        }

        #endregion


        #region IEnumerator<object>

        public object Current
        {
            get {
                return arr.GetItem(current.ToInt64()-arr.UnsafeAddress.ToInt64());
            }
        }


        public bool MoveNext()
        {
            current = NpyCoreApi.IterNext(core);
            return (current != IntPtr.Zero);
        }

        public void Reset()
        {
            current = IntPtr.Zero;
            NpyCoreApi.IterReset(core);
        }

        #endregion

        #region internal methods

        internal void SingleAssign(IntPtr index, object value)
        {
            IntPtr pos = NpyCoreApi.IterGoto1D(core, index);
            if (pos == IntPtr.Zero)
            {
                NpyCoreApi.CheckError();
            }
            arr.SetItem(value, pos.ToInt64() - arr.UnsafeAddress.ToInt64());
        }

        internal object Get(IntPtr index) {
            IntPtr pos = NpyCoreApi.IterGoto1D(core, index);
            if (pos == IntPtr.Zero) {
                NpyCoreApi.CheckError();
            }
            return arr.GetItem(pos.ToInt64() - arr.UnsafeAddress.ToInt64());
        }

        #endregion

        internal IntPtr Iter
        {
            get
            {
                return core;
            }
        }

        private IntPtr current;
        private ndarray arr;
    }
}
