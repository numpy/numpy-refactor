using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumpyDotNet
{
    public class flatiter : IDisposable, IEnumerator<object>
    {
        internal flatiter(IntPtr coreIter)
        {
            iter = coreIter;
            arr = NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.IterArray(iter));
        }

        ~flatiter()
        {
            Dispose(false);
        }

        protected void Dispose(bool disposing)
        {
            if (iter != IntPtr.Zero)
            {
                lock (this) {
                    IntPtr a = iter;
                    iter = IntPtr.Zero;
                    NpyCoreApi.Dealloc(a);
                }
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
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
                    // TODO: Should return a numpy scalar
                    return result.GetItem(0);
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

        #region IEnumerator<object>

        public object Current
        {
            get {
                return arr.GetItem(current.ToInt64()-arr.data.ToInt64());
            }
        }


        public bool MoveNext()
        {
            current = NpyCoreApi.IterNext(iter);
            return (current != IntPtr.Zero);
        }

        public void Reset()
        {
            current = IntPtr.Zero;
            NpyCoreApi.IterReset(iter);
        }

        #endregion

        #region internal methods

        private void SingleAssign(IntPtr index, object value)
        {
            IntPtr pos = NpyCoreApi.IterGoto1D(iter, index);
            if (pos == IntPtr.Zero)
            {
                NpyCoreApi.CheckError();
            }
            arr.SetItem(value, pos.ToInt64() - arr.data.ToInt64());
        }

        #endregion

        internal IntPtr Iter
        {
            get
            {
                return iter;
            }
        }

        private IntPtr iter;
        private IntPtr current;
        private ndarray arr;
    }
}
