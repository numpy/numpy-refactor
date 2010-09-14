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
