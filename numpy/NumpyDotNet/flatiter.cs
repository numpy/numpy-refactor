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


        private IntPtr iter;
        private IntPtr current;
        private ndarray arr;
    }
}
