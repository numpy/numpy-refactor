using System;
using System.Collections.Generic;
using IronPython.Runtime;
using Microsoft.Scripting;

namespace NumpyDotNet
{
    interface IArray
    {
        object all(object axis = null, ndarray @out = null);
        object any(object axis = null, ndarray @out = null);
        object argmax(object axis = null, ndarray @out = null);
        object argmin(object axis = null, ndarray @out = null);
        object argsort(object axis = null, string kind = null, object order = null);
        ndarray astype(IronPython.Runtime.CodeContext cntx, object dtype = null);
        ndarray byteswap(bool inplace = false);
        object choose([ParamDictionary] IDictionary<object,object> kwargs, params object[] args);
        object clip(object min = null, object max = null, ndarray @out = null);
        ndarray compress(object condition, object axis = null, ndarray @out = null);
        ndarray conj(ndarray @out = null);
        ndarray conjugate(ndarray @out = null);
        ndarray copy(object order = null);
        object cumprod(IronPython.Runtime.CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null);
        object cumsum(IronPython.Runtime.CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null);
        PythonBuffer data { get; }
        ndarray diagonal(int offset = 0, int axis1 = 0, int axis2 = 1);
        dtype dtype { get; set; }
        void fill(object scalar);
        flagsobj flags { get; }
        object flat { get; set; }
        ndarray flatten(object order = null);
        ndarray getfield(IronPython.Runtime.CodeContext cntx, object dtype, int offset = 0);
        object imag { get; set; }
        object item(params object[] args);
        void itemset(params object[] args);
        object max(object axis = null, ndarray @out = null);
        object mean(IronPython.Runtime.CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null);
        object min(object axis = null, ndarray @out = null);
        int ndim { get; }
        ndarray newbyteorder(string endian = null);
        IronPython.Runtime.PythonTuple nonzero();
        object prod(IronPython.Runtime.CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null);
        object ptp(object axis = null, ndarray @out = null);
        void put(object indices, object values, object mode = null);
        ndarray ravel(object order = null);
        object real { get; set; }
        object repeat(object repeats, object axis = null);
        ndarray reshape(IDictionary<object,object> kwds, params object[] args);
        void resize(IDictionary<object,object> kwds, params object[] args);
        object round(int decimals = 0, ndarray @out = null);
        object searchsorted(object keys, string side = null);
        void setfield(IronPython.Runtime.CodeContext cntx, object value, object dtype, int offset = 0);
        void setflags(object write = null, object align = null, object uic = null);
        object shape { get; }
        object size { get; }
        void sort(int axis = -1, string kind = null, object order = null);
        object squeeze();
        object std(IronPython.Runtime.CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null, int ddof = 0);
        long[] Strides { get; }
        PythonTuple strides { get; }
        object sum(IronPython.Runtime.CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null);
        ndarray swapaxes(int a1, int a2);
        ndarray swapaxes(object a1, object a2);
        object take(object indices, object axis = null, ndarray @out = null, object mode = null);
        object this[params object[] args] { get; set; }
        object this[int index] { get; }
        object this[long index] { get; }
        object this[IntPtr index] { get; }
        object this[System.Numerics.BigInteger index] { get; }
        void tofile(IronPython.Runtime.CodeContext cntx, IronPython.Runtime.PythonFile file, string sep = null, string format = null);
        void tofile(IronPython.Runtime.CodeContext cntx, string filename, string sep = null, string format = null);
        object tolist();
        IronPython.Runtime.Bytes tostring(object order = null);
        object trace(IronPython.Runtime.CodeContext cntx, int offset = 0, int axis1 = 0, int axis2 = 1, object dtype = null, ndarray @out = null);
        ndarray transpose(params object[] args);
        object var(IronPython.Runtime.CodeContext cntx, object axis = null, object dtype = null, ndarray @out = null, int ddof = 0);
        ndarray view(IronPython.Runtime.CodeContext cntx, object dtype = null, object type = null);
    }
}
