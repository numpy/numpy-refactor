using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using IronPython.Runtime;
using IronPython.Modules;
using Microsoft.Scripting;
using NumpyDotNet;

namespace NumpyDotNet
{
    public class ndarray
    {
        public ndarray(CodeContext cntx, [ParamDictionary] IAttributesCollection kwargs)
        {
            String[] unsupportedArgs = { "buffer", "offset", "strides", "order" };

            if (pyContext == null && cntx != null) pyContext = (PythonContext)cntx.LanguageContext;

            Dictionary<String, Object> y = kwargs
                .Select((k, v) => new KeyValuePair<String, Object>(k.ToString(), v))
                .ToDictionary((kvPair => kvPair.Key), (kvPair => kvPair.Value));

            foreach(String bad in unsupportedArgs) {
                if (y.ContainsKey(bad))
                    throw new NotImplementedException(String.Format("ndarray argument '%s' is not yet implemented.", bad));
            }

            Object dims;
            Object descr;
            
            if (!y.TryGetValue("shape", out dims)) dims = null;
            if (!y.TryGetValue("dtype", out descr))
                descr = NpyArray.DescrFromType(NpyArray.NPY_DEFAULT);

            array = IntPtr.Zero;
        }

        // Creates a wrapper for an array created on the native side, such as the result of a slice operation.
        internal ndarray(IntPtr a)
        {
            array = a;
        }

        ~ndarray()
        {
            Dispose(false);
        }

        protected void Dispose(bool disposing)
        {
            if (array != IntPtr.Zero)
            {
                IntPtr a = array;
                array = IntPtr.Zero;
                //SimpleArray_delete(a);
                //PythonStub.CheckError();
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }



        private static PythonContext pyContext = null;

        /// <summary>
        ///  Pointer to the native object 
        /// </summary>
        private IntPtr array;

        public static void Main(String[] args) {
            NpyArray.SimpleArray_create(42);
        }
    }
}
