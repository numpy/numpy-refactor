using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using IronPython.Runtime;
using IronPython.Modules;
using Microsoft.Scripting;
using NumpyDotNet;

namespace NumpyDotNet {
    public class dtype {
        public dtype(CodeContext cntx, [ParamDictionary] IAttributesCollection kwargs) {
            String[] unsupportedArgs = { };

            if (pyContext == null && cntx != null) pyContext = (PythonContext)cntx.LanguageContext;

            Dictionary<String, Object> y = kwargs
                .Select((k, v) => new KeyValuePair<String, Object>(k.ToString(), v))
                .ToDictionary((kvPair => kvPair.Key), (kvPair => kvPair.Value));

            foreach (String bad in unsupportedArgs) {
                if (y.ContainsKey(bad))
                    throw new NotImplementedException(String.Format("ndarray argument '%s' is not yet implemented.", bad));
            }
        }

        // Creates a wrapper for an array created on the native side, such as the result of a slice operation.
        private dtype(IntPtr d) {
            descr = d;
        }

        ~dtype() {
            Dispose(false);
        }

        protected void Dispose(bool disposing) {
            if (descr != IntPtr.Zero) {
                IntPtr a = descr;
                descr = IntPtr.Zero;
                //SimpleArray_delete(a);
                //PythonStub.CheckError();
            }
        }

        public void Dispose() {
            Dispose(true);
            GC.SuppressFinalize(this);
        }



        private static PythonContext pyContext = null;

        /// <summary>
        ///  Pointer to the native object 
        /// </summary>
        private IntPtr descr;
    }
}
