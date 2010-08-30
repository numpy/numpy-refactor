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
    /// <summary>
    /// Implements the Numpy python 'ndarray' object and acts as an interface to
    /// the core NpyArray data structure.  Npy_INTERFACE(NpyArray *) points an 
    /// instance of this class.
    /// </summary>
    public class ndarray
    {
        private static String[] ndarryArgNames = { "object", "dtype", "copy",
                                                     "order", "subok", "ndwin" };

        public ndarray(CodeContext cntx, [ParamDictionary] IAttributesCollection kwargs)
        {
            Object[] posArgs = { };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx, args);
        }

        public ndarray(CodeContext cntx, Object a1, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx, args);
        }

        public ndarray(CodeContext cntx, Object a1, Object a2, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1, a2 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx, args);
        }

        public ndarray(CodeContext cntx, Object a1, Object a2, Object a3, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1, a2, a3 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx, args);
        }

        public ndarray(CodeContext cntx, Object a1, Object a2, Object a3, Object a4, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1, a2, a3, a4 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx, args);
        }

        public ndarray(CodeContext cntx, Object a1, Object a2, Object a3, Object a4, 
            Object a5, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1, a2, a3, a4, a5 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx, args);
        }

        public ndarray(CodeContext cntx, Object a1, Object a2, Object a3, Object a4, 
            Object a5, Object a6, [ParamDictionary] IAttributesCollection kwargs) {
            Object[] posArgs = { a1, a2, a3, a4, a5, a6 };
            Object[] args = NpyUtil_ArgProcessing.BuildArgsArray(posArgs, ndarryArgNames, kwargs);

            Construct(cntx, args);
        }


        /// <summary>
        /// Arguments are: object, dtype, copy, order, subok
        /// </summary>
        /// <param name="cntx"></param>
        /// <param name="args"></param>
        private void Construct(CodeContext cntx, Object[] args) {
            if (pyContext == null && cntx != null) pyContext = (PythonContext)cntx.LanguageContext;

            Object dims = args[0];
            dtype type = (dtype)args[1];

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

        public dtype Descr {
            get { return null; }
        }

        public bool IsContiguous {
            get { return true; }        // TODO: Need real value
        }

        public bool IsFortran {
            get { return false; }       // TODO: Need real value
        }


        public bool StridingOk(NpyArray.NPY_ORDER order) {
            return order == NpyArray.NPY_ORDER.NPY_ANYORDER ||
                order == NpyArray.NPY_ORDER.NPY_CORDER && IsContiguous ||
                order == NpyArray.NPY_ORDER.NPY_FORTRANORDER && IsFortran;
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
