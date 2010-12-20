using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Reflection;
using IronPython.Runtime;
using IronPython.Runtime.Operations;
using IronPython.Runtime.Types;
using IronPython.Modules;
using Microsoft.Scripting;

namespace NumpyDotNet {
    /// <summary>
    /// ModuleMethods implements the module-level numpy functions.
    /// </summary>
    public static class ScalarMathModule
    {
        public const string __module__ = "numpy.core.scalarmath";


        private static object GenericOper<T>(Func<T, ScalarGeneric> newScalar, Func<T, T, T> f, object aObj, object bObj) {
            T a, b;
            switch (Convert2ToNative<T>(aObj, bObj, out a, out b)) {
                case 0:
                    break;
                case -1:
                    // One can't be cast safely, must be mixed-type.
                    throw new NotImplementedException();
                case -2:
                    throw new NotImplementedException();
                case -3:
                    throw new NotImplementedException();
                default:
                    Contract.Assert(false, "Unhandled return value from Convert2ToNative");
                    return null;
            }
            return newScalar(f(a, b));
        }

        private static int Convert2ToNative<TReturn>(object a, object b, out TReturn aResult, out TReturn bResult) {
            // TODO: This is wrong, needs to be really implemented.
            try {
                aResult = (TReturn)a;
                bResult = (TReturn)b;
            } catch {
                aResult = default(TReturn);
                bResult = default(TReturn);
                return -1;
            }
            return 0;
        }
    }
}
