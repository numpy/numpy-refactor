using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Scripting;
using Microsoft.Scripting.Runtime;
using IronPython.Runtime;

namespace NumpyDotNet {
    /// <summary>
    /// Package of extension methods.
    /// </summary>
    public static class NpyUtils_Extensions {

        /// <summary>
        /// Applies function f to all elements in 'input'. Same as Select() but
        /// with no result.
        /// </summary>
        /// <typeparam name="Tin">Element type</typeparam>
        /// <param name="input">Input sequence</param>
        /// <param name="f">Function to be applied</param>
        public static void Iter<Tin>(this IEnumerable<Tin> input, Action<Tin> f) {
            foreach (Tin x in input) {
                f(x);
            }
        }

        /// <summary>
        /// Applies function f to all elements in 'input' plus the index of each
        /// element.
        /// </summary>
        /// <typeparam name="Tin">Type of input elements</typeparam>
        /// <param name="input">Input sequence</param>
        /// <param name="f">Function to be applied</param>
        public static void Iteri<Tin>(this IEnumerable<Tin> input, Action<Tin, int> f) {
            int i = 0;
            foreach (Tin x in input) {
                f(x, i);
                i++;
            }
        }
    }


    internal static class NpyUtil_ArgProcessing {

        internal static bool BoolConverter(Object o) {
            if (o is Boolean) return (bool)o;
            else if (o is IConvertible) return Convert.ToBoolean(o);

            throw new ArgumentException(String.Format("Unable to convert argument '{0}' to Boolean value.", o));
        }


        internal static int IntConverter(Object o) {
            if (o is int) return (int)o;
            else if (o is IConvertible) return Convert.ToInt32(o);

            throw new ArgumentException(String.Format("Unable to convert argument '{0}' to Boolean value.", o));
        }

        internal static Object[] BuildArgsArray(Object[] posArgs, String[] kwds,
            IAttributesCollection namedArgs) {
            // For some reason the name of the attribute can only be access via ToString
            // and not as a key so we fix that here.
            Dictionary<String, Object> argsDict = namedArgs
                .Select(kvPair => new KeyValuePair<String, Object>(kvPair.Key.ToString(), kvPair.Value))
                .ToDictionary((kvPair => kvPair.Key), (kvPair => kvPair.Value));

            // The result, filled in as we go.
            Object[] args = new Object[kwds.Length];
            int i;

            // Copy in the position arguments.
            for (i = 0; i < posArgs.Length; i++) {
                if (argsDict.ContainsKey(kwds[i])) {
                    throw new ArgumentException(String.Format("Argument '{0}' is specified both positionally and by name.", kwds[i]));
                }
                args[i] = posArgs[i];
            }

            // Now insert any named arguments into the correct position.
            for (i = posArgs.Length; i < kwds.Length; i++) {
                if (argsDict.TryGetValue(kwds[i], out args[i])) {
                    argsDict.Remove(kwds[i]);
                } else {
                    args[i] = null;
                }
            }
            if (argsDict.Count > 0) {
                throw new ArgumentException("Unknown named arguments were specified.");
            }
            return args;
        }

        // Don't like how this turned out, will probably go away, but might
        // be useful for another function.
#if goaway
        internal static bool parseArgKeywords<T0>(Object[] posArgs, String[] kwds,
            IAttributesCollection namedArgs, 
            Func<Object, T0> f0, out T0 arg0) {

            if (kwds.Length != 1)
                throw new ArgumentOutOfRangeException(String.Format("Internal error: parseArgKeywords call with incorrect number of argument keywords ({0}, expected {1}.",
                    kwds.Length, 1));

            Object[] args = buildArgsArray(posArgs, kwds, namedArgs);

            if (f0 == null) arg0 = (T0)args[0];
            else arg0 = f0(args[0]);

            return false;
        }

        internal static bool parseArgKeywords<T0, T1>(Object[] posArgs, String[] kwds,
            IAttributesCollection namedArgs, 
            Func<Object, T0> f0, out T0 arg0,
            Func<Object, T1> f1, out T1 arg1) {

            if (kwds.Length != 2)
                throw new ArgumentOutOfRangeException(String.Format("Internal error: parseArgKeywords call with incorrect number of argument keywords ({0}, expected {1}.",
                    kwds.Length, 2));

            Object[] args = buildArgsArray(posArgs, kwds, namedArgs);

            if (f0 == null) arg0 = (T0)args[0];
            else arg0 = f0(args[0]);

            if (f1 == null) arg1 = (T1)args[1];
            else arg1 = f1(args[1]);

            return false;
        }

        internal static bool parseArgKeywords<T0, T1, T2>(Object[] posArgs, String[] kwds,
            IAttributesCollection namedArgs, 
            Func<Object, T0> f0, out T0 arg0,
            Func<Object, T1> f1, out T1 arg1,
            Func<Object, T2> f2, out T2 arg2) {

            if (kwds.Length != 3)
                throw new ArgumentOutOfRangeException(String.Format("Internal error: parseArgKeywords call with incorrect number of argument keywords ({0}, expected {1}.",
                    kwds.Length, 3));

            Object[] args = buildArgsArray(posArgs, kwds, namedArgs);

            if (f0 == null) arg0 = (T0)args[0];
            else arg0 = f0(args[0]);

            if (f1 == null) arg1 = (T1)args[1];
            else arg1 = f1(args[1]);

            if (f2 == null) arg2 = (T2)args[2];
            else arg2 = f2(args[2]);

            return false;
        }

        internal static bool parseArgKeywords<T0, T1, T2, T3, T4, T5>(Object[] posArgs, String[] kwds,
            IAttributesCollection namedArgs, 
            Func<Object, T0> f0, out T0 arg0,
            Func<Object, T1> f1, out T1 arg1,
            Func<Object, T2> f2, out T2 arg2,
            Func<Object, T3> f3, out T3 arg3,
            Func<Object, T4> f4, out T4 arg4,
            Func<Object, T5> f5, out T5 arg5) {

            if (kwds.Length != 3)
                throw new ArgumentOutOfRangeException(String.Format("Internal error: parseArgKeywords call with incorrect number of argument keywords ({0}, expected {1}.",
                    kwds.Length, 3));

            Object[] args = buildArgsArray(posArgs, kwds, namedArgs);

            if (f0 == null) arg0 = (T0)args[0];
            else arg0 = f0(args[0]);

            if (f1 == null) arg1 = (T1)args[1];
            else arg1 = f1(args[1]);

            if (f2 == null) arg2 = (T2)args[2];
            else arg2 = f2(args[2]);

            if (f3 == null) arg3 = (T3)args[3];
            else arg3 = f3(args[3]);

            if (f4 == null) arg4 = (T4)args[4];
            else arg4 = f4(args[4]);

            if (f5 == null) arg5 = (T5)args[5];
            else arg5 = f5(args[5]);

            return false;
        }
#endif
    }


    internal static class NpyUtil_IndexProcessing
    {
        public static void IndexConverter(Object[] indexArgs, NpyIndexes indexes)
        {
            if (indexArgs.Length == 1 && indexArgs[0] is PythonTuple) {
                // Treat a single tuple as a tuple of args.
                PythonTuple tuple = (PythonTuple)indexArgs[0];
                if (tuple.Count > NpyCoreApi.IndexInfo.max_dims) {
                    throw new IndexOutOfRangeException("Too many indices.");
                }
                foreach (object arg in tuple) {
                    ConvertSingleIndex(arg, indexes);
                }
            } 
            else {
                if (indexArgs.Length > NpyCoreApi.IndexInfo.max_dims) {
                    throw new IndexOutOfRangeException("Too many indices.");
                }
                foreach (Object arg in indexArgs)
                {
                    ConvertSingleIndex(arg, indexes);
                }
            }
        }

        private static void ConvertSingleIndex(Object arg, NpyIndexes indexes)
        {
            if (arg == null)
            {
                indexes.AddNewAxis();
            }
            else if (arg is IronPython.Runtime.Types.Ellipsis)
            {
                indexes.AddEllipsis();
            }
            else if (arg is bool)
            {
                indexes.AddIndex((bool)arg);
            }
            else if (arg is ISlice)
            {
                indexes.AddIndex((ISlice)arg);
            }
            else if (arg is int)
            {
                indexes.AddIndex((IntPtr)(int)arg);
            }
            else if (arg is long)
            {
                indexes.AddIndex((IntPtr)(long)arg);
            }
            else if (arg is IConvertible)
            {
                if (IntPtr.Size == 4)
                {
                    indexes.AddIndex((IntPtr)Convert.ToInt32(arg));
                }
                else
                {
                    indexes.AddIndex((IntPtr)Convert.ToInt64(arg));
                }
            }
            else
            {
                throw new ArgumentException(String.Format("Argument '{0}' is not a valid index.", arg));
            }
        }
    }
}
