using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using IronPython.Runtime;
using IronPython.Runtime.Operations;

namespace NumpyDotNet
{
    public partial class ndarray : IEnumerable<object> 
    {
        internal ndarray TakeFrom(ndarray indices, int axis, ndarray ret, NpyDefs.NPY_CLIPMODE clipMode) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_TakeFrom(Array, indices.Array, axis, (ret != null ? ret.Array : IntPtr.Zero), (int)clipMode)
                );
        }

        internal void PutTo(ndarray values, ndarray indices, NpyDefs.NPY_CLIPMODE mode) {
            if (NpyCoreApi.NpyArray_PutTo(Array, values.Array, indices.Array, (int)mode) < 0) {
                NpyCoreApi.CheckError();
            }
        }

        internal void PutMask(ndarray values, ndarray mask) {
            if (NpyCoreApi.NpyArray_PutMask(Array, values.Array, mask.Array) < 0) {
                NpyCoreApi.CheckError();
            }
        }

        internal ndarray Repeat(ndarray repeats, int axis) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Repeat(Array, repeats.Array, axis));
        }

        internal ndarray Choose(IEnumerable<object> objs, ndarray ret = null, NpyDefs.NPY_CLIPMODE clipMode = NpyDefs.NPY_CLIPMODE.NPY_RAISE) {
            ndarray[] arrays = NpyUtil_ArgProcessing.ConvertToCommonType(objs);
            IntPtr[] coreArrays = arrays.Select(x => x.Array).ToArray();
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Choose(Array, coreArrays, coreArrays.Length,
                ret==null ? IntPtr.Zero : ret.Array, (int)clipMode));
        }

        internal void Sort(int axis, NpyDefs.NPY_SORTKIND sortkind) {
            if (NpyCoreApi.NpyArray_Sort(Array, axis, (int)sortkind) < 0) {
                NpyCoreApi.CheckError();
            }
        }

        internal ndarray ArgSort(int axis, NpyDefs.NPY_SORTKIND sortkind) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_ArgSort(Array, axis, (int)sortkind));
        }

        internal static ndarray LexSort(ndarray[] arrays, int axis) {
            int n = arrays.Length;
            IntPtr[] coreArrays = arrays.Select(x => x.Array).ToArray();
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_LexSort(coreArrays, n, axis));
        }

        internal ndarray SearchSorted(ndarray keys, NpyDefs.NPY_SEARCHSIDE side) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_SearchSorted(Array, keys.Array, (int)side));
        }

        internal ndarray Diagonal(int offset, int axis1, int axis2) {
            int n = ndim;
            IntPtr[] newaxes = new IntPtr[n];

            // Check the axes
            if (n < 2) {
                throw new ArgumentException("array.ndim must be >= 2");
            }
            if (axis1 < 0) {
                axis1 += n;
            }
            if (axis2 < 0) {
                axis2 += n;
            }
            if (axis1 == axis2 ||
                axis1 < 0 || axis1 >= n ||
                axis2 < 0 || axis2 >= n) {
                throw new ArgumentException(
                    String.Format("axis1(={0}) and axis2(={1}) must be different and within range (nd={3})",
                        axis1, axis2, n));
            }

            // Transpose to be axis1 and axis2 at the end
            newaxes[n - 2] = (IntPtr)axis1;
            newaxes[n - 1] = (IntPtr)axis2;
            int pos = 0;
            for (int i = 0; i < n; i++) {
                if (i != axis1 && i != axis2) {
                    newaxes[pos++] = (IntPtr)i;
                }
            }
            ndarray newarray = Transpose(newaxes);

            if (n == 2) {
                long n1, n2, start, stop, step;

                n1 = newarray.Dims[0];
                n2 = newarray.Dims[1];
                step = n2 + 1;
                if (offset < 0) {
                    start = -n2 * offset;
                    stop = Math.Min(2, n1 + offset) * (n2 + 1) - n2 * offset;
                } else {
                    start = offset;
                    stop = Math.Min(n1, n2 - offset) * (n2 + 1) + offset;
                }

                Slice slice = new Slice(((IntPtr)start).ToPython(), ((IntPtr)stop).ToPython(), ((IntPtr)step).ToPython());
                return (ndarray)newarray.Flat[slice];
            } else {
                // my_diagonal = []
                // for i in range(s[0]):
                // my_diagonal.append(diagonal(a[i], offset))
                // return array(my_diagonal);
                dtype typecode = newarray.dtype;
                List my_diagonal = new List();
                long n1 = newarray.Dims[0];
                for (long i=0; i<n1; i++) {
                    ndarray sub = (ndarray)newarray[i];
                    my_diagonal.append(sub.Diagonal(offset, n-3, n-2));
                }
                return NpyArray.FromAny(my_diagonal, typecode, 0, 0, 0, null);
            }
        }


        internal ndarray[] NonZero() {
            int nd = ndim;
            IntPtr[] coreArrays = new IntPtr[nd];
            // TODO: We should be passing the managed array as the last arg for subtypes.
            if (NpyCoreApi.NpyArray_NonZero(Array, coreArrays, IntPtr.Zero) < 0) {
                NpyCoreApi.CheckError();
            }
            ndarray[] result = new ndarray[nd];
            for (int i = 0; i < nd; i++) {
                result[i] = NpyCoreApi.DecrefToInterface<ndarray>(coreArrays[i]);
            }
            return result;
        }

    }


}
