using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumpyDotNet
{
    public partial class ndarray
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

        internal void Sort(int axis, NpyDefs.NPY_SORTKIND sortkind) {
            if (NpyCoreApi.NpyArray_Sort(Array, axis, (int)sortkind) < 0) {
                NpyCoreApi.CheckError();
            }
        }

        internal ndarray ArgSort(int axis, NpyDefs.NPY_SORTKIND sortkind) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_ArgSort(Array, axis, (int)sortkind));
        }
    }
}
