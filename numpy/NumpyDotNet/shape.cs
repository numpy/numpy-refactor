using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using IronPython.Runtime;

namespace NumpyDotNet
{
    public partial class ndarray {

        internal ndarray Flatten(NpyDefs.NPY_ORDER order) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Flatten(Array, (int)order));
        }

        internal ndarray Ravel(NpyDefs.NPY_ORDER fortran) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Ravel(Array, (int)fortran));
        }

        internal void Resize(IntPtr[] newdims, bool refcheck, NpyDefs.NPY_ORDER fortran) {
            long oldsize = Size;
            long newsize = 1;
            foreach (IntPtr dim in newdims) {
                newsize *= dim.ToInt64();
            }

            if (NpyCoreApi.NpyArrayAccess_Resize(Array, newdims.Length, newdims, (refcheck ? 1 : 0), (int)fortran) < 0) {
                NpyCoreApi.CheckError();
            }

            if (newsize > oldsize && dtype.IsObject) {
                ndarray view = (ndarray)Ravel(fortran)[new Slice(oldsize, null)];
                NpyArray.FillObjects(view, 0);
            }
        }


        internal ndarray Squeeze() {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_Squeeze(Array));
        }

        internal ndarray SwapAxes(int a1, int a2) {
            return NpyCoreApi.DecrefToInterface<ndarray>(
                NpyCoreApi.NpyArray_SwapAxes(Array, a1, a2));
        }
        
        internal ndarray Transpose(IntPtr[] permute = null) {
            if (permute == null) {
                return NpyCoreApi.DecrefToInterface<ndarray>(
                    NpyCoreApi.NpyArrayAccess_Transpose(Array, 0, null));
            } else {
                return NpyCoreApi.DecrefToInterface<ndarray>(
                    NpyCoreApi.NpyArrayAccess_Transpose(Array, permute.Length, permute));
            }
        }
    }

}
