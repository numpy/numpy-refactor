using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

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
            if (newsize > oldsize && IsObject) {
                throw new NotImplementedException("Expanding arrays with Resize not yet supported.");
            }

            if (NpyCoreApi.NpyArrayAccess_Resize(Array, newdims.Length, newdims, (refcheck ? 1 : 0), (int)fortran) < 0) {
                NpyCoreApi.CheckError();
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
