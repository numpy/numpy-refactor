#include <assert.h>

extern "C" {
#include <npy_api.h>
#include <npy_defs.h>
#include <npy_loops.h>
#include <npy_ufunc_object.h>
}

/* This code comes from __umath_generated.c and each array is indexed by type where
   the entries are the looping function for each type and the signature. The functions
   like npy_BOOL_add are from another generated file. 

   Some of these functions will need to be passed in from the managed world.  This is
   where it gets ugly because each type and all methods for some types will be passed
   in - a lot of funcs.  Passing them in through the npy_init stage (npy_multiarray.c)
   might make the most sense or might not. */
static NpyUFuncGenericFunction add_functions[] = { npy_BOOL_add, npy_BYTE_add, npy_UBYTE_add, npy_SHORT_add, npy_USHORT_add, npy_INT_add, npy_UINT_add, npy_LONG_add, npy_ULONG_add, npy_LONGLONG_add, npy_ULONGLONG_add, npy_FLOAT_add, npy_DOUBLE_add, npy_LONGDOUBLE_add, npy_CFLOAT_add, npy_CDOUBLE_add, npy_CLONGDOUBLE_add, npy_DATETIME_Mm_M_add, npy_TIMEDELTA_mm_m_add, npy_DATETIME_mM_M_add, NULL };
static void * add_data[] = { (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL };
static char add_signatures[] = { NPY_BOOL, NPY_BOOL, NPY_BOOL, NPY_BYTE, NPY_BYTE, NPY_BYTE, NPY_UBYTE, NPY_UBYTE, NPY_UBYTE, NPY_SHORT, NPY_SHORT, NPY_SHORT, NPY_USHORT, NPY_USHORT, NPY_USHORT, NPY_INT, NPY_INT, NPY_INT, NPY_UINT, NPY_UINT, NPY_UINT, NPY_LONG, NPY_LONG, NPY_LONG, NPY_ULONG, NPY_ULONG, NPY_ULONG, NPY_LONGLONG, NPY_LONGLONG, NPY_LONGLONG, NPY_ULONGLONG, NPY_ULONGLONG, NPY_ULONGLONG, NPY_FLOAT, NPY_FLOAT, NPY_FLOAT, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_CFLOAT, NPY_CFLOAT, NPY_CFLOAT, NPY_CDOUBLE, NPY_CDOUBLE, NPY_CDOUBLE, NPY_CLONGDOUBLE, NPY_CLONGDOUBLE, NPY_CLONGDOUBLE, NPY_DATETIME, NPY_TIMEDELTA, NPY_DATETIME, NPY_TIMEDELTA, NPY_TIMEDELTA, NPY_TIMEDELTA, NPY_TIMEDELTA, NPY_DATETIME, NPY_DATETIME, NPY_OBJECT, NPY_OBJECT, NPY_OBJECT };


// Initializes the ufuncs.  
extern "C" __declspec(dllexport)
void _cdecl NpyUFuncAccess_Init()
{
    NpyUFuncObject *f;

    // This list code is similar to the function at the end of __umath_generated.c that
    // registers each of the arrays of functions.  That code sticks everything in a PyDict
    // that goes to a function in number.c that calls NpyArray_SetNumericOp.  The reason
    // for using the PyDict is that it allows any PyCallable object to be given.  In this
    // case for now we can just register all of these directly.
    f = NpyUFunc_FromFuncAndData(add_functions, add_data, add_signatures, 21,
                                 2, 1, NpyUFunc_Zero, "add",
                                 "Add arguments element-wise.\n""\n""Parameters\n""----------\n""x1, x2 : array_like\n""    The arrays to be added.\n""\n""Returns\n""-------\n""y : {ndarray, scalar}\n""    The sum of `x1` and `x2`, element-wise.  Returns scalar if\n""    both  `x1` and `x2` are scalars.\n""\n""Notes\n""-----\n""Equivalent to `x1` + `x2` in terms of array broadcasting.\n""\n""Examples\n""--------\n"">>> np.add(1.0, 4.0)\n""5.0\n"">>> x1 = np.arange(9.0).reshape((3, 3))\n"">>> x2 = np.arange(3.0)\n"">>> np.add(x1, x2)\n""array([[  0.,   2.,   4.],\n""       [  3.,   5.,   7.],\n""       [  6.,   8.,  10.]])", 0);
    NpyArray_SetNumericOp(npy_op_add, f);

}
