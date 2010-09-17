#include <assert.h>

extern "C" {
#include <npy_api.h>
#include <npy_defs.h>
#include <npy_loops.h>
#include <npy_number.h>
#include <npy_math.h>
#include <npy_funcs.h>
#include <npy_ufunc_object.h>
}



typedef void *(*unaryfunc)(void *);
typedef void *(*binaryfunc)(void *, void *);


// Defined in __umath_generated.c, included below.
static void InitOperators(void *);


// Utility calls provided by the managed layer.
void *(*IPyCallMethod)(void *obj, char *meth, void *arg);
void (*IPyAddToDict)(void *dictObj, char *funcStr, void *ufuncObj);


// This structure defines all of the artimetic functions not provided by
// the core, such as those that operate on objects.
struct ExternFuncs {
    // Loop functions, may be provided by the managed or native layer.
    void (*loop_equal)(char **args, npy_intp *dimensions, npy_intp *steps, 
        void *NPY_UNUSED(func));
    void (*loop_not_equal)(char **args, npy_intp *dimensions, npy_intp *steps, 
        void *NPY_UNUSED(func));
    void (*loop_greater)(char **args, npy_intp *dimensions, npy_intp *steps, 
        void *NPY_UNUSED(func));
    void (*loop_greater_equal)(char **args, npy_intp *dimensions, npy_intp *steps, 
        void *NPY_UNUSED(func));
    void (*loop_less)(char **args, npy_intp *dimensions, npy_intp *steps, 
        void *NPY_UNUSED(func));
    void (*loop_less_equal)(char **args, npy_intp *dimensions, npy_intp *steps, 
        void *NPY_UNUSED(func));
    void (*loop_sign)(char **args, npy_intp *dimensions, npy_intp *steps, 
        void *NPY_UNUSED(func));

    // Generic arithmatic functions that operate on objects and are provided by
    // the managed layer.  These typically call into IronPython to perform the
    // operation.
    unaryfunc absolute;
    binaryfunc add, subtract, multiply, divide;
    binaryfunc trueDivide, floorDivide;
    unaryfunc invert, negative;
    binaryfunc remainder, square, power;
    binaryfunc min, max, reciprocal;
    binaryfunc and, or, xor;
    binaryfunc lshift, rshift, get_one;

    int sentinel;   // Used to verify matching structure sizes, must be last.
};

static ExternFuncs managedFuncs;



// These defines re-write the naming used by the CPython layer so we can re-use the
// generated file __umath_generated.c.
// TODO: Would be nice to refactor the CPython interface to use the same structure
// as above so we don't need this and both can share the same naming.  However, that
// isn't a priority right now.
#define npy_OBJECT_equal managedFuncs.loop_equal
#define npy_OBJECT_not_equal managedFuncs.loop_not_equal
#define npy_OBJECT_greater managedFuncs.loop_greater
#define npy_OBJECT_greater_equal managedFuncs.loop_greater_equal
#define npy_OBJECT_less managedFuncs.loop_less
#define npy_OBJECT_less_equal managedFuncs.loop_less_equal
#define npy_OBJECT_sign managedFuncs.loop_sign
#define PyNumber_Absolute managedFuncs.absolute
#define PyNumber_Add managedFuncs.add
#define PyNumber_Subtract managedFuncs.subtract
#define PyNumber_Multiply managedFuncs.multiply
#define PyNumber_Divide managedFuncs.divide
#define PyNumber_TrueDivide managedFuncs.trueDivide
#define PyNumber_FloorDivide managedFuncs.floorDivide
#define PyNumber_Invert managedFuncs.invert
#define PyNumber_Negative managedFuncs.negative
#define PyNumber_Remainder managedFuncs.remainder
#define Py_square managedFuncs.square
#define npy_ObjectPower managedFuncs.power
#define npy_ObjectMax managedFuncs.max
#define npy_ObjectMin managedFuncs.min
#define Py_reciprocal managedFuncs.reciprocal
#define PyNumber_And managedFuncs.and
#define PyNumber_Or managedFuncs.or
#define PyNumber_Xor managedFuncs.xor
#define PyNumber_Lshift managedFuncs.lshift
#define PyNumber_Rshift managedFuncs.rshift
#define Py_get_one managedFuncs.get_one


// Initializes the ufuncs.  
extern "C" __declspec(dllexport)
    void _cdecl NpyUFuncAccess_Init(void *dictionary, ExternFuncs *funcs,
    void *(*callMethod)(void *obj, char *meth, void *arg),
    void (*addToDict)(void *dictObj, char *funcStr, void *ufuncObj))
{
    // Copies the provided function pointers to our local storage.  The
    // sentinel field is used to verify that the managed structure
    // lines up with the native one.
    assert(NPY_VALID_MAGIC == funcs->sentinel);
    managedFuncs.sentinel = NPY_VALID_MAGIC;
    memcpy(&managedFuncs, funcs, sizeof(managedFuncs));
    assert(NPY_VALID_MAGIC == managedFuncs.sentinel);

    IPyCallMethod = callMethod;
    IPyAddToDict = addToDict;

    // Populates the dictionary with the function names and corresponding ufunc
    // instance.
    InitOperators(dictionary);
}



/******************************************************************************
 **                         GENERIC OBJECT lOOPS                             **
 *****************************************************************************/

/*UFUNC_API*/
void
NpyUFunc_O_O(char **args, npy_intp *dimensions, npy_intp *steps, void *func)
{
    unaryfunc f = (unaryfunc)func;
    UNARY_LOOP {
        void *in1 = *(void **)ip1;
        void **out = (void **)op1;
        void *ret = f(in1);
        if (NULL == ret) {
            return;
        }
        NpyInterface_DECREF(*out);
        *out = ret;
    }
}

/*UFUNC_API*/
void
NpyUFunc_O_O_method(char **args, npy_intp *dimensions, npy_intp *steps, void *func)
{
    char *meth = (char *)func;
    UNARY_LOOP {
        void *in1 = *(void **)ip1;
        void **out = (void **)op1;
        void *ret = IPyCallMethod(in1, meth, NULL);
        if (NULL == ret) {
            return;
        }
        NpyInterface_DECREF(*out);
        *out = ret;
    }
}

/*UFUNC_API*/
void
NpyUFunc_OO_O(char **args, npy_intp *dimensions, npy_intp *steps, void *func)
{
    binaryfunc f = (binaryfunc)func;
    BINARY_LOOP {
        void *in1 = *(void **)ip1;
        void *in2 = *(void **)ip2;
        void **out = (void **)op1;
        void *ret = f(in1, in2);
        if (NULL == ret) {
            return;
        }
        NpyInterface_DECREF(*out);
        *out = ret;
    }
}

/*UFUNC_API*/
void
NpyUFunc_OO_O_method(char **args, npy_intp *dimensions, npy_intp *steps, void *func)
{
    char *meth = (char *)func;
    BINARY_LOOP {
        void *in1 = *(void **)ip1;
        void *in2 = *(void **)ip2;
        void **out = (void **)op1;
        void *ret = IPyCallMethod(in1, meth, in2);
        if (NULL == ret) {
            return;
        }
        NpyInterface_DECREF(*out);
        *out = ret;
    }
}


// This macro is called by the code in __umath_generated.c to create the ufunc
// object and register it with the core.  The macro is needed because the same
// __umath_generated.c file is used by multiple interfaces.
#define AddFunction(func, numTypes, nin, nout, identity, nameStr, doc, check_return) \
    do {                                                                             \
        NpyUFuncObject *f = NpyUFunc_FromFuncAndData(func ## _functions,                 \
                                                     func ## _data,                      \
                                                     func ## _signatures, numTypes, nin, \
                                                     nout, identity, nameStr, doc,   \
                                                     check_return);                  \
        IPyAddToDict(dictionary, nameStr, Npy_INTERFACE(f));                         \
        Npy_DECREF(f);                                                               \
    } while (0);

#include "..\..\core\__umath_generated.c"

#undef AddFunction
