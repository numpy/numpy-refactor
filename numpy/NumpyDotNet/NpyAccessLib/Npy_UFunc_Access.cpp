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
typedef int (*cmpfunc)(void *, void *);

// Defined in __umath_generated.c, included below.
static void InitOperators(void *);
static void InitOtherOperators(void *dictionary, 
    void (*addToDict)(void *dictObj, const char *funcStr, void *ufuncobj));


// Utility calls provided by the managed layer.
void *(*IPyCallMethod)(void *obj, const char *meth, void *arg);
void (*IPyAddToDict)(void *dictObj, const char *funcStr, void *ufuncObj);


// This structure defines all of the artimetic functions not provided by
// the core, such as those that operate on objects.
struct ExternFuncs {
    cmpfunc cmp_equal;
    cmpfunc cmp_notEqual;
    cmpfunc cmp_greater;
    cmpfunc cmp_greaterEqual;
    cmpfunc cmp_less;
    cmpfunc cmp_lessEqual;
    cmpfunc cmp_sign;
    
    // Generic arithmatic functions that operate on objects and are provided by
    // the managed layer.  These typically call into IronPython to perform the
    // operation.
    unaryfunc absolute;
    binaryfunc add, subtract, multiply, divide;
    binaryfunc trueDivide, floorDivide;
    unaryfunc invert, negative;
    binaryfunc remainder;
    unaryfunc square;
    binaryfunc power;
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
//#define npy_OBJECT_equal managedFuncs.loop_equal
//#define npy_OBJECT_not_equal managedFuncs.loop_not_equal
//#define npy_OBJECT_greater managedFuncs.loop_greater
//#define npy_OBJECT_greater_equal managedFuncs.loop_greater_equal
//#define npy_OBJECT_less managedFuncs.loop_less
//#define npy_OBJECT_less_equal managedFuncs.loop_less_equal
//#define npy_OBJECT_sign managedFuncs.loop_sign
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

static void *blank3_data[] = { NULL, NULL, NULL };

static NpyUFuncGenericFunction frexp_functions[] = {
#if NPY_HAVE_DECL_FREXPF
    npy_FLOAT_frexp,
#endif
    npy_DOUBLE_frexp
#if NPY_HAVE_DECL_FREXPL
    , npy_LONGDOUBLE_frexp
#endif
};

static char frexp_signatures[] = {
#if NPY_HAVE_DECL_FREXPF
    NPY_FLOAT, NPY_FLOAT, NPY_INT,
#endif
    NPY_DOUBLE, NPY_DOUBLE, NPY_INT
#if NPY_HAVE_DECL_FREXPL
    , NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_INT
#endif
};

static NpyUFuncGenericFunction ldexp_functions[] = {
#if NPY_HAVE_DECL_LDEXPF
    npy_FLOAT_ldexp,
#endif
    npy_DOUBLE_ldexp
#if NPY_HAVE_DECL_LDEXPL
    , npy_LONGDOUBLE_ldexp
#endif
};


static char ldexp_signatures[] = {
#if NPY_HAVE_DECL_LDEXPF
    NPY_FLOAT, NPY_INT, NPT_FLOAT,
#endif
    NPY_DOUBLE, NPY_LONG, NPY_DOUBLE
#if NPY_HAVE_DECL_LDEXPL
    , NPY_LONGDOUBLE, NPY_LONG, NPY_LONGDOUBLE
#endif
};



// Initializes the ufuncs.  
extern "C" __declspec(dllexport)
    void _cdecl NpyUFuncAccess_Init(void *dictionary, ExternFuncs *funcs,
    void *(*callMethod)(void *obj, const char *meth, void *arg),
    void (*addToDict)(void *dictObj, const char *funcStr, void *ufuncObj))
{
    // Copies the provided function pointers to our local storage.  The
    // sentinel field is used to verify that the managed structure
    // lines up with the native one.
    assert(NPY_VALID_MAGIC == funcs->sentinel);
    managedFuncs.sentinel = NPY_INVALID_MAGIC;  // Ensure it gets overwritten
    memcpy(&managedFuncs, funcs, sizeof(managedFuncs));
    assert(NPY_VALID_MAGIC == managedFuncs.sentinel);

    IPyCallMethod = callMethod;
    IPyAddToDict = addToDict;

    // Populates the dictionary with the function names and corresponding ufunc
    // instance.
    InitOperators(dictionary);
    InitOtherOperators(dictionary, addToDict);
}




static void InitOtherOperators(void *dictionary, 
    void (*addToDict)(void *dictObj, const char *funcStr, void *ufuncobj)) {
    int num=1;

#if NPY_HAVE_DECL_FREXPF
    num += 1;
#endif
#if NPY_HAVE_DECL_FREXPL
    num += 1;
#endif
    NpyUFuncObject *f = NpyUFunc_FromFuncAndData(frexp_functions, blank3_data,
        frexp_signatures, num, 1, 2, NpyUFunc_None, "frexp",
        "Split the number, x, into a normalized fraction (y1) and exponent (y2)", 0);
    addToDict(dictionary, "frexp", Npy_INTERFACE(f));
    Npy_DECREF(f);                                

    num = 1;
#if NPY_HAVE_DECL_LDEXPF
    num += 1;
#endif
#if NPY_HAVE_DECL_LDEXPL
    num += 1;
#endif
    f = NpyUFunc_FromFuncAndData(ldexp_functions, blank3_data,
        ldexp_signatures, num, 2, 1, NpyUFunc_None, "ldexp",
        "Compute y = x1 * 2**x2.", 0);
    addToDict(dictionary, "ldexp", Npy_INTERFACE(f));
    Npy_DECREF(f);
}


/******************************************************************************
 **                         GENERIC OBJECT lOOPS                             **
 *****************************************************************************/

void
npy_OBJECT_compare(char **args, npy_intp *dimensions, npy_intp *steps, cmpfunc op) {
    BINARY_LOOP {
        void *in1 = *(void **)ip1;
        void *in2 = *(void **)ip2;
        int ret = op(in1, in2);
        if (ret == -1) {
            return;
        }
        *((npy_bool *)op1) = (npy_bool)ret;
    }
}

void
npy_OBJECT_equal(char **args, npy_intp *dimensions, npy_intp *steps, void *NPY_UNUSED(func)) {
    npy_OBJECT_compare(args, dimensions, steps, managedFuncs.cmp_equal);
}

void
npy_OBJECT_not_equal(char **args, npy_intp *dimensions, npy_intp *steps, void *NPY_UNUSED(func)) {
    npy_OBJECT_compare(args, dimensions, steps, managedFuncs.cmp_notEqual);
}

void
npy_OBJECT_greater(char **args, npy_intp *dimensions, npy_intp *steps, void *NPY_UNUSED(func)) {
    npy_OBJECT_compare(args, dimensions, steps, managedFuncs.cmp_greater);
}

void
npy_OBJECT_greater_equal(char **args, npy_intp *dimensions, npy_intp *steps, void *NPY_UNUSED(func)) {
    npy_OBJECT_compare(args, dimensions, steps, managedFuncs.cmp_greaterEqual);
}

void
npy_OBJECT_less(char **args, npy_intp *dimensions, npy_intp *steps, void *NPY_UNUSED(func)) {
    npy_OBJECT_compare(args, dimensions, steps, managedFuncs.cmp_less);
}

void
npy_OBJECT_less_equal(char **args, npy_intp *dimensions, npy_intp *steps, void *NPY_UNUSED(func)) {
    npy_OBJECT_compare(args, dimensions, steps, managedFuncs.cmp_lessEqual);
}



void
npy_OBJECT_sign(char **args, npy_intp *dimensions, npy_intp *steps, void *NPY_UNUSED(func))
{
#if defined(NPY_PY3K)
    PyObject *zero = PyLong_FromLong(0);
    UNARY_LOOP {
        PyObject *in1 = *(PyObject **)ip1;
        PyObject **out = (PyObject **)op1;
        int v;
        PyObject *ret;
        PyObject_Cmp(in1, zero, &v);
        ret = PyLong_FromLong(v);
        if (PyErr_Occurred()) {
            return;
        }
        Py_XDECREF(*out);
        *out = ret;
    }
    Py_DECREF(zero);
#elif 0
    PyObject *zero = PyInt_FromLong(0);
    UNARY_LOOP {
        PyObject *in1 = *(PyObject **)ip1;
        PyObject **out = (PyObject **)op1;
        PyObject *ret = PyInt_FromLong(PyObject_Compare(in1, zero));
        if (PyErr_Occurred()) {
            return;
        }
        Py_XDECREF(*out);
        *out = ret;
    }
    Py_DECREF(zero);
#endif
}


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

#include "__umath_generated.c"

#undef AddFunction
