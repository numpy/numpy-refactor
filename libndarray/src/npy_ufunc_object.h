
#if !defined(NPY_UFUNC_OBJECT_H)
#define NPY_UFUNC_OBJECT_H

#include "npy_object.h"


typedef void (*NpyUFuncGenericFunction) (char **, npy_intp *, npy_intp *, void *);


struct NpyDict_struct;

struct NpyUFuncObject {
    NpyObject_HEAD
    int magic_number;

    int nin, nout, nargs;
    int identity;
    NpyUFuncGenericFunction *functions;
    void **data;
    int ntypes;
    int check_return;
    char *name, *types;
    char *doc;
    void *ptr;
    struct NpyDict_struct *userloops;

    /* generalized ufunc */
    int core_enabled;      /* 0 for scalar ufunc; 1 for generalized ufunc */
    int core_num_dim_ix;   /* number of distinct dimension names in
                            signature */

    /* dimension indices of input/output argument k are stored in
     core_dim_ixs[core_offsets[k]..core_offsets[k]+core_num_dims[k]-1] */
    int *core_num_dims;    /* numbers of core dimensions of each argument */
    int *core_dim_ixs;     /* dimension indices in a flatted form; indices
                            are in the range of [0,core_num_dim_ix) */
    int *core_offsets;     /* positions of 1st core dimensions of each
                            argument in core_dim_ixs */
    char *core_signature;  /* signature string for printing purpose */
};

typedef struct NpyUFuncObject NpyUFuncObject;

extern NpyTypeObject NpyUFunc_Type;


extern struct NpyDict_struct *npy_create_userloops_table();



enum NpyArray_Ops {
    npy_op_add,
    npy_op_subtract,
    npy_op_multiply,
    npy_op_divide,
    npy_op_remainder,
    npy_op_power,
    npy_op_square,
    npy_op_reciprocal,
    npy_op_ones_like,
    npy_op_sqrt,
    npy_op_negative,
    npy_op_absolute,
    npy_op_invert,
    npy_op_left_shift,
    npy_op_right_shift,
    npy_op_bitwise_and,
    npy_op_bitwise_xor,
    npy_op_bitwise_or,
    npy_op_less,
    npy_op_less_equal,
    npy_op_equal,
    npy_op_not_equal,
    npy_op_greater,
    npy_op_greater_equal,
    npy_op_floor_divide,
    npy_op_true_divide,
    npy_op_logical_or,
    npy_op_logical_and,
    npy_op_floor,
    npy_op_ceil,
    npy_op_maximum,
    npy_op_minimum,
    npy_op_rint,
    npy_op_conjugate
};

NpyUFuncObject *NpyArray_GetNumericOp(enum NpyArray_Ops);
int NpyArray_SetNumericOp(enum NpyArray_Ops, NpyUFuncObject *);
NpyUFuncObject *
    NpyUFunc_FromFuncAndData(NpyUFuncGenericFunction *func, void **data,
                             char *types, int ntypes,
                             int nin, int nout, int identity,
                             char *name, char *doc, int check_return);

#endif
