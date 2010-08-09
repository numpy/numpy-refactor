
#if !defined(NPY_UFUNC_OBJECT_H)
#define NPY_UFUNC_OBJECT_H

#include <numpy/npy_object.h>


typedef void (*PyUFuncGenericFunction) (char **, npy_intp *, npy_intp *, void *);


struct NpyDict_struct;

struct NpyUFuncObject {
    NpyObject_HEAD
    int magic_number;
    
    int nin, nout, nargs;
    int identity;
    PyUFuncGenericFunction *functions;
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

extern _NpyTypeObject NpyUFunc_Type;


extern struct NpyDict_struct *npy_create_userloops_table();


#endif
