/*
 * Python Universal Functions Object -- CPython-independent portion
 *
 */
#include "npy_config.h"

#include "noprefix.h"
#include "npy_dict.h"
#include "ufuncobject.h"  // TODO: Fix this
#include "numpy_api.h"
#include "npy_iterators.h"

#include "ufunc_object.h"       // TODO: Fix this



/*
 * Userloops dictionary implementation
 */

static int npy_compare_ints(const void *a, const void *b)
{
    if (a < b ) return -1;
    else if ( a > b ) return 1;
    return 0;
}

static int npy_hash_int(const void *a)
{
    return (int)a;          /* Size change is safe - just a hash function */
}

/*
 * This frees the linked-list structure when the CObject
 * is destroyed (removed from the internal dictionary)
 */
static void npy_free_loop1d_list(PyUFunc_Loop1d *data)
{
    while (data != NULL) {
        PyUFunc_Loop1d *next = data->next;
        NpyArray_free(data->arg_types);
        NpyArray_free(data);
        data = next;
    }
}


NpyDict *npy_create_userloops_table()
{
    NpyDict *new = NpyDict_CreateTable(7);  /* 7 is a guess at enough */
    NpyDict_SetKeyComparisonFunction(new, (int (*)(const void *, const void *))npy_compare_ints);
    NpyDict_SetHashFunction(new, (int (*)(const void *))npy_hash_int);
    NpyDict_SetDeallocationFunctions(new, NULL, (void (*)(void *))npy_free_loop1d_list); 
    return new;
}
