/*
 *  npy_descriptor.c - 
 *  
 */

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include "npy_config.h"
#include "numpy/numpy_api.h"
#include "numpy/npy_object.h"


#if !defined(MAX)
#define MAX(x,y) (((x)>=(y)) ? (x) : (y))
#endif


/* Local functions */
static NpyDict *npy_create_fields_table(void);
static void *npy_copy_fields_key(void *key);
static void npy_dealloc_fields_key(void *key);
static void *npy_copy_fields_value(void *value);
static void npy_dealloc_fields_value(void *value);

void 
NpyArray_DescrDestroy(NpyArray_Descr *self);


_NpyTypeObject NpyArrayDescr_Type = {
    (npy_destructor)NpyArray_DescrDestroy,
};




/*NUMPY_API*/
NpyArray_Descr *
NpyArray_DescrNewFromType(int type_num)
{
    NpyArray_Descr *old;
    NpyArray_Descr *new;
    
    old = NpyArray_DescrFromType(type_num);
    new = NpyArray_DescrNew(old);
    _Npy_DECREF(old);
    return new;
}



/** Array Descr Objects for dynamic types **/

/*
 * There are some statically-defined NpyArray_Descr objects corresponding
 * to the basic built-in types.
 * These can and should be DECREF'd and INCREF'd as appropriate, anyway.
 * If a mistake is made in reference counting, deallocation on these
 * builtins will be attempted leading to problems.
 *
 * This let's us deal with all NpyArray_Descr objects using reference
 * counting (regardless of whether they are statically or dynamically
 * allocated).
 */

/*NUMPY_API
 * base cannot be NULL
 */
NpyArray_Descr *
NpyArray_DescrNew(NpyArray_Descr *base)
{
    NpyArray_Descr *new;
    
    assert(NULL != base && NPY_VALID_MAGIC == base->magic_number);

    new = (NpyArray_Descr *)malloc(sizeof(NpyArray_Descr));
    if (new == NULL) {
        return NULL;
    }
    _NpyObject_Init(new, &NpyArrayDescr_Type);
    new->magic_number = NPY_VALID_MAGIC;
    
    /* Don't copy NpyObject_HEAD part */
    memcpy((char *)new + sizeof(struct _NpyObject),
           (char *)base + sizeof(struct _NpyObject),
           sizeof(NpyArray_Descr) - sizeof(struct _NpyObject));
    
    assert((NULL == new->fields && NULL == new->names) || (NULL != new->fields && NULL != new->names));
    if (NULL != new->fields) {
        new->names = NpyArray_DescrNamesCopy(new->names);
        new->fields = NpyDict_Copy(new->fields, npy_copy_fields_key, npy_copy_fields_value);
    }
    if (new->subarray) {
        new->subarray = NpyArray_DupSubarray(base->subarray);
    }

    if (new->dtinfo) {
        new->dtinfo = NpyArray_malloc(sizeof(NpyArray_DateTimeInfo));
        memcpy(new->dtinfo, base->dtinfo, sizeof(NpyArray_DateTimeInfo));
    }

    /* Allocate the interface wrapper object. */
    if (NPY_FALSE == NpyInterface_DescrNewFromWrapper(Npy_INTERFACE(base), new, &new->nob_interface)) {
        Npy_INTERFACE(new) = NULL;
        _Npy_DECREF(new);
        return NULL;
    }
    
    /* Note on reference counts: At this point if there is an inteface object, it's refcnt should
       be == 1 because the refcnt on the core object == 1. That is, the core object is holding a
       single reference to the interface object. */
    return new;
}



/*
 * new reference
 * doesn't alter refcount of chktype or mintype ---
 * unless one of them is returned
 */
NPY_NO_EXPORT NpyArray_Descr *
NpyArray_SmallType(NpyArray_Descr *chktype, NpyArray_Descr *mintype)
{
    NpyArray_Descr *outtype;
    int outtype_num, save_num;
    
    assert(NULL != chktype && NULL != mintype &&
           NPY_VALID_MAGIC == chktype->magic_number && 
           NPY_VALID_MAGIC == mintype->magic_number);

    if (NpyArray_EquivTypes(chktype, mintype)) {
        _Npy_INCREF(mintype);
        return mintype;
    }
    
    
    if (chktype->type_num > mintype->type_num) {
        outtype_num = chktype->type_num;
    }
    else {
        if (NpyTypeNum_ISOBJECT(chktype->type_num) &&
            NpyDataType_ISSTRING(mintype)) {
            return NpyArray_DescrFromType(NPY_OBJECT);
        }
        else {
            outtype_num = mintype->type_num;
        }
    }
    
    save_num = outtype_num;
    while (outtype_num < NPY_NTYPES &&
           !(NpyArray_CanCastSafely(chktype->type_num, outtype_num)
             && NpyArray_CanCastSafely(mintype->type_num, outtype_num))) {
               outtype_num++;
           }
    if (outtype_num == NPY_NTYPES) {
        outtype = NpyArray_DescrFromType(save_num);
    }
    else {
        outtype = NpyArray_DescrFromType(outtype_num);
    }
    if (NpyTypeNum_ISEXTENDED(outtype->type_num)) {
        int testsize = outtype->elsize;
        int chksize, minsize;
        chksize = chktype->elsize;
        minsize = mintype->elsize;
        /*
         * Handle string->unicode case separately
         * because string itemsize is 4* as large
         */
        if (outtype->type_num == NPY_UNICODE &&
            mintype->type_num == NPY_STRING) {
            testsize = MAX(chksize, 4*minsize);
        }
        else if (chktype->type_num == NPY_STRING &&
                 mintype->type_num == NPY_UNICODE) {
            testsize = MAX(chksize*4, minsize);
        }
        else {
            testsize = MAX(chksize, minsize);
        }
        if (testsize != outtype->elsize) {
            NpyArray_DESCR_REPLACE(outtype);
            outtype->elsize = testsize;
            NpyArray_DescrDeallocNamesAndFields(outtype);
        }
    }
    return outtype;
}



/*
 * op is an object to be converted to an ndarray.
 *
 * minitype is the minimum type-descriptor needed.
 *
 * returns new reference
 */
NpyArray_Descr *
NpyArray_DescrFromArray(NpyArray *ap, NpyArray_Descr *mintype)
{
    NpyArray_Descr *chktype = NULL;
    NpyArray_Descr *outtype = NULL;
    
    assert(NULL != ap && 
           NPY_VALID_MAGIC == ap->magic_number && 
           (NULL == mintype || NPY_VALID_MAGIC == mintype->magic_number));
    
    chktype = NpyArray_DESCR(ap);
    _Npy_INCREF(chktype);
    if (mintype == NULL) {
        return chktype;
    }
    _Npy_INCREF(mintype);
    
    outtype = NpyArray_SmallType(chktype, mintype);
    _Npy_DECREF(chktype);
    _Npy_DECREF(mintype);
    
    /*
     * VOID Arrays should not occur by "default"
     * unless input was already a VOID
     */
    if (outtype->type_num == NPY_VOID &&
        mintype->type_num != NPY_VOID) {
        _Npy_DECREF(outtype);
        outtype = NpyArray_DescrFromType(PyArray_OBJECT);
    }
    return outtype;
}




/* Duplicates an array descr structure and the sub-structures. */
NpyArray_ArrayDescr *
NpyArray_DupSubarray(NpyArray_ArrayDescr *src)
{
    NpyArray_ArrayDescr *dest = (NpyArray_ArrayDescr *)NpyArray_malloc(sizeof(NpyArray_ArrayDescr));
    assert((0 == src->shape_num_dims && NULL == src->shape_dims) || (0 < src->shape_num_dims && NULL != src->shape_dims));
    
    dest->base = src->base;
    _Npy_INCREF(dest->base);
    
    dest->shape_num_dims = src->shape_num_dims;
    if (0 < dest->shape_num_dims) {
        dest->shape_dims = (npy_intp *)NpyArray_malloc(dest->shape_num_dims * sizeof(npy_intp *));
        memcpy(dest->shape_dims, src->shape_dims, dest->shape_num_dims * sizeof(npy_intp *));
    } else {
        dest->shape_dims = NULL;
    }
    return dest;
}

/*NUMPY_API
 * self cannot be NULL
 * Destroys the given descriptor and deallocates the memory for it.
 */
void 
NpyArray_DescrDestroy(NpyArray_Descr *self)
{
    assert(NPY_VALID_MAGIC == self->magic_number);

    NpyArray_DescrDeallocNamesAndFields(self);
    
    if (self->subarray) {
        NpyArray_DestroySubarray(self->subarray);
        self->subarray = NULL;
    }
    if (self->dtinfo) {
        NpyArray_free(self->dtinfo);
    }

    self->magic_number = NPY_INVALID_MAGIC;
    
    free(self);
}


void 
NpyArray_DestroySubarray(NpyArray_ArrayDescr *self)
{
    _Npy_DECREF(self->base);
    if (0 < self->shape_num_dims) {
        NpyArray_free(self->shape_dims);
    }
    self->shape_dims = NULL;
    NpyArray_free(self);
}


/*NUMPY_API
 *
 * returns a copy of the NpyArray_Descr structure with the byteorder
 * altered:
 * no arguments:  The byteorder is swapped (in all subfields as well)
 * single argument:  The byteorder is forced to the given state
 * (in all subfields as well)
 *
 * Valid states:  ('big', '>') or ('little' or '<')
 * ('native', or '=')
 *
 * If a descr structure with | is encountered it's own
 * byte-order is not changed but any fields are:
 *
 *
 * Deep bytorder change of a data-type descriptor
 * *** Leaves reference count of self unchanged --- does not DECREF self ***
 */
NpyArray_Descr *
NpyArray_DescrNewByteorder(NpyArray_Descr *self, char newendian)
{
    NpyArray_Descr *new;
    char endian;
    
    new = NpyArray_DescrNew(self);
    endian = new->byteorder;
    if (endian != NPY_IGNORE) {
        if (newendian == NPY_SWAP) {
            /* swap byteorder */
            if NpyArray_ISNBO(endian) {
                endian = NPY_OPPBYTE;
            }
            else {
                endian = NPY_NATBYTE;
            }
            new->byteorder = endian;
        }
        else if (newendian != NPY_IGNORE) {
            new->byteorder = newendian;
        }
    }
    if (NULL != new->names) {
        const char *key;
        NpyArray_Descr *newdescr;
        NpyArray_DescrField *value;
        NpyDict_Iter pos;
        
        NpyDict_IterInit(&pos);
        while (NpyDict_IterNext(new->fields, &pos, (void **)&key, (void **)&value)) {
            if (NULL != value->title && !strcmp(key, value->title)) {
                continue;
            }
            newdescr = NpyArray_DescrNewByteorder(value->descr, newendian);
            if (newdescr == NULL) {
                _Npy_DECREF(new);
                return NULL;
            }
            _Npy_DECREF(value->descr);
            value->descr = newdescr;
        }
    }
    if (NULL != new->subarray) {
        NpyArray_Descr *old = new->subarray->base;
        new->subarray->base = NpyArray_DescrNewByteorder(self->subarray->base, newendian);
        _Npy_DECREF(old);
    }
    return new;
}


/*NUMPY_API
 * Allocates a new names array. Since this array is NULL-terminated, this function is just
 * a bit safer because it makes sure size is n+1 and zeros all of memory.
 */
char **
NpyArray_DescrAllocNames(int n)
{
    char **nameslist = (char **)malloc((n+1) * sizeof(char *));
    if (NULL != nameslist) {
        memset(nameslist, 0, (n+1)*sizeof(char *));
    }
    return nameslist;
}



/*NUMPY_API
 * base cannot be NULL
 * Allocates a new fields dictionary and returns it.
 */
NpyDict *
NpyArray_DescrAllocFields()
{
    return npy_create_fields_table();
}



/*NUMPY_API
 * self cannot be NULL
 */
void
NpyArray_DescrDeallocNamesAndFields(NpyArray_Descr *self)
{
    int i;

    if (NULL != self->names) {
        for (i=0; NULL != self->names[i]; i++) {
            free(self->names[i]);
            self->names[i] = NULL;
        }
        free(self->names);
        self->names = NULL;
    }
    
    if (NULL != self->fields) {
        NpyDict_Destroy(self->fields);        
        self->fields = NULL;
    }
}

/*NUMPY_API
 * self cannot be NULL
 * Replaces the existing list of names with a new set of names.  This also re-keys the
 * fields dictionary.  The passed in nameslist must be exactly the same length as the
 * existing names array.  All memory in nameslist, including the string points, will
 * be managed by the NpyArray_Descr instance as it's own.
 *
 * Returns: 1 on success, 0 on error.
 */
int 
NpyArray_DescrReplaceNames(NpyArray_Descr *self, char **nameslist)
{
    int i, n;
    
    for (n = 0; NULL != nameslist[n] && NULL != self->names[n]; n++) ;
    if (NULL != nameslist[n] || NULL != self->names[n]) {
        return 0;
    }
    
    for (i = 0; i < n; i++) {
        NpyDict_Rekey(self->fields, self->names[i], strdup(nameslist[i]));
        free(self->names[i]);
    }
    free(self->names);
    
    self->names = nameslist;
    return 1;
}




/*NUMPY_API
 * self cannot be NULL
 * Sets the existing list of names.  The fields are not change or checked -- it
 * is assumed that the caller will update the fields as appropropiate.
 */
void 
NpyArray_DescrSetNames(NpyArray_Descr *self, char **nameslist)
{
    int i;

    if (NULL != self->names) {
        for (i = 0; NULL != self->names[i]; i++) {
            free(self->names[i]);
        }
        free(self->names);
    }    
    self->names = nameslist;
}


/*NUMPY_API
 * base cannot be NULL
 * Adds the tripple of { descriptor, offset, [title] } to the given field dictionary.
 *
 * NOTE: This routine steals a reference to descr!
 */
void
NpyArray_DescrSetField(NpyDict *self, const char *key, NpyArray_Descr *descr,
                       int offset, const char *title)
{
    NpyArray_DescrField *field;
    
    field = (NpyArray_DescrField *)malloc(sizeof(NpyArray_DescrField));
    field->descr = descr;
    field->offset = offset;
    field->title = (NULL == title) ? NULL : strdup(title);
    
    NpyDict_Put(self, strdup(key), field);
}




/*NUMPY_API
 * base cannot be NULL
 * Duplicates the fields dictionary usinga deep-copy.  The returned version is
 * completely independent of the original */
NpyDict *
NpyArray_DescrFieldsCopy(NpyDict *fields)
{
    return NpyDict_Copy(fields, npy_copy_fields_key, npy_copy_fields_value);
}



/*NUMPY_API
 * base cannot be NULL
 * Allocates a new NpyDict contaning NpyArray_DescrField value object and performs a 
 * deep-copy of the passed-in fields structure to populate them. The descriptor field 
 * structure contains a pointer to another NpyArray_Descr instance, which must be 
 * reference counted. */
char **
NpyArray_DescrNamesCopy(char **names)
{
    char **copy = NULL;
    int n, i;
    
    if (NULL != names) {
        for (n=0; NULL != names[n]; n++) ;
        
        copy = malloc((n+1) * sizeof(char *));
        for (i=0; i < n; i++) {
            copy[i] = strdup(names[i]);
        }
        copy[n] = NULL;
    }
    return copy;
}


static NpyDict *npy_create_fields_table()
{
    NpyDict *new = NpyDict_CreateTable(7);  /* TODO: Should be price, 7 is a guess at size that avoids rehashing most of the time. */
    NpyDict_SetKeyComparisonFunction(new, (int (*)(const void *, const void *))strcmp);
    NpyDict_SetHashFunction(new, NpyDict_StringHashFunction);
    NpyDict_SetDeallocationFunctions(new, npy_dealloc_fields_key, npy_dealloc_fields_value);
    return new;
}


/* Keys are c-strings */
static void *npy_copy_fields_key(void *key)
{
    return strdup((const char *)key);
}

static void npy_dealloc_fields_key(void *key)
{
    free(key);
}





/* Values are NpyArray_DescrField structures. */
static void *npy_copy_fields_value(void *value_tmp)
{
    NpyArray_DescrField *value = (NpyArray_DescrField *)value_tmp;
    NpyArray_DescrField *copy = (NpyArray_DescrField *)malloc(sizeof(NpyArray_DescrField));
    
    copy->descr = value->descr;
    _Npy_XINCREF(copy->descr);
    copy->offset = value->offset;
    copy->title = (NULL == value->title) ? NULL : strdup(value->title);
    
    return copy;
}

static void npy_dealloc_fields_value(void *value_tmp)
{
    NpyArray_DescrField *value = (NpyArray_DescrField *)value_tmp;

    _Npy_XDECREF(value->descr);
    if (NULL != value->title) free(value->title);
    free(value);
}

