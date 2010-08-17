/*
 * Python Universal Functions Object -- CPython-independent portion
 *
 */
#include <stdlib.h>
#include <memory.h>
#include "npy_config.h"

#include "npy_dict.h"
#include "npy_api.h"
#include "npy_arrayobject.h"
#include "npy_iterators.h"
#include "npy_ufunc_object.h"


/* 
 * Forward decls 
 */
static NpyUFuncLoopObject *
construct_loop(NpyUFuncObject *self);
static size_t
construct_arrays(NpyUFuncLoopObject *loop, size_t nargs, NpyArray **mps,
                 int *rtypenums, npy_prepare_outputs_func prepare, void *prepare_data);
static int
select_types(NpyUFuncObject *self, int *arg_types,
             NpyUFuncGenericFunction *function, void **data,
             NPY_SCALARKIND *scalars,
             int *rtypenums);
static int
extract_specified_loop(NpyUFuncObject *self, int *arg_types,
                       NpyUFuncGenericFunction *function, void **data,
                       int *rtypenums, int userdef);
static void
ufuncloop_dealloc(NpyUFuncLoopObject *self);
static int
_create_copies(NpyUFuncLoopObject *loop, int *arg_types, NpyArray **mps);
static int
_compute_dimension_size(NpyUFuncLoopObject *loop, NpyArray **mps, int i);
static npy_intp*
_compute_output_dims(NpyUFuncLoopObject *loop, int iarg,
                     int *out_nd, npy_intp *tmp_dims);
static NpyArray *
_trunc_coredim(NpyArray *ap, int core_nd);
static int
_find_matching_userloop(NpyUFunc_Loop1d *funcdata, int *arg_types,
                        NPY_SCALARKIND *scalars,
                        NpyUFuncGenericFunction *function, void **data,
                        int nargs, int nin);
static int
_does_loop_use_arrays(void *data);
static char _lowest_type(char intype);



/* Global floating-point error handling.  This is set by the interface layer or, if NULL
   defaults to a trival internal one. */
static void default_fp_error_handler(int errormask, void *errobj, int retstatus, int *first)
{
    const char *msg = "unknown";
    
    switch (errormask) {
        case UFUNC_FPE_DIVIDEBYZERO:
            msg = "division by zero"; break;
        case UFUNC_FPE_OVERFLOW:
            msg = "overflow"; break;
        case UFUNC_FPE_UNDERFLOW:
            msg = "underflow"; break;
        case UFUNC_FPE_INVALID:
            msg = "invalid"; break;
    }
    printf("libndarray floating point %s warning.", msg);
}

static void (*fp_error_handler)(int, void *, int, int*) = &default_fp_error_handler;






int NpyUFunc_GenericFunction(NpyUFuncObject *self, int nargs, NpyArray **mps,
                             int *rtypenums,
                             int bufsize, int errormask, void *errobj, 
                             int originalArgWasObjArray, npy_prepare_outputs_func prepare_outputs,
                             void *prepare_out_args)
{
    NpyUFuncLoopObject *loop;
    int res;
    int i;
    
    
    /* Build the loop. */
    loop = construct_loop(self);
    if (loop == NULL) {
        return -1;
    }
    loop->bufsize = bufsize;
    loop->errormask = errormask;
    loop->errobj = errobj;

    /* Setup the arrays */
    res = construct_arrays(loop, nargs, mps, rtypenums, 
                           prepare_outputs, prepare_out_args);
    
    if (res < 0) {
        ufuncloop_dealloc(loop);
        return -1;
    }

    /*
     * FAIL with NotImplemented if the other object has
     * the __r<op>__ method and has __array_priority__ as
     * an attribute (signalling it can handle ndarray's)
     * and is not already an ndarray or a subtype of the same type.
     */
    if (self->nin == 2 && self->nout == 1 &&
        NpyArray_TYPE(mps[1]) == NPY_OBJECT && originalArgWasObjArray) {
        /* Return -2 for notimplemented. */
        ufuncloop_dealloc(loop);
        return -2;
    }

    if (loop->notimplemented) {
        ufuncloop_dealloc(loop);
        return -2;
    }
    if (self->core_enabled && loop->meth != SIGNATURE_NOBUFFER_UFUNCLOOP) {
        NpyErr_SetString(NpyExc_RuntimeError,
                         "illegal loop method for ufunc with signature");
        goto fail;
    }
    
    //    NPY_LOOP_BEGIN_THREADS;
    switch(loop->meth) {
        case ONE_UFUNCLOOP:
            /*
             * Everything is contiguous, notswapped, aligned,
             * and of the right type.  -- Fastest.
             * Or if not contiguous, then a single-stride
             * increment moves through the entire array.
             */
            /*fprintf(stderr, "ONE...%d\n", loop->size);*/
            loop->function((char **)loop->bufptr, &(loop->iter->size),
                           loop->steps, loop->funcdata);
            UFUNC_CHECK_ERROR(loop);
            break;
        case NOBUFFER_UFUNCLOOP:
            /*
             * Everything is notswapped, aligned and of the
             * right type but not contiguous. -- Almost as fast.
             */
            while (loop->iter->index < loop->iter->size) {
                for (i = 0; i < self->nargs; i++) {
                    loop->bufptr[i] = loop->iter->iters[i]->dataptr;
                }
                loop->function((char **)loop->bufptr, &(loop->bufcnt),
                               loop->steps, loop->funcdata);
                UFUNC_CHECK_ERROR(loop);

                /* Adjust loop pointers */
                for (i = 0; i < self->nargs; i++) {
                    NpyArray_ITER_NEXT(loop->iter->iters[i]);
                }
                loop->iter->index++;
            }
            break;
        case SIGNATURE_NOBUFFER_UFUNCLOOP:
            while (loop->iter->index < loop->iter->size) {
                for (i = 0; i < self->nargs; i++) {
                    loop->bufptr[i] = loop->iter->iters[i]->dataptr;
                }
                loop->function((char **)loop->bufptr, loop->core_dim_sizes,
                               loop->core_strides, loop->funcdata);
                UFUNC_CHECK_ERROR(loop);

                /* Adjust loop pointers */
                for (i = 0; i < self->nargs; i++) {
                    NpyArray_ITER_NEXT(loop->iter->iters[i]);
                }
                loop->iter->index++;
            }
            break;
        case BUFFER_UFUNCLOOP: {
            /* This should be a function */
            NpyArray_CopySwapNFunc *copyswapn[NPY_MAXARGS];
            NpyArrayIterObject **iters=loop->iter->iters;
            int *swap=loop->swap;
            char **dptr=loop->dptr;
            int mpselsize[NPY_MAXARGS];
            npy_intp laststrides[NPY_MAXARGS];
            int fastmemcpy[NPY_MAXARGS];
            int *needbuffer = loop->needbuffer;
            npy_intp index=loop->iter->index, size=loop->iter->size;
            int bufsize;
            npy_intp bufcnt;
            int copysizes[NPY_MAXARGS];
            char **bufptr = loop->bufptr;
            char **buffer = loop->buffer;
            char **castbuf = loop->castbuf;
            npy_intp *steps = loop->steps;
            char *tptr[NPY_MAXARGS];
            int ninnerloops = loop->ninnerloops;
            npy_bool pyobject[NPY_MAXARGS];
            int datasize[NPY_MAXARGS];
            int j, k, stopcondition;
            char *myptr1, *myptr2;

            for (i = 0; i <self->nargs; i++) {
                copyswapn[i] = NpyArray_DESCR(mps[i])->f->copyswapn;
                mpselsize[i] = NpyArray_DESCR(mps[i])->elsize;
                pyobject[i] = ((loop->obj & UFUNC_OBJ_ISOBJECT)
                               && (NpyArray_TYPE(mps[i]) == NPY_OBJECT));
                laststrides[i] = iters[i]->strides[loop->lastdim];
                if (steps[i] && laststrides[i] != mpselsize[i]) {
                    fastmemcpy[i] = 0;
                }
                else {
                    fastmemcpy[i] = 1;
                }
            }
            /* Do generic buffered looping here (works for any kind of
             * arrays -- some need buffers, some don't.
             *
             *
             * New algorithm: N is the largest dimension.  B is the buffer-size.
             * quotient is loop->ninnerloops-1
             * remainder is loop->leftover
             *
             * Compute N = quotient * B + remainder.
             * quotient = N / B  # integer math
             * (store quotient + 1) as the number of innerloops
             * remainder = N % B # integer remainder
             *
             * On the inner-dimension we will have (quotient + 1) loops where
             * the size of the inner function is B for all but the last when the niter size is
             * remainder.
             *
             * So, the code looks very similar to NOBUFFER_LOOP except the inner-most loop is
             * replaced with...
             *
             * for(i=0; i<quotient+1; i++) {
             * if (i==quotient+1) make itersize remainder size
             * copy only needed items to buffer.
             * swap input buffers if needed
             * cast input buffers if needed
             * call loop_function()
             * cast outputs in buffers if needed
             * swap outputs in buffers if needed
             * copy only needed items back to output arrays.
             * update all data-pointers by strides*niter
             * }
             */


            /*
             * fprintf(stderr, "BUFFER...%d,%d,%d\n", loop->size,
             * loop->ninnerloops, loop->leftover);
             */
            /*
             * for(i=0; i<self->nargs; i++) {
             * fprintf(stderr, "iters[%d]->dataptr = %p, %p of size %d\n", i,
             * iters[i], iters[i]->ao->data, PyArray_NBYTES(iters[i]->ao));
             * }
             */
            stopcondition = ninnerloops;
            if (loop->leftover == 0) {
                stopcondition--;
            }
            while (index < size) {
                bufsize=loop->bufsize;
                for(i = 0; i<self->nargs; i++) {
                    tptr[i] = loop->iter->iters[i]->dataptr;
                    if (needbuffer[i]) {
                        dptr[i] = bufptr[i];
                        datasize[i] = (steps[i] ? bufsize : 1);
                        copysizes[i] = datasize[i] * mpselsize[i];
                    }
                    else {
                        dptr[i] = tptr[i];
                    }
                }

                /* This is the inner function over the last dimension */
                for (k = 1; k<=stopcondition; k++) {
                    if (k == ninnerloops) {
                        bufsize = loop->leftover;
                        for (i=0; i<self->nargs;i++) {
                            if (!needbuffer[i]) {
                                continue;
                            }
                            datasize[i] = (steps[i] ? bufsize : 1);
                            copysizes[i] = datasize[i] * mpselsize[i];
                        }
                    }
                    for (i = 0; i < self->nin; i++) {
                        if (!needbuffer[i]) {
                            continue;
                        }
                        if (fastmemcpy[i]) {
                            memcpy(buffer[i], tptr[i], copysizes[i]);
                        }
                        else {
                            myptr1 = buffer[i];
                            myptr2 = tptr[i];
                            for (j = 0; j < bufsize; j++) {
                                memcpy(myptr1, myptr2, mpselsize[i]);
                                myptr1 += mpselsize[i];
                                myptr2 += laststrides[i];
                            }
                        }

                        /* swap the buffer if necessary */
                        if (swap[i]) {
                            /* fprintf(stderr, "swapping...\n");*/
                            copyswapn[i](buffer[i], mpselsize[i], NULL, -1,
                                         (npy_intp) datasize[i], 1,
                                         mps[i]);
                        }
                        /* cast to the other buffer if necessary */
                        if (loop->cast[i]) {
                            /* fprintf(stderr, "casting... %d, %p %p\n", i, buffer[i]); */
                            loop->cast[i](buffer[i], castbuf[i],
                                          (npy_intp) datasize[i],
                                          NULL, NULL);
                        }
                    }

                    bufcnt = (npy_intp) bufsize;
                    loop->function((char **)dptr, &bufcnt, steps, loop->funcdata);
                    UFUNC_CHECK_ERROR(loop);

                    for (i = self->nin; i < self->nargs; i++) {
                        if (!needbuffer[i]) {
                            continue;
                        }
                        if (loop->cast[i]) {
                            /* fprintf(stderr, "casting back... %d, %p", i, castbuf[i]); */
                            loop->cast[i](castbuf[i],
                                          buffer[i],
                                          (npy_intp) datasize[i],
                                          NULL, NULL);
                        }
                        if (swap[i]) {
                            copyswapn[i](buffer[i], mpselsize[i], NULL, -1,
                                         (npy_intp) datasize[i], 1,
                                         mps[i]);
                        }
                        /*
                         * copy back to output arrays
                         * decref what's already there for object arrays
                         */
                        if (pyobject[i]) {
                            myptr1 = tptr[i];
                            for (j = 0; j < datasize[i]; j++) {
                                NpyInterface_DECREF(*((void **)myptr1));
                                myptr1 += laststrides[i];
                            }
                        }
                        if (fastmemcpy[i]) {
                            memcpy(tptr[i], buffer[i], copysizes[i]);
                        }
                        else {
                            myptr2 = buffer[i];
                            myptr1 = tptr[i];
                            for (j = 0; j < bufsize; j++) {
                                memcpy(myptr1, myptr2, mpselsize[i]);
                                myptr1 += laststrides[i];
                                myptr2 += mpselsize[i];
                            }
                        }
                    }
                    if (k == stopcondition) {
                        continue;
                    }
                    for (i = 0; i < self->nargs; i++) {
                        tptr[i] += bufsize * laststrides[i];
                        if (!needbuffer[i]) {
                            dptr[i] = tptr[i];
                        }
                    }
                }
                /* end inner function over last dimension */

                if (loop->objfunc) {
                    /*
                     * DECREF castbuf when underlying function used
                     * object arrays and casting was needed to get
                     * to object arrays
                     */
                    for (i = 0; i < self->nargs; i++) {
                        if (loop->cast[i]) {
                            if (steps[i] == 0) {
                                NpyInterface_DECREF(*((void **)castbuf[i]));
                            }
                            else {
                                int size = loop->bufsize;

                                void **objptr = (void **)castbuf[i];
                                /*
                                 * size is loop->bufsize unless there
                                 * was only one loop
                                 */
                                if (ninnerloops == 1) {
                                    size = loop->leftover;
                                }
                                for (j = 0; j < size; j++) {
                                    NpyInterface_DECREF(*objptr);
                                    *objptr = NULL;
                                    objptr += 1;
                                }
                            }
                        }
                    }
                    /* Prevent doing the decref twice on an error. */
                    loop->objfunc = 0;
                }
                /* fixme -- probably not needed here*/
                UFUNC_CHECK_ERROR(loop);

                for (i = 0; i < self->nargs; i++) {
                    NpyArray_ITER_NEXT(loop->iter->iters[i]);
                }
                index++;
            }
        } /* end of last case statement */
    }
    
    //    NPY_LOOP_END_THREADS;
    ufuncloop_dealloc(loop);
    return 0;

fail:
    //    NPY_LOOP_END_THREADS;
    if (loop) {
        if (loop->objfunc) {
            char **castbuf = loop->castbuf;
            npy_intp *steps = loop->steps;
            int ninnerloops = loop->ninnerloops;
            int j;

            /*
             * DECREF castbuf when underlying function used
             * object arrays and casting was needed to get
             * to object arrays
             */
            for (i = 0; i < self->nargs; i++) {
                if (loop->cast[i]) {
                    if (steps[i] == 0) {
                        NpyInterface_DECREF(*((void **)castbuf[i]));
                    }
                    else {
                        int size = loop->bufsize;

                        void **objptr = (void **)castbuf[i];
                        /*
                         * size is loop->bufsize unless there
                         * was only one loop
                         */
                        if (ninnerloops == 1) {
                            size = loop->leftover;
                        }
                        for (j = 0; j < size; j++) {
                            NpyInterface_DECREF(*objptr);
                            *objptr = NULL;
                            objptr += 1;
                        }
                    }
                }
            }
        }
        ufuncloop_dealloc(loop);
    }
    return -1;
}



int
NpyUFunc_SetUsesArraysAsData(void **data, size_t i)
{
    data[i] = (void*)NpyUFunc_SetUsesArraysAsData;
    return 0;
}


static NpyUFuncLoopObject *
construct_loop(NpyUFuncObject *self)
{
    NpyUFuncLoopObject *loop;
    int i;
    char *name;
    
    if (self == NULL) {
        NpyErr_SetString(NpyExc_ValueError, "function not supported");
        return NULL;
    }
    if ((loop = malloc(sizeof(NpyUFuncLoopObject))) == NULL) {
        NpyErr_SetString(NpyExc_MemoryError, "no memory");
        return loop;
    }
    
    loop->iter = NpyArray_MultiIterNew();
    if (loop->iter == NULL) {
        free(loop);
        return NULL;
    }
    
    loop->iter->index = 0;
    loop->iter->numiter = self->nargs;
    loop->ufunc = self;
    Npy_INCREF(loop->ufunc);
    loop->buffer[0] = NULL;
    for (i = 0; i < self->nargs; i++) {
        loop->iter->iters[i] = NULL;
        loop->cast[i] = NULL;
    }
    loop->errobj = NULL;
    loop->notimplemented = 0;
    loop->first = 1;
    loop->core_dim_sizes = NULL;
    loop->core_strides = NULL;
    
    if (self->core_enabled) {
        int num_dim_ix = 1 + self->core_num_dim_ix;
        int nstrides = self->nargs 
        + self->core_offsets[self->nargs - 1]
        + self->core_num_dims[self->nargs - 1];
        loop->core_dim_sizes = malloc(sizeof(npy_intp)*num_dim_ix);
        loop->core_strides = malloc(sizeof(npy_intp)*nstrides);
        if (loop->core_dim_sizes == NULL || loop->core_strides == NULL) {
            NpyErr_SetString(NpyExc_MemoryError, "no memory");
            goto fail;
        }
        memset(loop->core_strides, 0, sizeof(npy_intp) * nstrides);
        for (i = 0; i < num_dim_ix; i++) {
            loop->core_dim_sizes[i] = 1;
        }
    }
    name = self->name ? self->name : "";
    
    return loop;
    
fail:
    ufuncloop_dealloc(loop);
    return NULL;
}




static size_t
construct_arrays(NpyUFuncLoopObject *loop, size_t nargs, NpyArray **mps,
                 int *rtypenums, npy_prepare_outputs_func prepare, void *prepare_data)
{
    int i;
    int arg_types[NPY_MAXARGS];
    NPY_SCALARKIND scalars[NPY_MAXARGS];
    NPY_SCALARKIND maxarrkind, maxsckind, new;
    NpyUFuncObject *self = loop->ufunc;
    npy_bool allscalars = NPY_TRUE;
    int flexible = 0;
    int object = 0;
    
    npy_intp temp_dims[NPY_MAXDIMS];
    npy_intp *out_dims;
    int out_nd;
    
    /* Get each input argument */
    maxarrkind = NPY_NOSCALAR;
    maxsckind = NPY_NOSCALAR;
    for(i = 0; i < self->nin; i++) {
        arg_types[i] = NpyArray_TYPE(mps[i]);
        if (!flexible && NpyTypeNum_ISFLEXIBLE(arg_types[i])) {
            flexible = 1;
        }
        if (!object && NpyTypeNum_ISOBJECT(arg_types[i])) {
            object = 1;
        }
        /*
         * debug
         * fprintf(stderr, "array %d has reference %d\n", i,
         * (mps[i])->ob_refcnt);
         */
        
        /*
         * Scalars are 0-dimensional arrays at this point
         */
        
        /*
         * We need to keep track of whether or not scalars
         * are mixed with arrays of different kinds.
         */
        
        if (NpyArray_NDIM(mps[i]) > 0) {
            scalars[i] = NPY_NOSCALAR;
            allscalars = NPY_FALSE;
            new = NpyArray_ScalarKind(arg_types[i], NULL);
            maxarrkind = NpyArray_MAX(new, maxarrkind);
        }
        else {
            scalars[i] = NpyArray_ScalarKind(arg_types[i], &(mps[i]));
            maxsckind = NpyArray_MAX(scalars[i], maxsckind);
        }
    }
    
    /* We don't do strings */
    if (flexible && !object) {
        loop->notimplemented = 1;
        return nargs;
    }
    
    /*
     * If everything is a scalar, or scalars mixed with arrays of
     * different kinds of lesser kinds then use normal coercion rules
     */
    if (allscalars || (maxsckind > maxarrkind)) {
        for (i = 0; i < self->nin; i++) {
            scalars[i] = NPY_NOSCALAR;
        }
    }
    
    /* Select an appropriate function for these argument types. */
    if (select_types(loop->ufunc, arg_types, &(loop->function),
                     &(loop->funcdata), scalars, rtypenums) == -1) {
        return -1;
    }
    
    /*
     * Create copies for some of the arrays if they are small
     * enough and not already contiguous
     */
    if (_create_copies(loop, arg_types, mps) < 0) {
        return -1;
    }
    
    /*
     * Only use loop dimensions when constructing Iterator:
     * temporarily replace mps[i] (will be recovered below).
     */
    if (self->core_enabled) {
        for (i = 0; i < self->nin; i++) {
            NpyArray *ao;
            
            if (_compute_dimension_size(loop, mps, i) < 0) {
                return -1;
            }
            ao = _trunc_coredim(mps[i], self->core_num_dims[i]);
            if (ao == NULL) {
                return -1;
            }
            mps[i] = ao;
        }
    }
    
    /* Create Iterators for the Inputs */
    for (i = 0; i < self->nin; i++) {
        loop->iter->iters[i] = NpyArray_IterNew(mps[i]);
        if (loop->iter->iters[i] == NULL) {
            return -1;
        }
    }
    
    /* Recover mps[i]. */
    if (self->core_enabled) {
        for (i = 0; i < self->nin; i++) {
            NpyArray *ao = mps[i];
            mps[i] = NpyArray_BASE_ARRAY(mps[i]);
            Npy_DECREF(ao);
        }
    }
    
    /* Broadcast the result */
    loop->iter->numiter = self->nin;
    if (NpyArray_Broadcast(loop->iter) < 0) {
        return -1;
    }
    
    /* Get any return arguments */
    for (i = self->nin; i < nargs; i++) {
        if (self->core_enabled) {
            if (_compute_dimension_size(loop, mps, i) < 0) {
                return -1;
            }
        }
        out_dims = _compute_output_dims(loop, i, &out_nd, temp_dims);
        if (!out_dims) {
            return -1;
        }
        if (NpyArray_NDIM(mps[i]) != out_nd
            || !NpyArray_CompareLists(NpyArray_DIMS(mps[i]), out_dims, out_nd)) {
            NpyErr_SetString(NpyExc_ValueError, "invalid return array shape");
            Npy_DECREF(mps[i]);
            mps[i] = NULL;
            return -1;
        }
        if (!NpyArray_ISWRITEABLE(mps[i])) {
            NpyErr_SetString(NpyExc_ValueError, "return array is not writeable");
            Npy_DECREF(mps[i]);
            mps[i] = NULL;
            return -1;
        }
    }
    
    /* construct any missing return arrays and make output iterators */
    for(i = self->nin; i < self->nargs; i++) {
        NpyArray_Descr *ntype;
        
        if (mps[i] == NULL) {
            out_dims = _compute_output_dims(loop, i, &out_nd, temp_dims);
            if (!out_dims) {
                return -1;
            }
            mps[i] = NpyArray_New(NULL, 
                                  out_nd,
                                  out_dims,
                                  arg_types[i],
                                  NULL, NULL,
                                  0, 0, NULL);
            if (mps[i] == NULL) {
                return -1;
            }
        }
        
        /*
         * reset types for outputs that are equivalent
         * -- no sense casting uselessly
         */
        else {
            if (NpyArray_TYPE(mps[i]) != arg_types[i]) {
                NpyArray_Descr *atype;
                ntype = NpyArray_DESCR(mps[i]);
                atype = NpyArray_DescrFromType(arg_types[i]);
                if (NpyArray_EquivTypes(atype, ntype)) {
                    arg_types[i] = ntype->type_num;
                }
                Npy_DECREF(atype);
            }
            
            /* still not the same -- or will we have to use buffers?*/
            if (NpyArray_TYPE(mps[i]) != arg_types[i]
                || !NpyArray_ISBEHAVED_RO(mps[i])) {
                if (loop->iter->size < loop->bufsize || self->core_enabled) {
                    NpyArray *new;
                    /*
                     * Copy the array to a temporary copy
                     * and set the UPDATEIFCOPY flag
                     */
                    ntype = NpyArray_DescrFromType(arg_types[i]);
                    new = NpyArray_FromArray(mps[i], ntype, NPY_FORCECAST | NPY_ALIGNED | NPY_UPDATEIFCOPY);
                    if (new == NULL) {
                        return -1;
                    }
                    Npy_DECREF(mps[i]);
                    mps[i] = new;
                }
            }
        }
        
        if (self->core_enabled) {
            NpyArray *ao;
            
            /* computer for all output arguments, and set strides in "loop" */
            if (_compute_dimension_size(loop, mps, i) < 0) {
                return -1;
            }
            ao = _trunc_coredim(mps[i], self->core_num_dims[i]);
            if (ao == NULL) {
                return -1;
            }
            /* Temporarily modify mps[i] for constructing iterator. */
            mps[i] = ao;
        }
        
        loop->iter->iters[i] = NpyArray_IterNew(mps[i]);
        if (loop->iter->iters[i] == NULL) {
            return -1;
        }
        
        /* Recover mps[i]. */
        if (self->core_enabled) {
            NpyArray *ao = mps[i];
            mps[i] = NpyArray_BASE_ARRAY(mps[i]);
            Npy_DECREF(ao);
        }
        
    }
    
    /* wrap outputs */
    if (prepare) {
        if (prepare(self, mps, prepare_data) < 0) {
            return -1;
        }
    }
    
    /*
     * If any of different type, or misaligned or swapped
     * then must use buffers
     */
    loop->bufcnt = 0;
    loop->obj = 0;
    /* Determine looping method needed */
    loop->meth = NO_UFUNCLOOP;
    if (loop->iter->size == 0) {
        return nargs;
    }
    if (self->core_enabled) {
        loop->meth = SIGNATURE_NOBUFFER_UFUNCLOOP;
    }
    for (i = 0; i < self->nargs; i++) {
        loop->needbuffer[i] = 0;
        if (arg_types[i] != NpyArray_TYPE(mps[i])
            || !NpyArray_ISBEHAVED_RO(mps[i])) {
            if (self->core_enabled) {
                NpyErr_SetString(NpyExc_RuntimeError,
                                "never reached; copy should have been made");
                return -1;
            }
            loop->meth = BUFFER_UFUNCLOOP;
            loop->needbuffer[i] = 1;
        }
        if (!(loop->obj & UFUNC_OBJ_ISOBJECT)
            && ((NpyArray_TYPE(mps[i]) == NPY_OBJECT)
                || (arg_types[i] == NPY_OBJECT))) {
                loop->obj = UFUNC_OBJ_ISOBJECT|UFUNC_OBJ_NEEDS_API;
            }
        if (!(loop->obj & UFUNC_OBJ_NEEDS_API)
            && ((NpyArray_TYPE(mps[i]) == NPY_DATETIME)
                || (NpyArray_TYPE(mps[i]) == NPY_TIMEDELTA)
                || (arg_types[i] == NPY_DATETIME)
                || (arg_types[i] == NPY_TIMEDELTA))) {
                loop->obj = UFUNC_OBJ_NEEDS_API;
            }
    }
    
    if (self->core_enabled && (loop->obj & UFUNC_OBJ_ISOBJECT)) {
        NpyErr_SetString(NpyExc_TypeError,
                         "Object type not allowed in ufunc with signature");
        return -1;
    }
    if (loop->meth == NO_UFUNCLOOP) {
        loop->meth = ONE_UFUNCLOOP;
        
        /* All correct type and BEHAVED */
        /* Check for non-uniform stridedness */
        for (i = 0; i < self->nargs; i++) {
            if (!(loop->iter->iters[i]->contiguous)) {
                /*
                 * May still have uniform stride
                 * if (broadcast result) <= 1-d
                 */
                if (NpyArray_NDIM(mps[i]) != 0 &&                  \
                    (loop->iter->iters[i]->nd_m1 > 0)) {
                    loop->meth = NOBUFFER_UFUNCLOOP;
                    break;
                }
            }
        }
        if (loop->meth == ONE_UFUNCLOOP) {
            for (i = 0; i < self->nargs; i++) {
                loop->bufptr[i] = NpyArray_BYTES(mps[i]);
            }
        }
    }
    
    loop->iter->numiter = self->nargs;
    
    /* Fill in steps  */
    if (loop->meth == SIGNATURE_NOBUFFER_UFUNCLOOP && loop->iter->nd == 0) {
        /* Use default core_strides */
    }
    else if (loop->meth != ONE_UFUNCLOOP) {
        int ldim;
        npy_intp minsum;
        npy_intp maxdim;
        NpyArrayIterObject *it;
        npy_intp stride_sum[NPY_MAXDIMS];
        int j;
        
        /* Fix iterators */
        
        /*
         * Optimize axis the iteration takes place over
         *
         * The first thought was to have the loop go
         * over the largest dimension to minimize the number of loops
         *
         * However, on processors with slow memory bus and cache,
         * the slowest loops occur when the memory access occurs for
         * large strides.
         *
         * Thus, choose the axis for which strides of the last iterator is
         * smallest but non-zero.
         */
        for (i = 0; i < loop->iter->nd; i++) {
            stride_sum[i] = 0;
            for (j = 0; j < loop->iter->numiter; j++) {
                stride_sum[i] += loop->iter->iters[j]->strides[i];
            }
        }
        
        ldim = loop->iter->nd - 1;
        minsum = stride_sum[loop->iter->nd - 1];
        for (i = loop->iter->nd - 2; i >= 0; i--) {
            if (stride_sum[i] < minsum ) {
                ldim = i;
                minsum = stride_sum[i];
            }
        }
        maxdim = loop->iter->dimensions[ldim];
        loop->iter->size /= maxdim;
        loop->bufcnt = maxdim;
        loop->lastdim = ldim;
        
        /*
         * Fix the iterators so the inner loop occurs over the
         * largest dimensions -- This can be done by
         * setting the size to 1 in that dimension
         * (just in the iterators)
         */
        for (i = 0; i < loop->iter->numiter; i++) {
            it = loop->iter->iters[i];
            it->contiguous = 0;
            it->size /= (it->dims_m1[ldim] + 1);
            it->dims_m1[ldim] = 0;
            it->backstrides[ldim] = 0;
            
            /*
             * (won't fix factors because we
             * don't use PyArray_ITER_GOTO1D
             * so don't change them)
             *
             * Set the steps to the strides in that dimension
             */
            loop->steps[i] = it->strides[ldim];
        }
        
        /*
         * Set looping part of core_dim_sizes and core_strides.
         */
        if (loop->meth == SIGNATURE_NOBUFFER_UFUNCLOOP) {
            loop->core_dim_sizes[0] = maxdim;
            for (i = 0; i < self->nargs; i++) {
                loop->core_strides[i] = loop->steps[i];
            }
        }
        
        /*
         * fix up steps where we will be copying data to
         * buffers and calculate the ninnerloops and leftover
         * values -- if step size is already zero that is not changed...
         */
        if (loop->meth == BUFFER_UFUNCLOOP) {
            loop->leftover = maxdim % loop->bufsize;
            loop->ninnerloops = (maxdim / loop->bufsize) + 1;
            for (i = 0; i < self->nargs; i++) {
                if (loop->needbuffer[i] && loop->steps[i]) {
                    loop->steps[i] = NpyArray_ITEMSIZE(mps[i]);
                }
                /* These are changed later if casting is needed */
            }
        }
    }
    else if (loop->meth == ONE_UFUNCLOOP) {
        /* uniformly-strided case */
        for (i = 0; i < self->nargs; i++) {
            if (NpyArray_SIZE(mps[i]) == 1) {
                loop->steps[i] = 0;
            }
            else {
                loop->steps[i] = NpyArray_STRIDE(mps[i], NpyArray_NDIM(mps[i])-1);
            }
        }
    }
    
    
    /* Finally, create memory for buffers if we need them */
    
    /*
     * Buffers for scalars are specially made small -- scalars are
     * not copied multiple times
     */
    if (loop->meth == BUFFER_UFUNCLOOP) {
        int cnt = 0, cntcast = 0;
        int scnt = 0, scntcast = 0;
        char *castptr;
        char *bufptr;
        int last_was_scalar = 0;
        int last_cast_was_scalar = 0;
        int oldbufsize = 0;
        int oldsize = 0;
        int scbufsize = 4*sizeof(double);
        int memsize;
        NpyArray_Descr *descr;
        
        /* compute the element size */
        for (i = 0; i < self->nargs; i++) {
            if (!loop->needbuffer[i]) {
                continue;
            }
            if (arg_types[i] != mps[i]->descr->type_num) {
                descr = NpyArray_DescrFromType(arg_types[i]);
                if (loop->steps[i]) {
                    cntcast += descr->elsize;
                }
                else {
                    scntcast += descr->elsize;
                }
                if (i < self->nin) {
                    loop->cast[i] = NpyArray_GetCastFunc(NpyArray_DESCR(mps[i]),
                                                         arg_types[i]);
                }
                else {
                    loop->cast[i] = NpyArray_GetCastFunc \
                    ( descr, NpyArray_DESCR(mps[i])->type_num);
                }
                Npy_DECREF(descr);
                if (!loop->cast[i]) {
                    return -1;
                }
            }
            loop->swap[i] = !(NpyArray_ISNOTSWAPPED(mps[i]));
            if (loop->steps[i]) {
                cnt += NpyArray_ITEMSIZE(mps[i]);
            }
            else {
                scnt += NpyArray_ITEMSIZE(mps[i]);
            }
        }
        memsize = loop->bufsize*(cnt+cntcast) + scbufsize*(scnt+scntcast);
        loop->buffer[0] = NpyDataMem_NEW(memsize);
        
        /*
         * debug
         * fprintf(stderr, "Allocated buffer at %p of size %d, cnt=%d, cntcast=%d\n",
         *               loop->buffer[0], loop->bufsize * (cnt + cntcast), cnt, cntcast);
         */
        if (loop->buffer[0] == NULL) {
            NpyErr_SetString(NpyExc_MemoryError, "no memory");
            return -1;
        }
        if (loop->obj & UFUNC_OBJ_ISOBJECT) {
            memset(loop->buffer[0], 0, memsize);
        }
        castptr = loop->buffer[0] + loop->bufsize*cnt + scbufsize*scnt;
        bufptr = loop->buffer[0];
        loop->objfunc = 0;
        for (i = 0; i < self->nargs; i++) {
            if (!loop->needbuffer[i]) {
                continue;
            }
            loop->buffer[i] = bufptr + (last_was_scalar ? scbufsize :
                                        loop->bufsize)*oldbufsize;
            last_was_scalar = (loop->steps[i] == 0);
            bufptr = loop->buffer[i];
            oldbufsize = NpyArray_ITEMSIZE(mps[i]);
            /* fprintf(stderr, "buffer[%d] = %p\n", i, loop->buffer[i]); */
            if (loop->cast[i]) {
                NpyArray_Descr *descr;
                loop->castbuf[i] = castptr + (last_cast_was_scalar ? scbufsize :
                                              loop->bufsize)*oldsize;
                last_cast_was_scalar = last_was_scalar;
                /* fprintf(stderr, "castbuf[%d] = %p\n", i, loop->castbuf[i]); */
                descr = NpyArray_DescrFromType(arg_types[i]);
                oldsize = descr->elsize;
                Npy_DECREF(descr);
                loop->bufptr[i] = loop->castbuf[i];
                castptr = loop->castbuf[i];
                if (loop->steps[i]) {
                    loop->steps[i] = oldsize;
                }
            }
            else {
                loop->bufptr[i] = loop->buffer[i];
            }
            if (!loop->objfunc && (loop->obj & UFUNC_OBJ_ISOBJECT)) {
                if (arg_types[i] == NPY_OBJECT) {
                    loop->objfunc = 1;
                }
            }
        }
    }
    
    if (_does_loop_use_arrays(loop->funcdata)) {
        loop->funcdata = (void*)mps;
    }
    
    return nargs;
}



static char *_types_msg =  "function not supported for these types, "   \
    "and can't coerce safely to supported types";


/*
 * Called to determine coercion
 * Can change arg_types.
 */
static int
select_types(NpyUFuncObject *self, int *arg_types,
             NpyUFuncGenericFunction *function, void **data,
             NPY_SCALARKIND *scalars,
             int *rtypenums)
{
    int i, j;
    char start_type;
    int userdef = -1;
    int userdef_ind = -1;
    
    if (self->userloops) {
        for(i = 0; i < self->nin; i++) {
            if (NpyTypeNum_ISUSERDEF(arg_types[i])) {
                userdef = arg_types[i];
                userdef_ind = i;
                break;
            }
        }
    }
    
    if (rtypenums != NULL)
        return extract_specified_loop(self, arg_types, function, data,
                                      rtypenums, userdef);
    
    if (userdef > 0) {
        int ret = -1;
        
        /*
         * Look through all the registered loops for all the user-defined
         * types to find a match.
         */
        while (ret == -1) {
            NpyUFunc_Loop1d *funcdata;
            npy_intp userdefP;
            
            if (userdef_ind >= self->nin) {
                break;
            }
            userdef = arg_types[userdef_ind++];
            if (!(NpyTypeNum_ISUSERDEF(userdef))) {
                continue;
            }
            userdefP = (npy_intp)userdef;
            funcdata = NpyDict_Get(self->userloops, (void *)userdefP);
            /*
             * extract the correct function
             * data and argtypes for this user-defined type.
             */
            ret = _find_matching_userloop(funcdata, arg_types, scalars,
                                          function, data, self->nargs,
                                          self->nin);
        }
        if (ret == 0) {
            return ret;
        }
        NpyErr_SetString(NpyExc_TypeError, _types_msg);
        return ret;
    }
    
    start_type = arg_types[0];
    /*
     * If the first argument is a scalar we need to place
     * the start type as the lowest type in the class
     */
    if (scalars[0] != NPY_NOSCALAR) {
        start_type = _lowest_type(start_type);
    }
    
    i = 0;
    while (i < self->ntypes && start_type > self->types[i*self->nargs]) {
        i++;
    }
    for (; i < self->ntypes; i++) {
        for (j = 0; j < self->nin; j++) {
            if (!NpyArray_CanCoerceScalar(arg_types[j],
                                          self->types[i*self->nargs + j],
                                          scalars[j]))
                break;
        }
        if (j == self->nin) {
            break;
        }
    }
    if (i >= self->ntypes) {
        NpyErr_SetString(NpyExc_TypeError, _types_msg);
        return -1;
    }
    for (j = 0; j < self->nargs; j++) {
        arg_types[j] = self->types[i*self->nargs+j];
    }
    if (self->data) {
        *data = self->data[i];
    }
    else {
        *data = NULL;
    }
    *function = self->functions[i];
    
    return 0;
}



/*
 * if only one type is specified then it is the "first" output data-type
 * and the first signature matching this output data-type is returned.
 *
 * if a tuple of types is specified then an exact match to the signature
 * is searched and it much match exactly or an error occurs
 */
static int
extract_specified_loop(NpyUFuncObject *self, int *arg_types,
                       NpyUFuncGenericFunction *function, void **data,
                       int *rtypenums, int userdef)
{
    static char msg[] = "loop written to specified type(s) not found";
    int nargs;
    int i, j;
    
    nargs = self->nargs;
    if (userdef > 0) {
        /* search in the user-defined functions */
        NpyUFunc_Loop1d *funcdata;
        
        funcdata = NpyDict_Get(self->userloops, (void *)(npy_intp)userdef);
        if (NULL == funcdata) {
            NpyErr_SetString(NpyExc_TypeError,
                             "user-defined type used in ufunc" \
                             " with no registered loops");
            return -1;
        }
        /*
         * extract the correct function
         * data and argtypes
         */
        while (funcdata != NULL) {
            if (rtypenums[0] == funcdata->arg_types[self->nin]) {
                i = nargs;
            }
            else {
                i = -1;
            }
            if (i == nargs) {
                *function = funcdata->func;
                *data = funcdata->data;
                for(i = 0; i < nargs; i++) {
                    arg_types[i] = funcdata->arg_types[i];
                }
                return 0;
            }
            funcdata = funcdata->next;
        }
        NpyErr_SetString(NpyExc_TypeError, msg);
        return -1;
    }
    
    /* look for match in self->functions */
    for (j = 0; j < self->ntypes; j++) {
        if (rtypenums[0] == self->types[j*nargs+self->nin]) {
            i = nargs;
        }
        else {
            i = -1;
        }
        if (i == nargs) {
            *function = self->functions[j];
            *data = self->data[j];
            for (i = 0; i < nargs; i++) {
                arg_types[i] = self->types[j*nargs+i];
            }
            return 0;
        }
    }
    NpyErr_SetString(NpyExc_TypeError, msg);
    
    return -1;
}




static void
ufuncloop_dealloc(NpyUFuncLoopObject *self)
{
    if (self->ufunc != NULL) {
        if (self->core_dim_sizes) {
            free(self->core_dim_sizes);
        }
        if (self->core_strides) {
            free(self->core_strides);
        }
        self->iter->numiter = self->ufunc->nargs;
        Npy_DECREF(self->iter);
        if (self->buffer[0]) {
            NpyDataMem_FREE(self->buffer[0]);
        }
        NpyInterface_DECREF(self->errobj);
        Npy_DECREF(self->ufunc);
    }
    self->magic_number = NPY_INVALID_MAGIC;
    free(self);
}



/*
 * Create copies for any arrays that are less than loop->bufsize
 * in total size (or core_enabled) and are mis-behaved or in need
 * of casting.
 */
static int
_create_copies(NpyUFuncLoopObject *loop, int *arg_types, NpyArray **mps)
{
    int nin = loop->ufunc->nin;
    int i;
    npy_intp size;
    NpyArray_Descr *ntype;
    NpyArray_Descr *atype;
    
    for (i = 0; i < nin; i++) {
        size = NpyArray_SIZE(mps[i]);
        /*
         * if the type of mps[i] is equivalent to arg_types[i]
         * then set arg_types[i] equal to type of mps[i] for later checking....
         */
        if (NpyArray_TYPE(mps[i]) != arg_types[i]) {
            ntype = mps[i]->descr;
            atype = NpyArray_DescrFromType(arg_types[i]);
            if (NpyArray_EquivTypes(atype, ntype)) {
                arg_types[i] = ntype->type_num;
            }
            Npy_DECREF(atype);
        }
        if (size < loop->bufsize || loop->ufunc->core_enabled) {
            if (!(NpyArray_ISBEHAVED_RO(mps[i]))
                || NpyArray_TYPE(mps[i]) != arg_types[i]) {
                NpyArray *new;
                ntype = NpyArray_DescrFromType(arg_types[i]);
                
                /* Move reference to interface. */
                new = NpyArray_FromArray(mps[i], ntype, NPY_FORCECAST | NPY_ALIGNED);
                if (new == NULL) {
                    return -1;
                }
                Npy_DECREF(mps[i]);
                mps[i] = new;
            }
        }
    }
    return 0;
}



/* Check and set core_dim_sizes and core_strides for the i-th argument. */
static int
_compute_dimension_size(NpyUFuncLoopObject *loop, NpyArray **mps, int i)
{
    NpyUFuncObject *ufunc = loop->ufunc;
    int j = ufunc->core_offsets[i];
    int k = NpyArray_NDIM(mps[i]) - ufunc->core_num_dims[i];
    int ind;
    
    for (ind = 0; ind < ufunc->core_num_dims[i]; ind++, j++, k++) {
        npy_intp dim = k < 0 ? 1 : NpyArray_DIM(mps[i], k);
        /* First element of core_dim_sizes will be used for looping */
        int dim_ix = ufunc->core_dim_ixs[j] + 1;
        if (loop->core_dim_sizes[dim_ix] == 1) {
            /* broadcast core dimension  */
            loop->core_dim_sizes[dim_ix] = dim;
        }
        else if (dim != 1 && dim != loop->core_dim_sizes[dim_ix]) {
            NpyErr_SetString(NpyExc_ValueError, "core dimensions mismatch");
            return -1;
        }
        /* First ufunc->nargs elements will be used for looping */
        loop->core_strides[ufunc->nargs + j] =
        dim == 1 ? 0 : NpyArray_STRIDE(mps[i], k);
    }
    return 0;
}


/*
 * Concatenate the loop and core dimensions of
 * PyArrayMultiIterObject's iarg-th argument, to recover a full
 * dimension array (used for output arguments).
 */
static npy_intp*
_compute_output_dims(NpyUFuncLoopObject *loop, int iarg,
                     int *out_nd, npy_intp *tmp_dims)
{
    int i;
    NpyUFuncObject *ufunc = loop->ufunc;
    
    if (ufunc->core_enabled == 0) {
        /* case of ufunc with trivial core-signature */
        *out_nd = loop->iter->nd;
        return loop->iter->dimensions;
    }
    
    *out_nd = loop->iter->nd + ufunc->core_num_dims[iarg];
    if (*out_nd > NPY_MAXARGS) {
        NpyErr_SetString(NpyExc_ValueError,
                         "dimension of output variable exceeds limit");
        return NULL;
    }
    
    /* copy loop dimensions */
    memcpy(tmp_dims, loop->iter->dimensions, sizeof(npy_intp) * loop->iter->nd);
    
    /* copy core dimension */
    for (i = 0; i < ufunc->core_num_dims[iarg]; i++) {
        tmp_dims[loop->iter->nd + i] = loop->core_dim_sizes[1 +
                                                            ufunc->core_dim_ixs[ufunc->core_offsets[iarg] + i]];
    }
    return tmp_dims;
}



/* Return a view of array "ap" with "core_nd" dimensions cut from tail. */
static NpyArray *
_trunc_coredim(NpyArray *ap, int core_nd)
{
    NpyArray *ret;
    int nd = NpyArray_NDIM(ap) - core_nd;
    
    if (nd < 0) {
        nd = 0;
    }
    /* The following code is basically taken from PyArray_Transpose */
    /* NewFromDescr will steal this reference */
    Npy_INCREF(ap->descr);
    ret = NpyArray_NewFromDescr(ap->descr,
                                nd, ap->dimensions,
                                ap->strides, 
                                ap->data, 
                                ap->flags,
                                NPY_FALSE, NULL, Npy_INTERFACE(ap));
    if (ret == NULL) {
        return NULL;
    }
    /* point at true owner of memory: */
    NpyArray_BASE_ARRAY(ret) = ap;
    Npy_INCREF(ap);
    assert(NULL == NpyArray_BASE(ret));
    NpyArray_UpdateFlags(ret, NPY_CONTIGUOUS | NPY_FORTRAN);
    return ret;
}



/*
 * Called for non-NULL user-defined functions.
 * The object should be a CObject pointing to a linked-list of functions
 * storing the function, data, and signature of all user-defined functions.
 * There must be a match with the input argument types or an error
 * will occur.
 */
static int
_find_matching_userloop(NpyUFunc_Loop1d *funcdata, int *arg_types,
                        NPY_SCALARKIND *scalars,
                        NpyUFuncGenericFunction *function, void **data,
                        int nargs, int nin)
{
    int i;
    
    while (funcdata != NULL) {
        for (i = 0; i < nin; i++) {
            if (!NpyArray_CanCoerceScalar(arg_types[i],
                                          funcdata->arg_types[i],
                                          scalars[i]))
                break;
        }
        if (i == nin) {
            /* match found */
            *function = funcdata->func;
            *data = funcdata->data;
            /* Make sure actual arg_types supported by the loop are used */
            for (i = 0; i < nargs; i++) {
                arg_types[i] = funcdata->arg_types[i];
            }
            return 0;
        }
        funcdata = funcdata->next;
    }
    return -1;
}



static char
_lowest_type(char intype) 
{
    /* TODO: Ready to move */
    switch(intype) {
            /* case PyArray_BYTE */
        case NPY_SHORT:
        case NPY_INT:
        case NPY_LONG:
        case NPY_LONGLONG:
        case NPY_DATETIME:
        case NPY_TIMEDELTA:
            return NPY_BYTE;
            /* case NPY_UBYTE */
        case NPY_USHORT:
        case NPY_UINT:
        case NPY_ULONG:
        case NPY_ULONGLONG:
            return NPY_UBYTE;
            /* case PyArray_FLOAT:*/
        case NPY_DOUBLE:
        case NPY_LONGDOUBLE:
            return NPY_FLOAT;
            /* case PyArray_CFLOAT:*/
        case NPY_CDOUBLE:
        case NPY_CLONGDOUBLE:
            return NPY_CFLOAT;
        default:
            return intype;
    }
}


/* Return 1 if the given data pointer for the loop specifies that it needs the
 * arrays as the data pointer.
 */
static int
_does_loop_use_arrays(void *data)
{
    return (data == NpyUFunc_SetUsesArraysAsData);
}


/*
 * Floating point error handling.
 */

void 
NpyUFunc_SetFpErrHandler(void (*handler)(int, void *, int, int *))
{
    fp_error_handler = handler;
}



int
NpyUFunc_getfperr(void)
{
    int retstatus;
    UFUNC_CHECK_STATUS(retstatus);
    return retstatus;
}


int
NpyUFunc_checkfperr(int errmask, void *errobj, int *first)
{
    int retstatus;
    
    /* 1. check hardware flag --- this is platform dependent code */
    retstatus = NpyUFunc_getfperr();
    fp_error_handler(errmask, errobj, retstatus, first);
    return 0;
}


/* Checking the status flag clears it */
void
NpyUFunc_clearfperr()
{
    NpyUFunc_getfperr();
}




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
static void npy_free_loop1d_list(NpyUFunc_Loop1d *data)
{
    while (data != NULL) {
        NpyUFunc_Loop1d *next = data->next;
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





