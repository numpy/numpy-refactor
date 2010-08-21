/*
 * Python Universal Functions Object -- CPython-independent portion
 *
 */
#include <stdlib.h>
#include <memory.h>
#include "npy_config.h"

#include "npy_common.h"
#include "npy_dict.h"
#include "npy_api.h"
#include "npy_arrayobject.h"
#include "npy_iterators.h"
#include "npy_ufunc_object.h"
#include "npy_os.h"
#include "npy_math.h"


/*
 * Forward decls
 */
static NpyUFuncLoopObject *
construct_loop(NpyUFuncObject *self);
static size_t
construct_arrays(NpyUFuncLoopObject *loop, size_t nargs, NpyArray **mps,
                 int *rtypenums, npy_prepare_outputs_func prepare,
                 void *prepare_data);
static NpyUFuncReduceObject *
construct_reduce(NpyUFuncObject *self, NpyArray **arr, NpyArray *out,
                 int axis, int otype, int operation, npy_intp ind_size,
                 char *str, int bufsize, int errormask, void *errobj);
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
static void
ufuncreduce_dealloc(NpyUFuncReduceObject *self);
static int
_create_reduce_copy(NpyUFuncReduceObject *loop, NpyArray **arr, int rtype);
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
cmp_arg_types(int *arg1, int *arg2, int n);
static int
_find_matching_userloop(NpyUFunc_Loop1d *funcdata, int *arg_types,
                        NPY_SCALARKIND *scalars,
                        NpyUFuncGenericFunction *function, void **data,
                        int nargs, int nin);
static int
_does_loop_use_arrays(void *data);
static char _lowest_type(char intype);
static int
_parse_signature(NpyUFuncObject *self, const char *signature);
static NpyArray *
_getidentity(NpyUFuncObject *self, int otype, char *str);



NpyTypeObject NpyUFunc_Type = {
    (void (*)(_NpyObject *))npy_ufunc_dealloc
};





static void NpyErr_NoMemory()
{
    NpyErr_SetString(NpyExc_MemoryError, "no memory");
}


static void default_fp_error_state(char *name, int *bufsizeRet,
                                   int *errormaskRet, void **errobjRet)
{
    *bufsizeRet = NPY_BUFSIZE;
    *errormaskRet = NPY_UFUNC_ERR_DEFAULT;
    *errobjRet = NULL;
}

/* Global floating-point error handling.  This is set by the interface
   layer or, if NULL defaults to a trival internal one.
*/
static void default_fp_error_handler(int errormask, void *errobj,
                                     int retstatus, int *first)
{
    const char *msg = "unknown";

    switch (errormask) {
        case NPY_UFUNC_FPE_DIVIDEBYZERO:
            msg = "division by zero"; break;
        case NPY_UFUNC_FPE_OVERFLOW:
            msg = "overflow"; break;
        case NPY_UFUNC_FPE_UNDERFLOW:
            msg = "underflow"; break;
        case NPY_UFUNC_FPE_INVALID:
            msg = "invalid"; break;
    }
    printf("libndarray floating point %s warning.", msg);
}

static void (*fp_error_state)(char *, int *, int *,
                              void **) = &default_fp_error_state;
static void (*fp_error_handler)(int, void *, int,
                                int *) = &default_fp_error_handler;


int NpyUFunc_GenericFunction(NpyUFuncObject *self, int nargs, NpyArray **mps,
                             int *rtypenums,
                             int originalArgWasObjArray,
                             npy_prepare_outputs_func prepare_outputs,
                             void *prepare_out_args)
{
    NpyUFuncLoopObject *loop;
    char *name = (NULL != self->name) ? self->name : "";
    int res;
    int i;

    assert(NPY_VALID_MAGIC == self->magic_number);

    /* Build the loop. */
    loop = construct_loop(self);
    if (loop == NULL) {
        return -1;
    }
    fp_error_state(name, &loop->bufsize, &loop->errormask, &loop->errobj);

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
            NPY_UFUNC_CHECK_ERROR(loop);
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
                NPY_UFUNC_CHECK_ERROR(loop);

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
                NPY_UFUNC_CHECK_ERROR(loop);

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
                pyobject[i] = ((loop->obj & NPY_UFUNC_OBJ_ISOBJECT)
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
             * the size of the inner function is B for all but the last when
             * the niter size is remainder.
             *
             * So, the code looks very similar to NOBUFFER_LOOP except the
             * inner-most loop is replaced with...
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
                    loop->function((char **)dptr, &bufcnt, steps,
                                   loop->funcdata);
                    NPY_UFUNC_CHECK_ERROR(loop);

                    for (i = self->nin; i < self->nargs; i++) {
                        if (!needbuffer[i]) {
                            continue;
                        }
                        if (loop->cast[i]) {
                            /* fprintf(stderr, "casting back... %d, %p", i,
                               castbuf[i]); */
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
                        /* copy back to output arrays decref what's already
                           there for object arrays */
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
                NPY_UFUNC_CHECK_ERROR(loop);

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




/*
 * We have two basic kinds of loops. One is used when arr is not-swapped
 * and aligned and output type is the same as input type.  The other uses
 * buffers when one of these is not satisfied.
 *
 *  Zero-length and one-length axes-to-be-reduced are handled separately.
 */
NpyArray *
NpyUFunc_Reduce(NpyUFuncObject *self, NpyArray *arr, NpyArray *out,
               int axis, int otype, int bufsize, int errormask, void *errobj)
{
    NpyArray *ret = NULL;
    NpyUFuncReduceObject *loop;
    npy_intp i, n;
    char *dptr;
//    NPY_BEGIN_THREADS_DEF

    assert(arr == NULL ||
           (NPY_VALID_MAGIC == arr->magic_number &&
            NPY_VALID_MAGIC == NpyArray_DESCR(arr)->magic_number));
    assert(NPY_VALID_MAGIC == self->magic_number);

    /* Construct loop object */
    loop = construct_reduce(self, &arr, out, axis, otype, NPY_UFUNC_REDUCE, 0,
                            "reduce", bufsize, errormask, errobj);
    if (!loop) {
        return NULL;
    }

//    NPY_LOOP_BEGIN_THREADS;
    switch(loop->meth) {
        case ZERO_EL_REDUCELOOP:
            /* fprintf(stderr, "ZERO..%d\n", loop->size); */
            for (i = 0; i < loop->size; i++) {
                if (loop->obj & NPY_UFUNC_OBJ_ISOBJECT) {
                    NpyInterface_INCREF(*((void **)loop->idptr));
                }
                memmove(loop->bufptr[0], loop->idptr, loop->outsize);
                loop->bufptr[0] += loop->outsize;
            }
            break;
        case ONE_EL_REDUCELOOP:
            /*fprintf(stderr, "ONEDIM..%d\n", loop->size); */
            while (loop->index < loop->size) {
                if (loop->obj & NPY_UFUNC_OBJ_ISOBJECT) {
                    NpyInterface_INCREF(*((void **)loop->it->dataptr));
                }
                memmove(loop->bufptr[0], loop->it->dataptr, loop->outsize);
                NpyArray_ITER_NEXT(loop->it);
                loop->bufptr[0] += loop->outsize;
                loop->index++;
            }
            break;
        case NOBUFFER_UFUNCLOOP:
            /*fprintf(stderr, "NOBUFFER..%d\n", loop->size); */
            while (loop->index < loop->size) {
                /* Copy first element to output */
                if (loop->obj & NPY_UFUNC_OBJ_ISOBJECT) {
                    NpyInterface_INCREF(*((void **)loop->it->dataptr));
                }
                memmove(loop->bufptr[0], loop->it->dataptr, loop->outsize);
                /* Adjust input pointer */
                loop->bufptr[1] = loop->it->dataptr+loop->steps[1];
                loop->function((char **)loop->bufptr, &(loop->N),
                               loop->steps, loop->funcdata);
                NPY_UFUNC_CHECK_ERROR(loop);
                NpyArray_ITER_NEXT(loop->it);
                loop->bufptr[0] += loop->outsize;
                loop->bufptr[2] = loop->bufptr[0];
                loop->index++;
            }
            break;
        case BUFFER_UFUNCLOOP:
            /*
             * use buffer for arr
             *
             * For each row to reduce
             * 1. copy first item over to output (casting if necessary)
             * 2. Fill inner buffer
             * 3. When buffer is filled or end of row
             * a. Cast input buffers if needed
             * b. Call inner function.
             * 4. Repeat 2 until row is done.
             */
            /* fprintf(stderr, "BUFFERED..%d %d\n", loop->size, loop->swap); */
            while(loop->index < loop->size) {
                loop->inptr = loop->it->dataptr;
                /* Copy (cast) First term over to output */
                if (loop->cast) {
                    /* A little tricky because we need to cast it first */
                    NpyArray_DESCR(arr)->f->copyswap(loop->buffer, loop->inptr,
                                                     loop->swap, NULL);
                    loop->cast(loop->buffer, loop->castbuf, 1, NULL, NULL);
                    if ((loop->obj & NPY_UFUNC_OBJ_ISOBJECT) &&
                        !NpyArray_ISOBJECT(arr)) {
                        /*
                         * In this case the cast function is creating
                         * an object reference so we need to incref
                         * it since we care copying it to bufptr[0].
                         */
                        NpyInterface_INCREF(*((void **)loop->castbuf));
                    }
                    memcpy(loop->bufptr[0], loop->castbuf, loop->outsize);
                }
                else {
                    /* Simple copy */
                    NpyArray_DESCR(arr)->f->copyswap(loop->bufptr[0],
                                                     loop->inptr,
                                                     loop->swap, NULL);
                }
                loop->inptr += loop->instrides;
                n = 1;
                while(n < loop->N) {
                    /* Copy up to loop->bufsize elements to buffer */
                    dptr = loop->buffer;
                    for (i = 0; i < loop->bufsize; i++, n++) {
                        if (n == loop->N) {
                            break;
                        }
                        NpyArray_DESCR(arr)->f->copyswap(dptr, loop->inptr,
                                                         loop->swap, NULL);
                        loop->inptr += loop->instrides;
                        dptr += loop->insize;
                    }
                    if (loop->cast) {
                        loop->cast(loop->buffer, loop->castbuf, i, NULL, NULL);
                    }
                    loop->function((char **)loop->bufptr, &i,
                                   loop->steps, loop->funcdata);
                    loop->bufptr[0] += loop->steps[0]*i;
                    loop->bufptr[2] += loop->steps[2]*i;
                    NPY_UFUNC_CHECK_ERROR(loop);
                }
                NpyArray_ITER_NEXT(loop->it);
                loop->bufptr[0] += loop->outsize;
                loop->bufptr[2] = loop->bufptr[0];
                loop->index++;
            }

            if (loop->obj & NPY_UFUNC_OBJ_ISOBJECT) {
                /*
                 * DECREF left-over objects if buffering was used.
                 * There are 2 cases here.
                 * 1. The output is an object. In this case the
                 * castfunc will produce objects and castbuf needs
                 * to be decrefed.
                 * 2. The input is an object array.  In this case
                 * the copyswap will produce object references and
                 * the buffer needs to be decrefed.
                 */
                if (!NpyArray_ISOBJECT(arr)) {
                    for (i=0; i<loop->bufsize; i++) {
                        NpyInterface_CLEAR(((void **)loop->castbuf)[i]);
                    }
                } else {
                    for (i=0; i<loop->bufsize; i++) {
                        NpyInterface_CLEAR(((void **)loop->buffer)[i]);
                    }
                }
            }
    }

//    NPY_LOOP_END_THREADS;
    /* Hang on to this reference -- will be decref'd with loop */
    if (loop->retbase) {
        ret = loop->ret->base_arr;
    }
    else {
        ret = loop->ret;
    }
    Npy_INCREF(ret);
    ufuncreduce_dealloc(loop);
    return ret;

fail:
//    NPY_LOOP_END_THREADS;
    if (loop) {
        ufuncreduce_dealloc(loop);
    }
    return NULL;
}



NpyArray *
NpyUFunc_Accumulate(NpyUFuncObject *self, NpyArray *arr, NpyArray *out,
                    int axis, int otype, int bufsize, int errormask,
                    void *errobj)
{
    NpyArray *ret = NULL;
    NpyUFuncReduceObject *loop;
    npy_intp i, n;
    char *dptr;
//    NPY_BEGIN_THREADS_DEF

    assert(NPY_VALID_MAGIC == self->magic_number);

    /* Construct loop object */
    loop = construct_reduce(self, &arr, out, axis, otype,
                            NPY_UFUNC_ACCUMULATE, 0, "accumulate",
                            bufsize, errormask, errobj);
    if (!loop) {
        return NULL;
    }

//    NPY_LOOP_BEGIN_THREADS;
    switch(loop->meth) {
        case ZERO_EL_REDUCELOOP:
            /* Accumulate */
            /* fprintf(stderr, "ZERO..%d\n", loop->size); */
            for (i = 0; i < loop->size; i++) {
                if (loop->obj & NPY_UFUNC_OBJ_ISOBJECT) {
                    NpyInterface_INCREF(*((void **)loop->idptr));
                }
                memcpy(loop->bufptr[0], loop->idptr, loop->outsize);
                loop->bufptr[0] += loop->outsize;
            }
            break;
        case ONE_EL_REDUCELOOP:
            /* Accumulate */
            /* fprintf(stderr, "ONEDIM..%d\n", loop->size); */
            while (loop->index < loop->size) {
                if (loop->obj & NPY_UFUNC_OBJ_ISOBJECT) {
                    NpyInterface_INCREF(*((void **)loop->it->dataptr));
                }
                memmove(loop->bufptr[0], loop->it->dataptr, loop->outsize);
                NpyArray_ITER_NEXT(loop->it);
                loop->bufptr[0] += loop->outsize;
                loop->index++;
            }
            break;
        case NOBUFFER_UFUNCLOOP:
            /* Accumulate */
            /* fprintf(stderr, "NOBUFFER..%d\n", loop->size); */
            while (loop->index < loop->size) {
                /* Copy first element to output */
                if (loop->obj & NPY_UFUNC_OBJ_ISOBJECT) {
                    NpyInterface_INCREF(*((void **)loop->it->dataptr));
                }
                memmove(loop->bufptr[0], loop->it->dataptr, loop->outsize);
                /* Adjust input pointer */
                loop->bufptr[1] = loop->it->dataptr + loop->steps[1];
                loop->function((char **)loop->bufptr, &(loop->N),
                               loop->steps, loop->funcdata);
                NPY_UFUNC_CHECK_ERROR(loop);
                NpyArray_ITER_NEXT(loop->it);
                NpyArray_ITER_NEXT(loop->rit);
                loop->bufptr[0] = loop->rit->dataptr;
                loop->bufptr[2] = loop->bufptr[0] + loop->steps[0];
                loop->index++;
            }
            break;
        case BUFFER_UFUNCLOOP:
            /* Accumulate
             *
             * use buffer for arr
             *
             * For each row to reduce
             * 1. copy identity over to output (casting if necessary)
             * 2. Fill inner buffer
             * 3. When buffer is filled or end of row
             * a. Cast input buffers if needed
             * b. Call inner function.
             * 4. Repeat 2 until row is done.
             */
            /* fprintf(stderr, "BUFFERED..%d %p\n", loop->size, loop->cast); */
            while (loop->index < loop->size) {
                loop->inptr = loop->it->dataptr;
                /* Copy (cast) First term over to output */
                if (loop->cast) {
                    /* A little tricky because we need to
                     cast it first */
                    NpyArray_DESCR(arr)->f->copyswap(loop->buffer, loop->inptr,
                                                     loop->swap, NULL);
                    loop->cast(loop->buffer, loop->castbuf, 1, NULL, NULL);
                    if (loop->obj & NPY_UFUNC_OBJ_ISOBJECT) {
                        NpyInterface_INCREF(*((void **)loop->castbuf));
                    }
                    memcpy(loop->bufptr[0], loop->castbuf, loop->outsize);
                }
                else {
                    /* Simple copy */
                    NpyArray_DESCR(arr)->f->copyswap(loop->bufptr[0],
                                                     loop->inptr,
                                                     loop->swap, NULL);
                }
                loop->inptr += loop->instrides;
                n = 1;
                while (n < loop->N) {
                    /* Copy up to loop->bufsize elements to buffer */
                    dptr = loop->buffer;
                    for (i = 0; i < loop->bufsize; i++, n++) {
                        if (n == loop->N) {
                            break;
                        }
                        NpyArray_DESCR(arr)->f->copyswap(dptr, loop->inptr,
                                                         loop->swap, NULL);
                        loop->inptr += loop->instrides;
                        dptr += loop->insize;
                    }
                    if (loop->cast) {
                        loop->cast(loop->buffer, loop->castbuf, i, NULL, NULL);
                    }
                    loop->function((char **)loop->bufptr, &i,
                                   loop->steps, loop->funcdata);
                    loop->bufptr[0] += loop->steps[0]*i;
                    loop->bufptr[2] += loop->steps[2]*i;
                    NPY_UFUNC_CHECK_ERROR(loop);
                }
                NpyArray_ITER_NEXT(loop->it);
                NpyArray_ITER_NEXT(loop->rit);
                loop->bufptr[0] = loop->rit->dataptr;
                loop->bufptr[2] = loop->bufptr[0] + loop->steps[0];
                loop->index++;
            }

            /*
             * DECREF left-over objects if buffering was used.
             * It is needed when casting created new objects in
             * castbuf.  Intermediate copying into castbuf (via
             * loop->function) decref'd what was already there.

             * It's the final copy into the castbuf that needs a DECREF.
             */

            /* Only when casting needed and it is from a non-object array */
            if ((loop->obj & NPY_UFUNC_OBJ_ISOBJECT) && loop->cast &&
                (!NpyArray_ISOBJECT(arr))) {
                for (i=0; i<loop->bufsize; i++) {
                    NpyInterface_CLEAR(((void **)loop->castbuf)[i]);
                }
            }

    }
//    NPY_LOOP_END_THREADS;
    /* Hang on to this reference -- will be decref'd with loop */
    if (loop->retbase) {
        ret = NpyArray_BASE_ARRAY(loop->ret);
    }
    else {
        ret = loop->ret;
    }
    Npy_INCREF(ret);
    ufuncreduce_dealloc(loop);
    return ret;

fail:
//    NPY_LOOP_END_THREADS;
    if (loop) {
        ufuncreduce_dealloc(loop);
    }
    return NULL;
}

/*
 * Reduceat performs a reduce over an axis using the indices as a guide
 *
 * op.reduceat(array,indices)  computes
 * op.reduce(array[indices[i]:indices[i+1]]
 * for i=0..end with an implicit indices[i+1]=len(array)
 * assumed when i=end-1
 *
 * if indices[i+1] <= indices[i]+1
 * then the result is array[indices[i]] for that value
 *
 * op.accumulate(array) is the same as
 * op.reduceat(array,indices)[::2]
 * where indices is range(len(array)-1) with a zero placed in every other sample
 * indices = zeros(len(array)*2-1)
 * indices[1::2] = range(1,len(array))
 *
 * output shape is based on the size of indices
 */
NpyArray *
NpyUFunc_Reduceat(NpyUFuncObject *self, NpyArray *arr, NpyArray *ind,
                  NpyArray *out, int axis, int otype,
                  int bufsize, int errormask, void *errobj)
{
    NpyArray *ret;
    NpyUFuncReduceObject *loop;
    npy_intp *ptr = (npy_intp *)NpyArray_BYTES(ind);
    npy_intp nn = NpyArray_DIM(ind, 0);
    npy_intp mm = NpyArray_DIM(arr, axis) - 1;
    npy_intp n, i, j;
    char *dptr;
//    NPY_BEGIN_THREADS_DEF;

    assert(NPY_VALID_MAGIC == self->magic_number);

    /* Check for out-of-bounds values in indices array */
    for (i = 0; i<nn; i++) {
        if ((*ptr < 0) || (*ptr > mm)) {
            char buf[256];

            NpyOS_snprintf(buf, 256, "index out-of-bounds (0, %d)", (int)mm);
            NpyErr_SetString(NpyExc_IndexError, buf);
            return NULL;
        }
        ptr++;
    }

    ptr = (npy_intp *)NpyArray_BYTES(ind);
    /* Construct loop object */
    loop = construct_reduce(self, &arr, out, axis, otype, NPY_UFUNC_REDUCEAT,
                            nn, "reduceat", bufsize, errormask, errobj);
    if (!loop) {
        return NULL;
    }

//    NPY_LOOP_BEGIN_THREADS;
    switch(loop->meth) {
        case ZERO_EL_REDUCELOOP:
            /* zero-length index -- return array immediately */
            /* fprintf(stderr, "ZERO..\n"); */
            break;
        case NOBUFFER_UFUNCLOOP:
            /* Reduceat
             * NOBUFFER -- behaved array and same type
             */
            /* fprintf(stderr, "NOBUFFER..%d\n", loop->size); */
            while (loop->index < loop->size) {
                ptr = (npy_intp *)NpyArray_BYTES(ind);
                for (i = 0; i < nn; i++) {
                    loop->bufptr[1] = loop->it->dataptr + (*ptr)*loop->steps[1];
                    if (loop->obj & NPY_UFUNC_OBJ_ISOBJECT) {
                        NpyInterface_INCREF(*((void **)loop->bufptr[1]));
                    }
                    memcpy(loop->bufptr[0], loop->bufptr[1], loop->outsize);
                    mm = (i == nn - 1 ? NpyArray_DIM(arr, axis) - *ptr :
                          *(ptr + 1) - *ptr) - 1;
                    if (mm > 0) {
                        loop->bufptr[1] += loop->steps[1];
                        loop->bufptr[2] = loop->bufptr[0];
                        loop->function((char **)loop->bufptr, &mm,
                                       loop->steps, loop->funcdata);
                        NPY_UFUNC_CHECK_ERROR(loop);
                    }
                    loop->bufptr[0] += NpyArray_STRIDE(loop->ret, axis);
                    ptr++;
                }
                NpyArray_ITER_NEXT(loop->it);
                NpyArray_ITER_NEXT(loop->rit);
                loop->bufptr[0] = loop->rit->dataptr;
                loop->index++;
            }
            break;

        case BUFFER_UFUNCLOOP:
            /* Reduceat
             * BUFFER -- misbehaved array or different types
             */
            /* fprintf(stderr, "BUFFERED..%d\n", loop->size); */
            while (loop->index < loop->size) {
                ptr = (npy_intp *)NpyArray_BYTES(ind);
                for (i = 0; i < nn; i++) {
                    if (loop->obj & NPY_UFUNC_OBJ_ISOBJECT) {
                        NpyInterface_INCREF(*((void **)loop->idptr));
                    }
                    memcpy(loop->bufptr[0], loop->idptr, loop->outsize);
                    n = 0;
                    mm = (i == nn - 1 ? NpyArray_DIM(arr, axis) - *ptr :
                          *(ptr + 1) - *ptr);
                    if (mm < 1) {
                        mm = 1;
                    }
                    loop->inptr = loop->it->dataptr + (*ptr)*loop->instrides;
                    while (n < mm) {
                        /* Copy up to loop->bufsize elements to buffer */
                        dptr = loop->buffer;
                        for (j = 0; j < loop->bufsize; j++, n++) {
                            if (n == mm) {
                                break;
                            }
                            NpyArray_DESCR(arr)->f->copyswap(dptr, loop->inptr,
                                                             loop->swap, NULL);
                            loop->inptr += loop->instrides;
                            dptr += loop->insize;
                        }
                        if (loop->cast) {
                            loop->cast(loop->buffer, loop->castbuf, j,
                                       NULL, NULL);
                        }
                        loop->bufptr[2] = loop->bufptr[0];
                        loop->function((char **)loop->bufptr, &j,
                                       loop->steps, loop->funcdata);
                        NPY_UFUNC_CHECK_ERROR(loop);
                        loop->bufptr[0] += j*loop->steps[0];
                    }
                    loop->bufptr[0] += NpyArray_STRIDE(loop->ret, axis);
                    ptr++;
                }
                NpyArray_ITER_NEXT(loop->it);
                NpyArray_ITER_NEXT(loop->rit);
                loop->bufptr[0] = loop->rit->dataptr;
                loop->index++;
            }

            /*
             * DECREF left-over objects if buffering was used.
             * It is needed when casting created new objects in
             * castbuf.  Intermediate copying into castbuf (via
             * loop->function) decref'd what was already there.

             * It's the final copy into the castbuf that needs a DECREF.
             */

            /* Only when casting needed and it is from a non-object array */
            if ((loop->obj & NPY_UFUNC_OBJ_ISOBJECT) && loop->cast &&
                (!NpyArray_ISOBJECT(arr))) {
                for (i=0; i<loop->bufsize; i++) {
                    NpyInterface_CLEAR(((void **)loop->castbuf)[i]);
                }
            }

            break;
    }
//    NPY_LOOP_END_THREADS;
    /* Hang on to this reference -- will be decref'd with loop */
    if (loop->retbase) {
        ret = NpyArray_BASE_ARRAY(loop->ret);
    }
    else {
        ret = loop->ret;
    }
    Npy_INCREF(ret);
    ufuncreduce_dealloc(loop);
    return ret;

fail:
//    NPY_LOOP_END_THREADS;
    if (loop) {
        ufuncreduce_dealloc(loop);
    }
    return NULL;
}

#if 0
/*
 * This code handles reduce, reduceat, and accumulate
 * (accumulate and reduce are special cases of the more general reduceat
 * but they are handled separately for speed)
 */
static PyObject *
PyUFunc_GenericReduction(PyUFuncObject *self, PyObject *args,
                         PyObject *kwds, int operation)
{
    int axis=0;
    PyArrayObject *mp, *ret = NULL;
    PyObject *op, *res = NULL;
    PyObject *obj_ind, *context;
    PyArrayObject *indices = NULL;
    PyArray_Descr *otype = NULL;
    PyArrayObject *out = NULL;
    static char *kwlist1[] = {"array", "axis", "dtype", "out", NULL};
    static char *kwlist2[] = {"array", "indices", "axis", "dtype", "out", NULL};
    static char *_reduce_type[] = {"reduce", "accumulate", "reduceat", NULL};

    if (self == NULL) {
        PyErr_SetString(PyExc_ValueError, "function not supported");
        return NULL;
    }
    if (PyUFunc_UFUNC(self)->core_enabled) {
        PyErr_Format(PyExc_RuntimeError,
                     "Reduction not defined on ufunc with signature");
        return NULL;
    }
    if (PyUFunc_UFUNC(self)->nin != 2) {
        PyErr_Format(PyExc_ValueError,
                     "%s only supported for binary functions",
                     _reduce_type[operation]);
        return NULL;
    }
    if (PyUFunc_UFUNC(self)->nout != 1) {
        PyErr_Format(PyExc_ValueError,
                     "%s only supported for functions " \
                     "returning a single value",
                     _reduce_type[operation]);
        return NULL;
    }

    if (operation == NPY_UFUNC_REDUCEAT) {
        PyArray_Descr *indtype;
        indtype = PyArray_DescrFromType(PyArray_INTP);
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO|iO&O&", kwlist2,
                                        &op, &obj_ind, &axis,
                                        PyArray_DescrConverter2,
                                        &otype,
                                        PyArray_OutputConverter,
                                        &out)) {
            Py_XDECREF(otype);
            return NULL;
        }

        indices = (PyArrayObject *)PyArray_FromAny(obj_ind, indtype,
                                                   1, 1, CARRAY, NULL);
        if (indices == NULL) {
            Py_XDECREF(otype);
            return NULL;
        }
    }
    else {
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|iO&O&", kwlist1,
                                        &op, &axis,
                                        PyArray_DescrConverter2,
                                        &otype,
                                        PyArray_OutputConverter,
                                        &out)) {
            Py_XDECREF(otype);
            return NULL;
        }
    }
    /* Ensure input is an array */
    if (!PyArray_Check(op) && !PyArray_IsScalar(op, Generic)) {
        context = Py_BuildValue("O(O)i", self, op, 0);
    }
    else {
        context = NULL;
    }

    mp = (PyArrayObject *)PyArray_FromAny(op, NULL, 0, 0, 0, context);
    Py_XDECREF(context);
    if (mp == NULL) {
        return NULL;
    }
    assert(PyArray_ISVALID(mp));

    /* Check to see if input is zero-dimensional */
    if (PyArray_NDIM(mp) == 0) {
        PyErr_Format(PyExc_TypeError, "cannot %s on a scalar",
                     _reduce_type[operation]);
        Py_XDECREF(otype);
        Py_DECREF(mp);
        return NULL;
    }
    /* Check to see that type (and otype) is not FLEXIBLE */
    if (PyArray_ISFLEXIBLE(mp) ||
        (otype && NpyTypeNum_ISFLEXIBLE(otype->descr->type_num))) {
        PyErr_Format(PyExc_TypeError,
                     "cannot perform %s with flexible type",
                     _reduce_type[operation]);
        Py_XDECREF(otype);
        Py_DECREF(mp);
        return NULL;
    }

    if (axis < 0) {
        axis += PyArray_NDIM(mp);
    }
    if (axis < 0 || axis >= PyArray_NDIM(mp)) {
        PyErr_SetString(PyExc_ValueError, "axis not in array");
        Py_XDECREF(otype);
        Py_DECREF(mp);
        return NULL;
    }
    /*
     * If out is specified it determines otype
     * unless otype already specified.
     */
    if (otype == NULL && out != NULL) {
        otype = PyArray_Descr_WRAP( PyArray_DESCR(out) );
        Py_INCREF(otype);
    }
    if (otype == NULL) {
        /*
         * For integer types --- make sure at least a long
         * is used for add and multiply reduction to avoid overflow
         */
        int typenum = PyArray_TYPE(mp);
        if ((typenum < NPY_FLOAT)
            && ((strcmp(PyUFunc_UFUNC(self)->name,"add") == 0)
                || (strcmp(PyUFunc_UFUNC(self)->name,"multiply") == 0))) {
                if (NpyTypeNum_ISBOOL(typenum)) {
                    typenum = PyArray_LONG;
                }
                else if ((size_t)PyArray_ITEMSIZE(mp) < sizeof(long)) {
                    if (NpyTypeNum_ISUNSIGNED(typenum)) {
                        typenum = PyArray_ULONG;
                    }
                    else {
                        typenum = PyArray_LONG;
                    }
                }
            }
        otype = PyArray_DescrFromType(typenum);
    }


    switch(operation) {
        case NPY_UFUNC_REDUCE:
            ret = (PyArrayObject *)PyUFunc_Reduce(self, mp, out, axis,
                                                  otype->descr->type_num);
            break;
        case NPY_UFUNC_ACCUMULATE:
            ret = (PyArrayObject *)PyUFunc_Accumulate(self, mp, out, axis,
                                                      otype->descr->type_num);
            break;
        case NPY_UFUNC_REDUCEAT:
            ret = (PyArrayObject *)PyUFunc_Reduceat(self, mp, indices, out,
                                                    axis,
                                                    otype->descr->type_num);
            Py_DECREF(indices);
            break;
    }
    Py_DECREF(mp);
    Py_DECREF(otype);
    if (ret == NULL) {
        return NULL;
    }
    if (Py_TYPE(op) != Py_TYPE(ret)) {
        res = PyObject_CallMethod(op, "__array_wrap__", "O", ret);
        if (res == NULL) {
            PyErr_Clear();
        }
        else if (res == Py_None) {
            Py_DECREF(res);
        }
        else {
            Py_DECREF(ret);
            return res;
        }
    }
    return PyArray_Return(ret);
}

#endif



int
NpyUFunc_RegisterLoopForType(NpyUFuncObject *ufunc,
                             int usertype,
                             NpyUFuncGenericFunction function,
                             int *arg_types,
                             void *data)
{
    NpyArray_Descr *descr;
    NpyUFunc_Loop1d *funcdata, *current = NULL;
    int i;
    int *newtypes=NULL;

    descr = NpyArray_DescrFromType(usertype);
    if ((usertype < NPY_USERDEF) || (descr==NULL)) {
        NpyErr_SetString(NpyExc_TypeError, "unknown user-defined type");
        return -1;
    }
    Npy_DECREF(descr);

    if (ufunc->userloops == NULL) {
        ufunc->userloops = npy_create_userloops_table();
    }
    funcdata = malloc(sizeof(NpyUFunc_Loop1d));
    if (funcdata == NULL) {
        goto fail;
    }
    newtypes = malloc(sizeof(int)*ufunc->nargs);
    if (newtypes == NULL) {
        goto fail;
    }
    if (arg_types != NULL) {
        for (i = 0; i < ufunc->nargs; i++) {
            newtypes[i] = arg_types[i];
        }
    }
    else {
        for (i = 0; i < ufunc->nargs; i++) {
            newtypes[i] = usertype;
        }
    }

    funcdata->func = function;
    funcdata->arg_types = newtypes;
    funcdata->data = data;
    funcdata->next = NULL;

    /* Get entry for this user-defined type*/
    current = (NpyUFunc_Loop1d *)NpyDict_Get(ufunc->userloops,
                                             (void *)(npy_intp)usertype);
    /* If it's not there, then make one and return. */
    if (NULL == current) {
        NpyDict_Put(ufunc->userloops, (void *)(npy_intp)usertype, funcdata);
        return 0;
    }
    else {
        NpyUFunc_Loop1d *prev = NULL;
        int cmp = 1;
        /*
         * There is already at least 1 loop. Place this one in
         * lexicographic order.  If the next one signature
         * is exactly like this one, then just replace.
         * Otherwise insert.
         */
        while (current != NULL) {
            cmp = cmp_arg_types(current->arg_types, newtypes, ufunc->nargs);
            if (cmp >= 0) {
                break;
            }
            prev = current;
            current = current->next;
        }
        if (cmp == 0) {
            /* just replace it with new function */
            current->func = function;
            current->data = data;
            free(newtypes);
            free(funcdata);
        }
        else {
            /*
             * insert it before the current one by hacking the internals
             * of cobject to replace the function pointer --- can't use
             * CObject API because destructor is set.
             */
            funcdata->next = current;
            if (prev == NULL) {
                /* place this at front */
                NpyDict_ForceValue(ufunc->userloops,
                                   (void *)(npy_intp)usertype, funcdata);
            }
            else {
                prev->next = funcdata;
            }
        }
    }
    return 0;

fail:
    free(funcdata);
    free(newtypes);
    if (!NpyErr_Occurred()) NpyErr_NoMemory();
    return -1;
}


NpyUFuncObject *
NpyUFunc_FromFuncAndDataAndSignature(NpyUFuncGenericFunction *func,
                                     void **data, char *types, int ntypes,
                                     int nin, int nout, int identity,
                                     char *name, char *doc,
                                     int check_return, const char *signature)
{
    NpyUFuncObject *self;

    self = (NpyUFuncObject *)malloc(sizeof(NpyUFuncObject));
    if (NULL == self) {
        return NULL;
    }
    NpyObject_Init(self, &NpyUFunc_Type);
    self->magic_number = NPY_VALID_MAGIC;

    self->nin = nin;
    self->nout = nout;
    self->nargs = nin+nout;
    self->identity = identity;

    self->functions = func;
    self->data = data;
    self->types = types;
    self->ntypes = ntypes;
    self->check_return = check_return;
    self->ptr = NULL;
    self->userloops=NULL;

    if (name == NULL) {
        self->name = "?";
    }
    else {
        self->name = name;
    }
    if (doc == NULL) {
        self->doc = "NULL";
    }
    else {
        self->doc = doc;
    }

    /* generalized ufunc */
    self->core_enabled = 0;
    self->core_num_dim_ix = 0;
    self->core_num_dims = NULL;
    self->core_dim_ixs = NULL;
    self->core_offsets = NULL;
    self->core_signature = NULL;
    if (signature != NULL) {
        if (0 != _parse_signature(self, signature)) {
            Npy_DECREF(self);
            return NULL;
        }
    }
    return self;
}


NpyUFuncObject *
NpyUFunc_FromFuncAndData(NpyUFuncGenericFunction *func, void **data,
                         char *types, int ntypes,
                         int nin, int nout, int identity,
                         char *name, char *doc, int check_return)
{
    return NpyUFunc_FromFuncAndDataAndSignature(func, data, types, ntypes,
                                                nin, nout, identity, name,
                                                doc, check_return, NULL);
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
    loop->magic_number = NPY_VALID_MAGIC;

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
    loop->leftover = 0;

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
                 int *rtypenums, npy_prepare_outputs_func prepare,
                 void *prepare_data)
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
        if (NULL != mps[i] && (NpyArray_NDIM(mps[i]) != out_nd ||
                               !NpyArray_CompareLists(NpyArray_DIMS(mps[i]),
                                                      out_dims, out_nd))) {
            NpyErr_SetString(NpyExc_ValueError, "invalid return array shape");
            Npy_DECREF(mps[i]);
            mps[i] = NULL;
            return -1;
        }
        if (NULL != mps[i] && !NpyArray_ISWRITEABLE(mps[i])) {
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
                    new = NpyArray_FromArray(mps[i], ntype,
                             NPY_FORCECAST | NPY_ALIGNED | NPY_UPDATEIFCOPY);
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
        if (!(loop->obj & NPY_UFUNC_OBJ_ISOBJECT)
            && ((NpyArray_TYPE(mps[i]) == NPY_OBJECT)
                || (arg_types[i] == NPY_OBJECT))) {
                loop->obj = NPY_UFUNC_OBJ_ISOBJECT | NPY_UFUNC_OBJ_NEEDS_API;
            }
        if (!(loop->obj & NPY_UFUNC_OBJ_NEEDS_API)
            && ((NpyArray_TYPE(mps[i]) == NPY_DATETIME)
                || (NpyArray_TYPE(mps[i]) == NPY_TIMEDELTA)
                || (arg_types[i] == NPY_DATETIME)
                || (arg_types[i] == NPY_TIMEDELTA))) {
                loop->obj = NPY_UFUNC_OBJ_NEEDS_API;
            }
    }

    if (self->core_enabled && (loop->obj & NPY_UFUNC_OBJ_ISOBJECT)) {
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
                if (NpyArray_NDIM(mps[i]) != 0 &&
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
                loop->steps[i] = NpyArray_STRIDE(mps[i],
                                                 NpyArray_NDIM(mps[i])-1);
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
                    loop->cast[i] = NpyArray_GetCastFunc(
                        descr, NpyArray_DESCR(mps[i])->type_num);
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
        if (loop->obj & NPY_UFUNC_OBJ_ISOBJECT) {
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
            if (!loop->objfunc && (loop->obj & NPY_UFUNC_OBJ_ISOBJECT)) {
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





static NpyUFuncReduceObject *
construct_reduce(NpyUFuncObject *self, NpyArray **arr, NpyArray *out,
                 int axis, int otype, int operation, npy_intp ind_size,
                 char *str, int bufsize, int errormask, void *errobj)
{
    NpyUFuncReduceObject *loop;
    NpyArray *idarr;
    NpyArray *aar;
    npy_intp loop_i[NPY_MAXDIMS], outsize = 0;
    int arg_types[3];
    NPY_SCALARKIND scalars[3] = { NPY_NOSCALAR, NPY_NOSCALAR,
        NPY_NOSCALAR };
    int i, j, nd;
    int flags;

    assert(NPY_VALID_MAGIC == self->magic_number);

    /* Reduce type is the type requested of the input during reduction */
    if (self->core_enabled) {
        NpyErr_SetString(NpyExc_RuntimeError,
                         "construct_reduce not allowed on ufunc with signature");
        return NULL;
    }

    nd = NpyArray_NDIM(*arr);
    arg_types[0] = otype;
    arg_types[1] = otype;
    arg_types[2] = otype;
    if ((loop = malloc(sizeof(NpyUFuncReduceObject))) == NULL) {
        NpyErr_NoMemory();
        return loop;
    }
    loop->magic_number = NPY_VALID_MAGIC;

    loop->retbase = 0;
    loop->swap = 0;
    loop->index = 0;
    loop->ufunc = self;
    Npy_INCREF(self);
    loop->cast = NULL;
    loop->buffer = NULL;
    loop->ret = NULL;
    loop->it = NULL;
    loop->rit = NULL;
    loop->errobj = NULL;
    loop->first = 1;
    loop->decref_arr = NULL;
    loop->N = NpyArray_DIM(*arr,axis);
    loop->instrides = NpyArray_STRIDE(*arr, axis);
    loop->bufsize = bufsize;
    loop->errormask = errormask;
    loop->errobj = errobj;
    if (select_types(loop->ufunc, arg_types, &(loop->function),
                     &(loop->funcdata), scalars, NULL) == -1) {
        goto fail;
    }
    /*
     * output type may change -- if it does
     * reduction is forced into that type
     * and we need to select the reduction function again
     */
    if (otype != arg_types[2]) {
        otype = arg_types[2];
        arg_types[0] = otype;
        arg_types[1] = otype;
        if (select_types(loop->ufunc, arg_types, &(loop->function),
                         &(loop->funcdata), scalars, NULL) == -1) {
            goto fail;
        }
    }

    /* Make copy if misbehaved or not otype for small arrays */
    if (_create_reduce_copy(loop, arr, otype) < 0) {
        goto fail;
    }
    aar = *arr;

    if (loop->N == 0) {
        loop->meth = ZERO_EL_REDUCELOOP;
    }
    else if (NpyArray_ISBEHAVED_RO(aar) && (otype == NpyArray_TYPE(aar))) {
        if (loop->N == 1) {
            loop->meth = ONE_EL_REDUCELOOP;
        }
        else {
            loop->meth = NOBUFFER_UFUNCLOOP;
            loop->steps[1] = NpyArray_STRIDE(aar, axis);
            loop->N -= 1;
        }
    }
    else {
        loop->meth = BUFFER_UFUNCLOOP;
        loop->swap = !(NpyArray_ISNOTSWAPPED(aar));
    }

    /* Determine if object arrays are involved */
    if (otype == NPY_OBJECT || NpyArray_TYPE(aar) == NPY_OBJECT) {
        loop->obj = NPY_UFUNC_OBJ_ISOBJECT | NPY_UFUNC_OBJ_NEEDS_API;
    }
    else if ((otype == NPY_DATETIME)
             || (NpyArray_TYPE(aar) == NPY_DATETIME)
             || (otype == NPY_TIMEDELTA)
             || (NpyArray_TYPE(aar) == NPY_TIMEDELTA)) {
        loop->obj = NPY_UFUNC_OBJ_NEEDS_API;
    } else {
        loop->obj = 0;
    }
    if ((loop->meth == ZERO_EL_REDUCELOOP)
        || ((operation == NPY_UFUNC_REDUCEAT)
            && (loop->meth == BUFFER_UFUNCLOOP))) {
            idarr = _getidentity(self, otype, str);
            if (idarr == NULL) {
                goto fail;
            }

            if (NpyArray_ITEMSIZE(idarr) > NPY_UFUNC_MAXIDENTITY) {
                char buf[256];

                NpyOS_snprintf(buf, 256, "UFUNC_MAXIDENTITY (%d) is too small"\
                               "(needs to be at least %d)",
                               NPY_UFUNC_MAXIDENTITY, NpyArray_ITEMSIZE(idarr));
                NpyErr_SetString(NpyExc_RuntimeError, buf);

                Npy_DECREF(idarr);
                goto fail;
            }
            memcpy(loop->idptr, NpyArray_BYTES(idarr), NpyArray_ITEMSIZE(idarr));
            Npy_DECREF(idarr);
        }

    /* Construct return array */
    flags = NPY_CARRAY | NPY_UPDATEIFCOPY | NPY_FORCECAST;
    switch(operation) {
        case NPY_UFUNC_REDUCE:
            for (j = 0, i = 0; i < nd; i++) {
                if (i != axis) {
                    loop_i[j++] = NpyArray_DIM(aar, i);
                }
            }
            if (out == NULL) {
                loop->ret = NpyArray_New(NULL, NpyArray_NDIM(aar)-1, loop_i,
                                         otype, NULL, NULL, 0, 0,
                                         Npy_INTERFACE(aar));
            }
            else {
                outsize = NpyArray_MultiplyList(loop_i, NpyArray_NDIM(aar) - 1);
            }
            break;
        case NPY_UFUNC_ACCUMULATE:
            if (out == NULL) {
                loop->ret = NpyArray_New(NULL, NpyArray_NDIM(aar),
                                         NpyArray_DIMS(aar),
                                         otype, NULL, NULL, 0, 0,
                                         Npy_INTERFACE(aar));
            }
            else {
                outsize = NpyArray_MultiplyList(NpyArray_DIMS(aar),
                                                NpyArray_NDIM(aar));
            }
            break;
        case NPY_UFUNC_REDUCEAT:
            memcpy(loop_i, NpyArray_DIMS(aar), nd*sizeof(npy_intp));
            /* Index is 1-d array */
            loop_i[axis] = ind_size;
            if (out == NULL) {
                loop->ret = NpyArray_New(NULL, NpyArray_NDIM(aar), loop_i, otype,
                                         NULL, NULL, 0, 0, Npy_INTERFACE(aar));
            }
            else {
                outsize = NpyArray_MultiplyList(loop_i, NpyArray_NDIM(aar));
            }
            if (ind_size == 0) {
                loop->meth = ZERO_EL_REDUCELOOP;
                return loop;
            }
            if (loop->meth == ONE_EL_REDUCELOOP) {
                loop->meth = NOBUFFER_REDUCELOOP;
            }
            break;
    }
    if (out) {
        if (NpyArray_SIZE(out) != outsize) {
            NpyErr_SetString(NpyExc_ValueError,
                             "wrong shape for output");
            goto fail;
        }
        loop->ret = NpyArray_FromArray(out, NpyArray_DescrFromType(otype),
                                       flags);
        if (loop->ret && loop->ret != out) {
            loop->retbase = 1;
        }
    }
    if (loop->ret == NULL) {
        goto fail;
    }
    loop->insize = NpyArray_ITEMSIZE(aar);
    loop->outsize = NpyArray_ITEMSIZE(loop->ret);
    loop->bufptr[0] = NpyArray_BYTES(loop->ret);

    if (loop->meth == ZERO_EL_REDUCELOOP) {
        loop->size = NpyArray_SIZE(loop->ret);
        return loop;
    }

    loop->it = NpyArray_IterNew(aar);
    if (loop->it == NULL) {
        return NULL;
    }
    if (loop->meth == ONE_EL_REDUCELOOP) {
        loop->size = loop->it->size;
        return loop;
    }

    /*
     * Fix iterator to loop over correct dimension
     * Set size in axis dimension to 1
     */
    loop->it->contiguous = 0;
    loop->it->size /= (loop->it->dims_m1[axis]+1);
    loop->it->dims_m1[axis] = 0;
    loop->it->backstrides[axis] = 0;
    loop->size = loop->it->size;
    if (operation == NPY_UFUNC_REDUCE) {
        loop->steps[0] = 0;
    }
    else {
        loop->rit = NpyArray_IterNew(loop->ret);
        if (loop->rit == NULL) {
            return NULL;
        }
        /*
         * Fix iterator to loop over correct dimension
         * Set size in axis dimension to 1
         */
        loop->rit->contiguous = 0;
        loop->rit->size /= (loop->rit->dims_m1[axis] + 1);
        loop->rit->dims_m1[axis] = 0;
        loop->rit->backstrides[axis] = 0;

        if (operation == NPY_UFUNC_ACCUMULATE) {
            loop->steps[0] = NpyArray_STRIDE(loop->ret, axis);
        }
        else {
            loop->steps[0] = 0;
        }
    }
    loop->steps[2] = loop->steps[0];
    loop->bufptr[2] = loop->bufptr[0] + loop->steps[2];
    if (loop->meth == BUFFER_UFUNCLOOP) {
        int _size;

        loop->steps[1] = loop->outsize;
        if (otype != NpyArray_TYPE(aar)) {
            _size=loop->bufsize*(loop->outsize + NpyArray_ITEMSIZE(aar));
            loop->buffer = NpyDataMem_NEW(_size);
            if (loop->buffer == NULL) {
                goto fail;
            }
            if (loop->obj & NPY_UFUNC_OBJ_ISOBJECT) {
                memset(loop->buffer, 0, _size);
            }
            loop->castbuf = loop->buffer + loop->bufsize*NpyArray_ITEMSIZE(aar);
            loop->bufptr[1] = loop->castbuf;
            loop->cast = NpyArray_GetCastFunc(NpyArray_DESCR(aar), otype);
            if (loop->cast == NULL) {
                goto fail;
            }
        }
        else {
            _size = loop->bufsize * loop->outsize;
            loop->buffer = NpyDataMem_NEW(_size);
            if (loop->buffer == NULL) {
                goto fail;
            }
            if (loop->obj & NPY_UFUNC_OBJ_ISOBJECT) {
                memset(loop->buffer, 0, _size);
            }
            loop->bufptr[1] = loop->buffer;
        }
    }
    NpyUFunc_clearfperr();
    return loop;

fail:
    ufuncreduce_dealloc(loop);
    return NULL;
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



NpyUFuncObject *
npy_ufunc_frompyfunc(int nin, int nout, char *fname, size_t fname_len,
                     NpyUFuncGenericFunction *gen_funcs, void *function) {
    NpyUFuncObject *self;
    NpyUFunc_FuncData *fdata;
    char *str;
    int i;
    int offset[2];

    self = (NpyUFuncObject *)malloc(sizeof(NpyUFuncObject));
    if (NULL == self) {
        return NULL;
    }
    NpyObject_Init(self, &NpyUFunc_Type);
    self->magic_number = NPY_VALID_MAGIC;

    self->userloops = NULL;
    self->nin = nin;
    self->nout = nout;
    self->nargs = nin + nout;
    self->identity = NpyUFunc_None;
    self->functions = gen_funcs;
    self->ntypes = 1;
    self->check_return = 0;

    /* generalized ufunc */
    self->core_enabled = 0;
    self->core_num_dim_ix = 0;
    self->core_num_dims = NULL;
    self->core_dim_ixs = NULL;
    self->core_offsets = NULL;
    self->core_signature = NULL;

    /*
     * self->ptr holds a pointer for enough memory for
     * self->data[0] (fdata)
     * self->data
     * self->name
     * self->types
     *
     * To be safest, all of these need their memory aligned on void * pointers
     * Therefore, we may need to allocate extra space.
     */
    offset[0] = sizeof(NpyUFunc_FuncData);
    i = (sizeof(NpyUFunc_FuncData) % sizeof(void *));
    if (i) {
        offset[0] += (sizeof(void *) - i);
    }
    offset[1] = self->nargs;
    i = (self->nargs % sizeof(void *));
    if (i) {
        offset[1] += (sizeof(void *)-i);
    }
    self->ptr = malloc(offset[0] + offset[1] + sizeof(void *) +
                       (fname_len + 14));
    if (NULL == self->ptr) {
        return NULL;
    }

    fdata = (NpyUFunc_FuncData *)(self->ptr);
    fdata->nin = nin;
    fdata->nout = nout;
    fdata->callable = function;

    self->data = (void **)(((char *)self->ptr) + offset[0]);
    self->data[0] = (void *)fdata;
    self->types = (char *)self->data + sizeof(void *);
    for (i = 0; i < self->nargs; i++) {
        self->types[i] = NPY_OBJECT;
    }
    str = self->types + offset[1];
    memcpy(str, fname, fname_len);
    memcpy(str+fname_len, " (vectorized)", 14);
    self->name = str;

    /* Do a better job someday */
    self->doc = "dynamic ufunc based on a python function";

    return self;
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


void
npy_ufunc_dealloc(NpyUFuncObject *self)
{
    /* TODO: Ready to move */
    if (self->core_num_dims) {
        free(self->core_num_dims);
    }
    if (self->core_dim_ixs) {
        free(self->core_dim_ixs);
    }
    if (self->core_offsets) {
        free(self->core_offsets);
    }
    if (self->core_signature) {
        free(self->core_signature);
    }
    if (self->ptr) {
        free(self->ptr);
    }
    if (NULL != self->userloops) {
        NpyDict_Destroy(self->userloops);
    }
    self->magic_number = NPY_INVALID_MAGIC;
    free(self);
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


static void
ufuncreduce_dealloc(NpyUFuncReduceObject *self)
{
    if (self->ufunc) {
        Npy_XDECREF(self->it);
        Npy_XDECREF(self->rit);
        Npy_XDECREF(self->ret);
        NpyInterface_DECREF(self->errobj);
        Npy_XDECREF(self->decref_arr);
        if (self->buffer) {
            NpyDataMem_FREE(self->buffer);
        }
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
                new = NpyArray_FromArray(mps[i], ntype,
                                         NPY_FORCECAST | NPY_ALIGNED);
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


static NpyArray *
_getidentity(NpyUFuncObject *self, int otype, char *str)
{
    NpyArray *arr;
    NpyArray_Descr *descr, *indescr;
    unsigned char identity;
    NpyArray_VectorUnaryFunc *castfunc;

    if (self->identity == NpyUFunc_None) {
        char buf[256];

        NpyOS_snprintf(buf, 256,
                       "zero-size array to ufunc.%s without identity", str);
        NpyErr_SetString(NpyExc_ValueError, buf);

        return NULL;
    }

    /* Get the identity as an unsigned char. */
    if (self->identity == NpyUFunc_One) {
        identity = 1;
    } else {
        identity = 0;
    }

    /* Build the output 0-d array. */
    descr = NpyArray_DescrFromType(otype);
    if (descr == NULL) {
        return NULL;
    }
    arr = NpyArray_Alloc(descr, 0, NULL, NPY_FALSE, NULL);
    if (arr == NULL) {
        return NULL;
    }

    indescr = NpyArray_DescrFromType(NPY_UBYTE);
    assert(indescr != NULL);

    castfunc = NpyArray_GetCastFunc(indescr, otype);
    Npy_DECREF(indescr);
    if (castfunc == NULL) {
        NpyErr_SetString(NpyExc_ValueError,
                         "Can't cast identity to output type.");
        return NULL;
    }

    /* Use the castfunc to fill in the array. */
    castfunc(&identity, arr->data, 1, NULL, arr);

    return arr;
}


/* return 1 if arg1 > arg2, 0 if arg1 == arg2, and -1 if arg1 < arg2 */
static int
cmp_arg_types(int *arg1, int *arg2, int n)
{
    for (; n > 0; n--, arg1++, arg2++) {
        if (NpyArray_EquivTypenums(*arg1, *arg2)) {
            continue;
        }
        if (NpyArray_CanCastSafely(*arg1, *arg2)) {
            return -1;
        }
        return 1;
    }
    return 0;
}


static int
_create_reduce_copy(NpyUFuncReduceObject *loop, NpyArray **arr, int rtype)
{
    npy_intp maxsize;
    NpyArray *new;
    NpyArray_Descr *ntype;

    maxsize = NpyArray_SIZE(*arr);
    if (maxsize < loop->bufsize) {
        if (!(NpyArray_ISBEHAVED_RO(*arr))
            || NpyArray_TYPE(*arr) != rtype) {
            ntype = NpyArray_DescrFromType(rtype);

            new = NpyArray_FromArray(*arr, ntype, NPY_FORCECAST | NPY_ALIGNED);
            if (new == NULL) {
                return -1;
            }
            *arr = new;
            loop->decref_arr = new;
        }
    }

    /*
     * Don't decref *arr before re-assigning
     * because it was not going to be DECREF'd anyway.
     *
     * If a copy is made, then the copy will be removed
     * on deallocation of the loop structure by setting
     * loop->decref_arr.
     */
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
        tmp_dims[loop->iter->nd + i] = loop->core_dim_sizes[
            1 + ufunc->core_dim_ixs[ufunc->core_offsets[iarg] + i]];
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


/* Return the position of next non-white-space char in the string */
static int
_next_non_white_space(const char* str, int offset)
{
    int ret = offset;
    while (str[ret] == ' ' || str[ret] == '\t') {
        ret++;
    }
    return ret;
}


static int
_is_alpha_underscore(char ch)
{
    return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch == '_';
}


static int
_is_alnum_underscore(char ch)
{
    return _is_alpha_underscore(ch) || (ch >= '0' && ch <= '9');
}


/*
 * Return the ending position of a variable name
 */
static int
_get_end_of_name(const char* str, int offset)
{
    int ret = offset;
    while (_is_alnum_underscore(str[ret])) {
        ret++;
    }
    return ret;
}


/*
 * Returns 1 if the dimension names pointed by s1 and s2 are the same,
 * otherwise returns 0.
 */
static int
_is_same_name(const char* s1, const char* s2)
{
    while (_is_alnum_underscore(*s1) && _is_alnum_underscore(*s2)) {
        if (*s1 != *s2) {
            return 0;
        }
        s1++;
        s2++;
    }
    return !_is_alnum_underscore(*s1) && !_is_alnum_underscore(*s2);
}


/*
 * Sets core_num_dim_ix, core_num_dims, core_dim_ixs, core_offsets,
 * and core_signature in PyUFuncObject "self".  Returns 0 unless an
 * error occured.
 */
static int
_parse_signature(NpyUFuncObject *self, const char *signature)
{
    size_t len;
    char const **var_names;
    int nd = 0;             /* number of dimension of the current argument */
    int cur_arg = 0;        /* index into core_num_dims&core_offsets */
    int cur_core_dim = 0;   /* index into core_dim_ixs */
    int i = 0;
    char *parse_error = NULL;

    if (signature == NULL) {
        NpyErr_SetString(NpyExc_RuntimeError,
                         "_parse_signature with NULL signature");
        return -1;
    }

    len = strlen(signature);
    self->core_signature = malloc(sizeof(char) * (len+1));
    if (self->core_signature) {
        strcpy(self->core_signature, signature);
    }
    /* Allocate sufficient memory to store pointers to all dimension names */
    var_names = malloc(sizeof(char const*) * len);
    if (var_names == NULL) {
        NpyErr_NoMemory();
        return -1;
    }

    self->core_enabled = 1;
    self->core_num_dim_ix = 0;
    self->core_num_dims = malloc(sizeof(int) * self->nargs);
    self->core_dim_ixs = malloc(sizeof(int) * len); /* shrink this later */
    self->core_offsets = malloc(sizeof(int) * self->nargs);
    if (self->core_num_dims == NULL || self->core_dim_ixs == NULL
        || self->core_offsets == NULL) {
        NpyErr_NoMemory();
        goto fail;
    }

    i = _next_non_white_space(signature, 0);
    while (signature[i] != '\0') {
        /* loop over input/output arguments */
        if (cur_arg == self->nin) {
            /* expect "->" */
            if (signature[i] != '-' || signature[i+1] != '>') {
                parse_error = "expect '->'";
                goto fail;
            }
            i = _next_non_white_space(signature, i + 2);
        }

        /*
         * parse core dimensions of one argument,
         * e.g. "()", "(i)", or "(i,j)"
         */
        if (signature[i] != '(') {
            parse_error = "expect '('";
            goto fail;
        }
        i = _next_non_white_space(signature, i + 1);
        while (signature[i] != ')') {
            /* loop over core dimensions */
            int j = 0;
            if (!_is_alpha_underscore(signature[i])) {
                parse_error = "expect dimension name";
                goto fail;
            }
            while (j < self->core_num_dim_ix) {
                if (_is_same_name(signature+i, var_names[j])) {
                    break;
                }
                j++;
            }
            if (j >= self->core_num_dim_ix) {
                var_names[j] = signature+i;
                self->core_num_dim_ix++;
            }
            self->core_dim_ixs[cur_core_dim] = j;
            cur_core_dim++;
            nd++;
            i = _get_end_of_name(signature, i);
            i = _next_non_white_space(signature, i);
            if (signature[i] != ',' && signature[i] != ')') {
                parse_error = "expect ',' or ')'";
                goto fail;
            }
            if (signature[i] == ',')
            {
                i = _next_non_white_space(signature, i + 1);
                if (signature[i] == ')') {
                    parse_error = "',' must not be followed by ')'";
                    goto fail;
                }
            }
        }
        self->core_num_dims[cur_arg] = nd;
        self->core_offsets[cur_arg] = cur_core_dim-nd;
        cur_arg++;
        nd = 0;

        i = _next_non_white_space(signature, i + 1);
        if (cur_arg != self->nin && cur_arg != self->nargs) {
            /*
             * The list of input arguments (or output arguments) was
             * only read partially
             */
            if (signature[i] != ',') {
                parse_error = "expect ','";
                goto fail;
            }
            i = _next_non_white_space(signature, i + 1);
        }
    }
    if (cur_arg != self->nargs) {
        parse_error = "incomplete signature: not all arguments found";
        goto fail;
    }
    self->core_dim_ixs = realloc(self->core_dim_ixs,
                                 sizeof(int)*cur_core_dim);
    /* check for trivial core-signature, e.g. "(),()->()" */
    if (cur_core_dim == 0) {
        self->core_enabled = 0;
    }
    free((void*)var_names);
    return 0;

fail:
    free((void*)var_names);
    if (parse_error) {
        char *buf = malloc(sizeof(char) * (len + 200));
        if (buf) {
            sprintf(buf, "%s at position %d in \"%s\"",
                    parse_error, i, signature);
            NpyErr_SetString(NpyExc_ValueError, signature);
            free(buf);
        }
        else {
            NpyErr_NoMemory();
        }
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
NpyUFunc_SetFpErrFuncs(void (*state)(char *, int *, int *, void **),
                       void (*handler)(int, void *, int, int *))
{
    fp_error_state = state;
    fp_error_handler = handler;
}


int
NpyUFunc_getfperr(void)
{
    int retstatus;
    NPY_UFUNC_CHECK_STATUS(retstatus);
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

static int
compare_ints(const void *a, const void *b)
{
    if (a < b ) return -1;
    else if ( a > b ) return 1;
    return 0;
}

static int
hash_int(const void *a)
{
    return (int)a;  /* Size change is safe - just a hash function */
}

/* This frees the linked-list structure when the CObject is destroyed (removed
   from the internal dictionary) */
static void
free_loop1d_list(NpyUFunc_Loop1d *data)
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
    NpyDict_SetKeyComparisonFunction(
        new, (int (*)(const void *, const void *))compare_ints);
    NpyDict_SetHashFunction(new, (int (*)(const void *))hash_int);
    NpyDict_SetDeallocationFunctions(new, NULL,
                                     (void (*)(void *))free_loop1d_list);
    return new;
}
