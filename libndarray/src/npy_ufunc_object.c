/*
 * Python Universal Functions Object -- CPython-independent portion
 *
 */
#include "npy_config.h"

#include "npy_dict.h"
#include "ufuncobject.h"  // TODO: Fix this
#include "npy_api.h"
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







int NpyUFunc_GenericFunction(NpyUFuncObject *self, int nargs, NpyArray **mps,
                             int bufsize, int errormask, void *errobj, int originalArgWasObjArray)
{
    NpyUFuncLoopObject *loop;
    Py_ssize_t nargs;
    int typenumbuf[NPY_MAXARGS];
    int *rtypenums;
    int res;
    int i;
    char* name = self->name ? self->name : "";


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
                           (prepare_outputs_func)prepare_outputs,
                           (void*) args);

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
            intp laststrides[NPY_MAXARGS];
            int fastmemcpy[NPY_MAXARGS];
            int *needbuffer = loop->needbuffer;
            intp index=loop->iter->index, size=loop->iter->size;
            int bufsize;
            intp bufcnt;
            int copysizes[NPY_MAXARGS];
            char **bufptr = loop->bufptr;
            char **buffer = loop->buffer;
            char **castbuf = loop->castbuf;
            intp *steps = loop->steps;
            char *tptr[NPY_MAXARGS];
            int ninnerloops = loop->ninnerloops;
            Bool pyobject[NPY_MAXARGS];
            int datasize[NPY_MAXARGS];
            int j, k, stopcondition;
            char *myptr1, *myptr2;

            for (i = 0; i <self->nargs; i++) {
                copyswapn[i] = PyArray_DESCR(mps[i])->f->copyswapn;
                mpselsize[i] = PyArray_DESCR(mps[i])->elsize;
                pyobject[i] = ((loop->obj & UFUNC_OBJ_ISOBJECT)
                               && (PyArray_TYPE(mps[i]) == PyArray_OBJECT));
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
                                         (intp) datasize[i], 1,
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
                                NpyInterface_Decref(*((void **)myptr1));
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
                                NpyInterface_Decref(*((void **)castbuf[i]));
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
                                    NpyInterface_Decref(*objptr);
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
            intp *steps = loop->steps;
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
                        NpyInterface_Decref(*((void **)castbuf[i]));
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
                            NpyInterface_Decref(*objptr);
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

