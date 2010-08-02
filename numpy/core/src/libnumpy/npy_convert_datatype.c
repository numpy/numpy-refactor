/*
 *  npy_convert_datatype.c -
 *
 */

#define _MULTIARRAYMODULE
#define PY_SSIZE_T_CLEAN
#include "npy_config.h"
#include "numpy/numpy_api.h"


/*
 * Reference counts:
 * copyswapn is used which increases and decreases reference counts for OBJECT
 * arrays.
 * All that needs to happen is for any reference counts in the buffers to be
 * decreased when completely finished with the buffers.
 *
 * buffers[0] is the destination
 * buffers[1] is the source
 */

static void
_strided_buffered_cast(char *dptr, npy_intp dstride, int delsize, int dswap,
                       NpyArray_CopySwapNFunc *dcopyfunc,
                       char *sptr, npy_intp sstride, int selsize, int sswap,
                       NpyArray_CopySwapNFunc *scopyfunc,
                       npy_intp N, char **buffers, int bufsize,
                       NpyArray_VectorUnaryFunc *castfunc,
                       NpyArray *dest, NpyArray *src)
{
    int i;
    if (N <= bufsize) {
        /*
         * 1. copy input to buffer and swap
         * 2. cast input to output
         * 3. swap output if necessary and copy from output buffer
         */
        scopyfunc(buffers[1], selsize, sptr, sstride, N, sswap, src);
        castfunc(buffers[1], buffers[0], N, src, dest);
        dcopyfunc(dptr, dstride, buffers[0], delsize, N, dswap, dest);
        return;
    }

    /* otherwise we need to divide up into bufsize pieces */
    i = 0;
    while (N > 0) {
        int newN = NPY_MIN(N, bufsize);

        _strided_buffered_cast(dptr+i*dstride, dstride, delsize,
                               dswap, dcopyfunc,
                               sptr+i*sstride, sstride, selsize,
                               sswap, scopyfunc,
                               newN, buffers, bufsize, castfunc, dest, src);
        i += newN;
        N -= bufsize;
    }
    return;
}



static int
_broadcast_cast(NpyArray *out, NpyArray *in,
                NpyArray_VectorUnaryFunc *castfunc, int iswap, int oswap)
{
    int delsize, selsize, maxaxis, i, N;
    NpyArrayMultiIterObject *multi;
    npy_intp maxdim, ostrides, istrides;
    char *buffers[2];
    NpyArray_CopySwapNFunc *ocopyfunc, *icopyfunc;
    char *obptr;
    NPY_BEGIN_THREADS_DEF;

    delsize = NpyArray_ITEMSIZE(out);
    selsize = NpyArray_ITEMSIZE(in);
    multi = NpyArray_MultiIterFromArrays(NULL, 0, 2, out, in);
    if (multi == NULL) {
        return -1;
    }

    if (multi->size != NpyArray_SIZE(out)) {
        NpyErr_SetString(PyExc_ValueError,
                         "array dimensions are not "\
                         "compatible for copy");
        _Npy_DECREF(multi);
        return -1;
    }

    icopyfunc = in->descr->f->copyswapn;
    ocopyfunc = out->descr->f->copyswapn;
    maxaxis = NpyArray_RemoveSmallest(multi);
    if (maxaxis < 0) {
        /* cast 1 0-d array to another */
        N = 1;
        maxdim = 1;
        ostrides = delsize;
        istrides = selsize;
    }
    else {
        maxdim = multi->dimensions[maxaxis];
        N = (int) (NPY_MIN(maxdim, NPY_BUFSIZE));
        ostrides = multi->iters[0]->strides[maxaxis];
        istrides = multi->iters[1]->strides[maxaxis];

    }
    buffers[0] = malloc(N*delsize);
    if (buffers[0] == NULL) {
        NpyErr_NoMemory();
        return -1;
    }
    buffers[1] = malloc(N*selsize);
    if (buffers[1] == NULL) {
        free(buffers[0]);
        NpyErr_NoMemory();
        return -1;
    }
    if (NpyDataType_FLAGCHK(out->descr, NPY_NEEDS_INIT)) {
        memset(buffers[0], 0, N*delsize);
    }
    if (NpyDataType_FLAGCHK(in->descr, NPY_NEEDS_INIT)) {
        memset(buffers[1], 0, N*selsize);
    }

#if NPY_ALLOW_THREADS
    if (NpyArray_ISNUMBER(in) && NpyArray_ISNUMBER(out)) {
        NPY_BEGIN_THREADS;
    }
#endif

    while (multi->index < multi->size) {
        _strided_buffered_cast(multi->iters[0]->dataptr,
                               ostrides,
                               delsize, oswap, ocopyfunc,
                               multi->iters[1]->dataptr,
                               istrides,
                               selsize, iswap, icopyfunc,
                               maxdim, buffers, N,
                               castfunc, out, in);
        NpyArray_MultiIter_NEXT(multi);
    }
#if NPY_ALLOW_THREADS
    if (NpyArray_ISNUMBER(in) && NpyArray_ISNUMBER(out)) {
        NPY_END_THREADS;
    }
#endif
    _Npy_DECREF(multi);
    if (NpyDataType_REFCHK(in->descr)) {
        obptr = buffers[1];
        for (i = 0; i < N; i++, obptr+=selsize) {
            NpyArray_Item_XDECREF(obptr, in->descr);
        }
    }
    if (NpyDataType_REFCHK(out->descr)) {
        obptr = buffers[0];
        for (i = 0; i < N; i++, obptr+=delsize) {
            NpyArray_Item_XDECREF(obptr, out->descr);
        }
    }
    free(buffers[0]);
    free(buffers[1]);
    if (NpyErr_Occurred()) {
        return -1;
    }

    return 0;
}





/*NUMPY_API
 * Get a cast function to cast from the input descriptor to the
 * output type_number (must be a registered data-type).
 * Returns NULL if un-successful.
 */
NpyArray_VectorUnaryFunc *
NpyArray_GetCastFunc(NpyArray_Descr *descr, int type_num)
{
    NpyArray_VectorUnaryFunc *castfunc = NULL;

    if (type_num < NPY_NTYPES) {
        castfunc = descr->f->cast[type_num];
    } else {
        /* Check castfuncs for casts to user defined types. */
        NpyArray_CastFuncsItem* pitem = descr->f->castfuncs;
        if (pitem != NULL) {
            while (pitem->totype != NPY_NOTYPE) {
                if (pitem->totype == type_num) {
                    castfunc = pitem->castfunc;
                    break;
                }
            }
        }
    }
    if (NpyTypeNum_ISCOMPLEX(descr->type_num) &&
        !NpyTypeNum_ISCOMPLEX(type_num) &&
        NpyTypeNum_ISNUMBER(type_num) &&
        !NpyTypeNum_ISBOOL(type_num)) {

        /* TODO: Need solution for using ComplexWarning class as object to
           NpyErr_WarnEx. Callback or just classify as RuntimeErr? */
        PyObject *cls = NULL, *obj = NULL;
        obj = PyImport_ImportModule("numpy.core");
        if (obj) {
            cls = PyObject_GetAttrString(obj, "ComplexWarning");
            Py_DECREF(obj);
        }
        NpyErr_WarnEx(cls,
                      "Casting complex values to real discards the imaginary "
                      "part", 0);
        Npy_Interface_XDECREF(cls);
    }

    if (NULL == castfunc) {
        NpyErr_SetString(NpyExc_ValueError, "No cast function available.");
        return NULL;
    }
    return castfunc;
}




/*
 * Must be broadcastable.
 * This code is very similar to NpyArray_CopyInto/NpyArray_MoveInto
 * except casting is done --- NPY_BUFSIZE is used
 * as the size of the casting buffer.
 */

/*NUMPY_API
 * Cast to an already created array.
 */
int
NpyArray_CastTo(NpyArray *out, NpyArray *mp)
{
    int simple;
    int same;
    NpyArray_VectorUnaryFunc *castfunc = NULL;
    npy_intp mpsize = NpyArray_SIZE(mp);
    int iswap, oswap;
    NPY_BEGIN_THREADS_DEF;

    if (mpsize == 0) {
        return 0;
    }
    if (!NpyArray_ISWRITEABLE(out)) {
        NpyErr_SetString(NpyExc_ValueError, "output array is not writeable");
        return -1;
    }

    castfunc = NpyArray_GetCastFunc(mp->descr, out->descr->type_num);
    if (castfunc == NULL) {
        return -1;
    }

    same = NpyArray_SAMESHAPE(out, mp);
    simple = same && ((NpyArray_ISCARRAY_RO(mp) && NpyArray_ISCARRAY(out)) ||
                      (NpyArray_ISFARRAY_RO(mp) && NpyArray_ISFARRAY(out)));
    if (simple) {
#if NPY_ALLOW_THREADS
        if (NpyArray_ISNUMBER(mp) && NpyArray_ISNUMBER(out)) {
            NPY_BEGIN_THREADS;
        }
#endif
        castfunc(mp->data, out->data, mpsize, mp, out);

#if NPY_ALLOW_THREADS
        if (NpyArray_ISNUMBER(mp) && NpyArray_ISNUMBER(out)) {
            NPY_END_THREADS;
        }
#endif
        if (NpyErr_Occurred()) {
            return -1;
        }
        return 0;
    }

    /*
     * If the input or output is OBJECT, STRING, UNICODE, or VOID
     *  then getitem and setitem are used for the cast
     *  and byteswapping is handled by those methods
     */
    if (NpyArray_ISFLEXIBLE(mp) || NpyArray_ISOBJECT(mp) ||
              NpyArray_ISOBJECT(out) || NpyArray_ISFLEXIBLE(out)) {
        iswap = oswap = 0;
    }
    else {
        iswap = NpyArray_ISBYTESWAPPED(mp);
        oswap = NpyArray_ISBYTESWAPPED(out);
    }

    return _broadcast_cast(out, mp, castfunc, iswap, oswap);
}




/*NUMPY_API
 * For backward compatibility
 *
 * Cast an array using typecode structure.
 * steals reference to at --- cannot be NULL
 */
NpyArray *
NpyArray_CastToType(NpyArray *mp, NpyArray_Descr *at, int fortran)
{
    NpyArray *out;
    int ret;
    NpyArray_Descr *mpd;

    mpd = mp->descr;

    if (((mpd == at) ||
         ((mpd->type_num == at->type_num) &&
          NpyArray_EquivByteorders(mpd->byteorder, at->byteorder) &&
          ((mpd->elsize == at->elsize) || (at->elsize==0)))) &&
        NpyArray_ISBEHAVED_RO(mp)) {
        _Npy_DECREF(at);
        _Npy_INCREF(mp);
        return mp;
    }

    if (at->elsize == 0) {
        NpyArray_DESCR_REPLACE(at);
        if (at == NULL) {
            return NULL;
        }
        if (mpd->type_num == NPY_STRING &&
            at->type_num == NPY_UNICODE) {
            at->elsize = mpd->elsize << 2;
        }
        if (mpd->type_num == NPY_UNICODE &&
            at->type_num == NPY_STRING) {
            at->elsize = mpd->elsize >> 2;
        }
        if (at->type_num == NPY_VOID) {
            at->elsize = mpd->elsize;
        }
    }

    out = NpyArray_NewFromDescr(at, mp->nd,
                                mp->dimensions,
                                NULL, NULL,
                                fortran, NPY_FALSE,
                                NULL, Npy_INTERFACE(mp));

    if (out == NULL) {
        return NULL;
    }
    ret = NpyArray_CastTo(out, mp);
    if (ret != -1) {
        return out;
    }

    _Npy_DECREF(out);
    return NULL;
}






static int
_bufferedcast(NpyArray *out, NpyArray *in,
              NpyArray_VectorUnaryFunc *castfunc)
{
    char *inbuffer, *bptr, *optr;
    char *outbuffer=NULL;
    NpyArrayIterObject *it_in = NULL, *it_out = NULL;
    npy_intp i, index;
    npy_intp ncopies = NpyArray_SIZE(out) / NpyArray_SIZE(in);
    int elsize=in->descr->elsize;
    int nels = NPY_BUFSIZE;
    int el;
    int inswap, outswap = 0;
    int obuf=!NpyArray_ISCARRAY(out);
    int oelsize = out->descr->elsize;
    NpyArray_CopySwapFunc *in_csn;
    NpyArray_CopySwapFunc *out_csn;
    int retval = -1;

    in_csn = in->descr->f->copyswap;
    out_csn = out->descr->f->copyswap;

    /*
     * If the input or output is STRING, UNICODE, or VOID
     * then getitem and setitem are used for the cast
     *  and byteswapping is handled by those methods
     */

    inswap = !(NpyArray_ISFLEXIBLE(in) || NpyArray_ISNOTSWAPPED(in));

    inbuffer = NpyDataMem_NEW(NPY_BUFSIZE*elsize);
    if (inbuffer == NULL) {
        return -1;
    }
    if (NpyArray_ISOBJECT(in)) {
        memset(inbuffer, 0, NPY_BUFSIZE*elsize);
    }
    it_in = NpyArray_IterNew(in);
    if (it_in == NULL) {
        goto exit;
    }
    if (obuf) {
        outswap = !(NpyArray_ISFLEXIBLE(out) ||
                    NpyArray_ISNOTSWAPPED(out));
        outbuffer = NpyDataMem_NEW(NPY_BUFSIZE*oelsize);
        if (outbuffer == NULL) {
            goto exit;
        }
        if (NpyArray_ISOBJECT(out)) {
            memset(outbuffer, 0, NPY_BUFSIZE*oelsize);
        }
        it_out = NpyArray_IterNew(out);
        if (it_out == NULL) {
            goto exit;
        }
        nels = NPY_MIN(nels, NPY_BUFSIZE);
    }

    optr = (obuf) ? outbuffer: out->data;
    bptr = inbuffer;
    el = 0;
    while (ncopies--) {
        index = it_in->size;
        NpyArray_ITER_RESET(it_in);
        while (index--) {
            in_csn(bptr, it_in->dataptr, inswap, in);
            bptr += elsize;
            NpyArray_ITER_NEXT(it_in);
            el += 1;
            if ((el == nels) || (index == 0)) {
                /* buffer filled, do cast */
                castfunc(inbuffer, optr, el, in, out);
                if (obuf) {
                    /* Copy from outbuffer to array */
                    for (i = 0; i < el; i++) {
                        out_csn(it_out->dataptr,
                                optr, outswap,
                                out);
                        optr += oelsize;
                        NpyArray_ITER_NEXT(it_out);
                    }
                    optr = outbuffer;
                }
                else {
                    optr += out->descr->elsize * nels;
                }
                el = 0;
                bptr = inbuffer;
            }
        }
    }
    retval = 0;

exit:
    _Npy_XDECREF(it_in);
    NpyDataMem_FREE(inbuffer);
    NpyDataMem_FREE(outbuffer);
    if (obuf) {
        _Npy_XDECREF(it_out);
    }
    return retval;
}

/*NUMPY_API
 * Cast to an already created array.  Arrays don't have to be "broadcastable"
 * Only requirement is they have the same number of elements.
 */
int
NpyArray_CastAnyTo(NpyArray *out, NpyArray *mp)
{
    int simple;
    NpyArray_VectorUnaryFunc *castfunc = NULL;
    npy_intp mpsize = NpyArray_SIZE(mp);

    if (mpsize == 0) {
        return 0;
    }
    if (!NpyArray_ISWRITEABLE(out)) {
        NpyErr_SetString(NpyExc_ValueError, "output array is not writeable");
        return -1;
    }

    if (!(mpsize == NpyArray_SIZE(out))) {
        NpyErr_SetString(NpyExc_ValueError,
                         "arrays must have the same number of"
                         " elements for the cast.");
        return -1;
    }

    castfunc = NpyArray_GetCastFunc(mp->descr, out->descr->type_num);
    if (castfunc == NULL) {
        return -1;
    }
    simple = ((NpyArray_ISCARRAY_RO(mp) && NpyArray_ISCARRAY(out)) ||
              (NpyArray_ISFARRAY_RO(mp) && NpyArray_ISFARRAY(out)));
    if (simple) {
        castfunc(mp->data, out->data, mpsize, mp, out);
        return 0;
    }
    if (NpyArray_SAMESHAPE(out, mp)) {
        int iswap, oswap;
        iswap = NpyArray_ISBYTESWAPPED(mp) && !NpyArray_ISFLEXIBLE(mp);
        oswap = NpyArray_ISBYTESWAPPED(out) && !NpyArray_ISFLEXIBLE(out);
        return _broadcast_cast(out, mp, castfunc, iswap, oswap);
    }
    return _bufferedcast(out, mp, castfunc);
}

/*NUMPY_API
 *Check the type coercion rules.
 */
int
NpyArray_CanCastSafely(int fromtype, int totype)
{
    NpyArray_Descr *from, *to;
    int felsize, telsize;

    if (fromtype == totype) {
        return 1;
    }
    if (fromtype == NPY_BOOL) {
        return 1;
    }
    if (totype == NPY_BOOL) {
        return 0;
    }
    if (fromtype == NPY_DATETIME || fromtype == NPY_TIMEDELTA ||
        totype == NPY_DATETIME || totype == NPY_TIMEDELTA) {
        return 0;
    }
    if (totype == NPY_OBJECT || totype == NPY_VOID) {
        return 1;
    }
    if (fromtype == NPY_OBJECT || fromtype == NPY_VOID) {
        return 0;
    }
    from = NpyArray_DescrFromType(fromtype);
    /*
     * cancastto is a NPY_NOTYPE terminated C-int-array of types that
     * the data-type can be cast to safely.
     */
    if (from->f->cancastto) {
        int *curtype;
        curtype = from->f->cancastto;
        while (*curtype != NPY_NOTYPE) {
            if (*curtype++ == totype) {
                return 1;
            }
        }
    }
    if (NpyTypeNum_ISUSERDEF(totype)) {
        return 0;
    }
    to = NpyArray_DescrFromType(totype);
    telsize = to->elsize;
    felsize = from->elsize;
    _Npy_DECREF(from);
    _Npy_DECREF(to);

    switch(fromtype) {
        case NPY_BYTE:
        case NPY_SHORT:
        case NPY_INT:
        case NPY_LONG:
        case NPY_LONGLONG:
            if (NpyTypeNum_ISINTEGER(totype)) {
                if (NpyTypeNum_ISUNSIGNED(totype)) {
                    return 0;
                }
                else {
                    return telsize >= felsize;
                }
            }
            else if (NpyTypeNum_ISFLOAT(totype)) {
                if (felsize < 8) {
                    return telsize > felsize;
                }
                else {
                    return telsize >= felsize;
                }
            }
            else if (NpyTypeNum_ISCOMPLEX(totype)) {
                if (felsize < 8) {
                    return (telsize >> 1) > felsize;
                }
                else {
                    return (telsize >> 1) >= felsize;
                }
            }
            else {
                return totype > fromtype;
            }
        case NPY_UBYTE:
        case NPY_USHORT:
        case NPY_UINT:
        case NPY_ULONG:
        case NPY_ULONGLONG:
            if (NpyTypeNum_ISINTEGER(totype)) {
                if (NpyTypeNum_ISSIGNED(totype)) {
                    return telsize > felsize;
                }
                else {
                    return telsize >= felsize;
                }
            }
            else if (NpyTypeNum_ISFLOAT(totype)) {
                if (felsize < 8) {
                    return telsize > felsize;
                }
                else {
                    return telsize >= felsize;
                }
            }
            else if (NpyTypeNum_ISCOMPLEX(totype)) {
                if (felsize < 8) {
                    return (telsize >> 1) > felsize;
                }
                else {
                    return (telsize >> 1) >= felsize;
                }
            }
            else {
                return totype > fromtype;
            }
        case NPY_FLOAT:
        case NPY_DOUBLE:
        case NPY_LONGDOUBLE:
            if (NpyTypeNum_ISCOMPLEX(totype)) {
                return (telsize >> 1) >= felsize;
            }
            else {
                return totype > fromtype;
            }
        case NPY_CFLOAT:
        case NPY_CDOUBLE:
        case NPY_CLONGDOUBLE:
            return totype > fromtype;
        case NPY_STRING:
        case NPY_UNICODE:
            return totype > fromtype;
        default:
            return 0;
    }
}

/*NUMPY_API
 * leaves reference count alone --- cannot be NULL
 */
npy_bool
NpyArray_CanCastTo(NpyArray_Descr *from, NpyArray_Descr *to)
{
    int fromtype=from->type_num;
    int totype=to->type_num;
    npy_bool ret;

    ret = NpyArray_CanCastSafely(fromtype, totype);
    if (ret) {
        /* Check String and Unicode more closely */
        if (fromtype == NPY_STRING) {
            if (totype == NPY_STRING) {
                ret = (from->elsize <= to->elsize);
            }
            else if (totype == NPY_UNICODE) {
                ret = (from->elsize << 2 <= to->elsize);
            }
        }
        else if (fromtype == NPY_UNICODE) {
            if (totype == NPY_UNICODE) {
                ret = (from->elsize <= to->elsize);
            }
        }
        /*
         * TODO: If totype is STRING or unicode
         * see if the length is long enough to hold the
         * stringified value of the object.
         */
    }
    return ret;
}


/*NUMPY_API
 * Is the typenum valid?
 */
int
NpyArray_ValidType(int type)
{
    NpyArray_Descr *descr;
    int res=NPY_TRUE;

    descr = NpyArray_DescrFromType(type);
    if (descr == NULL) {
        res = NPY_FALSE;
    }
    _Npy_DECREF(descr);
    return res;
}
