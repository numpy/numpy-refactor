#ifndef _NPY_INTERNAL_H_
#define _NPY_INTERNAL_H_


extern struct NpyInterface_WrapperFuncs _NpyArrayWrapperFuncs;

#define NpyInterface_ArrayNewWrapper(a, b, c, d, e, f)                               \
    (NULL != _NpyArrayWrapperFuncs.array_new_wrapper ?                               \
        (_NpyArrayWrapperFuncs.array_new_wrapper)((a), (b), (c), (d), (e), (f)) :    \
        NPY_TRUE)

#define NpyInterface_IterNewWrapper(a, b)                                            \
    (NULL != _NpyArrayWrapperFuncs.iter_new_wrapper ?                                \
        (_NpyArrayWrapperFuncs.iter_new_wrapper)((a), (b)) :                         \
        NPY_TRUE)

#define NpyInterface_MultiIterNewWrapper(a, b)                                       \
    (NULL != _NpyArrayWrapperFuncs.multi_iter_new_wrapper ?                          \
        (_NpyArrayWrapperFuncs.multi_iter_new_wrapper)((a), (b)) :                   \
        NPY_TRUE)

#define NpyInterface_NeighborhoodIterNewWrapper(a, b)                                \
    (NULL != _NpyArrayWrapperFuncs.neighbor_iter_new_wrapper ?                       \
        (_NpyArrayWrapperFuncs.neighbor_iter_new_wrapper)((a), (b)) :                \
        NPY_TRUE)

#define NpyInterface_MapIterNewWrapper(a, b)                                         \
    (NULL != _NpyArrayWrapperFuncs.map_iter_new_wrapper ?                            \
        (_NpyArrayWrapperFuncs.map_iter_new_wrapper)((a), (b)) :                     \
        NPY_TRUE)

#define NpyInterface_DescrNewFromType(a, b, c)                                       \
    (NULL != _NpyArrayWrapperFuncs.descr_new_from_type ?                             \
        (_NpyArrayWrapperFuncs.descr_new_from_type)((a), (b), (c)) :                 \
        NPY_TRUE)

#define NpyInterface_DescrNewFromWrapper(a, b, c)                                    \
    (NULL != _NpyArrayWrapperFuncs.descr_new_from_wrapper ?                          \
        (_NpyArrayWrapperFuncs.descr_new_from_wrapper)((a), (b), (c)) :              \
        NPY_TRUE)


#endif
