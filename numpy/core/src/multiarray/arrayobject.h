#ifndef _NPY_INTERNAL_ARRAYOBJECT_H_
#define _NPY_INTERNAL_ARRAYOBJECT_H_

#ifndef _MULTIARRAYMODULE
#error You should not include this
#endif

NPY_NO_EXPORT PyObject *
_strings_richcompare(PyArrayObject *self, PyArrayObject *other, int cmp_op,
                     int rstrip);

NPY_NO_EXPORT PyObject *
array_richcompare(PyArrayObject *self, PyObject *other, int cmp_op);

#define RETURN_PYARRAY(arr)                     \
    do {                                        \
        NpyArray* a_ = (arr);                    \
        PyArrayObject* ret_;                     \
        if (a_ == NULL) return NULL;             \
        ret_ = Npy_INTERFACE(a_);                 \
        Py_INCREF(ret_);                         \
        Npy_DECREF(a_);                         \
        return (PyObject*) ret_;                 \
    } while (0)

#define ASSIGN_TO_PYARRAY(pya, arr)             \
    do {                                        \
        NpyArray* a_ = (arr);                   \
        if (a_ == NULL) {                       \
            pya = NULL;                         \
        } else {                                \
            pya = Npy_INTERFACE(a_);            \
            Py_INCREF(pya);                     \
            Npy_DECREF(a_);                    \
        }                                       \
    } while (0)


#endif
