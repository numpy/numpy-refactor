AUTOMAKE_OPTIONS = foreign


# Headers which are installed to support the library
INSTINCLUDES = \
        npy_config.h \
        src/_neighborhood_iterator_imp.h \
        src/npy_api.h \
        src/npy_arrayobject.h \
        src/npy_common.h \
        src/npy_cpu.h \
        src/npy_defs.h \
        src/npy_descriptor.h \
        src/npy_dict.h \
        src/npy_endian.h \
        src/npy_index.h \
        src/npy_iterators.h \
        src/npy_math.h \
        src/npy_object.h \
        src/npy_os.h \
        src/npy_ufunc_object.h \
        src/npy_utils.h

# Headers which are not installed
OTHERINCLUDES = \
	src/npy_fpmath.h \
	src/npy_math_private.h \
        src/npy_noprefix.h \
        src/npy_number.h \
        src/npy_internal.h

# Sources to build library
LIBSOURCES = \
        src/npy_arrayobject.c \
        src/npy_arraytypes.c \
        src/npy_calculation.c \
        src/npy_common.c \
        src/npy_conversion_utils.c \
        src/npy_convert.c \
        src/npy_convert_datatype.c \
        src/npy_ctors.c \
        src/npy_datetime.c \
        src/npy_descriptor.c \
        src/npy_dict.c \
        src/npy_flagsobject.c \
        src/npy_getset.c \
        src/npy_ieee754.c \
        src/npy_index.c \
        src/npy_item_selection.c \
        src/npy_iterators.c \
        src/npy_mapping.c \
        src/npy_math.c \
        src/npy_math_complex.c \
        src/npy_methods.c \
        src/npy_multiarray.c \
        src/npy_number.c \
        src/npy_refcount.c \
        src/npy_shape.c \
        src/npy_usertypes.c \
        src/npy_os.c


# libtool libraries to build
lib_LTLIBRARIES = libndarray.la

# Library sources for libndarray.la
libndarray_la_SOURCES = $(LIBSOURCES)

# Headers to install
include_HEADERS = $(INSTINCLUDES)

# Other distributed headers
noinst_HEADERS = $(OTHERINCLUDES)


src/npy_arraytypes.c: src/npy_arraytypes.c.src
	python src/conv_template.py src/npy_arraytypes.c.src

src/npy_ieee754.c: src/npy_ieee754.c.src
	python src/conv_template.py src/npy_ieee754.c.src

src/npy_math.c: src/npy_math.c.src
	python src/conv_template.py src/npy_math.c.src

src/npy_math_complex.c: src/npy_math_complex.c.src
	python src/conv_template.py src/npy_math_complex.c.src
