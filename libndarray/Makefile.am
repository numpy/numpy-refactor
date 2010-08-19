AUTOMAKE_OPTIONS = foreign


# Headers which are installed to support the library
INSTINCLUDES = \
        src/_neighborhood_iterator_imp.h \
        src/npy_api.h \
        src/npy_arrayobject.h \
        src/npy_common.h \
        src/npy_config.h \
        src/npy_cpu.h \
        src/npy_defs.h \
        src/npy_descriptor.h \
        src/npy_dict.h \
        src/npy_endian.h \
        src/npy_funcs.h \
        src/npy_index.h \
        src/npy_iterators.h \
        src/npy_loops.h \
        src/npy_math.h \
        src/npy_object.h \
        src/npy_os.h \
        src/npy_ufunc_object.h \
        src/npy_utils.h

# Headers which are not installed
OTHERINCLUDES = \
	src/npy_math_private.h \
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
        src/npy_funcs.c \
        src/npy_getset.c \
        src/npy_ieee754.c \
        src/npy_index.c \
        src/npy_item_selection.c \
        src/npy_iterators.c \
        src/npy_loops.c \
        src/npy_mapping.c \
        src/npy_math.c \
        src/npy_math_complex.c \
        src/npy_methods.c \
        src/npy_multiarray.c \
        src/npy_number.c \
        src/npy_os.c \
        src/npy_refcount.c \
        src/npy_shape.c \
        src/npy_ufunc_object.c \
        src/npy_usertypes.c \
        tools/mk_config.py \
        tools/long_double.c

# libtool libraries to build
lib_LTLIBRARIES = libndarray.la

# Library sources for libndarray.la
libndarray_la_SOURCES = $(LIBSOURCES)

# Headers to install
include_HEADERS = $(INSTINCLUDES)

# Other distributed headers
noinst_HEADERS = $(OTHERINCLUDES)

CLEANFILES = \
        src/npy_arraytypes.c \
        src/npy_ieee754.c \
        src/npy_funcs.c \
        src/npy_funcs.h \
        src/npy_loops.c \
        src/npy_loops.h \
        src/npy_math.c \
        src/npy_math_complex.c \
        src/npy_config.c \
        tools/long_double.o


src/npy_config.h: tools/mk_config.py config.h tools/long_double.o
	python $<

tools/long_double.o: tools/long_double.c
	$(COMPILE) -c -o $@ $<


CONV_TMPL = python tools/conv_template.py

src/npy_arraytypes.c: src/npy_arraytypes.c.src
	$(CONV_TMPL) $<

src/npy_ieee754.c: src/npy_ieee754.c.src
	$(CONV_TMPL) $<

src/npy_math.c: src/npy_math.c.src
	$(CONV_TMPL) $<

src/npy_math_complex.c: src/npy_math_complex.c.src
	$(CONV_TMPL) $<

src/npy_funcs.c: src/npy_funcs.c.src src/npy_funcs.h
	$(CONV_TMPL) $<

src/npy_funcs.h: src/npy_funcs.h.src
	$(CONV_TMPL) $<

src/npy_loops.c: src/npy_loops.c.src src/npy_loops.h
	$(CONV_TMPL) $<

src/npy_loops.h: src/npy_loops.h.src
	$(CONV_TMPL) $<
