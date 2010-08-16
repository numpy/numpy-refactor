import re



ADD_NPY = '''
HAVE_DECL_ISNAN
HAVE_DECL_ISFINITE
HAVE_DECL_ISINF

SIZEOF_SHORT
SIZEOF_INT
SIZEOF_LONG
SIZEOF_FLOAT
SIZEOF_DOUBLE
'''.split()


MAP = {
    'HAVE_LONG_LONG_INT':         'NPY_HAVE_LONGLONG',
    'SIZEOF_LONG_DOUBLE':         'NPY_SIZEOF_LONGDOUBLE',
    'SIZEOF_LONG_LONG':           'NPY_SIZEOF_LONGLONG',
    'SIZEOF_VOID_P':              'NPY_SIZEOF_PTR',
    'SIZEOF_FLOAT_COMPLEX':       'NPY_SIZEOF_COMPLEX_FLOAT',
    'SIZEOF_DOUBLE_COMPLEX':      'NPY_SIZEOF_COMPLEX_DOUBLE',
    'SIZEOF_LONG_DOUBLE_COMPLEX': 'NPY_SIZEOF_COMPLEX_LONGDOUBLE',
}


def define_repl(match):
    name = match.group(2)

    if name in ADD_NPY:
        name = 'NPY_' + name

    if name in MAP:
        name = MAP[name]

    return match.group(1) + name + match.group(3)


def rename_defs(data):
    p = re.compile(r'^(#\s*define\s+)([A-Z0-9_]+)(\s+\S+)$', re.M)
    return p.sub(define_repl, data)


if __name__ == '__main__':
    src = 'config.h'
    dst = 'src/npy_config.h'

    print "\treading:", src
    data = open(src).read()

    data = '''\
/* npy_config.h   Generated from config.h by tools/mk_config.py */
''' + rename_defs(data)

    print "\twriting:", dst
    fo = open(dst, 'w')
    fo.write(data)
    fo.close()
