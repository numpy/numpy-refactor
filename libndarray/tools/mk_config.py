import re
import sys



DEFINE_MAP = {
    'VERSION':                    'DUMMY_VERSION',
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
    if name in DEFINE_MAP:
        name = DEFINE_MAP[name]
    else:
        name = 'NPY_' + name
    return match.group(1) + name + match.group(3)

def rename_defs(data):
    p = re.compile(r'^(#\s*define\s+)([A-Z0-9_]+)(\s+\S+)$', re.M)
    return p.sub(define_repl, data)



_IEEE_DOUBLE_BE = '\xc1\x9d\x6f\x34\x54\x00\x00\x00'
_IEEE_QUAD_BE = '\xc0\x19\xd6\xf3\x45\x40' + 10 * '\x00'
_INTEL_EXT_12_LE = '\x00\x00\x00\x00\xa0\xa2\x79\xeb\x19\xc0\x00\x00'
_INTEL_EXT_16_LE = _INTEL_EXT_12_LE + 4 * '\x00'

LD_MAP = {
    _IEEE_DOUBLE_BE:       'IEEE_DOUBLE_BE',
    _IEEE_DOUBLE_BE[::-1]: 'IEEE_DOUBLE_LE',
    _IEEE_QUAD_BE:         'IEEE_QUAD_BE',
    _IEEE_QUAD_BE[::-1]:   'IEEE_QUAD_LE',
    _INTEL_EXT_12_LE:      'INTEL_EXT_12BYTES_LE',
    _INTEL_EXT_16_LE:      'INTEL_EXT_16BYTES_LE',
}

def check_long_double_repr(path):
    print "\tcheck_long_double_repr:", path
    data = open(path, 'rb').read()

    p = re.compile('\0{8}aR8qyb1W(.+)z7pLC3Si')
    m = p.search(data)
    if m is None:
        raise ValueError("Could not locate sequence")

    content = m.group(1)
    if content not in LD_MAP:
        raise ValueError("Unrecognized format: %r" % content)

    res = LD_MAP[content]
    print "\tdetected:", res
    return res


def main():
    src = 'config.h'
    dst = 'src/npy_config.h'

    print "\treading:", src
    data = open(src).read()

    data = '''\
/* npy_config.h   Generated from config.h by tools/mk_config.py */
''' + rename_defs(data)

    if sys.platform == 'darwin':
        data += '''
/* long double representation is not added by tools/mk_config.py on OSX
   because NPY_LDOUBLE_??? is definied in src/npy_math_private.h on OSX
*/
'''
    else:
        data += '''
/* long double representation (added by tools/mk_config.py) */
#define NPY_LDOUBLE_%s 1
''' % check_long_double_repr('tools/long_double.o')

    print "\twriting:", dst
    fo = open(dst, 'w')
    fo.write(data)
    fo.close()


if __name__ == '__main__':
    main()
