import os
import sys
import shutil
from glob import glob
from os.path import expanduser, join


sys.path.insert(0, r'..\tools')

from conv_template import process_file
from mk_config import check_long_double_repr

src_dir = r'..\src'


def write_config():
    os.system(r"cl ..\tools\long_double.c")
    data = open('npy_config.h').read()
    data += '''
/* long double representation */
#define NPY_LDOUBLE_%s 1
''' % check_long_double_repr('long_double.obj')
    fo = open(join(src_dir, 'npy_config.h'), 'w')
    fo.write(data)
    fo.close()


def convert_templates():
    for path in glob(join(src_dir, '*.src')):
        process_file(path)


def main():
    convert_templates()
    write_config()
    os.system("msbuild /v:diag msvc2008.vcproj")
    # shutil.copy(r'Release\ndarray.dll', sys.prefix)


if __name__ == '__main__':
    main()
