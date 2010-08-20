import os
import sys
import shutil
from glob import glob
from os.path import expanduser, join

from egginst.utils import rm_rf


assert sys.platform == 'win32'
sys.path.insert(0, r'..\tools')

from conv_template import process_file
from mk_config import check_long_double_repr

src_dir = r'..\src'


def write_config():
    os.system(r"..\tools\long_double.c")
    data = open('npy_config.h').read()
    data += '''
/* long double representation */
#define NPY_LDOUBLE_%s 1
''' % check_long_double_repr('long_double.obj')
    fo = open(join(src_dir, 'npy_config.h'))
    fo.write(data)
    fo.close()


def convert_templates():
    for path in glob(join(src_dir, '*.src')):
        process_file(path)


def install():
    usr_dir = expanduser(r'~\usr')
    rm_rf(usr_dir)

    usr_inc_dir = join(usr_dir, 'include')
    os.makedirs(usr_inc_dir)
    for path in glob(join(src_dir, '*.h')):
        shutil.copy(path, usr_inc_dir)

    usr_lib_dir = join(usr_dir, 'lib')
    os.makedirs(usr_lib_dir)
    shutil.copy(r'Release\ndarray.lib', usr_lib_dir)
    shutil.copy(join(sys.prefix, 'libs', 'python26.lib'), usr_lib_dir)
    shutil.copy(r'Release\ndarray.dll', sys.prefix)


def main():
    convert_templates()
    write_config()
    os.system("msbuild /v:diag")
    install()


if __name__ == '__main__':
    main()
