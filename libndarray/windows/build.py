import os
import sys
import shutil
from glob import glob
from os.path import expanduser, join

from egginst.utils import rm_rf


assert sys.platform == 'win32'
sys.path.insert(0, r'..\tools')

from conv_template import process_file

src_dir = r'..\src'

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
    shutil.copy(r'Release\libndarray.dll', usr_lib_dir)


def main():
    convert_templates()
    shutil.copy('npy_config.h', src_dir)
    os.system("msbuild")
    install()


if __name__ == '__main__':
    main()
