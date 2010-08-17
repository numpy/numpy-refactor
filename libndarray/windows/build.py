import os
import sys
import shutil
from glob import glob
from os.path import join

assert sys.platform == 'win32'
sys.path.insert(0, r'..\tools')

from conv_template import process_file

src_dir = r'..\src'

def convert_templates():
    for path in glob(join(src_dir, '*.src')):
        process_file(path)


def main():
    convert_templates()
    shutil.copy('npy_config.h', src_dir)
    os.system("msbuild")


if __name__ == '__main__':
    main()
