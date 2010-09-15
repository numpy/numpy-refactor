import os
import sys
from os.path import join


if sys.platform != 'cli':
    print "ERROR: This setup script only works under IronPython"
    sys.exit(1)

src_dir = os.getcwd()


def msbuild():
    os.chdir(join(src_dir, 'numpy\NumpyDotNet'))
    os.system('msbuild')
    os.chdir(src_dir)


if __name__ == '__main__':
    msbuild()
