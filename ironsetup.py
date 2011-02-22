import os
import sys
import shutil
import tempfile
from os.path import dirname, isdir, isfile, join
from iron_install import install


if sys.platform != 'cli':
    print "ERROR: This setup script only works under IronPython"
    sys.exit(1)

src_dir = os.getcwd()


def msbuild():
    pass
    os.chdir(join(src_dir, r'numpy\NumpyDotNet'))
    os.system('msbuild /p:Configuration=Debug')
    os.chdir(src_dir)


if __name__ == '__main__':
    msbuild()
    install()
