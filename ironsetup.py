import os
import sys
import shutil
import tempfile
from os.path import isdir, isfile, join


if sys.platform != 'cli':
    print "ERROR: This setup script only works under IronPython"
    sys.exit(1)

src_dir = os.getcwd()


def msbuild():
    os.chdir(join(src_dir, 'numpy\NumpyDotNet'))
    os.system('msbuild')
    os.chdir(src_dir)


def install():
    print "INSTALLING ..."
    sp_dir = join(sys.prefix, r'Lib\site-packages')
    bin_dir = join(src_dir, r'numpy\NumpyDotNet\bin\Debug')
    dll_dir = join(sys.prefix, 'DLLs')
    if not isdir(dll_dir):
        os.mkdir(dll_dir)
    shutil.copy(join(bin_dir, 'numpy.py'), sp_dir)
    for fn in ['ndarray.dll', 'NpyAccessLib.dll', 'NumpyDotNet.dll']:
        src = join(bin_dir, fn)
        dst = join(dll_dir, fn)
        if isfile(dst):
            tmp_dir = tempfile.mkdtemp()
            os.rename(dst, join(tmp_dir, fn))
        shutil.copyfile(src, dst)


if __name__ == '__main__':
    msbuild()
    install()
