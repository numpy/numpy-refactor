import os
import sys
import shutil
import tempfile
from os.path import dirname, isdir, isfile, join


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
    numpy_dir = join(src_dir, 'numpy')
    bin_dir = join(numpy_dir, r'NumpyDotNet\bin\Debug')
    dll_dir = join(sys.prefix, 'DLLs')
    if not isdir(dll_dir):
        os.mkdir(dll_dir)
    for fn in ['ndarray.dll', 'NpyAccessLib.dll', 'NumpyDotNet.dll']:
        src = join(bin_dir, fn)
        dst = join(dll_dir, fn)
        if isfile(dst):
            tmp_dir = tempfile.mkdtemp()
            os.rename(dst, join(tmp_dir, fn))
        shutil.copyfile(src, dst)

    for root, dirs, files in os.walk(numpy_dir):
        for fn in files:
            if not fn.endswith('.py'):
                continue
            abs_path = join(root, fn)
            rel_path = abs_path[len(numpy_dir) + 1:]
            dst_dir = dirname(join(sp_dir, 'numpy', rel_path))
            if not isdir(dst_dir):
                 os.makedirs(dst_dir)
            shutil.copy(abs_path, dst_dir)


if __name__ == '__main__':
    #msbuild()
    install()
