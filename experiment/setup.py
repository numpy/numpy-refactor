from os.path import expanduser, join

from distutils.core import setup
from distutils.sysconfig import get_python_lib
from distutils.extension import Extension
from Cython.Distutils import build_ext

usr_dir = join(expanduser('~'), 'usr')

setup(
    cmdclass = {"build_ext": build_ext},
    ext_modules = [
        Extension(
            "interface", ["interface.pyx"],
            include_dirs = [
                join(usr_dir, 'include'),
                join(get_python_lib(), 'numpy', 'core', 'include')
            ],
            library_dirs = [join(usr_dir, 'lib')],
            libraries = ['ndarray'],
        )
    ],
)
