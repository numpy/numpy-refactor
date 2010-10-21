import sys

from info import __doc__
from numpy.version import version as __version__

if sys.platform != 'cli':
    import multiarray
    import umath
else:
    import multiarray_cli as multiarray
    import umath_cli as umath

import _internal # for freeze programs

if sys.platform != 'cli':
    import numerictypes as nt
    multiarray.set_typeDict(nt.sctypeDict)

    import _sort
else:
    import numerictypes as nt

from numeric import *
from fromnumeric import *
import defchararray as char
import records as rec
from records import *
from memmap import *
from defchararray import chararray

if sys.platform != 'cli':
    import scalarmath

from function_base import *
from machar import *

if sys.platform != 'cli':
    from getlimits import *

from shape_base import *

del nt

from fromnumeric import amax as max, amin as min, round_ as round

if sys.platform != 'cli':
    from numeric import absolute as abs

__all__ = ['char','rec','memmap']
__all__ += numeric.__all__
__all__ += fromnumeric.__all__
__all__ += rec.__all__
__all__ += ['chararray']
__all__ += function_base.__all__
__all__ += machar.__all__
if sys.platform != 'cli':
    __all__ += getlimits.__all__
__all__ += shape_base.__all__


from numpy.testing import Tester
test = Tester(__file__).test
bench = Tester(__file__).bench
