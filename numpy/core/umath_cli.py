
import sys

if sys.platform == 'cli':
    import clr
    clr.AddReference("NumpyDotNet")
    import NumpyDotNet
    NumpyDotNet.umath.__init__()

