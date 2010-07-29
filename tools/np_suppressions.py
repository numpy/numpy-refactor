suppressions = [
    # This one cannot be covered by any Python language test because there is
    # no code pathway to it.  But it is part of the C API, so must not be
    # excised from the code.
    [ r".*/multiarray/mapping\.", "PyArray_MapIterReset" ],

    # PyArray_Std trivially forwards to and appears to be superceded by
    # __New_PyArray_Std, which is exercised by the test framework.
    [ r".*/multiarray/calculation\.", "PyArray_Std" ],

    # PyCapsule_Check is declared in a header, and used in
    # multiarray/ctors.c.  So it isn't really untested.
    [ r".*/multiarray/common\.", "PyCapsule_Check" ],

    # It is unclear why these aren't called by the array casting tests in
    # test_npy_arraytypes.py, when other X_to_X functions are called.
    [ r".*/libnumpy/npy_arraytypes\.", "DATETIME_to_DATETIME" ],
    [ r".*/libnumpy/npy_arraytypes\.", "TIMEDELTA_to_TIMEDELTA" ],
    [ r".*/libnumpy/npy_arraytypes\.", "BOOL_to_BOOL" ],
    [ r".*/libnumpy/npy_arraytypes\.", "BYTE_to_BYTE" ],
    [ r".*/libnumpy/npy_arraytypes\.", "UBYTE_to_UBYTE" ],
    [ r".*/libnumpy/npy_arraytypes\.", "LONGLONG_to_LONGLONG" ],
    [ r".*/libnumpy/npy_arraytypes\.", "ULONGLONG_to_ULONGLONG" ],

    ]
