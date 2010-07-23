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
    ]
