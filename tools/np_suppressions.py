suppressions = [
    [ ".*/multiarray/mapping\.", "PyArray_MapIterReset" ],

    # PyArray_Std trivially forwards to and appears to be superceded by
    # __New_PyArray_Std, which is exercised by the test framework.
    [ ".*/multiarray/calculation\.", "PyArray_Std" ],

    # PyCapsule_Check is declared in a header, and used in
    # multiarray/ctors.c.  So it isn't really untested.
    [ ".*/multiarray/common\.", "PyCapsule_Check" ],
    ]
