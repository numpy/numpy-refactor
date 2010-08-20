cl /c "/IC:\\Documents and Settings\\builder\\usr\\include" main.c

link /nologo /OUT:main.exe \
    "/LIBPATH:C:\\Documents and Settings\\builder\\usr\\lib" \
    main.obj ndarray.lib
