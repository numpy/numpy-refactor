#!/bin/tcsh -f
#

setenv NPY_SEPARATE_COMPILATION 1
setenv CFLAGS "-DDEBUG -g -Wall -Wextra"
setenv LDFLAGS "-g"

#setenv CLFAGS "$CFLAGS -fprofile-arcs -fprofile-arcs -ftest-coverage"
#setenv LINKFLAGSEND "-lgov"

python setupscons.py install --prefix=./install 
#python setup.py install --prefix=./install
