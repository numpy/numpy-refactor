"""nptool - A tool to assist with numpy development and testing.

More documentation here.
"""

import sys
import os
import re
import shutil
from subprocess import *
from np_suppressions import suppressions

def is_suppressed( file, func ):
    "Check to see if file:func matches a registered suppression."

    # Iterate through all file/func pairs in our suppressions list.
    for sfile, sfunc in suppressions:

        # Match each suppression file against the current file.
        m = re.match( sfile, file )
        if m:
            # The file was matched by this regexp.  Now check the function.
            m = re.match( sfunc, func )
            if m:
                # The function also matched, so this is suppressed.
                return True

    return False

class nptool:

    def __init__ ( self ):
        pass

    def cmd ( self, argv ):
        "Analyze command, branch to appropriate handler."

        # Break up the arglist into a command with optional args.
        try:
            cmd = argv[0]
            args = argv[1:]

        except:
            self.usage()

        # Process the commands we recognize, or issue usage output.
        if cmd == "clean":
            if os.path.exists( 'build' ):
                shutil.rmtree( 'build' )

        elif cmd == "build":
            self.build()

        elif cmd == "test" or cmd == "run_tests":
            self.run_tests()

        elif cmd == "gcov" or cmd == "run_gcov":
            self.run_gcov()

        else:
            self.usage()

    def build( self ):
        "Build numpy with gcov enabled."

        p = Popen( """/usr/bin/env \
 NPY_SEPARATE_COMPILATION=1 \
 CFLAGS=\"-DDEBUG -g -Wall --coverage\" \
 LDFLAGS=\"-g --coverage\" \
 \
python setupscons.py build install""", shell=True )
        status = os.waitpid( p.pid, 0 )[1]

        if not status == 0:
            print "Some trouble with the build, investigate."
            sys.exit(1)

    def run_tests( self ):
        "Execute the numpy unit tests."

        # Make sure we have a parallel directory, outside the numpy-refactor
        # sandbox, in which to run the unit tests themselves.
        if not os.path.exists( "../runtests" ):
            os.mkdir( "../runtests" )

        # Make sure the driver script is ready to go.
        if not os.path.exists( "../runtests/t.py" ):
            f = open( "../runtests/t.py", "w" )
            print >>f, """import sys
import ctypes

_old_rtld = sys.getdlopenflags()
sys.setdlopenflags(_old_rtld | ctypes.RTLD_GLOBAL)

import numpy

numpy.test()
"""
            f.close()

        # Run the unit tests.
        p = Popen( "cd ../runtests; python t.py |& tee test.log", shell=True )
        status = os.waitpid( p.pid, 0 )[1]

        if status:
            print "Trouble running tests, investigate."
            sys.exit(1)

    def run_gcov( self ):
        "Perform coverage analysis and generate report."

        if not os.path.exists( 'build' ):
            print "Looks like you haven't built numpy yet."
            print "Issue  : nptool build"
            print "Then do: nptool test"
            sys.exit(1)

        # Find the files which contain coverage instrumentation data.
        files = os.popen( 'find build -name "*.gcno" \
        -path \*/core/\* -a ! -path \*/sconf/\*', 'r' ).readlines()

        da_files = os.popen( 'find build -name "*.gcda" \
        -path \*/core/\* -a ! -path \*/sconf/\*', 'r' ).readlines()

        if len( files ) == 0 or len( da_files ) == 0:
            print "No coverage instrumentation data available."
            print "Issue: nptool test"
            sys.exit(1)

        #print "files:"
        #print files

        self.data = {}

        # Analyze each file for which instrumentation data is present.
        for file in files:
            # Strip off the trailing newline from the file name.
            self.run_gcov_on_file( file[:-1] )

            # For testing, it may be convenient to bail after analyzing the
            # first file, or maybe after a few files...
            #if len( self.data.keys() ) == 10:
            #    break

        # Emit the final summary report.
        self.report_gcov_data()

    def run_gcov_on_file( self, file ):
        "Run gcov on a single file."

        t = os.path.basename( file )
        b = os.path.splitext( t )[0]
        d = os.path.dirname( file )

        src = os.path.splitext( file )[0] + '.c'

        #print "file: ", file
        #print "t: ", t
        #print "b: ", b
        #print "d: ", d
        #print "src: ", src

        self.data[ file ] = {}
        fd = self.data[ file ]

        cmd = "gcov -f --object-directory %s %s.c" % ( d, b )

        #print "Executing: ", cmd

        log = os.popen( "gcov -f --object-directory %s %s.c" % ( d, b ),
                        'r').readlines()

        gcov_files = []

        for line in log:
            #print "line: %s" % line

            # Notice lines declaring new .gcov output files.
            m = re.match( "(.*):creating \'(.*).gcov\'", line )
            if m:
                gcov_file = m.group(2) + ".gcov"
                #print "Found new gcov file: ", gcov_file

                gcov_files.append( gcov_file )

            # Notice lines introducing a new function data summary.
            m = re.match( "^Function '(.*)'", line )
            if m:
                function = m.group(1)
                #print "Working on function: %s" % function

            # Notice lines providing coverage data on functions.
            m = re.match( "^Lines executed:([0-9.]+)% of ([0-9]+)", line )
            if m:
                fcov = float( m.group( 1 ) )
                flines = int( m.group( 2 ) )

                fd[function] = { 'covpct' : fcov,
                                 'lines' : flines }

                #print "fd[function]=", fd[function]

        #print "After scanning lines, fd=", fd

        #print "self.data=", self.data

        # Scan the .gcov files to see what we can learn from them.
        for gfile in gcov_files:
            self.scan_gcov_file( file, gfile )

        # Now move the gcov files over to the target dir, since there doesn't
        # seem to be a way to tell gcov to emit them there itself.
        for file in gcov_files:

            # Construct the destination file path.
            dfile = os.path.join( d, file )

            # Clear it out if necessary.
            if os.path.exists( dfile ):
                #print "Imminent collision!"
                os.remove( dfile )

            #print "Moving %s to %s." % ( file, d )
            shutil.move( file, d )

    def scan_gcov_file( self, mfile, gfile ):
        "Scan a .gcov file to glean it's data."

        # The mfile is the "master file", the one which triggered the gcov
        # run.  The gfile is a single specific .gcov output file produced
        # by gcov while processing mfile.  It is unclear why gcov produces
        # multiple .gcov files per master file.  It is also unclear how we
        # should use this data.  It seems like we get a distinct .gcov file
        # for each header used in the code, and the true usage stats for the
        # functions within should be composed by summing across all master
        # files which include a given header.  This needs to be studied and
        # worked through.

        pass

    def report_gcov_data( self ):
        "Summarize what we learned from the gcov coverage analysis."

        print "Results:"

        # Let's collect all the function names.
        all_func_names = {}

        untested = {}

        for file in self.data.keys():
            print "Data for file: ", file
            # print self.data[file]
            untested_funcs = []
            for func in self.data[file].keys():
                all_func_names[ func ] = 0

                print "%40s  %6.2f  %4d" % \
                      ( func, self.data[file][func]['covpct'],
                        self.data[file][func]['lines'] )

                if self.data[file][func]['covpct'] == 0. \
                   and not is_suppressed( file, func ):
                    untested_funcs.append( func )

            if len( untested_funcs ):
                untested[ file ] = untested_funcs

        print
        print "Untested functions:"

        nfiles_with_untested_funcs = 0
        nuntested_funcs = 0

        # We'll collect the names of untested functions in a dictionary,
        # which will help us weed out the duplicates which presumably come
        # from header files included in multiple translation units.
        untested_func_names = {}

        # Emit the table of files and their untested functions.
        for file in untested.keys():
            print "    %s" % file
            nfiles_with_untested_funcs = nfiles_with_untested_funcs + 1

            for func in untested[file]:
                print "        %s" % func
                nuntested_funcs = nuntested_funcs + 1

                untested_func_names[ func ] = 0

        # Final, interesting cummulative data.
        print
        print "Altogether, there are %d untested functions in %d files." % \
              ( nuntested_funcs, nfiles_with_untested_funcs )

        print "Total number of untested function names: ", \
              len( untested_func_names.keys() )

        print "Total number of functions: ", len( all_func_names.keys() )

    def usage( self ):
        "xxx"

        print """nptool cmd args

Commands:
   clean     Wipe out build products.
   build     Build numpy instrumented for coverage analysis.
   test      Run the unit tests, generage coverage data.
   gcov      Perform the coverage analysis, produce report."""

        sys.exit(1)

