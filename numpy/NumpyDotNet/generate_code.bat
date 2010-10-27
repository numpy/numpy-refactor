@echo off

cd "%1%\..\core"
"%IRONPYTHON_HOME%/ipy.exe" code_generators\generate_umath.py


