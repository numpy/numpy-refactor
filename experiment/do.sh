#!/bin/bash

python setup.py build_ext --inplace

export LD_LIBRARY_PATH="$HOME/usr/lib"

python main.py
