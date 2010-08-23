#!/bin/bash


export CFLAGS="-DDEBUG -g -Wall" 
export LDFLAGS=-g

pushd numpy/random/mtrand
cython mtrand.pyx
popd

python setupscons.py build install
