#!/bin/bash


export CFLAGS="-DDEBUG -g -Wall" 
export LDFLAGS=-g

python setupscons.py build install
