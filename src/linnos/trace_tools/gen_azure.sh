#!/bin/bash
#original is 77us arrival
#2x 38
#3x is 25us
#4x is 19

mkdir -p azure
python3 gen.py azure/azure1.trace 0.25 10 25/64 17/64 38
python3 gen.py azure/azure2.trace 0.25 10 25/64 17/64 38
python3 gen.py azure/azure3.trace 0.25 10 25/64 17/64 38