#!/bin/bash

pip install batchgenerators==0.21
pip install build/SimpleITK-2.0.2-cp35-cp35m-manylinux1_x86_64.whl
pip install medpy

# set encoding
export LANG="en_US.UTF-8"
export PYTHONIOENCODING=utf-8