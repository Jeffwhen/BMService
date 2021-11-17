#!/bin/bash

apt-get install libglib2.0-dev python-pip python3-pip
pip2 install absl-py numpy
pip3 install absl-py numpy
pip install absl-py numpy

pushd /workspace/loadgen_build/mlperf_inference/loadgen
CFLAGS="-std=c++14 -O3" python3 setup.py bdist_wheel
pip install --force-reinstall dist/mlperf_loadgen-1.1-cp35-cp35m-linux_x86_64.whl
popd

pip install batchgenerators==0.21
pip install build/SimpleITK-2.0.2-cp35-cp35m-manylinux1_x86_64.whl
pip install medpy

# set encoding
export LANG="en_US.UTF-8"
export PYTHONIOENCODING=utf-8