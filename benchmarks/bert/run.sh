#!/bin/bash

# export BMSERVICE_USE_DEVICE="0 1 2"
export PYTHONPATH=/workspace/BMService/python:$PYTHONPATH

bmodel_path=/workspace/models/bert/squad_fp32/compilation.bmodel
python3 bm_run.py --backend bm \
                  --model ${bmodel_path} \
                  --scenario SingleStream \
                  --performance_count 20 \
                  --accuracy \
                  2>&1 | tee squad_fp32.log