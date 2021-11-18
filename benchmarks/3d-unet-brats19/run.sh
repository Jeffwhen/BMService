#!/bin/bash

# export BMSERVICE_USE_DEVICE="0 1 2"
export PYTHONPATH=/workspace/BMService/python:$PYTHONPATH

bmodel_path=/workspace/models/3dunet/compilation.bmodel

echo "bmodel path: ${bmodel_path}"
python3 bm_run.py --backend bm \
                  --model /workspace/models/3dunet/compilation.bmodel \
                  --scenario SingleStream \
                  --performance_count 4 \
                  --accuracy \
                  2>&1 | tee 3dunet_fp32.log