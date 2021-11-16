MLPerf Resnet50 Benchmark
=========================

### Prerequisites

Download resnet50-v1.5/PyTorch from [MLPerf inference repo](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection#supported-models).

### Model compilation

```shell
python3 parse_net.py resnet50-19c8e357.pth # Compile fp32 bmodel and UFW models for calibration
```

### Calibration

```
protoc ./caffe.proto --python_out=./ # Gen protobuf python
python3 calib_lmdb.py /workspace/ILSVRC2012_val/ ./cal_image_list_option_1.txt ./resnet50_calib.lmdb # Generate calib lmdb dataset
cp torch-resnet50-v1_bmnetp_test_fp32.prototxt torch-resnet50-v1_bmnetp_calib_fp32.prototxt
```

Output bmodel resides in `compilation` directory.

Now edit `torch-resnet50-v1_bmnetp_calib_fp32.prototxt`'s input layer as following.

```
layer {
  name: "input.1"
  type: "Data"
  top: "input.1"
  transform_param {
    transform_op {
      op: STAND
      mean_value: 123.675
      mean_value: 116.28
      mean_value: 103.53
      scale: 0.017124753831663668
      scale: 0.01750700280112045
      scale: 0.017429193899782137
    }
  }
  data_param {
    source: "/path/to/calib.lmdb"
    batch_size: 1
    backend: LMDB
  }
}
```

Quantize and build bmodel.

```
calibration_use_pb quantize -model ./torch-resnet50-v1_bmnetp_calib_fp32.prototxt -weights ./torch-resnet50-v1_bmnetp.fp32umodel
bmnetu -model ./torch-resnet50-v1_bmnetp_deploy_int8_unique_top.prototxt -weight ./torch-resnet50-v1_bmnetp.int8umodel -input_as_fp32 input.1 -output_as_fp32 45
```

### Inference

```
python3 main.py --dataset imagenet_pytorch --dataset-path /workspace/ILSVRC2012_val --data-format NCHW --scenario SingleStream --model /workspace/resnet50.fp32.bmodel --accuracy
```

### TODO

Write multi-batch c++ harness.

