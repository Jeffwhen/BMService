bm1684 MLPerf inference codebase
================================

### Introduction

This code base implements MLPerf SUT.

### Benchmarks

| model | framework | accuracy | dataset | model link | model source | sut |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | pytorch | 76.014% | imagenet2012 validation | [from zenodo](https://zenodo.org/record/4588417/files/resnet50-19c8e357.pth) | [from TorchVision](https://github.com/pytorch/vision/blob/v0.8.2/torchvision/models/resnet.py) | [benchmarks/resnet50](https://github.com/Jeffwhen/BMService/tree/mlperf/benchmarks/resnet50) |
| ssd-mobilenet 300x300 | tensorflow | mAP 0.23 | coco resized to 300x300 | [from tensorflow](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | [from tensorflow](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | [benchmarks/ssd](https://github.com/Jeffwhen/BMService/tree/mlperf/benchmarks/ssd) |
| ssd-resnet34 1200x1200 | onnx | mAP 0.20 | coco resized to 1200x1200 | from zenodo [opset-8](https://zenodo.org/record/3228411/files/resnet34-ssd1200.onnx) | [from mlperf](https://github.com/mlperf/inference/tree/master/others/cloud/single_stage_detector) converted using the these [instructions](https://github.com/BowenBao/inference/tree/master/cloud/single_stage_detector/pytorch#6-onnx) | [benchmarks/ssd](https://github.com/Jeffwhen/BMService/tree/mlperf/benchmarks/ssd) |

### BMService installation

``` shell
git submodule update --init --recursive
mkdir build && cd build
cmake ..
make -j
```

