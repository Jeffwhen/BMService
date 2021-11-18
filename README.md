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
| BERT-Large | TensorFlow | f1_score=90.874% | SQuAD v1.1 validation set | [from zenodo](https://zenodo.org/record/3733868) [from zenodo](https://zenodo.org/record/3939747) | [BERT-Large](https://github.com/google-research/bert), trained with [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) |  [benchmarks/bert](https://github.com/Jeffwhen/BMService/tree/mlperf/benchmarks/bert) |
| 3D-Unet | PyTorch | **mean = 0.85300** (whole tumor = 0.9141, tumor core = 0.8679, enhancing tumor = 0.7770) | [Fold 1](https://github.com/mlcommons/inference/blob/master/vision/medical_imaging/3d-unet-brats19/folds/fold1_validation.txt) of [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) Training Dataset | [from zenodo](https://zenodo.org/record/3904106) | [Supported Models -> pytorch](https://github.com/mlcommons/inference/blob/master/vision/medical_imaging/3d-unet-brats19/README.md#supported-models) | [benchmarks/3d-unet](https://github.com/Jeffwhen/BMService/tree/mlperf/benchmarks/3d-unet-brats19) |

### BMService installation

``` shell
git submodule update --init --recursive
mkdir build && cd build
cmake ..
make -j
```

### Build loadgen
Run the script below, follow the [reference](https://github.com/mlcommons/inference/blob/master/loadgen/README_BUILD.md)
```
source env.sh
```
