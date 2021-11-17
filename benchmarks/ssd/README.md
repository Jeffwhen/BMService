MLPerf SSD Benchmark
====================

### Prerequisites

Download ssd-mobilenet/TensorFlow and ssd-resnet34/ONNX from [MLPerf inference repo](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection#supported-models).

### Model compilation

```
python3 parse_net.py ./resnet34-ssd1200.onnx
```

To compile ssd-mobilenet on the other hand, is rather tricky. In order to compile it, we have to use upstream tool script to simplify it first. In order to utilize the script, we have to use tf1. [Docker image](https://hub.docker.com/r/tensorflow/tensorflow/) is recommended.

```
git clone https://github.com/tensorflow/models.git
```

Then install it according to [google's guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md#python-package-installation). Note we have to use tf1 version. Behold.

```
python3 -m pip install tf-slim
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf1/setup.py . # IMPORTANT, select tf1
python -m pip install --use-feature=2020-resolver .

# Simplify the graph
python3 /path/to/tf-models/research/object_detection/export_tflite_ssd_graph.py --input_type image_tensor --pipeline_config_path ssd_mobilenet_v1_coco_2018_01_28/pipeline.config -trained_checkpoint_prefix ssd_mobilenet_v1_coco_2018_01_28/model.ckpt --output_directory ssd_mobilenet_v1_coco_2018_01_28.simplifiedgraph --add_postprocessing_op=false
```

Now that we have a simplified sane network, we can use tf frontend to compile it.

```
python3 parse_net.py ssd_mobilenet_v1_coco_2018_01_28.simplifiedgraph/tflite_graph.pb
```

### Calibration

```
protoc ./caffe.proto --python_out=./ # Gen protobuf python
python3 calib_lmdb_resnet50.py /workspace/data/coco2017/train2017/ ./coco_cal_images_list.txt ./coco_calib_1200x1200.lmdb
cp ssd-resnet34_bmneto_test_fp32.prototxt ssd-resnet34_bmneto_calib_fp32.prototxt
```

Now edit `ssd-resnet34_bmneto_calib_fp32.prototxt`'s input layer as following.

```
layer {
  name: "image"
  type: "Data"
  top: "image"
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
    source: "/path/to/coco_calib.lmdb"
    batch_size: 1
    backend: LMDB
  }
}
```

Quantize and build bmodel.

```
calibration_use_pb quantize -model=./ssd-resnet34_bmneto_calib_fp32.prototxt -weights=./ssd-resnet34_bmneto.fp32umodel -iterations 20 -fpfwd_outputs Conv_338,Conv_360,Conv_382,Conv_404,Conv_426,Conv_448,Conv_349,Conv_371,Conv_393,Conv_415,Conv_437,Conv_459
bmnetu -model ./ssd-resnet34_bmneto_deploy_int8_unique_top.prototxt -weight ./ssd-resnet34_bmneto.int8umodel -input_as_fp32 image -output_as_fp32 Concat_659,Softmax_660
```

```
protoc ./ufw.proto --python_out=./
python3 calib_lmdb_mobilenet.py /workspace/data/coco2017/train2017 ./coco_cal_images_list.txt ./coco_calib_300x300.lmdb
cp ssd-mobilenet_bmnett_test_fp32.prototxt ssd-mobilenet_bmnett_calib_fp32.prototxt
```

Edit `ssd-mobilenet_bmnett_calib_fp32.prototxt`'s input layer as following.

```
layer {
  name: "normalized_input_image_tensor"
  type: "Data"
  top: "normalized_input_image_tensor"
  data_param {
    source: "/path/to/coco_calib_300x300.lmdb"
    batch_size: 1
    backend: LMDB
  }
}
```

Quantize and build bmodel.

```
calibration_use_pb quantize -model ./ssd-mobilenet_bmnett_calib_fp32.prototxt -weights ./ssd-mobilenet_bmnett.fp32umodel -iterations 500
bmnetu -model ./ssd-mobilenet_bmnett_deploy_int8_unique_top.prototxt -weight ./ssd-mobilenet_bmnett.int8umodel -net_name ssd-mobilenet-int8 -target BM1684 -input_as_fp32 normalized_input_image_tensor -output_as_fp32 raw_outputs/class_predictions,raw_outputs/box_encodings
```

### Dataset

You should download `coco2017` val set and annotations. Make sure it is organized as following directory structure.

```
.
|-- annotations
|   |-- captions_train2017.json
|   |-- captions_val2017.json
|   |-- image_info_test-dev2017.json
|   |-- image_info_test2017.json
|   |-- image_info_unlabeled2017.json
|   |-- instances_train2017.json
|   |-- instances_val2017.json
|   |-- person_keypoints_train2017.json
|   |-- person_keypoints_val2017.json
|   |-- stuff_train2017.json
|   |-- stuff_train2017_pixelmaps.zip
|   |-- stuff_val2017.json
|   `-- stuff_val2017_pixelmaps.zip
|-- cocoval.list
|-- train2017
|   |-- 000000391895.jpg
|   |-- 000000522418.jpg
|   `-- 000000184613.jpg
`-- val2017
    |-- 000000000139.jpg
    |-- 000000581615.jpg
    `-- 000000581781.jpg
```

### Inference

```
python3 main.py --dataset coco-1200 --profile ssd-1200 --scenario SingleStream --model /workspace/ssd-resnet34.fp32.bmodel --data-format NCHW --dataset-path /workspace/coco2017/ --accuracy
python3 main.py --dataset coco-300 --profile ssd-mobilenet --scenario SingleStream --model /workspace/ssd-mobilenet.fp32.bmodel --dataset-path /workspace/coco2017/ --accuracy
```

### TODO

+ Current post-process in `interface.cpp` is rather hacky
+ Write multi-batch c++ harness.
+ ssd-mobilenet suffers precision loss when quantized
