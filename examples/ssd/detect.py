import bmservice
import coco
import cv2
import numpy as np
import onnxruntime
from dataset import pre_process_coco_resnet34
from main import SUPPORTED_DATASETS

def main():
    #model = '/workspace/mlperf_models/ssd/compilation/compilation.bmodel'
    model = '/workspace/mlperf_models/ssd/ssd-resnet34-int8fp32.bmodel'
    onnx_model = '/workspace/mlperf_models/ssd/resnet34-ssd1200.onnx'
    dataset = 'coco-1200'
    dataset_path = '/workspace/coco2017'
    image_format="NCHW"
    test_id = 206
    count = test_id + 1
    wanted_dataset, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[dataset]
    ds = wanted_dataset(data_path=dataset_path,
                        image_list=None,
                        name=dataset,
                        image_format=image_format,
                        count=count,
                        pre_process=pre_proc,
                        use_cache=True,
                        **kwargs)
    runner = bmservice.BMService(model)
    sample_ids = [test_id]
    use_dataset = False
    if use_dataset:
        ds.load_query_samples(sample_ids)
        img, label = ds.get_samples(sample_ids)
    else:
        #fn = 'sample.jpg'
        fn = '/workspace/coco2017/val2017/000000000139.jpg'
        img = cv2.imread(fn)
        img = pre_process_coco_resnet34(img, (1200, 1200, 3), True)
        img = np.expand_dims(img, axis=0)

    runner.put(img)
    task_id, results, valid = runner.get()
    bboxes, labels, scores = [u.squeeze(0) for u in results]
    if use_dataset:
        fn = ds.get_item_loc(test_id)
    rawImg = cv2.imread(fn)
    ih, iw = rawImg.shape[:2]
    for idx in range(len(labels)):
        if scores[idx] < 0.2:
            continue
        xmin, xmax = (round(v * iw) for v in bboxes[idx][0::2])
        ymin, ymax = (round(v * ih) for v in bboxes[idx][1::2])
        cv2.rectangle(rawImg, (xmin, ymin), (xmax, ymax), (0, 255, 0))
    cv2.imwrite('out.jpg', rawImg)

    ort_session = onnxruntime.InferenceSession(onnx_model)
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)
    bboxes, labels, scores = [u.squeeze(0) for u in ort_outs]
    if use_dataset:
        fn = ds.get_item_loc(test_id)
    rawImg = cv2.imread(fn)
    ih, iw = rawImg.shape[:2]
    for idx in range(len(labels)):
        if scores[idx] < 0.2:
            continue
        xmin, xmax = (round(v * iw) for v in bboxes[idx][0::2])
        ymin, ymax = (round(v * ih) for v in bboxes[idx][1::2])
        cv2.rectangle(rawImg, (xmin, ymin), (xmax, ymax), (0, 255, 0))
    cv2.imwrite('onnxout.jpg', rawImg)

    ds.unload_query_samples(None)

def test_mobilenet_ssd():
    model = '/workspace/mlperf_models/ssd/ssd-mobilenet.fp32.bmodel'
    dataset = 'coco-300'
    dataset_path = '/workspace/coco2017'
    image_format="NHWC"
    test_id = 206
    count = test_id + 1
    wanted_dataset, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[dataset]
    ds = wanted_dataset(data_path=dataset_path,
                        image_list=None,
                        name=dataset,
                        image_format=image_format,
                        count=count,
                        pre_process=pre_proc,
                        use_cache=True,
                        **kwargs)
    runner = bmservice.BMService(model)
    sample_ids = [test_id]
    use_dataset = True
    if use_dataset:
        ds.load_query_samples(sample_ids)
        img, label = ds.get_samples(sample_ids)
    else:
        #fn = 'sample.jpg'
        fn = '/workspace/coco2017/val2017/000000000139.jpg'
        img = cv2.imread(fn)
        img = pre_process_coco_resnet34(img, (1200, 1200, 3), True)
        img = np.expand_dims(img, axis=0)

    runner.put(img)
    task_id, results, valid = runner.get()
    num, bboxes, scores, labels = [u.squeeze(0) for u in results]
    if use_dataset:
        fn = ds.get_item_loc(test_id)
    rawImg = cv2.imread(fn)
    ih, iw = rawImg.shape[:2]
    for idx in range(num):
        if scores[idx] < 0.2:
            continue
        ymin, ymax = (round(v * ih) for v in bboxes[idx][0::2])
        xmin, xmax = (round(v * iw) for v in bboxes[idx][1::2])
        print(labels[idx], xmin, xmax, ymin, ymax)
        cv2.rectangle(rawImg, (xmin, ymin), (xmax, ymax), (0, 255, 0))
    cv2.imwrite('out.jpg', rawImg)

    #ort_session = onnxruntime.InferenceSession(onnx_model)
    #ort_inputs = {ort_session.get_inputs()[0].name: img}
    #ort_outs = ort_session.run(None, ort_inputs)
    #bboxes, labels, scores = [u.squeeze(0) for u in ort_outs]
    #if use_dataset:
    #    fn = ds.get_item_loc(test_id)
    #rawImg = cv2.imread(fn)
    #ih, iw = rawImg.shape[:2]
    #for idx in range(len(labels)):
    #    if scores[idx] < 0.2:
    #        continue
    #    xmin, xmax = (round(v * iw) for v in bboxes[idx][0::2])
    #    ymin, ymax = (round(v * ih) for v in bboxes[idx][1::2])
    #    cv2.rectangle(rawImg, (xmin, ymin), (xmax, ymax), (0, 255, 0))
    #cv2.imwrite('onnxout.jpg', rawImg)

    ds.unload_query_samples(None)

if __name__ == '__main__':
    #main()
    test_mobilenet_ssd()

