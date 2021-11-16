import caffe_pb2
import lmdb
import cv2
import numpy as np
from dataset import resize_with_aspectratio, center_crop

def convert_calib_list(data_dir, fn, out_dir):
    from PIL import Image
    import torchvision.transforms.functional as F
    output_height = output_width = 224
    with open(fn) as f:
        total_num = sum(1 for i in f) * 1.1
    env = lmdb.open(out_dir, map_size=output_height*output_width*3*total_num)
    with env.begin(write=True) as txn, open(fn) as lf:
        for line in lf:
            line = line.strip(' \n')
            img = cv2.imread(os.path.join(data_dir, line))
            is_torch = True
            datum = caffe_pb2.Datum()
            datum.height, datum.width, datum.channels = 224, 224, 3
            if is_torch:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = F.resize(img, 256, Image.BILINEAR)
                img = F.center_crop(img, 224)
                img = np.asarray(img, dtype=np.uint8)
            else:
                cv2_interpol = cv2.INTER_AREA
                img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2_interpol)
                img = center_crop(img, output_height, output_width)
            cv2.imwrite(os.path.join('preview', line), img)
            img = img.transpose([2, 0, 1])
            datum.data = img.tobytes()
            txn.put(line.encode(), datum.SerializeToString())

if __name__ == '__main__':
    import sys
    import os
    if len(sys.argv) != 4:
        print('{} <dataset_dir> <calib_list.txt> <lmdb_dir>'.format(sys.argv[0]))
        sys.exit(1)
    convert_calib_list(*sys.argv[1:])

