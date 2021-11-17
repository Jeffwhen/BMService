import ufw_pb2
import lmdb
import cv2
import numpy as np
from dataset import maybe_resize

def convert_calib_list(data_dir, fn, out_dir):
    dims = [300, 300, 3]
    with open(fn) as f:
        total_num = sum(1 for i in f) * 1.1
    env = lmdb.open(out_dir, map_size=dims[0]*dims[1]*3*total_num)
    index = 0
    with env.begin(write=True) as txn, open(fn) as lf:
        for line in lf:
            if index >= 500:
                break
            index += 1
            line = line.strip(' \n')
            print(os.path.join(data_dir, line))
            img = cv2.imread(os.path.join(data_dir, line))
            img = maybe_resize(img, dims)

            datum = ufw_pb2.Datum()
            cv2.imwrite(os.path.join('preview', line), img)
            img = img.astype(np.float32)
            datum.float_data.extend(img.flat)
            datum.shape.dim.extend(dims)
            txn.put(line.encode(), datum.SerializeToString())

if __name__ == '__main__':
    import sys
    import os
    if len(sys.argv) != 4:
        print('{} <dataset_dir> <calib_list.txt> <lmdb_dir>'.format(sys.argv[0]))
        sys.exit(1)
    convert_calib_list(*sys.argv[1:])

