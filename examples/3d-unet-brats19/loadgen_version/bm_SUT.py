#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2021/11/9 3:55 下午
import array
import os
import sys
import threading
import time

import bmservice
import mlperf_loadgen as lg
import numpy as np

from brats_QSL import get_brats_QSL

sys.path.insert(0, os.getcwd())


class BMServiceSUT(object):
    def __init__(self, model_path, preprocessed_data_dir, performance_count, batch_size):
        print("Loading bmodel...")
        self.runner = bmservice.BMService(model_path)
        # After model conversion output name could be any
        # So we are looking for output with max number of channels
        self.batch_size = batch_size

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries, self.process_latencies)
        self.qsl = get_brats_QSL(preprocessed_data_dir, performance_count)
        print("Finished constructing SUT.")
        self.query_count = 0
        # task id sample id pair info
        self.task_map = {}
        self.map_lock = threading.Lock()

    def wait_result(self):
        print('Waiting results')
        while not (self.query_count <= 0):
            task_id, values, valid = self.runner.try_get()
            if task_id == 0:
                time.sleep(0.00001)
                continue

            self.map_lock.acquire()
            ids = self.task_map[task_id]
            del self.task_map[task_id]
            self.map_lock.release()

            outputs = values[-1].astype(np.float16)
            print("Got task_id={:d} with shape={}".format(task_id, outputs.shape), flush=True)

            responses = []
            for sample_id, output in zip(ids, outputs):
                response_array = array.array("B", output.tobytes())
                bi = response_array.buffer_info()
                response = lg.QuerySampleResponse(sample_id, bi[0], bi[1])
                responses.append(response)
            self.query_count -= 1
            lg.QuerySamplesComplete(responses)
        print('Come out of waiting results')

    def issue_queries(self, query_samples):
        query_count = len(query_samples)
        print('Query count: {}'.format(query_count))
        # make up batch
        if len(query_samples) % self.batch_size != 0:
            padding_sample_num = self.batch_size - (len(query_samples) % self.batch_size)
            query_samples += query_samples[-1 * padding_sample_num:]

        # process samples
        datas = [self.qsl.get_features(qs.index) for qs in query_samples]
        datas = [data[np.newaxis, ...] for data in datas]
        self.query_count += len(datas) // self.batch_size
        # receiver thread focus on get result
        receiver = threading.Thread(target=self.wait_result)
        receiver.start()

        for idx in range(0, len(datas), self.batch_size):
            st = idx * self.batch_size
            part_datas = datas[st: st + self.batch_size]
            part_datas_array = np.concatenate(part_datas, axis=0)
            task_id = self.runner.put(part_datas_array)
            print("Put task_id {:d} with shape = {:}".format(task_id, part_datas_array.shape), flush=True)
            batch_query_samples = query_samples[st: st + self.batch_size]
            self.map_lock.acquire()
            self.task_map[task_id] = [qs.id for qs in batch_query_samples]
            self.map_lock.release()

    def flush_queries(self):
        print("flush_queries finish", flush=True)

    def process_latencies(self, latencies_ns):
        print("Average latency: ")
        print(np.mean(latencies_ns))
        print("Median latency: ")
        print(np.percentile(latencies_ns, 50))
        print("90 percentile latency: ")
        print(np.percentile(latencies_ns, 90))


def get_bm_sut(model_path, preprocessed_data_dir, performance_count, batch_size):
    return BMServiceSUT(model_path, preprocessed_data_dir, performance_count, batch_size)
