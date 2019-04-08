#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

import sys
import os
import system
import importlib
from ioutil import checkfilegz, loadedgelist


#engine
engine = system.Engine()

#model
anomaly_detection = system.AnomalyDetection()
eigen_decompose = system.EigenDecompose()
traingle_count = system.TraingleCount()

'''Input graph format: 
    src1 dst1 value1
    src2 dst2 value2
    ...
'''


def loadTensor(name, path, col_ids = ["uid", "oid", "ts", "rating"], col_types = [int, int, int, float]):
    if path == None:
        path = "inputData/"
    full_path = path + name
    tensor_file = checkfilegz(full_path + '.tensor')

    if tensor_file is None:
        print("Can not find this file, please check the file path!\n")
        sys.exit()

    edgelist = loadedgelist(tensor_file, col_ids, col_types)

    return edgelist


def config(frame_name):
    global ad_policy, tc_policy, ed_policy
    frame = importlib.import_module(frame_name)
    
    #algorithm list
    ad_policy = frame.AnomalyDetection()
    tc_policy = frame.TriangleCount()
    ed_policy = frame.EigenDecompose()