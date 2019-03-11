#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao


import system
import importlib


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


def SFrame(file_name):
    return file_name


def config(frame_name):
    global ad_policy, tc_policy, ed_policy
    frame = importlib.import_module(frame_name)
    #algorithm list
    ad_policy = frame.AnomalyDetection()
    tc_policy = frame.TriangleCount()
    ed_policy = frame.EigenDecompose()