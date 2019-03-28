#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

import os, sys

alg_list = {
    "AnomalyDetection":{
        "HOLOSCOPE": "HOLOSCOPE",
        "FRAUDAR": "FRAUDAR"
    },
    "EigenDecompose":{
        "SVDS": "SVDS"
    },
    "TriangleCount":{
        "THINKD": "THINKD"
    }
}

class Engine:
    SINGLEMACHINE = "framework.SingleMachine"

class Model():
    def __init__(self):
        self.name = None
        self.edgelist = None
        self.out_path = "./outputData/"

    def create(self, input_data, model_name):
        self.name = model_name
        self.edgelist = input_data
        return self

    def showResults(self, plot=False):
        #TODO
        pass

class TraingleCount(Model):
    def run(self, algorithm, sampling_ratio, number_of_trials, mode="batch"):
        pass
        
class AnomalyDetection(Model):        
    def run(self, algorithm, k = None):
        alg_name = str(algorithm)
        if alg_name.find(alg_list["AnomalyDetection"]["HOLOSCOPE"]) != -1:
            algorithm(self.edgelist, self.out_path, self.name, k)
        elif alg_name.find(alg_list["AnomalyDetection"]["FRAUDAR"]) != -1:
            algorithm(self.edgelist, self.out_path, self.name)
        else:
            print("Can not find this algorithm!\n")
            sys.exit()

class EigenDecompose(Model):
    def run(self, algorithm, k):
        algorithm(self.edgelist, self.out_path, self.name, k)