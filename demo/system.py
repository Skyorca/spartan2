#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

import os

class Engine:
    SINGLEMACHINE = "framework.SingleMachine"
    POWERGRAPH = "POWERGRAPH" #TODO

#tc:triangle count
class TcPolicy:
    def __init__(self, engine):
        self.DOULION = engine.TriangleCount().doulion

#ad:anomaly detection
class AdPolicy:
    def __init__(self, engine):
        self.HOLOSCOPE = engine.AnomalyDetection().holoscope

#ed:eigen decomposition
class EdPolicy:
    SVDS = "SVDS"

class Model():
    def __init__(self):
        self.name = None
        self.source_path = None
        self.out_path = "./outputData/"
        self.file_name = None


    def create(self, input_data, model_name):
        self.name = model_name
        self.source_path = input_data[0]
        self.file_name = input_data[1]
        self.sparse_matrix = input_data[2]
        return self

    def showResults(self, plot=False):
        #TODO
        pass

class TraingleCount(Model):
    def run(self, algorithm, p):
        #TODO
        print "run is finished successfully\n"

class AnomalyDetection(Model):
    def run(self, algorithm, k):
        self.result = algorithm(self.sparse_matrix, self.source_path, self.out_path, self.file_name, k)
    def run(self, algorithm):
        self.result = algorithm(self.sparse_matrix, self.out_path, self.file_name)

class EigenDecompose(Model):
    def run(self, algorithm, k):
        self.result = algorithm(k, self.sparse_matrix, self.out_path, self.file_name)