#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

import os

class Engine:
    SINGLEMACHINE = "framework.SingleMachine"
    POWERGRAPH = "POWERGRAPH"

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
        self.__name = None
        self.file_path = "./inputData/"
        self.file_name = None


    def create(self, file_name, model_name):
        self.__name = model_name
        if (file_name.find('/') == -1):
            self.file_name = file_name
        else:
            self.file_path, self.file_name = os.path.split(file_name)
        return self

    def preProcess(self):
        #TODO
        print "preProcess is finished successfully\n"

    def showResults(self, plot=False):
        #TODO
        pass

class TraingleCount(Model):
    def run(self, algorithm, p):
        #TODO
        print "run is finished successfully\n"

class AnomalyDetection(Model):
    def run(self, algorithm, k):
        algorithm(k, self.file_path, self.file_name)

class EigenDecompose(Model):
    def run(self, algorithm, k):
        #TODO
        print "run is finished successfully\n"