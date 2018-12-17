#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

class Model:
    def __init__(self, graph_data):
        #TODO
        self.__input = graph_data

    def preProcess(self):
        #TODO
        print("preProcess is finished successfully\n")

    def run(self, run_policy):
        #TODO
        print("run is finished successfully\n")
    
    def showResult(self):
        #TODO
        with open("mockOutput.spt", "r") as output:
            self.__output_data = output.read()
        print(self.__output_data + "\n")

    def export(self, file_name):
        #TODO
        with open(file_name, "w") as output:
            output.write(self.__output_data)
        print("export is finished successfully\n")