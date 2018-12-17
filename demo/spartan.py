#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

from spartan_engine import Model
from run_policy import Policy

run_policy = Policy()
__graph_list = {}

'''Input data format: 
        src1 dst1 value1
        src2 dst2 value2
        ...
'''
def showList():
    print("=== GRAPH LIST ===\n")
    for graph_name in sorted(__graph_list):
        print("- " + graph_name)
    print("\n")

def addGraph(file_name, graph_name):
    with open(file_name, "r") as input:
        __graph_list[graph_name] = input.read()

def showGraph(graph_name):
    if(graph_name in __graph_list):
        print(__graph_list[graph_name])
    else:
        print("Not Found Graph With This Name!")
    print("\n")

def createModel(graph_name):
    return Model(__graph_list[graph_name])