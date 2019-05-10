#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

from singleMachine.holoscope.holoscopeFraudDect import Ptype, HoloScope
from singleMachine.fraudar.greedy import logWeightedAveDegree, np
from singleMachine.eaglemine.eaglemine import graph_to_cluster, map_label_to_point, get_subgraph
from singleMachine.ioutil import saveSimpleListData, loadedgelist2sm
import scipy.sparse.linalg as slin


class AnomalyDetection:
    def HOLOSCOPE(self, edgelist, out_path, file_name, k):
        sparse_matrix = loadedgelist2sm(edgelist[2])
        sparse_matrix = sparse_matrix.asfptype()
        ptype = [Ptype.freq]
        alg = 'fastgreedy'
        qfun, b = 'exp', 8  # 10 #4 #8 # 32
        tunit = 'd'
        bdres = HoloScope(sparse_matrix, alg, ptype, qfun=qfun, b=b, tunit=tunit, nblock=k)
        opt = bdres[-1]
        for nb in xrange(k):
            res = opt.nbests[nb]
            print 'block{}: \n\tobjective value {}'.format(nb + 1, res[0])
            export_file = out_path + file_name + '.blk{}'.format(nb + 1)
            saveSimpleListData(res[1][0], export_file + '.rows')
            saveSimpleListData(res[1][1], export_file + '.colscores')
    
    def FRAUDAR(self, edgelist, out_path, file_name):
        sparse_matrix = loadedgelist2sm(edgelist[2])
        sparse_matrix = sparse_matrix.asfptype()
        res = logWeightedAveDegree(sparse_matrix)
        print res
        np.savetxt("%s.rows" % (out_path + file_name, ), np.array(list(res[0][0])), fmt='%d')
        np.savetxt("%s.cols" % (out_path + file_name, ), np.array(list(res[0][1])), fmt='%d')
        print "score obtained is ", res[1]

    def EAGLEMINE(self, edgelist, feature, isBigraph):
        graph_to_cluster(edgelist[2], feature, isBigraph)
        label_point = map_label_to_point("temp/point2pos.out", "outputData/hpos2label.out")
        subgraph = get_subgraph(edgelist, label_point, feature)

        return subgraph

class EigenDecompose:
    def SVDS(self, edgelist, out_path, file_name, k):
        sparse_matrix = loadedgelist2sm(edgelist[2])
        sparse_matrix = sparse_matrix.asfptype()
        res = slin.svds(sparse_matrix, k)
        export_file =out_path + file_name
        saveSimpleListData(res[0], export_file + '.leftSV')
        saveSimpleListData(res[1], export_file + '.singularValue')
        saveSimpleListData(res[2], export_file + '.rightSV')

class TriangleCount:
    #arg mode: batch or incremental
    def THINKD(self, in_path, out_path, sampling_ratio, number_of_trials, mode):
        pass