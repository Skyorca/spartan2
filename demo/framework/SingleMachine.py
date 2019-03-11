#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

from singleMachine.holoscope.holoscopeFraudDect import Ptype, HoloScope
from singleMachine.holoscope.mytools.ioutil import checkfilegz, saveSimpleListData
from singleMachine.fraudar.greedy import logWeightedAveDegree, np
import scipy.sparse.linalg as slin

class AnomalyDetection:
    def HOLOSCOPE(self, sparse_matrix, source_path, out_path, file_name, k):
        sparse_matrix = sparse_matrix.asfptype()
        ptype = [Ptype.freq]
        tsfile = checkfilegz(source_path + file_name + 'ts.dict')
        if (tsfile):
            ptype.append(Ptype.ts)
        ratefile = checkfilegz(source_path + file_name + 'rate.dict')
        if (ratefile):
            ptype.append(Ptype.rate)
        alg = 'fastgreedy'
        qfun, b = 'exp', 8  # 10 #4 #8 # 32
        tunit = 'd'
        bdres = HoloScope(sparse_matrix, alg, ptype, qfun=qfun, b=b,
                        ratefile=ratefile, tsfile=tsfile, tunit=tunit,
                        nblock=k)
        opt = bdres[-1]
        for nb in xrange(k):
            res = opt.nbests[nb]
            print 'block{}: \n\tobjective value {}'.format(nb + 1, res[0])
            export_file = out_path + file_name + '.blk{}'.format(nb + 1)
            saveSimpleListData(res[1][0], export_file + '.rows')
            saveSimpleListData(res[1][1], export_file + '.colscores')
        return res
    
    def FRAUDAR(self, sparse_matrix, out_path, file_name):
        res = logWeightedAveDegree(sparse_matrix)
        print res
        np.savetxt("%s.rows" % (out_path + file_name, ), np.array(list(res[0][0])), fmt='%d')
        np.savetxt("%s.cols" % (out_path + file_name, ), np.array(list(res[0][1])), fmt='%d')
        print "score obtained is ", res[1]
        return res

class EigenDecompose:
    def SVDS(self, k, sparse_matrix, out_path, file_name):
        res = slin.svds(sparse_matrix, k)
        export_file =out_path + file_name
        saveSimpleListData(res[0], export_file + '.leftSV')
        saveSimpleListData(res[1], export_file + '.singularValue')
        saveSimpleListData(res[2], export_file + '.rightSV')
        return res

class TriangleCount:
    def DOULION(self, p):
        pass