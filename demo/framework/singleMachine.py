#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

from singleMachine.holoscope.holoscopeFraudDect import *
from singleMachine. holoscope.run_holoscope import *

class AnomalyDetection:
    def HOLOSCOPE(source, k, file_path, file_name):
        ptype = [Ptype.freq]
        freqfile, tsfile, ratefile = checkfilegz(file_path + file_name + '.edgelist'), None, None
        tsfile = checkfilegz(file_path + file_name + 'ts.dict')
        if (tsfile):
            ptype.append(Ptype.ts)
        ratefile = checkfilegz(file_path + file_name + 'rate.dict')
        if (ratefile):
            ptype.append(Ptype.rate)
        M = loadedge2sm(freqfile, coo_matrix, weighted=True)
        alg = 'fastgreedy'
        qfun, b = 'exp', 8  # 10 #4 #8 # 32
        tunit = 'd'
        M = M.asfptype()
        bdres = HoloScope(M, alg, ptype, qfun=qfun, b=b,
                        ratefile=ratefile, tsfile=tsfile, tunit=tunit,
                        nblock=k)
        opt = bdres[-1]
        for nb in xrange(k):
            res = opt.nbests[nb]
            print 'block{}: \n\tobjective value {}'.format(nb + 1, res[0])
            T = './outputData/' + file_name + '.blk{}'.format(nb + 1)
            saveSimpleListData(res[1][0], T + '.rows')
            saveSimpleListData(res[1][1], T + '.colscores')

class EigenDecompose:
    def SVDS(k):
        pass

class TriangleCount:
    def DOULION(p):
        pass