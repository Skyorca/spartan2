#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

#  Project: eaglemine
#    eaglemine_test.py
#      Version:  1.0
#      Goal: Test script
#      Created by @wenchieh  on <12/17/2017>
#

__author__ = 'wenchieh'


import time
import numpy as np
from utils.loader import Loader
from eaglemine_model import EaglemineModel


VERBOSE=True


def load_hcube_weights(histogram_infn, hpos2avgfeat_infn,
                       mode=2, wtcol_index=1, sep=','):
    loader = Loader()
    _, _, hist_arr = loader.load_histogram(histogram_infn)

    nhcubes = len(hist_arr)
    hcube2index = dict(zip(map(tuple, hist_arr[:, :2]), range(nhcubes)))
    hcube_weight = np.zeros(nhcubes)  #np.empty((nhcubes, nfeat))
    with open(hpos2avgfeat_infn, 'r') as fp:
        for line in fp.readlines():
            if line.startswith('#'):  continue
            tok = line.strip().split(sep)
            pos = tuple(map(int, tok[:mode]))
            hcube_weight[hcube2index[pos]] = float(tok[wtcol_index + mode - 1])
        fp.close()

    return hcube_weight


def eaglemine(histogram_infn, node2pos_infn, hpos2avgdeg_infn, outfd,
              degree_index=1, mode=2, mix_component=2):
    if VERBOSE:
        print("Histogram: %s;  Node2pos: %s;  Hpos_avgfeat: %s" % (histogram_infn, node2pos_infn, hpos2avgdeg_infn))
    eaglemodel = EaglemineModel(mode, mix_component)
    eaglemodel.set_vocabulary("dtmnorm")  # "dmgauss"
    eaglemodel.load_histogram(histogram_infn)

    start_tm = time.time()
    eaglemodel.leveltree_build(outfd, prune_alpha=0.80, verbose=VERBOSE)
    end_tm1 = time.time()
    eaglemodel.search(VERBOSE)
    # eaglemodel.save(outfd, '_search_res')
    eaglemodel.post_stitch(verbose=VERBOSE)
    eaglemodel.save(outfd, '_stitch_res')
    eaglemodel.graph_node_cluster(node2pos_infn, outfd+'nodelabel.out', outfd+'hpos2label.out', verbose=VERBOSE)
    end_tm2 = time.time()
    eaglemodel.cluster_histogram(verbose=VERBOSE)

    if degree_index >= 0:
        histpos_avgdeg = load_hcube_weights(histogram_infn, hpos2avgdeg_infn, mode, degree_index)
        eaglemodel.cluster_weighted_suspicious(histpos_avgdeg, verbose=VERBOSE)

    print("running times: {}(s) (water-level tree build {}(s))\n".format(end_tm2 - start_tm, end_tm1 - start_tm))
    print('Eaglemine model done!')




if __name__ == '__main__':
    path = '../output/'
    infn_histogram = 'histogram.out'
    infn_node2pos = 'point2pos.out'
    infn_hpos2avgfeat = 'hpos2avgfeat.out'
    degree_index = 1
    mode, n_components = 2, 2

    eaglemine(path+infn_histogram, path+infn_node2pos, path+infn_hpos2avgfeat, path,
              degree_index, mode, n_components)