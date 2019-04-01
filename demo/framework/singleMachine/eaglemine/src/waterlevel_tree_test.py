#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

######################################################
#  METHOD WaterLevel tree test script
#  Author: wenchieh
#
#  Project: eaglemine
#      waterlevel_tree_test.py
#      Version: 
#      Date: November 29 2017
#      Main Contact: Wenchieh Feng (wenchiehfeng.us@gmail.com)
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <11/30/2017>
#

__author__ = 'wenchieh'


import numpy as np
from core.leveltree import LevelTree
from utils.loader import Loader


VERBOSE = False


def leveltree_test(coord2cnt_arr, outpath):
    tsr_arr = np.asarray(coord2cnt_arr)
    values = np.log2(1.0 + tsr_arr[:, -1])
    max_level = np.max(values)
    step = 0.2

    tree = LevelTree()
    tree.build_level_tree(tsr_arr[:, :-1], values, 1.0, max_level, step,
                          verbose=False, outfn=outpath + 'tiny_blob2cnt.out')
    print("Initial build Level-tree!")
    tree.tree_contract(VERBOSE)
    tree.save_leveltree(outpath + 'level_tree_raw.out')

    print("Level-tree pruned!")
    tree.tree_prune(alpha=0.8, verbose=VERBOSE)
    tree.save_leveltree(outpath + 'level_tree_prune.out')

    print("Level-tree node expand!")
    tree.tree_node_expand(VERBOSE)
    tree.save_leveltree(outpath + 'level_tree_final.out')
    print("Refined Level-tree:")
    # tree.dump()



if __name__ == '__main__':
    path = '../output/'
    histogram_infn = 'histogram.out'

    loader = Loader()
    shape, ticks_vec, hist_arr = loader.load_multi_histogram(path+histogram_infn)
    mode = len(shape)
    print("Info: mode:{} shape:{}".format(mode, shape))
    leveltree_test(hist_arr, path)