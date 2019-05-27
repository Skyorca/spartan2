#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

#  Project: eaglemine-master
#       function: provide visualization tools for viewing the clustering results
#    ealgemine_cluster_view.py
#      Version:  1.0
#      Goal: Subroutine script
#      Created by @wenchieh  on <12/17/2018>
#

__author__ = 'wenchieh'


import numpy as np
from scipy.sparse import csr_matrix
from utils.ploter import plot_heatmap, plot_clusters

from desc.gaussian_describe import GaussianDescribe
from desc.dtmnorm_describe import DTMNormDescribe
from utils.loader import Loader
from utils.ploter_aux import plot_heatmap_ellipse_covs


def size_relabels(labels):
    clsdic = {}
    for l in labels:
        if l not in clsdic:
            if l != -1: clsdic[l] = len(clsdic)
            else:       clsdic[l] = l
    rlbs = [clsdic[l] for l in labels]
    return np.array(rlbs)



def eaglemine_describe_view(histogram_infn, describe_infn, xlabel, ylabel, outfn):
    loader = Loader()
    desc_parms = loader.load_describes_parms(describe_infn, DTMNormDescribe, mode=2)
    h_shape, ticks_vec, hist_arr = loader.load_histogram(histogram_infn)
    csr_mat = csr_matrix((hist_arr[:, -1], (hist_arr[:, 0], hist_arr[:, 1])), shape=h_shape, dtype=int)

    # plot_heatmap(ticks_vec[1], ticks_vec[0], csr_mat.toarray(), xlabel=xlabel, ylabel=ylabel, outfn=outfn)
    plot_heatmap_ellipse_covs(ticks_vec[1], ticks_vec[0], csr_mat.toarray(), desc_parms, base=10,
                              scales=(1.5, 3), xlabel=xlabel, ylabel=ylabel, outfn=outfn)


def eaglemine_cluster_view(hpos2label_infn, outfn=None, outlier_label=-1):
    hcube_label = np.loadtxt(hpos2label_infn, int, delimiter=',')
    outs_index = hcube_label[:, -1] == outlier_label
    outs = hcube_label[outs_index, :-1]
    others_cls = hcube_label[~outs_index]
    labels = size_relabels(others_cls[:, -1])
    cls_fig = plot_clusters(others_cls[:, :-1], [], labels, outliers=outs[::-1], ticks=False)
    if outfn is not None:
        cls_fig.savefig(outfn)
    # cls_fig.show()



if __name__ == '__main__':
    path = '../output/'
    infn_histogram = 'histogram.out'
    infn_describe = 'describe_stitch_res.out'
    infn_hpos2label = 'hpos2label.out'
    axis_labels = ["Hubness", "Out-degree"]

    eaglemine_describe_view(path+infn_histogram, path+infn_describe, axis_labels[0], axis_labels[1],
                            path+'histogram_describes.png')
    eaglemine_cluster_view(path+infn_hpos2label, path+'histogram_cluster.png')