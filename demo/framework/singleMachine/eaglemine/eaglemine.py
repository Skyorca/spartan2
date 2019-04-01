import os
import time
import numpy as np
from src.graph2histogram import histogram_construct, histogram_view
from src.utils.loader import Loader
from src.utils.ploter import plot_clusters
from src.tools.graph import BipartiteGraph, UnipartiteGraph
from src.eaglemine_test import eaglemine

# feature: ind2aut or outd2hub
def graph_to_cluster(edgelist, feature = "outd2hub", isBigraph = True):
    # bipartite graph
    loader = Loader()
    edgelist_data = loader.load_edgelist(edgelist)
    print("load graph edgelist done!")
    print("bipartite graph feature extraction")
    bi_graph = BipartiteGraph()
    bi_graph.set_edgelist(edgelist_data)
    bi_graph.get_node_degree()
    bi_graph.get_hits_score()

    tempdir = "temp/"
    tempfile = {
        "feature": "outd2hub_feature",
        "histogram": "histogram.out",
        "pts2pos": "point2pos.out",
        "hpos2avgfeat": "hpos2avgfeat.out"
    }

    outdir = "outputData/"
    output = {
        "hpos2label": "hpos2label.out",
        "histogram_cluster": "histogram_cluster.png"
    }
    if feature == "outd2hub":
        tempfile["feature"] = "outd2hub_feature"
        with open(tempdir + tempfile["feature"], 'w') as ofp:
            ofp.writelines("# {},{}\n".format(len(bi_graph.src_outd), 0))
            for src in range(len(bi_graph.src_outd)):
                ofp.writelines("{},{}\n".format(bi_graph.src_outd[src], bi_graph.src_hub[src]))
    else:
        tempfile["feature"] = "ind2aut_feature"
        with open(tempdir + tempfile["feature"], 'w') as ofp:
            ofp.writelines("# {},{}\n".format(len(bi_graph.dest_ind), 0))
            for dest in range(len(bi_graph.dest_ind)):
                ofp.writelines("{},{}\n".format(bi_graph.dest_ind[dest], bi_graph.dest_auth[dest]))

    gfeat = np.loadtxt(tempdir + tempfile["feature"], float, comments='#', delimiter=',')
    histogram_construct(gfeat, int(1), tempdir + tempfile["histogram"], tempdir + tempfile["pts2pos"], tempdir + tempfile["hpos2avgfeat"], int(2))
    eaglemine(tempdir + tempfile["histogram"], tempdir + tempfile["pts2pos"], tempdir + tempfile["hpos2avgfeat"], outdir, int(1), int(2), int(2))
    eaglemine_cluster_view(outdir + output["hpos2label"], outdir + output["histogram_cluster"])

def size_relabels(labels):
    clsdic = {}
    for l in labels:
        if l not in clsdic:
            if l != -1: clsdic[l] = len(clsdic)
            else:       clsdic[l] = l
    rlbs = [clsdic[l] for l in labels]
    return np.array(rlbs)

def eaglemine_cluster_view(hpos2label_infn, outfn=None, outlier_label=-1):
    hcube_label = np.loadtxt(hpos2label_infn, int, delimiter=',')
    outs_index = hcube_label[:, -1] == outlier_label
    outs = hcube_label[outs_index, :-1]
    others_cls = hcube_label[~outs_index]
    labels = size_relabels(others_cls[:, -1])
    cls_fig = plot_clusters(others_cls[:, :-1], [], labels, outliers=outs[::-1], ticks=False)
    if outfn is not None:
        cls_fig.savefig(outfn)

if __name__ == "__main__":
    tempdir = "../../../temp/"
    tempfile = {
        "feature": "outd2hub_feature",
        "histogram": "histogram.out",
        "pts2pos": "point2pos.out",
        "hpos2avgfeat": "hpos2avgfeat.out",
        "hpos2label": "hpos2label.out"
    }
    outdir = "../../../outputData/"
    eaglemine(tempdir + tempfile["histogram"], tempdir + tempfile["pts2pos"], tempdir + tempfile["hpos2avgfeat"], outdir, int(1), int(2), int(2))