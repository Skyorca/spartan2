#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

# EagleMine: Vision-Guided Mining in Large Graphs
# Authors: Wenjie Feng, Shenghua Liu, Christos Faloutsos, Bryan Hooi,
#                 and Huawei Shen, and Xueqi Cheng
#
#  Project: eaglemine
#      eaglemine_model.py
#      Version:  1.0
#      Date: December 17 2017
#      Main Contact: Wenchieh Feng (wenchiehfeng.us@gmail.com)
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <12/17/2017>
#
#      Main contributor:  Wenjie Feng, Shenghua Liu.
#

__author__ = 'wenchieh'


import operator
import numpy as np
from collections import deque
from utils.loader import Loader
from utils.mdlbase import MDLBase
from core.leveltree import LevelTree
from desc.dtmnorm_describe import DTMNormDescribe
from desc.truncated_gaussian import TruncatedGaussian
from desc.gaussian_describe import GaussianDescribe
from desc.normal_gaussian import NormalGaussian
from desc.statistic_hypothesis_test import StatisticHypothesisTest
from desc.statistic_hypothesis_test_truncate import StatisticHypothesisTestTruncate


class EaglemineModel(object):
    _valid_vocabularies_ = ['dtmnorm', 'dmgauss']

    _raw_leveltree_outs_ = 'level_tree.out'
    _prune_leveltree_outs_ = 'level_tree_prune.out'
    _expand_leveltree_outs_ = 'level_tree_expand.out'

    def __init__(self, mode=2, mixture_components=2, power_base=2):
        self.leveltree = None
        self.describes = None
        self.mode = mode
        self._hpos = None
        self._count = None
        self._total = None
        self.shape = None
        self.descvoc = None
        self._mixcomps = mixture_components
        self.power_base = power_base

    def set_vocabulary(self, init_parames='dtmnorm'):
        if init_parames == 'dtmnorm':
            self.descvoc = {"name": 'dtmnorm', "voc": DTMNormDescribe, "dist": TruncatedGaussian}
        elif init_parames == 'dmgauss':
            self.descvoc = {"name": 'dmgauss', "voc": GaussianDescribe, "dist": NormalGaussian}
        else:
            raise ValueError("Unimplemented initialization method {}, "
                             "valid vocabularies {}".format(init_parames, self._valid_vocabularies_))

    def load_histogram(self, infn_hist):
        loader = Loader()
        shape, ticks, hist_arr = loader.load_histogram(infn_hist)
        self._hpos = hist_arr[:, :-1]
        self._count = hist_arr[:, -1]
        n_pos, n_dim = self._hpos.shape
        if n_dim != self.mode:
            raise ValueError("Input histogram dimension does match with the initial dimension.")
        self._total = n_pos
        self.shape = shape

    def set_histogram(self, pos2cnt):
        self._hpos = np.asarray(pos2cnt[:, :-1])
        self._count = np.asarray(pos2cnt[:, -1])
        n_pos, n_dim = self._hpos.shape
        if n_dim != self.mode:
            raise ValueError("Input histogram dimension does match with the initial dimension.")
        self._total = n_pos
        self.shape = np.max(self._hpos, axis=0)

    def load_leveltree(self, infn, sep=',', verbose=True):
        self.leveltree = LevelTree()
        self.leveltree.load_leveltree(infn, sep, verbose)

    def load_describes(self, infn, sep=';', P=2, verbose=True):
        self.describes = dict()
        hpos2cnt = dict()
        for k in range(self._total):
            hpos2cnt[tuple(self._hpos[k])] = self._count[k]

        with open(infn, 'r') as ifp:
            for line in ifp:
                line = line.strip()
                splits = line.find(sep)
                node_id = int(line[:splits])
                desc = self.descvoc["voc"](self.mode)
                desc.load(line[splits+1:])
                tree_node = self.leveltree.get_node(node_id)
                node_counts = list()
                for hcb in tree_node.get_covers():
                    node_counts.append(hpos2cnt.get(tuple(hcb)))
                values = np.array(node_counts)
                values = np.log(values + 1) / np.log(P)
                desc.set_data(tree_node.get_covers(), values)
                self.describes[node_id] = desc
            ifp.close()

        if verbose:
            print('load describe done!')

    def leveltree_build(self, outfd, step=0.2, prune_alpha=0.95,
                        min_level=1.0, max_level=None, verbose=True):
        values = np.log2(1.0 + np.asarray(self._count, float))
        if max_level is None or max_level < min_level or max_level > np.max(values):
            max_level = np.max(values)

        self.leveltree = LevelTree()
        self.leveltree.build_level_tree(self._hpos, values, min_level, max_level, step,
                                        verbose=verbose, outfn = outfd+'tiny_blob2cnt.out')
        self.leveltree.tree_contract(verbose=verbose)
        if verbose:
            print("Info: Raw level-tree:")
            self.leveltree.dump()
        self.leveltree.tree_prune(alpha=prune_alpha, verbose=verbose)
        if verbose:
            print("Info: Pruned level-tree:")
            self.leveltree.dump()
        self.leveltree.tree_node_expand(verbose)
        self.leveltree.save_leveltree(outfd + self._expand_leveltree_outs_)
        if verbose:
            print("Info: Expanded level-tree:")
            self.leveltree.dump()

    def _describe_singular_check_(self, hcubes):
        hcubes = np.asarray(hcubes)
        ndims = hcubes.ndim
        sing_dims = list()
        for dm in range(ndims):
            if len(np.unique(hcubes[:, dm])) <= 1:
                sing_dims.append(dm)

        if len(sing_dims) > 0:
            raise ValueError("{}-dimensional data degenerate in "
                             "{}-st dimension(s)".format(ndims, sing_dims))

    def describe_all(self):
        self.describes = dict()
        hpos2cnt = dict()
        for k in range(self._total):
            hpos2cnt[tuple(self._hpos[k])] = self._count[k]

        # tree nodes will be fitted with mixture distribution
        heavynode2level = self.leveltree.get_heavynodes()
        nodes = self.leveltree.get_nodes()  # self.leveltree.get_leaves()
        for ndid in nodes:
            tree_node = self.leveltree.get_node(ndid)
            hcubes = tree_node.get_covers()
            node_counts = [hpos2cnt.get(tuple(hcb)) for hcb in hcubes]
            is_mix = ndid in heavynode2level.keys()

            ## logP_count
            values = np.array(node_counts)
            values = np.log(values + 1) / np.log(self.power_base)

            comps = self._mixcomps if is_mix else 1
            desc = self.descvoc["voc"](self.mode, is_mix, comps)
            if self.descvoc["name"] == 'dtmnorm':
                desc.set_bounds()

            self._describe_singular_check_(hcubes)
            desc.fit(hcubes, values)
            self.describes[ndid] = desc

    def dump(self):
        print("Information: ")
        print("#mode: {}, input-shape: {}".format(self.mode, self.shape))
        print("#non-zeros: {}, #totals: {}".format(len(self._hpos), np.sum(self._count)))
        print("Level-tree information:")
        self.leveltree.dump()
        print("Leaves describe information:")
        for id, desc in self.describes.items():
            print("{}: {}".format(id, str(desc)))

        print("done!")

    def save(self, outfd, suffix=''):
        describe_outfn = 'describe%s.out' % suffix
        # self.leveltree.save_leveltree(outfd+'level_tree_expand-00.out')
        with open(outfd + describe_outfn, 'w') as ofp:
            desc_ids = sorted(self.describes.keys())
            for id in desc_ids:
                desc = self.describes[id]
                line = str(id) + ";" + str(desc)
                ofp.writelines(line + '\n')
            ofp.close()
        print("save done!")


    def _model_hcubes_prob_(self, pos_left, pos_right, paras, is_mix):
        npts = len(pos_left)
        mus, covs, weights = paras["mus"], paras["covs"], paras["weights"]
        probs = None
        if is_mix:
            nmix = len(mus)
            mix_dists = list()
            for i in range(nmix):
                if self.descvoc["name"] == "dtmnorm":
                    desc = self.describes.values()[0]
                    lower_bnd, upper_bnd = desc.get_bounds()
                    idist = self.descvoc["dist"](lower_bnd, upper_bnd)
                else:
                    idist = self.descvoc["dist"]()
                idist.set_para(mus[i], covs[i])
                mix_dists.append(idist)

            comp_probs = list()
            for k in range(npts):
                ps = [mix_dists[i].range_cdf(pos_left[k, :], pos_right[k, :]) for i in range(nmix)]
                comp_probs.append(ps)
            probs = np.array(comp_probs)
        else:
            if self.descvoc["name"] == "dtmnorm":
                desc = self.describes.values()[0]
                lower_bnd, upper_bnd = desc.get_bounds()
                dist = self.descvoc["dist"](lower_bnd, upper_bnd)
            else:
                dist = self.descvoc["dist"]()
            dist.set_para(mus[0], covs[0])
            probs = np.array([dist.range_cdf(pos_left[k, :], pos_right[k, :]) for k in range(npts)])

        return probs

    def search(self, min_pts=20, strictness=4, verbose=True):
        search_tree = dict()
        blob_nodes = list()
        hpos2cnt = dict()
        min_pts = np.min([min_pts, int(np.mean(self._count))])
        for k in range(self._total):
            hpos2cnt[tuple(self._hpos[k])] = self._count[k]

        heavynode2level = self.leveltree.get_heavynodes()

        stat_tester = None
        if self.descvoc['name'] == 'dtmnorm':
            stat_tester = StatisticHypothesisTestTruncate(alpha_level=0.01, n_jobs=3)
        elif self.descvoc['name'] == 'dmgauss':
            stat_tester = StatisticHypothesisTest(strictness)
        else:
            raise ValueError("Unimplemented vocabulary, valid vocabularies {}".format(self._valid_vocabularies_))

        Q = deque()
        roots = self.leveltree.get_nodesid_atlevel(self.leveltree.levels[0])
        Q.extend(roots)

        # BFS search
        if self.describes is None:
            self.describes = dict()
        while len(Q) > 0:
            ndid = Q.popleft()
            tree_node = self.leveltree.get_node(ndid)
            hcubes = np.array(tree_node.get_covers())
            node_counts = [hpos2cnt.get(tuple(hcb)) for hcb in hcubes]

            if np.max(node_counts) < min_pts:
                blob_nodes.append(tree_node)

            is_mix = ndid in heavynode2level.keys()

            ## logP
            values = np.array(node_counts)
            values = np.log(values + 1) / np.log(self.power_base)

            self._describe_singular_check_(hcubes)
            comps = self._mixcomps if is_mix else 1
            if ndid not in self.describes:
                desc = self.descvoc["voc"](self.mode, is_mix, comps)
                if self.descvoc["name"] == 'dtmnorm':
                    desc.set_bounds()
                desc.fit(hcubes, values)
                self.describes[ndid] = desc
            else:
                desc = self.describes[ndid]
                desc.data, desc.values = hcubes, values

            hcubes_prob = None
            if is_mix:
                hcubes_prob = self._model_hcubes_prob_(hcubes, hcubes + 1, desc.paras, is_mix)

            weights = np.ones(len(hcubes), int) # values #

            gaussian = False
            if self.descvoc['name'] == 'dmgauss':
                gaussian = stat_tester.apply(hcubes, weights, hcubes_prob, desc.paras, is_mix)
            elif self.descvoc['name'] == 'dtmnorm':
                lower_bnd, _ = desc.get_bounds()
                gaussian = stat_tester.apply(hcubes, weights, hcubes_prob, desc.paras, is_mix, lower_bnd)

            if gaussian:
                search_tree[ndid] = desc
                # if verbose:
                #     print("Info: island: {} hypothesis * Accept. ^~^".format(ndid))
            else:
                if tree_node.child is not None:
                    Q.extend(tree_node.child)
                    # if verbose:
                    #     print("Info island: {} hypothesis & Rejected. >_<".format(ndid))
                else:
                    search_tree[ndid] = desc
                    # if verbose:
                    #     print("Info: island: {} hypothesis $ pseudo-Accept. ^..^".format(ndid))

        self.describes = search_tree

    def _close_check(self, gaus_voc1, gaus_voc2, threshold=None):
        mu1, cov1 = np.array(gaus_voc1.get("mus")[0]), np.array(gaus_voc1.get("covs")[0])
        mu2, cov2 = np.array(gaus_voc2.get("mus")[0]), np.array(gaus_voc2.get("covs")[0])

        diff = mu1 - mu2 # distance between two cluster centers.
        distance = np.sqrt(diff.dot(diff.T))
        if threshold is not None:
            return distance <= threshold
        else:
            # default distance threshold
            cov_dist = np.max([1, np.max(np.sqrt(np.diag(cov1)) + np.sqrt(np.diag(cov2)))])
            return distance < 2 * cov_dist  #3 * np.sum(axes_distance)

    def _greedy_select(self, candidates, content):
        '''
        select the optimal merged cluster with max score (minimum decrease of log-likelihood)
        :param candidates: selection candidates
        :param clusters: all clusters dictionary
        :return:  merged cluster id
        '''
        if len(candidates) <= 0: return None

        cands_score = list()
        for cand in candidates:
            cs = content.get(cand)
            if len(cs.get("cs_id")) > 0:
                score = 0.0
                for cs_id in cs.get("cs_id"):
                    desc = content[cs_id].get("desc")
                    score += desc.paras["loss"]
                score -= cs.get("desc").paras["loss"]
                score /= 1.0 * cs.get("npts")            # average score over points
                # log-likelihood will decrease (need to select the one keep best model.)
                cands_score.append((cand, score))
        sorted_score = sorted(cands_score, cmp=lambda x, y: cmp(x[1], y[1]), reverse=False)
        return sorted_score[0][0]

    def post_stitch(self, strictness=4, verbose=True):
        optimals = list()
        mixturec, singlesc = None, dict()
        hpos2cnt = dict()
        for k in range(self._total):
            hpos2cnt[tuple(self._hpos[k])] = self._count[k]

        ider = -1
        for cid, desc in self.describes.items():
            if cid > ider: ider = cid
            if desc.is_mix: mixturec = cid
            else:
                npts = np.sum([hpos2cnt.get(tuple(pos)) for pos in desc.data])
                singlesc[cid] = {"id": cid, "desc": desc, "cs_id": [cid], "islnds": [cid], "npts": npts}
                optimals.append(cid)

        # stitching islands iteratively
        tested_cands = dict()

        stat_tester = None
        if self.descvoc['name'] == 'dtmnorm':
            stat_tester = StatisticHypothesisTestTruncate(alpha_level=0.05, n_jobs=3)
        elif self.descvoc['name'] == 'dmgauss':
            stat_tester = StatisticHypothesisTest(strictness)
        else:
            raise ValueError("Unimplemented vocabulary, valid vocabularies {}".format(self._valid_vocabularies_))

        while True:
            update = False
            candidates = list()
            n_cands = len(optimals)

            # test each clusters pair.
            for i in range(n_cands):
                ci = singlesc.get(optimals[i])
                for j in range(i + 1, n_cands):
                    cj = singlesc.get(optimals[j])
                    # have tested tuple
                    merge_tuple = (ci.get("id"), cj.get("id"))
                    merge_id = tested_cands.get(merge_tuple, None)
                    if verbose:
                        print merge_tuple
                    if merge_id is not None:
                        if merge_id != -1:
                            candidates.append(merge_id)
                        else:
                            continue
                    else:
                        desci, descj = ci.get("desc"), cj.get("desc")
                        tested_cands[merge_tuple] = -1
                        # merge limits: two islands are closed.
                        # distance = 0.3 * np.min(self.shape)
                        closed = self._close_check(desci.paras, descj.paras)
                        if not closed: continue

                        ## output testing cases
                        if verbose:
                            print("Info: test new merging node: {} ({}, {})".format(
                                merge_tuple, ci.get("cs_id"), cj.get("cs_id")))

                        merged_hcubes = np.vstack([desci.data, descj.data])
                        merged_values = np.hstack([desci.values, descj.values])
                        desc = self.descvoc["voc"](self.mode, False, 1)
                        if self.descvoc["name"] == 'dtmnorm':
                            desc.set_bounds()
                        desc.fit(merged_hcubes, merged_values)

                        weights = np.ones(len(merged_hcubes), int) # merged_values #

                        gaussian = False
                        if self.descvoc['name'] == 'dmgauss':
                            gaussian = stat_tester.apply(merged_hcubes, weights, None, desc.paras, False)
                        elif self.descvoc['name'] == 'dtmnorm':
                            lower_bnd, _ = desc.get_bounds()
                            gaussian = stat_tester.apply(merged_hcubes, weights, None, desc.paras, False, lower_bnd)

                        if gaussian:
                            ider += 1
                            cs_id = [ci.get("id")] + [cj.get("id")]
                            islnds = list(set(ci.get("islnds") + cj.get("islnds")))
                            npts = ci.get("npts") + cj.get("npts")
                            singlesc[ider] = {"id": ider, "desc": desc, "cs_id": cs_id, "islnds": islnds, "npts": npts}
                            candidates.append(ider)
                            tested_cands[merge_tuple] = ider
                            update = True
                            if verbose:
                                print("new node: {}-{}-{}".format(ider, cs_id, islnds))
                        else:
                            if verbose:
                                print("Info:  ------ node {} check failed ------".format(merge_tuple))

            # select the most promising merge-node with greedy select from candidates
            #     (minimum decrease of log-likelihood)
            opt_merge_id = self._greedy_select(candidates, singlesc)
            if opt_merge_id is not None:
                update = True
                opt_c = singlesc.get(opt_merge_id)
                for cid in opt_c.get("cs_id"):
                    if cid in optimals:
                        optimals.remove(cid)
                    if cid in singlesc:
                        for islnd_id in singlesc.get(cid).get("islnds"):
                            if islnd_id in optimals:
                                optimals.remove(islnd_id)
                optimals.append(opt_merge_id)
                if verbose:
                    print("current optimals: {}".format(optimals))
            else:
                update = False

            if not update:
                break    # # no further update and stitch finished

        # if verbose:
        #     print("\nFinal optimal cluster result: {}".format(optimals))

        desc_stitch = dict()
        desc_stitch[mixturec] = self.describes.get(mixturec)
        for cid in optimals:
            c_nd = singlesc.get(cid)
            desc_stitch[cid] = c_nd.get("desc")
        self.describes = desc_stitch

    def _measure_model_mdl_(self, hcubes, counts, hcubes_label, hcubes_prob, outlier_marker=-1):
        counts = np.asarray(counts)
        hcubes_label = np.asarray(hcubes_label)

        descs = self.describes.values()
        N_cls = len(descs)

        mdl = MDLBase()
        L_clusters = mdl.integer_mdl(N_cls)
        L_paras, L_assign, L_nis, L_error = 0.0, 0.0, 0.0, 0.0
        for k in range(N_cls):
            lb = k
            index = np.where(hcubes_label == lb)
            k_hcubes, k_cnts = hcubes[index], counts[index]
            k_probs = hcubes_prob[index]

            # model error code-length
            knis = np.sum(descs[k].values)
            exp_logPvs = k_probs * knis
            exp_counts = np.array(np.power(self.power_base, exp_logPvs), int) #exp_logPvs  #
            L_error += mdl.seq_diff_mdl(exp_counts, k_cnts)

            # model parameters code-length
            comp_parm = descs[k].compact_parm()
            L_paras += np.sum([mdl.float_mdl(p) for p in comp_parm])
            L_assign += 1.0  # for encoding the indicator for mixture or single
            L_nis += mdl.float_mdl(knis)  # for encoding #pts of this cluster

        L_outs = 0.0
        outs_index = np.where(hcubes_label == outlier_marker)
        outs_hcubs, outs_vals = hcubes[outs_index], counts[outs_index]
        L_outs += mdl.seq_diff_mdl(np.zeros_like(outs_vals, int), outs_vals)
        L_shape = np.sum([mdl.integer_mdl(dm) for dm in self.shape])

        C = L_clusters + L_assign + L_shape + L_paras + L_nis + L_outs + L_error
        return C


    def _smooth_entropy_(self, ps, qs, weights, smoother=1e-15):
        n = len(ps)
        ps = np.asarray(ps) + smoother
        qs = np.asarray(qs) + smoother
        w_en = [ weights[i] * ps[i] * np.log2(ps[i] * 1.0 / qs[i])
                 if qs[i] != 0 else 0 for i in range(n) ]
        return np.sum(w_en)

    def _measure_suspicious_(self, hcubes, values):
        majority_normal = None
        cls_id2prob = dict()
        for id, desc in self.describes.items():
            hcb_prob = self._model_hcubes_prob_(hcubes, hcubes + 1, desc.paras, desc.is_mix)
            if desc.is_mix:
                majority_normal = id
                hcb_prob = np.sum(hcb_prob * desc.paras['weights'], axis=1)
            cls_id2prob[id] = hcb_prob

        dists = dict()
        for id, probs in cls_id2prob.items():
            dists[id] = self._smooth_entropy_(probs, cls_id2prob[majority_normal], values)

        return sorted(dists.items(), key=operator.itemgetter(1), reverse=True)

    def _hcubes_labeling_(self, hcubes, outliers_marker=-1, strictness=4):
        criticals = [1e-1, 5e-2, 1e-2, 1e-3, 1e-4, 1e-5]
        hcubes = np.asarray(hcubes)
        hcube_clsprob = list()
        hcube_labels = list()

        descs = self.describes.values()
        mixId = -1
        for k in range(len(descs)):
            desc = descs[k]
            clsprob = self._model_hcubes_prob_(hcubes, hcubes + 1, desc.paras, desc.is_mix)
            if desc.is_mix:
                mixId = k
                clsprob = np.sum(clsprob * desc.paras['weights'], axis = 1)
            hcube_clsprob.append(clsprob)
        hcube_clsprob = np.array(hcube_clsprob).T

        for k in range(self._total):
            prob, cls = np.max(hcube_clsprob[k, :]), np.argmax(hcube_clsprob[k, :])
            if prob <= criticals[strictness]:
                label = outliers_marker
            else:
                label = cls

            hcube_labels.append(label)

        probs = np.max(hcube_clsprob, axis=1)
        return hcube_labels, probs

    def cluster_histogram(self, outfn=None, strictness=4, verbose=True):
        outliers_marker = -1
        hcubes_label, hcubes_prob = self._hcubes_labeling_(self._hpos, outliers_marker, strictness)
        if outfn is not None:
            np.savetxt(outfn, np.vstack((self._hpos.T, hcubes_label)).T, '%i', delimiter=',',
                       header='#hcube: {}, #label: {}'.format(self._total, len(np.unique(hcubes_label))))

        mdls = self._measure_model_mdl_(self._hpos, self._count, hcubes_label, hcubes_prob, outliers_marker)
        # suspicious = self._measure_suspicious_(self._hpos, np.log2(1 + self._count))

        if verbose:
            print("Graph Model Description Length: {}".format(mdls))
            # print("Island suspicious orders: {}".format(suspicious))

        return self._hpos, hcubes_label, mdls


    def graph_node_cluster(self, node2hpos_infn, nodelabel_outfn,
                           hcubelabel_outfn=None, strictness=4, verbose=True):
        loader = Loader()

        ptsidx_pos = loader.load_pt2pos(node2hpos_infn)
        outliers_marker = -1
        hcbs_labels, _ = self._hcubes_labeling_(self._hpos, outliers_marker, strictness)
        hcube2label = dict(zip(map(tuple, self._hpos), hcbs_labels))

        if hcubelabel_outfn is not None:
            np.savetxt(hcubelabel_outfn, np.vstack((self._hpos.T, hcbs_labels)).T, '%i', delimiter=',',
                       header='#hcube: {}, #label: {}'.format(self._total, len(np.unique(hcbs_labels))))

        with open(nodelabel_outfn, 'w') as ofp:
            ofp.writelines("# #pt: {}, #label: {}\n".format(len(ptsidx_pos), len(np.unique(hcbs_labels))))
            for k in range(len(ptsidx_pos)):
                ptidx, pos = ptsidx_pos[k, 0], tuple(ptsidx_pos[k, 1:])
                ptlab = hcube2label.get(pos, -1)
                ofp.writelines("{},{}\n".format(ptidx, ptlab))
            ofp.close()

        if verbose:
            print("clustering done!")


    def cluster_weighted_suspicious(self, hcube_weights, strictness=4, verbose=True):
        outliers_marker = -1
        hcubes_label, hcubes_prob = self._hcubes_labeling_(self._hpos, outliers_marker, strictness)
        _susp_ = self._measure_suspicious_(self._hpos, np.log2(1 + self._count))

        cls2susp = dict(_susp_)
        cls2wtsusp = dict()

        hcubes_label = np.asarray(hcubes_label)
        clsid2labels = dict(zip(self.describes.keys(), range(len(self.describes.values()))))
        for clsid, desc in self.describes.items():
            lb = clsid2labels[clsid]
            lb_hcubes_index = np.arange(len(hcubes_label))[hcubes_label == lb]
            cls_wt, npts = 0, 0
            for hidx in lb_hcubes_index:
                cls_wt += hcube_weights[hidx] * self._count[hidx]
                npts += self._count[hidx]
            cls_wt /= 1.0 * npts
            cls2wtsusp[clsid] = cls2susp[clsid] * np.log(cls_wt)

        sorted_wtsusp = sorted(cls2wtsusp.items(), key=operator.itemgetter(1), reverse=True)

        if verbose:
            print("KL-divergence based suspicious: {}".format(_susp_))
            print("Weighted suspicious orders: {}".format(sorted_wtsusp))

        return sorted_wtsusp

