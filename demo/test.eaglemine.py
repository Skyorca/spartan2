#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

import spartan as st


def test():
    # set the computing engine
    st.config(st.engine.SINGLEMACHINE)

    # load graph data
    data = st.loadTensor(name = "example_graph", path = "inputData/", col_ids = ["uid", "oid", "rating"],  col_types = [int, int, int])

    # count degree
    Du, Dv = st.bidegree(data)
    # D = st.degree(data)

    # create a anomaly detection model
    emmodel = st.anomaly_detection.create(data, st.ad_policy.EAGLEMINE, "my_eaglemine_model")
    emmodel.setbipartite(True)
    
    # run the eaglemine model
    emmodel.run(emmodel.U, Du)
    emmodel.run(emmodel.V, Dv)

    A, B = emmodel.nodes(n=0)

    A = [0, 1, 2, 3]
    B = [19, 32, 201]
    g = st.subgraph(data, A, B)
    # g = st.subgraph(data, A)


if __name__ == "__main__":
    test()
