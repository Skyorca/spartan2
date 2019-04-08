#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

import spartan as st


def test():
    # set the computing engine
    st.config(st.engine.SINGLEMACHINE)

    # load graph data
    data = st.loadTensor(name = "example", path = "inputData/", col_ids = ["uid", "oid", "rating"],  col_types = [int, int, int])

    # create a eigen decomposition model
    edmodel = st.eigen_decompose.create(data, "eigen decomposition")

    # run the model
    edmodel.run(st.ed_policy.SVDS, k=10)

    # show the result
    edmodel.showResults()


if __name__ == "__main__":
    test()
