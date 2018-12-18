#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

import spartan as st

def test():
    # set the computing engine
    st.config(st.engine.POWERGRAPH)

    # load graph data
    st.addGraph("mockInput.spt", "facebook")
    st.showList()

    # create a model
    model = st.createModel("facebook")

    # preProcess the graph
    model.preProcess()

    # run the model
    model.run(st.run_policy.PAGERANK)

    # show the result
    model.showResult()

    # export result to local file
    model.export("result.spt")

if __name__ == "__main__":
    test()
