#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

import spartan as st


def test():
    # set the computing engine
    st.config(st.engine.SINGLEMACHINE)

    # load graph data
    data = st.loadTensor(name = "example", path = "inputData/", col_ids = ["uid", "oid", "rating"], col_types = [int, int, int])

    # create a anomaly detection model
    admodel = st.anomaly_detection.create(data, st.ad_policy.FRAUDAR, "my_holoscope_model")

    # run the model
    admodel.run()

    # show the results
    admodel.showResults()


if __name__ == "__main__":
    test()
