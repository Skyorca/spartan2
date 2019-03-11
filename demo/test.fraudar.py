#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

import spartan as st


def test():
    # set the computing engine
    st.config(st.engine.SINGLEMACHINE)

    # load graph data
    data = st.SFrame("example")

    # create a anomaly detection model
    admodel = st.anomaly_detection.create(data, "anomaly detection")

    # run the model
    admodel.run(st.ad_policy.FRAUDAR)

    # show the results
    admodel.showResults()


if __name__ == "__main__":
    test()
