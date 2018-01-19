#!/usr/bin/env python
# -*- coding: utf-8 -*-

if __name__ == "__main__":

    # import libraries
    #from lib.alr import alr_AWT
    import scipy.io as sio

    from lib.io.matlab import ReadMatfile
    from lib.custom_dateutils import datevec2datetime
    from lib.alr import AutoRegLogisticReg

    ## ----------------------------------
    ## AUTOREGRESSIVE LOGISTIC REGRESSION

    # path to AWT data (using matlab)
    p_AWT = '/Users/ripollcab/Projects/TESLA-kit/teslakit/data/TESLA_AWT.mat'

    # need bmus and dates
    d_AWT = ReadMatfile(p_AWT)
    bmus = d_AWT['AWT']['bmus']
    dvec = d_AWT['AWT']['Dates']
    #dtime = datevec2datetime(dvec)  # use python dtime instead of date vector

    # solve alr
    num_wts = 6  # or len(set(bmus))
    num_sims = 100
    sim_start = 1700
    sim_end = 3201
    
    evbmusd_sim = AutoRegLogisticReg(bmus, num_wts, num_sims, sim_start, sim_end)




