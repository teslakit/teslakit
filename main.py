#!/usr/bin/env python
# -*- coding: utf-8 -*-

if __name__ == "__main__":

    # import libraries
    from lib.alr import alr_AWT
    import scipy.io as sio

    # path to AWT data (using matlab)
    p_AWT ='/Users/ripollcab/Projects/TESLA-kit/teslakit/TESLA_AWT.mat'

    # need bmus and dates
    #test = loadmat(p_AWT,variable_names=['Dates','bmus'])
    test = sio.loadmat(p_AWT, squeeze_me=True, struct_as_record=False)

    print test['AWT'].bmus
    print test['AWT'].Dates
    #print test['AWT']['bmus']

