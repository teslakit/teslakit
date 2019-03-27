#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
from datetime import datetime

def GetMJOCategories(rmm1, rmm2, phase):
    '''
    Divides MJO data in 25 categories.
    returns array with categories time series
    and corresponding rmm
    '''

    rmm = np.sqrt(rmm1**2+rmm2**2)
    categ = np.empty(rmm.shape)*np.nan

    for i in range(1,9):
        s = np.squeeze(np.where(phase == i))
        rmm_p = rmm[s]

        # TODO: VA LENTO. OPTIMIZAR
        # usar busqueda y remplazar por indices .where
        for j in s:
            if rmm[j] > 2.5:
                categ[j] = i
            elif rmm[j] > 1.5:
                categ[j] = i+8
            elif rmm[j] > 1:
                categ[j] = i+8*2
            elif rmm[j] <= 1:
                categ[j] = 25

    # get rmm_categ
    rmm_categ = {}
    for i in range(1,26):
        s = np.squeeze(np.where(categ == i))
        rmm_categ['cat_{0}'.format(i)] = np.column_stack((rmm1[s],rmm2[s]))

    return categ.astype(int), rmm_categ

