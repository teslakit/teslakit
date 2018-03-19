#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as op

from lib.objs.project import Project


# data storage
p_data = '/Users/ripollcab/Projects/TESLA-kit/teslakit/data/'
p_predictor = op.join(p_data, 'TKPRED_SST.nc')

# study site
p_sites = '/Users/ripollcab/Projects/TESLA-kit/teslakit/sites/'
n_site = 'KWAJALEI'
p_case = op.join(p_sites, n_site)



# Tesla-Kit project
tkp = Project(p_case)

# Generate 1000y Annual Weather Types (AWT)
tkp.Generate_AWT(wp)



# -------------------------------------------------------------------




