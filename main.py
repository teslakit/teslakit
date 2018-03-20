#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as op

from lib.objs.project import Project


# -----------------------------------------------

# study site
p_sites = '/Users/ripollcab/Projects/TESLA-kit/teslakit/sites/'
n_site = 'KWAJALEI'
p_case = op.join(p_sites, n_site)

# Tesla-Kit project
tkp = Project(p_case)


# -----------------------------------------------

# Generate predictor
# TODO
#tkp.Generate_Predictor()

# Generate 1000y Annual Weather Types (AWT)
#tkp.Generate_AWT_simulation()

# TODO: incorporar plots de esta zona

# -----------------------------------------------

# Download MJO
#tkp.Download_MJO()

# Generate 1000y Daily Weather Types (MJO)
tkp.Generate_MJO_simulation()

# Plot historical MJO data
#tkp.Plot_MJO_historical()




