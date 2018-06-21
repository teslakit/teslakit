#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# basemap: ortho
m = Basemap(
    projection='ortho',
    lon_0=-35,lat_0=0,
    resolution='l',
)
plt.show()






