#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op

# geographic lib
from geographiclib.geodesic import Geodesic


lat1, lon1 = 5.7, 158.3
lat2, lon2 = 9.5, 167.5

out = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)

# print todo
for k in out.keys():
    print('{0} --> {1}'.format(k, out[k]))
print('')


# azimuth and distance
print(out['a12'])   # distancia entre puntos en degrees
print(out['azi1'])  # azimuth de la linea en el punto 1
print(out['azi2'])  # azimuth de la linea en el punto 2

