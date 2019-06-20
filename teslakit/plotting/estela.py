
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# fig aspect and size
_faspect = (1+5**0.5)/2.0
_fsize = 7

def Plot_ESTELA_Globe(pnt_lon, pnt_lat, estela_D, p_export=None):
    'Plots astronomical tide temporal series'

    estela_max = np.ceil(estela_D.max().values)

    # plot axes
    axes = plt.axes(
        projection=ccrs.Orthographic(pnt_lon+40, pnt_lat)
    )

    qmesh = estela_D.sortby(estela_D.longitude).plot(
        ax=axes, transform=ccrs.PlateCarree()
    )
    qmesh.set_clim(0, estela_max)
    axes.plot(pnt_lon, pnt_lat, 'oy', transform=ccrs.PlateCarree())

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=128)
        plt.close()

