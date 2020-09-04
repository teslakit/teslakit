# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:47:05 2020

@author: lcag075
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import gridspec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


os.chdir (r'C:\Users\lcag075\Dropbox\MAJURO-teslakit')
data_path=r'C:\Users\lcag075\Dropbox\MAJURO-teslakit\teslakit\DATA\sites\MAJURO'

data=xr.open_dataset(os.path.join(data_path,'Seiche','hm0_daily_60secfrequency.nc'))
data['time']=data.time.dt.floor('d')

kma = xr.open_dataset(os.path.join(r'C:\Users\lcag075\Dropbox\Culebras-uoa\MAJURO\DATA\DWTs', "kma.nc"))
kma = xr.Dataset(
        {   'bmus':(('time'),kma.sorted_bmus.values),
         },coords = {'time': kma.time.values})
kma=kma.sel(time = slice(data.time[0],data.time[-1]))

data['bmus']=kma.bmus

#%%

num_clusters=36
l_colors_dwt = np.array([ (1.0000, 0.1344, 0.0021), (1.0000, 0.2669, 0.0022), (1.0000, 0.5317, 0.0024), (1.0000, 0.6641, 0.0025), (1.0000, 0.9287, 0.0028),
    (0.9430, 1.0000, 0.0029),(0.6785, 1.0000, 0.0031), (0.5463, 1.0000, 0.0032),(0.2821, 1.0000, 0.0035),(0.1500, 1.0000, 0.0036),(0.0038, 1.0000, 0.1217),
    (0.0039, 1.0000, 0.2539),(0.0039, 1.0000, 0.4901),(0.0039, 1.0000, 0.6082),(0.0039, 1.0000, 0.8444), (0.0039, 1.0000, 0.9625), (0.0039, 0.8052, 1.0000),
    (0.0039, 0.6872, 1.0000),(0.0040, 0.4510, 1.0000),(0.0040, 0.3329, 1.0000),(0.0040, 0.0967, 1.0000),(0.1474, 0.0040, 1.0000),(0.2655, 0.0040, 1.0000),
    (0.5017, 0.0040, 1.0000),(0.6198, 0.0040, 1.0000),(0.7965, 0.0040, 1.0000),(0.8848, 0.0040, 1.0000),(1.0000, 0.0040, 0.9424),(1.0000, 0.0040, 0.8541),
    (1.0000, 0.0040, 0.6774),(1.0000, 0.0040, 0.5890),(1.0000, 0.0040, 0.4124),(1.0000, 0.0040, 0.3240),(1.0000, 0.0040, 0.1473),(0.9190, 0.1564, 0.2476),
    (0.7529, 0.3782, 0.4051),(0.6699, 0.4477, 0.4584),(0.5200, 0.5200, 0.5200),(0.4595, 0.4595, 0.4595),(0.4100, 0.4100, 0.4100),(0.3706, 0.3706, 0.3706),
    (0.2000, 0.2000, 0.2000),(     0, 0, 0)])

# get first N colors 
np_colors_base = np.array(l_colors_dwt)
np_colors_rgb = np_colors_base[:num_clusters]

newcmp = ListedColormap(np_colors_rgb)

fig = plt.figure(figsize=[18.5,9])
gs1=gridspec.GridSpec(4,1)
ax1=fig.add_subplot(gs1[0])
ax2=fig.add_subplot(gs1[1],sharex=ax1)
ax3=fig.add_subplot(gs1[2],sharex=ax1)
ax4=fig.add_subplot(gs1[3],sharex=ax1)


ax1.plot(data.time,data.hm0_41320,'k:',linewidth=0.8)
ax1.scatter(data.time,data.hm0_41320,15,data.bmus+1,cmap=newcmp)
ax1.set_ylabel('Hm0 (m)',fontsize=12)
ax1.text(.5,.9,'41320', horizontalalignment='center', transform=ax1.transAxes,fontsize=13,fontweight='bold')

ax4.plot(data.time,data.hm0_41323,'k:',linewidth=0.8)
cs=ax4.scatter(data.time,data.hm0_41323,15,data.bmus+1,cmap=newcmp)
ax4.set_ylabel('Hm0 (m)',fontsize=13)
ax4.text(.5,.9,'41323', horizontalalignment='center', transform=ax4.transAxes,fontsize=13,fontweight='bold')

ax2.plot(data.time,data.hm0_41321,'k:',linewidth=0.8)
ax2.scatter(data.time,data.hm0_41321,15,data.bmus+1,cmap=newcmp)
ax2.set_ylabel('Hm0 (m)',fontsize=13)
ax2.text(.5,.9,'41321', horizontalalignment='center', transform=ax2.transAxes,fontsize=13,fontweight='bold')

ax3.plot(data.time,data.hm0_41322,'k:',linewidth=0.8)
ax3.scatter(data.time,data.hm0_41322,15,data.bmus+1,cmap=newcmp)
ax3.set_ylabel('Hm0 (m)',fontsize=13)
ax3.text(.5,.9,'41322', horizontalalignment='center', transform=ax3.transAxes,fontsize=13,fontweight='bold')



ax1.set_xlim(data.time[0],data.time[-1])

gs1.tight_layout(fig, rect=[0.05, [], 0.93, []])

gs2=gridspec.GridSpec(1,1)
ax1=fig.add_subplot(gs2[0])
plt.colorbar(cs,cax=ax1)
ax1.set_ylabel('DWT')
gs2.tight_layout(fig, rect=[0.94, 0.05, 0.995, 0.95])
