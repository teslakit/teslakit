#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.basemap import Basemap
from datetime import datetime, timedelta

def WorldMap(bk='simple'):
    'Returns customized world map (mercator)'

    # basemap: mercator
    m = Basemap(
        projection='merc',
        llcrnrlat=-70,urcrnrlat=70,
        llcrnrlon=-180,urcrnrlon=180,
        lat_ts=20,
        resolution='c')

    # draw parallels.
    parallels = np.arange(-90.,90.,10.)
    m.drawparallels(parallels, labels=[1,0,0,0],fontsize=6)
    # draw meridians
    meridians = np.arange(0.,360.,10.)
    m.drawmeridians(meridians, labels=[0,0,0,1],fontsize=6)

    # draw background
    if bk == 'simple':
        m.drawcoastlines()
        m.fillcontinents(color='coral',lake_color='aqua')
        m.drawmapboundary(fill_color='aqua')
    elif bk == 'shaderelief':
        m.shadedrelief()
    elif bk == 'etopo':
        m.etopo()
    elif bk == 'bluemarble':
        m.bluemarble()

    # return basemap
    return m

def WorldMap_Stations(xds_stations, bk='simple', p_export=None):
    '''
    Plot mercator world map with CSIRO spec stations
    bk (background)= 'simple', 'shaderelief', 'etopo', 'bluemarble'
    '''

    # xds_station lon lat
    lon = xds_stations.longitude.values[:]
    lat = xds_stations.latitude.values[:]
    nms = xds_stations.station_name[:]

    # plot figure
    fig = plt.figure(figsize=(16,9))

    # Get customized basemap
    m = WorldMap(bk)

    # add stations
    m.scatter(lon, lat, s=6, c='r', latlon=True)

    # more info
    plt.title('CSIRO spec stations')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=96)
        plt.close()

def WorldGlobe_Stations(xds_stations, bk='simple', lon0=0, lat0=0, p_export=None):
    '''
    Plot orthogonal world globe with CSIRO spec stations
    bk (background): 'simple', 'shaderelief', 'etopo', 'bluemarble'
    lonlat0: center focus, tuple (lon0, lat0)
    '''

    # xds_station lon lat
    lon = xds_stations.longitude.values[:]
    lat = xds_stations.latitude.values[:]
    nms = xds_stations.station_name[:]

    # plot figure
    fig = plt.figure(figsize=(12,12))

    # basemap: ortho
    m = Basemap(
        projection='ortho',
        lon_0=lon0,
        lat_0=lat0,
        resolution='l',
    )

    # draw parallels.
    parallels = np.arange(-90.,90.,10.)
    m.drawparallels(parallels, labels=[1,0,0,0],fontsize=6)
    # draw meridians
    meridians = np.arange(0.,360.,10.)
    m.drawmeridians(meridians, labels=[0,0,0,1],fontsize=6)

    # draw background
    if bk == 'simple':
        m.drawcoastlines()
        m.fillcontinents(color='coral',lake_color='aqua')
        m.drawmapboundary(fill_color='aqua')
    elif bk == 'shaderelief':
        m.shadedrelief()
    elif bk == 'etopo':
        m.etopo()
    elif bk == 'bluemarble':
        m.bluemarble()

    # add stations
    m.scatter(lon, lat, s=6, c='r', latlon=True)

    # more info
    plt.title('CSIRO spec stations')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=96)
        plt.close()

def WorldGlobeZoom_Stations(xds_stations, bk='simple', lon0=0, lat0=0, p_export=None):
    '''
    Plot orthogonal world globe zoom with CSIRO spec stations
    bk (background): 'simple', 'shaderelief', 'etopo', 'bluemarble'
    lonlat0: center focus, tuple (lon0, lat0)
    '''
    # TODO: no sale bien

    delta_x = 5000000
    delta_y = 5000000

    # xds_station lon lat
    lon = xds_stations.longitude.values[:]
    lat = xds_stations.latitude.values[:]
    nms = xds_stations.station_name[:]

    # plot figure
    fig = plt.figure(figsize=(12,12))

    # basemap: ortho
    m1 = Basemap(
            projection='ortho',
            lon_0=lon0,
            lat_0=lat0,
            resolution=None
        )
    # zoom
    #ax = fig.add_axes([0.1,0.1,0.8,0.8],facecolor='k')
    xc,yc = m1(lon0, lat0)
    print xc, yc
    m = Basemap(
        projection='ortho',
        lon_0=lon0,
        lat_0=lat0,
        resolution='l',
        llcrnrx=xc-delta_x, llcrnry=yc-delta_y,
        urcrnrx=yc+delta_x, urcrnry=yc+delta_y,
    )
    print m

    # draw parallels.
    parallels = np.arange(-90.,90.,10.)
    # m1.drawparallels(parallels, labels=[1,0,0,0],fontsize=6)
    # draw meridians
    meridians = np.arange(0.,360.,10.)
    #m1.drawmeridians(meridians, labels=[0,0,0,1],fontsize=6)

    # draw background
    if bk == 'simple':
        m.drawcoastlines()
        m.fillcontinents(color='coral',lake_color='aqua')
        m.drawmapboundary(fill_color='aqua')
    elif bk == 'shaderelief':
        m.shadedrelief()
    elif bk == 'etopo':
        m.etopo()
    elif bk == 'bluemarble':
        m.bluemarble()

    # add stations
    m.scatter(lon, lat, s=6, c='r', latlon=True)

    # more info
    plt.title('CSIRO spec stations')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=96)
        plt.close()


def WorldMap_GriddedCoords(xds_gridded, bk='simple', p_export=None):
    '''
    Plot mercator world map with CSIRO gridded coordinates
    bk (background)= 'simple', 'shaderelief', 'etopo', 'bluemarble'
    '''

    # xds_station lon lat
    lon = xds_gridded.longitude.values[:]
    lat = xds_gridded.latitude.values[:]

    # fix lon for plotting
    lon[np.where(lon>180.0)] = lon[np.where(lon>180.0)]-360

    # plot figure
    fig = plt.figure(figsize=(16,9))

    # Get customized basemap
    m = WorldMap(bk)

    # add stations
    xx,yy = np.meshgrid(lon, lat)
    m.scatter(xx, yy, s=6, c='r', latlon=True)

    # more info
    plt.title('CSIRO gridded coords')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=96)
        plt.close()

def WorldMap_GriddedCoords(xds_gridded, bk='simple', p_export=None):
    '''
    Plot mercator world map with all CSIRO gridded coordinates
    bk (background)= 'simple', 'shaderelief', 'etopo', 'bluemarble'
    '''

    # xds_station lon lat
    lon = xds_gridded.longitude.values[:]
    lat = xds_gridded.latitude.values[:]

    lon[np.where(lon>180.0)] = lon[np.where(lon>180.0)]-360  # fix lon for plotting

    # plot figure
    fig = plt.figure(figsize=(16,9))

    # Get customized basemap
    m = WorldMap(bk)

    # add stations
    xx,yy = np.meshgrid(lon, lat)
    m.scatter(xx, yy, s=6, c='r', latlon=True)

    # more info
    plt.title('CSIRO gridded coords')

    # show / export
    if not p_export:
        plt.show()

    else:
        fig.savefig(p_export, dpi=96)
        plt.close()
