#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
from scipy.spatial import distance


def Extract_Circle(xds_hurr, p_lon, p_lat, r):
    '''
    Extracts hurricanes inside circle

    xds_hurr: storms database with tracks lon,lat variables and storm dimension

    circle defined by:
        p_lon, p_lat  -  circle center
        r             -  circle radius (degree)
    '''

    lonlat_p = np.array([[p_lon, p_lat]])

    lon_hurr = xds_hurr.lon.values[:]
    lat_hurr = xds_hurr.lat.values[:]

    # get storms inside circle area
    n_storms = xds_hurr.storm.shape[0]
    l_storms_area = []

    for i_storm in range(n_storms):
        lonlat_s = np.column_stack(
            (lon_hurr[i_storm], lat_hurr[i_storm])
        )

        # TODO: cambiar de distancia euclidea a arclen great circle
        dist = distance.cdist(lonlat_s, lonlat_p)
        pos_in = np.where(dist<r)[0][:]
        if pos_in.any():
            l_storms_area.append(i_storm)

    # cut storm dataset to selection
    xds_area = xds_hurr.isel(storm=l_storms_area)
    return xds_area

def Extract_Square(xds_wmo):
    '''
    Extracts hurricanes inside square

    xds_wmo: all storms database downlaoded from
    ftp://eclipse.ncdc.noaa.gov/pub/ibtracs/v03r10/wmo/netcdf/Allstorms.ibtracs_wmo.v03r10.nc.gz

    square defined by:
    '''
    # TODO

    return None

def Extract_Polygon(xds_wmo):
    '''
    Extracts hurricanes inside polygon

    xds_wmo: all storms database downlaoded from
    ftp://eclipse.ncdc.noaa.gov/pub/ibtracs/v03r10/wmo/netcdf/Allstorms.ibtracs_wmo.v03r10.nc.gz

    polygon defined by:
    '''
    # TODO

    return None
