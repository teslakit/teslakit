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
    l_pos_in = []

    for i_storm in range(n_storms):
        lonlat_s = np.column_stack(
            (lon_hurr[i_storm], lat_hurr[i_storm])
        )

        # TODO: cambiar de distancia euclidea a arclen great circle
        dist = distance.cdist(lonlat_s, lonlat_p)
        pos_in = np.where(dist<r)[0][:]
        if pos_in.any():
            l_storms_area.append(i_storm)
            l_pos_in.append(np.array(pos_in))

    # cut storm dataset to selection
    xds_area = xds_hurr.isel(storm=l_storms_area)
    xds_area['pos_inside'] =(('storm',), np.array(l_pos_in))
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

def GetStormCategory(pres_min):
    '''
    Returns storm category (int 5-0)
    '''

    pres_lims = [920, 944, 964, 979, 1000]

    if pres_min < pres_lims[0]:
        return 5
    elif pres_min < pres_lims[1]:
        return 4
    elif pres_min < pres_lims[2]:
        return 3
    elif pres_min < pres_lims[3]:
        return 2
    elif pres_min < pres_lims[4]:
        return 1
    else:
        return 0

def SortCategoryCount(np_categ, nocat=9):
    '''
    Sort category change - count matrix
    np_categ = [[category1, category2, count], ...]
    '''


    categs = [0,1,2,3,4,5,9]

    np_categ = np_categ.astype(int)
    np_sort = np.empty((len(categs)*(len(categs)-1),3))
    rc=0
    for c1 in categs[:-1]:
        for c2 in categs:
            p_row = np.where((np_categ[:,0]==c1) & (np_categ[:,1]==c2))
            if p_row[0].size:
                np_sort[rc,:]=[c1,c2,np_categ[p_row,2]]
            else:
                np_sort[rc,:]=[c1,c2,0]

            rc+=1

    return np_sort.astype(int)

