#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
from scipy.spatial import distance


def Extract_Circle(xds_TCs, p_lon, p_lat, r):
    '''
    Extracts TCs inside circle

    xds_TCs: tropical cyclones track database
        lon,lat,pressure variables
        storm dimension

    circle defined by:
        p_lon, p_lat  -  circle center
        r             -  circle radius (degree)

    returns:
        xds_area: selection of xds_TCs inside circle
        xds_inside: contains TCs custom data inside circle
    '''

    # TODO REFACTOR
    lonlat_p = np.array([[p_lon, p_lat]])

    lon = xds_TCs.lon.values[:]
    lat = xds_TCs.lat.values[:]
    press = xds_TCs.pressure.values[:]

    store_date = 'dates' in xds_TCs.variables
    if store_date:
        time = xds_TCs.dates.values[:]

    # get storms inside circle area
    n_storms = xds_TCs.storm.shape[0]
    l_storms_area = []
    l_pos_in = []  # inside circle position
    l_press_in = []  # inside circle pressure
    l_min_press_in = []  # inside circle min pressure
    l_categ_in = []  # inside circle storm category
    l_date_in = []  # inside circle date (day)
    l_date_last = []  # last cyclone date 

    for i_storm in range(n_storms):
        lonlat_s = np.column_stack(
            (lon[i_storm], lat[i_storm])
        )
        press_s = press[i_storm]

        # TODO: cambiar de distancia euclidea a arclen great circle
        dist = distance.cdist(lonlat_s, lonlat_p)
        pos_in = np.where(dist<r)[0][:]
        if pos_in.any():
            l_storms_area.append(i_storm)
            l_pos_in.append(pos_in)

            # pressure, min pressure and category inside
            press_s_in = press_s[pos_in]
            press_s_min = np.min(press_s_in)

            l_press_in.append(press_s_in)
            l_min_press_in.append(np.array(press_s_min))
            l_categ_in.append(np.array(GetStormCategory(press_s_min)))

            if store_date:
                time_s_in = time[i_storm][pos_in]
                dist_in = dist[pos_in]
                p_dm = np.where((dist_in==np.min(dist_in)))[0]
                l_date_in.append(np.datetime64(time_s_in[p_dm][0],'D'))

                # store last cyclone date too
                all_dates = filter(lambda a: a!= np.datetime64('NaT'),
                                   time[i_storm][:])
                l_date_last.append(all_dates[-1])

    # cut storm dataset to selection
    xds_area = xds_TCs.isel(storm=l_storms_area)

    # add data from inside the circle to a dataset
    # TODO: CORREGIR DATASET, NO PUEDO USAR DIMENSIONES PARA SELECIONAR
    xds_inside = xr.Dataset(
        {
            'inside_pos':(('storm'), np.array(l_pos_in)),
            'inside_pressure':(('storm'), np.array(l_press_in)),
            'inside_pressure_min':(('storm'), np.array(l_min_press_in)),
            'inside_category':(('storm'), np.array(l_categ_in)),
        },
        coords = {
            'storm':(('storm'), xds_area.storm.values[:])
        },
        attrs = {
            'point_lon' : p_lon,
            'point_lat' : p_lat,
            'point_r' : r,
        }
    )
    if store_date:
        xds_inside['dmin_date'] = (('storm',), np.array(l_date_in))
        xds_inside['last_date'] = (('storm',), np.array(l_date_last))

    return xds_area, xds_inside

def Extract_Square(xds_wmo):
    '''
    Extracts storms inside square

    xds_wmo: all storms database downlaoded from
    ftp://eclipse.ncdc.noaa.gov/pub/ibtracs/v03r10/wmo/netcdf/Allstorms.ibtracs_wmo.v03r10.nc.gz

    square defined by:
    '''
    # TODO

    return None

def Extract_Polygon(xds_wmo):
    '''
    Extracts storms inside polygon

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

