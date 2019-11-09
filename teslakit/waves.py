#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr

# tk
from teslakit.custom_dateutils import get_years_months_days, npdt64todatetime
from datetime import datetime, timedelta

# hide numpy warnings
np.warnings.filterwarnings('ignore')


def GetDistribution(xds_wps, swell_sectors):
    '''
    Separates wave partitions (0-5) into families.
    Default: sea, swl1, swl2

    xds_wps (waves partitionss):
        xarray.Dataset (time,), phs, pspr, pwfrac... {0-5 partitions}

    sectors: list of degrees to cut wave energy [(a1, a2), (a2, a3), (a3, a1)]

    returns
        xarray.Dataset (time,), fam_V, {fam: sea,swell_1,swell2. V: Hs,Tp,Dir}
    '''

    # fix data
    hs_fix_data = 50
    for i in range(6):
        phs = xds_wps['phs{0}'.format(i)].values
        p_fix = np.where(phs >= hs_fix_data)[0]

        # fix data
        xds_wps['phs{0}'.format(i)][p_fix] = np.nan
        xds_wps['ptp{0}'.format(i)][p_fix] = np.nan
        xds_wps['pdir{0}'.format(i)][p_fix] = np.nan

    # sea (partition 0)
    sea_Hs = xds_wps['phs0'].values
    sea_Tp = xds_wps['ptp0'].values
    sea_Dir = xds_wps['pdir0'].values
    time = xds_wps['time'].values

    # concatenate energy groups 
    cat_hs = np.column_stack(
        (xds_wps.phs1.values,
        xds_wps.phs2.values,
        xds_wps.phs3.values,
        xds_wps.phs4.values,
        xds_wps.phs5.values )
    )
    cat_tp = np.column_stack(
        (xds_wps.ptp1.values,
        xds_wps.ptp2.values,
        xds_wps.ptp3.values,
        xds_wps.ptp4.values,
        xds_wps.ptp5.values )
    )
    cat_dir = np.column_stack(
        (xds_wps.pdir1.values,
        xds_wps.pdir2.values,
        xds_wps.pdir3.values,
        xds_wps.pdir4.values,
        xds_wps.pdir5.values )
    )

    # prepare output array
    xds_parts = xr.Dataset({
        'sea_Hs':('time',sea_Hs),
        'sea_Tp':('time',sea_Tp),
        'sea_Dir':('time',sea_Dir)
    },
        coords = {'time':time}
    )

    # solve sectors
    c = 1
    for s_ini, s_end in swell_sectors:
        if s_ini < s_end:
            p_sw = np.where((cat_dir <= s_end) & (cat_dir > s_ini))
        else:
            p_sw = np.where((cat_dir <= s_end) | (cat_dir > s_ini))

        # get data inside sector
        sect_dir = np.zeros(cat_dir.shape)*np.nan
        sect_hs = np.zeros(cat_dir.shape)*np.nan
        sect_tp = np.zeros(cat_dir.shape)*np.nan

        sect_dir[p_sw] = cat_dir[p_sw]
        sect_hs[p_sw] = cat_hs[p_sw]
        sect_tp[p_sw] = cat_tp[p_sw]

        # calculate swell Hs, Tp, Dir
        swell_Hs = np.sqrt(np.nansum(np.power(sect_hs,2), axis=1))

        swell_Tp = np.sqrt(
            np.nansum(np.power(sect_hs,2), axis=1) /
            np.nansum(np.power(sect_hs,2)/np.power(sect_tp,2), axis=1)
        )
        swell_Dir = np.arctan2(
            np.nansum(np.power(sect_hs,2) * sect_tp * np.sin(sect_dir*np.pi/180), axis=1),
            np.nansum(np.power(sect_hs,2) * sect_tp * np.cos(sect_dir*np.pi/180), axis=1)
        )

        # dir correction and denormalization 
        swell_Dir[np.where((swell_Dir<0))] = swell_Dir[np.where((swell_Dir<0))]+2*np.pi
        swell_Dir = swell_Dir*180/np.pi

        # dont do arctan2 if there is only one dir
        i_onedir = np.where(
            (np.count_nonzero(~np.isnan(sect_dir),axis=1)==1)
        )[0]
        swell_Dir[i_onedir] = np.nanmin(sect_dir[i_onedir], axis=1)

        # out of bound dir correction
        swell_Dir[np.where((swell_Dir>360))] = swell_Dir[np.where((swell_Dir>360))]-360
        swell_Dir[np.where((swell_Dir<0))] = swell_Dir[np.where((swell_Dir<0))]+360


        # fix swell all-nans to 0s nansum behaviour
        p_fix = np.where(swell_Hs==0)
        swell_Hs[p_fix] = np.nan
        swell_Dir[p_fix] = np.nan

        # append data to partitons dataset
        xds_parts['swell_{0}_Hs'.format(c)] = ('time', swell_Hs)
        xds_parts['swell_{0}_Tp'.format(c)] = ('time', swell_Tp)
        xds_parts['swell_{0}_Dir'.format(c)] = ('time', swell_Dir)
        c+=1

    return xds_parts

def GetDistribution_ws(xds_wps, swell_sectors):
    '''
    Separates wave partitions (0-5) into families.
    Default: sea, swl1, swl2

    xds_wps (waves partitionss):
        xarray.Dataset (time,), phs, pspr, pwfrac... {0-5 partitions}

    sectors: list of degrees to cut wave energy [(a1, a2), (a2, a3), (a3, a1)]

    returns
        xarray.Dataset (time,), fam_V, {fam: sea,swell_1,swell2. V: Hs,Tp,Dir}
    '''

    sea_Hs= xds_wps.isel(part=0).hs.values
    sea_Tp=xds_wps.isel(part=0).tp.values
    sea_Dir = xds_wps.isel(part=0).dpm.values
    time= xds_wps.time.values

    # concatenate energy groups 
    cat_hs = np.column_stack(
        (xds_wps.isel(part=1).hs.values,
        xds_wps.isel(part=2).hs.values,
        xds_wps.isel(part=3).hs.values,
        xds_wps.isel(part=4).hs.values,
        xds_wps.isel(part=5).hs.values)
    )
    cat_tp = np.column_stack(
        (xds_wps.isel(part=1).tp.values,
        xds_wps.isel(part=2).tp.values,
        xds_wps.isel(part=3).tp.values,
        xds_wps.isel(part=4).tp.values,
        xds_wps.isel(part=5).tp.values)
    )
    cat_dir = np.column_stack(
        (xds_wps.isel(part=1).dpm.values,
        xds_wps.isel(part=2).dpm.values,
        xds_wps.isel(part=3).dpm.values,
        xds_wps.isel(part=4).dpm.values,
        xds_wps.isel(part=5).dpm.values)
    )

    # prepare output array
    xds_parts = xr.Dataset({
        'sea_Hs':('time',sea_Hs),
        'sea_Tp':('time',sea_Tp),
        'sea_Dir':('time',sea_Dir)
    },
        coords = {'time':time}
    )

    # solve sectors
    c = 1
    for s_ini, s_end in swell_sectors:
        if s_ini < s_end:
            p_sw = np.where((cat_dir <= s_end) & (cat_dir > s_ini))
        else:
            p_sw = np.where((cat_dir <= s_end) | (cat_dir > s_ini))

        # get data inside sector
        sect_dir = np.zeros(cat_dir.shape)*np.nan
        sect_hs = np.zeros(cat_dir.shape)*np.nan
        sect_tp = np.zeros(cat_dir.shape)*np.nan

        sect_dir[p_sw] = cat_dir[p_sw]
        sect_hs[p_sw] = cat_hs[p_sw]
        sect_tp[p_sw] = cat_tp[p_sw]

        # calculate swell Hs, Tp, Dir
        swell_Hs = np.sqrt(np.nansum(np.power(sect_hs,2), axis=1))

        swell_Tp = np.sqrt(
            np.nansum(np.power(sect_hs,2), axis=1) /
            np.nansum(np.power(sect_hs,2)/np.power(sect_tp,2), axis=1)
        )
        swell_Dir = np.arctan2(
            np.nansum(np.power(sect_hs,2) * sect_tp * np.sin(sect_dir*np.pi/180), axis=1),
            np.nansum(np.power(sect_hs,2) * sect_tp * np.cos(sect_dir*np.pi/180), axis=1)
        )

        # dir correction and denormalization 
        swell_Dir[np.where((swell_Dir<0))] = swell_Dir[np.where((swell_Dir<0))]+2*np.pi
        swell_Dir = swell_Dir*180/np.pi

        # dont do arctan2 if there is only one dir
        i_onedir = np.where(
            (np.count_nonzero(~np.isnan(sect_dir),axis=1)==1)
        )[0]
        swell_Dir[i_onedir] = np.nanmin(sect_dir[i_onedir], axis=1)

        # out of bound dir correction
        swell_Dir[np.where((swell_Dir>360))] = swell_Dir[np.where((swell_Dir>360))]-360
        swell_Dir[np.where((swell_Dir<0))] = swell_Dir[np.where((swell_Dir<0))]+360


        # fix swell all-nans to 0s nansum behaviour
        p_fix = np.where(swell_Hs==0)
        swell_Hs[p_fix] = np.nan
        swell_Dir[p_fix] = np.nan

        # append data to partitons dataset
        xds_parts['swell_{0}_Hs'.format(c)] = ('time', swell_Hs)
        xds_parts['swell_{0}_Tp'.format(c)] = ('time', swell_Tp)
        xds_parts['swell_{0}_Dir'.format(c)] = ('time', swell_Dir)
        c+=1

    return xds_parts

def AWL(hs, tp):
    'Returns Atmospheric Water Level'

    return 0.043*(hs*1.56*(tp/1.00)**2)**(0.5)

def TWL(awl, ss, at, mmsl):
    'Returns Total Water Level'

    return awl + ss + at + mmsl

def AnnualMaxima(xds_data, var_name):
    '''
    Calculate annual maxima for "var_name" (time index not monotonic)
    requires xarray.Dataset with var_name (time)

    returns xarray.Dataset with selection of annual maxima
    '''

    # get TWL and times
    ts = xds_data.time.values[:]
    vs = xds_data[var_name].values[:]

    # years array
    ys, _, _ = get_years_months_days(ts)  # aux. avoid time type problems 
    us = np.unique(ys)

    # iterate over years
    p_amax = []
    for y in us:

        # find year max TWL position
        y_vs = vs[np.where(y==ys)]
        y_time = ts[np.where(y==ys)]
        p_mt = np.where(y_vs == np.max(y_vs))[0][0]

        yt_mt = y_time[p_mt]
        p_mt = np.where(ts == yt_mt)[0][0]

        p_amax.append(p_mt)

    # Select annual maxima
    xds_AMAX = xds_data.isel(time=p_amax)

    return xds_AMAX

def Aggregate_WavesFamilies(wvs_fams):
    '''
    Aggregate Hs, Tp and Dir from waves families data

    wvs_fams (waves families):
        xarray.Dataset (time,), fam1_Hs, fam1_Tp, fam1_Dir, ...
        {any number of families}

    returns Hs, Tp, Dir (numpy.array)
    '''

    # get variable names
    vs = [str(x) for x in wvs_fams.keys()]
    vs_Hs = [x for x in vs if x.endswith('_Hs')]
    vs_Tp = [x for x in vs if x.endswith('_Tp')]
    vs_Dir = [x for x in vs if x.endswith('_Dir')]

    # join variable values
    vv_Hs = np.column_stack([wvs_fams[v].values[:] for v in vs_Hs])
    vv_Tp = np.column_stack([wvs_fams[v].values[:] for v in vs_Tp])
    vv_Dir = np.column_stack([wvs_fams[v].values[:] for v in vs_Dir])

    # Hs from families
    HS = np.sqrt(np.nansum(np.power(vv_Hs,2), axis=1))

    # Hs maximun position 
    p_max_hs = np.nanargmax(vv_Hs, axis=1)

    # Tp from families (Hs max pos)
    TP = np.array([r[i] for r,i in zip(vv_Tp, p_max_hs)])

    # Dir from families (Hs max pos)
    DIR = np.array([r[i] for r,i in zip(vv_Dir, p_max_hs)])

    # TP from families 
    #tmp1 = np.power(vv_Hs,2)
    #tmp2 = np.divide(np.power(vv_Hs,2), np.power(vv_Tp,2))
    #TP = np.sqrt(np.nansum(tmp1, axis=1) / np.nansum(tmp2, axis=1))

    # Dir from families
    #tmp3 = np.arctan2(
    #    np.sum(np.power(vv_Hs,2) * vv_Tp * np.sin(vv_Dir * np.pi/180), axis=1),
    #    np.sum(np.power(vv_Hs,2) * vv_Tp * np.cos(vv_Dir * np.pi/180), axis=1)
    #)
    #tmp3[tmp3<0] = tmp3[tmp3<0] + 2*np.pi
    #DIR = tmp3 * 180/np.pi

    # return xarray.Dataset
    xds_AGGR = xr.Dataset(
        {
            'Hs': (('time',), HS),
            'Tp': (('time',), TP),
            'Dir': (('time',), DIR),
        },
        coords = {
            'time': wvs_fams.time.values[:]  # get time from input
        }
    )

    return xds_AGGR

def Intradaily_Hydrograph(xds_wvs, xds_tcs):
    '''
    Calculates intradaily hydrograph (hourly) from a time series of storms.
    storms waves data (hs, tp, dir) and TCs data (mu, tau, ss) is needed.

    xds_wvs (waves aggregated):
        xarray.Dataset (time,), Hs, Tp, Dir

    xds_tcs (TCs):
        xarray.Dataset (time,), mu, tau, ss

    returns xarray.Dataset (time,), Hs, Tp, Dir, SS  (hourly)
    '''

    # input data (storms aggregated waves)
    Hs = xds_wvs.Hs.values[:]
    Tp = xds_wvs.Tp.values[:]
    Dir = xds_wvs.Dir.values[:]
    ts = xds_wvs.time.values[:]

    # fix times
    # TODO: this should not be needed
    if isinstance(ts[0], np.datetime64):
        ts = [npdt64todatetime(x) for x in ts]

    # input data (storms TCs)
    tau = xds_tcs.tau.values[:]  # storm max. instant (0-1)
    mu = xds_tcs.mu.values[:]
    ss = xds_tcs.ss.values[:]

    # storm durations
    s_dur_d = np.array([x.days for x in np.diff(ts)])  # days
    s_dur_h = s_dur_d * 24  # hours
    s_cs_h = np.cumsum(s_dur_h)  # hours since time start
    s_cs_h = np.insert(s_cs_h,0,0)

    # storm tau max (hourly)
    tau_h = np.floor(s_cs_h[:-1] + s_dur_h * tau[:-1])

    # aux function
    def CalcHydro(vv, vt, tt, mt):
        '''
        Calculate variable hourly hydrograph.
        vv - var value at max.
        vt - var time (hours since start, at hydrograph extremes)
        tt - tau max time (hours since start).
        mt - mu value
        '''

        # var value at hydrographs extremes
        vv_extr = vv * np.power(2*mt-1, 2)

        # make it continuous
        vv_extr_cont = (np.roll(vv_extr,1) + vv_extr) / 2
        vv_extr_cont[0] = vv_extr_cont[1]
        vv_extr_cont[-1] = vv_extr_cont[-2]

        # join hydrograph max. and extremes variable data
        vt_full = np.concatenate([vt, tt])  # concatenate times (used for sorting)
        vv_full = np.concatenate([vv_extr_cont, vv])

        # sort data
        ix = np.argsort(vt_full)
        vt_sf = vt_full[ix]
        vv_sf = vv_full[ix]

        # interpolate to fill all hours
        h_times = np.arange(vt_sf[0], vt_sf[-1] + 1, 1)
        h_values = np.interp(h_times, vt_sf, vv_sf)

        # fix times
        h_times = h_times.astype(int)

        return h_values, h_times

    # hydrograph variables: hs and ss
    hourly_Hs, hourly_times = CalcHydro(Hs, s_cs_h, tau_h, mu)
    hourly_ss, _ = CalcHydro(ss, s_cs_h, tau_h, mu)

    # resample waves data to hourly (pad Tp and Dir)
    xds_wvs_h = xds_wvs.resample(time='1H').pad()

    # add Hs and SS 
    xds_wvs_h['Hs'] =(('time',), hourly_Hs)
    xds_wvs_h['SS'] =(('time',), hourly_ss)

    return xds_wvs_h

