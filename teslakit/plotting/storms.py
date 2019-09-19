#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib.patches import Circle

# teslakit
from ..storms import GeoAzimuth

# import constants
from .config import _faspect, _fsize, _fdpi

def get_storm_linestyle(vel):
    if vel >= 60:
        return 'solid'
    elif vel >= 40:
        return 'dashed'
    elif vel >= 20:
        return 'dashdot'
    else:
        return 'dotted'

def get_storm_color(categ):

    dcs = {
        0 : 'green',
        1 : 'yellow',
        2 : 'orange',
        3 : 'red',
        4 : 'purple',
        5 : 'black',
    }

    return dcs[categ]


def axplot_Tracks_Circle(ax, lon_point, lat_point, r_point,
                         lon_storms, lat_storms, index_in, index_out,
                         categs, vel):
    'axes plot storm tracks inside radius circle'

    n_storms = len(categs)

    # plot point and  circle
    ax.plot(lon_point, lat_point, '.b', markersize=12, zorder=100)
    circle = plt.Circle(
        (lon_point, lat_point), r_point,
        edgecolor='k', fill=False)
    ax.add_patch(circle)

    # plot storms
    for lon_s, lat_s, i_i, i_o, v, c in zip(
        lon_storms, lat_storms, index_in, index_out, vel, categs):

        # get storm data
        lon_in = lon_s[i_i:i_o+1]
        lat_in = lat_s[i_i:i_o+1]

        # linestyle (velocity) and linecolor (category)
        ls = get_storm_linestyle(v)
        cs = get_storm_color(c)

        # plot storm and point at entry
        ax.plot(lon_in, lat_in, linestyle=ls, color=cs)
        ax.plot(lon_in[0], lat_in[0], '.', markersize=8, color=cs, zorder=99)

    # customize axes
    ax.axis('equal')
    ax.axis('off')
    ax.set_title(
        'Historical Tracks ({0} TCs inside R={1}º)'.format(n_storms, r_point),
        {'fontsize': 14, 'fontweight':'bold'}
    )

def axplot_Parametrized_Circle(ax, lon_point, lat_point, r_point,
                               delta_storms, gamma_storms, categs, vel):
    'axes plot storm parametrized tracks inside radius circle'

    n_storms = len(categs)

    # plot point and  circle
    ax.plot(lon_point, lat_point, '.b', markersize=12, zorder=100)
    circle = plt.Circle(
        (lon_point, lat_point), r_point,
        edgecolor='k', fill=False)
    ax.add_patch(circle)

    # TODO: def functions  track --> params / params --> track 

    # start-end points aux. opts.
    n = 1000
    ang = np.linspace(0, 2*np.pi, n)
    x_ang = r_point * np.cos(ang) + lat_point
    y_ang = r_point * np.sin(ang) + lon_point

    center_radius = np.asarray(
        [GeoAzimuth(lat_point, lon_point, x, y) for x, y in zip(x_ang, y_ang)])

    # plot storms
    for delta, gamma, v, c in zip(delta_storms, gamma_storms, vel, categs):

        # find most similar angle
        im = np.argmin(np.absolute(center_radius - delta))
        lon_entry = y_ang[im]
        lat_entry = x_ang[im]

        ent_radius = np.asarray(
            [GeoAzimuth(x, y, x_ang[im], y_ang[im]) for x, y in zip(
                x_ang, y_ang)])

        im2 = np.argmin(np.absolute(ent_radius - gamma))
        lon_exit = y_ang[im2]
        lat_exit = x_ang[im2]

        # linestyle (velocity) and linecolor (category)
        ls = get_storm_linestyle(v)
        cs = get_storm_color(c)

        # plot storm and point at entry
        ax.plot([lon_entry, lon_exit], [lat_entry, lat_exit], linestyle=ls, color=cs)
        ax.plot(lon_entry, lat_entry, '.', markersize=8, color=cs, zorder=99)

    # customize axes 
    ax.axis('equal')
    ax.axis('off')
    ax.set_title(
        'Parametrized Tracks',
        {'fontsize': 14, 'fontweight':'bold'}
    )

def axlegend_categ(ax):
    'add custom legend (storm category) to axes'

    # category legend
    lls_cat = [
        Line2D([0], [0], color = 'black'),
        Line2D([0], [0], color = 'purple'),
        Line2D([0], [0], color = 'red'),
        Line2D([0], [0], color = 'orange'),
        Line2D([0], [0], color = 'yellow'),
        Line2D([0], [0], color = 'green'),
    ]
    leg_cat = Legend(
        ax, lls_cat, ['5','4','3','2','1','0'],
        title = 'Category', bbox_to_anchor = (1.01, 1), loc='upper left',
    )
    ax.add_artist(leg_cat)

def axlegend_vel(ax):
    'add custom legend (storm velocity) to axes'

    # velocity legend
    lls_vel = [
        Line2D([0], [0], color = 'black', ls = 'solid'),
        Line2D([0], [0], color = 'black', ls = 'dashed'),
        Line2D([0], [0], color = 'black', ls = 'dashdot'),
        Line2D([0], [0], color = 'black', ls = 'dotted'),
    ]
    leg_vel = Legend(
        ax, lls_vel, ['> 60 km/h', '> 40 km/h', '> 20 km/h', '< 20 km/h'],
        title = 'Velocity', bbox_to_anchor = (1.01, 0.7), loc= 'upper left',
    )
    ax.add_artist(leg_vel)


def Plot_TCs_TracksParams(TCs_tracks, TCs_params, p_export=None):
    'Plots storms tracks and storms parametrized'

    # custom wmo names
    nm_lon = 'lon_wmo'
    nm_lat = 'lat_wmo'

    # get data from historical tracks
    n_storms = TCs_tracks.storm.shape[0]
    lon_storms = TCs_tracks[nm_lon].values[:]
    lat_storms = TCs_tracks[nm_lat].values[:]

    # get data from parametrized tracks
    lon_point = TCs_params.point_lon
    lat_point = TCs_params.point_lat
    r_point = TCs_params.point_r

    index_in = TCs_params.index_in.values[:]
    index_out = TCs_params.index_out.values[:]
    categs_storms = TCs_params.category.values[:]
    vel_storms = TCs_params.velocity_mean.values[:]
    gamma_storms = TCs_params.gamma.values[:]
    delta_storms = TCs_params.delta.values[:]

    # figure
    fig, axs = plt.subplots(ncols=2, figsize=(_faspect*_fsize, _fsize))

    # storm tracks
    axplot_Tracks_Circle(
        axs[0], lon_point, lat_point, r_point,
        lon_storms, lat_storms, index_in, index_out,
        categs_storms, vel_storms
    )

    # storms  parametrized
    axplot_Parametrized_Circle(
        axs[1], lon_point, lat_point, r_point,
        delta_storms, gamma_storms, categs_storms, vel_storms
    )

    # small axis lims fix
    axs[0].set_xlim(axs[1].get_xlim())
    axs[0].set_ylim(axs[1].get_ylim())

    # add custom legends
    axlegend_categ(axs[1])
    axlegend_vel(axs[1])

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

# TODO: small refactor 
def Plot_Historical_TCs_Tracks(xds_TCs_r1, xds_TCs_r2,
                               lon1, lon2, lat1, lat2,
                               pnt_lon, pnt_lat, r1, r2,
                               p_export=None):
    'Plot Historical TCs tracks map, requires basemap module'

    try:
        from mpl_toolkits.basemap import Basemap
    except:
        print('basemap module required.')
        return

    fig, ax = plt.subplots(1, figsize=(_faspect*_fsize, _fsize))

    # setup mercator map projection.
    m = Basemap(
        llcrnrlon = lon1, llcrnrlat = lat1,
        urcrnrlon = lon2, urcrnrlat = lat2,
        resolution = 'l', projection = 'cyl',
        lat_0 = lat1, lon_0 = lon1, area_thresh = 0.01,
    )
    m.drawcoastlines()
    m.fillcontinents(color = 'silver')
    m.drawmapboundary(fill_color = 'lightcyan')
    m.drawparallels(np.arange(lat1, lat2, 20), labels = [1,1,0,0])
    m.drawmeridians(np.arange(lon1, lon2, 20), labels = [0,0,0,1])

    # plot r1 storms
    for s in range(len(xds_TCs_r1.storm)):
        lon = xds_TCs_r1.isel(storm = s).lon_wmo.values[:]
        lon[np.where(lon<0)] = lon[np.where(lon<0)] + 360

        if s==0:
            ax.plot(
                lon, xds_TCs_r1.isel(storm = s).lat_wmo.values[:],
                '-', color = 'grey', alpha = 0.5,
                label = 'Enter {0}° radius'.format(r1)
            )
        else:
            ax.plot(
                lon, xds_TCs_r1.isel(storm = s).lat_wmo.values[:],
                '-', color = 'grey',alpha = 0.5
            )
            ax.plot(
                lon[0], xds_TCs_r1.isel(storm = s).lat_wmo.values[0],
                '.', color = 'grey', markersize = 10
            )

    # plot r2 storms
    for s in range(len(xds_TCs_r2.storm)):
        lon = xds_TCs_r2.isel(storm = s).lon_wmo.values[:]
        lon[np.where(lon<0)] = lon[np.where(lon<0)] + 360

        if s==0:
            ax.plot(
                lon, xds_TCs_r2.isel(storm = s).lat_wmo.values[:],
                color = 'indianred', alpha = 0.8,
                label = 'Enter {0}° radius'.format(r2)
            )
        else:
            ax.plot(
                lon, xds_TCs_r2.isel(storm = s).lat_wmo.values[:],
                color = 'indianred', alpha = 0.8
            )
            ax.plot(
                lon[0], xds_TCs_r2.isel(storm = s).lat_wmo.values[0],
                '.', color = 'indianred', markersize = 10
            )

    # plot point
    ax.plot(
        pnt_lon, pnt_lat, '.',
        markersize = 15, color = 'brown',
        label = 'STUDY SITE'
    )

    # plot r1 circle
    circle = Circle(
        m(pnt_lon, pnt_lat), r1,
        facecolor = 'grey', edgecolor = 'grey',
        linewidth = 3, alpha = 0.5,
        label='{0}° Radius'.format(r1)
    )
    ax.add_patch(circle)

    # plot r2 circle
    circle2 = Circle(
        m(pnt_lon, pnt_lat), r2,
        facecolor = 'indianred', edgecolor = 'indianred',
        linewidth = 3, alpha = 0.8,
        label='{0}° Radius'.format(r2))
    ax.add_patch(circle2)

    # customize axes
    ax.set_aspect(1.0)
    ax.set_ylim(lat1, lat2)
    ax.set_title('Historical TCs', fontsize=15)
    ax.legend(loc=0, fontsize=14)

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

def Plot_Historical_TCs_Tracks_Category(xds_TCs_r1, cat,
                                        lon1, lon2, lat1, lat2,
                                        pnt_lon, pnt_lat, r1,
                                        p_export=None):
    'Plot Historical TCs category map, requires basemap module'

    try:
        from mpl_toolkits.basemap import Basemap
    except:
        print('basemap module required.')
        return

    fig, ax = plt.subplots(1, figsize=(_faspect*_fsize, _fsize))

    # setup mercator map projection.
    m = Basemap(
        llcrnrlon = lon1, llcrnrlat = lat1,
        urcrnrlon = lon2, urcrnrlat = lat2,
        resolution = 'l', projection = 'cyl',
        lat_0 = lat1, lon_0 = lon1, area_thresh=0.01
    )
    m.drawcoastlines()
    m.fillcontinents(color = 'silver')
    m.drawmapboundary(fill_color = 'lightcyan')
    m.drawparallels(np.arange(lat1, lat2, 20), labels = [1,0,0,0])
    m.drawmeridians(np.arange(lon1, lon2, 20), labels = [0,0,0,1])

    for s in range(len(xds_TCs_r1.storm)):
        lon = xds_TCs_r1.isel(storm = s).lon_wmo.values[:]
        lon[np.where(lon<0)] = lon[np.where(lon<0)] + 360

        if s==0:
            ax.plot(
                lon, xds_TCs_r1.isel(storm = s).lat_wmo.values[:],
                '-', color = get_storm_color(int(cat[s].values)),
                alpha = 0.5, label = 'Enter {0}° radius'.format(r1)
            )
        else:
            ax.plot(
                lon, xds_TCs_r1.isel(storm = s).lat_wmo.values[:],
                '-', color = get_storm_color(int(cat[s].values)),
                alpha = 0.5,
            )
            ax.plot(
                lon[0], xds_TCs_r1.isel(storm = s).lat_wmo.values[0],
                '.', color = get_storm_color(int(cat[s].values)),
                markersize = 10,
            )

    # plot point
    ax.plot(
        pnt_lon, pnt_lat, '.',
        markersize = 15, color = 'brown',
        label = 'STUDY SITE'
    )

    # plot circle
    circle = Circle(
        m(pnt_lon, pnt_lat), r1,
        facecolor = 'grey', edgecolor = 'grey',
        linewidth = 3, alpha = 0.5,
        label='Radius {0}º'.format(r1)
    )
    ax.add_patch(circle)

    # customize axes
    ax.set_aspect(1.0)
    ax.set_ylim(lat1,lat2)
    ax.set_title('Historical TCs', fontsize=15)
    ax.legend(loc=0, fontsize=14)
    axlegend_categ(ax)

    # show / export
    if not p_export:
        plt.show()
    else:
        fig.savefig(p_export, dpi=_fdpi)
        plt.close()

