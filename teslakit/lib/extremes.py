#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from scipy.stats import  gumbel_l, genextreme, spearmanr
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.special import ndtri  # norm inv
from itertools import permutations
import xarray as xr

def ChromMatrix(vs):
    'Return chromosome matrix for np.array vs (n x nvars)'

    n_cols = vs.shape[1]
    chrom = np.empty((0,n_cols), int)
    b = np.zeros(n_cols)
    for c in range(n_cols):
        b[c] = 1
        for r in set(permutations(b.tolist())):
            chrom = np.row_stack([chrom, np.array(r)])

    return chrom

def ChromosomesProbabilities_KMA(bmus, n_clusters, vars_chrom):
    '''
    Calculate chromosomes probabilities for KMA bmus and variables at
    vars_chrom

    bmus - KMA bmus (time series of KMA centroids)
    n_clusters - number of KMA clusters
    vars_chrom - vars to get chromosomes probabilities, np.array (n x nvars)
                 nan data is considered a "0" chromosome, "1" otherwise
    '''

    # get chromosomes matrix
    chrom = ChromMatrix(vars_chrom)

    # calculate chromosomes probabilities
    probs = np.zeros((n_clusters, chrom.shape[0]))
    for i in range(n_clusters):
        c = i+1
        pos = np.where((bmus==c))[0]

        # get variables chromosomes at cluster
        var_c = vars_chrom[pos,:]
        var_c[~np.isnan(var_c)] = 1
        var_c[np.isnan(var_c)] = 0

        # count chromosomes
        ucs, ccs = np.unique(var_c, return_counts=True, axis=0)
        tcs = var_c.shape[0]

        # get probs of each chromosome
        for uc, cc in zip(ucs, ccs):

            # skip all empty chromosomes
            if ~uc.any(): continue

            pc = np.where(np.all(uc == chrom, axis=1))[0][0]
            probs[i, pc] = cc / tcs

    return chrom, probs

def FitGEV_KMA_Frechet(bmus, n_clusters, var):
    '''
    Returns stationary GEV/Gumbel_L params for KMA bmus and varible series

    bmus - KMA bmus (time series of KMA centroids)
    n_clusters - number of KMA clusters
    var - time series of variable to fit to GEV/Gumbel_L

    returns np.array [n_clusters x 3 parameters (shape, loc, scale)]
    for gumbel distributions shape value will be ~0 (0.0000000001)
    '''

    param_GEV = np.empty((n_clusters, 3))
    for i in range(n_clusters):
        c = i+1
        pos = np.where((bmus==c))[0]

        if len(pos) == 0:
            param_GEV[i,:] = [np.nan, np.nan, np.nan]

        else:

            # get variable at cluster position
            var_c = var[pos]
            var_c = var_c[~np.isnan(var_c)]

            # fit to Gumbel_l and get negative loglikelihood
            loc_gl, scale_gl = gumbel_l.fit(-var_c)
            theta_gl = (0.0000000001, -1*loc_gl, scale_gl)
            nLogL_gl = genextreme.nnlf(theta_gl, var_c)

            # fit to GEV and get negative loglikelihood
            c = -0.1
            shape_gev, loc_gev, scale_gev = genextreme.fit(var_c, c)
            theta_gev = (shape_gev, loc_gev, scale_gev)
            nLogL_gev = genextreme.nnlf(theta_gev, var_c)

            # store negative shape
            theta_gev_fix = (-shape_gev, loc_gev, scale_gev)

            # apply significance test if Frechet
            if shape_gev < 0:  # TODO: signo alreves ??? hablar con Fer
                if nLogL_gl - nLogL_gev >= 1.92:
                    param_GEV[i,:] = list(theta_gev_fix)
                else:
                    param_GEV[i,:] = list(theta_gl)
            else:
                param_GEV[i,:] = list(theta_gev_fix)

    return param_GEV

def SampleGEV_KMA_Smooth(bmus, n_clusters, param_GEV, var):
    '''
    TODO: resolver/encontrar la matriz de informacion Fisher
    Generar un gev_params_sample (n_clustersx3) por simulacion
    respetando los param_GEV (n_clusterx3) input
    '''

    print('SampleGEV_KMA_Smooth no programada. devuelve param_GEV')
    return param_GEV
    # TODO: NECESITO SACAR LA MATRIZ DE INFORMACION FISHER

    # gev parameters
    shape_gev = param_GEV[:,0]
    loc_gev = param_GEV[:,1]
    scale_gev = param_GEV[:,2]

    # location parameter
    index = np.ones((1, n_clusters))
    mu_strips = loc_gev - np.multiply(
        np.divide(scale_gev, shape_gev),
        1 - np.power(index, shape_gev)
    ).squeeze()

    psi_strips = np.multiply(
        scale_gev, np.power(index, shape_gev)
    )

    # Gumbel parameters
    pos_gumbel = np.where((shape_gev==0.0000000001))[0]
    loc_gumbel = -1*loc_gev[pos_gumbel]
    scale_gumbel = scale_gev[pos_gumbel]

    mu_strips[pos_gumbel] = loc_gumbel + np.multiply(
        scale_gumbel, np.log(index[:,pos_gumbel].squeeze())
    )

    # KMA bmus
    for i in range(n_clusters):
        c = i+1
        pos = np.where((bmus==c))[0]

        if len(pos) == 0:
            #param_SIM[i,:] = [np.nan, np.nan, np.nan]
            pass

        else:

            # get variable at cluster position
            var_c = var[pos]
            var_c = var_c[~np.isnan(var_c)]

            if i in pos_gumbel:
                pass

            else:
                # TODO ACOV?
                acov = genextreme.stats(
                    -1*shape_gev[i], loc_gev[i], scale_gev[i],
                    moments='sk'
                )
                pass

                #param_SIM = [shape_gev[i], mu_strips[i], psi_strips[i]]

    return None

def Correlation_Smooth_Partitions(
    bmus, cenEOFs, n_clusters, xds_waves, wvs_fams,
    xds_GEV_params, chrom):
    '''
    TODO: Documentar
    ojo, bmus y waves tienen que compartir tiempo
    ojo, mas adelante hacer la libreria homogenea
    meter kma en xarray, indicar que son waves_WT y KMA_WT (maxTWL)

    '''

    def Smooth_GEV_Shape(cenEOFs, param):
        '''
        TODO: Documentar
        '''
        n_cs = cenEOFs.shape[0]

        D = np.empty((n_cs, n_cs))
        sort_ord = np.empty((n_cs, n_cs), dtype=int)
        D_sorted = np.empty((n_cs, n_cs))

        param_c = np.empty(n_cs)

        for i in range(n_cs):
            for k in range(n_cs):
                D[i,k] = np.sqrt(np.sum(np.power(cenEOFs[i,:]-cenEOFs[k,:], 2)))
            D[i,i] = np.nan

            order = np.argsort(D[i,:])
            sort_ord[i,:] = order
            D_sorted[i,:] = D[i, order]

        for i in range(n_cs):
            denom = np.sum(
                [1/D_sorted[i,0], 1/D_sorted[i,1],
                 1/D_sorted[i,2], 1/D_sorted[i,3]]
            )

            param_c[i] = 0.5 * np.sum(
                [
                    param[i],
                    param[sort_ord[i,0]] * np.divide(1/D_sorted[i,0], denom),
                    param[sort_ord[i,1]] * np.divide(1/D_sorted[i,1], denom),
                    param[sort_ord[i,2]] * np.divide(1/D_sorted[i,2], denom),
                    param[sort_ord[i,3]] * np.divide(1/D_sorted[i,3], denom),
                ]
            )

        return param_c


    # smooth GEV shape parameter 
    d_shape = {}
    for wf in wvs_fams:
        for wv in ['Hs', 'Tp']:
            vn = '{0}_{1}'.format(wf, wv)
            sh_GEV = xds_GEV_params.sel(parameter='shape')[vn].values[:]
            d_shape[vn] = Smooth_GEV_Shape(cenEOFs, sh_GEV)

    # Get sigma correlation for each KMA cluster 
    l_sigma = []
    for i in range(n_clusters):
        c = i+1
        pos = np.where((bmus==c))[0]

        # current cluster waves
        xds_K_wvs = xds_waves.isel(time=pos)

        # get chromosomes from waves (0/1)
        var_c = np.column_stack(
            [xds_K_wvs['{0}_Hs'.format(x)].values[:] for x in wvs_fams]
        )
        var_c[~np.isnan(var_c)] = 1
        var_c[np.isnan(var_c)] = 0

        # get general chromosomes matrix
        chrom = ChromMatrix(var_c)

        # get sigma for each chromosome
        for uc in chrom:
            wt_crom = 1  # data / no data 

            # find data position for this chromosome
            p_c = np.where((var_c == uc).all(axis=1))[0]

            # TODO: comentar con Fer , por que coger todos esos??
            # if not enought data, get all chromosomes with shared 1s
            if len(p_c) < 20:
                p1s = np.where(uc==1)[0]
                p_c = np.where((var_c[:,p1s] == uc[p1s]).all(axis=1))[0]

                wt_crom = 0  # data / no data 

            # select waves chrom data 
            xds_chr_wvs = xds_K_wvs.isel(time=p_c)

            # solve normal inverse CDF for each active chromosome
            to_corr = np.empty((0,len(p_c)))  # append for spearman correlation
            for i_c in np.where(uc==1)[0]:

                # get wave family chromosome variables
                fam_n = wvs_fams[i_c]
                vv_Hs = xds_chr_wvs['{0}_Hs'.format(fam_n)].values[:]
                vv_Tp = xds_chr_wvs['{0}_Tp'.format(fam_n)].values[:]
                vv_Dir = xds_chr_wvs['{0}_Dir'.format(fam_n)].values[:]

                # GEV cdf Hs 
                vn = '{0}_Hs'.format(fam_n)
                sha_g = d_shape[vn][i]
                loc_g = xds_GEV_params.sel(parameter='location')[vn].values[i]
                sca_g = xds_GEV_params.sel(parameter='scale')[vn].values[i]
                norm_Hs = genextreme.cdf(vv_Hs, -1*sha_g, loc_g, sca_g)

                # GEV cdf Tp 
                vn = '{0}_Tp'.format(fam_n)
                sha_g = d_shape[vn][i]
                loc_g = xds_GEV_params.sel(parameter='location')[vn].values[i]
                sca_g = xds_GEV_params.sel(parameter='scale')[vn].values[i]
                norm_Tp = genextreme.cdf(vv_Tp, -1*sha_g, loc_g, sca_g)

                # ECDF dir
                ecdf = ECDF(vv_Dir)
                norm_Dir = ecdf(vv_Dir)

                # normal inverse CDF 
                u_cdf = np.column_stack([norm_Hs, norm_Tp, norm_Dir])
                u_cdf[u_cdf>=1.0] = 0.999999
                inv_n = ndtri(u_cdf)

                # concatenate data for correlation
                to_corr = np.concatenate((to_corr, inv_n.T), axis=0)

            # sigma: spearman correlation
            corr, pval = spearmanr(to_corr, axis=1)

            # store data
            sg = {}
            sg['wt'] = c
            sg['crom'] = uc
            sg['corr'] = corr
            sg['data'] = len(p_c)
            sg['wt_crom'] = wt_crom
            l_sigma.append(sg)

    return l_sigma

def Climate_Emulator(xds_WVS_MS, xds_KMA_MS):
    '''
    TODO: Doc

    xds_WVS_MaxStorm - 
    xds_KMA_MaxStorm - 
    '''

    # get variables
    bmus = xds_KMA_MS.bmus.values[:]
    cenEOFs = xds_KMA_MS.cenEOFs.values[:]
    n_clusters = len(xds_KMA_MS.n_clusters)


    # Fit each wave family var to GEV distribution (using KMA bmus)
    xds_gev_params = xr.Dataset(
        coords={
            'n_cluster' : np.arange(n_clusters)+1,
            'parameter' : ['shape', 'location', 'scale'],
        }
    )
    vars_fit = ['sea_Hs', 'sea_Tp', 'swell_1_Hs', 'swell_1_Tp', 'swell_2_Hs', 'swell_2_Tp']
    for vn in vars_fit:
        gp_pars = FitGEV_KMA_Frechet(bmus, n_clusters, xds_WVS_MS[vn].values)
        xds_gev_params[vn] = (('n_cluster', 'parameter',), gp_pars)


    # Calculate chromosomes and chromosomes probabilities
    vars_chrom = ['sea_Hs', 'swell_1_Hs', 'swell_2_Hs']
    np_vars_chrom = np.column_stack(
        [xds_WVS_MS[vn].values for vn in vars_chrom]
    )

    chrom, chrom_probs = ChromosomesProbabilities_KMA(bmus, n_clusters, np_vars_chrom)

    # Calculate sigma spearman for each KMA - waves chromosome
    wvs_fams = ['sea', 'swell_1', 'swell_2']
    sigma = Correlation_Smooth_Partitions(
        bmus, cenEOFs, n_clusters, xds_WVS_MS, wvs_fams, xds_gev_params, chrom
    )



