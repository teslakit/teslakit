#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
import time
import pickle
from itertools import permutations

# pip
import numpy as np
import xarray as xr
from scipy.special import ndtri  # norm inv
from scipy.stats import  genextreme, gumbel_l, spearmanr, norm
from statsmodels.distributions.empirical_distribution import ECDF
from numpy.random import choice, multivariate_normal, randint, rand

# fix tqdm for notebook 
from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

# tk
from .statistical import Empirical_ICDF
from .waves import AWL
from .extremes import FitGEV_KMA_Frechet, Smooth_GEV_Shape, ACOV
from .plotting.extremes import Plot_GEVParams, Plot_ChromosomesProbs, \
        Plot_SigmaCorrelation

# TODO: CREAR LOG
# TODO: introducir switch log on / log off para ejecuciones silenciosas

class Climate_Emulator(object):
    'KMA - DWTs Climate Emulator'

    def __init__(self, p_base):

        # max. Total Water level for each storm data
        self.KMA_MS = None
        self.WVS_MS = None

        # extremes model params
        self.GEV_Par = None         # GEV fitting parameters
        self.GEV_Par_S = None       # GEV simulation sampled parameters
        self.sigma = None           # Pearson sigma correlation

        # chromosomes
        self.chrom = None           # chromosomes and chromosomes probs

        # waves families and variables related aprameters
        self.fams = []              # waves families to use in emulator
        self.vars_GEV = []          # variables handled with GEV fit
        self.vars_EMP = []          # varibles handled with empirical fit

        # paths
        self.p_base      = p_base
        self.p_config    = op.join(p_base, 'config.pk')
        self.p_WVS_MS    = op.join(p_base, 'WVS_MaxStorm.nc')
        self.p_KMA_MS    = op.join(p_base, 'KMA_MaxStorm.nc')
        self.p_chrom     = op.join(p_base, 'chromosomes.nc')
        self.p_GEV_Par   = op.join(p_base, 'GEV_Parameters.nc')
        self.p_GEV_Sigma = op.join(p_base, 'GEV_SigmaCorrelation.nc')

        self.p_report_fit = op.join(p_base, 'report_fit')
        self.p_report_sim = op.join(p_base, 'report_sim')

        # output simulation storage paths
        self.p_sim           = op.join(p_base, 'Simulation')
        self.p_sim_wvs_notcs = op.join(self.p_sim, 'WAVES_noTCs')
        self.p_sim_wvs_tcs   = op.join(self.p_sim, 'WAVES_TCs')
        self.p_sim_tcs       = op.join(self.p_sim, 'TCs')  # simulated TCs

    def ConfigVariables(self, config):
        '''
        Set wich waves families variables will be handled with GEV or EMP

        config = {
            'name_fams': ['sea', 'swell_1', ...]
            'force_empirical': ['swell_2_Tp', 'swell_3_Hs', ...]
        }
        '''

        # get data from config dict
        fams = config['name_fams']  # waves families to use
        force_empircal = config['force_empirical']

        # Hs, Tp, Dir separated by GEV / EMPIRICAL
        GEV_vn = ['Hs', 'Tp']
        EMP_vn = ['Dir']

        # mount family_variable lists
        l_GEV_vars = []
        l_EMP_vars = []
        for f in fams:
            for v in GEV_vn:
                l_GEV_vars.append('{0}_{1}'.format(f,v))
            for v in EMP_vn:
                l_EMP_vars.append('{0}_{1}'.format(f,v))

        # force empirical
        for vf in force_empircal:
            if vf in l_GEV_vars:
                l_GEV_vars.pop(l_GEV_vars.index(vf))
            if vf not in l_EMP_vars:
                l_EMP_vars.append(vf)

        # set properties
        self.fams = fams
        self.vars_GEV = l_GEV_vars
        self.vars_EMP = l_EMP_vars

    def FitExtremes(self, xds_KMA, xds_WVS_parts, xds_WVS_fams, config):
        '''
        GEV extremes fitting.
        Input data (waves vars series and bmus) shares time dimension

        xds_KMA        - xarray.Dataset, vars: bmus (time,), cenEOFs(n_clusters,n_features)
        xds_WVS_parts  - xarray.Dataset: (time,), phs, pspr, pwfrac... {0-5 partitions}
        xds_WVS_fams   - xarray.Dataset: (time,), fam_V, {fam: sea,swell_1,swell2. V: Hs,Tp,Dir}
        config         - dictionary: name_fams, force_empirical
        '''

        # configure waves fams variables parameters from config dict
        self.ConfigVariables(config)

        # get start and end dates for each storm
        lt_storm_dates = self.Calc_StormsDates(xds_KMA)

        # calculate max. TWL for each storm
        xds_max_TWL = self.Calc_StormsMaxTWL(xds_WVS_parts, lt_storm_dates)

        # select WVS_families data at storms max. TWL 
        xds_WVS_MS = xds_WVS_fams.sel(time = xds_max_TWL.time)
        xds_WVS_MS['max_TWL'] = ('time', xds_max_TWL.TWL.values[:])

        # select KMA data at storms max. TWL 
        xds_KMA_MS = xds_KMA.sel(time = xds_max_TWL.time)

        # calculate chromosomes and probabilities
        xds_chrom = self.Calc_Chromosomes(xds_KMA_MS, xds_WVS_MS)

        # GEV: Fit each wave family to a GEV distribution (KMA bmus)
        xds_GEV_Par = self.Calc_GEVParams(xds_KMA_MS, xds_WVS_MS)

        # Calculate sigma spearman for each KMA - fams chromosome
        d_sigma = self.Calc_SigmaCorrelation(
            xds_KMA_MS, xds_WVS_MS, xds_GEV_Par
        )

        # store data
        self.WVS_MS = xds_WVS_MS
        self.KMA_MS = xds_KMA_MS
        self.GEV_Par = xds_GEV_Par
        self.chrom = xds_chrom
        self.sigma = d_sigma
        self.Save()

    def Save(self):
        'Saves fitted climate emulator data'

        if not op.isdir(self.p_base): os.makedirs(self.p_base)

        # store .nc files    
        self.WVS_MS.to_netcdf(self.p_WVS_MS)
        self.KMA_MS.to_netcdf(self.p_KMA_MS)
        self.chrom.to_netcdf(self.p_chrom)
        self.GEV_Par.to_netcdf(self.p_GEV_Par)

        # store pickle
        pickle.dump(self.sigma, open(self.p_GEV_Sigma, 'wb'))

        # store config
        pickle.dump(
            (self.fams, self.vars_GEV, self.vars_EMP),
            open(self.p_config, 'wb')
        )

    def Load(self):
        'Loads fitted climate emulator data'

        # store .nc files    
        self.WVS_MS = xr.open_dataset(self.p_WVS_MS)
        self.KMA_MS = xr.open_dataset(self.p_KMA_MS)
        self.chrom = xr.open_dataset(self.p_chrom)
        self.GEV_Par = xr.open_dataset(self.p_GEV_Par)

        # store pickle
        self.sigma = pickle.load(open(self.p_GEV_Sigma, 'rb'))

        # load config
        self.fams, self.vars_GEV, self.vars_EMP = pickle.load(
            open(self.p_config, 'rb')
        )

    def StoreSim(self, p_store, ls_xdsets, code):
        'Store waves and TCs simulations (list of xr.Dataset) at p_store folder'

        if not op.isdir(p_store): os.makedirs(p_store)

        for c, xds in enumerate(ls_xdsets):
            p_nc = op.join(p_store, '{0}{1:02}.nc'.format(code, c+1))
            xds.to_netcdf(p_nc, 'w')

    def LoadSim(self, TCs=False):
        'Load waves and TCs simulations'

        def lsncs(p):
            fs = sorted(
                [op.join(p,f) for f in os.listdir(p) if f.endswith('.nc')]
                )
            xrs = [xr.open_dataset(f) for f in fs]

            return xrs

        if TCs:
            return lsncs(self.p_sim_wvs_tcs), lsncs(self.p_sim_tcs)

        else:
            return lsncs(self.p_sim_wvs_notcs)

    def Calc_StormsDates(self, xds_KMA):
        'Returns list of tuples with each storm start and end times'

        # locate dates where KMA WT changes (bmus series)
        bmus_diff = np.diff(xds_KMA.bmus.values)
        ix_ch = np.where((bmus_diff != 0))[0]+1
        ix_ch = np.insert(ix_ch, 0,0)
        ds_ch = xds_KMA.time.values[ix_ch]  # dates where WT changes

        # list of tuples with (date start, date end) for each storm (WT window)
        dates_tup_WT = [(ds_ch[c], ds_ch[c+1]-np.timedelta64(1,'D')) for c in range(len(ds_ch)-1)]
        dates_tup_WT.append((dates_tup_WT[-1][1]+np.timedelta64(1,'D'), xds_KMA.time.values[-1]))

        return dates_tup_WT

    def Calc_StormsMaxTWL(self, xds_WVS_pts, lt_storm_dates):
        'Returns xarray.Dataset with max. TWL value and time'

        # Get TWL from waves partitions data 
        xda_TWL = AWL(xds_WVS_pts.hs, xds_WVS_pts.tp)

        # find max TWL inside each storm 
        TWL_WT_max = []
        times_WT_max = []
        for d1, d2 in lt_storm_dates:

            # get TWL inside WT window
            wt_TWL = xda_TWL.sel(time=slice(d1,d2))[:]

            # get window maximum TWL date
            wt_max_TWL = wt_TWL.where(wt_TWL==wt_TWL.max(), drop=True).squeeze()
            max_TWL = wt_max_TWL.values
            max_date = wt_max_TWL.time.values

            # append data
            TWL_WT_max.append(max_TWL)
            times_WT_max.append(max_date)

        return xr.Dataset(
            {
                'TWL': (('time',), TWL_WT_max),
            },
            coords = {'time': times_WT_max}
        )

    def Calc_GEVParams(self, xds_KMA_MS, xds_WVS_MS):
        '''
        Fits each WT (KMA.bmus) waves families data to a GEV distribtion
        Requires KMA and WVS families at storms max. TWL

        Returns xarray.Dataset with GEV shape, location and scale parameters
        '''

        vars_gev = self.vars_GEV
        bmus = xds_KMA_MS.bmus.values[:]
        cenEOFs = xds_KMA_MS.cenEOFs.values[:]
        n_clusters = len(xds_KMA_MS.n_clusters)

        xds_GEV_Par = xr.Dataset(
            coords = {
                'n_cluster' : np.arange(n_clusters)+1,
                'parameter' : ['shape', 'location', 'scale'],
            }
        )

        # Fit each wave family var to GEV distribution (using KMA bmus)
        for vn in vars_gev:
            gp_pars = FitGEV_KMA_Frechet(
                bmus, n_clusters, xds_WVS_MS[vn].values[:])
            xds_GEV_Par[vn] = (('n_cluster', 'parameter',), gp_pars)

        return xds_GEV_Par

    def Calc_Chromosomes(self, xds_KMA_MS, xds_WVS_MS):
        '''
        Calculate chromosomes and probabilities from KMA.bmus data

        Returns xarray.Dataset vars: chrom, chrom_probs. dims: WT, wave_family
        '''

        bmus = xds_KMA_MS.bmus.values[:]
        n_clusters = len(xds_KMA_MS.n_clusters)
        fams_chrom = self.fams
        l_vc = [ '{0}_Hs'.format(x) for x in fams_chrom]

        # get chromosomes matrix
        np_vc = np.column_stack([xds_WVS_MS[vn].values for vn in l_vc])
        chrom = ChromMatrix(np_vc)

        # calculate chromosomes probabilities
        probs = np.zeros((n_clusters, chrom.shape[0]))
        for i in range(n_clusters):
            c = i+1
            pos = np.where((bmus==c))[0]

            # get variables chromosomes at cluster
            var_c = np_vc[pos,:]
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

        # chromosomes dataset
        return xr.Dataset(
            {
                'chrom': (('n','wave_family',), chrom),
                'probs': (('WT','n',), probs),
            },
            coords={
                'WT': np.arange(n_clusters)+1,
                'wave_family': fams_chrom,
            }
        )

    def norm_GEV_or_EMP(self, vn, vv, xds_GEV_Par, d_shape, i_wt):
        '''
        Switch function
        Check climate emulator config and calculate NORM for the variable (GEV / EMPIRICAL)

        vn - var name
        vv - var value
        i_wt - Weather Type index
        xds_GEV_Par , d_shape: GEV data used in sigma correlation
        '''

        # get GEV and EMPIRICAL variables list
        vars_GEV = self.vars_GEV
        vars_EMP = self.vars_EMP

        # switch variable name
        if vn in vars_GEV:

            # gev CDF
            sha_g = d_shape[vn][i_wt]
            loc_g = xds_GEV_Par.sel(parameter='location')[vn].values[i_wt]
            sca_g = xds_GEV_Par.sel(parameter='scale')[vn].values[i_wt]
            norm_VV = genextreme.cdf(vv, -1*sha_g, loc_g, sca_g)

        elif vn in vars_EMP:

            # empirical CDF
            ecdf = ECDF(vv)
            norm_VV = ecdf(vv)

        return norm_VV

    def Calc_SigmaCorrelation(self, xds_KMA_MS, xds_WVS_MS, xds_GEV_Par):
        'Calculate Sigma Pearson correlation for each WT-chromosome combo'

        bmus = xds_KMA_MS.bmus.values[:]
        cenEOFs = xds_KMA_MS.cenEOFs.values[:]
        n_clusters = len(xds_KMA_MS.n_clusters)
        wvs_fams = self.fams
        vars_GEV = self.vars_GEV

        # smooth GEV shape parameter 
        d_shape = {}
        for vn in vars_GEV:
            sh_GEV = xds_GEV_Par.sel(parameter='shape')[vn].values[:]
            d_shape[vn] = Smooth_GEV_Shape(cenEOFs, sh_GEV)

        # Get sigma correlation for each KMA cluster 
        d_sigma = {}  # nested dict [WT][crom]
        for iwt in range(n_clusters):
            c = iwt+1
            pos = np.where((bmus==c))[0]
            d_sigma[c] = {}

            # current cluster waves
            xds_K_wvs = xds_WVS_MS.isel(time=pos)

            # get chromosomes from waves (0/1)
            var_c = np.column_stack(
                [xds_K_wvs['{0}_Hs'.format(x)].values[:] for x in wvs_fams]
            )
            var_c[~np.isnan(var_c)] = 1
            var_c[np.isnan(var_c)] = 0
            chrom = ChromMatrix(var_c)

            # get sigma for each chromosome
            for ucix, uc in enumerate(chrom):
                wt_crom = 1  # data / no data 

                # find data position for this chromosome
                p_c = np.where((var_c == uc).all(axis=1))[0]

                # if not enought data, get all chromosomes with shared 1s
                if len(p_c) < 20:
                    p1s = np.where(uc==1)[0]
                    p_c = np.where((var_c[:,p1s] == uc[p1s]).all(axis=1))[0]

                    wt_crom = 0  # data / no data 

                # select waves chrom data 
                xds_chr_wvs = xds_K_wvs.isel(time=p_c)

                # solve normal inverse GEV/EMP CDF for each active chromosome
                to_corr = np.empty((0,len(p_c)))  # append for spearman correlation
                for i_c in np.where(uc==1)[0]:

                    # get wave family chromosome variables
                    fam_n = wvs_fams[i_c]
                    vn_Hs = '{0}_Hs'.format(fam_n)
                    vn_Tp = '{0}_Tp'.format(fam_n)
                    vn_Dir = '{0}_Dir'.format(fam_n)

                    vv_Hs = xds_chr_wvs[vn_Hs].values[:]
                    vv_Tp = xds_chr_wvs[vn_Tp].values[:]
                    vv_Dir = xds_chr_wvs[vn_Dir].values[:]

                    # Hs 
                    norm_Hs = self.norm_GEV_or_EMP(
                        vn_Hs, vv_Hs, xds_GEV_Par, d_shape, iwt)

                    # Tp 
                    norm_Tp = self.norm_GEV_or_EMP(
                        vn_Tp, vv_Tp, xds_GEV_Par, d_shape, iwt)

                    # Dir 
                    norm_Dir = self.norm_GEV_or_EMP(
                        vn_Dir, vv_Dir, xds_GEV_Par, d_shape, iwt)

                    # normal inverse CDF 
                    u_cdf = np.column_stack([norm_Hs, norm_Tp, norm_Dir])
                    u_cdf[u_cdf>=1.0] = 0.999999
                    inv_n = ndtri(u_cdf)

                    # concatenate data for correlation
                    to_corr = np.concatenate((to_corr, inv_n.T), axis=0)

                # sigma: spearman correlation
                corr, pval = spearmanr(to_corr, axis=1)

                # store data at dict
                d_sigma[c][ucix] = {
                    'corr': corr, 'data': len(p_c), 'wt_crom': wt_crom
                }

        return d_sigma

    def GEV_Parameters_Sampling(self, n_sims):
        '''
        Sample new GEV/GUMBELL parameters using GEV/GUMBELL asymptotic variances

        num_sims  - number of GEV parameters to sample
        '''

        xds_GEV_Par = self.GEV_Par
        vars_gev = self.vars_GEV
        xds_KMA_MS = self.KMA_MS
        xds_WVS_MS = self.WVS_MS

        # get KMA data
        bmus = xds_KMA_MS.bmus.values[:]
        n_clusters = len(xds_KMA_MS.n_clusters)
        cenEOFs = xds_KMA_MS.cenEOFs.values[:]

        # dataset for storing parameters
        xds_par_samp = xr.Dataset(
            {
            },
            coords={
                'parameter' : ['shape', 'location', 'scale'],
                'n_cluster' : np.arange(n_clusters)+1,
                'simulation': range(n_sims),
            },
        )

        # simulate variables
        for vn in vars_gev:

            # GEV/GUMBELL parameters
            pars_GEV = xds_GEV_Par[vn]
            sha = pars_GEV.sel(parameter='shape').values[:]
            sca = pars_GEV.sel(parameter='scale').values[:]
            loc = pars_GEV.sel(parameter='location').values[:]

            # location parameter Extremal Index (Gev) 
            index = np.ones(sha.shape)
            mu_b = loc - (sca/sha) * (1-np.power(index, sha))
            psi_b = sca * np.power(index, sha)

            # location parameter Extremal Index (Gumbell) 
            sha_gbl = 0.0000000001
            pos_gbl = np.where(sha == sha_gbl)[0]
            cls_gbl = pos_gbl + 1  # Gumbell Weather Types

            # update mu_b
            mu_b[pos_gbl] = -loc[pos_gbl] + sca[pos_gbl] * np.log(index[pos_gbl])

            # output holder 
            out_ps = np.ndarray((n_clusters, n_sims, 3)) * np.nan

            # sample Gumbel or GEV parameters for each WT 
            for i in range(n_clusters):
                c = i+1  # WT ID

                # get var values at cluster and remove nans
                p_bmus = np.where((bmus==c))[0]
                var_wvs = xds_WVS_MS[vn].isel(time=p_bmus).values[:]
                var_wvs = var_wvs[~np.isnan(var_wvs)]

                # Gumbel WTs: parameters sampling
                if c in cls_gbl:

                    # GUMBELL Loglikelihood function acov
                    theta = (loc[i], sca[i])
                    # TODO comprobar gumbel_l acov y evlike acov son similares
                    acov = ACOV(gumbel_l.nnlf, theta, var_wvs)

                    # GUMBELL params used for multivar. normal random generation
                    theta_gen = np.array([mu_b[i], sca[i]])
                    theta_gbl = multivariate_normal(theta_gen, acov, n_sims)

                    # mount "GEV" params for simulation
                    theta_sim = np.ones((n_sims,3))*sha_gbl
                    theta_sim[:,1:] = theta_gbl

                # GEV WTs: parameters sampling
                else:
                    # TODO: signo acov correcto?
                    # TODO: PARECE QUE ACOV NO FUNCIONA SIEMPRE, NLOGL inf

                    # GEV Loglikelihood function acov
                    theta = (sha[i], loc[i], sca[i])
                    acov = ACOV(genextreme.nnlf, theta, var_wvs)

                    # GEV params used for multivar. normal random generation
                    theta_gen = np.array([sha[i], mu_b[i], psi_b[i]])
                    theta_sim = multivariate_normal(theta_gen, acov, n_sims)

                # store sampled GEV/GUMBELL params
                out_ps[i,:,:] = theta_sim[:,:]

            # smooth shape parameter
            for j in range(n_sims):
                shape_wts = out_ps[:,j,0]
                out_ps[:,j,0] = Smooth_GEV_Shape(cenEOFs, shape_wts)

            # append output to dataset
            xds_par_samp[vn] = (('n_cluster','simulation', 'parameter'), out_ps)

        return xds_par_samp

    def Simulate_Waves(self, xds_DWT, dict_WT_TCs_wvs):
        '''
        Climate Emulator DWTs waves simulation

        xds_DWT          - xarray.Dataset, vars: evbmus_sims (time, n_sim,)
        dict_WT_TCs_wvs  - dict of xarray.Dataset (waves data) for TCs WTs
        '''

        # max. storm waves and KMA
        xds_KMA_MS = self.KMA_MS
        xds_WVS_MS = self.WVS_MS
        xds_chrom = self.chrom
        xds_GEV_Par = self.GEV_Par
        sigma = self.sigma

        # vars needed
        dwt_bmus_sim = xds_DWT.evbmus_sims.values[:]
        dwt_time_sim = xds_DWT.time.values[:]
        bmus = xds_KMA_MS.bmus.values[:]
        n_clusters = len(xds_KMA_MS.n_clusters)
        chrom = xds_chrom.chrom.values[:]
        chrom_probs = xds_chrom.probs.values[:]

        # iterate DWT simulations
        ls_wvs_sim = []
        for dwt in dwt_bmus_sim.T:

            # get number of simulations
            idw, iuc = np.unique(dwt, return_counts=True)
            num_sims = np.max(iuc)

            # Sample GEV/GUMBELL parameters 
            xds_GEV_Par_Sampled = self.GEV_Parameters_Sampling(num_sims)

            # generate waves
            wvs_sim = self.GenerateWaves(
                bmus, n_clusters, chrom, chrom_probs, sigma, xds_WVS_MS,
                xds_GEV_Par_Sampled, dict_WT_TCs_wvs, dwt, dwt_time_sim
            )
            ls_wvs_sim.append(wvs_sim)

        # store simulations
        self.StoreSim(self.p_sim_wvs_notcs, ls_wvs_sim, 'wvs_sim_noTCs_')

        return ls_wvs_sim

    def Simulate_TCs(self, xds_DWT, dict_WT_TCs_wvs, xds_TCs_params,
                     xds_TCs_simulation, prob_change_TCs, MU_WT, TAU_WT):
        '''
        Climate Emulator DWTs TCs simulation

        xds_DWT             - xarray.Dataset, vars: evbmus_sims (time,n_sim,)
        dict_WT_TCs_wvs     - dict of xarray.Dataset (waves data) for TCs WTs

        xds_TCs_params      - xr.Dataset. vars(storm): pressure_min
        xds_TCs_simulation  - xr.Dataset. vars(storm): mu, hs, ss, tp, dir
        prob_change_TCs     - cumulative probabilities of TC category change
        MU_WT, TAU_WT       - intradaily hidrographs for each WT
        '''

        # max. storm waves and KMA
        xds_KMA_MS = self.KMA_MS

        # vars needed
        dwt_bmus_sim = xds_DWT.evbmus_sims.values[:]
        dwt_time_sim = xds_DWT.time.values[:]
        n_clusters = len(xds_KMA_MS.n_clusters)

        # iterate DWT simulations
        ls_tcs_sim = []
        ls_wvs_upd = []
        for n_sim, dwt in enumerate(dwt_bmus_sim.T):

            # load waves simulation, will be modified
            p_wvs_nc = op.join(self.p_sim_wvs_notcs, 'wvs_sim_noTCs_{0:02}.nc'.format(n_sim+1))
            wvs_sim = xr.open_dataset(p_wvs_nc)

            # generate TCs
            tcs_sim, wvs_upd_sim = self.GenerateTCs(
                n_clusters, dwt, dwt_time_sim,
                xds_TCs_params, xds_TCs_simulation, prob_change_TCs, MU_WT, TAU_WT,
                wvs_sim
            )
            ls_tcs_sim.append(tcs_sim)
            ls_wvs_upd.append(wvs_upd_sim)

        # store simulations
        self.StoreSim(self.p_sim_tcs, ls_tcs_sim, 'TCs_sim_')
        self.StoreSim(self.p_sim_wvs_tcs, ls_wvs_upd, 'wvs_sim_TCs_')

        return ls_tcs_sim, ls_wvs_upd

    def icdf_GEV_or_EMP(self, vn, vv, pb, xds_GEV_Par, i_wt):
        '''
        Switch function
        Check climate emulator config and calculate ICDF for the variable (GEV / EMPIRICAL)

        vn - var name
        vv - var value
        pb - var simulation probs
        i_wt - Weather Type index
        xds_GEV_Par: GEV parameters
        '''

        # get GEV and EMPIRICAL variables list
        vars_GEV = self.vars_GEV
        vars_EMP = self.vars_EMP

        # switch variable name
        if vn in vars_GEV and not vn =='sea_Tp':

            # gev ICDF
            sha_g = xds_GEV_Par.sel(parameter='shape')[vn].values[i_wt]
            loc_g = xds_GEV_Par.sel(parameter='location')[vn].values[i_wt]
            sca_g = xds_GEV_Par.sel(parameter='scale')[vn].values[i_wt]
            ppf_VV = genextreme.ppf(pb, -1*sha_g, loc_g, sca_g)

        elif vn in vars_EMP:

            # empirical ICDF
            ppf_VV = Empirical_ICDF(vv, pb)

        return ppf_VV

    def GenerateWaves(self, bmus, n_clusters, chrom, chrom_probs, sigma,
                      xds_WVS_MS, xds_GEV_Par_Sampled, TC_WVS, DWT, DWT_time):
        '''
        Climate Emulator DWTs waves simulation

        bmus                 - KMA max. storms bmus series
        n_clusters           - KMA number of clusters
        chrom, chrom_probs   - chromosomes and probabilities
        sigma                - pearson correlation for each WT
        xds_GEV_Par_Sampled  - GEV/GUMBELL parameters sampled for simulation
        TC_WVS               - dictionary. keys: WT, vals: xarray.Dataset TCs waves fams
        DWT                  - np.array with DWT bmus sim series (dims: time,)

        Returns xarray.Dataset with simulated storm data
            vars:
                *fam*_*vn* (fam: sea, swell_1, swell_2, vn: Hs, Tp, Dir),
                DWT_sim
            dims: storm
        '''

        # filter parameters
        hs_min, hs_max = 0, 15
        tp_min, tp_max = 2, 25
        ws_min, ws_max = 0.001, 0.06

        # waves families - variables (sorted for simulation output)
        wvs_fams = self.fams
        wvs_fams_vars = [
            ('{0}_{1}'.format(f,vn)) for f in wvs_fams for vn in['Hs', 'Tp','Dir']
            ]

        # simulate one value for each storm 
        dwt_df = np.diff(DWT)
        ix_ch = np.where((dwt_df != 0))[0]+1
        ix_ch = np.insert(ix_ch, 0,0)
        DWT_sim = DWT[ix_ch]
        DWT_time_sim = DWT_time[ix_ch]

        # new progress bar 
        pbar = tqdm(
            total=len(DWT_sim),
            desc = 'C.E: Sim. Waves'
        )

        # Simulate
        sims_out = np.zeros((len(DWT_sim), 9))
        c = 0
        while c < len(DWT_sim):
            WT = DWT_sim[c]
            iwt = WT - 1

            # KMA Weather Types waves generation
            if WT <= n_clusters:

                # get random chromosome (weigthed choice)
                pr = chrom_probs[iwt] / np.sum(chrom_probs[iwt])
                ci = choice(range(chrom.shape[0]), 1, p=pr)
                crm = chrom[ci].astype(int).squeeze()


                # get sigma correlation for this WT - crm combination 
                corr = sigma[WT][int(ci)]['corr']
                mvn_m = np.zeros(corr.shape[0])
                sims = multivariate_normal(mvn_m, corr)
                prob_sim = norm.cdf(sims, 0, 1)

                # solve normal inverse CDF for each active chromosome
                ipbs = 0  # prob_sim aux. index
                sim_row = np.zeros(9)
                for i_c in np.where(crm == 1)[0]:

                    # random sampled GEV 
                    # TODO hacerlo excluyente, no repetir opciones 
                    rd = np.random.randint(0,len(xds_GEV_Par_Sampled.simulation))
                    xds_GEV_Par = xds_GEV_Par_Sampled.isel(simulation=rd)

                    # get wave family chromosome variables
                    fam_n = wvs_fams[i_c]
                    vn_Hs = '{0}_Hs'.format(fam_n)
                    vn_Tp = '{0}_Tp'.format(fam_n)
                    vn_Dir = '{0}_Dir'.format(fam_n)

                    vv_Hs = xds_WVS_MS[vn_Hs].values[:]
                    vv_Tp = xds_WVS_MS[vn_Tp].values[:]
                    vv_Dir = xds_WVS_MS[vn_Dir].values[:]

                    pb_Hs = prob_sim[ipbs+0]
                    pb_Tp = prob_sim[ipbs+1]
                    pb_Dir = prob_sim[ipbs+2]
                    ipbs +=3

                    # Hs
                    ppf_Hs = self.icdf_GEV_or_EMP(
                        vn_Hs, vv_Hs, pb_Hs, xds_GEV_Par, iwt)

                    # Tp
                    ppf_Tp = self.icdf_GEV_or_EMP(
                        vn_Tp, vv_Tp, pb_Tp, xds_GEV_Par, iwt)

                    # Dir
                    ppf_Dir = self.icdf_GEV_or_EMP(
                        vn_Dir, vv_Dir, pb_Dir, xds_GEV_Par, iwt)


                    # store simulation data
                    is0,is1 = wvs_fams.index(fam_n)*3, (wvs_fams.index(fam_n)+1)*3
                    sim_row[is0:is1] = [ppf_Hs, ppf_Tp, ppf_Dir]

            # TCs Weather Types waves generation
            else:

                # Get TC-WT waves fams data 
                tws = TC_WVS['{0}'.format(WT)]

                # select random state
                ri = randint(len(tws.time))

                # generate sim_row with sorted waves families variables
                sim_row = np.stack([tws[vn].values[ri] for vn in wvs_fams_vars])

            # Filters

            # all 0 chromosomes
            if all(c == 0 for c in crm):
                continue

            # nan / negative values
            if np.isnan(sim_row).any() or len(np.where(sim_row<0)[0])!=0:
                continue

            # Hs and Tp
            hs_s = sim_row[0::3][crm==1]
            tp_s = sim_row[1::3][crm==1]
            if any(v <= hs_min for v in hs_s) or any(v >= hs_max for v in hs_s) \
               or any(v <= tp_min for v in tp_s) or any(v >= tp_max for v in tp_s):
                continue

            # wave stepness 
            ws_s = hs_s / (1.56 * tp_s**2 )
            if any(v <= ws_min for v in ws_s) or any(v >= ws_max for v in ws_s):
                continue

            # store simulation
            sim_row[sim_row==0] = np.nan  # nan data at crom 0 
            sims_out[c] = sim_row
            c+=1

            # progress bar
            pbar.update(1)

        pbar.close()

        # dataset for storing output
        xds_wvs_sim = xr.Dataset(
            {
                'DWT': (('time',), DWT_sim),
            },
            coords = {'time': DWT_time_sim}
        )
        for c,vn in enumerate(wvs_fams_vars):
            xds_wvs_sim[vn] = (('time',), sims_out[:,c])

        return xds_wvs_sim

    def GenerateTCs(self, n_clusters, DWT, DWT_time,
                    TCs_params, TCs_simulation, prob_TCs, MU_WT, TAU_WT,
                    xds_wvs_sim):
        '''
        Climate Emulator DWTs TCs simulation

        n_clusters      - KMA number of clusters
        DWT             - np.array with DWT bmus sim series (dims: time,)

        TCs_params      - xr.Dataset. vars(storm): pressure_min
        TCs_simulation  - xr.Dataset. vars(storm): mu, hs, ss, tp, dir
        prob_TCs        - cumulative probabilities of TC category change
        MU_WT, TAU_WT   - intradaily hidrographs for each WT
        xds_wvs_sim     - xr.Dataset, waves simulated without TCs (for updating)

        returns xarray.Datasets with updated Waves and simulated TCs data
            vars waves:
                *fam*_*vn* (fam: sea, swell_1, swell_2 ..., vn: Hs, Tp, Dir),
            vars TCS:
                mu, tau, ss
            dims: storm
        '''

        # wave family to modify
        mod_fam = 'sea'  # TODO input parameter

        # waves families - variables (sorted for simulation output)
        wvs_fams = self.fams
        wvs_fams_vars = [
            ('{0}_{1}'.format(f,vn)) for f in wvs_fams for vn in['Hs', 'Tp','Dir']
            ]

        # simulate one value for each storm 
        dwt_df = np.diff(DWT)
        ix_ch = np.where((dwt_df != 0))[0]+1
        ix_ch = np.insert(ix_ch, 0,0)
        DWT_sim = DWT[ix_ch]
        DWT_time_sim = DWT_time[ix_ch]

        # get simulated waves for updating
        sim_wvs = np.column_stack([
            xds_wvs_sim[vn].values[:] for vn in wvs_fams_vars
        ])

        # new progress bar 
        pbar = tqdm(
            total=len(DWT_sim),
            desc = 'C.E: Sim. TCs  '
        )

        # Simulate TCs (mu, ss, tau)
        sims_out = np.zeros((len(DWT_sim), 3))
        c = 0
        while c < len(DWT_sim):
            WT = DWT_sim[c]
            iwt = WT - 1

            # KMA Weather Types tcs generation
            if WT <= n_clusters:

                # get random MU,TAU from current WT
                # TODO: random excluyente?
                ri = randint(len(MU_WT[iwt]))
                mu_s = MU_WT[iwt][ri]
                tau_s = TAU_WT[iwt][ri]
                ss_s = 0

            # TCs Weather Types waves generation
            else:
                # get probability of category change for this WT
                prob_t = np.append(prob_TCs[:, iwt-n_clusters], 1)

                ri = rand()
                si = np.where(prob_t >= ri)[0][0]

                if si == len(prob_t)-1:
                    # TC does not enter. random mu_s, 0.5 tau_s, 0 ss_s
                    all_MUs = np.concatenate(MU_WT)
                    ri = randint(len(all_MUs))
                    # TODO: check mu 0s, set nans (?)

                    mu_s = all_MUs[ri]
                    tau_s = 0.5
                    ss_s = 0

                else:
                    s_pmin = TCs_params.pressure_min.values[:]
                    p1, p2 = {
                        0:(1000, np.nanmax(s_pmin)+1),
                        1:(979, 1000),
                        2:(964, 979),
                        3:(944, 964),
                        4:(920, 944),
                        5:(np.nanmin(s_pmin)-1, 920),
                    }[si]

                    psi = np.where((s_pmin > p1) & (s_pmin <= p2))[0]

                    if psi.any():
                        ri = randint(len(psi))
                        mu_s = TCs_simulation.mu.values[ri]
                        ss_s = TCs_simulation.ss.values[ri]
                        tau_s = 0.5

                        # Get waves family data from simulated TCs (numerical+rbf)
                        mod_fam_Hs = TCs_simulation.hs.values[ri]
                        mod_fam_Tp = TCs_simulation.tp.values[ri]
                        mod_fam_Dir = TCs_simulation.dir.values[ri]

                        # locate index of wave family to modify
                        ixu = wvs_fams.index(mod_fam) * 3

                        # replace waves simulation value
                        sim_wvs[c, :] = sim_wvs[c,:] * 0
                        sim_wvs[c, ixu:ixu+3] = [
                            mod_fam_Hs, mod_fam_Tp, mod_fam_Dir]

                    else:
                        # TODO: no deberia caer aqui. comentar duda
                        mu_s = 0
                        ss_s = 0
                        tau_s = 0

            sim_row = np.array([mu_s, tau_s, ss_s])

            # no nans or values < 0 stored 
            if ~np.isnan(sim_row).any() and len(np.where(sim_row<0)[0])==0:
                sims_out[c] = sim_row
                c+=1

                # progress bar
                pbar.update(1)

        pbar.close()

        # update waves simulation
        xds_WVS_sim_updated = xr.Dataset(
            {
                'DWT': (('time',), DWT_sim),
            },
            coords = {'time': DWT_time_sim}
        )
        for c, vn in enumerate(wvs_fams_vars):
            xds_WVS_sim_updated[vn] = (('time',), sim_wvs[:,c])

        # generated TCs 
        xds_TCs_sim = xr.Dataset(
            {
                'mu':  (('time',), sims_out[:,0]),
                'tau': (('time',), sims_out[:,1]),
                'ss':  (('time',), sims_out[:,2]),

                'DWT': (('time',), DWT_sim),
            },
            coords = {'time': DWT_time_sim}
        )

        return xds_TCs_sim, xds_WVS_sim_updated


    def Report_Fit(self, export=False):
        'Report for extremes model fitting'

        # get data (fams hs)
        vars_gev_params = [x for x in self.vars_GEV if 'Hs' in x]
        xds_GEV_Par = self.GEV_Par
        xds_chrom = self.chrom
        d_sigma = self.sigma

        if export:

            # report folder
            p_save = self.p_report_fit
            if not op.isdir(p_save):
                os.mkdir(p_save)

            # Plot GEV params for each WT
            for gvn in vars_gev_params:
                p_plot = op.join(p_save, 'GEV_params_{0}.png'.format(gvn))
                Plot_GEVParams(xds_GEV_Par[gvn], p_export=p_plot)

            # Plot cromosomes probabilities
            p_plot = op.join(p_save, 'chromosomes_probs.png')
            Plot_ChromosomesProbs(xds_chrom, p_export=p_plot)

            # Plot sigma correlation triangle
            p_plot = op.join(p_save, 'sigma_correlation.png')
            Plot_SigmaCorrelation(xds_chrom, d_sigma, p_export=p_plot)

        else:

            # Plot GEV params for each WT
            for gvn in vars_gev_params:
                Plot_GEVParams(xds_GEV_Par[gvn])

            # Plot cromosomes probabilities
            Plot_ChromosomesProbs(xds_chrom)

            # Plot sigma correlation triangle
            Plot_SigmaCorrelation(xds_chrom, d_sigma)


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

