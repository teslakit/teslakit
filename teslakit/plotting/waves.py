import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

def Waves_Distributions_HsTpDir(xds_wvs_pts):
    
    clusters=np.where(np.unique(xds_wvs_pts.bmus)>=0)[0]
    n_clusters=len(clusters)
    n_rows=int(np.sqrt(n_clusters+1))
    n_cols=n_rows
    
    fig = plt.figure(figsize=[20,12])
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.0, hspace=0.0)
    grid_row = 0
    grid_col = 0
    for ic in clusters:
        # data mean
        pos_cluster = np.where(xds_wvs_pts.bmus==ic)[0][:]
        ax = plt.subplot(gs[grid_row, grid_col])
        sns.distplot(xds_wvs_pts.sea_Hs[pos_cluster], bins=40, color='gold');
        sns.distplot(xds_wvs_pts.sea_Hs[pos_cluster], bins=40, color='gold', hist=False);
        sns.distplot(xds_wvs_pts.swell_1_Hs[pos_cluster], bins=40, color='darkgreen');
        sns.distplot(xds_wvs_pts.swell_1_Hs[pos_cluster], bins=40, color='darkgreen', hist=False);
        sns.distplot(xds_wvs_pts.swell_2_Hs[pos_cluster], bins=40, color='royalblue');
        sns.distplot(xds_wvs_pts.swell_2_Hs[pos_cluster], bins=40, color='royalblue', hist=False);
    
        if grid_row == n_rows-1:
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_xlabel('Hs',fontsize=14)    
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    
        grid_row += 1
        if grid_row >= n_rows:
            grid_row = 0
            grid_col += 1
    
    #Tp
    fig = plt.figure(figsize=[20,12])
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.0, hspace=0.0)
    grid_row = 0
    grid_col = 0
    for ic in clusters:
        # data mean
        pos_cluster = np.where(xds_wvs_pts.bmus==ic)[0][:]
        ax = plt.subplot(gs[grid_row, grid_col])
        sns.distplot(xds_wvs_pts.sea_Tp[pos_cluster], bins=40, color='gold');
        sns.distplot(xds_wvs_pts.sea_Tp[pos_cluster], bins=40, color='gold', hist=False);
        sns.distplot(xds_wvs_pts.swell_1_Tp[pos_cluster], bins=40, color='darkgreen');
        sns.distplot(xds_wvs_pts.swell_1_Tp[pos_cluster], bins=40, color='darkgreen', hist=False);
        sns.distplot(xds_wvs_pts.swell_2_Tp[pos_cluster], bins=40, color='royalblue');
        sns.distplot(xds_wvs_pts.swell_2_Tp[pos_cluster], bins=40, color='royalblue', hist=False);
    
        if grid_row == n_rows-1:
            ax.set_yticks([])
            #ax.set_xticks(np.arange(5,21, step=5))
            ax.set_xticks([])
            ax.set_xlabel('Tp',fontsize=14)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    
        grid_row += 1
        if grid_row >= n_rows:
            grid_row = 0
            grid_col += 1
    
    #dir    
    fig = plt.figure(figsize=[20,12])
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.0, hspace=0.0)
    grid_row = 0
    grid_col = 0
    for ic in clusters:
        # data mean
        pos_cluster = np.where(xds_wvs_pts.bmus==ic)[0][:]
        ax = plt.subplot(gs[grid_row, grid_col], projection='polar')
        plt.hist(np.deg2rad(xds_wvs_pts.sea_Dir[pos_cluster]),range=[0,np.deg2rad(360)],bins=50,color='gold',histtype='stepfilled', alpha=0.5)
        plt.hist(np.deg2rad(xds_wvs_pts.swell_1_Dir[pos_cluster]),range=[0,np.deg2rad(360)],bins=50,color='darkgreen',histtype='stepfilled', alpha=0.5)
        plt.hist(np.deg2rad(xds_wvs_pts.swell_2_Dir[pos_cluster]),range=[0,np.deg2rad(360)],bins=50,color='royalblue',histtype='stepfilled', alpha=0.5)
    
        ax.set_xticks([])
        ax.set_yticks([])
    
    grid_row += 1
    if grid_row >= n_rows:
        grid_row = 0
        grid_col += 1
