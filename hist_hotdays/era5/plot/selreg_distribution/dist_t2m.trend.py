import os
import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from tqdm import tqdm

varn='t2m'
lre=['sea']
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','son','jja'] # season (ann, djf, mam, jja, son)

for re in lre:
    for se in lse:
        idir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)
        odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/era5/%s/%s' % (se,varn)

        if not os.path.exists(odir):
            os.makedirs(odir)

        c = 0
        # Load climatology
        [ht2m,lpc]=pickle.load(open('%s/clmean.%s.%s.1950.2020.%s.pickle' % (idir,varn,re,se), 'rb'))
        clt2m={}
        for pc in [5,50,95]:
            clt2m[str(pc)]=ht2m[np.where(np.equal(lpc,pc))[0][0]]

        # KDE
        [stats,kbm] = pickle.load(open('%s/regress.kde.%s.%s.%s.pickle' % (idir,varn,re,se), 'rb'))
        dect=10*stats['slope'] # convert to decadal trend
        sgft=np.where(stats['pvalue']<0.05) # find bins of significant trend
        fig,ax=plt.subplots(figsize=(4,3))
        ax.axhline(0,linewidth=0.5,color='k')
        ax.axvline(clt2m['5'],linewidth=0.5,color='tab:blue')
        ax.axvline(clt2m['50'],linewidth=0.5,color='k')
        ax.axvline(clt2m['95'],linewidth=0.5,color='tab:orange')
        ax.plot(kbm[sgft],dect[sgft],'.',color='tab:red',alpha=0.5)
        ax.plot(kbm,dect,'k')
        # ax.set_xlim([285,310])
        ax.set_xlabel(r'$T_{2\,m}$ (K)')
        ax.set_ylabel(r'Probability Density Trend (K$^{-1}$ dec$^{-1}$)')
        ax.set_title(r'%s ERA5 %s' % (se.upper(),re.upper()))
        fig.tight_layout()
        plt.savefig('%s/kde.%s.%s.%s.trend.pdf' % (odir,varn,re,se), format='pdf', dpi=300)
        plt.close()

