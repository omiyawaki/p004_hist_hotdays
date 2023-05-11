import os
import sys
sys.path.append('../../')
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from tqdm import tqdm
from regbins import rbin

varn='qt2m'
lre=['swus']
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','son','jja'] # season (ann, djf, mam, jja, son)
yr='1950'

for re in lre:
    mt,mq=rbin(re)
    abm=np.vstack([mt.ravel(),mq.ravel()])

    for se in lse:
        idir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)
        odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/era5/%s/%s' % (se,varn)

        if not os.path.exists(odir):
            os.makedirs(odir)

        # KDE
        kde = pickle.load(open('%s/k%s_%s.%s.%s.pickle' % (idir,varn,yr,re,se), 'rb'))
        pdf=np.reshape(kde(abm).T,mt.shape)
        
        fig,ax=plt.subplots(figsize=(4,3))
        ax.contour(mt,mq,pdf)
        ax.set_xlim([285,310])
        ax.set_xlabel(r'$T_{2\,m}$ (K)')
        ax.set_ylabel(r'$q_{2\,m}$ (kg kg$^{-1}$)')
        ax.set_title(r'%s ERA5 %s' % (se.upper(),re.upper()))
        fig.tight_layout()
        plt.savefig('%s/kde.%s.%s.%s.pdf' % (odir,varn,re,se), format='pdf', dpi=300)
        plt.close()
