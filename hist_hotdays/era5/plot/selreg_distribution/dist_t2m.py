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
yr='1950'

for re in lre:
    for se in lse:
        idir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)
        odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/era5/%s/%s' % (se,varn)

        if not os.path.exists(odir):
            os.makedirs(odir)

        c = 0
        [hg, bn] = pickle.load(open('%s/f%s_%s.%s.%s.pickle' % (idir,varn,yr,re,se), 'rb'))
        bm=1/2*(bn[1:]+bn[:-1]) # bin midpoints
        pdf=hg/hg.sum()/(bn[1:]-bn[:-1]) # convert to PDF
        
        # KDE
        kde = pickle.load(open('%s/k%s_%s.%s.%s.pickle' % (idir,varn,yr,re,se), 'rb'))
        kbm=np.arange(285,310,0.1) # use finer bins for KDE
        fig,ax=plt.subplots(figsize=(4,3))
        ax.plot(kbm,kde(kbm))
        ax.set_xlim([285,310])
        ax.set_xlabel(r'$T_{2\,m}$ (K)')
        ax.set_ylabel(r'Probability Density (K$^{-1}$)')
        ax.set_title(r'%s ERA5 %s' % (se.upper(),re.upper()))
        fig.tight_layout()
        plt.savefig('%s/kde.%s.%s.%s.pdf' % (odir,varn,re,se), format='pdf', dpi=300)
        plt.close()

        # Histogram
        fig,ax=plt.subplots(figsize=(4,3))
        ax.stairs(hg,bn)
        ax.set_xlim([285,310])
        ax.set_xlabel(r'$T_{2\,m}$ (K)')
        ax.set_ylabel(r'Days (Total=%g)' % hg.sum())
        ax.set_title(r'%s ERA5 %s' % (se.upper(),re.upper()))
        fig.tight_layout()
        plt.savefig('%s/hgram.%s.%s.%s.pdf' % (odir,varn,re,se), format='pdf', dpi=300)
        plt.close()

        # PDF
        fig,ax=plt.subplots(figsize=(4,3))
        ax.stairs(pdf,bn)
        ax.set_xlim([285,310])
        ax.set_xlabel(r'$T_{2\,m}$ (K)')
        ax.set_ylabel(r'Probability Density (K$^{-1}$)')
        ax.set_title(r'%s ERA5 %s' % (se.upper(),re.upper()))
        fig.tight_layout()
        plt.savefig('%s/pdf.%s.%s.%s.pdf' % (odir,varn,re,se), format='pdf', dpi=300)
        plt.close()

