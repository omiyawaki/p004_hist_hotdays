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
xpc='95'
lre=['sea_mp']
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','son','jja'] # season (ann, djf, mam, jja, son)
by=[1950,2020]
ts=np.arange(by[0],by[1]+1)

for re in lre:
    for se in lse:
        idir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)
        odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/era5/%s/%s' % (se,varn)

        if not os.path.exists(odir):
            os.makedirs(odir)

        c = 0
        # Load timeseries
        tst2m=pickle.load(open('%s/tseries.dw%s.%s.%s.%g.%g.%s.pickle' % (idir,varn,xpc,re,by[0],by[1],se), 'rb'))

        # Timeseries
        fig,ax=plt.subplots(figsize=(4,3))
        ax.axhline(0,linewidth=0.5,color='k')
        ax.plot(ts,tst2m,'k')
        ax.set_xlabel(r'Year')
        ax.set_ylabel(r'$T^{%s}_\mathrm{2\,m}-T^{50}_\mathrm{2\,m}$ (K)'%xpc)
        ax.set_title(r'%s ERA5 %s' % (se.upper(),re.upper()))
        fig.tight_layout()
        plt.savefig('%s/tseries.%s.%s.%s.%s.trend.pdf' % (odir,varn,xpc,re,se), format='pdf', dpi=300)
        plt.close()

