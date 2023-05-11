import os
import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import gaussian_kde
from tqdm import tqdm

varn='t2m'
lre=['sea']
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','son','jja'] # season (ann, djf, mam, jja, son)
by0=[1950,1970]
by1=[2000,2020]

for re in lre:
    for se in lse:
        idir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)
        odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/era5/%s/%s' % (se,varn)

        if not os.path.exists(odir):
            os.makedirs(odir)

        c = 0
        # Load climatology
        [ht2m0,lpc0]=pickle.load(open('%s/clmean.%s.%s.%g.%g.%s.pickle' % (idir,varn,re,by0[0],by0[1],se), 'rb'))
        [ht2m1,lpc1]=pickle.load(open('%s/clmean.%s.%s.%g.%g.%s.pickle' % (idir,varn,re,by1[0],by1[1],se), 'rb'))
        clt2m0={}
        clt2m1={}
        for pc in [5,50,95]:
            clt2m0[str(pc)]=ht2m0[np.where(np.equal(lpc0,pc))[0][0]]
            clt2m1[str(pc)]=ht2m1[np.where(np.equal(lpc1,pc))[0][0]]

        # Compare KDE past and future
        [kt2m0,kbm0] = pickle.load(open('%s/clmean.k%s.%s.%g.%g.%s.pickle' % (idir,varn,re,by0[0],by0[1],se), 'rb'))
        [kt2m1,kbm1] = pickle.load(open('%s/clmean.k%s.%s.%g.%g.%s.pickle' % (idir,varn,re,by1[0],by1[1],se), 'rb'))
        fig,ax=plt.subplots(figsize=(4,3))
        ax.axhline(0,linewidth=0.5,color='k')
        ax.plot(kbm0,kt2m0,'k',label='%g-%g'%(by0[0],by0[1]))
        ax.plot(kbm1,kt2m1,'tab:orange',label='%g-%g'%(by1[0],by1[1]))
        # ax.set_xlim([285,310])
        ymin,ymax=ax.get_ylim()
        ax.plot(clt2m0['5'],ymin,'|',color='k')
        ax.plot(clt2m1['5'],ymin,'|',color='tab:orange')
        ax.plot(clt2m0['50'],ymin,'|',color='k')
        ax.plot(clt2m1['50'],ymin,'|',color='tab:orange')
        ax.plot(clt2m0['95'],ymin,'|',color='k')
        ax.plot(clt2m1['95'],ymin,'|',color='tab:orange')
        ax.set_ylim([ymin,ymax])
        ax.set_xlabel(r'$T_{2\,m}$ (K)')
        ax.set_ylabel(r'Probability Density (K$^{-1}$)')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_title(r'%s ERA5 %s' % (se.upper(),re.upper()))
        ax.legend()
        fig.tight_layout()
        plt.savefig('%s/kde.%s.%s.%s.comp.pdf' % (odir,varn,re,se), format='pdf', dpi=300)
        plt.close()

