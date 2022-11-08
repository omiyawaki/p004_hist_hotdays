import os
import sys
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from tqdm import tqdm
from regions import rbin,rlev,rtlm

varn='qt2m'
lpc=['95']
lre=['sea','swus']
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','son','jja'] # season (ann, djf, mam, jja, son)
# by=[1950,1970]
by=[2000,2020]

for pc in lpc:
    for re in lre:
        for se in lse:
            idir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)
            odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/era5/%s/%s' % (se,varn)

            if not os.path.exists(odir):
                os.makedirs(odir)

            # KDE
            if pc=='':
                [pdf,mt,mq] = pickle.load(open('%s/clmean.k%s.%s.%g.%g.%s.pickle' % (idir,varn,re,by[0],by[1],se), 'rb'))
            else:
                [pdf,mt,mq] = pickle.load(open('%s/clmean.k%s.gt%s.%s.%g.%g.%s.pickle' % (idir,varn,pc,re,by[0],by[1],se), 'rb'))
            
            fig,ax=plt.subplots(figsize=(4,3))
            print(pdf.max())
            ax.contour(mt,mq,pdf,rlev(re,pc))
            ax.set_xlim(rtlm(re,pc))
            ax.set_xlabel(r'$T_{2\,m}$ (K)')
            ax.set_ylabel(r'$q_{2\,m}$ (kg kg$^{-1}$)')
            if pc=='':
                ax.set_title(r'%s ERA5 %s %g-%g (all days)' % (se.upper(),re.upper(),by[0],by[1]))
            else:
                ax.set_title(r'%s ERA5 %s %g-%g ($>T^{%s}_{2\,m}$ days)' % (se.upper(),re.upper(),by[0],by[1],pc))
            fig.tight_layout()
            if pc=='':
                plt.savefig('%s/kde.%s.%s.%g.%g.%s.pdf' % (odir,varn,re,by[0],by[1],se), format='pdf', dpi=300)
            else:
                plt.savefig('%s/kde.%s.gt%s.%s.%g.%g.%s.pdf' % (odir,varn,pc,re,by[0],by[1],se), format='pdf', dpi=300)
            plt.close()
