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
from metpy.units import units
import constants as c

varn='qt2m'
lpc=['','95']
lre=['sea','swus']
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','son','jja'] # season (ann, djf, mam, jja, son)
by0=[1950,1970]
by1=[2000,2020]

def cc(t,t0,q0):
    t=t*units.kelvin
    t0=t0*units.kelvin
    q=q0*np.exp(c.Lv/c.Rv*(1/t0-1/t))
    return q 

for pc in lpc:
    for re in lre:
        for se in lse:
            idir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)
            odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/era5/%s/%s' % (se,varn)

            if not os.path.exists(odir):
                os.makedirs(odir)

            # KDE
            if pc=='':
                [pdf0,mt,mq] = pickle.load(open('%s/clmean.k%s.%s.%g.%g.%s.pickle' % (idir,varn,re,by0[0],by0[1],se), 'rb'))
                [pdf1,mt,mq] = pickle.load(open('%s/clmean.k%s.%s.%g.%g.%s.pickle' % (idir,varn,re,by1[0],by1[1],se), 'rb'))
            else:
                [pdf0,mt,mq] = pickle.load(open('%s/clmean.k%s.gt%s.%s.%g.%g.%s.pickle' % (idir,varn,pc,re,by0[0],by0[1],se), 'rb'))
                [pdf1,mt,mq] = pickle.load(open('%s/clmean.k%s.gt%s.%s.%g.%g.%s.pickle' % (idir,varn,pc,re,by1[0],by1[1],se), 'rb'))
            dpdf=pdf1-pdf0
            
            fig,ax=plt.subplots(figsize=(4,3))
            vmax=np.abs(dpdf).max()
            xlim=rtlm(re,pc)
            mloc0=np.where(pdf0==pdf0.max())
            t0,q0=mt[mloc0],mq[mloc0]
            mloc1=np.where(pdf1==pdf1.max())
            t1,q1=mt[mloc1],mq[mloc1]
            tr=np.linspace(xlim[0],xlim[1],101)
            qr=cc(tr,t0,q0)
            ax.contour(mt,mq,dpdf,vmax=vmax,vmin=-vmax,cmap='RdBu_r')
            ylim=ax.get_ylim()
            ax.plot(t0,q0,'*',color='black')
            ax.plot(t1,q1,'*',color='tab:purple')
            ax.plot(tr,qr,color='gray')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(r'$T_{2\,m}$ (K)')
            ax.set_ylabel(r'$q_{2\,m}$ (kg kg$^{-1}$)')
            if pc=='':
                ax.set_title(r'%s ERA5 %s (all days)' % (se.upper(),re.upper()))
            else:
                ax.set_title(r'%s ERA5 %s ($>T^{%s}_{2\,m}$ days)' % (se.upper(),re.upper(),pc))
            fig.tight_layout()
            if pc=='':
                plt.savefig('%s/kde.diff.%s.%s.%g.%g.%g.%g.%s.pdf' % (odir,varn,re,by0[0],by0[1],by1[0],by1[1],se), format='pdf', dpi=300)
            else:
                plt.savefig('%s/kde.diff.%s.gt%s.%s.%g.%g.%g.%g.%s.pdf' % (odir,varn,pc,re,by0[0],by0[1],by1[0],by1[1],se), format='pdf', dpi=300)
            plt.close()
