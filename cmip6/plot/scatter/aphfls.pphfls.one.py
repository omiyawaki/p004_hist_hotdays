import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import pickle
import pwlf
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from scipy.stats import gaussian_kde
from tqdm import tqdm
from cmip6util import mods
from utils import corr,corr2d,monname
from regions import pointlocs

# lre=['amazon','sahara','sea','zambia']
lre=['sea','amazon','yuma']
# lre=['zambia']

nt=7 # window size in days
nseg=2 # number of line segments for pwlf
ndeg=1 # degree of polynomial fit
pref1='p'
varn1='hfls'
pref2='p'
varn2='mrsos'
pref3='p'
varn3='pr'
pref4='p'
varn4='tas'
varn='%s%s+%s%s'%(pref1,varn1,pref2,varn2)
se = 'sc'

fo='historical' # forcings 
yr='1980-2000'

# fo='ssp370' # forcings 
# yr='2080-2100'

mmm=True
ann=False

troplat=20    # latitudinal bound of tropics

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

# load land mask
lm,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomask.pickle','rb'))

if mmm:
    md='mmm'
else:
    md='mi'

for re in lre:
    iloc=pointlocs(re)
    la=gr['lat'][iloc[0]]
    lo=gr['lon'][iloc[1]]

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/selpoint/%s' % (se,fo,'mmm',varn,re)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/selpoint/%s' % (se,fo,md,varn,re)
    if not os.path.exists(odir):
        os.makedirs(odir)

    iname='%s/pwlf.%gd%g.bc.%s_%s.%s' % (idir,nseg,ndeg,varn,yr,se)
    [a,p,pm,pct]=pickle.load(open('%s.pickle' %iname, 'rb'))

    oname='%s/pred.pwlf.%gd%g.bc.%s_%s.%s' % (odir,nseg,ndeg,varn,yr,se)

    if mmm:
        tname=r'%s, [%+05.1f, %+05.1f]' % (md.upper(),la,lo)

        pct=np.nanmean(pct,axis=0)

        if ann:
            fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
            for m in range(a.shape[0]):
                clf=ax.scatter(p[m,:],a[m,:],marker='$%g$'%(m+1),s=30,c=pct[m,:])
            ax.set_title(r'%s' % (tname),fontsize=16)
            ax.set_xlabel('$Predicted \Delta LH(\Delta SM)$ (W m$^{-2}$)')
            ax.set_ylabel('$Actual \Delta LH$ (W m$^{-2}$)')
            fig.savefig('%s.ann.pdf'%oname, format='pdf', dpi=300)

        else:
            fig,ax=plt.subplots(figsize=(12,8),nrows=3,ncols=4,constrained_layout=True)
            fig.suptitle(r'%s' % (tname),fontsize=16)
            ax=ax.flatten()
            for mon in range(12):
                ma=a[mon,...]
                mp=p[mon,...]
                # mpm=pm[mon,...]
                mpct=pct[mon,...]
                i95=np.where(mpct==95)
                i50=np.where(mpct==50)

                ax[mon].scatter(mp,ma,s=20,c=mpct)
                ax[mon].scatter(mp[i95],ma[i95],marker='$95$',s=100,c='k')
                ax[mon].scatter(mp[i50],ma[i50],marker='$50$',s=100,c='k')
                ax[mon].text(0.1,0.9,r'$\Delta\delta LH=%g$ W m$^{-2}$'%(ma[i95]-ma[i50]),transform=ax[mon].transAxes)
                oto=np.linspace(ax[mon].get_xlim()[0],ax[mon].get_xlim()[1],101)
                ax[mon].plot(oto,oto,'--k')
                ax[mon].set_title(r'%s' % (monname(mon)),fontsize=16)
                ax[mon].set_xlabel('$Predicted \Delta LH(\Delta SM)$ (W m$^{-2}$)')
                ax[mon].set_ylabel('$Actual \Delta LH$ (W m$^{-2}$)')
                fig.savefig('%s.pdf'%oname, format='pdf', dpi=300)

    else:
        for imd,md in enumerate(mods(fo)):
            tname=r'%s, [%+05.1f, %+05.1f]' % (md.upper(),la,lo)

            odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/selpoint/%s' % (se,fo,md,varn,re)
            if not os.path.exists(odir):
                os.makedirs(odir)

            oname='%s/pred.pwlf.%gd%g.bc.%s_%s.%s' % (odir,nseg,ndeg,varn,yr,se)

            fig,ax=plt.subplots(figsize=(12,8),nrows=3,ncols=4,constrained_layout=True)
            fig.suptitle(r'%s' % (tname),fontsize=16)
            ax=ax.flatten()
            for mon in range(12):
                ma=a[imd,mon,...]
                mp=p[imd,mon,...]
                # mpm=pm[mon,...]
                mpct=pct[imd,mon,...]
                i95=np.where(mpct==95)
                i50=np.where(mpct==50)

                ax[mon].scatter(mp,ma,s=20,c=mpct)
                ax[mon].scatter(mp[i95],ma[i95],marker='$95$',s=100,c='k')
                ax[mon].scatter(mp[i50],ma[i50],marker='$50$',s=100,c='k')
                oto=np.linspace(ax[mon].get_xlim()[0],ax[mon].get_xlim()[1],101)
                ax[mon].plot(oto,oto,'--k')
                ax[mon].set_title(r'%s' % (monname(mon)),fontsize=16)
                ax[mon].set_xlabel('$Predicted \Delta LH$ (W m$^{-2}$)')
                ax[mon].set_ylabel('$Actual \Delta LH(\Delta SM)$ (W m$^{-2}$)')
                fig.savefig('%s.pdf'%oname, format='pdf', dpi=300)
