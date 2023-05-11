import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import pickle
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

# lre=['zambia','amazon','sahara','sea']
lre=['zambia']

nt=7 # window size in days
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
idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mi',varn1)
idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mi',varn2)
idir3 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mi',varn3)
idir4 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mi',varn4)
idirg = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mmm',varn4)

ri1=pickle.load(open('%s/%s%s_%s.%s.pickle' % (idir1,pref1,varn1,yr,se), 'rb'))
ri2=pickle.load(open('%s/%s%s_%s.%s.pickle' % (idir2,pref2,varn2,yr,se), 'rb'))
# ri3=pickle.load(open('%s/%s%s_%s.%s.pickle' % (idir3,pref3,varn3,yr,se), 'rb'))
ri4=pickle.load(open('%s/%s%s_%s.%s.pickle' % (idir4,pref4,varn4,yr,se), 'rb'))
_,_,gr=pickle.load(open('%s/%s%s_%s.%s.pickle' % (idirg,pref4,varn4,yr,se), 'rb'))

rs1=pickle.load(open('%s/std%s%s_%s.%s.pickle' % (idir1,pref1,varn1,yr,se), 'rb'))
rs2=pickle.load(open('%s/std%s%s_%s.%s.pickle' % (idir2,pref2,varn2,yr,se), 'rb'))

for re in lre:
    iloc=pointlocs(re)
    la=gr['lat'][iloc[0]]
    lo=gr['lon'][iloc[1]]

    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/selpoint/%s' % (se,cl,fo,md,varn,re)
    if not os.path.exists(odir):
        os.makedirs(odir)

    i1=ri1[...,iloc[0],iloc[1]]
    i2=ri2[...,iloc[0],iloc[1]]
    # i3=ri3[...,iloc[0],iloc[1]]
    i4=ri4[...,iloc[0],iloc[1]]

    s1=rs1[...,iloc[0],iloc[1]]
    s2=rs2[...,iloc[0],iloc[1]]

    pct=np.tile(gr['pct'].data,(i4.shape[0],i4.shape[1],1))

    oname='%s/bc.%s_%s.%s' % (odir,varn,yr,se)

    if mmm:
        tname=r'%s, [%+05.1f, %+05.1f]' % (md.upper(),la,lo)

        i1=np.nanmean(i1,axis=0)
        i2=np.nanmean(i2,axis=0)
        # i3=np.nanmean(i3,axis=0)
        i4=np.nanmean(i4,axis=0)
        pct=np.nanmean(pct,axis=0)

        s1=np.nanmean(s1,axis=0)
        s2=np.nanmean(s2,axis=0)

        if ann:
            fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
            for m in range(i1.shape[0]):
                clf=ax.scatter(i2[m,:],i1[m,:],marker='$%g$'%(m+1),s=30,c=pct[m,:])
            ax.set_title(r'%s' % (tname),fontsize=16)
            ax.set_xlabel('${SM}$ (kg m$^{-2}$)')
            ax.set_ylabel('$ LH$ (W m$^{-2}$)')
            fig.savefig('%s.ann.pdf'%oname, format='pdf', dpi=300)

        else:
            fig,ax=plt.subplots(figsize=(12,8),nrows=3,ncols=4,constrained_layout=True)
            fig.suptitle(r'%s' % (tname),fontsize=16)
            ax=ax.flatten()
            for mon in range(12):
                mi1=i1[mon,...]
                mi2=i2[mon,...]
                # mi3=i3[mon,...]
                mpct=pct[mon,...]
                ms1=s1[mon,...]
                ms2=s2[mon,...]
                i95=np.where(mpct==95)
                i50=np.where(mpct==50)
                # ax[mon].scatter(mi2,mi1,s=20,c=mpct)
                ax[mon].errorbar(mi2,mi1,xerr=ms2,yerr=ms1,fmt='.',elinewidth=0.5,color=mpct)
                ax[mon].scatter(mi2[i95],mi1[i95],marker='$95$',s=100,c='k')
                ax[mon].scatter(mi2[i50],mi1[i50],marker='$50$',s=100,c='k')
                ax[mon].set_title(r'%s' % (monname(mon)),fontsize=16)
                ax[mon].set_xlabel('${SM}$ (kg m$^{-2}$)')
                ax[mon].set_ylabel('$ LH$ (W m$^{-2}$)')
                fig.savefig('%s.pdf'%oname, format='pdf', dpi=300)

    else:
        tname=r'%s' % (md.upper())

        if ann:
            lmd=mods(fo1)
            fig,ax=plt.subplots(figsize=(12,8),nrows=4,ncols=5,constrained_layout=True)
            fig.suptitle(r'%s, ANN' % (tname),fontsize=16)
            ax=ax.flatten()

            #mmm
            mi1=np.nanmean(i1,axis=0).flatten()
            mi2=np.nanmean(i2,axis=0).flatten()
            mi3=np.nanmean(i3,axis=0).flatten()
            mi1=mi1[~np.isnan(mi1)]
            mi2=mi2[~np.isnan(mi2)]
            mi3=mi3[~np.isnan(mi3)]
            # stack=np.vstack([mi2,mi1])
            # kde=gaussian_kde(stack)
            # abm=np.vstack(stack)
            # pdf=kde(abm)
            ax[-1].scatter(mi2,mi1,s=0.1,c=mi3)
            ax[-1].set_title(r'%s' % (md),fontsize=16)
            ax[-1].set_xlabel('${SM}$ (kg m$^{-2}$)')
            ax[-1].set_ylabel('$ LH$ (W m$^{-2}$)')
            fig.savefig('%s.ann.mi.pdf'%oname, format='pdf', dpi=300)

            for imd,md in enumerate(lmd):
                mi1=i1[imd,...].flatten()
                mi2=i2[imd,...].flatten()
                mi3=i3[imd,...].flatten()
                mi1=mi1[~np.isnan(mi1)]
                mi2=mi2[~np.isnan(mi2)]
                mi3=mi3[~np.isnan(mi3)]
                # stack=np.vstack([mi2,mi1])
                # kde=gaussian_kde(stack)
                # abm=np.vstack(stack)
                # pdf=kde(abm)
                ax[imd].scatter(mi2,mi1,s=0.1,c=mi3)
                ax[imd].set_title(r'%s' % (md),fontsize=16)
                ax[imd].set_xlabel('${SM}$ (kg m$^{-2}$)')
                ax[imd].set_ylabel('$ LH$ (W m$^{-2}$)')
                fig.savefig('%s.ann.mi.pdf'%oname, format='pdf', dpi=300)

