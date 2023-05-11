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

nt=7 # window size in days
p=95
pref1='ddp'
varn1='hfls'
pref2='ddp'
varn2='pr'
pref3='p'
varn3='hurs'
phurs0=30 # threshold for dry hot days mm/d
varn='%s%s+%s%s+%s%s'%(pref1,varn1,pref2,varn2,pref3,varn3)
se = 'sc'
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'
mmm=True
ann=True

troplat=20    # latitudinal bound of tropics

largs=[
    # {
    # 'landonly':False, # only use land grid points for rsq
    # 'troponly':False, # only look at tropics
    # },
    {
    'landonly':True, # only use land grid points for rsq
    'troponly':True, # only look at tropics
    },
    {
    'landonly':True, # only use land grid points for rsq
    'troponly':False, # only look at tropics
    },
]

for args in largs:
    # grid
    rgdir='/project/amp/miyawaki/data/share/regrid'
    # open CESM data to get output grid
    cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
    cdat=xr.open_dataset(cfil)
    gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

    # load land mask
    lm,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomask.pickle','rb'))

    if args['troponly']:
        lats=np.transpose(np.tile(gr['lat'].data,(len(gr['lon']),1)),[1,0])
        lm[np.abs(lats)>troplat]=np.nan

    md='mi'
    idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
    idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn2)
    idir3 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn3)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    i1=pickle.load(open('%s/%s%s_%s_%s.%s.%s.pickle' % (idir1,pref1,varn1,his,fut,p,se), 'rb'))
    i2=pickle.load(open('%s/%s%s_%s_%s.%s.%s.pickle' % (idir2,pref2,varn2,his,fut,p,se), 'rb'))
    i3=pickle.load(open('%s/%s%s_%s.%s.pickle' % (idir3,pref3,varn3,his,se), 'rb'))

    if p==95:
        i3=i3[:,:,-2,...]

    if i2.shape != i1.shape:
        i2=np.transpose(i2[...,None],[0,1,4,2,3])
        i3=np.transpose(i3[...,None],[0,1,4,2,3])

    if args['landonly']:
        i1=i1*lm
        i2=i2*lm
        i3=i3*lm
        # r=corr2d(i1,i2,gr,(3,4),lm=lm)
    # else:
        # r=corr2d(i1,i2,gr,(3,4))

    oname='%s/sp.rsq.%s_%s_%s.%s' % (odir,varn,his,fut,se)
    if args['landonly']:
        oname='%s.land'%oname
    if args['troponly']:
        oname='%s.trop'%oname

    if mmm:
        md='mmm'
        tname=r'%s, $RH^{%02d}<%g$ %%'%(md.upper(),p,phurs0)
        if args['landonly']:
            tname='%s, Land'%tname
        if args['troponly']:
            tname='%s, Tropics'%tname

        i1=np.nanmean(i1,axis=0)
        i2=np.nanmean(i2,axis=0)
        i3=np.nanmean(i3,axis=0)

        if ann:
            fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
            mi1=i1.flatten()
            mi2=i2.flatten()
            mi3=i3.flatten()
            mi1=mi1[~np.isnan(mi1)]
            mi2=mi2[~np.isnan(mi2)]
            mi3=mi3[~np.isnan(mi3)]
            stack=np.vstack([86400*mi2,mi1])
            kde=gaussian_kde(stack)
            pdf=kde(stack)
            rsq=corr(mi2,mi1,dim=0)
            clf=ax.scatter(86400*mi2,mi1,s=0.1,c=mi3)
            ax.annotate(r'$R^2=%.2f$'%rsq, xy=(0.05, 0.9), xycoords='axes fraction')
            ax.set_title(r'%s, ANN' % (tname),fontsize=12)
            ax.set_xlabel('$\Delta\delta{P}^{%02d}$ (mm d$^{-1}$)'%p)
            ax.set_ylabel('$\Delta\delta LH^{%02d}$ (W m$^{-2}$)'%(p))
            cb=fig.colorbar(clf,location='right',aspect=50)
            cb.ax.tick_params(labelsize=12)
            cb.set_label(label=r'$RH^{%02d}$ (%%)'%(p),size=12)
            fig.savefig('%s.ann.pdf'%oname, format='pdf', dpi=300)

        else:
            fig,ax=plt.subplots(figsize=(12,8),nrows=3,ncols=4,constrained_layout=True)
            fig.suptitle(r'%s' % (tname),fontsize=16)
            ax=ax.flatten()
            for mon in range(12):
                mi1=i1[mon,...]
                mi2=i2[mon,...]
                mi3=i3[mon,...]

                mi1=mi1.flatten()
                mi2=mi2.flatten()
                mi3=mi3.flatten()
                mi1=mi1[~np.isnan(mi1)]
                mi2=mi2[~np.isnan(mi2)]
                mi3=mi3[~np.isnan(mi3)]
                ax[mon].scatter(86400*mi2,mi1,s=0.1,c=mi3)
                ax[mon].set_title(r'%s' % (monname(mon)),fontsize=16)
                ax[mon].set_xlabel('$\Delta\delta{P}^{%02d}$ (mm d$^{-1}$)'%p)
                ax[mon].set_ylabel('$\Delta\delta LH^{%02d}$ (W m$^{-2}$)'%(p))
                fig.savefig('%s.pdf'%oname, format='pdf', dpi=300)

    else:
        md='mi'
        tname=r'%s' % (md.upper())
        if args['landonly']:
            tname='%s, Land'%tname
        if args['troponly']:
            tname='%s, Tropics'%tname

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
            # stack=np.vstack([86400*mi2,mi1])
            # kde=gaussian_kde(stack)
            # abm=np.vstack(stack)
            # pdf=kde(abm)
            ax[-1].scatter(86400*mi2,mi1,s=0.1,c=mi3)
            ax[-1].set_title(r'%s' % (md),fontsize=16)
            ax[-1].set_xlabel('$\Delta\delta{P}^{%02d}$ (mm d$^{-1}$)'%p)
            ax[-1].set_ylabel('$\Delta\delta LH^{%02d}$ (W m$^{-2}$)'%(p))
            fig.savefig('%s.ann.mi.pdf'%oname, format='pdf', dpi=300)

            for imd,md in enumerate(lmd):
                mi1=i1[imd,...].flatten()
                mi2=i2[imd,...].flatten()
                mi3=i3[imd,...].flatten()
                mi1=mi1[~np.isnan(mi1)]
                mi2=mi2[~np.isnan(mi2)]
                mi3=mi3[~np.isnan(mi3)]
                # stack=np.vstack([86400*mi2,mi1])
                # kde=gaussian_kde(stack)
                # abm=np.vstack(stack)
                # pdf=kde(abm)
                ax[imd].scatter(86400*mi2,mi1,s=0.1,c=mi3)
                ax[imd].set_title(r'%s' % (md),fontsize=16)
                ax[imd].set_xlabel('$\Delta\delta{P}^{%02d}$ (mm d$^{-1}$)'%p)
                ax[imd].set_ylabel('$\Delta\delta LH^{%02d}$ (W m$^{-2}$)'%(p))
                fig.savefig('%s.ann.mi.pdf'%oname, format='pdf', dpi=300)

