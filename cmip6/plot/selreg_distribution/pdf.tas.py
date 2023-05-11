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
from util import mods
from utils import corr,corr2d,monname
from regions import pointlocs
from CASutils import shapefile_utils as shp

re=['Utah','Colorado','Arizona','New Mexico']

nt=7 # window size in days
lpc=np.concatenate((np.arange(0,50,10),np.arange(50,75,5),np.arange(75,95,2.5),np.arange(95,100,1)))
nseg=2 # number of line segments for pwlf
ndeg=1 # degree of polynomial fit
varn='tas'
se = 'sc'

fo1='historical' # forcings 
yr1='1980-2000'

fo2='ssp370' # forcings 
yr2='2080-2100'

md='CanESM5'

# load land indices
lmi,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

# mask
shpfile = "/project/cas/islas/shapefiles/usa/gadm36_USA_1.shp"
mask = shp.maskgen(shpfile,gr,re)
print(mask.shape)
sys.exit()

idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo1,md,varn)
idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo2,md,varn)

ds=xr.open_dataset('%s/p.%s%03d_%s.%s.nc' % (idir1,varn,nt,yr1,se))
vn1=ds['__xarray_dataarray_variable__']
pct=ds['percentile']
gpi=ds['gpi']

odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/selpoint/%s' % (se,fo,md,varn,re)
if not os.path.exists(odir):
    os.makedirs(odir)

pct=np.tile(lpc,(i4.shape[0],i4.shape[1],1))

oname='%s/pwlf.%gd%g.bc.%s_%s.%s' % (odir,nseg,ndeg,varn,yr,se)

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
            mpct=pct[mon,...]
            ms1=s1[mon,...]
            ms2=s2[mon,...]
            i95=np.where(mpct==95)
            i0=np.where(mpct==0)

            # use local maximum of a cubic polynomial fit as breaking point of piecewise linear fit
            pa=np.polynomial.polynomial.polyfit(mi2,mi1,3) # polynomial fit
            xv1=-(2*pa[2]+np.sqrt(4*pa[2]**2-12*pa[1]*pa[3]))/(6*pa[3])
            xv2=-(2*pa[2]-np.sqrt(4*pa[2]**2-12*pa[1]*pa[3]))/(6*pa[3])
            cc1=2*pa[2]+6*pa[3]*xv1
            cc2=2*pa[2]+6*pa[3]*xv2
            if cc1<0:
                bp=xv1
            else:
                bp=xv2
            mp=pwlf.PiecewiseLinFit(mi2,mi1,degree=ndeg)
            fp=mp.fit_with_breaks([np.min(mi2),bp,np.max(mi2)])
            xh=np.linspace(min(mi2),max(mi2),101)
            yh=mp.predict(xh)
            yp=mp.predict(xh)

            ax[mon].scatter(mi2,mi1,s=20,c=mpct)
            # ax[mon].errorbar(mi2,mi1,xerr=ms2,yerr=ms1,fmt='.',elinewidth=0.5,color=mpct)
            ax[mon].scatter(mi2[i95],mi1[i95],marker='$95$',s=100,c='k')
            ax[mon].scatter(mi2[i0],mi1[i0],marker='$M$',s=100,c='k')
            ax[mon].plot(xh,yh,'-k')
            ax[mon].set_title(r'%s' % (monname(mon)),fontsize=16)
            ax[mon].set_xlabel('${SM}$ (kg m$^{-2}$)')
            ax[mon].set_ylabel('$ LH$ (W m$^{-2}$)')
            fig.savefig('%s.pdf'%oname, format='pdf', dpi=300)

else:
    for imd,md in enumerate(mods(fo)):
        tname=r'%s' % (md.upper())

        odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/selpoint/%s' % (se,fo,md,varn,re)
        if not os.path.exists(odir):
            os.makedirs(odir)

        oname='%s/pwlf.%gd%g.bc.%s_%s.%s' % (odir,nseg,ndeg,varn,yr,se)

        fig,ax=plt.subplots(figsize=(12,8),nrows=3,ncols=4,constrained_layout=True)
        fig.suptitle(r'%s' % (tname),fontsize=16)
        ax=ax.flatten()
        for mon in range(12):
            mi1=i1[imd,mon,...]
            mi2=i2[imd,mon,...]
            mpct=pct[imd,mon,...]
            ms1=s1[imd,mon,...]
            ms2=s2[imd,mon,...]
            i95=np.where(mpct==95)
            i50=np.where(mpct==50)

            mp=pwlf.PiecewiseLinFit(mi2,mi1,degree=ndeg)
            fp=mp.fit(nseg)
            xh=np.linspace(min(mi2),max(mi2),101)
            yh=mp.predict(xh)

            ax[mon].scatter(mi2,mi1,s=20,c=mpct)
            # ax[mon].errorbar(mi2,mi1,xerr=ms2,yerr=ms1,fmt='.',elinewidth=0.5,color=mpct)
            ax[mon].scatter(mi2[i95],mi1[i95],marker='$95$',s=100,c='k')
            ax[mon].scatter(mi2[i50],mi1[i50],marker='$50$',s=100,c='k')
            ax[mon].plot(xh,yh,'-k')
            ax[mon].set_title(r'%s' % (monname(mon)),fontsize=16)
            ax[mon].set_xlabel('${SM}$ (kg m$^{-2}$)')
            ax[mon].set_ylabel('$ LH$ (W m$^{-2}$)')
            fig.savefig('%s.pdf'%oname, format='pdf', dpi=300)

    #if ann:
    #    lmd=mods(fo1)
    #    fig,ax=plt.subplots(figsize=(12,8),nrows=4,ncols=5,constrained_layout=True)
    #    fig.suptitle(r'%s, ANN' % (tname),fontsize=16)
    #    ax=ax.flatten()

    #    #mmm
    #    mi1=np.nanmean(i1,axis=0).flatten()
    #    mi2=np.nanmean(i2,axis=0).flatten()
    #    mpct=pct[mon,...]
    #    mi1=mi1[~np.isnan(mi1)]
    #    mi2=mi2[~np.isnan(mi2)]

    #    mp=pwlf.PiecewiseLinFit(mi2,mi1,degree=ndeg)
    #    fp=mp.fit(nseg)
    #    xh=np.linspace(min(mi2),max(mi2),101)
    #    yh=mp.predict(xh)

    #    # stack=np.vstack([mi2,mi1])
    #    # kde=gaussian_kde(stack)
    #    # abm=np.vstack(stack)
    #    # pdf=kde(abm)
    #    ax[-1].scatter(mi2,mi1,s=0.1,c=mi3)
    #    ax[-1].set_title(r'%s' % (md),fontsize=16)
    #    ax[-1].set_xlabel('${SM}$ (kg m$^{-2}$)')
    #    ax[-1].set_ylabel('$ LH$ (W m$^{-2}$)')
    #    fig.savefig('%s.ann.mi.pdf'%oname, format='pdf', dpi=300)

    #    for imd,md in enumerate(lmd):
    #        mi1=i1[imd,...].flatten()
    #        mi2=i2[imd,...].flatten()
    #        mpct=pct[mon,...]
    #        mi1=mi1[~np.isnan(mi1)]
    #        mi2=mi2[~np.isnan(mi2)]

    #        mp=pwlf.PiecewiseLinFit(mi2,mi1,degree=ndeg)
    #        fp=mp.fit(nseg)
    #        xh=np.linspace(min(mi2),max(mi2),101)
    #        yh=mp.predict(xh)

    #        # stack=np.vstack([mi2,mi1])
    #        # kde=gaussian_kde(stack)
    #        # abm=np.vstack(stack)
    #        # pdf=kde(abm)
    #        ax[imd].scatter(mi2,mi1,s=0.1,c=mi3)
    #        ax[imd].set_title(r'%s' % (md),fontsize=16)
    #        ax[imd].set_xlabel('${SM}$ (kg m$^{-2}$)')
    #        ax[imd].set_ylabel('$ LH$ (W m$^{-2}$)')
    #        fig.savefig('%s.ann.mi.pdf'%oname, format='pdf', dpi=300)

