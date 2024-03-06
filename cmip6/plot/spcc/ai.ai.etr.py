import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor as Pool
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
from scipy import ndimage
from scipy.stats import linregress
from tqdm import tqdm
from util import mods
from utils import monname,varnlb,unitlb

lre=['tr','ml','hl'] # tr=tropics, ml=midlatitudes, hl=high lat, et=extratropics
tlat=30 # latitude bound for tropics
plat=50 # midlatitude bound
cval=0.4# threshold DdT value for drawing contour line
npai=20 # number of bins for AI percentiles
dpai=100/npai
lpai=np.arange(0,100+dpai,dpai)
mppai=1/2*(lpai[1:]+lpai[:-1])
filt=False # only look at gridpoints with max exceeding value below
fmax=0.5
title=True

bi=0.025
lb=np.arange(bi,1+bi,bi) # area bins
p=95 # percentile
varn='annai'
varn1='mrsos'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'
skip507599=True

md='mmm'

def vmax(vn):
    d={ 'cat':     9,
            }
    return d[vn]

def plot(re):
    # plot strings
    if re=='tr':
        tstr='Tropics'
    elif re=='ml':
        tstr='Midlatitudes'
    elif re=='hl':
        tstr='High latitudes'
    elif re=='et':
        tstr='Extratropics'
    fstr='.filt' if filt else ''

    # load land indices
    lmi,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

    # grid
    rgdir='/project/amp/miyawaki/data/share/regrid'
    # open CESM data to get output grid
    cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
    cdat=xr.open_dataset(cfil)
    gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

    def remap(v,gr):
        llv=np.nan*np.ones([12,gr['lat'].size*gr['lon'].size])
        llv[:,lmi]=v.data
        llv=np.reshape(llv,(12,gr['lat'].size,gr['lon'].size))
        return llv

    def regsl(v,ma):
        v=v*ma
        v=np.reshape(v,[v.shape[0],v.shape[1]*v.shape[2]])
        kidx=~np.isnan(v).any(axis=0)
        return v[:,kidx],kidx

    def regsla(v,gr,ma):
        return regsl(v,ma)

    def regsl2d(v,ma,kidx):
        v=v*ma
        v=np.reshape(v,[v.shape[0]*v.shape[1]])
        return v[kidx]

    def sortmax(v):
        im=np.argmax(v,axis=0)
        idx=np.argsort(im)
        return v[:,idx],idx

    def loadcsm(fo,yr,vn):
        idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
        ds=xr.open_dataset('%s/%s.%s.%s.nc' % (idir,vn,yr,se))
        gpi=ds['gpi']
        try:
            pvn=ds[vn]
        except:
            pvn=ds['__xarray_dataarray_variable__']
        return pvn,gpi

    def load_vn(varn,fo,byr,px='m'):
        idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        return xr.open_dataarray('%s/%s.%s_%s.%s.nc' % (idir,px,varn,byr,se))

    ai=load_vn(varn,fo1,his,px='m') # historical AI
    aimon=load_vn('ooai',fo1,his,px='m') # historical AI (monthly)

    def loadmvn(fo,yr,vn):
        idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
        ds=xr.open_dataset('%s/anom.m.%s_%s.%s.nc' % (idir,vn,yr,se))
        gpi=ds['gpi']
        try:
            pvn=ds[vn]
        except:
            pvn=ds['__xarray_dataarray_variable__']
        return pvn,gpi

    def loadpvn(fo,yr,vn):
        idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
        ds=xr.open_dataset('%s/anom.pc.%s_%s.%s.nc' % (idir,vn,yr,se))
        pct=ds['percentile']
        gpi=ds['gpi']
        try:
            pvn=ds[vn]
        except:
            pvn=ds['__xarray_dataarray_variable__']
        return pvn,pct,gpi

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    # variable of interest
    idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn1)
    odir1 = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn1)
    if not os.path.exists(odir1):
        os.makedirs(odir1)

    msm1,_=loadmvn(fo1,his,varn1)
    msm2,_=loadmvn(fo2,fut,varn1)
    psm1,_,_=loadpvn(fo1,his,varn1)
    psm2,_,_=loadpvn(fo2,fut,varn1)
    psm1=psm1.sel(percentile=p)
    psm2=psm2.sel(percentile=p)

    # reorder months by AI
    idxai=np.argsort(aimon.data,axis=0)
    msm1.data=np.take_along_axis(msm1.data,idxai,axis=0)
    msm2.data=np.take_along_axis(msm2.data,idxai,axis=0)
    psm1.data=np.take_along_axis(psm1.data,idxai,axis=0)
    psm2.data=np.take_along_axis(psm2.data,idxai,axis=0)

    # remap to lat x lon
    ai=remap(ai,gr)
    msm1=remap(msm1,gr)
    msm2=remap(msm2,gr)
    psm1=remap(psm1,gr)
    psm2=remap(psm2,gr)

    # mask greenland and antarctica
    aagl=pickle.load(open('/project/amp/miyawaki/data/share/aa_gl/cesm2/aa_gl.pickle','rb'))
    ai=ai*aagl
    msm1=msm1*aagl
    msm2=msm2*aagl
    psm1=psm1*aagl
    psm2=psm2*aagl

    mai=np.nanmean(ai,axis=0) # annual mean

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')
    awgt=np.cos(np.deg2rad(mlat)) # area weight

    # subselect by latitude and/or high seasonal cycle
    if filt:
        ah=np.nan*np.ones_like(mlat)
        ah[mddpvn>fmax]=1
    else:
        ah=np.ones_like(mlat)
    if re=='tr':
        ah[np.abs(gr['lat'])>tlat]=np.nan
    elif re=='ml':
        ah[np.logical_or(np.abs(gr['lat'])<=tlat,np.abs(gr['lat'])>plat)]=np.nan
    elif re=='hl':
        ah[np.abs(gr['lat'])<=plat]=np.nan
    elif re=='et':
        ah[np.abs(gr['lat'])<=tlat]=np.nan

    # make sure nans are consistent accross T and SM
    nidx=np.logical_or(np.isnan(psm2),np.logical_or(np.isnan(psm1),np.logical_or(np.isnan(msm2),np.isnan(msm1))))
    msm1[nidx]=np.nan
    msm2[nidx]=np.nan
    psm1[nidx]=np.nan
    psm2[nidx]=np.nan

    # select region of interest
    ahmsm1,kidx=regsla(msm1,gr,ah)
    ahmsm2,_=regsla(msm2,gr,ah)
    ahpsm1,_=regsla(psm1,gr,ah)
    ahpsm2,_=regsla(psm2,gr,ah)
    ahw=regsl2d(awgt,ah,kidx)
    mai=regsl2d(mai,ah,kidx)

    # bin into 12
    def binned(vn,w,idx,nb):
        bvn=np.empty([12,nb])
        for im in range(12):
            bvn[im,:]=[np.nansum(w[idx==i]*vn[im,idx==i])/np.nansum(w[idx==i]) for i in 1+np.arange(nb)]
        return bvn
    mai[np.isinf(mai)]=np.nan
    pai=np.nanpercentile(mai,lpai)
    iai=np.digitize(mai,pai)
    nb=np.nanmax(iai)-1

    ahmsm1=binned(ahmsm1,ahw,iai,nb)
    ahmsm2=binned(ahmsm2,ahw,iai,nb)
    ahpsm1=binned(ahpsm1,ahw,iai,nb)
    ahpsm2=binned(ahpsm2,ahw,iai,nb)

    # categorize according to pathways
    def categorize(msm1,msm2,psm1,psm2):
        c=np.nan*np.ones_like(msm1)
        # H_his=WL & H_fut=WL & M_his=EL & M_fut=EL
        c[np.logical_and(psm2<0,np.logical_and(psm1<0,np.logical_and(msm1>0,msm2>0)))]=0
        # H_his=WL & H_fut=WL & M_his=EL & M_fut=WL
        c[np.logical_and(psm2<0,np.logical_and(psm1<0,np.logical_and(msm1>0,msm2<0)))]=1
        # H_his=WL & H_fut=WL & M_his=WL & M_fut=WL
        c[np.logical_and(psm2<0,np.logical_and(psm1<0,np.logical_and(msm1<0,msm2<0)))]=2
        # H_his=WL & H_fut=WL & M_his=EL & M_fut=WL
        c[np.logical_and(psm2<0,np.logical_and(psm1<0,np.logical_and(msm1<0,msm2>0)))]=3
        # H_his=EL & H_fut=WL & M_his=EL & M_fut=WL
        c[np.logical_and(psm2<0,np.logical_and(psm1>0,np.logical_and(msm1>0,msm2<0)))]=4
        # H_his=EL & H_fut=EL & M_his=EL & M_fut=EL
        c[np.logical_and(psm2>0,np.logical_and(psm1>0,np.logical_and(msm1>0,msm2>0)))]=5
        # H_his=EL & H_fut=WL & M_his=EL & M_fut=EL
        c[np.logical_and(psm2<0,np.logical_and(psm1>0,np.logical_and(msm1>0,msm2>0)))]=6
        # H_his=EL & H_fut=EL & M_his=EL & M_fut=WL
        c[np.logical_and(psm2>0,np.logical_and(psm1>0,np.logical_and(msm1>0,msm2<0)))]=7
        # H_his=EL & H_fut=EL & M_his=EL & M_fut=EL
        c[np.logical_and(psm2>0,np.logical_and(psm1>0,np.logical_and(msm1<0,msm2>0)))]=8
        # H_his=EL & H_fut=EL & M_his=EL & M_fut=EL
        c[np.logical_and(psm2>0,np.logical_and(psm1>0,np.logical_and(msm1<0,msm2<0)))]=9
        return c

    ahc=categorize(ahmsm1,ahmsm2,ahpsm1,ahpsm2)

    # save et regimes data
    varnc='cat'
    s=ahc.shape
    xahc=xr.DataArray(ahc,coords={'month':1+np.arange(s[0]),'bin':1+np.arange(s[1])},dims=('month','bin'))
    ddir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn1)
    if not os.path.exists(ddir): os.makedirs(ddir)
    xahc.to_netcdf('%s/ai.%s.%s_%s.%s.nc'%(ddir,varnc,his,fut,re))

    mon=range(12)
    [ahmmon,ahmbin] = np.meshgrid(mon,mppai, indexing='ij')

    # plot gp vs seasonal cycle of varnc PCOLORMESH
    fig,ax=plt.subplots(figsize=(2,2),constrained_layout=True)
    clf=ax.pcolormesh(ahmmon,ahmbin,ahc,vmin=0,vmax=vmax(varnc),cmap='Pastel1')
    ax.text(0.5,1.05,tstr,ha='center',va='center',transform=ax.transAxes)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xticks(np.arange(1,11+2,2))
    ax.set_xticklabels(np.arange(2,12+2,2))
    ax.set_yticks(100*np.arange(0,1+0.2,0.2))
    # ax.set_xticklabels(np.arange(2,12+2,2))
    ax.set_ylabel('AI Percentile')
    ax.set_xlim([-0.5,11.5])
    ax.set_ylim([0,100])
    fig.savefig('%s/ai.ddp%02d%s.%s%s.ah.ai.%s.png' % (odir1,p,varnc,fo,fstr,re), format='png', dpi=600)
    fig.savefig('%s/ai.ddp%02d%s.%s%s.ah.ai.%s.pdf' % (odir1,p,varnc,fo,fstr,re), format='pdf', dpi=600)

    # save colorbar only
    fig,ax=plt.subplots(figsize=(1,2))
    cb=plt.colorbar(clf,ax=ax,location='right')
    ax.remove()
    plt.savefig('%s/ai.ddp%02d%s.%s.ah.ai.cb.pdf' % (odir1,p,varnc,fo), format='pdf', dpi=600,bbox_inches='tight')

[plot(re) for re in tqdm(lre)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lre)) as p:
#         p.map(plot,lre)
