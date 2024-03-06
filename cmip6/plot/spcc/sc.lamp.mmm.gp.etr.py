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
filt=False # only look at gridpoints with max exceeding value below
fmax=0.5
title=True

bi=0.025
lb=np.arange(bi,1+bi,bi) # area bins
p=95 # percentile
varn='tas'
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
        return v[:,~np.isnan(v).any(axis=0)]

    def regsla(v,gr,ma):
        sv=np.roll(v,6,axis=0) # seasonality shifted by 6 months
        v[:,gr['lat']<0,:]=sv[:,gr['lat']<0,:]
        return regsl(v,ma)

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

    # warming
    ds=xr.open_dataset('%s/ddpc.%s_%s_%s.%s.nc' % (idir,varn,his,fut,se))
    ddpvn=ds[varn]
    pct=ds['percentile']
    gpi=ds['gpi']
    ddpvn=ddpvn.sel(percentile=pct==p).squeeze()

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

    # remap to lat x lon
    ddpvn=remap(ddpvn,gr)
    msm1=remap(msm1,gr)
    msm2=remap(msm2,gr)
    psm1=remap(psm1,gr)
    psm2=remap(psm2,gr)

    addpvn = np.max(ddpvn,axis=0)-np.min(ddpvn,axis=0)
    mddpvn = np.max(ddpvn,axis=0)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')
    awgt=np.cos(np.deg2rad(mlat)) # area weight
    awgt=np.tile(awgt,(12,1,1))

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
    nh=ah.copy()
    nh[gr['lat']<=0]=np.nan
    sh=ah.copy()
    sh[gr['lat']>0]=np.nan

    # make sure nans are consistent accross T and SM
    nidx=np.logical_or(np.isnan(psm2),np.logical_or(np.isnan(psm1),np.logical_or(np.isnan(msm2),np.logical_or(np.isnan(ddpvn),np.isnan(msm1)))))
    ddpvn[nidx]=np.nan
    msm1[nidx]=np.nan
    msm2[nidx]=np.nan
    psm1[nidx]=np.nan
    psm2[nidx]=np.nan

    # select region of interest
    nhmsm1=regsl(msm1,nh)
    shmsm1=regsl(msm1,sh)
    ahmsm1=regsla(msm1,gr,ah)
    nhmsm2=regsl(msm2,nh)
    shmsm2=regsl(msm2,sh)
    ahmsm2=regsla(msm2,gr,ah)
    nhpsm1=regsl(psm1,nh)
    shpsm1=regsl(psm1,sh)
    ahpsm1=regsla(psm1,gr,ah)
    nhpsm2=regsl(psm2,nh)
    shpsm2=regsl(psm2,sh)
    ahpsm2=regsla(psm2,gr,ah)

    # convert sh to months since winter solstice
    shmsm1=np.roll(shmsm1,6,axis=0)
    shmsm2=np.roll(shmsm2,6,axis=0)
    shpsm1=np.roll(shpsm1,6,axis=0)
    shpsm2=np.roll(shpsm2,6,axis=0)

    # repeat above for tas if varn1 is not tas
    nhddp=regsl(ddpvn,nh)
    shddp=regsl(ddpvn,sh)
    ahddp=regsla(ddpvn,gr,ah)
    shddp=np.roll(shddp,6,axis=0)
    # get index when sorted by month of maximum ddptas response
    nhddp,nhidx=sortmax(nhddp)
    shddp,shidx=sortmax(shddp)
    ahddp,ahidx=sortmax(ahddp)
    # sort
    nhmsm1=nhmsm1[:,nhidx]
    shmsm1=shmsm1[:,shidx]
    ahmsm1=ahmsm1[:,ahidx]
    nhmsm2=nhmsm2[:,nhidx]
    shmsm2=shmsm2[:,shidx]
    ahmsm2=ahmsm2[:,ahidx]
    nhpsm1=nhpsm1[:,nhidx]
    shpsm1=shpsm1[:,shidx]
    ahpsm1=ahpsm1[:,ahidx]
    nhpsm2=nhpsm2[:,nhidx]
    shpsm2=shpsm2[:,shidx]
    ahpsm2=ahpsm2[:,ahidx]
    # resorted idx
    nhidxrs=np.argmax(nhddp,axis=0)
    shidxrs=np.argmax(shddp,axis=0)
    ahidxrs=np.argmax(ahddp,axis=0)

    # make weights and reorder
    nhw=regsl(awgt,nh)
    shw=regsl(awgt,sh)
    ahw=regsla(awgt,gr,ah)
    nhw=nhw[:,nhidx]
    shw=shw[:,shidx]
    ahw=ahw[:,ahidx]

    # cumulatively add weights
    nhcw=np.flip(np.cumsum(nhw,axis=1),axis=1)/np.sum(nhw,axis=1,keepdims=True)
    shcw=np.flip(np.cumsum(shw,axis=1),axis=1)/np.sum(shw,axis=1,keepdims=True)
    ahcw=np.flip(np.cumsum(ahw,axis=1),axis=1)/np.sum(ahw,axis=1,keepdims=True)

    # bin into 12
    def binned(vn,w,idx):
        bvn=np.empty([12,12])
        bw=np.zeros(13)
        for im in range(12):
            bvn[im,:]=[np.sum(w[im,idx==i]*vn[im,idx==i])/np.sum(w[im,idx==i]) for i in range(12)]
            bw[im+1]=np.sum(w[im,idx==im])+bw[im]
        return bvn,bw/bw[-1]
    nhmsm1,nhcw=binned(nhmsm1,nhw,nhidxrs)
    shmsm1,shcw=binned(shmsm1,shw,shidxrs)
    ahmsm1,ahcw=binned(ahmsm1,ahw,ahidxrs)
    nhmsm2,_=binned(nhmsm2,nhw,nhidxrs)
    shmsm2,_=binned(shmsm2,shw,shidxrs)
    ahmsm2,_=binned(ahmsm2,ahw,ahidxrs)
    nhpsm1,_=binned(nhpsm1,nhw,nhidxrs)
    shpsm1,_=binned(shpsm1,shw,shidxrs)
    ahpsm1,_=binned(ahpsm1,ahw,ahidxrs)
    nhpsm2,_=binned(nhpsm2,nhw,nhidxrs)
    shpsm2,_=binned(shpsm2,shw,shidxrs)
    ahpsm2,_=binned(ahpsm2,ahw,ahidxrs)
    # repeat for DdT
    nhddp,nhcw=binned(nhddp,nhw,nhidxrs)
    shddp,shcw=binned(shddp,shw,shidxrs)
    ahddp,ahcw=binned(ahddp,ahw,ahidxrs)

    # convert from edge to mp values
    nhcwmp=1/2*(nhcw[1:]+nhcw[:-1])
    shcwmp=1/2*(shcw[1:]+shcw[:-1])
    ahcwmp=1/2*(ahcw[1:]+ahcw[:-1])

    # categorize according to pathways
    def categorize(msm1,msm2,psm1,psm2):
        c=np.nan*np.ones_like(msm1)
        # H_his=WL & H_fut=WL & M_his=EL & M_fut=EL
        c[np.logical_and(psm2<0,np.logical_and(psm1<0,np.logical_and(msm1>0,msm2>0)))]=0
        # H_his=WL & H_fut=WL & M_his=EL & M_fut=WL
        c[np.logical_and(psm2<0,np.logical_and(psm1<0,np.logical_and(msm1>0,msm2<0)))]=1
        # H_his=WL & H_fut=WL & M_his=WL & M_fut=WL
        c[np.logical_and(psm2<0,np.logical_and(psm1<0,np.logical_and(msm1<0,msm2<0)))]=2
        # H_his=EL & H_fut=WL & M_his=EL & M_fut=WL
        c[np.logical_and(psm2<0,np.logical_and(psm1>0,np.logical_and(msm1>0,msm2<0)))]=3
        # H_his=EL & H_fut=EL & M_his=EL & M_fut=EL
        c[np.logical_and(psm2>0,np.logical_and(psm1>0,np.logical_and(msm1>0,msm2>0)))]=4
        # H_his=EL & H_fut=WL & M_his=EL & M_fut=EL
        c[np.logical_and(psm2<0,np.logical_and(psm1>0,np.logical_and(msm1>0,msm2>0)))]=5
        # H_his=EL & H_fut=EL & M_his=EL & M_fut=WL
        c[np.logical_and(psm2>0,np.logical_and(psm1>0,np.logical_and(msm1>0,msm2<0)))]=6
        return c

    nhc=categorize(nhmsm1,nhmsm2,nhpsm1,nhpsm2)
    shc=categorize(shmsm1,shmsm2,shpsm1,shpsm2)
    ahc=categorize(ahmsm1,ahmsm2,ahpsm1,ahpsm2)

    mon=range(12)
    [nhmbin,nhmmon] = np.meshgrid(nhcwmp,mon, indexing='ij')
    [shmbin,shmmon] = np.meshgrid(shcwmp,mon, indexing='ij')
    [ahmbin,ahmmon] = np.meshgrid(ahcwmp,mon, indexing='ij')

    varnc='cat'

    # plot gp vs seasonal cycle of varnc PCOLORMESH
    fig,ax=plt.subplots(figsize=(2,2),constrained_layout=True)
    clf=ax.pcolormesh(ahmmon,ahmbin,np.transpose(ahc),vmin=0,vmax=vmax(varnc),cmap='Pastel1')
    ax.set_title(tstr)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xticks(np.arange(1,11+2,2))
    ax.set_xticklabels(np.arange(2,12+2,2))
    ax.set_yticks(np.arange(0,1+0.2,0.2))
    ax.set_ylabel('Cumulative area fraction')
    ax.set_xlim([-0.5,11.5])
    ax.set_ylim([0,1])
    ax.set_ylim(ax.get_ylim()[::-1])
    x=np.arange(12)
    y=ahcwmp
    z=np.diag(np.diag(np.ones_like(ahc)))
    # make dense grid
    scale=100
    yy=ndimage.zoom(y,scale,order=0)
    zz=ndimage.zoom(z,scale,order=0)
    xx=np.linspace(x.min(),x.max(),zz.shape[0])
    # extend out to edge
    xx=np.insert(xx,0,-0.5)
    yy=np.insert(yy,0,0)
    zz=np.insert(zz,0,zz[0,:],axis=0)
    zz=np.insert(zz,0,zz[:,0],axis=1)
    xx=np.append(xx,12.5)
    yy=np.append(yy,1)
    zz=np.insert(zz,-1,zz[-1,:],axis=0)
    zz=np.insert(zz,-1,zz[:,-1],axis=1)
    ax.contour(xx,yy,zz,levels=[0.5],colors='k',linewidths=0.5,corner_mask=False)
    fig.savefig('%s/sc.ddp%02d%s.%s%s.ah.gp.%s.png' % (odir1,p,varnc,fo,fstr,re), format='png', dpi=600)
    fig.savefig('%s/sc.ddp%02d%s.%s%s.ah.gp.%s.pdf' % (odir1,p,varnc,fo,fstr,re), format='pdf', dpi=600)

    # save colorbar only
    fig,ax=plt.subplots(figsize=(1,2))
    cb=plt.colorbar(clf,ax=ax,location='right')
    ax.remove()
    plt.savefig('%s/sc.ddp%02d%s.%s.ah.gp.cb.pdf' % (odir1,p,varnc,fo), format='pdf', dpi=600,bbox_inches='tight')

[plot(re) for re in tqdm(lre)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lre)) as p:
#         p.map(plot,lre)
