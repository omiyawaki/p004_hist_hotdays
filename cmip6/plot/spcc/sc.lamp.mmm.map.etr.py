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
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator
from cartopy.mpl.ticker import LatitudeFormatter
from scipy.stats import linregress
from tqdm import tqdm
from util import mods
from utils import monname,varnlb,unitlb

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

# categorize according to pathways
def categorize(msm1,msm2,psm1,psm2):
    c=np.nan*np.ones_like(msm1)
    # H_his=WL & H_fut=WL & M_his=EL & M_fut=EL
    c[np.logical_and(psm2<0,np.logical_and(psm1<0,np.logical_and(msm1>0,msm2>0)))]=0
    # H_his=EL & H_fut=WL & M_his=EL & M_fut=EL
    c[np.logical_and(psm2<0,np.logical_and(psm1>0,np.logical_and(msm1>0,msm2>0)))]=1
    # H_his=WL & H_fut=WL & M_his=WL & M_fut=WL
    c[np.logical_and(psm2<0,np.logical_and(psm1<0,np.logical_and(msm1<0,msm2<0)))]=2
    # H_his=EL & H_fut=EL & M_his=EL & M_fut=EL
    c[np.logical_and(psm2>0,np.logical_and(psm1>0,np.logical_and(msm1>0,msm2>0)))]=3
    # H_his=WL & H_fut=WL & M_his=EL & M_fut=WL
    c[np.logical_and(psm2<0,np.logical_and(psm1<0,np.logical_and(msm1>0,msm2<0)))]=4
    # H_his=EL & H_fut=WL & M_his=EL & M_fut=WL
    c[np.logical_and(psm2>0,np.logical_and(psm1<0,np.logical_and(msm1>0,msm2<0)))]=5
    return c

cat=categorize(msm1,msm2,psm1,psm2)

varnc='cat'

for i in range(12):
    # plot gp vs seasonal cycle of varnc
    fig,ax=plt.subplots(subplot_kw={'projection':ccrs.Robinson(central_longitude=240)},figsize=(5,3),constrained_layout=True)
    clf=ax.pcolormesh(mlon,mlat,cat[i,...],vmin=0,vmax=vmax(varnc),transform=ccrs.PlateCarree(),cmap='Pastel1')
    ax.coastlines()
    ax.set_extent((-180,180,-30,30),crs=ccrs.PlateCarree())
    gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.5,color='gray',y_inline=False)
    gl.ylocator=mticker.FixedLocator([-50,-30,0,30,50])
    gl.yformatter=LatitudeFormatter()
    gl.xlines=False
    gl.left_labels=False
    gl.bottom_labels=False
    gl.right_labels=True
    gl.top_labels=False
    # cb=fig.colorbar(clf,location='bottom')
    fig.savefig('%s/sc.%s.%s.map.%02d.png' % (odir1,varnc,fo,i+1), format='png', dpi=600)
