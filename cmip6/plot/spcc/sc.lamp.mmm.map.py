import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
import seaborn as sns
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LatitudeFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cmasher as cmr
from scipy.stats import linregress
from tqdm import tqdm
from util import mods
from utils import monname,varnlb,unitlb

p=97.5
varn='tas'
varn1='tas'
tr=True
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'
skip507599=True
dd0=0.5

lcolors=sns.color_palette('RdBu_r',3)
ccustom=mcolors.LinearSegmentedColormap.from_list('custom',lcolors)
lcolors=np.vstack((sns.color_palette('RdBu_r',7),np.flip(sns.color_palette('RdBu_r',7),axis=0)[1:-1,:]))
BuRdBu=mcolors.LinearSegmentedColormap.from_list('custom',lcolors)

# limited range modified colormaps
lrseasons=cmr.get_sub_cmap('cmr.seasons',0,0.9)
lrcopper=cmr.get_sub_cmap('cmr.copper',0.15,0.85)

md='mmm'

# load land indices
lmi,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

def vmin(vn):
    d={ 'hfls':     [-8,1],
        'tas':      [-0.05,0.5],
            }
    return d[vn]

def remap(v,gr):
    llv=np.nan*np.ones([12,gr['lat'].size*gr['lon'].size])
    llv[:,lmi]=v.data
    llv=np.reshape(llv,(12,gr['lat'].size,gr['lon'].size))
    return llv

def replo(v):
    return np.append(v, v[...,0][...,None],axis=2)

def regav(v,ma,w):
    return np.nansum(w*ma*v,axis=(1,2))/np.nansum(w*ma)

def regsp(v,ma,w,vav):
    return 1.96*np.sqrt(np.nansum(w*ma*(v-vav[:,None,None])**2,axis=(1,2))/np.nansum(w*ma))/np.sqrt(np.sum(~np.isnan(ma)))

def regava(v,gr,ma,w):
    sv=np.roll(v,6,axis=0) # seasonality shifted by 6 months
    v[:,gr['lat']<0,:]=sv[:,gr['lat']<0,:]
    return np.nansum(w*ma*v,axis=(1,2))/np.nansum(w*ma)

def regspa(v,gr,ma,w,vav):
    sv=np.roll(v,6,axis=0) # seasonality shifted by 6 months
    v[:,gr['lat']<0,:]=sv[:,gr['lat']<0,:]
    return 1.96*np.sqrt(np.nansum(w*ma*(v-vav[:,None,None])**2,axis=(1,2))/np.nansum(w*ma))/np.sqrt(np.sum(~np.isnan(ma)))

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

ds=xr.open_dataset('%s/ddpc.%s_%s_%s.%s.nc' % (idir1,varn1,his,fut,se))
ddpvn1=ds[varn1]
pct=ds['percentile']
gpi=ds['gpi']
ddpvn1=ddpvn1.sel(percentile=pct==p).squeeze()

# remap to lat x lon
llddpvn=remap(ddpvn,gr)
llddpvn1=remap(ddpvn1,gr)

# repeat 0 deg lon info to 360 deg to prevent a blank line in contour
pgr=gr.copy()
pgr['lon'] = np.append(pgr['lon'].data,360)
ddpvn=replo(llddpvn)
ddpvn1=replo(llddpvn1)

amddpvn=np.array(np.argmax(ddpvn,axis=0),dtype='float') # month of maximum
amddpvn[np.isnan(ddpvn).all(axis=0)]=np.nan

# convert sh to months since winter solstice
ammsws=np.copy(amddpvn)
ammsws[gr['lat']<0]=np.mod(amddpvn[gr['lat']<0]+6,12)

# seasons version
ams=np.copy(ammsws)
w=np.isin(ams,[0,1,11]) # winter months
e=np.isin(ams,[2,3,4,8,9,10]) # equinox months
s=np.isin(ams,[5,6,7]) # summer months
ams[w]=0
ams[e]=1
ams[s]=2

# where maximum exceeds threshold
wkmax=np.nanmax(ddpvn,axis=0)<dd0
wkmax[np.isnan(ddpvn).all(axis=0)]=True
ma_amddpvn=np.ma.masked_where(wkmax,amddpvn)
ma_ammsws=np.ma.masked_where(wkmax,ammsws)
ma_ams=np.ma.masked_where(wkmax,ams)

[mlat,mlon] = np.meshgrid(pgr['lat'], pgr['lon'], indexing='ij')

# plot seasonal cycle of varn1
fig,ax=plt.subplots(subplot_kw={'projection':ccrs.Robinson(central_longitude=240)},figsize=(4,2.5),constrained_layout=True)
clf=ax.pcolormesh(mlon,mlat,ammsws,vmin=0,vmax=11,transform=ccrs.PlateCarree(),cmap=lrseasons)
ax.coastlines()
cb=fig.colorbar(clf,location='bottom',aspect=50,boundaries=np.arange(-0.5,12.5))
cb.set_ticks(np.arange(0,12,1))
cb.set_ticklabels(1+np.arange(0,12,1))
cb.set_label('Months since winter solstice')
gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.5,color='gray',y_inline=False)
gl.ylocator=mticker.FixedLocator([-50,-30,0,30,50])
gl.yformatter=LatitudeFormatter()
gl.xlines=False
gl.left_labels=False
gl.bottom_labels=False
gl.right_labels=True
gl.top_labels=False
fig.savefig('%s/maxmon.ddp%02d%s.%s.msws.png' % (odir1,p,varn1,fo), format='png', dpi=600)
fig.savefig('%s/maxmon.ddp%02d%s.%s.msws.pdf' % (odir1,p,varn1,fo), format='pdf', dpi=600)

# # plot seasonal cycle of varn1
# fig,ax=plt.subplots(subplot_kw={'projection':ccrs.Robinson(central_longitude=240)},figsize=(5,3),constrained_layout=True)
# clf=ax.pcolormesh(mlon,mlat,amddpvn,vmin=0,vmax=11,transform=ccrs.PlateCarree(),cmap=lrseasons)
# ax.fill_between([0,360],-90,90,transform=ccrs.PlateCarree(),hatch='///',alpha=0)
# clf=ax.pcolormesh(mlon,mlat,ma_amddpvn,vmin=0,vmax=12,transform=ccrs.PlateCarree(),cmap=lrseasons)
# ax.coastlines()
# cb=fig.colorbar(clf,location='bottom',aspect=50,boundaries=np.arange(-0.5,12.5))
# cb.set_ticks(np.arange(0,12,1))
# cb.set_ticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
# fig.savefig('%s/maxmon.ddp%02d%s.%s.png' % (odir1,p,varn1,fo), format='png', dpi=600)

if tr:
    # plot seasonal cycle of varn1
    fig,ax=plt.subplots(subplot_kw={'projection':ccrs.Robinson(central_longitude=240)},figsize=(5,3),constrained_layout=True)
    clf=ax.pcolormesh(mlon,mlat,ams,vmin=0,vmax=2,transform=ccrs.PlateCarree(),cmap=ccustom)
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
    cb=fig.colorbar(clf,location='bottom',aspect=50,boundaries=np.arange(-0.5,3.5))
    cb.set_ticks(np.arange(3))
    cb.set_ticklabels(['Winter','Equinox','Summer'])
    fig.savefig('%s/maxmon.ddp%02d%s.%s.seasons.tr.png' % (odir1,p,varn1,fo), format='png', dpi=600)

# plot seasonal cycle of varn1
fig,ax=plt.subplots(subplot_kw={'projection':ccrs.Robinson(central_longitude=240)},figsize=(5,3),constrained_layout=True)
clf=ax.pcolormesh(mlon,mlat,ams,vmin=0,vmax=2,transform=ccrs.PlateCarree(),cmap=ccustom)
ax.coastlines()
cb=fig.colorbar(clf,location='bottom',aspect=50,boundaries=np.arange(-0.5,3.5))
cb.set_ticks(np.arange(3))
cb.set_ticklabels(['Winter','Equinox','Summer'])
fig.savefig('%s/maxmon.ddp%02d%s.%s.seasons.png' % (odir1,p,varn1,fo), format='png', dpi=600)

