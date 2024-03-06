import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm
from util import mods
from utils import monname,varnlb,unitlb

nt=7 # window size in days
tlat=30 # latitude bound for tropics
p=95
varn='tas'
varn1='hfls'
varn2='ooplh_fixbc'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'
skip507599=True

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

addpvn = np.max(ddpvn,axis=0)-np.min(ddpvn,axis=0)
mddpvn = np.max(ddpvn,axis=0)

[mlat,mlon] = np.meshgrid(pgr['lat'], pgr['lon'], indexing='ij')

# identify region of high seasonal cycle in tropics
ah=np.nan*np.ones_like(mlat)
ah[mddpvn>0.5]=1
ah[np.abs(pgr['lat'])>tlat]=np.nan
nh=ah.copy()
nh[pgr['lat']<0]=np.nan
sh=ah.copy()
sh[pgr['lat']>=0]=np.nan

# compute averaged seasonal cycle of x
w=np.cos(np.deg2rad(mlat))

nhddp=regav(ddpvn,nh,w)
nhddp1=regav(ddpvn1,nh,w)
shddp=regav(ddpvn,sh,w)
shddp1=regav(ddpvn1,sh,w)
ahddp=regava(ddpvn,pgr,ah,w)
ahddp1=regava(ddpvn1,pgr,ah,w)

# stdev
snhddp =regsp(ddpvn,nh,w,       nhddp)
snhddp1=regsp(ddpvn1,nh,w,     nhddp1)
sshddp =regsp(ddpvn,sh,w,       shddp)
sshddp1=regsp(ddpvn1,sh,w,     shddp1)
sahddp =regspa(ddpvn,pgr,ah,w,  ahddp)
sahddp1=regspa(ddpvn1,pgr,ah,w,ahddp1)

# plot seasonal cycle of varn1
mon=np.arange(1,13,1)
fig,ax=plt.subplots(figsize=(3,3),constrained_layout=True)
ax.fill_between(mon,nhddp1-snhddp1,nhddp1+snhddp1,color='k',alpha=0.2,edgecolor=None)
ax.plot(mon,nhddp1,'k')
ax.set_xticks(mon)
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_ylabel('$\Delta\delta %s^{%02d}$ (%s)'%(varnlb(varn1),p,unitlb(varn1)))
ax.set_ylim(vmin(varn1))
fig.savefig('%s/sc.ddp%02d%s.%s.nh.png' % (odir1,p,varn1,fo), format='png', dpi=600)

# plot seasonal cycle of varn1
mon=np.arange(1,13,1)
fig,ax=plt.subplots(figsize=(3,3),constrained_layout=True)
ax.fill_between(mon,shddp1-sshddp1,shddp1+sshddp1,color='k',alpha=0.2,edgecolor=None)
ax.plot(mon,shddp1,'k')
ax.set_xticks(mon)
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_ylabel('$\Delta\delta %s^{%02d}$ (%s)'%(varnlb(varn1),p,unitlb(varn1)))
ax.set_ylim(vmin(varn1))
fig.savefig('%s/sc.ddp%02d%s.%s.sh.png' % (odir1,p,varn1,fo), format='png', dpi=600)

# plot seasonal cycle of varn1
mon=np.arange(1,13,1)
fig,ax=plt.subplots(figsize=(3,3),constrained_layout=True)
ax.fill_between(mon,ahddp1-sahddp1,ahddp1+sahddp1,color='k',alpha=0.2,edgecolor=None)
ax.plot(mon,ahddp1,'k')
ax.set_xticks(mon)
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_ylabel('$\Delta\delta %s^{%02d}$ (%s)'%(varnlb(varn1),p,unitlb(varn1)))
ax.set_ylim(vmin(varn1))
fig.savefig('%s/sc.ddp%02d%s.%s.ah.png' % (odir1,p,varn1,fo), format='png', dpi=600)

