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
from matplotlib.ticker import MultipleLocator
from scipy.stats import linregress
from tqdm import tqdm
from util import mods
from utils import monname,varnlb,unitlb

re='tr' # tr=tropics, ml=midlatitudes, hl=high lat, et=extratropics
tlat=30 # latitude bound for tropics
plat=50 # midlatitude bound
filt=False # only look at gridpoints with max exceeding value below
fmax=0.5
title=True # show title string?

p=97.5 # percentile
varn='tas'
varn1='hfls'
varnp='hfls'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'
skip507599=True

# md='mmm'
md='CESM2'

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

def vmin(vn):
    if re=='tr':
        d={ 'hfls':     [-1,0],
            'hfss':    [-1,8],
            'ooplh':    [-8,1],
            'ooplh_fixbc':    [-8,1],
            'ooplh_fixmsm':    [-8,1],
            'tas':      [-0.5,0.5],
                }
        return d[vn]
    elif re=='et':
        d={ 'hfls':     [-8,1],
            'tas':      [-1.5,1.5],
                }
        return d[vn]

def remap(v,gr):
    llv=np.nan*np.ones([12,gr['lat'].size*gr['lon'].size])
    llv[:,lmi]=v.data
    llv=np.reshape(llv,(12,gr['lat'].size,gr['lon'].size))
    return llv

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
ddpvn1=ds[varnp]
pct=ds['percentile']
gpi=ds['gpi']
ddpvn1=ddpvn1.sel(percentile=pct==p).squeeze()

print(ddpvn.shape)
print(ddpvn1.shape)

# remap to lat x lon
ddpvn=remap(ddpvn,gr)
ddpvn1=remap(ddpvn1,gr)

addpvn = np.max(ddpvn,axis=0)-np.min(ddpvn,axis=0)
mddpvn = np.max(ddpvn,axis=0)

[mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

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

# compute averaged seasonal cycle of x
w=np.cos(np.deg2rad(mlat))

nhddp=regav(ddpvn,nh,w)
nhddp1=regav(ddpvn1,nh,w)
shddp=regav(ddpvn,sh,w)
shddp1=regav(ddpvn1,sh,w)
ahddp=regava(ddpvn,gr,ah,w)
ahddp1=regava(ddpvn1,gr,ah,w)

# stdev
snhddp =regsp(ddpvn,nh,w,       nhddp)
snhddp1=regsp(ddpvn1,nh,w,     nhddp1)
sshddp =regsp(ddpvn,sh,w,       shddp)
sshddp1=regsp(ddpvn1,sh,w,     shddp1)
sahddp =regspa(ddpvn,gr,ah,w,  ahddp)
sahddp1=regspa(ddpvn1,gr,ah,w,ahddp1)

# # convert sh to months since winter solstice
# nhddp1=np.roll(nhddp1,1,axis=0)
# snhddp1=np.roll(snhddp1,1,axis=0)
# shddp1=np.roll(shddp1,7,axis=0)
# sshddp1=np.roll(sshddp1,7,axis=0)
# ahddp1=np.roll(ahddp1,1,axis=0)
# sahddp1=np.roll(sahddp1,1,axis=0)

# plot seasonal cycle of varn1
mon=np.arange(1,13,1)
fig,ax=plt.subplots(figsize=(3,3),constrained_layout=True)
ax.fill_between(mon,nhddp1-snhddp1,nhddp1+snhddp1,color='k',alpha=0.2,edgecolor=None)
ax.axhline(0,color='k',linewidth=0.5)
ax.plot(mon,nhddp1,'k')
if title: ax.set_title(tstr)
ax.set_xticks(mon)
# ax.set_xticklabels(['D','J','F','M','A','M','J','J','A','S','O','N'])
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_ylabel('$\Delta\delta %s^{%02d}$ (%s)'%(varnlb(varn1),p,unitlb(varn1)))
ax.set_ylim(vmin(varn1))
fig.savefig('%s/sc.ddp%02d%s.%s%s.nh.av.%s.png' % (odir1,p,varn1,fo,fstr,re), format='png', dpi=600)

# plot seasonal cycle of varn1
mon=np.arange(1,13,1)
fig,ax=plt.subplots(figsize=(3,3),constrained_layout=True)
ax.fill_between(mon,shddp1-sshddp1,shddp1+sshddp1,color='k',alpha=0.2,edgecolor=None)
ax.axhline(0,color='k',linewidth=0.5)
ax.plot(mon,shddp1,'k')
if title: ax.set_title(tstr)
ax.set_xticks(mon)
# ax.set_xticklabels(['J','J','A','S','O','N','D','J','F','M','A','M'])
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_ylabel('$\Delta\delta %s^{%02d}$ (%s)'%(varnlb(varn1),p,unitlb(varn1)))
ax.set_ylim(vmin(varn1))
fig.savefig('%s/sc.ddp%02d%s.%s%s.sh.av.%s.png' % (odir1,p,varn1,fo,fstr,re), format='png', dpi=600)

# # plot seasonal cycle of varn1
# mon=np.arange(1,13,1)
# fig,ax=plt.subplots(figsize=(2,2),constrained_layout=True)
# ax.fill_between(mon,ahddp1-sahddp1,ahddp1+sahddp1,color='k',alpha=0.2,edgecolor=None)
# if title: ax.set_title(tstr)
# ax.axhline(0,color='k',linewidth=0.5)
# ax.plot(mon,ahddp1,'k')
# ax.set_xticks(np.arange(2,12+2,2))
# ax.set_xticklabels(np.arange(2,12+2,2))
# # ax.set_xlabel('Months since winter solstice')
# ax.set_ylabel('$\Delta\delta %s^{%02d}$ (%s)'%(varnlb(varn1),p,unitlb(varn1)))
# ax.xaxis.set_minor_locator(MultipleLocator(1))
# ax.yaxis.set_minor_locator(MultipleLocator(0.1))
# # ax.set_ylim(vmin(varn1))
# fig.savefig('%s/sc.ddp%02d%s.%s%s.ah.av.%s.png' % (odir1,p,varn1,fo,fstr,re), format='png', dpi=600)
# fig.savefig('%s/sc.ddp%02d%s.%s%s.ah.av.%s.pdf' % (odir1,p,varn1,fo,fstr,re), format='pdf', dpi=600)

