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
varn1='ooplh_fixbc'
varn2='ooplh_fixmsm'
varn3='ooplh_rddsm'
styl1,styl2,styl3='--','-','-'
colo1,colo2,colo3='tab:blue','tab:purple','tab:orange'
varnc='%s+%s+%s'%(varn1,varn2,varn3)
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'
skip507599=True

md='mmm'
# md='CESM2'

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
idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s' % (se,fo,md)
odir1 = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varnc)
if not os.path.exists(odir1):
    os.makedirs(odir1)

def load_pvn(vn):
    return xr.open_dataarray('%s/%s/ddpc.%s_%s_%s.%s.nc' % (idir1,vn,vn,his,fut,se)).sel(percentile=pct==p).squeeze()

# load pvn
ddpvn1=load_pvn(varn1)
ddpvn2=load_pvn(varn2)
ddpvn3=load_pvn(varn3)

# remap to lat x lon
ddpvn=remap(ddpvn,gr)
ddpvn1=remap(ddpvn1,gr)
ddpvn2=remap(ddpvn2,gr)
ddpvn3=remap(ddpvn3,gr)

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

def hemiavg(vn):
    return regav(vn,nh,w),regav(vn,sh,w),regava(vn,gr,ah,w)

nhddp,shddp,ahddp=hemiavg(ddpvn)
nhddp1,shddp1,ahddp1=hemiavg(ddpvn1)
nhddp2,shddp2,ahddp2=hemiavg(ddpvn2)
nhddp3,shddp3,ahddp3=hemiavg(ddpvn3)

# stdev
def hemistd(vn,nhvn,shvn,ahvn):
    return regsp(vn,nh,w,nhvn),regsp(vn,sh,w,shvn),regspa(vn,gr,ah,w,ahvn)

snhddp,sshddp,sahddp=hemistd(ddpvn,nhddp,shddp,ahddp)
snhddp1,sshddp1,sahddp1=hemistd(ddpvn1,nhddp1,shddp1,ahddp1)
snhddp2,sshddp2,sahddp2=hemistd(ddpvn2,nhddp2,shddp2,ahddp2)
snhddp3,sshddp3,sahddp3=hemistd(ddpvn3,nhddp3,shddp3,ahddp3)

# plot seasonal cycle of varn1
mon=np.arange(1,13,1)
fig,ax=plt.subplots(figsize=(3,3),constrained_layout=True)
ax.axhline(0,color='k',linewidth=0.5)
ax.fill_between(mon,nhddp1-snhddp1,nhddp1+snhddp1,color='k',alpha=0.2,edgecolor=None)
ax.plot(mon,nhddp1,'k')
ax.fill_between(mon,nhddp2-snhddp2,nhddp2+snhddp2,color='tab:blue',alpha=0.2,edgecolor=None)
ax.plot(mon,nhddp2,'tab:blue')
ax.fill_between(mon,nhddp3-snhddp3,nhddp3+snhddp3,color='tab:blue',alpha=0.2,edgecolor=None)
ax.plot(mon,nhddp3,'--',color='tab:blue')
if title: ax.set_title(tstr)
ax.set_xticks(mon)
# ax.set_xticklabels(['D','J','F','M','A','M','J','J','A','S','O','N'])
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_ylabel('$\Delta\delta %s^{%02d}$ (%s)'%(varnlb(varn1),p,unitlb(varn1)))
ax.set_ylim(vmin(varn1))
fig.savefig('%s/sc.ddp%02d%s.%s%s.nh.av.%s.png' % (odir1,p,varnc,fo,fstr,re), format='png', dpi=600)

# plot seasonal cycle of varn1
mon=np.arange(1,13,1)
fig,ax=plt.subplots(figsize=(3,3),constrained_layout=True)
ax.axhline(0,color='k',linewidth=0.5)
ax.fill_between(mon,shddp1-sshddp1,shddp1+sshddp1,color='k',alpha=0.2,edgecolor=None)
ax.plot(mon,shddp1,'k')
ax.fill_between(mon,shddp2-sshddp2,shddp2+sshddp2,color='tab:blue',alpha=0.2,edgecolor=None)
ax.plot(mon,shddp2,'tab:blue')
ax.fill_between(mon,shddp3-sshddp3,shddp3+sshddp3,color='tab:blue',alpha=0.2,edgecolor=None)
ax.plot(mon,shddp3,'--',color='tab:blue')
if title: ax.set_title(tstr)
ax.set_xticks(mon)
# ax.set_xticklabels(['J','J','A','S','O','N','D','J','F','M','A','M'])
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_ylabel('$\Delta\delta %s^{%02d}$ (%s)'%(varnlb(varn1),p,unitlb(varn1)))
ax.set_ylim(vmin(varn1))
fig.savefig('%s/sc.ddp%02d%s.%s%s.sh.av.%s.png' % (odir1,p,varnc,fo,fstr,re), format='png', dpi=600)

# plot seasonal cycle of varn1
mon=np.arange(1,13,1)
fig,ax=plt.subplots(figsize=(3,3),constrained_layout=True)
if title: ax.set_title(tstr)
ax.axhline(0,color='k',linewidth=0.5)
ax.fill_between(mon,ahddp1-sahddp1,ahddp1+sahddp1,color=colo1,alpha=0.2,edgecolor=None)
ax.plot(mon,ahddp1,linestyle=styl1,color=colo1)
ax.fill_between(mon,ahddp2-sahddp2,ahddp2+sahddp2,color=colo2,alpha=0.2,edgecolor=None)
ax.plot(mon,ahddp2,linestyle=styl2,color=colo2)
ax.fill_between(mon,ahddp3-sahddp3,ahddp3+sahddp3,color=colo3,alpha=0.2,edgecolor=None)
ax.plot(mon,ahddp3,linestyle=styl3,color=colo3)
ax.set_xticks(np.arange(2,12+2,2))
ax.set_xticklabels(np.arange(2,12+2,2))
# ax.set_xlabel('Months since winter solstice')
ax.set_ylabel('$\Delta\delta %s^{%02d}$ (%s)'%(varnlb(varn1),p,unitlb(varn1)))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
# ax.set_ylim(vmin(varn1))
fig.savefig('%s/sc.ddp%02d%s.%s%s.ah.av.%s.png' % (odir1,p,varnc,fo,fstr,re), format='png', dpi=600)
fig.savefig('%s/sc.ddp%02d%s.%s%s.ah.av.%s.pdf' % (odir1,p,varnc,fo,fstr,re), format='pdf', dpi=600)

