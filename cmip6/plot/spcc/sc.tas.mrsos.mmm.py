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
from utils import monname

nt=7 # window size in days
tlat=30 # latitude bound for tropics
p=95
varn='tas'
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

def remap(v,gr):
    llv=np.nan*np.ones([12,gr['lat'].size*gr['lon'].size])
    llv[:,lmi]=v.data
    llv=np.reshape(llv,(12,gr['lat'].size,gr['lon'].size))
    return llv

def replo(v):
    return np.append(v, v[...,0][...,None],axis=2)

def regav(v,ma,w):
    return np.nansum(w*ma*v,axis=(1,2))/np.nansum(w*ma)

def regava(v,gr,ma,w):
    sv=np.roll(v,6,axis=0) # seasonality shifted by 6 months
    v[:,gr['lat']<0,:]=sv[:,gr['lat']<0,:]
    return np.nansum(w*ma*v,axis=(1,2))/np.nansum(w*ma)

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

# sm anomaly from crit
idir0 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo1,md,'mrsos')
odir0 = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo1,md,'mrsos')
if not os.path.exists(odir0):
    os.makedirs(odir0)

msm1=xr.open_dataarray('%s/anom.m.%s_%s.%s.nc' % (idir0,'mrsos',his,se))
psm1=xr.open_dataarray('%s/anom.pc.%s_%s.%s.nc' % (idir0,'mrsos',his,se))
psm1=psm1.sel(percentile=p).squeeze()

# # precip
# idir0='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo1,md,'pr')
# ds=xr.open_dataset('%s/m.%s_%s.%s.nc' % (idir0,'pr',his,se))
# ddpvn=ds['pr']
# pct=ds['percentile']
# gpi=ds['gpi']
# ddpvn=ddpvn.sel(percentile=pct==p).squeeze()

# remap to lat x lon
llddpvn=remap(ddpvn,gr)
llmsm1=remap(msm1,gr)
llpsm1=remap(psm1,gr)

# repeat 0 deg lon info to 360 deg to prevent a blank line in contour
pgr=gr.copy()
pgr['lon'] = np.append(pgr['lon'].data,360)
ddpvn=replo(llddpvn)
msm1=replo(llmsm1)
psm1=replo(llpsm1)

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
nhmsm1=regav(msm1,nh,w)
nhpsm1=regav(psm1,nh,w)

shddp=regav(ddpvn,sh,w)
shmsm1=regav(msm1,sh,w)
shpsm1=regav(psm1,sh,w)

ahddp=regava(ddpvn,pgr,ah,w)
ahmsm1=regava(msm1,pgr,ah,w)
ahpsm1=regava(psm1,pgr,ah,w)

# plot seasonal cycle of etr
fig,ax=plt.subplots(figsize=(3,3),constrained_layout=True)
ax.axhline(0,color='k',linestyle='--')
ax.plot(ahmsm1,'k')
ax.plot(ahpsm1,color='tab:red')
ax.set_xticks(range(len(ahddp)))
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_ylabel('$SM-SM_\mathrm{crit}$ (kg m$^{-2}$)')
# ax.set_ylim([-0.05,0.5])
fig.savefig('%s/sc.%s.%s.ah.png' % (odir0,'dsm',fo1), format='png', dpi=600)

# fig,ax=plt.subplots(figsize=(3,3),constrained_layout=True)
# ax.axhline(0)
# ax.plot(nhmsm1,'k')
# ax.plot(nhpsm1,color='tab:red')
# ax.set_xticks(range(len(nhddp)))
# ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
# ax.set_ylabel('$SM-SM_\mathrm{crit}$')
# # ax.set_ylim([-0.05,0.5])
# fig.savefig('%s/sc.%s.%s.nh.png' % (odir0,'dsm',fo1), format='png', dpi=600)

# fig,ax=plt.subplots(figsize=(3,3),constrained_layout=True)
# ax.axhline(0)
# ax.plot(shmsm1,'k')
# ax.plot(shpsm1,color='tab:red')
# ax.set_xticks(range(len(shddp)))
# ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
# ax.set_ylabel('$SM-SM_\mathrm{crit}$')
# # ax.set_ylim([-0.05,0.5])
# fig.savefig('%s/sc.%s.%s.sh.png' % (odir0,'dsm',fo1), format='png', dpi=600)

# plot seasonal cycle of warming
fig,ax=plt.subplots(figsize=(3,3),constrained_layout=True)
ax.plot(ahddp,'k')
ax.set_xticks(range(len(ahddp)))
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_ylabel('$\Delta\delta T^{%02d}$'%(p))
ax.set_ylim([-0.05,0.5])
fig.savefig('%s/sc.ddp%02d%s.%s.ah.png' % (odir,p,varn,fo), format='png', dpi=600)

fig,ax=plt.subplots(figsize=(3,3),constrained_layout=True)
ax.plot(nhddp,'k')
ax.set_xticks(range(len(nhddp)))
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_ylabel('$\Delta\delta T^{%02d}$'%(p))
ax.set_ylim([-0.05,0.5])
fig.savefig('%s/sc.ddp%02d%s.%s.nh.png' % (odir,p,varn,fo), format='png', dpi=600)

fig,ax=plt.subplots(figsize=(3,3),constrained_layout=True)
ax.plot(np.roll(shddp,6),'k')
ax.set_xticks(range(len(shddp)))
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_ylabel('$\Delta\delta T^{%02d}$'%(p))
ax.set_ylim([-0.05,0.5])
fig.savefig('%s/sc.ddp%02d%s.%s.sh.png' % (odir,p,varn,fo), format='png', dpi=600)

sys.exit()

# (NH) plot pct warming - mean warming
fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
clf=ax.contourf(mlon, mlat, mddpvn, np.arange(0,1+0.1,0.1),extend='both', vmax=1, vmin=0, transform=ccrs.PlateCarree(),cmap='RdBu_r')
ax.contourf(mlon, mlat, nh,np.arange(0,1+0.1,1),vmax=1,vmin=0, transform=ccrs.PlateCarree(),cmap='Reds')
ax.contourf(mlon, mlat, sh,np.arange(0,1+0.1,1),vmax=1,vmin=0, transform=ccrs.PlateCarree(),cmap='Blues')
ax.coastlines()
cb=fig.colorbar(clf,location='bottom',aspect=50)
cb.ax.tick_params(labelsize=12)
cb.set_label(label=r'Seasonal maximum of $\Delta \delta T^{%02d}_\mathrm{2\,m}$ (K)'%(p),size=16)
fig.savefig('%s/max.ddp%02d%s.%s.nhsh.png' % (odir,p,varn,fo), format='png', dpi=300)

