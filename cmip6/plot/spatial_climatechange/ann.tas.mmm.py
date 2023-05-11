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
p=95
varn='tas'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'
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

idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)

if not os.path.exists(odir):
    os.makedirs(odir)

# warming
dvn=xr.open_dataset('%s/d.%s_%s_%s.%s.nc' % (idir,varn,his,fut,se))['__xarray_dataarray_variable__']
dpvn=xr.open_dataset('%s/dp.%s_%s_%s.%s.nc' % (idir,varn,his,fut,se))[varn]
ds=xr.open_dataset('%s/ddp.%s_%s_%s.%s.nc' % (idir,varn,his,fut,se))
ddpvn=ds[varn]
pct=ds['percentile']
gpi=ds['gpi']
dpvn=dpvn.sel(percentile=pct==p).squeeze()
ddpvn=ddpvn.sel(percentile=pct==p).squeeze()

# annual mean
dvn=dvn.mean(dim='month')
dpvn=dpvn.mean(dim='month')
ddpvn=ddpvn.mean(dim='month')

# remap to lat x lon
lldvn=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
lldpvn=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
llddpvn=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
lldvn[lmi]=dvn.data
lldpvn[lmi]=dpvn.data
llddpvn[lmi]=ddpvn.data
lldvn=np.reshape(lldvn,(gr['lat'].size,gr['lon'].size))
lldpvn=np.reshape(lldpvn,(gr['lat'].size,gr['lon'].size))
llddpvn=np.reshape(llddpvn,(gr['lat'].size,gr['lon'].size))

# repeat 0 deg lon info to 360 deg to prevent a blank line in contour
gr['lon'] = np.append(gr['lon'].data,360)
lldvn = np.append(lldvn, lldvn[...,0][...,None],axis=1)
lldpvn = np.append(lldpvn, lldpvn[...,0][...,None],axis=1)
llddpvn = np.append(llddpvn, llddpvn[...,0][...,None],axis=1)

[mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

# plot pct warming - mean warming
fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
clf=ax.contourf(mlon, mlat, llddpvn, np.arange(-1.5,1.5+0.1,0.1),extend='both', vmax=1.5, vmin=-1.5, transform=ccrs.PlateCarree(), cmap='RdBu_r')
ax.coastlines()
ax.set_title(r'%s %s ANN' % (md.upper(),fo.upper()),fontsize=16)
cb=fig.colorbar(clf,location='bottom',aspect=50)
cb.ax.tick_params(labelsize=16)
cb.set_label(label=r'$\Delta \delta T^{%02d}$ (K)'%(p),size=16)
fig.savefig('%s/ann.ddp%02d%s.%s.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)

# plot pct warming
fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
clf=ax.contourf(mlon, mlat, lldpvn, np.arange(-8,8+0.25,0.25),extend='both', vmax=8, vmin=-8, transform=ccrs.PlateCarree(), cmap='RdBu_r')
ax.coastlines()
ax.set_title(r'%s %s ANN' % (md.upper(),fo.upper()),fontsize=16)
cb=fig.colorbar(clf,location='bottom',aspect=50)
cb.ax.tick_params(labelsize=12)
cb.set_label(label=r'$\Delta T^{%02d}$ (K)'%(p),size=16)
fig.savefig('%s/ann.dp%02d%s.%s.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)

# plot mean warming
fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
clf=ax.contourf(mlon, mlat, lldvn, np.arange(-8,8+0.25,0.25),extend='both', vmax=8, vmin=-8, transform=ccrs.PlateCarree(), cmap='RdBu_r')
ax.coastlines()
ax.set_title(r'%s %s ANN' % (md.upper(),fo.upper()),fontsize=16)
cb=fig.colorbar(clf,location='bottom',aspect=50)
cb.ax.tick_params(labelsize=12)
cb.set_label(label=r'$\Delta \overline{T}$ (K)',size=16)
fig.savefig('%s/ann.d%s.%s.pdf' % (odir,varn,fo), format='pdf', dpi=300)
