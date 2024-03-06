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
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LatitudeFormatter
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
# fut='2080-2100'
fut='gwl2.0'
skip507599=True
vm=1.5

# md='mmm'
md='CESM2'

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
dvn=xr.open_dataarray('%s/d.%s_%s_%s.%s.nc' % (idir,varn,his,fut,se))
dpvn=xr.open_dataarray('%s/dpc.%s_%s_%s.%s.nc' % (idir,varn,his,fut,se))
ds=xr.open_dataset('%s/ddpc.%s_%s_%s.%s.nc' % (idir,varn,his,fut,se))
ddpvn=ds[varn]
pct=ds['percentile']
gpi=ds['gpi']
dpvn=dpvn.sel(percentile=pct==p).squeeze()
ddpvn=ddpvn.sel(percentile=pct==p).squeeze()

# remap to lat x lon
lldvn=np.nan*np.ones([12,gr['lat'].size*gr['lon'].size])
lldpvn=np.nan*np.ones([12,gr['lat'].size*gr['lon'].size])
llddpvn=np.nan*np.ones([12,gr['lat'].size*gr['lon'].size])
lldvn[:,lmi]=dvn.data
lldpvn[:,lmi]=dpvn.data
llddpvn[:,lmi]=ddpvn.data
lldvn=np.reshape(lldvn,(12,gr['lat'].size,gr['lon'].size))
lldpvn=np.reshape(lldpvn,(12,gr['lat'].size,gr['lon'].size))
llddpvn=np.reshape(llddpvn,(12,gr['lat'].size,gr['lon'].size))

# repeat 0 deg lon info to 360 deg to prevent a blank line in contour
gr['lon'] = np.append(gr['lon'].data,360)
dvn=lldvn
dpvn=lldpvn
ddpvn=llddpvn
dvn=np.append(dvn, dvn[...,0][...,None],axis=2)
dpvn=np.append(dpvn, dpvn[...,0][...,None],axis=2)
ddpvn=np.append(ddpvn, ddpvn[...,0][...,None],axis=2)

addpvn = np.max(ddpvn,axis=0)-np.min(ddpvn,axis=0)
mxddpvn = np.max(ddpvn,axis=0)
mnddpvn = np.min(ddpvn,axis=0)

[mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

# plot pct warming - mean warming
fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
ax.set_title(r'Seasonal Amplitude',fontsize=16)
clf=ax.contourf(mlon, mlat, addpvn, np.arange(0,vm+0.1,0.1),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(),cmap='RdBu_r')
ax.coastlines()
gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.5,color='gray',y_inline=False)
gl.ylocator=mticker.FixedLocator([-50,-30,0,30,50])
gl.yformatter=LatitudeFormatter()
gl.xlines=False
gl.left_labels=False
gl.bottom_labels=False
gl.right_labels=True
gl.top_labels=False
cb=fig.colorbar(clf,location='bottom',aspect=50)
cb.ax.tick_params(labelsize=12)
cb.set_label(label=r'Seasonal amplitude of $\Delta \delta T^{%02d}_\mathrm{2\,m}$ (K)'%(p),size=16)
fig.savefig('%s/amp.ddp%02d%s.%s.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)
fig.savefig('%s/amp.ddp%02d%s.%s.png' % (odir,p,varn,fo), format='png', dpi=300)

# MIN plot pct warming - mean warming
fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
# ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
ax.set_title(r'Annual minimum',fontsize=16)
clf=ax.contourf(mlon, mlat, mnddpvn, np.arange(-vm,vm+0.1,0.1),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(),cmap='RdBu_r')
ax.coastlines()
gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.5,color='gray',y_inline=False)
gl.ylocator=mticker.FixedLocator([-50,-30,0,30,50])
gl.yformatter=LatitudeFormatter()
gl.xlines=False
gl.left_labels=False
gl.bottom_labels=False
gl.right_labels=True
gl.top_labels=False
cb=fig.colorbar(clf,location='bottom',aspect=50)
cb.ax.tick_params(labelsize=12)
cb.set_label(label=r'$\Delta \delta T^{%02d}$ (K)'%(p),size=16)
fig.savefig('%s/min.ddp%02d%s.%s.png' % (odir,p,varn,fo), format='png', dpi=300)
fig.savefig('%s/min.ddp%02d%s.%s.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)

# MAX plot pct warming - mean warming
fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
# ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
ax.set_title(r'Annual maximum',fontsize=16)
clf=ax.contourf(mlon, mlat, mxddpvn, np.arange(-vm,vm+0.1,0.1),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(),cmap='RdBu_r')
ax.coastlines()
gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.5,color='gray',y_inline=False)
gl.ylocator=mticker.FixedLocator([-50,-30,0,30,50])
gl.yformatter=LatitudeFormatter()
gl.xlines=False
gl.left_labels=False
gl.bottom_labels=False
gl.right_labels=True
gl.top_labels=False
cb=fig.colorbar(clf,location='bottom',aspect=50)
cb.ax.tick_params(labelsize=12)
cb.set_label(label=r'$\Delta \delta T^{%02d}$ (K)'%(p),size=16)
fig.savefig('%s/max.ddp%02d%s.%s.png' % (odir,p,varn,fo), format='png', dpi=300)
fig.savefig('%s/max.ddp%02d%s.%s.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)

# MAX plot pct warming - mean warming
fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
# ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
ax.set_title(r'Annual maximum',fontsize=16)
clf=ax.contourf(mlon, mlat, mxddpvn, np.arange(-vm,vm+0.1,0.1),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(),cmap='RdBu_r')
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
cb=fig.colorbar(clf,location='bottom',aspect=50)
cb.ax.tick_params(labelsize=12)
cb.set_label(label=r'$\Delta \delta T^{%02d}$ (K)'%(p),size=16)
fig.savefig('%s/max.ddp%02d%s.%s.tr.png' % (odir,p,varn,fo), format='png', dpi=300)
fig.savefig('%s/max.ddp%02d%s.%s.tr.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)

