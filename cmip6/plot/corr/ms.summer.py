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
p=95
pref1='ddp'
varn1='tas'
pref2='ddp'
varn2='hfss'
varn='%s%s+%s%s'%(pref1,varn1,pref2,varn2)
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'

md='mi'

# load land indices
lmi,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,'mmm',varn1)
odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)

print(odir)
if not os.path.exists(odir):
    os.makedirs(odir)

# correlation
[jja,djf]=pickle.load(open('%s/ms.rsq.%s_%s_%s.%s.summer.pickle' % (idir,varn,his,fut,se), 'rb'))	

# remap to lat x lon
lljja=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
lljja[lmi]=jja.data
lljja=np.reshape(lljja,(gr['lat'].size,gr['lon'].size))
lldjf=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
lldjf[lmi]=djf.data
lldjf=np.reshape(lldjf,(gr['lat'].size,gr['lon'].size))

# nh jja sh djf
inh=gr['lat']>0
ish=gr['lat']<=0
llr=np.empty_like(lljja)
llr[inh,:]=lljja[inh,:]
llr[ish,:]=lldjf[ish,:]

# global mean
clat=np.cos(np.deg2rad(gr['lat']))
clat=np.tile(clat,(llr.shape[1],1))
clat=np.transpose(clat,[1,0])
clat[np.isnan(llr)]=np.nan
gmllr=np.nansum(clat*llr)/np.nansum(clat)
print('Global mean R=%g'%gmllr)

# tropical mean
clat[np.abs(gr['lat'])>20]=np.nan
tmllr=np.nansum(clat*llr)/np.nansum(clat)
print('Tropical mean R=%g'%tmllr)

# repeat 0 deg lon info to 360 deg to prevent a blank line in contour
gr['lon'] = np.append(gr['lon'].data,360)
llr=np.append(llr,llr[...,0][...,None],axis=1)
rsq=llr**2

[mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

########### PLOT
fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(4,3),constrained_layout=True)
ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
clf=ax.contourf(mlon, mlat, rsq, np.arange(0,1+0.1,0.1), vmax=1, vmin=0, transform=ccrs.PlateCarree())
ax.coastlines()
cb=fig.colorbar(clf,location='bottom',aspect=50)
cb.ax.tick_params(labelsize=12)
cb.set_label(label=r'$R^2(\Delta\delta %s^{%02d}, \Delta\delta %s^{%02d})$'%(varnlb(varn1),p,varnlb(varn2),p),size=12)
fig.savefig('%s/ms.rsq%02d%s.%s.summer.%s.png' % (odir,p,varn,fo,fut), format='png', dpi=600)

fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(4,3),constrained_layout=True)
ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
clf=ax.contourf(mlon, mlat, llr, np.arange(-1,1+0.1,0.1), vmax=1, vmin=-1, transform=ccrs.PlateCarree(),cmap='RdBu_r')
ax.coastlines()
cb=fig.colorbar(clf,location='bottom',aspect=50)
cb.ax.tick_params(labelsize=12)
cb.set_label(label=r'$R(\Delta\delta %s^{%02d}, \Delta\delta %s^{%02d})$'%(varnlb(varn1),p,varnlb(varn2),p),size=12)
fig.savefig('%s/ms.r%02d%s.%s.summer.%s.png' % (odir,p,varn,fo,fut), format='png', dpi=600)
