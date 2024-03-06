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
varn='mrsos'
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
ds=xr.open_dataset('%s/ddp.%s_%s_%s.%s.nc' % (idir,varn,his,fut,se))
ddpvn=ds[varn]
pct=ds['percentile']
gpi=ds['gpi']
mean=ddpvn.sel(percentile=pct==0).squeeze()
hotd=ddpvn.sel(percentile=pct==p).squeeze()

# remap to lat x lon
llmean=np.nan*np.ones([mean.shape[0],gr['lat'].size*gr['lon'].size])
llhotd=np.nan*np.ones([hotd.shape[0],gr['lat'].size*gr['lon'].size])
llmean[:,lmi]=mean.data
llhotd[:,lmi]=hotd.data
llmean=np.reshape(llmean,(mean.shape[0],gr['lat'].size,gr['lon'].size))
llhotd=np.reshape(llhotd,(hotd.shape[0],gr['lat'].size,gr['lon'].size))

# repeat 0 deg lon info to 360 deg to prevent a blank line in contour
gr['lon'] = np.append(gr['lon'].data,360)
llmean = np.append(llmean, llmean[...,0][...,None],axis=2)
llhotd = np.append(llhotd, llhotd[...,0][...,None],axis=2)

[mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

# plot pct warming - mean warming
fig,ax=plt.subplots(nrows=3,ncols=4,subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(12,7),constrained_layout=True)
ax=ax.flatten()
fig.suptitle(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
for m in tqdm(range(12)):
    if m in [0,1,2,3,11]:
        clf=ax[m].contourf(mlon, mlat, llhotd[m,...]-llmean[m,...], np.arange(-1.5,1.5+0.1,0.1+1e-8),extend='both', vmax=1.5, vmin=-1.5, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    else:
        clf=ax[m].contourf(mlon, mlat, llhotd[m,...]-llmean[m,...], np.arange(-1.5,1.5+0.1,0.1),extend='both', vmax=1.5, vmin=-1.5, transform=ccrs.PlateCarree(), cmap='RdBu_r')
        if m==10:
            cb=fig.colorbar(clf,ax=ax.ravel().tolist(),location='bottom',aspect=50)
            cb.ax.tick_params(labelsize=16)
            cb.set_label(label=r'$\Delta \delta SM^{%02d}_\mathrm{2\,m}$ (kg m$^{-2}$)'%(p),size=16)
    ax[m].coastlines()
    ax[m].set_title(r'%s' % (monname(m)),fontsize=16)
    fig.savefig('%s/ddp%02d%s.%s.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)
fig.savefig('%s/ddp%02d%s.%s.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)
