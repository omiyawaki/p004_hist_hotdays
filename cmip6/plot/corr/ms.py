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
fut='2080-2100'
ann=True

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
if ann:
    r=pickle.load(open('%s/ms.rsq.%s_%s_%s.%s.ann.pickle' % (idir,varn,his,fut,se), 'rb'))	
else:
    r=pickle.load(open('%s/ms.rsq.%s_%s_%s.%s.pickle' % (idir,varn,his,fut,se), 'rb'))	

# remap to lat x lon
if ann:
    llr=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llr[lmi]=r.data
    llr=np.reshape(llr,(gr['lat'].size,gr['lon'].size))
else:
    llr=np.nan*np.ones([r.shape[0],gr['lat'].size*gr['lon'].size])
    llr[:,lmi]=r.data
    llr=np.reshape(llr,(r.shape[0],gr['lat'].size,gr['lon'].size))

# repeat 0 deg lon info to 360 deg to prevent a blank line in contour
gr['lon'] = np.append(gr['lon'].data,360)
if ann:
    llr=np.append(llr,llr[...,0][...,None],axis=1)
else:
    llr=np.append(llr,llr[...,0][...,None],axis=2)
rsq=llr**2
print(np.nanmax(rsq))

[mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

########### PLOT
if ann:
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(4,3),constrained_layout=True)
    ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
    clf=ax.contourf(mlon, mlat, rsq, np.arange(0,1+0.1,0.1), vmax=1, vmin=0, transform=ccrs.PlateCarree())
    ax.coastlines()
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=12)
    cb.set_label(label=r'$R^2(\Delta\delta %s^{%02d}, \Delta\delta %s^{%02d})$'%(varnlb(varn1),p,varnlb(varn2),p),size=12)
    fig.savefig('%s/ms.rsq%02d%s.%s.ann.png' % (odir,p,varn,fo), format='png', dpi=300)

    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(4,3),constrained_layout=True)
    ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
    clf=ax.contourf(mlon, mlat, llr, np.arange(-1,1+0.1,0.1), vmax=1, vmin=-1, transform=ccrs.PlateCarree(),cmap='RdBu_r')
    ax.coastlines()
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=12)
    cb.set_label(label=r'$R(\Delta\delta %s^{%02d}, \Delta\delta %s^{%02d})$'%(varnlb(varn1),p,varnlb(varn2),p),size=12)
    fig.savefig('%s/ms.r%02d%s.%s.ann.png' % (odir,p,varn,fo), format='png', dpi=300)

else:
    # plot rsq (pct warming - mean warming)
    fig,ax=plt.subplots(nrows=3,ncols=4,subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(12,7),constrained_layout=True)
    ax=ax.flatten()
    fig.suptitle(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
    for m in tqdm(range(12)):
        clf=ax[m].contourf(mlon, mlat, rsq[m,...], np.arange(0,1+0.1,0.1), vmax=1, vmin=0, transform=ccrs.PlateCarree())
        ax[m].coastlines()
        ax[m].set_title(r'%s' % (monname(m)),fontsize=16)
        fig.savefig('%s/ms.rsq%02d%s.%s.png' % (odir,p,varn,fo), format='png', dpi=300)
    cb=fig.colorbar(clf,ax=ax.ravel().tolist(),location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$R^2(\Delta\delta %s^{%02d}, \Delta\delta %s^{%02d})$'%(varnlb(varn1),p,varnlb(varn2),p),size=16)
    fig.savefig('%s/ms.rsq%02d%s.%s.png' % (odir,p,varn,fo), format='png', dpi=300)
