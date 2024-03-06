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
from cmip6util import mods
from utils import monname

nt=7 # window size in days
varn='predtm'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'
skip5075=True

md='mmm'

idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

if not os.path.exists(odir):
    os.makedirs(odir)

# warming
dvn,sdvn,gr=pickle.load(open('%s/d%s_%s_%s.%s.nc' % (idir,varn,his,fut,se), 'rb'))	
dpvn,sdpvn,_=pickle.load(open('%s/dp%s_%s_%s.%s.nc' % (idir,varn,his,fut,se), 'rb'))	
ddpvn,sddpvn,_=pickle.load(open('%s/ddp%s_%s_%s.%s.nc' % (idir,varn,his,fut,se), 'rb'))	
# repeat 0 deg lon info to 360 deg to prevent a blank line in contour
gr['lon'] = np.append(gr['lon'].data,360)
dvn = np.append(dvn, dvn[...,0][...,None],axis=2)
dpvn = np.append(dpvn, dpvn[...,0][...,None],axis=3)
ddpvn = np.append(ddpvn, ddpvn[...,0][...,None],axis=3)
sdvn = np.append(sdvn, sdvn[...,0][...,None],axis=2)
sdpvn = np.append(sdpvn, sdpvn[...,0][...,None],axis=3)
sddpvn = np.append(sddpvn, sddpvn[...,0][...,None],axis=3)

[mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

# plot pct warming - mean warming
for i,p in enumerate(gr['pct']):
    if skip5075 and (p==50 or p==75):
        continue
    fig,ax=plt.subplots(nrows=3,ncols=4,subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(12,7),constrained_layout=True)
    ax=ax.flatten()
    fig.suptitle(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
    for m in tqdm(range(12)):
        clf=ax[m].contourf(mlon, mlat, ddpvn[m,i,...], np.arange(-3.0,3.0+0.1,0.1),extend='both', vmax=3.0, vmin=-3.0, transform=ccrs.PlateCarree(), cmap='RdBu_r')
        ax[m].coastlines()
        ax[m].set_title(r'%s' % (monname(m)),fontsize=16)
        fig.savefig('%s/dp%02d%s.%s.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)
    cb=fig.colorbar(clf,ax=ax.ravel().tolist(),location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\Delta \delta T^{%02d}_{q=0}$ (K)'%(p),size=16)
    fig.savefig('%s/dp%02d%s.%s.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)

# plot STD(pct warming - mean warming)
for i,p in enumerate(gr['pct']):
    if skip5075 and (p==50 or p==75):
        continue
    fig,ax=plt.subplots(nrows=3,ncols=4,subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(12,7),constrained_layout=True)
    ax=ax.flatten()
    fig.suptitle(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
    for m in tqdm(range(12)):
        clf=ax[m].contourf(mlon, mlat, sddpvn[m,i,...], np.arange(0,3.0+0.1,0.1),extend='both', vmax=3.0, vmin=0, transform=ccrs.PlateCarree())
        ax[m].coastlines()
        ax[m].set_title(r'%s' % (monname(m)),fontsize=16)
        fig.savefig('%s/sdp%02d%s.%s.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)
    cb=fig.colorbar(clf,ax=ax.ravel().tolist(),location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\sigma(\Delta \delta T^{%02d}_{q=0})$ (K)'%(p),size=16)
    fig.savefig('%s/sdp%02d%s.%s.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)

# plot pct warming
for i,p in enumerate(gr['pct']):
    if skip5075 and (p==50 or p==75):
        continue
    fig,ax=plt.subplots(nrows=3,ncols=4,subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(12,7),constrained_layout=True)
    ax=ax.flatten()
    fig.suptitle(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
    for m in tqdm(range(12)):
        clf=ax[m].contourf(mlon, mlat, dpvn[m,i,...], np.arange(-10,10+1,1),extend='both', vmax=10, vmin=-10, transform=ccrs.PlateCarree(), cmap='RdBu_r')
        ax[m].coastlines()
        ax[m].set_title(r'%s' % (monname(m)),fontsize=16)
        fig.savefig('%s/p%02d%s.%s.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)
    cb=fig.colorbar(clf,ax=ax.ravel().tolist(),location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\Delta T^{%02d}_{q=0}$ (K)'%(p),size=16)
    fig.savefig('%s/p%02d%s.%s.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)

# plot mean warming
fig,ax=plt.subplots(nrows=3,ncols=4,subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(12,7),constrained_layout=True)
ax=ax.flatten()
fig.suptitle(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
for m in tqdm(range(12)):
    clf=ax[m].contourf(mlon, mlat, dvn[m,...], np.arange(-10,10+1,1),extend='both', vmax=10, vmin=-10, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    ax[m].coastlines()
    ax[m].set_title(r'%s' % (monname(m)),fontsize=16)
    fig.savefig('%s/m%s.%s.pdf' % (odir,varn,fo), format='pdf', dpi=300)
cb=fig.colorbar(clf,ax=ax.ravel().tolist(),location='bottom',aspect=50)
cb.ax.tick_params(labelsize=16)
cb.set_label(label=r'$\Delta \overline{T}_{q=0}$ (K)',size=16)
fig.savefig('%s/m%s.%s.pdf' % (odir,varn,fo), format='pdf', dpi=300)
