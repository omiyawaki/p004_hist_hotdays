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
varn='pr'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'

# lmd=mods(fo1)
lmd=['CanESM5']

for md in lmd:
    idirt = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,'tas')
    idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn)
    idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    c = 0
    dt={}

    # mean temp
    ds1=xr.open_dataset('%s/m%s_%s.%s.nc' % (idir1,varn,his,se))
    gr={}
    gr['lat']=ds1['lat']
    gr['lon']=ds1['lon']
    try:
        vn1=ds1[varn]
    except:
        vn1=ds1['__xarray_dataarray_variable__']
    vn1=vn1.groupby('time.month').mean('time') # monthly means
    ds2=xr.open_dataset('%s/m%s_%s.%s.nc' % (idir2,varn,fut,se))
    try:
        vn2=ds2[varn]
    except:
        vn2=ds2['__xarray_dataarray_variable__']
    vn2=vn2.groupby('time.month').mean('time') # monthly means
    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    evn1 = np.append(vn1.data, vn1.data[...,0][...,None],axis=2)
    evn2 = np.append(vn2.data, vn2.data[...,0][...,None],axis=2)

    # prc temp
    dst=xr.open_dataset('%s/wp%s%03d_%s.%s.nc' % (idirt,'tas',nt,his,se))
    pct=dst['percentile']
    ds1=xr.open_dataset('%s/mp%s%03d_%s.%s.nc' % (idir1,varn,nt,his,se))
    try:
        pvn1=ds1[varn]
    except:
        pvn1=ds1['__xarray_dataarray_variable__']
    pvn1=pvn1.groupby('time.month').mean('time') # monthly means
    ds2=xr.open_dataset('%s/mp%s%03d_%s.%s.nc' % (idir2,varn,nt,fut,se))
    try:
        pvn2=ds2[varn]
    except:
        pvn2=ds2['__xarray_dataarray_variable__']
    pvn2=pvn2.groupby('time.month').mean('time') # monthly means
    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    epvn1 = np.append(pvn1.data, pvn1.data[...,0][...,None],axis=3)
    epvn2 = np.append(pvn2.data, pvn2.data[...,0][...,None],axis=3)

    # warming
    dvn=evn2-evn1
    dpvn=epvn2-epvn1
    ddpvn=dpvn-np.transpose(dvn[...,None],[0,3,1,2])

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # plot pct warming - mean warming
    for i,p in enumerate(pct):
        fig,ax=plt.subplots(nrows=3,ncols=4,subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(12,7),constrained_layout=True)
        ax=ax.flatten()
        fig.suptitle(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
        for m in tqdm(range(12)):
            clf=ax[m].contourf(mlon, mlat, ddpvn[m,i,...], np.arange(-5,5+0.5,0.5),extend='both', vmax=5, vmin=-5, transform=ccrs.PlateCarree(), cmap='RdBu_r')
            ax[m].coastlines()
            ax[m].set_title(r'%s' % (monname(m)),fontsize=16)
            fig.savefig('%s/dp%02d%s.%s.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)
        cb=fig.colorbar(clf,ax=ax.ravel().tolist(),location='bottom',aspect=50)
        cb.ax.tick_params(labelsize=16)
        cb.set_label(label=r'$\Delta \delta P^{%02d}$ (mm d$^{-1}$)'%(p),size=16)
        fig.savefig('%s/dp%02d%s.%s.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)

    # plot pct warming
    for i,p in enumerate(pct):
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
        cb.set_label(label=r'$\Delta P^{%02d}$ (mm d$^{-1}$)'%(p),size=16)
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
    cb.set_label(label=r'$\Delta \overline{P}$ (mm d$^{-1}$)',size=16)
    fig.savefig('%s/m%s.%s.pdf' % (odir,varn,fo), format='pdf', dpi=300)
