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
lvn=['tas']
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'
# fut='gwl2.0'
dpi=600
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

def vmaxdd(vn):
    lvm={   'tas':  [1,0.1],
            'hfls': [30,1],
            'plh': [30,1],
            'plh_fixbc': [30,1],
            'ooplh': [30,1],
            'ooplh_orig': [30,1],
            'ooplh_fixbc': [30,1],
            }
    return lvm[vn]

def vmaxd(vn):
    lvm={   'tas':  [2,0.1],
            'hfls': [30,2],
            'plh': [30,2],
            'plh_fixbc': [30,2],
            'ooplh': [30,2],
            'ooplh_orig': [30,2],
            'ooplh_fixbc': [30,2],
            }
    return lvm[vn]

def varmean(data,axis):
    return np.sqrt(np.nanmean(data**2,axis=axis))

def plot(vn):
    vmdd,dvmdd=vmaxdd(vn)
    vmd,dvmd=vmaxd(vn)

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # warming
    dvn=xr.open_dataset('%s/std.d.%s_%s_%s.%s.nc' % (idir,vn,his,fut,se))['__xarray_dataarray_variable__']
    try:
        dpvn=xr.open_dataset('%s/std.dpc.%s_%s_%s.%s.nc' % (idir,vn,his,fut,se))[vn]
    except:
        dpvn=xr.open_dataset('%s/std.dpc.%s_%s_%s.%s.nc' % (idir,vn,his,fut,se))['ooplh']
    ds=xr.open_dataset('%s/std.ddpc.%s_%s_%s.%s.nc' % (idir,vn,his,fut,se))
    try:
        ddpvn=ds[vn]
    except:
        ddpvn=ds['ooplh']
    pct=ds['percentile']
    gpi=ds['gpi']
    dpvn=dpvn.sel(percentile=pct==p).squeeze()
    ddpvn=ddpvn.sel(percentile=pct==p).squeeze()

    # jja and djf means
    dvnj  =varmean(dvn.data[5:8,:]  ,axis=0)
    dpvnj =varmean(dpvn.data[5:8,:] ,axis=0)
    ddpvnj=varmean(ddpvn.data[5:8,:],axis=0)
    dvnd  =varmean(np.roll(dvn.data  ,1,axis=0)[:3,:],axis=0)
    dpvnd =varmean(np.roll(dpvn.data ,1,axis=0)[:3,:],axis=0)
    ddpvnd=varmean(np.roll(ddpvn.data,1,axis=0)[:3,:],axis=0)

    # remap to lat x lon
    lldvnj=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    lldpvnj=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llddpvnj=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    lldvnj[lmi]=dvnj.data
    lldpvnj[lmi]=dpvnj.data
    llddpvnj[lmi]=ddpvnj.data
    lldvnj=np.reshape(lldvnj,(gr['lat'].size,gr['lon'].size))
    lldpvnj=np.reshape(lldpvnj,(gr['lat'].size,gr['lon'].size))
    llddpvnj=np.reshape(llddpvnj,(gr['lat'].size,gr['lon'].size))

    lldvnd=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    lldpvnd=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llddpvnd=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    lldvnd[lmi]=dvnd.data
    lldpvnd[lmi]=dpvnd.data
    llddpvnd[lmi]=ddpvnd.data
    lldvnd=np.reshape(lldvnd,(gr['lat'].size,gr['lon'].size))
    lldpvnd=np.reshape(lldpvnd,(gr['lat'].size,gr['lon'].size))
    llddpvnd=np.reshape(llddpvnd,(gr['lat'].size,gr['lon'].size))

    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    lldvnj = np.append(lldvnj, lldvnj[...,0][...,None],axis=1)
    lldpvnj = np.append(lldpvnj, lldpvnj[...,0][...,None],axis=1)
    llddpvnj = np.append(llddpvnj, llddpvnj[...,0][...,None],axis=1)
    lldvnd = np.append(lldvnd, lldvnd[...,0][...,None],axis=1)
    lldpvnd = np.append(lldpvnd, lldpvnd[...,0][...,None],axis=1)
    llddpvnd = np.append(llddpvnd, llddpvnd[...,0][...,None],axis=1)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # use jja for nh, djf for sh
    lldvn=np.copy(lldvnd)
    lldpvn=np.copy(lldpvnd)
    llddpvn=np.copy(llddpvnd)
    lldvn[gr['lat']>0]=lldvnj[gr['lat']>0]
    lldpvn[gr['lat']>0]=lldpvnj[gr['lat']>0]
    llddpvn[gr['lat']>0]=llddpvnj[gr['lat']>0]

    # plot pct warming - mean warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, llddpvn, np.arange(0,vmdd+dvmdd,dvmdd),extend='both', vmax=vmdd, vmin=0, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(r'%s %s JJA+DJF' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\sigma(\Delta \delta %s^{%02d})$ (%s)'%(varnlb(vn),p,unitlb(vn)),size=16)
    fig.savefig('%s/summer.std.ddp%02d%s.%s.%s.png' % (odir,p,vn,fo,fut), format='png', dpi=dpi)

    # plot pct warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, lldpvn, np.arange(0,vmd+dvmd,dvmd),extend='both', vmax=vmd, vmin=0, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(r'%s %s JJA+DJF' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=12)
    cb.set_label(label=r'$\sigma(\Delta %s^{%02d})$ (%s)'%(varnlb(vn),p,unitlb(vn)),size=16)
    fig.savefig('%s/summer.std.dp%02d%s.%s.%s.png' % (odir,p,vn,fo,fut), format='png', dpi=dpi)

    # plot mean warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, lldvn, np.arange(0,vmd+dvmd,dvmd),extend='both', vmax=vmd, vmin=0, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(r'%s %s JJA+DJF' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=12)
    cb.set_label(label=r'$\sigma(\Delta \overline{%s})$ (%s)'%(varnlb(vn),unitlb(vn)),size=16)
    fig.savefig('%s/summer.std.d%s.%s.%s.png' % (odir,vn,fo,fut), format='png', dpi=dpi)

# run
[plot(vn) for vn in lvn]
