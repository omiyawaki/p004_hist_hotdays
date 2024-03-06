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
from utils import monname,varnlb,unitlb

nt=7 # window size in days
p=97.5
lvn=['tas']
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'
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
    lvm={   'tas':  [1.0,0.1],
            'hfls': [10,1],
            'hfss': [10,1],
            'ooef2': [0.5,0.01],
            }
    return lvm[vn]

def vmaxd(vn):
    lvm={   'tas':  [4,0.25],
            'hfls': [30,2],
            'hfss': [30,2],
            'ooef2': [0.5,0.01],
            }
    return lvm[vn]

def plot(vn):
    vmdd,dvmdd=vmaxdd(vn)
    vmd,dvmd=vmaxd(vn)

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # warming
    dvn=xr.open_dataset('%s/d.%s_%s_%s.%s.nc' % (idir,vn,his,fut,se))['__xarray_dataarray_variable__']
    dpvn=xr.open_dataset('%s/dpc.%s_%s_%s.%s.nc' % (idir,vn,his,fut,se))[vn]
    ds=xr.open_dataset('%s/ddpc.%s_%s_%s.%s.nc' % (idir,vn,his,fut,se))
    ddpvn=ds[vn]
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

    # plot TROPICS ONLY
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    # ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
    ax.set_title(r'%s %s ANN' % (md.upper(),fo.upper()),fontsize=16)
    clf=ax.contourf(mlon, mlat, llddpvn, np.arange(-vmdd,vmdd+dvmdd,dvmdd),extend='both', vmax=vmdd, vmin=-vmdd, transform=ccrs.PlateCarree(),cmap='RdBu_r')
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
    cb.set_label(label=r'$\Delta \delta %s$ (%s)'%(varnlb(vn),unitlb(vn)),size=16)
    fig.savefig('%s/ann.ddp%02d%s.%s.%s.tr.pdf' % (odir,p,vn,fo,fut), format='pdf', dpi=dpi)
    fig.savefig('%s/ann.ddp%02d%s.%s.%s.tr.png' % (odir,p,vn,fo,fut), format='png', dpi=dpi)

    # plot pct warming - mean warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, llddpvn, np.arange(-vmdd,vmdd+dvmdd,dvmdd),extend='both', vmax=vmdd, vmin=-vmdd, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    ax.coastlines()
    # ax.set_title(r'%s %s ANN' % (md.upper(),fo.upper()),fontsize=16)
    ax.set_title(r'Annual mean',fontsize=16)
    # gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.5,color='gray',y_inline=False)
    # gl.ylocator=mticker.FixedLocator([-50,-30,0,30,50])
    # gl.yformatter=LatitudeFormatter()
    # gl.xlines=False
    # gl.left_labels=False
    # gl.bottom_labels=False
    # gl.right_labels=True
    # gl.top_labels=False
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\Delta \delta %s^{%02d}$ (%s)'%(varnlb(vn),p,unitlb(vn)),size=16)
    fig.savefig('%s/ann.ddp%02d%s.%s.png' % (odir,p,vn,fo), format='png', dpi=dpi)
    fig.savefig('%s/ann.ddp%02d%s.%s.pdf' % (odir,p,vn,fo), format='pdf', dpi=dpi)

    # plot pct warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, lldpvn, np.arange(-vmd,vmd+dvmd,dvmd),extend='both', vmax=vmd, vmin=-vmd, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    ax.coastlines()
    ax.set_title(r'%s %s ANN' % (md.upper(),fo.upper()),fontsize=16)
    # gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.5,color='gray',y_inline=False)
    # gl.ylocator=mticker.FixedLocator([-50,-30,0,30,50])
    # gl.yformatter=LatitudeFormatter()
    # gl.xlines=False
    # gl.left_labels=False
    # gl.bottom_labels=False
    # gl.right_labels=True
    # gl.top_labels=False
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=12)
    cb.set_label(label=r'$\Delta %s^{%02d}$ (%s)'%(varnlb(vn),p,unitlb(vn)),size=16)
    fig.savefig('%s/ann.dp%02d%s.%s.png' % (odir,p,vn,fo), format='png', dpi=dpi)

    # plot mean warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, lldvn, np.arange(-vmd,vmd+dvmd,dvmd),extend='both', vmax=vmd, vmin=-vmd, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    ax.coastlines()
    ax.set_title(r'%s %s ANN' % (md.upper(),fo.upper()),fontsize=16)
    # gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.5,color='gray',y_inline=False)
    # gl.ylocator=mticker.FixedLocator([-50,-30,0,30,50])
    # gl.yformatter=LatitudeFormatter()
    # gl.xlines=False
    # gl.left_labels=False
    # gl.bottom_labels=False
    # gl.right_labels=True
    # gl.top_labels=False
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=12)
    cb.set_label(label=r'$\Delta \overline{%s}$ (%s)'%(varnlb(vn),unitlb(vn)),size=16)
    fig.savefig('%s/ann.d%s.%s.png' % (odir,vn,fo), format='png', dpi=dpi)

# run
[plot(vn) for vn in lvn]
