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
p=97.5
vn='mrsos'
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

def loadcsm(fo,yr,vn):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    ds=xr.open_dataset('%s/%s.%s.%s.nc' % (idir,vn,yr,se))
    gpi=ds['gpi']
    try:
        pvn=ds[vn]
    except:
        pvn=ds['__xarray_dataarray_variable__']
    return pvn,gpi

def loadmvn(fo,yr,vn):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    ds=xr.open_dataset('%s/m.%s_%s.%s.nc' % (idir,vn,yr,se))
    gpi=ds['gpi']
    try:
        pvn=ds[vn]
    except:
        pvn=ds['__xarray_dataarray_variable__']
        try:
            pvn=ds['plh']
        except:
            sys.exit()
    return pvn,gpi

def loadpvn(fo,yr,vn):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    ds=xr.open_dataset('%s/pc.%s_%s.%s.nc' % (idir,vn,yr,se))
    pct=ds['percentile']
    gpi=ds['gpi']
    try:
        pvn=ds[vn]
    except:
        pvn=ds['__xarray_dataarray_variable__']
        try:
            pvn=ds['plh']
        except:
            sys.exit()
    return pvn,pct,gpi

def vmaxd(vn):
    lvm={   'mrsos':  [50,1],
            }
    return lvm[vn]

def dvmaxd(vn):
    lvm={   'mrsos':  [10,1],
            }
    return lvm[vn]

def plot(vn):
    vmd,dvmd=vmaxd(vn)
    dvmd,ddvmd=dvmaxd(vn)

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    odir1 = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo1,md,'csm')
    odir2 = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo2,md,'csm')

    if not os.path.exists(odir1):
        os.makedirs(odir1)
    if not os.path.exists(odir2):
        os.makedirs(odir2)

    # load csm
    csm1,_=loadcsm(fo1,his,'csm')
    csm2,_=loadcsm(fo2,fut,'csm')

    # replace inf with nan
    csm1.data[np.logical_or(csm1.data==np.inf,csm1.data==-np.inf)]=np.nan
    csm2.data[np.logical_or(csm2.data==np.inf,csm2.data==-np.inf)]=np.nan

    # load sm
    msm1,gpi=loadmvn(fo1,his,vn)
    msm2,_=loadmvn(fo2,fut,vn)
    psm1,pct,_=loadpvn(fo1,his,vn)
    psm2,_,_=loadpvn(fo2,fut,vn)
    psm1=psm1.sel(percentile=p)
    psm2=psm2.sel(percentile=p)

    # jja and djf means
    csm1j  =np.nanmean(csm1.data[5:8,:]  ,axis=0)
    csm2j  =np.nanmean(csm2.data[5:8,:]  ,axis=0)
    msm1j  =np.nanmean(msm1.data[5:8,:]  ,axis=0)
    msm2j  =np.nanmean(msm2.data[5:8,:]  ,axis=0)
    psm1j =np.nanmean(psm1.data[5:8,:] ,axis=0)
    psm2j =np.nanmean(psm2.data[5:8,:] ,axis=0)
    csm1d  =np.nanmean(np.roll(csm1.data  ,1,axis=0)[:3,:],axis=0)
    csm2d  =np.nanmean(np.roll(csm2.data  ,1,axis=0)[:3,:],axis=0)
    msm1d  =np.nanmean(np.roll(msm1.data  ,1,axis=0)[:3,:],axis=0)
    msm2d  =np.nanmean(np.roll(msm2.data  ,1,axis=0)[:3,:],axis=0)
    psm1d  =np.nanmean(np.roll(psm1.data  ,1,axis=0)[:3,:],axis=0)
    psm2d  =np.nanmean(np.roll(psm2.data  ,1,axis=0)[:3,:],axis=0)

    # remap to lat x lon
    llcsm1j=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llcsm2j=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llmsm1j=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llmsm2j=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llpsm1j=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llpsm2j=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llcsm1j[lmi]=csm1j.data
    llcsm2j[lmi]=csm2j.data
    llmsm1j[lmi]=msm1j.data
    llmsm2j[lmi]=msm2j.data
    llpsm1j[lmi]=psm1j.data
    llpsm2j[lmi]=psm2j.data
    llcsm1j=np.reshape(llcsm1j,(gr['lat'].size,gr['lon'].size))
    llcsm2j=np.reshape(llcsm2j,(gr['lat'].size,gr['lon'].size))
    llmsm1j=np.reshape(llmsm1j,(gr['lat'].size,gr['lon'].size))
    llmsm2j=np.reshape(llmsm2j,(gr['lat'].size,gr['lon'].size))
    llpsm1j=np.reshape(llpsm1j,(gr['lat'].size,gr['lon'].size))
    llpsm2j=np.reshape(llpsm2j,(gr['lat'].size,gr['lon'].size))

    llcsm1d=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llcsm2d=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llmsm1d=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llmsm2d=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llpsm1d=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llpsm2d=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llcsm1d[lmi]=csm1d.data
    llcsm2d[lmi]=csm2d.data
    llmsm1d[lmi]=msm1d.data
    llmsm2d[lmi]=msm2d.data
    llpsm1d[lmi]=psm1d.data
    llpsm2d[lmi]=psm2d.data
    llcsm1d=np.reshape(llcsm1d,(gr['lat'].size,gr['lon'].size))
    llcsm2d=np.reshape(llcsm2d,(gr['lat'].size,gr['lon'].size))
    llmsm1d=np.reshape(llmsm1d,(gr['lat'].size,gr['lon'].size))
    llmsm2d=np.reshape(llmsm2d,(gr['lat'].size,gr['lon'].size))
    llpsm1d=np.reshape(llpsm1d,(gr['lat'].size,gr['lon'].size))
    llpsm2d=np.reshape(llpsm2d,(gr['lat'].size,gr['lon'].size))

    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    llcsm1j = np.append(llcsm1j, llcsm1j[...,0][...,None],axis=1)
    llcsm2j = np.append(llcsm2j, llcsm2j[...,0][...,None],axis=1)
    llmsm1j = np.append(llmsm1j, llmsm1j[...,0][...,None],axis=1)
    llmsm2j = np.append(llmsm2j, llmsm2j[...,0][...,None],axis=1)
    llpsm1j = np.append(llpsm1j, llpsm1j[...,0][...,None],axis=1)
    llpsm2j = np.append(llpsm2j, llpsm2j[...,0][...,None],axis=1)
    llcsm1d = np.append(llcsm1d, llcsm1d[...,0][...,None],axis=1)
    llcsm2d = np.append(llcsm2d, llcsm2d[...,0][...,None],axis=1)
    llmsm1d = np.append(llmsm1d, llmsm1d[...,0][...,None],axis=1)
    llmsm2d = np.append(llmsm2d, llmsm2d[...,0][...,None],axis=1)
    llpsm1d = np.append(llpsm1d, llpsm1d[...,0][...,None],axis=1)
    llpsm2d = np.append(llpsm2d, llpsm2d[...,0][...,None],axis=1)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # use jja for nh, djf for sh
    llcsm1=np.copy(llcsm1d)
    llcsm2=np.copy(llcsm2d)
    llmsm1=np.copy(llmsm1d)
    llmsm2=np.copy(llmsm2d)
    llpsm1=np.copy(llpsm1d)
    llpsm2=np.copy(llpsm2d)
    llcsm1[gr['lat']>0]=llcsm1j[gr['lat']>0]
    llcsm2[gr['lat']>0]=llcsm2j[gr['lat']>0]
    llmsm1[gr['lat']>0]=llmsm1j[gr['lat']>0]
    llmsm2[gr['lat']>0]=llmsm2j[gr['lat']>0]
    llpsm1[gr['lat']>0]=llpsm1j[gr['lat']>0]
    llpsm2[gr['lat']>0]=llpsm2j[gr['lat']>0]

    # deviation from csm
    dmsm1=llmsm1-llcsm1
    dpsm1=llpsm1-llcsm1
    dmsm2=llmsm2-llcsm2
    dpsm2=llpsm2-llcsm2

    # mean elim and hot elim
    mehw=np.logical_and(dmsm1>0,dpsm1<0)
    # mean and hot mlim
    mwhw=np.logical_and(dmsm1<0,dpsm1<0)
    # mean and hot elim
    mehe=np.logical_and(dmsm1>0,dpsm1>0)

    # plot dpsm1
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, mehw, [0,1], vmin=0,vmax=0.999, extend='both', transform=ccrs.PlateCarree(), cmap='Purples')
    ax.coastlines()
    ax.set_title(r'%s %s JJA+DJF' % (md.upper(),fo1.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\delta \overline{%s}>0$ and $\delta %s^{%02d}<0$ (%s)'%(varnlb(vn),varnlb(vn),p,'Boolean'),size=16)
    fig.savefig('%s/summer.%s.%s.%s.png' % (odir1,'mehw',fo1,his), format='png', dpi=dpi)

    # plot dpsm1
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, mwhw, [0,1],vmin=0,vmax=0.999, extend='both', transform=ccrs.PlateCarree(), cmap='Reds')
    ax.coastlines()
    ax.set_title(r'%s %s JJA+DJF' % (md.upper(),fo1.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\delta \overline{%s}<0$ and $\delta %s^{%02d}<0$ (%s)'%(varnlb(vn),varnlb(vn),p,'Boolean'),size=16)
    fig.savefig('%s/summer.%s.%s.%s.png' % (odir1,'mwhw',fo1,his), format='png', dpi=dpi)

    # plot dpsm1
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, mehe, [0,1],vmin=0,vmax=0.999, extend='both', transform=ccrs.PlateCarree(), cmap='Greens')
    ax.coastlines()
    ax.set_title(r'%s %s JJA+DJF' % (md.upper(),fo1.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\delta \overline{%s}>0$ and $\delta %s^{%02d}>0$ (%s)'%(varnlb(vn),varnlb(vn),p,'Boolean'),size=16)
    fig.savefig('%s/summer.%s.%s.%s.png' % (odir1,'mehe',fo1,his), format='png', dpi=dpi)

    # plot dpsm1
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, dpsm1, np.arange(-dvmd,dvmd+ddvmd,ddvmd),extend='both', vmax=dvmd, vmin=-dvmd, transform=ccrs.PlateCarree(), cmap='BrBG')
    ax.coastlines()
    ax.set_title(r'%s %s JJA+DJF' % (md.upper(),fo1.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$%s^{%02d}-%s$ (%s)'%(varnlb(vn),p,varnlb('csm'),unitlb('csm')),size=16)
    fig.savefig('%s/summer.%s.%s.%s.png' % (odir1,'dpsm',fo1,his), format='png', dpi=dpi)

    # plot dmsm1
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, dmsm1, np.arange(-dvmd,dvmd+ddvmd,ddvmd),extend='both', vmax=dvmd, vmin=-dvmd, transform=ccrs.PlateCarree(), cmap='BrBG')
    ax.coastlines()
    ax.set_title(r'%s %s JJA+DJF' % (md.upper(),fo1.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\overline{%s}-%s$ (%s)'%(varnlb(vn),varnlb('csm'),unitlb('csm')),size=16)
    fig.savefig('%s/summer.%s.%s.%s.png' % (odir1,'dmsm',fo1,his), format='png', dpi=dpi)

    # plot csm
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, llcsm1, np.arange(0,vmd+dvmd,dvmd),extend='both', vmax=vmd, vmin=0, transform=ccrs.PlateCarree(), cmap='Greens')
    ax.coastlines()
    ax.set_title(r'%s %s JJA+DJF' % (md.upper(),fo1.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$%s$ (%s)'%(varnlb('csm'),unitlb('csm')),size=16)
    fig.savefig('%s/summer.%s.%s.%s.png' % (odir1,'csm',fo1,his), format='png', dpi=dpi)

# run
plot(vn)
