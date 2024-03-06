import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import dask
import dask.multiprocessing
from dask.distributed import Client
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
from utils import monname,varnlb,unitlb,corr2d

nt=7 # window size in days
p=97.5
vn='ooplh_fixbc300'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='gwl2.0'
dpi=600
skip507599=True
reverse=False

lmd=mods(fo1)

# load land indices
lmi,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

def loadmvn(fo,yr,vn,md):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    pvn=xr.open_dataarray('%s/m.%s_%s.%s.nc' % (idir,vn,yr,se))
    gpi=pvn['gpi']
    return pvn,gpi

def loadpvn(fo,yr,vn,md):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    pvn=xr.open_dataarray('%s/pc.%s_%s.%s.nc' % (idir,vn,yr,se))
    pct=pvn['percentile']
    gpi=pvn['gpi']
    return pvn,pct,gpi

def vmaxdd(vn):
    lvm={   'tas':          [1.5,0.1+1e-2],
            'ta850':        [1.5,0.1],
            'hfls':         [30,1],
            'lhflx':        [30,1],
            'qsum':         [30,1],
            'qsoil':        [30,1],
            'qvege':        [30,1],
            'qvegt':        [30,1+1e-2],
            'ef':           [0.1,0.01],
            'ef2':          [0.1,0.01],
            'ef3':          [0.1,0.01],
            'plh':          [30,1],
            'plh_fixbc':    [30,1],
            'ooef':         [0.1,0.01],
            'ooef2':        [0.1,0.01],
            'ooef3':        [0.1,0.01],
            'ooblh':        [30,1],
            'rfa':          [30,3],
            'fat850':       [0.3,0.03],
            }
    if 'ooplh' in vn:
        return [30,1]
    else:
        return lvm[vn]

def vmaxd(vn):
    lvm={   'tas':          [8,0.25],
            'ta850':        [8,0.25],
            'hfls':         [30,2],
            'lhflx':        [30,2],
            'qsum':         [30,2],
            'qsoil':        [30,2],
            'qvege':        [30,2],
            'qvegt':        [30,2],
            'ef':           [0.1,0.01],
            'ef2':          [0.1,0.01],
            'ef3':          [0.1,0.01],
            'plh':          [30,2],
            'plh_fixbc':    [30,2],
            'ooef':         [0.1,0.01],
            'ooef2':        [0.1,0.01],
            'ooef3':        [0.1,0.01],
            'ooblh':        [30,2],
            'ooplh':        [30,2],
            'ooplh300':     [30,2],
            'ooplh_orig':   [30,2],
            'rfa':          [30,3],
            'fat850':       [0.3,0.03],
            }
    if 'ooplh' in vn:
        return [30,2]
    else:
        return lvm[vn]

def plot(md):
    if '_fixbc' in vn:
        vn0=vn.replace('_fixbc','')
    elif '_fixmsm' in vn:
        vn0=vn.replace('_fixmsm','')
    elif '_rddsm' in vn:
        vn0=vn.replace('_rddsm','')
    elif 'plh_' in vn:
        vn0='plh'
    else:
        vn0=vn

    vmdd,dvmdd=vmaxdd(vn)
    vmd,dvmd=vmaxd(vn)

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # warming
    pvn1,pct,gpi=loadpvn(fo1,his,vn0,md)
    pvn2,_,_=loadpvn(fo2,fut,vn,md)
    mvn1,_=loadmvn(fo1,his,vn0,md)
    mvn2,_=loadmvn(fo2,fut,vn,md)
    dvn=mvn2-mvn1
    dpvn=pvn2-pvn1
    ddpvn=dpvn-np.transpose(dvn.data[...,None],[0,2,1])
    dpvn=dpvn.sel(percentile=p).squeeze()
    ddpvn=ddpvn.sel(percentile=p).squeeze()

    if reverse and (vn in ['gflx','hfss','hfls','fa850','fat850','rfa'] or 'ooplh' in vn):
        dpvn=-dpvn
        ddpvn=-ddpvn

    # jja and djf means
    dvnj  =np.nanmean(dvn.data[5:8,:]  ,axis=0)
    dpvnj =np.nanmean(dpvn.data[5:8,:] ,axis=0)
    ddpvnj=np.nanmean(ddpvn.data[5:8,:],axis=0)
    dvnd  =np.nanmean(np.roll(dvn.data  ,1,axis=0)[:3,:],axis=0)
    dpvnd =np.nanmean(np.roll(dpvn.data ,1,axis=0)[:3,:],axis=0)
    ddpvnd=np.nanmean(np.roll(ddpvn.data,1,axis=0)[:3,:],axis=0)

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

    # plot TROPICS ONLY
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    # ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
    ax.set_title(r'%s %s JJA+DJF' % (md.upper(),fo.upper()),fontsize=16)
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
    fig.savefig('%s/summer.ddp%02d%s.%s.%s.tr.pdf' % (odir,p,vn,fo,fut), format='pdf', dpi=dpi)
    fig.savefig('%s/summer.ddp%02d%s.%s.%s.tr.png' % (odir,p,vn,fo,fut), format='png', dpi=dpi)

    # plot pct warming - mean warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, llddpvn, np.arange(-vmdd,vmdd+dvmdd,dvmdd),extend='both', vmax=vmdd, vmin=-vmdd, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    ax.coastlines()
    ax.set_title(r'%s %s JJA+DJF' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\Delta \delta %s^{%02d}$ (%s)'%(varnlb(vn),p,unitlb(vn)),size=16)
    fig.savefig('%s/summer.ddp%02d%s.%s.png' % (odir,p,vn,fo), format='png', dpi=dpi)

    # plot pct warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, lldpvn, np.arange(-vmd,vmd+dvmd,dvmd),extend='both', vmax=vmd, vmin=-vmd, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    ax.coastlines()
    ax.set_title(r'%s %s JJA+DJF' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=12)
    cb.set_label(label=r'$\Delta %s^{%02d}$ (%s)'%(varnlb(vn),p,unitlb(vn)),size=16)
    fig.savefig('%s/summer.dp%02d%s.%s.png' % (odir,p,vn,fo), format='png', dpi=dpi)

    # plot mean warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, lldvn, np.arange(-vmd,vmd+dvmd,dvmd),extend='both', vmax=vmd, vmin=-vmd, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    ax.coastlines()
    ax.set_title(r'%s %s JJA+DJF' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=12)
    cb.set_label(label=r'$\Delta \overline{%s}$ (%s)'%(varnlb(vn),unitlb(vn)),size=16)
    fig.savefig('%s/summer.d%s.%s.png' % (odir,vn,fo), format='png', dpi=dpi)

# run
plot('CESM2')

# if __name__=='__main__':
#     with Client(n_workers=len(lmd)):
#         tasks=[dask.delayed(plot)(md) for md in lmd]
#         dask.compute(*tasks)
