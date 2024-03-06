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

mn=7
nt=7 # window size in days
p=95
lvn=['csm']
vnp='csm'
se = 'sc' # season (ann, djf, mam, jja, son)

fo='historical' # forcings 
yr='1980-2000'

# fo='ssp370' # forcings 
# yr='gwl2.0'

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

def loadvn(fo,yr,vn):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    ds=xr.open_dataset('%s/%s.%s.%s.nc' % (idir,vn,yr,se))
    gpi=ds['gpi']
    try:
        pvn=ds[vn]
    except:
        pvn=ds['__xarray_dataarray_variable__']
        try:
            pvn=ds[varn[2:5]]
        except:
            sys.exit()
    return pvn,gpi

def loadmvn(fo,yr,vn):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    ds=xr.open_dataset('%s/m.%s_%s.%s.nc' % (idir,vn,yr,se))
    gpi=ds['gpi']
    try:
        mvn=ds[vn]
    except:
        mvn=ds['__xarray_dataarray_variable__']
        try:
            mvn=ds[varn[2:5]]
        except:
            sys.exit()
    return mvn

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
            pvn=ds[varn[2:5]]
        except:
            sys.exit()
    return pvn

def plot(vn):
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # load mean and p sm
    msm=loadmvn(fo,yr,'mrsos')
    psm=loadpvn(fo,yr,'mrsos')

    # load critical sm
    csm,_=loadvn(fo,yr,vn)

    # take difference from crit
    msm=msm-csm
    psm=psm-csm

    # jja and djf means
    msmj=msm.data[mn-1,:]
    msmd=msm.data[(mn+5)%12,:]
    psmj=psm.data[mn-1,:]
    psmd=psm.data[(mn+5)%12,:]

    # remap to lat x lon
    llmsmj=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llmsmj[lmi]=msmj.data
    llmsmj=np.reshape(llmsmj,(gr['lat'].size,gr['lon'].size))
    llpsmj=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llpsmj[lmi]=psmj.data
    llpsmj=np.reshape(llpsmj,(gr['lat'].size,gr['lon'].size))

    llmsmd=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llmsmd[lmi]=msmd.data
    llmsmd=np.reshape(llmsmd,(gr['lat'].size,gr['lon'].size))
    llpsmd=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llpsmd[lmi]=psmd.data
    llpsmd=np.reshape(llpsmd,(gr['lat'].size,gr['lon'].size))

    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    llmsmj = np.append(llmsmj, llmsmj[...,0][...,None],axis=1)
    llmsmd = np.append(llmsmd, llmsmd[...,0][...,None],axis=1)
    llpsmj = np.append(llpsmj, llpsmj[...,0][...,None],axis=1)
    llpsmd = np.append(llpsmd, llpsmd[...,0][...,None],axis=1)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # use jja for nh, djf for sh
    llmsm=np.copy(llmsmd)
    llmsm[gr['lat']>0]=llmsmj[gr['lat']>0]
    llpsm=np.copy(llpsmd)
    llpsm[gr['lat']>0]=llpsmj[gr['lat']>0]

    # plot pct warming - mean warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.pcolormesh(mlon,mlat,llmsm,vmin=-5,vmax=5,cmap='BrBG',transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(r'%s %s %s+%s' % (md.upper(),fo.upper(),monname(mn-1),monname((mn+5)%12)),fontsize=16)
    # cb=fig.colorbar(clf,location='bottom',aspect=50)
    # cb.ax.tick_params(labelsize=16)
    # cb.set_label(label=r'$\Delta \delta %s^{%02d}$ (%s)'%(varnlb(vn),p,unitlb(vn)),size=16)
    fig.savefig('%s/mn.%s.cat.m.%s.%s.%s.png' % (odir,vn,fo,yr,mn), format='png', dpi=dpi)

    # plot pct warming - mean warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.pcolormesh(mlon,mlat,llpsm,vmin=-5,vmax=5,cmap='BrBG',transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(r'%s %s %s+%s' % (md.upper(),fo.upper(),monname(mn-1),monname((mn+5)%12)),fontsize=16)
    # cb=fig.colorbar(clf,location='bottom',aspect=50)
    # cb.ax.tick_params(labelsize=16)
    # cb.set_label(label=r'$\Delta \delta %s^{%02d}$ (%s)'%(varnlb(vn),p,unitlb(vn)),size=16)
    fig.savefig('%s/mn.%s.cat.p.%s.%s.%s.png' % (odir,vn,fo,yr,mn), format='png', dpi=dpi)

# run
[plot(vn) for vn in lvn]
