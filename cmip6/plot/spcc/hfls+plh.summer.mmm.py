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
from utils import monname,varnlb,unitlb,corr2d

nt=7 # window size in days
p=95
vn1='ooplh_mmmsm'
vn2='hfls'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'
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
gr0=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})


def proc(vn,gr0):
    gr=gr0.copy()

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # warming
    dvn=xr.open_dataset('%s/d.%s_%s_%s.%s.nc' % (idir,vn,his,fut,se))['__xarray_dataarray_variable__']
    try:
        dpvn=xr.open_dataset('%s/dpc.%s_%s_%s.%s.nc' % (idir,vn,his,fut,se))[vn]
    except:
        dpvn=xr.open_dataset('%s/dpc.%s_%s_%s.%s.nc' % (idir,vn,his,fut,se))['ooplh']
    ds=xr.open_dataset('%s/ddpc.%s_%s_%s.%s.nc' % (idir,vn,his,fut,se))
    try:
        ddpvn=ds[vn]
    except:
        ddpvn=ds['ooplh']
    pct=ds['percentile']
    gpi=ds['gpi']
    dpvn=dpvn.sel(percentile=pct==p).squeeze()
    ddpvn=ddpvn.sel(percentile=pct==p).squeeze()

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

    return lldvn,lldpvn,llddpvn,gr

# run
dvn1,dpvn1,ddpvn1,_=proc(vn1,gr0)
dvn2,dpvn2,ddpvn2,gr=proc(vn2,gr0)

rvn=corr2d(dvn1,dvn2,gr,(0,1))
rpvn=corr2d(dpvn1,dpvn2,gr,(0,1))
rdpvn=corr2d(ddpvn1,ddpvn2,gr,(0,1))

print('R(%s,%s)=%g for mean-day response'%(vn1,vn2,rvn))
print('R(%s,%s)=%g for hot-day response'%(vn1,vn2,rpvn))
print('R(%s,%s)=%g for hot-mean response'%(vn1,vn2,rdpvn))
