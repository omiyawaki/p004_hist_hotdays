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
vn1='plh'
vn2='hfls'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'
dpi=600
skip507599=True

md='CESM2'

# load land indices
lmi,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr0=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

def loadpvn(fo,yr,vn):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    ds=xr.open_dataset('%s/pc.%s_%s.%s.nc' % (idir,vn,yr,se))
    pct=ds['percentile']
    gpi=ds['gpi']
    try:
        pvn=ds[vn]
    except:
        pvn=ds['__xarray_dataarray_variable__']
    return pvn,pct,gpi

def loadhotpvn(fo,yr,vn):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    ds=xr.open_dataset('%s/pc%03d.%s_%s.%s.nc' % (idir,p,vn,yr,se))
    gpi=ds['gpi']
    try:
        pvn=ds[vn]
    except:
        pvn=ds['__xarray_dataarray_variable__']
    return pvn,gpi

def vmaxd(vn):
    lvm={   'tas':  [8,0.25],
            'hfls': [30,2],
            'plh': [30,2],
            }
    return lvm[vn]

def proc(vn,gr0):
    gr=gr0.copy()
    vmd,dvmd=vmaxd(vn)

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # warming
    if vn=='plh':
        pvn1,gpi=loadhotpvn(fo1,his,vn)
        pvn2,_=loadhotpvn(fo2,fut,vn)
    else:
        pvn1,pct,gpi=loadpvn(fo1,his,vn)
        pvn2,_,_=loadpvn(fo2,fut,vn)
        pvn1=pvn1.sel(percentile=pct==p).squeeze()
        pvn2=pvn2.sel(percentile=pct==p).squeeze()
    dpvn=pvn2-pvn1

    # jja and djf means
    dpvnj =np.nanmean(dpvn.data[5:8,:] ,axis=0)
    dpvnd =np.nanmean(np.roll(dpvn.data ,1,axis=0)[:3,:],axis=0)

    # remap to lat x lon
    lldpvnj=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    lldpvnj[lmi]=dpvnj.data
    lldpvnj=np.reshape(lldpvnj,(gr['lat'].size,gr['lon'].size))

    lldpvnd=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    lldpvnd[lmi]=dpvnd.data
    lldpvnd=np.reshape(lldpvnd,(gr['lat'].size,gr['lon'].size))

    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    lldpvnj = np.append(lldpvnj, lldpvnj[...,0][...,None],axis=1)
    lldpvnd = np.append(lldpvnd, lldpvnd[...,0][...,None],axis=1)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # use jja for nh, djf for sh
    lldpvn=np.copy(lldpvnd)
    lldpvn[gr['lat']>0]=lldpvnj[gr['lat']>0]

    return lldpvn,gr

# run
dpvn1,_=proc(vn1,gr0)
dpvn2,gr=proc(vn2,gr0)

rpvn=corr2d(dpvn1,dpvn2,gr,(0,1))

print('R(%s,%s)=%g for hot-day response'%(vn1,vn2,rpvn))
