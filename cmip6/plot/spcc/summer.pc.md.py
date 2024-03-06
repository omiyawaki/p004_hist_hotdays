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
lvn=['plh']
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
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

def loadpvn(fo,yr,vn):
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
            'plh_fixbc': [30,2],
            }
    return lvm[vn]

def plot(vn):
    if 'plh_' in vn:
        vn0='plh'
    else:
        vn0=vn

    vmd,dvmd=vmaxd(vn)

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # warming
    pvn1,gpi=loadpvn(fo1,his,vn0)
    pvn2,_=loadpvn(fo2,fut,vn)
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

    # plot pct warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, lldpvn, np.arange(-vmd,vmd+dvmd,dvmd),extend='both', vmax=vmd, vmin=-vmd, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    ax.coastlines()
    ax.set_title(r'%s %s JJA+DJF' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=12)
    cb.set_label(label=r'$\Delta %s^{%02d}$ (%s)'%(varnlb(vn),p,unitlb(vn)),size=16)
    fig.savefig('%s/summer.dp%03d%s.%s.png' % (odir,p,vn,fo), format='png', dpi=dpi)

# run
[plot(vn) for vn in lvn]
