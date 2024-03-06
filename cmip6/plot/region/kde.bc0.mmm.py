import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import pickle
import pwlf
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from tqdm import tqdm
from util import mods
from utils import corr,corr2d,monname,varnlb,unitlb
from regions import pointlocs,masklev0,settype,retname,regionsets
from CASutils import shapefile_utils as shp

# relb='fourcorners'
# re=['Utah','Colorado','Arizona','New Mexico']

lrelb=['ic']

pct=np.linspace(1,99,101)
varn1='hfls'
varn2='mrsos'
varn='%s+%s'%(varn1,varn2)
ise='sc'
ose='ann'

fo1='historical' # forcings 
yr1='1980-2000'

fo2='ssp370' # forcings 
yr2='2080-2100'

fo='%s+%s'%(fo1,fo2)
fod='%s-%s'%(fo2,fo1)

lmd=mods(fo1)

def set_clim(relb):
    clim={  
            'ic':   [0,15],
            'fc':   [0,8],
            }
    return clim[relb]

# load ocean indices
_,omi=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

odir= '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (ose,fo1,'mmm',varn,'regions')
if not os.path.exists(odir):
    os.makedirs(odir)

def load_data(md,fo,varn,mask):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (ise,fo,md,varn)
    ds=xr.open_dataset('%s/lm.%s_%s.%s.nc' % (idir,varn,yr1,ise))
    vn=ds[varn]
    gpi=ds['gpi']
    # extract data for season
    if ose!='ann':
        vn=vn.sel(time=vn['time.season']==ose.upper())
    # delete data outside masked region
    vn=np.delete(vn.data,np.nonzero(np.isnan(mask)),axis=1)
    vn=vn.flatten()
    return vn

def selreg(md,varn,relb,fo=fo1):
    # mask
    mask=masklev0(re,gr,mtype).data
    mask=mask.flatten()
    mask=np.delete(mask,omi)
    print(mask.shape)

    print('Loading %s...'%varn1)
    vn=load_data(md,fo,varn,mask)
    return vn

for relb in lrelb:
    mtype=settype(relb)
    retn=retname(relb)
    re=regionsets(relb)
    vn1=[selreg(lmd[i],varn1,relb,fo=fo1) for i in tqdm(range(len(lmd)))]
    vn2=[selreg(lmd[i],varn2,relb,fo=fo1) for i in tqdm(range(len(lmd)))]
    vn1=np.concatenate(vn1)
    vn2=np.concatenate(vn2)

    # create kde
    nans=np.logical_or(np.isnan(vn1),np.isnan(vn2))
    vn1=vn1[~nans]
    vn2=vn2[~nans]
    kde=gaussian_kde(np.vstack([vn2,vn1]))
    x1=np.linspace(np.min(vn1),np.max(vn1),51)
    x2=np.linspace(np.min(vn2),np.max(vn2),51)
    [x1,x2]=np.meshgrid(x1,x2,indexing='ij')
    pdf=kde(np.vstack([x2.ravel(),x1.ravel()]))
    pdf=np.reshape(pdf,x1.shape)
    print(np.max(pdf))

    print('Plotting...')
    clim=set_clim(relb)
    oname1='%s/kde.bc.%s_%s.%s' % (odir,varn,yr1,ose)
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    tname=r'MMM %s' % (retn)
    ax.set_title(r'%s' % (tname))
    ax.set_xlabel('%s (%s)'%(varnlb(varn2),unitlb(varn2)))
    ax.set_ylabel('%s (%s)'%(varnlb(varn1),unitlb(varn1)))
    clf=ax.contourf(x2,x1,pdf,np.logspace(-5,-3,101),extend='both',norm=mcolors.LogNorm())
    cb=fig.colorbar(clf,location='right')
    cb.set_label(label='Kernel density (unitless)')
    cb.ax.set_yscale('log')
    fig.savefig('%s.png'%oname1, format='png', dpi=600)

