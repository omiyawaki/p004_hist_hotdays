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
varn3='pr'
varn='%s+%s'%(varn1,varn2)
ise='sc'
ose='jja'
lmo=[6,7,8]

fo1='historical' # forcings 
yr1='1980-2000'

fo2='ssp370' # forcings 
yr2='2080-2100'

fo='%s+%s'%(fo1,fo2)
fod='%s-%s'%(fo2,fo1)

# md='CanESM5'
# md='CESM2'
# md='MPI-ESM1-2-LR'
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

def load_data(md,fo,varn,mask):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (ise,fo,md,varn)
    ds=xr.open_dataset('%s/lm.%s_%s.%s.nc' % (idir,varn,yr1,ise))
    vn=ds[varn]
    gpi=ds['gpi']
    # extract data for season
    vn=vn.sel(time=vn['time.season']==ose.upper())
    # delete data outside masked region
    vn=np.delete(vn.data,np.nonzero(np.isnan(mask)),axis=1)
    vn=vn.flatten()
    return vn

def plot(relb,fo=fo1):
    # mask
    mask=masklev0(re,gr,mtype).data
    mask=mask.flatten()
    mask=np.delete(mask,omi)
    print(mask.shape)

    odir= '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (ose,fo,md,varn,'regions')
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('Loading %s...'%varn1)
    vn1=load_data(md,fo,varn1,mask)
    print('Loading %s...'%varn2)
    vn2=load_data(md,fo,varn2,mask)
    print('Loading %s...'%varn3)
    vn3=load_data(md,fo,varn3,mask)

    print('Plotting...')
    clim=set_clim(relb)
    tname=r'%s %s' % (md,retn)
    oname1='%s/bc0.%s_%s.%s' % (odir,varn,yr1,ose)
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    clf=ax.scatter(vn2,vn1,s=0.5,c=86400*vn3,vmin=clim[0],vmax=clim[1],cmap='BrBG')
    ax.set_title(r'%s' % (tname),fontsize=16)
    ax.set_xlabel('%s (%s)'%(varnlb(varn2),unitlb(varn2)))
    ax.set_ylabel('%s (%s)'%(varnlb(varn1),unitlb(varn1)))
    cb=fig.colorbar(clf,location='right')
    cb.set_label(label='%s (%s)'%(varnlb(varn3),unitlb(varn3)))
    fig.savefig('%s.png'%oname1, format='png', dpi=600)

for md in tqdm(lmd):
    for relb in lrelb:
        mtype=settype(relb)
        retn=retname(relb)
        re=regionsets(relb)
        plot(relb,fo=fo1)
