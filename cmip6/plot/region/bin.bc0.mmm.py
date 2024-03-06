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
vr=[0,15]
bvn2=np.arange(10,50+2,2)
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
    if varn=='pr':
        vn=86400*vn
    return vn

for relb in lrelb:
    mtype=settype(relb)
    retn=retname(relb)
    re=regionsets(relb)
    vn1=[selreg(lmd[i],varn1,relb,fo=fo1) for i in tqdm(range(len(lmd)))]
    vn2=[selreg(lmd[i],varn2,relb,fo=fo1) for i in tqdm(range(len(lmd)))]
    vn3=[selreg(lmd[i],varn3,relb,fo=fo1) for i in tqdm(range(len(lmd)))]
    vn1=np.concatenate(vn1)
    vn2=np.concatenate(vn2)
    vn3=np.concatenate(vn3)

    # bin
    nans=np.logical_or(np.isnan(vn1),np.isnan(vn2))
    vn1=vn1[~nans]
    vn2=vn2[~nans]
    vn3=vn3[~nans]
    dg=np.digitize(vn2,bvn2)
    bvn1=np.array([vn1[dg==i].mean() for i in range(1,len(bvn2)+1)])
    svn1=np.array([vn1[dg==i].std() for i in range(1,len(bvn2)+1)])
    bvn3=np.array([vn3[dg==i].mean() for i in range(1,len(bvn2)+1)])

    # find critical soil moisture
    nans=np.logical_or(np.isnan(bvn1),np.isnan(bvn2))
    nbvn1=bvn1[~nans]
    nbvn2=bvn2[~nans]
    pa=np.polynomial.polynomial.polyfit(nbvn2,nbvn1,3) # polynomial fit
    xv1=-(2*pa[2]+np.sqrt(4*pa[2]**2-12*pa[1]*pa[3]))/(6*pa[3])
    xv2=-(2*pa[2]-np.sqrt(4*pa[2]**2-12*pa[1]*pa[3]))/(6*pa[3])
    cc1=2*pa[2]+6*pa[3]*xv1
    cc2=2*pa[2]+6*pa[3]*xv2
    if cc1<0:
        csm=xv1
    else:
        csm=xv2
    print(csm)

    print('Plotting...')
    clim=set_clim(relb)
    oname1='%s/bin.bc.%s_%s.%s' % (odir,varn,yr1,ose)
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    tname=r'MMM %s' % (retn)
    ax.set_title(r'%s' % (tname))
    ax.set_xlabel('%s (%s)'%(varnlb(varn2),unitlb(varn2)))
    ax.set_ylabel('%s (%s)'%(varnlb(varn1),unitlb(varn1)))
    try:
        ax.annotate(r'$SM_c=%0.1d$'%csm, xy=(0.05, 0.85), xycoords='axes fraction')
    except:
        print('SMc could not be solved')
    ax.axvline(csm,color='tab:purple')
    clf=ax.errorbar(bvn2,bvn1,yerr=svn1,fmt='.',color='k',alpha=0.2)
    clf=ax.scatter(bvn2,bvn1,s=5,c=bvn3,vmin=vr[0],vmax=vr[1],cmap='BrBG')
    cb=fig.colorbar(clf,location='right')
    cb.set_label(label='%s (%s)'%(varnlb(varn3),unitlb(varn3)))
    fig.savefig('%s.png'%oname1, format='png', dpi=600)

    ddir= '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (ose,fo,'mmm',varn,'regions')
    if not os.path.exists(ddir):
        os.makedirs(ddir)
    
    csm=xr.DataArray(np.array([csm]),coords={'model':['mmm']},dims=('model'))
    csm.to_netcdf('%s/crit.sm.%s.nc'%(ddir,relb),format='NETCDF4')
