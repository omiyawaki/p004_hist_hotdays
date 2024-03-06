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
from etregimes import bestfit

# relb='fourcorners'
# re=['Utah','Colorado','Arizona','New Mexico']

relb='ic'

p=95
pct=np.linspace(1,99,101)
varn1='hfls'
varn2='mrsos'
varn3='pr'
vr=[0,15]
bvn2=np.arange(0,50+2,2)
varn='%s+%s'%(varn1,varn2)
se='sc'

fo1='historical' # forcings 
yr1='1980-2000'

fo2='ssp370' # forcings 
yr2='2080-2100'

fo='%s+%s'%(fo1,fo2)
fod='%s-%s'%(fo2,fo1)
yr='%s+%s'%(yr1,yr2)

lmd=mods(fo1)
mmm=True

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

def load_hotdata(md,fo,varn,mask,yr):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    ds=xr.open_dataset('%s/pc.%s_%s.%s.nc' % (idir,varn,yr,se))
    ds=ds.sel(percentile=p)
    vn=ds[varn].squeeze()
    gpi=ds['gpi']
    nd,ng=vn.shape
    if varn=='pr': vn=86400*vn # convert pr to mm/d
    month=ds['month']
    # delete data outside masked region
    vn=np.delete(vn.data,np.nonzero(np.isnan(mask)),axis=1)
    vn=np.nanmean(vn,axis=1)
    return vn

def load_meandata(md,fo,varn,mask,yr):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if md=='mmm':
        ds=xr.open_dataset('%s/cm.%s_%s.%s.nc' % (idir,varn,yr,se))
    else:
        ds=xr.open_dataset('%s/lm.%s_%s.%s.nc' % (idir,varn,yr,se))
        ds=ds.groupby('time.month').mean(dim='time')
    vn=ds[varn]
    if varn=='pr': vn=86400*vn # convert pr to mm/d
    # delete data outside masked region
    vn=np.delete(vn.data,np.nonzero(np.isnan(mask)),axis=1)
    vn=np.nanmean(vn,axis=1)
    return vn

def selreg(md,varn,relb,fo=fo1,yr=yr1,dtype='mean'):
    # mask
    mask=masklev0(re,gr,mtype).data
    mask=mask.flatten()
    mask=np.delete(mask,omi)
    print(mask.shape)

    print('Loading %s...'%varn)
    if dtype=='mean':
        vn=load_meandata(md,fo,varn,mask,yr)
    elif dtype=='hot':
        vn=load_hotdata(md,fo,varn,mask,yr)
    return vn

for md in lmd:
    mtype=settype(relb)
    retn=retname(relb)
    re=regionsets(relb)

    # load hot and mean SM and LH data
    if mmm:
        md='mmm'

    psm1=selreg(md,varn2,relb,fo=fo1,yr=yr1,dtype='hot')
    psm2=selreg(md,varn2,relb,fo=fo2,yr=yr2,dtype='hot')
    plh1=selreg(md,varn1,relb,fo=fo1,yr=yr1,dtype='hot')
    plh2=selreg(md,varn1,relb,fo=fo2,yr=yr2,dtype='hot')
    msm1=selreg(md,varn2,relb,fo=fo1,yr=yr1,dtype='mean')
    msm2=selreg(md,varn2,relb,fo=fo2,yr=yr2,dtype='mean')
    mlh1=selreg(md,varn1,relb,fo=fo1,yr=yr1,dtype='mean')
    mlh2=selreg(md,varn1,relb,fo=fo2,yr=yr2,dtype='mean')

    # load BC data
    ddir1= '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,fo1,md,varn,'regions')
    dname1='%s/bc.%s_%s.%s.%s' % (ddir1,varn,yr1,se,relb)
    xc1,bc1=pickle.load(open('%s.pickle'%dname1,'rb'))

    ddir2= '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,fo2,md,varn,'regions')
    dname2='%s/bc.%s_%s.%s.%s' % (ddir2,varn,yr2,se,relb)
    xc2,bc2=pickle.load(open('%s.pickle'%dname2,'rb'))

    odir= '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (se,fo,md,varn,'regions')
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('Plotting...')
    clim=set_clim(relb)
    oname1='%s/bc.lineonly.%s_%s.%s.%s' % (odir,varn,yr,se,relb)
    fig,ax=plt.subplots(nrows=3,ncols=4,figsize=(9,7),constrained_layout=True)
    tname=r'%s %s' % (md.upper(),retn)
    fig.suptitle(r'%s' % (tname))
    fig.supxlabel('%s (%s)'%(varnlb(varn2),unitlb(varn2)))
    fig.supylabel('%s (%s)'%(varnlb(varn1),unitlb(varn1)))
    ax=ax.flatten()
    for i,mon in enumerate(tqdm(np.arange(1,13,1))):
        ax[i].set_title(monname(i))
        # select month
        ax[i].plot(msm1[i],mlh1[i],'o',color='tab:blue')
        ax[i].plot(psm1[i],plh1[i],'+',color='tab:blue')
        # ax[i].plot(msm2[i],mlh2[i],'o',color='tab:orange')
        # ax[i].plot(psm2[i],plh2[i],'+',color='tab:orange')
        ax[i].axvline(xc1[i],linestyle=':',color='tab:blue')
        ax[i].plot(bc1[i][0],bc1[i][1],'-',color='tab:blue')
        # ax[i].axvline(xc2[i],linestyle=':',color='tab:orange')
        # ax[i].plot(bc2[i][0],bc2[i][1],'-',color='tab:orange')
        fig.savefig('%s.png'%oname1, format='png', dpi=600)

    if mmm:
        sys.exit()
