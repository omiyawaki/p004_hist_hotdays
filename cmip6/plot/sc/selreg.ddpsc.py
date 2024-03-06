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
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
from util import mods
from utils import monname,varnlb,unitlb
from regions import masklev0,regionsets,retname,settype

lrelb=['af','ic','sa']
# lrelb=['ic']
lvarn=['tas']

nt=30 # lead-lag window size in days
nf=15 # uniform filter length
p=95
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s+%s'%(fo1,fo2)
fod='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'

md='mmm'

# load land indices
_,omi=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

def plot(varn,relb,ll1,ll2):
    mtype=settype(relb)
    re=regionsets(relb)
    retn=retname(relb)

    # mask
    mask=masklev0(re,gr,mtype).data
    mask=mask.flatten()
    mask=np.delete(mask,omi)

    # delete data outside masked region and average
    ll1=np.nanmean(np.delete(ll1.data,np.nonzero(np.isnan(mask)),axis=1),axis=1)
    ll2=np.nanmean(np.delete(ll2.data,np.nonzero(np.isnan(mask)),axis=1),axis=1)

    # smooth
    ll1=uniform_filter1d(ll1,nf)
    ll2=uniform_filter1d(ll2,nf)

    # plot climo
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    ax.set_title(r'%s %s' % (md.upper(),retn))
    ax.set_ylabel('$\Delta$ %s (%s)'%(varnlb(varn),unitlb(varn)))
    [ax.axvline(_dmn,color='k',linewidth=0.5,alpha=0.2) for _dmn in dmn]
    ax.axhline(0,linewidth=0.5,color='k',alpha=0.2)
    ax.plot(doy,ll1,color='tab:red',label='Hot days')
    ax.plot(doy,ll2,color='k',label='All days')
    # ax.axhline(0,linewidth=0.5,color='k')
    ax.set_xlim([doy[0],doy[-1]])
    ax.set_xticks(dmnmp)
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax.legend(frameon=False)
    fig.savefig('%s/d.p%02d.%s.%s.%s.png' % (odir,p,varn,fo1,relb), format='png', dpi=600)

    # plot climo
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    ax.set_title(r'%s %s' % (md.upper(),retn))
    ax.set_ylabel('$\Delta\delta$ %s (%s)'%(varnlb(varn),unitlb(varn)))
    [ax.axvline(_dmn,color='k',linewidth=0.5,alpha=0.2) for _dmn in dmn]
    ax.axhline(0,linewidth=0.5,color='k',alpha=0.2)
    ax.plot(doy,ll1-ll2,color='k')
    # ax.axhline(0,linewidth=0.5,color='k')
    ax.set_xlim([doy[0],doy[-1]])
    ax.set_xticks(dmnmp)
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax.legend(frameon=False)
    fig.savefig('%s/dd.p%02d.%s.%s.%s.png' % (odir,p,varn,fo1,relb), format='png', dpi=600)

for relb in lrelb:
    for varn in lvarn:
        idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo1,md,varn)
        idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo2,md,varn)
        idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)

        odir1 = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/regions' % (se,fo1,md,varn)
        if not os.path.exists(odir1):
            os.makedirs(odir1)
        odir2 = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/regions' % (se,fo2,md,varn)
        if not os.path.exists(odir2):
            os.makedirs(odir2)
        odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/regions' % (se,fod,md,varn)
        if not os.path.exists(odir):
            os.makedirs(odir)

        # ll data
        ds=xr.open_dataset('%s/pc.doy.%s_%s.%s.nc' % (idir1,varn,his,se))
        ll1=ds[varn].squeeze()
        gpi=ds['gpi']
        nd,ng=ll1.shape
        ll2=xr.open_dataarray('%s/pc.doy.%s_%s.%s.nc' % (idir2,varn,fut,se)).squeeze()
        if varn=='pr':
            ll1=86400*ll1
            ll2=86400*ll2
        dp=ll2-ll1

        if md=='mmm':
            ll1=xr.open_dataarray('%s/cd.%s_%s.%s.nc' % (idir1,varn,his,se))
            ll2=xr.open_dataarray('%s/cd.%s_%s.%s.nc' % (idir2,varn,fut,se))
        else:
            ll1=xr.open_dataarray('%s/lm.%s_%s.%s.nc' % (idir1,varn,his,se))
            ll2=xr.open_dataarray('%s/lm.%s_%s.%s.nc' % (idir2,varn,fut,se))
            ll1=ll1.groupby('time.dayofyear').mean(dim='time')
            ll2=ll2.groupby('time.dayofyear').mean(dim='time')
        if varn=='pr':
            ll1=86400*ll1
            ll2=86400*ll2
        dm=ll2-ll1

        doy=np.arange(nd)+1
        dmn=np.arange(0,len(doy),np.ceil(len(doy)/12))
        dmnmp=np.arange(0,len(doy),np.ceil(len(doy)/12))+np.ceil(len(doy)/24)

        plot(varn,relb,dp,dm)

