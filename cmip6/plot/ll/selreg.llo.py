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
from regions import masklev0,countrysets,retname

mtype='country'
lrelb=['af','ic','sa']
lvarn=['pr','mrsos','hfls','tas']

nt=30 # lead-lag window size in days
p=95
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s+%s'%(fo1,fo2)
his='1980-2000'
fut='2080-2100'

md='CESM2'

def setylim(varn):
    lims={
            'tas':      [-1,5],
            'mrsos':    [-10,1],
            'pr':       [-8,2],
            'hfls':     [-35,15],
            }
    return lims[varn]

# load land indices
_,omi=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

def plot(varn,relb,ll1,ll2):
    re=countrysets(relb)
    retn=retname(relb)

    # mask
    mask=masklev0(re,gr,mtype).data
    mask=mask.flatten()
    mask=np.delete(mask,omi)

    # delete data outside masked region and average
    if varn=='mrsos':
        ll1.data[np.abs(ll1.data)>1e10]=np.nan
        ll2.data[np.abs(ll2.data)>1e10]=np.nan
    discard=np.nonzero(np.isnan(mask))
    ll1=np.nanmean(np.delete(ll1.data,discard,axis=2),axis=2)
    ll2=np.nanmean(np.delete(ll2.data,discard,axis=2),axis=2)

    # plot lead-lag in historical climate
    fig,ax=plt.subplots(nrows=3,ncols=4,figsize=(12,7),constrained_layout=True)
    ax=ax.flatten()
    fig.suptitle(r'%s %s %s' % (md.upper(),fo1.upper(),retn),fontsize=16)
    fig.supxlabel('Days relative to $T^{>%g}$ day'%p,fontsize=16)
    fig.supylabel('%s anomaly (%s)'%(varnlb(varn),unitlb(varn)),fontsize=16)
    for m in tqdm(range(nm)):
        clf=ax[m].plot(lday,ll1[:,m],color='tab:blue')
        clf=ax[m].plot(lday,ll2[:,m],color='tab:orange')
        ax[m].axhline(0,linewidth=0.5,color='k')
        ax[m].set_xlim([-30,30])
        ax[m].set_ylim(setylim(varn))
        ax[m].set_xticks(np.arange(-30,31,10))
        ax[m].set_title(r'%s' % (monname(m)),fontsize=16)
        fig.savefig('%s/ddp.ll.%02d%s.%s.%s.png' % (odir,p,varn,fo,relb), format='png', dpi=300)
    fig.savefig('%s/ddp.ll.%02d%s.%s.%s.png' % (odir,p,varn,fo,relb), format='png', dpi=300)

for relb in lrelb:
    for varn in lvarn:
        print(varn,relb)
        idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo1,md,varn)
        idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo2,md,varn)
        idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)

        odir1 = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/regions' % (se,fo1,md,varn)
        if not os.path.exists(odir1):
            os.makedirs(odir1)
        odir2 = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/regions' % (se,fo2,md,varn)
        if not os.path.exists(odir2):
            os.makedirs(odir2)
        odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/regions' % (se,fo,md,varn)
        if not os.path.exists(odir):
            os.makedirs(odir)

        # ll data
        ds=xr.open_dataset('%s/ll%g.p%g.%s_%s.%s.nc' % (idir1,nt,p,varn,his,se))
        ll1=ds[varn]
        lday=ds['lday']-30
        gpi=ds['gpi']
        nl,nm,ng=ll1.shape
        ll2=xr.open_dataset('%s/ll%g.p%g.%s_%s.%s.nc' % (idir2,nt,p,varn,fut,se))[varn]
        if varn=='pr':
            ll1=86400*ll1
            ll2=86400*ll2

        plot(varn,relb,ll1,ll2)
