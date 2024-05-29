import os,sys
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
sys.path.append('../../data')
from util import mods
sys.path.append('/home/miyawaki/scripts/common')
from utils import monname,varnlb,unitlb

se='sc'

fo='historical'
yr='1980-2000'

# fo='ssp370'
# yr='gwl2.0'

vn1='tas'
vn2='advt850'
vn='%s+%s'%(vn1,vn2)

lmd=mods(fo)

# (lat,lon) of interest
lat,lon=(70,205)

# identify igpi for select (lat,lon)
llat,llon=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lmilatlon.pickle','rb'))
dlat,dlon=(np.abs(llat-lat), np.abs(llon-lon))
ilat,ilon=(np.where(dlat==dlat.min()), np.where(dlon==dlon.min()))
igpi=np.intersect1d(ilat,ilon)[0]

# colors
lc=pl.cm.twilight(np.linspace(0,1,12))

def rgr(md):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s'%(se,fo)
    # load data
    lv1=[xr.open_dataarray('%s/%s/%s/lm.%s_%s.%s.nc'%(idir,md0,vn1,vn1,yr,se)).isel(gpi=igpi) for md0 in tqdm(lmd)]
    lv2=[xr.open_dataarray('%s/%s/%s/lm.%s_%s.%s.nc'%(idir,md0,vn2,vn2,yr,se)).isel(gpi=igpi) for md0 in tqdm(lmd)]

    # take monthly anomalies
    lav1=[v1.groupby('time.month')-v1.groupby('time.month').mean('time') for v1 in lv1]
    lav2=[v2.groupby('time.month')-v2.groupby('time.month').mean('time') for v2 in lv2]

    # plot scatter
    odir='/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,vn)
    if not os.path.exists(odir): os.makedirs(odir)
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    a=np.empty(12)
    m=np.arange(1,13,1)
    for i,mn in enumerate(tqdm(m)):
        sv1,sv2=([v1.sel(time=v1['time.month']==mn).data.flatten() for v1 in lav1], [v2.sel(time=v2['time.month']==mn).data.flatten() for v2 in lav2])
        sv1=np.concatenate(sv1)
        sv2=np.concatenate(sv2)
        a[i]=np.nanmean(sv1*sv2)/np.nanmean(sv2**2) # slope
        x=np.linspace(sv2.min(),sv2.max(),101)
        ax.plot(sv2,sv1,'.',color=lc[i],markersize=1)
        ax.plot(x,a[i]*x,'-',color=lc[i],label=monname(i))
    ax.set_xlabel(r'$%s$ (%s)'%(varnlb(vn2),unitlb(vn2)))
    ax.set_ylabel(r'$T$ (K)')
    ax.legend(frameon=False,fontsize=7)
    plt.savefig('%s/rgr.%s.%g.%g.png'%(odir,vn,lat,lon),format='png',dpi=600)

    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    ax.plot(m,a)
    ax.set_xlabel('Month')
    ax.set_ylabel(r'$\partial %s / \partial %s$ (%s / %s)'%(varnlb(vn1),varnlb(vn2),unitlb(vn1),unitlb(vn2)))
    plt.savefig('%s/mon.rgr.%s.%g.%g.png'%(odir,vn,lat,lon),format='png',dpi=600)

rgr('mmm')
