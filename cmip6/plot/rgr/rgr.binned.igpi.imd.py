import os,sys
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from scipy.stats import gaussian_kde
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
sys.path.append('/home/miyawaki/scripts/common')
from utils import monname,varnlb,unitlb

se='sc'

fo='historical'
yr='1980-2000'

# fo='ssp370'
# yr='gwl2.0'

vn1='tas'
vn2='advt_mon850'
vn='%s+%s'%(vn1,vn2)

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
    # load data
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s'%(se,fo,md)
    pv1=xr.open_dataarray('%s/%s/p.%s_%s.%s.nc'%(idir,vn1,vn1,yr,se)).isel(gpi=igpi)
    pv2=xr.open_dataarray('%s/%s/pc.%s_%s.%s.nc'%(idir,vn2,vn2,yr,se)).isel(gpi=igpi)
    pct1=pv1['percentile']
    pct2=pv2['percentile']
    # v1=xr.open_dataarray('%s/%s/lm.%s_%s.%s.nc'%(idir,vn1,vn1,yr,se)).isel(gpi=igpi)
    # v2=xr.open_dataarray('%s/%s/lm.%s_%s.%s.nc'%(idir,vn2,vn2,yr,se)).isel(gpi=igpi)
    if 'advt' in vn2:
        pv2=-pv2

    # take monthly anomalies
    mv1=pv1.sel(percentile=50)
    mv2=1/2*(pv2.sel(percentile=47.5)+pv2.sel(percentile=52.5))
    mv1=mv1.expand_dims(dim={'percentile':pct1},axis=1)
    mv2=mv2.expand_dims(dim={'percentile':pct2},axis=1)
    av1=pv1-mv1
    av2=pv2-mv2

    # plot regression slope
    odir='/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,vn)
    if not os.path.exists(odir): os.makedirs(odir)

    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    a=np.empty(12)
    m=np.arange(1,13,1)
    for i,mn in enumerate(tqdm(m)):
        sv1,sv2=(av1.sel(month=mn).data, av2.sel(month=mn).data)
        sv2=1/2*(sv2[1:]+sv2[:-1])
        a[i]=sv2.dot(sv1)/sv2.dot(sv2) # slope
        res=sv1-a[i]*sv2
        if i==0:
            lres=res
        else:
            lres=np.concatenate((lres,res))
        x=np.linspace(sv2.min(),sv2.max(),101)
        ax.plot(sv2,sv1,'+',color=lc[i],markersize=6)
        ax.plot(x,a[i]*x,'-',color=lc[i],label=monname(i))
    ax.set_xlabel(r'$\delta %s$ (%s)'%(varnlb(vn2),unitlb(vn2)))
    ax.set_ylabel(r'$\delta %s$ (%s)'%(varnlb(vn1),unitlb(vn1)))
    ax.legend(frameon=False,fontsize=7)
    plt.savefig('%s/rgr.%s.%g.%g.png'%(odir,vn,lat,lon),format='png',dpi=600)

    # residual distribution
    k=gaussian_kde(res.flatten())
    r=np.linspace(res.min(),res.max(),101)
    pdf=k(r)
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    ax.plot(r,pdf)
    ax.set_xlabel('Residual (K)')
    ax.set_ylabel(r'Probability density')
    plt.savefig('%s/pdf.res.%s.%g.%g.png'%(odir,vn,lat,lon),format='png',dpi=600)

    # regression slope vs month
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    ax.plot(m,a)
    ax.set_xlabel('Month')
    ax.set_ylabel(r'$\partial %s / \partial %s$ (%s / %s)'%(varnlb(vn1),varnlb(vn2),unitlb(vn1),unitlb(vn2)))
    plt.savefig('%s/mon.rgr.%s.%g.%g.png'%(odir,vn,lat,lon),format='png',dpi=600)

rgr('CESM2')
