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
from mrsos import bestfit

se='sc'

fo='historical'
yr='1980-2000'

# fo='ssp370'
# yr='gwl2.0'

vn1='tas'
vn2='mrsos'
vn='%s+%s'%(vn1,vn2)

# (lat,lon) of interest
# lat,lon=(70,205)
lat,lon=(18,102)

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
    v1=xr.open_dataarray('%s/%s/lm.%s_%s.%s.nc'%(idir,vn1,vn1,yr,se)).isel(gpi=igpi)
    v2=xr.open_dataarray('%s/%s/lm.%s_%s.%s.nc'%(idir,vn2,vn2,yr,se)).isel(gpi=igpi)

    # plot regression slope
    odir='/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,vn)
    if not os.path.exists(odir): os.makedirs(odir)

    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    a=np.empty(12)
    m=np.arange(1,13,1)
    for i,mn in enumerate(tqdm(m)):
        sv1,sv2=(v1.sel(time=v1['time.month']==mn), v2.sel(time=v2['time.month']==mn))
        fit=bestfit(sv2,sv1)
        [X,Y]=fit['line']
        res=sv1-fit['yp']
        if i==0:
            lres=res
        else:
            lres=xr.concat((lres,res),dim='time')
        x=np.linspace(sv2.min(),sv2.max(),101)
        ax.plot(sv2,sv1,'.',color=lc[i],markersize=2)
        ax.plot(X,Y,'-',color=lc[i],label=monname(i))
    ax.set_xlabel(r'$%s$ (%s)'%(varnlb(vn2),unitlb(vn2)))
    ax.set_ylabel(r'$%s$ (%s)'%(varnlb(vn1),unitlb(vn1)))
    ax.legend(frameon=False,fontsize=7)
    plt.savefig('%s/pw.rgr.%s.%g.%g.png'%(odir,vn,lat,lon),format='png',dpi=600)

    # residual distribution
    k=gaussian_kde(res.data.flatten())
    r=np.linspace(res.min(),res.max(),101)
    pdf=k(r)
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    ax.plot(r,pdf)
    ax.set_xlabel('Residual (K)')
    ax.set_ylabel(r'Probability density')
    plt.savefig('%s/pw.pdf.res.%s.%g.%g.png'%(odir,vn,lat,lon),format='png',dpi=600)

rgr('CESM2')
