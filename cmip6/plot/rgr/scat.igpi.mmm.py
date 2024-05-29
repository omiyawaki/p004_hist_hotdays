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

def scat(md):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s'%(se,fo)
    # load data
    lv1=[xr.open_dataarray('%s/%s/%s/lm.%s_%s.%s.nc'%(idir,md0,vn1,vn1,yr,se)).isel(gpi=igpi) for md0 in tqdm(lmd)]
    lv2=[xr.open_dataarray('%s/%s/%s/lm.%s_%s.%s.nc'%(idir,md0,vn2,vn2,yr,se)).isel(gpi=igpi) for md0 in tqdm(lmd)]

    # plot scatter
    odir='/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,vn)
    if not os.path.exists(odir): os.makedirs(odir)
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    for i,mn in enumerate(tqdm(np.arange(1,13,1))):
        sv1,sv2=([v1.sel(time=v1['time.month']==mn).data.flatten() for v1 in lv1], [v2.sel(time=v2['time.month']==mn).data.flatten() for v2 in lv2])
        sv1=np.concatenate(sv1)
        sv2=np.concatenate(sv2)
        ax.plot(sv2,sv1,'.',color=lc[i],label=monname(i),markersize=2)
    ax.set_xlabel(r'$%s$ (%s)'%(varnlb(vn2),unitlb(vn2)))
    ax.set_ylabel(r'$T$ (K)')
    ax.legend(frameon=False,fontsize=7)
    plt.savefig('%s/scat.%s.%g.%g.png'%(odir,vn,lat,lon),format='png',dpi=600)

scat('mmm')
