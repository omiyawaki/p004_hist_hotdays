import os,sys
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
sys.path.append('../')
from util import mods,simu,emem

se='sc'

fo='historical'
yr='1980-2000'

# fo='ssp370'
# yr='gwl2.0'

vn1='tas'
vn2='advt_doy850'
vn='%s+%s'%(vn1,vn2)

def rgr(md):
    # load data
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s'%(se,fo,md)
    v1=xr.open_dataarray('%s/%s/lm.%s_%s.%s.nc'%(idir,vn1,vn1,yr,se))
    v2=xr.open_dataarray('%s/%s/lm.%s_%s.%s.nc'%(idir,vn2,vn2,yr,se))

    # take monthly anomalies
    av1=v1.groupby('time.month')-v1.groupby('time.month').mean('time')
    av2=v2.groupby('time.month')-v2.groupby('time.month').mean('time')
    gpi=av1['gpi']

    # calculate regression slope
    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,vn)
    if not os.path.exists(odir): os.makedirs(odir)

    a=np.empty([12,len(gpi)])
    m=np.arange(1,13,1)
    for i,mn in enumerate(m):
        sv1,sv2=(av1.sel(time=av1['time.month']==mn), av2.sel(time=av2['time.month']==mn))
        a[i,:]=np.nanmean(sv1*sv2,axis=0)/np.nanmean(sv2*sv2,axis=0)

    a=xr.DataArray(a,coords={'month':m,'gpi':gpi},dims=('month','gpi'))
    a=a.rename('%s regressed on %s'%(vn1,vn2))
    a.to_netcdf('%s/rgr.%s.nc'%(odir,vn))

lmd=mods(fo) # create list of ensemble members
[rgr(md) for md in tqdm(lmd)]
