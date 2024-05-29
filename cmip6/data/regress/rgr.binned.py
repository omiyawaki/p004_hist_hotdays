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
vn2='advty_mon850'
vn='%s+%s'%(vn1,vn2)

def rgr(md):
    # load data
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s'%(se,fo,md)
    pv1=xr.open_dataarray('%s/%s/p.%s_%s.%s.nc'%(idir,vn1,vn1,yr,se))
    pv2=xr.open_dataarray('%s/%s/pc.%s_%s.%s.nc'%(idir,vn2,vn2,yr,se))
    pct1=pv1['percentile']
    pct2=pv2['percentile']

    # take monthly anomalies from median
    # v1=xr.open_dataarray('%s/%s/lm.%s_%s.%s.nc'%(idir,vn1,vn1,yr,se))
    # v2=xr.open_dataarray('%s/%s/lm.%s_%s.%s.nc'%(idir,vn2,vn2,yr,se))
    # mv1=v1.groupby('time.month').mean('time')
    # mv2=v2.groupby('time.month').mean('time')
    mv1=pv1.sel(percentile=50)
    mv2=1/2*(pv2.sel(percentile=47.5)+pv2.sel(percentile=52.5))
    mv1=mv1.expand_dims(dim={'percentile':pct1},axis=1)
    mv2=mv2.expand_dims(dim={'percentile':pct2},axis=1)
    av1=pv1-mv1
    av2=pv2-mv2
    gpi=av1['gpi']

    # calculate regression slope
    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,vn)
    if not os.path.exists(odir): os.makedirs(odir)

    a=np.empty([12,len(gpi)])
    m=np.arange(1,13,1)
    for i,mn in enumerate(m):
        sv1,sv2=(av1.sel(month=mn).data, av2.sel(month=mn).data)
        sv2=1/2*(sv2[1:]+sv2[:-1])
        a[i,:]=np.nanmean(sv1*sv2,axis=0)/np.nanmean(sv2*sv2,axis=0)

    a=xr.DataArray(a,coords={'month':m,'gpi':gpi},dims=('month','gpi'))
    a=a.rename('%s regressed on %s'%(vn1,vn2))
    a.to_netcdf('%s/rgr.%s.nc'%(odir,vn))

lmd=mods(fo) # create list of ensemble members
[rgr(md) for md in tqdm(lmd)]
