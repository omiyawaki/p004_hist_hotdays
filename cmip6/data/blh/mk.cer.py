import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client
import dask.multiprocessing
from concurrent.futures import ProcessPoolExecutor as Pool
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
import constants as c
import statsmodels.api as sm
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid
from etregimes import bestfit

# collect warmings across the ensembles

fo0 = 'historical' # forcing (e.g., ssp245)
byr0=[1980,2000]
varn='cer'
se='sc'

fo = 'historical' # forcing (e.g., ssp245)
byr=[1980,2000]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr=[2080,2100]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr='gwl2.0'

freq='day'

lmd=mods(fo) # create list of ensemble members
# lmd.reverse()

def get_fn(varn,md,fo,byr):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if 'gwl' in byr:
        fn='%s/lm.%s_%s.%s.nc' % (idir,varn,byr,se)
    else:
        fn='%s/lm.%s_%g-%g.%s.nc' % (idir,varn,byr[0],byr[1],se)
    return fn

def calc_cer(md):
    print(md)
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Loading data...')
    ds=xr.open_dataset(get_fn('hfls',md,fo,byr))
    gpi=ds['gpi']
    time=ds['time']
    vn1=ds['hfls']
    vn2=xr.open_dataset(get_fn('lh2ce',md,fo,byr))['lh2ce']
    print('\n Done.')

    print('\n Computing evaporation coefficient...')
    def cermon(mon,vn1,vn2):
        svn1=vn1.sel(time=vn1['time.month']==mon)
        svn2=vn2.sel(time=vn2['time.month']==mon)
        cer=np.nan*np.ones(len(gpi))
        for igpi in tqdm(range(len(gpi))):
            nvn1=svn1.data[...,igpi].flatten()
            nvn2=svn2.data[...,igpi].flatten()
            nans=np.logical_or(np.isnan(nvn1),np.isnan(nvn2))
            nvn1=nvn1[~nans]
            nvn2=nvn2[~nans]
            try:
                lrm=sm.OLS(nvn1,nvn2)
                lrr=lrm.fit()
                cer[igpi]=lrr.params[0]
            except:
                pass
        return cer 

    with Client(n_workers=12):
        tasks=[dask.delayed(cermon)(mon,vn1,vn2) for mon in np.arange(1,13,1)]
        cer=dask.compute(*tasks)
        cer=np.stack(cer)

    # save cer 
    cer=xr.DataArray(cer,coords={'month':np.arange(1,13,1),'gpi':np.arange(len(gpi))},dims=('month','gpi'))
    cer=cer.rename(varn)
    if 'gwl' in byr:
        oname='%s/%s.%s.%s.nc' % (odir,varn,byr,se)
    else:
        oname='%s/%s.%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se)
    cer.to_netcdf(oname,format='NETCDF4')

if __name__=='__main__':
    [calc_cer(md) for md in tqdm(lmd)]
    # calc_cer('CESM2')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_cer,lmd)
