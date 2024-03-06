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
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid
from etregimes import bestfit

# collect warmings across the ensembles

varn='ooef2'
se='sc'
nt=7
p=95
doy=False
only95=True

# fo = 'historical' # forcing (e.g., ssp245)
# byr=[1980,2000]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr=[2080,2100]

fo = 'ssp370' # forcing (e.g., ssp245)
byr='gwl2.0'

freq='day'

lmd=mods(fo) # create list of ensemble members

def get_fn0(varn,md,fo,byr):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if 'gwl' in byr:
        fn='%s/lm.%s_%s.%s.nc' % (idir,varn,byr,se)
    else:
        fn='%s/lm.%s_%g-%g.%s.nc' % (idir,varn,byr[0],byr[1],se)
    return fn

def get_fn(varn,md,px='m'):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if doy:
        px='%s.doy'%px
    if 'gwl' in byr:
        fn='%s/%s.%s_%s.%s.nc' % (idir,px,varn,byr,se)
    else:
        fn='%s/%s.%s_%g-%g.%s.nc' % (idir,px,varn,byr[0],byr[1],se)
    return fn

def calc_ef2(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Loading data...')
    ds=xr.open_dataset(get_fn('hfls',md,px='pc'))
    pct=ds['percentile']
    gpi=ds['gpi']
    phfls=ds['hfls']
    pstf=xr.open_dataarray(get_fn('stf',md,px='pc'))
    mhfls=xr.open_dataarray(get_fn('hfls',md,px='m'))
    mstf=xr.open_dataarray(get_fn('stf',md,px='m'))
    print('\n Done.')

    print('\n Computing EF...')
    mef2=mhfls.copy()
    mef2.data=mhfls.data/mstf.data
    pef2=phfls.copy()
    pef2.data=phfls.data/pstf.data
    if only95:
        pct=[95]
        pef2=pef2.sel(percentile=pct)
    print('\n Done.')

    # save ef2
    mef2=mef2.rename(varn)
    pef2=pef2.rename(varn)
    mef2.to_netcdf(get_fn(varn,md,px='m'),format='NETCDF4')
    pef2.to_netcdf(get_fn(varn,md,px='pc'),format='NETCDF4')

if __name__=='__main__':
    # calc_ef2('CESM2')
    [calc_ef2(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_ef2,lmd)
