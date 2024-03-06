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

# collect warmings across the ensembles

varn='annai'
se='sc'
nt=7
p=95
doy=False
only95=True

# fo = 'historical' # forcing (e.g., ssp245)
# byr=[1980,2000]

fo = 'ssp370' # forcing (e.g., ssp245)
byr='gwl2.0'

freq='day'

lmd=mods(fo) # create list of ensemble members

def get_fn(varn,md,px='m'):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if doy:
        px='%s.doy'%px
    if 'gwl' in byr:
        fn='%s/%s.%s_%s.%s.nc' % (idir,px,varn,byr,se)
    else:
        fn='%s/%s.%s_%g-%g.%s.nc' % (idir,px,varn,byr[0],byr[1],se)
    return fn

def calc_ai(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Loading data...')
    ds=xr.open_dataset(get_fn('pr',md,px='pc'))
    pct=ds['percentile']
    gpi=ds['gpi']
    ppr=ds['pr']
    prsfc=xr.open_dataarray(get_fn('rsfc',md,px='pc'))
    mpr=xr.open_dataarray(get_fn('pr',md,px='m'))
    mrsfc=xr.open_dataarray(get_fn('rsfc',md,px='m'))
    print('\n Done.')

    # take annual mean
    ppr=ppr.mean('month',keepdims=True)
    prsfc=prsfc.mean('month',keepdims=True)
    mpr=mpr.mean('month',keepdims=True)
    mrsfc=mrsfc.mean('month',keepdims=True)

    print('\n Computing AI...')
    mai=mpr.copy()
    mai.data=0.8*mrsfc.data/(c.Lv*mpr.data)
    pai=ppr.copy()
    pai.data=0.8*prsfc.data/(c.Lv*ppr.data)
    if only95:
        pct=[95]
        pai=pai.sel(percentile=pct)
    print('\n Done.')

    # save ai
    mai=mai.rename(varn)
    pai=pai.rename(varn)
    mai.to_netcdf(get_fn(varn,md,px='m'),format='NETCDF4')
    pai.to_netcdf(get_fn(varn,md,px='pc'),format='NETCDF4')

# if __name__=='__main__':
#     calc_ai('CESM2')
#     # [calc_ai(md) for md in tqdm(lmd)]

if __name__=='__main__':
    with Pool(max_workers=len(lmd)) as p:
        p.map(calc_ai,lmd)
