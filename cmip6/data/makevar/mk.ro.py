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

varn='ti_ro'
se='sc'
nt=7
p=95
doy=False
only95=True

fo = 'historical' # forcing (e.g., ssp245)
byr=[1980,2000]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr='gwl2.0'

freq='day'

lmd=mods(fo) # create list of ensemble members

def get_fn(varn,md,fo,byr):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if 'gwl' in byr:
        fn='%s/lm.%s_%s.%s.nc' % (idir,varn,byr,se)
    else:
        fn='%s/lm.%s_%g-%g.%s.nc' % (idir,varn,byr[0],byr[1],se)
    return fn

def calc_ro(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Loading data...')
    sm=xr.open_dataarray(get_fn('td_mrsos',md,fo,byr))
    pr=xr.open_dataarray(get_fn('ti_pr',md,fo,byr))
    ev=xr.open_dataarray(get_fn('ti_ev',md,fo,byr))
    print('\n Done.')

    print('\n Computing runoff...')
    ro=pr.copy()
    ro.data=pr.data-ev.data-sm.data
    print('\n Done.')

    # save ro
    ro=ro.rename(varn)
    ro.to_netcdf(get_fn(varn,md,fo,byr),format='NETCDF4')

if __name__=='__main__':
    # calc_ro('CESM2')
    [calc_ro(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_ro,lmd)
