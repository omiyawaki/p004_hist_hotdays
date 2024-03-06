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

varn='oopef_rddsm'
se='sc'
nt=7
p=95
doy=False
only95=True

fo0 = 'historical' # forcing (e.g., ssp245)
byr0=[1980,2000]

fo = 'ssp370' # forcing (e.g., ssp245)
byr='gwl2.0'

freq='day'

lmd=mods(fo) # create list of ensemble members

def get_fn(varn,fo,byr,md):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if doy:
        px='pc.doy'
    else:
        px='pc'
    if 'gwl' in byr:
        fn='%s/%s.%s_%s.%s.nc' % (idir,px,varn,byr,se)
    else:
        fn='%s/%s.%s_%g-%g.%s.nc' % (idir,px,varn,byr[0],byr[1],se)
    return fn

def calc_pef(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Loading data...')
    ef0=xr.open_dataarray(get_fn('oopef',fo0,byr0,md))
    effbc=xr.open_dataarray(get_fn('oopef_fixbc',fo,byr,md))
    effsm=xr.open_dataarray(get_fn('oopef_fixmsm',fo,byr,md))
    print('\n Done.')

    print('\n Computing residual ef...')
    pef=effbc.copy()
    pef.data=ef0.data+effbc.data-effsm.data
    pef=pef.rename('pef')

    # save pef
    oname=get_fn(varn,fo,byr,md)
    pef.to_netcdf(oname,format='NETCDF4')

if __name__=='__main__':
    calc_pef('CESM2')
    # [calc_pef(md) for md in lmd]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_pef,lmd)
