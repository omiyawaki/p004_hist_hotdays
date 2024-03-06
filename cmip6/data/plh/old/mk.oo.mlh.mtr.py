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

disc=True
kv=[0,100] # range of slopes to keep
varn='ooplh_mtr'
varnp='plh'
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

def get_mtr(varn,md,fo,byr):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if 'gwl' in byr:
        fn='%s/%s.%s.%s.nc' % (idir,varn,byr,se)
    else:
        fn='%s/%s.%g-%g.%s.nc' % (idir,varn,byr[0],byr[1],se)
    return fn

def get_fn(varn,md):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if doy:
        px='m.doy'
    else:
        px='m'
    if 'gwl' in byr:
        fn='%s/%s.%s_%s.%s.nc' % (idir,px,varn,byr,se)
    else:
        fn='%s/%s.%s_%g-%g.%s.nc' % (idir,px,varn,byr[0],byr[1],se)
    return fn

def get_fnh(varn,md):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo0,md,varn)
    if doy:
        px='m.doy'
    else:
        px='m'
    if 'gwl' in byr0:
        fn='%s/%s.%s_%s.%s.nc' % (idir,px,varn,byr0,se)
    else:
        fn='%s/%s.%s_%g-%g.%s.nc' % (idir,px,varn,byr0[0],byr0[1],se)
    return fn

def calc_plh(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Loading data...')
    ds=xr.open_dataset(get_fnh('mrsos',md)) # load historical hot sm
    gpi=ds['gpi']
    sm=ds['mrsos']
    # future hot sm
    smf=xr.open_dataarray(get_fn('mrsos',md))
    # mtr
    mtr=xr.open_dataarray(get_mtr('mtr',md,fo0,byr0))
    # discard bad mtr values
    if disc:
        mtr[np.where(mtr<kv[0])]=np.nan
        mtr[np.where(mtr>kv[1])]=np.nan
    # historical ooplh
    lh0=xr.open_dataarray(get_fnh('ooplh',md)) # load historical hot lh
    print('\n Done.')

    print('\n Computing lh using transition regime slope...')
    plh=smf.copy()
    plh.data=lh0.data+mtr*(smf.data-sm.data)

    # save plh
    plh=plh.rename(varnp)
    oname=get_fn(varn,md)
    plh.to_netcdf(oname,format='NETCDF4')

# if __name__=='__main__':
#     calc_plh('CESM2')
#     # [calc_plh(md) for md in lmd]

if __name__=='__main__':
    with Pool(max_workers=len(lmd)) as p:
        p.map(calc_plh,lmd)
