import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client
import dask.multiprocessing
import pickle
import numpy as np
import xarray as xr
import constants as c
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid
from etregimes import bestfit

# collect warmings across the ensembles

varn='ooblh'
se='sc'
doy=False

fo = 'historical' # forcing (e.g., ssp245)
byr=[1980,2000]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr=[2080,2100]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr='gwl2.0'

freq='day'

lmd=mods(fo) # create list of ensemble members

def get_fn0(varn,md,fo,byr):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if 'gwl' in byr:
        fn='%s/lm.%s_%s.%s.nc' % (idir,varn,byr,se)
    else:
        fn='%s/lm.%s_%g-%g.%s.nc' % (idir,varn,byr[0],byr[1],se)
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

def get_cer(md):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,'cer')
    if 'gwl' in byr:
        iname='%s/%s.%s.%s.nc' % (idir,'cer',byr,se)
    else:
        iname='%s/%s.%g-%g.%s.nc' % (idir,'cer',byr[0],byr[1],se)
    return xr.open_dataarray(iname)

def calc_blh(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Loading data...')
    ds=xr.open_dataset(get_fn('lh2ce',md))
    gpi=ds['gpi']
    lh2ce=ds['lh2ce']
    print('\n Done.')

    print('\n Computing lh using cer...')
    cer=get_cer(md)

    def blhmon(mon,lh2ce,cer):
        slh2ce=lh2ce.sel(month=[mon]).squeeze()
        slh=slh2ce.copy()
        for igpi in tqdm(range(len(gpi))):
            slh.data[igpi]=cer[mon-1,igpi]*slh2ce[igpi]
        return slh

    with Client(n_workers=12):
        tasks=[dask.delayed(blhmon)(mon,lh2ce,cer) for mon in np.arange(1,13,1)]
        blh=dask.compute(*tasks)
    blh=xr.concat(blh,'month').sortby('month')
    blh=blh.rename(varn)

    # save blh
    oname=get_fn(varn,md)
    blh.to_netcdf(oname,format='NETCDF4')

if __name__=='__main__':
    calc_blh('CESM2')
    # [calc_blh(md) for md in lmd]

# if __name__ == '__main__':
#     with Client(n_workers=len(lmd)):
#         tasks=[dask.delayed(calc_blh)(md) for md in lmd]
#         dask.compute(*tasks)
#         # dask.compute(*tasks,scheduler='single-threaded')
