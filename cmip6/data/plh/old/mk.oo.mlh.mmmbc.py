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
import xesmf as xe
import xarray as xr
import constants as c
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid
from etregimes import bestfit

# collect warmings across the ensembles

varn='ooplh_mmmbc'
se='sc'
nt=7
doy=False

fo0 = 'historical' # forcing (e.g., ssp245)
byr0=[1980,2000]

# fo = 'historical' # forcing (e.g., ssp245)
# byr=[1980,2000]

fo = 'ssp370' # forcing (e.g., ssp245)
byr=[2080,2100]

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

def get_bc(md):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,'bc')
    iname='%s/%s.%g-%g.%s.pickle' % (idir,'bc',byr[0],byr[1],se)
    return pickle.load(open(iname,'rb'))

def eval_bc(sm,bc):
    return np.interp(sm,bc[0],bc[1])

def calc_mlh(md,bc):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Loading data...')
    ds=xr.open_dataset(get_fn('mrsos',md))
    gpi=ds['gpi']
    sm=ds['mrsos']
    print('\n Done.')

    print('\n Computing soil moisture anomaly...')
    sm0=xr.open_dataset(get_fn0('mrsos',md,fo0,byr0))['mrsos']
    # sm=sm.groupby('month')-sm0.groupby('time.month').mean('time')
    sm=sm-sm0.mean('time')
    print('\n Done.')

    print('\n Computing lh using budyko curve...')

    def mlhmon(mon,sm,bc):
        ssm=sm.sel(month=[mon])
        slh=ssm.copy()
        for igpi in range(len(gpi)):
            try:
                slh.data[...,igpi]=eval_bc(ssm[...,igpi],bc[mon-1][igpi])
            except:
                slh.data[...,igpi]=np.nan
        return slh

    with ProgressBar():
        tasks=[dask.delayed(mlhmon)(mon,sm,bc) for mon in np.arange(1,13,1)]
        mlh=dask.compute(*tasks,scheduler='threads')

    mlh=xr.concat(mlh,'month').sortby('month')
    mlh=mlh.rename(varn)

    # save mlh
    oname=get_fn(varn,md)
    mlh.to_netcdf(oname,format='NETCDF4')

# if __name__=='__main__':
#     calc_mlh('CESM2')
#     # [calc_mlh(md) for md in lmd]

bc=get_bc('mmm')

if __name__ == '__main__':
    with Client(n_workers=len(lmd)):
        tasks=[dask.delayed(calc_mlh)(md,bc) for md in lmd]
        dask.compute(*tasks)
