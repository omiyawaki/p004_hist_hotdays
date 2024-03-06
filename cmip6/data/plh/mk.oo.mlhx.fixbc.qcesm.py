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

ld=[10,100,200,300]
se='sc'
nt=7
doy=False

fo0 = 'historical' # forcing (e.g., ssp245)
byr0=[1980,2000]

fo = 'ssp370' # forcing (e.g., ssp245)
byr='gwl2.0'

freq='day'

md='CESM2'
# lmd=mods(fo) # create list of ensemble members

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

def get_bc(md,fo,byr,bvar):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,bvar)
    iname='%s/%s.%g-%g.%s.pickle' % (idir,bvar,byr[0],byr[1],se)
    return pickle.load(open(iname,'rb'))

def eval_bc(sm,bc):
    return np.interp(sm,bc[0],bc[1])

def calc_mlh(depth):
    ens=emem(md)
    grd=grid(md)
    varn='ooplh_fixbc%g.%s'%(depth,qvar)
    vmrs='mrso%g'%depth
    bvar='bc%g.%s'%(depth,qvar)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Loading data...')
    ds=xr.open_dataset(get_fn(vmrs,md))
    gpi=ds['gpi']
    sm=ds[vmrs]
    print('\n Done.')

    print('\n Computing soil moisture anomaly...')
    sm0=xr.open_dataset(get_fn0(vmrs,md,fo0,byr0))[vmrs]
    # sm=sm.groupby('month')-sm0.groupby('time.month').mean('time')
    sm=sm-sm0.mean('time')
    print('\n Done.')

    print('\n Computing lh using budyko curve...')
    # NEW
    bc=get_bc(md,fo0,byr0,bvar)

    def mlhmon(mon,sm,bc):
        ssm=sm.sel(month=[mon])
        slh=ssm.copy()
        for igpi in tqdm(range(len(gpi))):
            try:
                slh.data[...,igpi]=eval_bc(ssm[...,igpi],bc[mon-1][igpi])
            except:
                slh.data[...,igpi]=np.nan
        return slh

    with Client(n_workers=12):
        tasks=[dask.delayed(mlhmon)(mon,sm,bc) for mon in np.arange(1,13,1)]
        mlh=dask.compute(*tasks)

    mlh=xr.concat(mlh,'month').sortby('month')
    mlh=mlh.rename('plh')

    # save mlh
    oname=get_fn(varn,md)
    mlh.to_netcdf(oname,format='NETCDF4')

# if __name__=='__main__':
#     # calc_mlh('TaiESM1')
#     [calc_mlh(depth) for depth in ld]

if __name__=='__main__':
    with Pool(max_workers=len(ld)) as p:
        p.map(calc_mlh,ld)
