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

varn='oopef3'
bct='bcef3'
se='sc'
nt=7
doy=False

fo0 = 'historical' # forcing (e.g., ssp245)
byr0=[1980,2000]

# fo = 'historical' # forcing (e.g., ssp245)
# byr=[1980,2000]

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
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,bct)
    if 'gwl' in byr:
        iname='%s/%s.%s.%s.pickle' % (idir,bct,byr,se)
    else:
        iname='%s/%s.%g-%g.%s.pickle' % (idir,bct,byr[0],byr[1],se)
    return pickle.load(open(iname,'rb'))

def eval_bc(sm,bc):
    return np.interp(sm,bc[0],bc[1])

def calc_mef(md):
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

    print('\n Computing ef using budyko curve...')
    bc=get_bc(md)

    def mefmon(mon,sm,bc):
        ssm=sm.sel(month=[mon])
        sef=ssm.copy()
        for igpi in range(len(gpi)):
            try:
                sef.data[...,igpi]=eval_bc(ssm[...,igpi],bc[mon-1][igpi])
            except:
                sef.data[...,igpi]=np.nan
        return sef

    with Client(n_workers=12):
        tasks=[dask.delayed(mefmon)(mon,sm,bc) for mon in np.arange(1,13,1)]
        mef=dask.compute(*tasks)

    mef=xr.concat(mef,'month').sortby('month')
    mef=mef.rename(varn)

    # save mef
    oname=get_fn(varn,md)
    mef.to_netcdf(oname,format='NETCDF4')

if __name__=='__main__':
    # calc_mef('CESM2')
    [calc_mef(md) for md in tqdm(lmd)]
