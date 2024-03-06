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

varn='ooplh_orig'
se='sc'
nt=7
p=95
doy=False
only95=True

fo = 'historical' # forcing (e.g., ssp245)
byr=[1980,2000]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr=[2080,2100]

freq='day'

lmd=mods(fo) # create list of ensemble members

def get_fn(varn,md):
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

def get_bc(md):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,'bc')
    iname='%s/%s.%g-%g.%s.pickle' % (idir,'bc_orig',byr[0],byr[1],se)
    return pickle.load(open(iname,'rb'))

def eval_bc(sm,bc):
    return np.interp(sm,bc[0],bc[1])

def calc_plh(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Loading data...')
    ds=xr.open_dataset(get_fn('mrsos',md))
    pct=ds['percentile']
    gpi=ds['gpi']
    sm=ds['mrsos']
    if only95:
        pct=[95]
        sm=sm.sel(percentile=pct)
    print('\n Done.')

    print('\n Computing lh using budyko curve...')
    bc=get_bc(md)

    def plhmon(mon,sm,bc):
        ssm=sm.sel(month=[mon])
        slh=ssm.copy()
        for igpi in tqdm(range(len(gpi))):
            try:
                slh.data[...,igpi]=eval_bc(ssm[...,igpi],bc[mon-1][igpi])
            except:
                slh.data[...,igpi]=np.nan
        return slh

    ooplh=[]
    for ip,p in enumerate(pct):
        psm=sm.sel(percentile=[p])
        with ProgressBar():
            tasks=[dask.delayed(plhmon)(mon,psm,bc) for mon in np.arange(1,13,1)]
            plh=dask.compute(*tasks,scheduler='threads')

        plh=xr.concat(plh,'month').sortby('month')
        plh=plh.rename(varn)
        ooplh.append(plh)

    ooplh=xr.concat(ooplh,'percentile').sortby('percentile')

    # save plh
    oname=get_fn(varn,md)
    plh.to_netcdf(oname,format='NETCDF4')

# if __name__=='__main__':
#     calc_plh('TaiESM1')
#     # [calc_plh(md) for md in lmd]

if __name__ == '__main__':
    with Client():
        tasks=[dask.delayed(calc_plh)(md) for md in lmd]
        dask.compute(*tasks)
        # dask.compute(*tasks,scheduler='single-threaded')
