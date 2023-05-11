import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
import constants as c
from tqdm import tqdm
from cmip6util import mods,simu,emem
from glade_utils import grid
# from metpy.calc import saturation_mixing_ratio,specific_humidity_from_mixing_ratio
# from metpy.units import units

# collect warmings across the ensembles

varn='predts'

# fo = 'historical' # forcing (e.g., ssp245)
# byr=[1980,2000]

fo = 'ssp370' # forcing (e.g., ssp245)
byr=[2080,2100]

freq='day'

lmd=mods(fo) # create list of ensemble members

def calc_predts(md):
    print(md)
    ens=emem(md)
    grd=grid(fo,cl,md)

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    idir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'m500s',md,ens,grd)
    if md=='IPSL-CM6A-LR':
        idir1='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'huss',md,ens,grd)
    else:
        idir1='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'huss',md,ens,grd)
    idir2='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,'fx','orog',md,ens,grd)
    ds = xr.open_mfdataset('%s/*.nc'%(idir1))
    huss=ds['huss']
    print('%s/*.nc'%(idir2))
    ds = xr.open_mfdataset('%s/*.nc'%(idir2))
    orog=ds['orog']
    for _,_,files in os.walk(idir):
        for fn in files:
            fn1='%s/%s'%(idir,fn)
            ds = xr.open_dataset(fn1)
            m500s=ds['m500s']
            print(m500s.shape)
            shuss=huss.sel(time=huss['time']>=m500s['time'][0])
            shuss=shuss.sel(time=shuss['time']<=m500s['time'][-1])
            print(shuss.shape)
            # compute predts
            if md in ['MPI-ESM1-2-LR','MPI-ESM1-2-HR']:
                predts=m500s.copy()
                predts.data=1/c.cpd*(m500s.data-c.Lv*shuss.data-c.g*orog.data)
            else:
                predts=1/c.cpd*(m500s-c.Lv*shuss-c.g*orog)

            predts=predts.rename(varn)
            predts.to_netcdf('%s/%s'%(odir,fn.replace('m500s',varn,1)))

# calc_predts('CESM2')
calc_predts('MPI-ESM1-2-HR')

# if __name__ == '__main__':
#     with ProgressBar():
#         tasks=[dask.delayed(calc_predts)(md) for md in lmd]
#         # dask.compute(*tasks,scheduler='processes')
#         dask.compute(*tasks,scheduler='single-threaded')
