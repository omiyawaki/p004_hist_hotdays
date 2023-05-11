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

varn='huss'

# fo = 'historical' # forcing (e.g., ssp245)
# byr=[1980,2000]

fo = 'ssp370' # forcing (e.g., ssp245)
byr=[2080,2100]

freq='day'

lmd=mods(fo) # create list of ensemble members

def calc_hurs2huss(md):
    ens=emem(md)
    grd=grid(fo,cl,md)

    odir='/project/amp/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    chk=0
    # idir0='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'hurs',md,ens,grd)
    idir0='/project/amp/miyawaki/temp/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'hurs',md,ens,grd)
    idir1='/project/amp/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'shuss',md,ens,grd)
    for _,_,files in os.walk(idir0):
        for fn in files:
            fn0='%s/%s'%(idir0,fn)
            ds = xr.open_dataset(fn0)
            hurs=1e-2*ds['hurs'] # convert to fraction
            fn1='%s/%s'%(idir1,fn.replace('hurs','shuss',1))
            ds = xr.open_dataset(fn1)
            shuss=ds['shuss']
            # compute sp humidity
            sr=shuss/(1-shuss)
            huss=hurs*sr/(1+hurs*sr)
            huss=huss.rename(varn)
            huss.to_netcdf('%s/%s'%(odir,fn.replace('hurs',varn,1)))

calc_hurs2huss('IPSL-CM6A-LR')

# if __name__ == '__main__':
#     with ProgressBar():
#         tasks=[dask.delayed(calc_hurs2huss)(md) for md in lmd]
#         dask.compute(*tasks,scheduler='processes')
