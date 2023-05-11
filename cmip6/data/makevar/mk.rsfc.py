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

varn='rsfc'

fo = 'historical' # forcing (e.g., ssp245)

# fo = 'ssp370' # forcing (e.g., ssp245)

freq='day'

lmd=mods(fo) # create list of ensemble members

def calc_rsfc(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    chk=0
    idir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'rsds',md,ens,grd)
    for _,_,files in os.walk(idir):
        for fn in files:
            ofn='%s/%s'%(odir,fn.replace('rsds',varn))
            if os.path.isfile(ofn):
                continue
            fn1='%s/%s'%(idir,fn)
            ds = xr.open_dataset(fn1)
            rsds=ds['rsds']
            fn1=fn1.replace('rsds','rsus')
            ds = xr.open_dataset(fn1)
            rsus=ds['rsus']
            fn1=fn1.replace('rsus','rlds')
            ds = xr.open_dataset(fn1)
            rlds=ds['rlds']
            fn1=fn1.replace('rlds','rlus')
            ds = xr.open_dataset(fn1)
            rlus=ds['rlus']
            # compute net surface radiative heating
            rsfc=rsds.copy()
            rsfc.data=rsds.data-rsus.data+rlds.data-rlus.data
            rsfc=rsfc.rename(varn)
            rsfc.to_netcdf(ofn)

# calc_rsfc('CanESM5')

if __name__ == '__main__':
    with ProgressBar():
        tasks=[dask.delayed(calc_rsfc)(md) for md in lmd]
        dask.compute(*tasks,scheduler='processes')
        # dask.compute(*tasks,scheduler='single-threaded')
