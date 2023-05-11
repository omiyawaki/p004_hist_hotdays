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

varn='ef2'

fo = 'historical' # forcing (e.g., ssp245)

# fo = 'ssp370' # forcing (e.g., ssp245)

freq='day'

lmd=mods(fo) # create list of ensemble members

def calc_ef2(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    chk=0
    idir0='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'hfss',md,ens,grd)
    idir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'hfls',md,ens,grd)
    for _,_,files in os.walk(idir0):
        for fn in files:
            print(fn)
            ofn='%s/%s'%(odir,fn.replace('hfss',varn))
            if os.path.isfile(ofn):
                continue
            fn1='%s/%s'%(idir0,fn)
            ds = xr.open_dataset(fn1)
            hfss=ds['hfss']
            fn1='%s/%s'%(idir,fn.replace('hfss','hfls',1))
            ds = xr.open_dataset(fn1)
            hfls=ds['hfls']
            # compute evaporative fraction (EF)
            ef2=hfss.copy()
            ef2.data=hfls.data/(hfls.data+hfss.data)
            ef2=ef2.rename(varn)
            ef2.to_netcdf(ofn)

# calc_ef2('CanESM5')

if __name__ == '__main__':
    with ProgressBar():
        tasks=[dask.delayed(calc_ef2)(md) for md in lmd]
        dask.compute(*tasks,scheduler='processes')
        # dask.compute(*tasks,scheduler='single-threaded')
