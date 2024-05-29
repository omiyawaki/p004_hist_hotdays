import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
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

# collect warmings across the ensembles

varn='stf'

fo = 'historical' # forcing (e.g., ssp245)
# fo = 'ssp370' # forcing (e.g., ssp245)

freq='day'

lmd=mods(fo) # create list of ensemble members

def calc_stf(md):
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
            ofn='%s/%s'%(odir,fn.replace('hfss',varn))
            if os.path.isfile(ofn):
                continue
            fn1='%s/%s'%(idir0,fn)
            ds = xr.open_dataset(fn1)
            hfss=ds['hfss']
            fn1='%s/%s'%(idir,fn.replace('hfss','hfls',1))
            ds = xr.open_dataset(fn1)
            hfls=ds['hfls']
            # compute surface turbulent flux
            stf=hfss.copy()
            stf.data=hfls.data+hfss.data
            stf=stf.rename(varn)
            stf.to_netcdf(ofn)

calc_stf('KACE-1-0-G')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_stf,lmd)
