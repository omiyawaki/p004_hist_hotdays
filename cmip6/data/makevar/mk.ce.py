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
from thermo import esat,desat
from glade_utils import grid
from scipy.optimize import newton

# collect warmings across the ensembles

varn='ce'
checkexist=False

# fo = 'historical' # forcing (e.g., ssp245)
fo = 'ssp370' # forcing (e.g., ssp245)

freq='day'

lmd=mods(fo) # create list of ensemble members

def calc_ce(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    idir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'hfls',md,ens,grd)
    idir0='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'lh2ce',md,ens,grd)
    for _,_,files in os.walk(idir):
        for fn in files:
            ofn='%s/%s'%(odir,fn.replace('hfls',varn))
            if os.path.isfile(ofn) and checkexist:
                continue

            print('Loading hfls...')
            fn1='%s/%s'%(idir,fn)
            ds = xr.open_dataset(fn1).load()
            hfls=ds['hfls']
            print('\nDone...')

            print('Loading lh2ce...')
            fn1='%s/%s'%(idir0,fn.replace('hfls','lh2ce'))
            ds = xr.open_dataset(fn1).load()
            lh2ce=ds['lh2ce']
            print('\nDone...')

            print('Computing ce...')
            ce=lh2ce.copy()
            ce.data=hfls.data/lh2ce.data
            # ce.data[np.isinf(ce.data)]=np.nan # remove infs
            ce.data[hfls.data<5]=0 # ignore small LH regions
            ce.data[lh2ce.data<500]=0 # ignore small LH regions
            # ce.data[np.abs(ce.data)>0.5]=np.nan # remove blowups
            print('\nDone...')

            print('Saving output...')
            ce=ce.rename(varn)
            ce.to_netcdf(ofn)
            print('\nDone...')
            sys.exit()

calc_ce('CESM2')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_ce,lmd)
