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

varn='lh2ce'
checkexist=True

# fo = 'historical' # forcing (e.g., ssp245)
fo = 'ssp370' # forcing (e.g., ssp245)

freq='day'

lmd=mods(fo) # create list of ensemble members

def calc_lh2ce(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('Loading ps...')
    psdir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,'Amon','ps',md,ens,grd)
    # load monthly ps
    ps=xr.open_mfdataset('%s/*.nc'%psdir)['ps'].load()
    print('\nDone...')


    chk=0
    idir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'tas',md,ens,grd)
    idir0='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'shuss',md,ens,grd)
    idir1='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'huss',md,ens,grd)
    for _,_,files in os.walk(idir):
        for fn in files:
            ofn='%s/%s'%(odir,fn.replace('tas',varn))
            if os.path.isfile(ofn) and checkexist:
                continue

            print('Loading tas...')
            # surface temp
            fn1='%s/%s'%(idir,fn)
            ds = xr.open_dataset(fn1).load()
            tas=ds['tas']
            print('\nDone...')

            print('Loading sfcWind...')
            # surface wind speed
            fn1=fn1.replace('tas','sfcWind')
            ds = xr.open_dataset(fn1).load()
            sfcWind=ds['sfcWind']
            print('\nDone...')

            print('Loading huss...')
            # surface sp hum
            if md=='IPSL-CM6A-LR':
                fn1='%s/%s'%(idir1,fn.replace('tas','huss'))
            else:
                fn1=fn1.replace('tas','huss')
            ds = xr.open_dataset(fn1).load()
            huss=ds['huss']
            print('\nDone...')

            print('Loading shuss...')
            # surface sat sp hum
            fn1='%s/%s'%(idir0,fn.replace('tas','shuss'))
            ds = xr.open_dataset(fn1).load()
            shuss=ds['shuss']
            print('\nDone...')

            print('Interpolating ps...')
            # interpolate to daily ps
            psd=ps.reindex_like(tas,method='nearest')
            print('\nDone...')

            print('Computing density...')
            rho=psd.copy()
            rho.data=psd.data/(c.Rd*tas.data)
            print('\nDone...')

            print('Computing lh2ce...')
            lh2ce=rho.copy()
            lh2ce.data=c.Lv*rho.data*sfcWind.data*(shuss.data-huss.data)
            print('\nDone...')

            print('Saving output...')
            lh2ce=lh2ce.rename(varn)
            lh2ce.to_netcdf(ofn)
            print('\nDone...')

calc_lh2ce('IPSL-CM6A-LR')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_lh2ce,lmd)

