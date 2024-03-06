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

varn='twas'

# fo = 'historical' # forcing (e.g., ssp245)

fo = 'ssp370' # forcing (e.g., ssp245)

freq='day'

lmd=mods(fo) # create list of ensemble members

def calc_twas(md):
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
    for _,_,files in os.walk(idir):
        for fn in files:
            ofn='%s/%s'%(odir,fn.replace('tas',varn))
            if os.path.isfile(ofn):
                continue

            print('Loading tas...')
            # surface temp
            fn1='%s/%s'%(idir,fn)
            ds = xr.open_dataset(fn1).load()
            tas=ds['tas']
            print('\nDone...')

            print('Loading huss...')
            # surface sp hum
            fn1=fn1.replace('tas','huss')
            ds = xr.open_dataset(fn1).load()
            huss=ds['huss']
            print('\nDone...')

            print('Interpolating ps...')
            # interpolate to daily ps
            psd=ps.interp_like(tas,kwargs={'fill_value':'extrapolate'})
            print('\nDone...')

            print('Computing wet-bulb temperature...')
            # compute wet-bulb temperature using Newton iteration
            f=lambda Tw,T,q,ps: c.cpd*T+c.Lv*q-c.cpd*Tw-c.ep*c.Lv*esat(Tw)/ps
            fp=lambda Tw,T,q,ps: -c.cpd-c.ep*c.Lv/ps*desat(Tw)
            tw=newton(f,tas,fprime=fp,args=(tas,huss,psd),tol=1e-2)
            print('\nDone...')

            print('Saving output...')
            twas=tas.copy()
            twas.data=tw
            twas=twas.rename(varn)
            twas.to_netcdf(ofn)
            print('\nDone...')

calc_twas('CESM2')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_twas,lmd)

