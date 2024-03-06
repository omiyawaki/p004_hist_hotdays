import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
from concurrent.futures import ProcessPoolExecutor as Pool
import numpy as np
import xarray as xr
import constants as c
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid

# collect warmings across the ensembles

varn='gflx'

fo = 'historical' # forcing (e.g., ssp245)

# fo = 'ssp370' # forcing (e.g., ssp245)

freq='day'

lmd=mods(fo) # create list of ensemble members

def calc_gflx(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    chk=0
    idir0='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'rsfc',md,ens,grd)
    for _,_,files in os.walk(idir0):
        for fn in files:
            print(fn)
            ofn='%s/%s'%(odir,fn.replace('rsfc',varn))
            if os.path.isfile(ofn):
                continue
            fn1='%s/%s'%(idir0,fn)
            ds = xr.open_dataset(fn1)
            rsfc=ds['rsfc']
            fn1=fn1.replace('rsfc','stf',2)
            ds = xr.open_dataset(fn1)
            stf=ds['stf']
            # compute ground flux
            gflx=rsfc.copy()
            gflx.data=rsfc.data-stf.data
            gflx=gflx.rename(varn)
            gflx.to_netcdf(ofn)

# calc_gflx('CanESM5')

if __name__=='__main__':
    with Pool(max_workers=len(lmd)) as p:
        p.map(calc_gflx,lmd)
