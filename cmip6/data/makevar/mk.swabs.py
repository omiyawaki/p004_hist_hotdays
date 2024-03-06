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

varn='swabs'

fo = 'historical' # forcing (e.g., ssp245)

# fo = 'ssp370' # forcing (e.g., ssp245)

freq='Amon'

lmd=mods(fo) # create list of ensemble members

def calc_swabs(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    chk=0
    idir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'rsds',md,ens,grd)
    for _,_,files in os.walk(idir):
        for fn in files:
            if '.nc' not in fn:
                continue
            ofn='%s/%s'%(odir,fn.replace('rsds',varn))
            if os.path.isfile(ofn):
                continue
            fn1='%s/%s'%(idir,fn)
            ds = xr.open_dataset(fn1)
            rsds=ds['rsds']
            fn1=fn1.replace('rsds','rsus')
            ds = xr.open_dataset(fn1)
            rsus=ds['rsus']
            fn1=fn1.replace('rsus','rsdt')
            ds = xr.open_dataset(fn1)
            rsdt=ds['rsdt']
            fn1=fn1.replace('rsdt','rsut')
            ds = xr.open_dataset(fn1)
            rsut=ds['rsut']
            # compute net surface radiative heating
            swabs=rsds.copy()
            swabs.data=rsdt.data-rsut.data-rsds.data+rsus.data
            swabs=swabs.rename(varn)
            swabs.to_netcdf(ofn)

# calc_swabs('CanESM5')

if __name__=='__main__':
    with Pool(max_workers=len(lmd)) as p:
        p.map(calc_swabs,lmd)
