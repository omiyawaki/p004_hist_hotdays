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
from scipy.interpolate import interp1d

# collect warmings across the ensembles

slev=850 # in hPa
ivar='wapt'
varn='%s%g'%(ivar,slev)

fo = 'historical' # forcing (e.g., ssp245)
# fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=True
freq='day'

lmd=mods(fo) # create list of ensemble members

def calc_wapt(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    chk=0
    idir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'wap850',md,ens,grd)
    for _,_,files in os.walk(idir):
        for fn in files:
            ofn='%s/%s'%(odir,fn.replace('wap850',varn))
            if checkexist and os.path.isfile(ofn):
                continue
            try:
                fn1='%s/%s'%(idir,fn)
                ds = xr.open_dataset(fn1)
                wap850=ds['wap850']
                fn1=fn1.replace('wap850','ta850')
                ds = xr.open_dataset(fn1)
                ta850=ds['ta850']
                lat=np.deg2rad(ds['lat'].data)
                lon=np.deg2rad(ds['lon'].data)

                # compute wapT
                wapt=wap850*ta850
                wapt=wapt.rename(varn)
                wapt.to_netcdf(ofn)

            except Exception as e:
                print(e)

# calc_wapt('UKESM1-0-LL')

if __name__=='__main__':
    with Pool(max_workers=len(lmd)) as p:
        p.map(calc_wapt,lmd)
