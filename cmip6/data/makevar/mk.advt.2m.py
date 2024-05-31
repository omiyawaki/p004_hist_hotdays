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

slev='s'
ivar='advt'
varn='%s%g'%(ivar,slev)
gvar='gradt%g'%slev
uvar='ua%g'%slev
vvar='va%g'%slev

# fo = 'historical' # forcing (e.g., ssp245)
fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=False
freq='Eday'

lmd=mods(fo) # create list of ensemble members

def calc_adv(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    chk=0
    idir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,gvar,md,ens,grd)
    for _,_,files in os.walk(idir):
        for fn in files:
            ofn='%s/%s'%(odir,fn.replace(gvar,varn))
            if checkexist and os.path.isfile(ofn):
                continue
            try:
                fn1='%s/%s'%(idir,fn)
                ds = xr.open_dataset(fn1)
                dx=ds['dx']
                dy=ds['dy']
                try:
                    fn1=fn1.replace(gvar,uvar)
                    ds = xr.open_dataset(fn1)
                    ua=ds[uvar]
                    fn1=fn1.replace(uvar,vvar)
                    ds = xr.open_dataset(fn1)
                    va=ds[vvar]
                except:
                    fn1=fn1.replace('day','Eday')
                    fn1=fn1.replace(gvar,uvar)
                    ds = xr.open_dataset(fn1)
                    ua=ds[uvar]
                    fn1=fn1.replace(uvar,vvar)
                    ds = xr.open_dataset(fn1)
                    va=ds[vvar]

                # total horizontal advection
                adv=c.cpd*(ua*dx+va*dy)

                adv=adv.rename(varn)
                adv.to_netcdf(ofn)
            except Exception as e:
                print(e)
                print('WARNING skipping %s'%ofn)

calc_adv('UKESM1-0-LL')
# [calc_adv(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_adv,lmd)