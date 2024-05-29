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

se='sc'
slev=925 # in hPa
ivar='advty_mon'
varn='%s%g'%(ivar,slev)
gvar='gradt_mon%g'%slev
vvar='va%g'%slev

fo='historical' # forcing (e.g., ssp245)
yr=[1980,2000]

# fo='ssp370' # forcing (e.g., ssp245)
# yr='gwl2.0'

checkexist=False
freq='Eday'

lmd=mods(fo) # create list of ensemble members

def calc_adv(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    # load gradt
    idir0='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,gvar)
    if 'gwl' in yr:
        ds=xr.open_dataset('%s/mon.%s_%s.%s.nc' % (idir0,gvar,yr,se),format='NETCDF4')
    else:
        ds=xr.open_dataset('%s/mon.%s_%g-%g.%s.nc' % (idir0,gvar,yr[0],yr[1],se),format='NETCDF4')
    dy=ds['dy']

    idir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,vvar,md,ens,grd)
    for _,_,files in os.walk(idir):
        for fn in files:
            ofn='%s/%s'%(odir,fn.replace(vvar,varn))
            if checkexist and os.path.isfile(ofn):
                continue
            fn1='%s/%s'%(idir,fn)
            va=xr.open_dataarray(fn1)

            # total horizontal advection
            slt=xr.DataArray(va['time.month'],dims=['time'],coords={'time':va['time']})
            sdy=dy.sel(month=slt)

            adv=c.cpd*(va*sdy)

            adv=adv.rename(varn)
            adv.to_netcdf(ofn)

calc_adv('UKESM1-0-LL')
# [calc_adv(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_adv,lmd)
