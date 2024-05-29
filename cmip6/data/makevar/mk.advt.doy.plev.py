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
ivar='advt_doy'
varn='%s%g'%(ivar,slev)
gvar='gradt_doy%g'%slev
uvar='ua%g'%slev
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
        ds=xr.open_dataset('%s/doy.%s_%s.%s.nc' % (idir0,gvar,yr,se),format='NETCDF4')
    else:
        ds=xr.open_dataset('%s/doy.%s_%g-%g.%s.nc' % (idir0,gvar,yr[0],yr[1],se),format='NETCDF4')
    dx=ds['dx']
    dy=ds['dy']

    idir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,uvar,md,ens,grd)
    for _,_,files in os.walk(idir):
        for fn in files:
            ofn='%s/%s'%(odir,fn.replace(uvar,varn))
            if checkexist and os.path.isfile(ofn):
                continue
            fn1='%s/%s'%(idir,fn)
            ua=xr.open_dataarray(fn1)
            try:
                fn1=fn1.replace(uvar,vvar)
                ds = xr.open_dataset(fn1)
                va=ds[vvar]
            except:
                fn1=fn1.replace('day','Eday')
                fn1=fn1.replace(uvar,vvar)
                ds = xr.open_dataset(fn1)
                va=ds[vvar]

            # total horizontal advection
            slt=xr.DataArray(ua['time.dayofyear'],dims=['time'],coords={'time':ua['time']})
            sdx=dx.sel(dayofyear=slt)
            sdy=dy.sel(dayofyear=slt)

            adv=c.cpd*(ua*sdx+va*sdy)

            adv=adv.rename(varn)
            adv.to_netcdf(ofn)

calc_adv('UKESM1-0-LL')
# [calc_adv(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_adv,lmd)
