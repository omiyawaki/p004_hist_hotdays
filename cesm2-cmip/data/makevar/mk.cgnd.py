import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
from concurrent.futures import ProcessPoolExecutor as Pool
import numpy as np
import xarray as xr
import constants as c
from tqdm import tqdm
from util import casename
from cesmutils import realm,history

# collect warmings across the ensembles

varn='CGND1'
dz1=0.02 # depth of first soil layer [m]
cliq=4.188e3 # specific heat capacity of liquid water [J kg**-1 K**-1]
cice=2.11727e3 # specific heat capacity of ice [J kg**-1 K**-1]

fo = 'historical' # forcing (e.g., ssp245)
sfx='0.9x1.25_hist_78pfts_CMIP6_simyr1850_c190214'

# fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=False
freq='day'

cname=casename(fo)

def calc_cgnd(md):
    rlm=realm(varn.lower())
    hst=history(varn.lower())

    # load dry soil heat capacity and porosity
    csoil=xr.open_dataarray('/project/amp02/miyawaki/data/share/cesm2/inputdata/csoil_%s.nc'%sfx)[[0],...]
    porosat=xr.open_dataarray('/project/amp02/miyawaki/data/share/cesm2/inputdata/porosat_%s.nc'%sfx)[[0],...]

    idir='/project/amp02/miyawaki/data/share/cesm2/%s/%s/proc/tseries/%s_1' % (cname,rlm,freq)
    odir=idir
    for _,_,files in os.walk(idir):
        files=[fn for fn in files if '.SOILLIQ.' in fn]
        for fn in tqdm(files):
            print(fn)
            ofn='%s/%s'%(odir,fn.replace('.SOILLIQ.','.%s.'%varn))
            if checkexist and os.path.isfile(ofn):
                continue
            fn1='%s/%s'%(idir,fn)
            sliq=xr.open_dataset(fn1)['SOILLIQ'][:,0,...]
            fn1=fn1.replace('.SOILLIQ.','.SOILICE.')
            sice=xr.open_dataset(fn1)['SOILICE'][:,0,...]

            cgnd=sliq.copy()
            cgnd.data=csoil.data*(1-porosat.data)+sliq.data/dz1*cliq+sice.data/dz1*cice

            # save
            cgnd=cgnd.rename(varn)
            cgnd.to_netcdf(ofn)

calc_cgnd('CESM2')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_cgnd,lmd)
