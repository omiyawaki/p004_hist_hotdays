import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
from tqdm import tqdm
from cmip6util import mods,simu,emem
from glade_utils import grid

# collect warmings across the ensembles

varn='tas'
ty='2d'
se='ts'

fo1='historical' # forcing (e.g., ssp245)

fo2='ssp370' # forcing (e.g., ssp245)

fo='%s+%s'%(fo1,fo2)

freq='day'

lmd=mods(fo1) # create list of ensemble members

c0=0 # first loop counter
def calc_gm(md):
    ens=emem(md)
    grd=grid(md)

    idir1='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo1,freq,varn,md,ens,grd)
    idir2='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo2,freq,varn,md,ens,grd)

    odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    # historical temp
    fn1 = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir1,varn,freq,md,fo1,ens,grd)
    ds1 = xr.open_mfdataset(fn1)
    tas1 = ds1[varn].load()
    # create area weights
    cosw=np.cos(np.deg2rad(ds1['lat']))
    # take global mean
    tas1=tas1.weighted(cosw).mean(('lon','lat'))

    # future temp
    fn2 = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir2,varn,freq,md,fo2,ens,grd)
    ds2 = xr.open_mfdataset(fn2)
    tas2 = ds2[varn].load()
    # create area weights
    cosw=np.cos(np.deg2rad(ds2['lat']))
    # take global mean
    tas2=tas2.weighted(cosw).mean(('lon','lat'))
    print(tas2.shape)

    # merge timeseries
    tas=xr.concat([tas1,tas2],dim='time')
    print(tas.shape)

    tas=tas.rename(varn)
    tas.to_netcdf('%s/gm%s.%s.nc' % (odir,varn,se))

if __name__=='__main__':
    with ProgressBar():
        tasks=[dask.delayed(calc_gm)(md) for md in lmd]
        dask.compute(*tasks,scheduler='processes')
        # dask.compute(*tasks,scheduler='single-threaded')
