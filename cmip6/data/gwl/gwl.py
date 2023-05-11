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
gwl=np.array([1.5,2,3,4]) # global warming levels

fo1='historical' # forcing (e.g., ssp245)

fo2='ssp370' # forcing (e.g., ssp245)

fo='%s+%s'%(fo1,fo2)
cl='%s+%s'%(cl1,cl2)

freq='day'

lmd=mods(fo1) # create list of ensemble members

def calc_gwl(md):
    ens=emem(md)
    grd=grid(fo1,cl1,md)

    idir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
    odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # load tas timeseries
    fn = '%s/gm%s.%s.nc' % (odir,varn,se)
    ds = xr.open_mfdataset(fn)
    tas = ds[varn].load()

    # annual means
    tas=tas.resample(time='1Y').mean(dim='time')

    # 1850-1900 baseline mean
    tas0=tas.sel(time=tas['time.year']>=1850)
    tas0=tas0.sel(time=tas0['time.year']<1900)
    tas0=tas0.mean(dim='time')

    # 20-year rolling means
    tas=tas.rolling(time=20+1,center=True,min_periods=1).mean('time',skipna=True)

    # warming
    dtas=tas-tas0

    # compute year corresponding to GWLs
    ddtas=dtas.data[:,None]-np.transpose(gwl[:,None])
    iygwl=np.argmax(ddtas>0,axis=0)
    ygwl=dtas['time.year'].isel(time=iygwl)
    print(ygwl)

    pickle.dump([ygwl,gwl],open('%s/gwl%s.%s.pickle' % (odir,varn,se),'wb'),protocol=5)

# calc_gwl('MPI-ESM1-2-HR')

if __name__=='__main__':
    with ProgressBar():
        tasks=[dask.delayed(calc_gwl)(md) for md in lmd]
        # dask.compute(*tasks,scheduler='processes')
        dask.compute(*tasks,scheduler='single-threaded')
