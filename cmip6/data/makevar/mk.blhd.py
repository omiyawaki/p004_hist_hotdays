import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
import constants as c
from winpct import get_window_indices,get_our_pct
from tqdm import tqdm
from cmip6util import mods,simu,emem
from glade_utils import grid
np.set_printoptions(threshold=sys.maxsize)

# colldsect warmings across the ensembles

nt=7 # number of days for window (nt days before and after)
varn='tas' # input1
ovar='blhd'
ty='2d'
skip507599=False

fo = 'historical' # forcing (e.g., ssp245)
byr=[1980,2000]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr=[2080,2100]

freq='day'
se='sc'

# for regridding
rgdir='/project/amp02/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
ogr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

lmd=mods(fo) # create list of ensemble members

def calc_blhd(md):
    ens=emem(md)
    grd=grid(fo,cl,md)

    idir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    # load percentile data
    ds=xr.open_dataset('%s/wp%s%03d_%g-%g.%s.native.nc' % (odir,varn,nt,byr[0],byr[1],se))
    pct=ds['percentile']
    try:
        pvn=ds[varn]
    except:
        pvn=ds['__xarray_dataarray_variable__']

    # load raw data
    fn = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir,varn,freq,md,fo,ens,grd)
    print('\n Loading data to composite...')
    ds = xr.open_mfdataset(fn)
    vn = ds[varn].load()
    print('\n Done.')
    # save grid info
    gr = {}
    gr['lon'] = ds['lon']
    gr['lat'] = ds['lat']
    ds=None

    # select data within time of interest
    print('\n Selecting data within range of interest...')
    svn=vn.sel(time=vn['time.year']>=byr[0])
    svn=svn.sel(time=svn['time.year']<byr[1])
    print('\n Done.')

    for i,p in enumerate(pct):
        if skip507599 and (p==50 or p==75 or p==99):
            continue

        ipvn=pvn[:,i,:,:]
        asvn=svn.groupby('time.dayofyear')-ipvn.groupby('time.dayofyear').mean('time')
        bsvn=np.ones_like(asvn.data)
        bsvn[asvn.data<0]=np.nan
        bsvn[asvn.data>=0]=1
        asvn.data=bsvn

        asvn=asvn.rename(ovar)
        asvn.to_netcdf('%s/%s_%g-%g.%g.%s.nc' % (odir,ovar,byr[0],byr[1],p,se))

if __name__ == '__main__':
    with ProgressBar():
        tasks=[dask.delayed(calc_blhd)(md) for md in lmd]
        # dask.compute(*tasks,scheduler='processes')
        dask.compute(*tasks,scheduler='single-threaded')
