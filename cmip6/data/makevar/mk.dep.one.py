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
p=95
varn='tas' # input1
ovar='dep'
ty='2d'
checkexist=True

fo = 'historical' # forcing (e.g., ssp245)
byr=[1980,2000]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr=[2080,2100]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr='gwl2.0'
# dyr=10

freq='day'
se='sc'

# for regridding
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
ogr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

lmd=mods(fo) # create list of ensemble members

def calc_dep(md):
    ens=emem(md)
    grd=grid(fo,cl,md)

    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    if 'gwl' in byr:
        ds=xr.open_dataset('%s/%s_%s.%s.nc' % (odir,ovar,byr,se))
    else:
        ds=xr.open_dataset('%s/%s_%g-%g.%s.nc' % (odir,ovar,byr[0],byr[1],se))

    ds=ds.sel(percentile=p)

    if 'gwl' in byr:
        ds.to_netcdf('%s/%s_%s.%s.%02d.nc' % (odir,ovar,byr,se,p))
    else:
        ds.to_netcdf('%s/%s_%g-%g.%s.%02d.nc' % (odir,ovar,byr[0],byr[1],se,p))


calc_dep('CanESM5')

# if __name__ == '__main__':
#     with ProgressBar():
#         tasks=[dask.delayed(calc_dep)(md) for md in lmd]
#         # dask.compute(*tasks,scheduler='processes')
#         dask.compute(*tasks,scheduler='single-threaded')
