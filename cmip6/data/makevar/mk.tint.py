import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client
import dask.multiprocessing
from concurrent.futures import ProcessPoolExecutor as Pool
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
import constants as c
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid
from etregimes import bestfit

# collect warmings across the ensembles

nd=3 # number of days to integrate
budget='energy'
varn0='fsm'
varn='ti_ev' if varn0=='hfls' and budget=='water' else 'ti_%s'%varn0
se='sc'
doy=False
only95=True

fo0 = 'historical' # forcing (e.g., ssp245)
byr0=[1980,2000]

fo = 'historical' # forcing (e.g., ssp245)
byr=[1980,2000]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr='gwl2.0'

freq='day'

lmd=mods(fo) # create list of ensemble members

def get_fn0(varn,md,fo,byr):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(idir):
        os.makedirs(idir)
    if 'gwl' in byr:
        fn='%s/lm.%s_%s.%s.nc' % (idir,varn,byr,se)
    else:
        fn='%s/lm.%s_%g-%g.%s.nc' % (idir,varn,byr[0],byr[1],se)
    return fn

def calc_tint(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Loading data...')
    vn=xr.open_dataarray(get_fn0(varn0,md,fo,byr))
    vn0=xr.open_dataarray(get_fn0(varn0,md,fo0,byr0)) # historical
    print('\n Done.')

    print('\n Computing anomaly from historical climatology...')
    vn=vn.groupby('time.dayofyear')-vn0.groupby('time.dayofyear').mean('time')
    print('\n Done.')

    print('\n Computing time integral...')
    nvn=vn.data # numpy array
    s=nvn.shape
    svn=np.cumsum(nvn,axis=0)
    ssvn=np.concatenate([np.transpose(np.tile(np.array([np.NaN]*(nd-1)),(s[1],1)),[1,0]), np.tile(np.array([0]),(1,s[1])), svn[:-nd,:]],axis=0)
    tivn=86400*(svn-ssvn) # convert time from day to second
    tivn=tivn/c.Lv if varn0=='hfls' and budget=='water' else tivn # convert LH to E
    print('\n Done.')

    # save
    vn.data=tivn
    vn=vn.rename(varn)
    vn.to_netcdf(get_fn0(varn,md,fo,byr),format='NETCDF4')

if __name__=='__main__':
    calc_tint('CESM2')
    # [calc_tint(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_tint,lmd)
