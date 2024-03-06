import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing
from concurrent.futures import ProcessPoolExecutor as Pool
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
import constants as c
from tqdm import tqdm
from util import mods,simu,emem,casename
from glade_utils import grid
from cesmutils import realm,history

# collect warmings across the ensembles

varn='rfa'

# fo = 'historical' # forcing (e.g., ssp245)
fo = 'ssp370' # forcing (e.g., ssp245)

freq='day'

md='CESM2'

ens=emem(md)
grd=grid(md)

odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
if not os.path.exists(odir):
    os.makedirs(odir)

# noncmip cesm data
cname=casename(fo)
rlm=realm('ra')
hst=history('ra')
idir0='/project/amp02/miyawaki/data/share/cesm2/%s/%s/proc/tseries/%s_1' % (cname,rlm,freq)

idir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'stf',md,ens,grd)
lfn=list(os.walk(idir))[0][2]

def calc_rfa(fn):
    sdate=fn[-20:-3]
    # fix end date inconsistencies between raw cesm2 and cmip output
    sdate='20100101-20141231' if sdate=='20100101-20150101' else sdate
    sdate='20950101-21001231' if sdate=='20950101-21010101' else sdate

    ofn='%s/%s'%(odir,fn.replace('stf',varn))
    if os.path.isfile(ofn):
        return

    # load data
    ra=xr.open_dataarray('%s/%s.%s.%s.%s.nc'%(idir0,cname,hst,'RA',sdate))
    stf=xr.open_dataarray('%s/%s'%(idir,fn))
    tend=xr.open_dataarray(('%s/%s'%(idir,fn)).replace('stf','tend'))

    # compute column-integrated mse flux divergence as residual
    rfa=stf.copy()
    rfa.data=ra.data+stf.data-tend.data
    rfa=rfa.rename(varn)
    rfa.to_netcdf(ofn)

# calc_rfa(lfn[0])
[calc_rfa(fn) for fn in tqdm(lfn)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lfn)) as p:
#         p.map(calc_rfa,lfn)
