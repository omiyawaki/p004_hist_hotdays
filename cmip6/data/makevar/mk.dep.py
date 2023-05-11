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
ovar='dep'
ty='2d'
checkexist=False

fo = 'historical' # forcing (e.g., ssp245)
byr=[1980,2000]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr=[2080,2100]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr='gwl2.0'
# dyr=10

freq='day'
se='sc'

# load ocean indices
_,omi=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

lmd=mods(fo) # create list of ensemble members

def calc_dep(md):
    print(md)
    ens=emem(md)
    grd=grid(md)

    idir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    if checkexist:
        if 'gwl' in byr:
            if os.path.isfile('%s/%s_%s.%s.nc' % (odir,ovar,byr,se)):
                print('Output file already exists, skipping...')
                return
        else:
            if os.path.isfile('%s/%s_%g-%g.%s.nc' % (odir,ovar,byr[0],byr[1],se)):
                print('Output file already exists, skipping...')
                return

    # load raw data
    if 'gwl' in byr:
        fn='%s/lm.%s_%s.%s.nc' % (odir,varn,byr,se)
    else:
        fn='%s/lm.%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se)
    print('\n Loading data to composite...')
    ds = xr.open_mfdataset(fn)
    time=ds['time']
    svn = ds[varn].load()
    print('\n Done.')

    # remove ocean points
    svn=svn.data
    svn=np.reshape(svn,(svn.shape[0],svn.shape[1]*svn.shape[2]))
    svn=np.delete(svn,omi,axis=1)
    ndy=svn.shape[0]
    ngp=svn.shape[1]
    svn=xr.DataArray(svn,coords={'time':time,'gpi':np.arange(ngp)},dims=('time','gpi'))

    # load percentile data
    if 'gwl' in byr:
        ds=xr.open_dataset('%s/p.%s%03d_%s.%s.nc' % (odir,varn,nt,byr,se))
    else:
        ds=xr.open_dataset('%s/p.%s%03d_%g-%g.%s.nc' % (odir,varn,nt,byr[0],byr[1],se))
    pct=ds['percentile']
    try:
        pvn=ds[varn]
    except:
        pvn=ds['__xarray_dataarray_variable__']

    # replace pvn time with actual dates
    time1=svn['time'].sel(time=svn['time.year']==svn['time.year'][0])
    pvn=xr.DataArray(pvn.data,coords={'time':time1,'percentile':pct,'gpi':svn['gpi']},dims=('time','percentile','gpi'))

    csvn=np.empty([ndy,len(pct),ngp])
    for i,p in enumerate(tqdm(pct)):
        ipvn=pvn.sel(percentile=p)
        asvn=svn.groupby('time.dayofyear')-ipvn.groupby('time.dayofyear').mean(dim='time')
        bsvn=np.ones_like(asvn.data)
        bsvn[asvn.data<0]=np.nan # i.e. keep days exceeding percentile value

        csvn[:,i,...]=bsvn

    # csvn=xr.DataArray(csvn, coords={'time':svn['time'],'percentile':pct,'gpi':svn['gpi']}, dims=('time','percentile','gpi'))
    # csvn=csvn.rename(ovar)

    if 'gwl' in byr:
        # csvn.to_netcdf('%s/%s_%s.%s.nc' % (odir,ovar,byr,se))
        pickle.dump(csvn,open('%s/%s_%s.%s.pickle' % (odir,ovar,byr,se),'wb'),protocol=5)
    else:
        # csvn.to_netcdf('%s/%s_%g-%g.%s.nc' % (odir,ovar,byr[0],byr[1],se))
        # pickle.dump(csvn,open('%s/%s_%g-%g.%s.pickle' % (odir,ovar,byr[0],byr[1],se),'wb'),protocol=5)
        np.save('%s/%s_%g-%g.%s.npy' % (odir,ovar,byr[0],byr[1],se),csvn)

calc_dep('CESM2')

# if __name__ == '__main__':
#     with ProgressBar():
#         tasks=[dask.delayed(calc_dep)(md) for md in lmd]
#         dask.compute(*tasks,scheduler='processes')
#         # dask.compute(*tasks,scheduler='single-threaded')
