import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client
import dask.multiprocessing
from concurrent.futures import ProcessPoolExecutor as Pool
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
import constants as c
from scipy.ndimage import generic_filter1d
from winpct import get_window_indices,get_our_pct
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid
np.set_printoptions(threshold=sys.maxsize)

# comdsect warmings across the ensembles

nt=7 # ndays for percentile window
lvn=['fsm'] # input1
ty='2d'
checkexist=False
doy=False
only95=True

# fo = 'historical' # forcing (e.g., ssp245)
# byr=[1980,2000]

fo = 'ssp370' # forcing (e.g., ssp245)
byr='gwl2.0'
dyr=10

freq='day'
se='sc'

lmd=mods(fo) # create list of ensemble members

def calc_p(md):
    ens=emem(md)
    grd=grid(md)

    for varn in lvn:
        print(md)
        print(varn)

        idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        tdir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,'tas')
        odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        if not os.path.exists(odir):
            os.makedirs(odir)

        if doy:
            if 'gwl' in byr:
                oname='%s/pc.doy.%s_%s.%s.nc' % (odir,varn,byr,se)
            else:
                oname='%s/pc.doy.%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se)
        else:
            if 'gwl' in byr:
                oname='%s/pc.%s_%s.%s.nc' % (odir,varn,byr,se)
            else:
                oname='%s/pc.%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se)

        if checkexist:
            if os.path.isfile(oname):
                print('Output file already exists, skipping...')
                continue

        # load raw data
        if 'gwl' in byr:
            fn='%s/lm.%s_%s.%s.nc' % (idir,varn,byr,se)
        else:
            fn='%s/lm.%s_%g-%g.%s.nc' % (idir,varn,byr[0],byr[1],se)
        print('\n Loading data to composite...')
        ds = xr.open_mfdataset(fn)
        try:
            vn = ds[varn]
        except:
            vn = ds['plh']
        print('\n Done.')

        # select data within time of interest
        if 'gwl' in byr:
            idirg='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % ('ts','historical+%s'%fo,md,'tas')
            [ygwl,gwl]=pickle.load(open('%s/gwl%s.%s.pickle' % (idirg,'tas','ts'),'rb'))
            print(ygwl)
            idx=np.where(gwl==float(byr[-3:]))
            print(idx)
            if ygwl[idx]==1850:
                print('\n %s does not warm to %s K. Skipping...'%(md,byr[-3:]))
                return

            print('\n Selecting data within range of interest...')
            otime=vn['time'].sel(time=vn['time.year']==2080)
            vn=vn.sel(time=vn['time.year']>=ygwl[idx].data-dyr)
            vn=vn.sel(time=vn['time.year']<ygwl[idx].data+dyr)
            print('\n Done.')

        else:
            print('\n Selecting data within range of interest...')
            vn=vn.sel(time=vn['time.year']>=byr[0])
            vn=vn.sel(time=vn['time.year']<byr[1])
            otime=vn['time'].sel(time=vn['time.year']==byr[0])
            print('\n Done.')

        time=vn['time']
        yfull=vn.sel(time=vn['time.dayofyear']==max(vn['time.dayofyear']))['time.year'][0]
        time1=time.sel(time=vn['time.year']==yfull)
        ndy=vn.shape[0]
        ngp=vn.shape[1]

        # compute hot days
        # load temp data
        if 'gwl' in byr:
            fn='%s/lm.%s_%s.%s.nc' % (tdir,'tas',byr,se)
        else:
            fn='%s/lm.%s_%g-%g.%s.nc' % (tdir,'tas',byr[0],byr[1],se)
        print('\n Loading data to composite...')
        ds = xr.open_mfdataset(fn)
        tvn = ds['tas']
        print('\n Done.')

        # load percentile data
        if 'gwl' in byr:
            ds=xr.open_dataset('%s/p.%s%03d_%s.%s.nc' % (tdir,'tas',nt,byr,se))
        else:
            ds=xr.open_dataset('%s/p.%s%03d_%g-%g.%s.nc' % (tdir,'tas',nt,byr[0],byr[1],se))
        pct=ds['percentile']
        try:
            pvn=ds['tas']
        except:
            pvn=ds['__xarray_dataarray_variable__']

        if only95:
            pct=[95]
            pvn=pvn.sel(percentile=pct)

        # replace pvn time with actual dates
        pvn=xr.DataArray(pvn.data,coords={'time':time1,'percentile':pct,'gpi':np.arange(ngp)},dims=('time','percentile','gpi'))

        dep=np.empty([ndy,len(pct),ngp])
        for i,p in enumerate(tqdm(pct)):
            ipvn=pvn.sel(percentile=p)
            asvn=tvn.groupby('time.dayofyear')-ipvn.groupby('time.dayofyear').mean(dim='time')
            bsvn=np.ones_like(asvn.data)
            bsvn[asvn.data<0]=np.nan # i.e. keep days exceeding percentile value

            dep[:,i,...]=bsvn

        dep=xr.DataArray(dep, coords={'time':time,'percentile':pct,'gpi':pvn['gpi']}, dims=('time','percentile','gpi'))

        svnhd=dep.copy()
        if 'gwl' in byr:
            svnhd=svnhd.sel(time=svnhd['time.year']==yfull)
        else:
            svnhd=svnhd.sel(time=svnhd['time.year']==byr[0])

        print('\n Transpose...')
        vn=np.transpose(vn.data[...,None],[0,2,1])
        print('\n Multiply...')
        dep.data=vn*dep.data
        print('\n Computing means conditioned on hot days...')
        svnhd.data=dep.groupby('time.dayofyear').mean('time',skipna=True).data
        # print('\n Rolling mean...')
        # # take rolling mean of window size
        # svnhd.data = svnhd.rolling(time=2*nt+1,center=True,min_periods=1).mean('time',skipna=True).data
        # print('\n Done.')
        if not doy:
            print('\n Monthly mean...')
            svnhd = svnhd.groupby('time.month').mean('time',skipna=True)
            print('\n Done.')
        print(svnhd.shape)

        print('\n Saving data...')
        svnhd=svnhd.rename(varn)
        svnhd.to_netcdf(oname,format='NETCDF4')
        print('\n Done.')

calc_p('CESM2')
# [calc_p(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_p,lmd)

# if __name__=='__main__':
#     with Client(n_workers=len(lmd)):
#         tasks=[dask.delayed(calc_p)(md) for md in lmd]
#         dask.compute(*tasks)
#         # dask.compute(*tasks,scheduler='single-threaded')
