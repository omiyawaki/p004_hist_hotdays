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
from scipy.ndimage import generic_filter1d
from winpct import get_window_indices,get_our_pct
from tqdm import tqdm
from cmip6util import mods,simu,emem
from glade_utils import grid
np.set_printoptions(threshold=sys.maxsize)

# comdsect warmings across the ensembles

nt=7 # half-window size
lvn=['huss','pr','hfss','hfls','rsds','rsus','rlds','rlus']
ty='2d'
checkexist=True

# fo = 'historical' # forcing (e.g., ssp245)
# byr=[1980,2000]

fo = 'ssp370' # forcing (e.g., ssp245)
byr=[2080,2100]

freq='day'
se='sc'

# for regridding
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
ogr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

lmd=mods(fo) # create list of ensemble members

def calc_p(md):
    ens=emem(md)
    grd=grid(fo,cl,md)

    for varn in lvn:
        if md=='IPSL-CM6A-LR' and varn=='huss':
            idir='/project/amp/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
        else:
            idir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)

        odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
        if not os.path.exists(odir):
            os.makedirs(odir)

        if checkexist and os.path.isfile('%s/dsp%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se)):
            print('Output file already exists, skipping...')
            continue

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
        vn=None

        # deseasonalize
        svn.data=svn.groupby('time.dayofyear')-svn.groupby('time.dayofyear').mean('time')

        # load hot days
        idir0='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,'tas')
        ds=xr.open_dataset('%s/%s_%g-%g.%s.nc' % (idir0,'tpd',byr[0],byr[1],se))
        tpd=ds['tpd'].load()

        svnhd=tpd.copy()
        svnhd=svnhd.sel(time=svnhd['time.year']==byr[0])

        tpd.data=np.transpose(svn.data[...,None],[0,3,1,2])*tpd.data
        svnhd.data=tpd.groupby('time.dayofyear').mean('time',skipna=True).data
        # take rolling mean of window size
        svnhd.data = svnhd.rolling(time=2*nt+1,center=True,min_periods=1).mean('time',skipna=True).data

        # regrid data
        if md!='CESM2':
            print('\n Regridding...')
            # path to weight file
            wf='%s/wgt.cmip6.%s.%s.cesm2.nc'%(rgdir,md,ty)
            # build regridder with existing weights
            rgd = xe.Regridder(svnhd,ogr, 'bilinear', periodic=True, reuse_weights=True, filename=wf)
            svnhd=rgd(svnhd)
            print('\n Done.')

        svnhd.to_netcdf('%s/dsp%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se),format='NETCDF4')

# calc_p('IPSL-CM6A-LR')

if __name__=='__main__':
    with ProgressBar():
        tasks=[dask.delayed(calc_p)(md) for md in lmd]
        dask.compute(*tasks,scheduler='processes')
        # dask.compute(*tasks,scheduler='single-threaded')
