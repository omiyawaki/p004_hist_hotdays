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
# lvn=['huss','pr','mrsos','hfss','hfls','rsds','rsus','rlds','rlus'] # input1
lvn=['mrsos'] # input1
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

def calc_p(md):
    print(md)
    ens=emem(md)
    grd=grid(fo,cl,md)

    for varn in lvn:
        print(varn)

        idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
        odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
        if not os.path.exists(odir):
            os.makedirs(odir)

        if checkexist:
            if 'gwl' in byr:
                if os.path.isfile('%s/semp%s_%s.%s.nc' % (odir,varn,byr,se)):
                    print('Output file already exists, skipping...')
                    continue
            else:
                if os.path.isfile('%s/semp%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se)):
                    print('Output file already exists, skipping...')
                    continue

        # load raw data
        if 'gwl' in byr:
            fn='%s/lm.%s_%s.%s.nc' % (odir,varn,byr,se)
        else:
            fn='%s/lm.%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se)
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
        if 'gwl' in byr:
            idirg='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % ('ts','his+%s'%cl,'historical+%s'%fo,md,'tas')
            [ygwl,gwl]=pickle.load(open('%s/gwl%s.%s.pickle' % (idirg,'tas','ts'),'rb'))
            idx=np.where(gwl==float(byr[-3:]))
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

        # load hot days
        idir0='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,'tas')
        if 'gwl' in byr:
            ds=xr.open_dataset('%s/%s_%s.%s.nc' % (idir0,'tpd',byr,se))
        else:
            ds=xr.open_dataset('%s/%s_%g-%g.%s.nc' % (idir0,'tpd',byr[0],byr[1],se))
        tpd=ds['tpd'].load()

        svnhd=tpd.copy()
        svnhd=svnhd.sel(time=svnhd['time.year']==byr[0])

        tpd.data=np.transpose(vn.data[...,None],[0,3,1,2])*tpd.data
        svnhd.data=tpd.groupby('time.dayofyear').std('time',skipna=True).data/np.sqrt(tpd.groupby('time.dayofyear').count(dim='time').data)
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

        svnhd=svnhd.rename(varn)
        if 'gwl' in byr:
            svnhd.to_netcdf('%s/semp%s_%s.%s.nc' % (odir,varn,byr,se),format='NETCDF4')
        else:
            svnhd.to_netcdf('%s/semp%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se),format='NETCDF4')

# calc_p('CESM2')

if __name__=='__main__':
    with ProgressBar():
        tasks=[dask.delayed(calc_p)(md) for md in lmd]
        # dask.compute(*tasks,scheduler='processes')
        dask.compute(*tasks,scheduler='single-threaded')
