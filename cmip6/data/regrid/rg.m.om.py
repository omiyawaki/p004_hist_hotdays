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

# colldsect warmings across the ensembles

lvn=['tas'] # input1
# lvn=['huss','pr','mrsos','hfss','hfls','rsds','rsus','rlds','rlus'] # input1
ty='2d'
checkexist=False
doy=False

fo = 'historical' # forcing (e.g., ssp245)
byr=[1980,2000]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr=[2080,2100]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr='gwl2.0'
# dyr=10

freq='day'
se = 'sc' # season (ann, djf, mam, jja, son)

# for regridding
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
ogr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

lmd=mods(fo) # create list of ensemble members

c0=0 # first loop counter

def calc_mvn(md):
    ens=emem(md)
    grd=grid(md)

    for varn in lvn:

        idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        if not os.path.exists(odir):
            os.makedirs(odir)

        if doy:
            if 'gwl' in byr:
                oname='%s/m.om.doy.%s_%s.%s.nc'%(odir,varn,byr,se)
            else:
                oname='%s/m.om.doy.%s_%g-%g.%s.nc'%(odir,varn,byr[0],byr[1],se)
        else:
            if 'gwl' in byr:
                oname='%s/m.om.%s_%s.%s.nc'%(odir,varn,byr,se)
            else:
                oname='%s/m.om.%s_%g-%g.%s.nc'%(odir,varn,byr[0],byr[1],se)

        if checkexist:
            if os.path.isfile(oname):
                print('Output file already exists, skipping...')
                continue

        # load raw data
        if 'gwl' in byr:
            fn='%s/om.%s_%s.%s.nc' % (idir,varn,byr,se)
        else:
            fn='%s/om.%s_%g-%g.%s.nc' % (idir,varn,byr[0],byr[1],se)
        print('\n Loading raw data...')
        ds = xr.open_mfdataset(fn)
        try:
            vn = ds[varn]
        except:
            vn = ds['plh']
        print('\n Done.')
        # # save grid info
        # gr = {}
        # gr['lon'] = ds['lon']
        # gr['lat'] = ds['lat']
        # ds=None

        # select data within time of interest
        if 'gwl' in byr:
            idirg='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % ('ts','historical+%s'%fo,md,'tas')
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

        if doy:
            # compute daily climatology
            print('\n Computing daily climatology...')
            mvn=vn.groupby('time.dayofyear').mean('time')
            mvn=mvn.rename({'dayofyear':'time'})
            mvn['time']=otime
            print('\n Done.')
        else:
            # compute monthly climatology
            print('\n Computing monthly climatology...')
            mvn=vn.groupby('time.month').mean('time')
            print('\n Done.')

        # # regrid data
        # if md!='CESM2':
        #     print('\n Regridding...')
        #     # path to weight file
        #     wf='%s/wgt.cmip6.%s.%s.cesm2.nc'%(rgdir,md,ty)
        #     # build regridder with existing weights
        #     rgd = xe.Regridder(mvn,ogr, 'bilinear', periodic=True, reuse_weights=True, filename=wf)
        #     # regrid
        #     mvn=rgd(mvn)
        #     print('\n Done.')

        mvn.to_netcdf(oname,format='NETCDF4')

# calc_mvn('CESM2')

if __name__=='__main__':
    with Pool(max_workers=len(lmd)) as p:
        p.map(calc_mvn,lmd)

# if __name__=='__main__':
#     with Client(n_workers=len(lmd)):
#         tasks=[dask.delayed(calc_mvn)(md) for md in lmd]
#         dask.compute(*tasks)
