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
import constants as c
from tqdm import tqdm
from cmip6util import mods,simu,emem
from glade_utils import grid

# colldsect warmings across the ensembles

# lvn=['hfss','hfls','rsds','rsus','rlds','rlus'] # input1
lvn=['huss'] # input1
ty='2d'
checkexist=True

# fo = 'historical' # forcing (e.g., ssp245)
# byr=[1980,2000]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr=[2080,2100]

fo = 'ssp370' # forcing (e.g., ssp245)
byr='gwl2.0'
dyr=10

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
    grd=grid(fo,cl,md)

    for varn in lvn:
        if md=='IPSL-CM6A-LR' and varn=='huss':
            idir='/project/amp/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
        else:
            idir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)

        odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
        if not os.path.exists(odir):
            os.makedirs(odir)

        if checkexist:
            if 'gwl' in byr:
                if os.path.isfile('%s/m%s_%s.%s.nc' % (odir,varn,byr,se)):
                    print('Output file already exists, skipping...')
                    continue
            else:
                if os.path.isfile('%s/m%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se)):
                    print('Output file already exists, skipping...')
                    continue

        # load raw data
        print('%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir,varn,freq,md,fo,ens,grd))
        fn = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir,varn,freq,md,fo,ens,grd)
        print('\n Loading raw data...')
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
            idirg='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % ('ts','his+%s'%cl,'historical+%s'%fo,md,'tas')
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

        # compute daily climatology
        print('\n Computing daily climatology...')
        mvn=vn.groupby('time.dayofyear').mean('time')
        mvn=mvn.rename({'dayofyear':'time'})
        mvn['time']=otime
        print('\n Done.')

        # regrid data
        if md!='CESM2':
            print('\n Regridding...')
            # path to weight file
            wf='%s/wgt.cmip6.%s.%s.cesm2.nc'%(rgdir,md,ty)
            # build regridder with existing weights
            rgd = xe.Regridder(mvn,ogr, 'bilinear', periodic=True, reuse_weights=True, filename=wf)
            # regrid
            mvn=rgd(mvn)
            print('\n Done.')

        if 'gwl' in byr:
            mvn.to_netcdf('%s/m%s_%s.%s.nc' % (odir,varn,byr,se),format='NETCDF4')
        else:
            mvn.to_netcdf('%s/m%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se),format='NETCDF4')

# calc_mvn('MPI-ESM1-2-HR')

if __name__=='__main__':
    with ProgressBar():
        tasks=[dask.delayed(calc_mvn)(md) for md in lmd]
        dask.compute(*tasks,scheduler='processes')
