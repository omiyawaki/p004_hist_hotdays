import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
from concurrent.futures import ProcessPoolExecutor as Pool
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid

# collect warmings across the ensembles

varn='tas'
ty='2d'
se='ts'

fo1='historical' # forcing (e.g., ssp245)
fo2='ssp370' # forcing (e.g., ssp245)
byr=[1950,2020]

fo='%s+%s'%(fo1,fo2)

freq='day'

lmd=mods(fo1) # create list of ensemble members

# load ocean indices
_,omi=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# for regridding
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
ogr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

def calc_ts(md):
    ens=emem(md)
    grd=grid(md)

    idir1='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo1,freq,varn,md,ens,grd)
    idir2='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo2,freq,varn,md,ens,grd)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    # historical temp
    fn1 = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir1,varn,freq,md,fo1,ens,grd)
    vn1 = xr.open_mfdataset(fn1)[varn]

    # future temp
    fn2 = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir2,varn,freq,md,fo2,ens,grd)
    vn2 = xr.open_mfdataset(fn2)[varn]

    # merge timeseries
    vn=xr.concat([vn1,vn2],dim='time')

    # select data within time of interest
    print('\n Selecting data within range of interest...')
    vn=vn.sel(time=vn['time.year']>=byr[0])
    vn=vn.sel(time=vn['time.year']<byr[1])
    print('\n Done.')

    # regrid data
    if md!='CESM2':
        print('\n Regridding...')
        # path to weight file
        wf='%s/wgt.cmip6.%s.%s.cesm2.nc'%(rgdir,md,ty)
        # build regridder with existing weights
        rgd = xe.Regridder(vn,ogr, 'bilinear', periodic=True, reuse_weights=True, filename=wf)
        vn=rgd(vn)
        print('\n Done.')

    # remove ocean points
    time=vn['time']
    vn=vn.data
    vn=np.reshape(vn,(vn.shape[0],vn.shape[1]*vn.shape[2]))
    vn=np.delete(vn,omi,axis=1)
    ngp=vn.shape[1]

    vn=xr.DataArray(vn,coords={'time':time,'gpi':np.arange(ngp)},dims=('time','gpi'))

    vn=vn.rename(varn)
    vn.to_netcdf('%s/lm.%s.%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se))

calc_ts('CESM2')
# [calc_ts(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_ts,lmd)
