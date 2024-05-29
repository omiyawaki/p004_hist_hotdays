import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
import constants as c
from tqdm import tqdm
from util import rename_vn

lvn=['T2m'] # input1
byr=[1950,2020]
checkexist=False
se='ts'

# load ocean indices
_,omi=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

for varn in lvn:
    ovn=rename_vn(varn)

    idir='/project/mojave/observations/ERA5_daily/%s' % varn

    odir='/project/amp02/miyawaki/data/p004/era5/%s/%s' % (se,ovn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    if checkexist:
        if os.path.isfile('%s/lm.%s_%g-%g.%s.nc' % (odir,ovn,byr[0],byr[1],se)):
            print('Output file already exists, skipping...')
            continue

    # list of data to load
    lyr=np.arange(byr[0],byr[1],1)
    def stryr(yr):
        if yr<1979:
            return '%s/prelim/%s_%g.nc'%(idir,varn.lower(),yr)
        else:
            return '%s/%s_%g.nc'%(idir,varn.lower(),yr)
    lfn=[stryr(yr) for yr in lyr]

    # load raw data
    print('\n Loading data to composite...')
    ds = xr.open_mfdataset(lfn)
    vn = ds[varn.lower()]
    print('Done')

    # remove ocean points
    time=vn['time']
    vn=vn.data
    vn=np.reshape(vn,(vn.shape[0],vn.shape[1]*vn.shape[2]))
    vn=np.delete(vn,omi,axis=1)
    ngp=vn.shape[1]

    vn=xr.DataArray(vn,coords={'time':time,'gpi':np.arange(ngp)},dims=('time','gpi'))

    vn=vn.rename(ovn)
    vn.to_netcdf('%s/lm.%s_%g-%g.%s.nc' % (odir,ovn,byr[0],byr[1],se),format='NETCDF4')
