import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
from concurrent.futures import ProcessPoolExecutor as Pool
import numpy as np
import xarray as xr
import constants as c
from scipy.ndimage import generic_filter1d
from tqdm import tqdm
from sfutil import casename,rename_vn
from cesmutils import realm,history
np.set_printoptions(threshold=sys.maxsize)

# comdsect warmings across the ensembles

# lvn=['qsum'] # input1
lvn=['trefht'] # input1

ty='2d'
checkexist=False

fo = 'lens' # forcing (e.g., ssp245)
byr=[1950,2020]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr='gwl2.0'
# dyr=10

freq='day'
se='sc'

# load ocean indices
_,omi=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

md='CESM2'

cname=casename(fo)

for varn in lvn:
    print(varn)
    rlm=realm(varn)
    hst=history(varn)
    ovn=rename_vn(varn)

    idir='/project/amp02/miyawaki/data/share/cesm2/%s/%s/proc/tseries/%s_1' % (cname,rlm,freq)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,ovn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    if checkexist:
        if 'gwl' in byr:
            if os.path.isfile('%s/lm.%s_%s.%s.nc' % (odir,vn,byr,se)):
                print('Output file already exists, skipping...')
                continue
        else:
            if os.path.isfile('%s/lm.%s_%g-%g.%s.nc' % (odir,vn,byr[0],byr[1],se)):
                print('Output file already exists, skipping...')
                continue

    # load raw data
    fn = '%s/%s.%s.%s.*.nc' % (idir,cname,hst,varn.upper())
    print('\n Loading data to composite...')
    ds = xr.open_mfdataset(fn)
    vn = ds[varn.upper()]
    # convert evapotranspiration to latent heat flux
    if varn in ['qsoil','qvege','qvegt','qsum']:
        vn=1e-3*c.rhow*c.Lv*vn
    print('\n Done.')
    # save grid info
    gr = {}
    gr['lon'] = ds['lon']
    gr['lat'] = ds['lat']
    ds=None

    # select data within time of interest
    if 'gwl' in byr:
        idirg='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % ('ts','historical+%s'%fo,md,'tas')
        [ygwl,gwl]=pickle.load(open('%s/gwl%s.%s.pickle' % (idirg,'tas','ts'),'rb'))
        print(ygwl)
        idx=np.where(gwl==float(byr[-3:]))
        print(idx)
        if ygwl[idx]==1850:
            print('\n %s does not warm to %s K. Skipping...'%(md,byr[-3:]))
            continue

        print('\n Selecting data within range of interest...')
        vn=vn.sel(time=vn['time.year']>=ygwl[idx].data-dyr)
        vn=vn.sel(time=vn['time.year']<ygwl[idx].data+dyr)
        print('\n Done.')

    else:
        yend=byr[1]
        print('\n Selecting data within range of interest...')
        vn=vn.sel(time=vn['time.year']>=byr[0])
        vn=vn.sel(time=vn['time.year']<yend)
        print('\n Done.')

    # remove ocean points
    time=vn['time']
    vn=vn.data
    vn=np.reshape(vn,(vn.shape[0],vn.shape[1]*vn.shape[2]))
    vn=np.delete(vn,omi,axis=1)
    ngp=vn.shape[1]

    vn=xr.DataArray(vn,coords={'time':time,'gpi':np.arange(ngp)},dims=('time','gpi'))

    vn=vn.rename(ovn)
    if 'gwl' in byr:
        vn.to_netcdf('%s/lm.%s_%s.%s.nc' % (odir,ovn,byr,se),format='NETCDF4')
    else:
        vn.to_netcdf('%s/lm.%s_%g-%g.%s.nc' % (odir,ovn,byr[0],byr[1],se),format='NETCDF4')
