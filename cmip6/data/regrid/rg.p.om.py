import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client
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
from util import mods,simu,emem
from glade_utils import grid
np.set_printoptions(threshold=sys.maxsize)

# collect warmings across the ensembles

nt=7 # number of days for window (nt days before and after)
lpc=np.concatenate((np.arange(0,50,10),np.arange(50,75,5),np.arange(75,95,2.5),np.arange(95,100,1)))
varn='tas' # input1
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

# for regridding
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
ogr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

lmd=mods(fo) # create list of ensemble members

def load_raw(odir,varn,byr,se):
    # load raw data
    if 'gwl' in byr:
        fn='%s/om.%s_%s.%s.nc' % (odir,varn,byr,se)
    else:
        fn='%s/om.%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se)
    print('\n Loading data to composite...')
    ds = xr.open_dataset(fn)
    print('\n Done.')
    return ds

def calc_pvn(md):
    ens=emem(md)
    grd=grid(md)

    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    if 'gwl' in byr:
        oname='%s/p.om.%s%03d_%s.%s.nc' % (odir,varn,nt,byr,se)
    else:
        oname='%s/p.om.%s%03d_%g-%g.%s.nc' % (odir,varn,nt,byr[0],byr[1],se)

    if checkexist:
        if os.path.isfile(oname):
            print('Output file already exists, skipping...')
            return

    ds=load_raw(odir,varn,byr,se)
    vn = ds[varn].load()

    time_ndx = vn.dims.index('time')
    if 'long_name' in vn.attrs:
        var_lon_name = vn.attrs['long_name']
    else:
        var_lon_name = vn.name
    if 'units' in vn.attrs:
        var_units = vn.attrs['units']
    else:
        logging.warning('NO UNITS ATTACHED TO VARIABLE.')
        var_units = 'N/A'
    ds=None

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
        # select data within time of interest
        print('\n Selecting data within range of interest...')
        vn=vn.sel(time=vn['time.year']>=byr[0])
        vn=vn.sel(time=vn['time.year']<byr[1])
        otime=vn['time'].sel(time=vn['time.year']==byr[0])
        print('\n Done.')

    time=vn['time']
    tday=time.dt.dayofyear.values
    vn=vn.data
    ngp=vn.shape[1]

    # compute daily climatology
    print('\n Computing percentile from larger sample...')

    doy=set(tday)
    doy_list=list(doy)
    doy_dict=dict()
    ndays=len(doy_list)-1
    for i,day in enumerate(tqdm(doy_list)):
        use_days=get_window_indices(doy_list,i,nt,nt)
        use_inds=np.concatenate([np.nonzero(tday==j)[0] for j in use_days])
        doy_dict[day]=vn[use_inds,...]

    ovn=np.empty([len(doy_list),len(lpc),ngp])
    for day,aggvn in tqdm(doy_dict.items()):
        ovn[day-1,...]=get_our_pct(aggvn,lpc)

    ovn=xr.DataArray(ovn,coords={'doy':doy_list,'percentile':lpc,'gpi':np.arange(ngp)},dims=('doy','percentile','gpi'))

    ovn.to_netcdf(oname,format='NETCDF4')

# calc_pvn('CanESM5')

if __name__=='__main__':
    with Client(n_workers=len(lmd)):
        tasks=[dask.delayed(calc_pvn)(md) for md in lmd]
        dask.compute(*tasks)
        # dask.compute(*tasks,scheduler='single-threaded')
