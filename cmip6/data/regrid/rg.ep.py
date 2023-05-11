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
# lpc=np.arange(2.5,97.5+5,5) # percentiles to compute
# lpc=np.insert(lpc,0,2)
# lpc=np.append(lpc,98)
varn='tas' # input1
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
se='sc'

# for regridding
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
ogr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

lmd=mods(fo) # create list of ensemble members

def calc_pvn(md):
    ens=emem(md)
    grd=grid(fo,cl,md)

    idir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)

    odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    if checkexist:
        if 'gwl' in byr:
            if os.path.isfile('%s/m%s_%s.%s.nc' % (odir,varn,byr,se)):
                print('Output file already exists, skipping...')
                return
        else:
            if os.path.isfile('%s/m%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se)):
                print('Output file already exists, skipping...')
                return

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
        # select data within time of interest
        print('\n Selecting data within range of interest...')
        vn=vn.sel(time=vn['time.year']>=byr[0])
        vn=vn.sel(time=vn['time.year']<byr[1])
        otime=vn['time'].sel(time=vn['time.year']==byr[0])
        print('\n Done.')

    # compute daily climatology
    print('\n Computing percentile from larger sample...')
    time=vn['time']
    tday=time.dt.dayofyear.values
    doy=set(tday)
    doy_list=list(doy)
    doy_dict=dict()
    ndays=len(doy_list)-1
    for i,day in enumerate(tqdm(doy_list)):
        use_days=get_window_indices(doy_list,i,nt,nt)
        use_inds=np.concatenate([np.nonzero(tday==j)[0] for j in use_days])
        doy_dict[day]=vn.data[use_inds,...]

    doy_pcts={}
    for day,aggvn in tqdm(doy_dict.items()):
        doy_pcts[day]=get_our_pct(aggvn,lpc)

    # with ProgressBar():
    #     lazy_results=[dask.delayed(get_our_pct)(i) for i in doy_dict]
    # with ProgressBar():
    #     doy_pcts=dask.compute(*lazy_results,scheduler='processes')

    xr_das = []
    for i,(day,pvn) in enumerate(tqdm(doy_pcts.items())):
        xr_das.append(xr.DataArray(pvn, coords={'percentile':lpc, 'lat':gr['lat'], 'lon':gr['lon']}, dims=('percentile', 'lat', 'lon')))

    xr_output = xr.concat(xr_das, dim='time')
    xr_output['time']=otime
    xr_output.name = vn.name
    xr_output.attrs['long_name'] = var_lon_name
    xr_output.attrs['units'] = var_units

    # regrid data
    if md!='CESM2':
        print('\n Regridding...')
        # path to weight file
        wf='%s/wgt.cmip6.%s.%s.cesm2.nc'%(rgdir,md,ty)
        # build regridder with existing weights
        rgd = xe.Regridder(xr_output,ogr, 'bilinear', periodic=True, reuse_weights=True, filename=wf)
        # regrid
        xr_output=rgd(xr_output)
        print('\n Done.')

    if 'gwl' in byr:
        xr_output.to_netcdf('%s/ep%s%03d_%s.%s.nc' % (odir,varn,nt,byr,se),format='NETCDF4')
    else:
        xr_output.to_netcdf('%s/ep%s%03d_%g-%g.%s.nc' % (odir,varn,nt,byr[0],byr[1],se),format='NETCDF4')


# calc_pvn('CESM2')

if __name__=='__main__':
    with ProgressBar():
        tasks=[dask.delayed(calc_pvn)(md) for md in lmd]
        dask.compute(*tasks,scheduler='processes')
