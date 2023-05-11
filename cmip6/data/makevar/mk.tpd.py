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
varn='tas' # input1
ovar='tpd'
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

def calc_tpd(md):
    ens=emem(md)
    grd=grid(fo,cl,md)

    idir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    if checkexist:
        if 'gwl' in byr:
            if os.path.isfile('%s/%s_%s.%s.nc' % (odir,ovar,byr,se)):
                print('Output file already exists, skipping...')
                return
        else:
            if os.path.isfile('%s/%s_%g-%g.%s.nc' % (odir,ovar,byr[0],byr[1],se)):
                print('Output file already exists, skipping...')
                return

    # load raw data
    fn = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir,varn,freq,md,fo,ens,grd)
    print('\n Loading data to composite...')
    ds = xr.open_mfdataset(fn)
    svn = ds[varn].load()
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
        otime=svn['time'].sel(time=svn['time.year']==2080)
        svn=svn.sel(time=svn['time.year']>=ygwl[idx].data-dyr)
        svn=svn.sel(time=svn['time.year']<ygwl[idx].data+dyr)
        print('\n Done.')

    else:
        print('\n Selecting data within range of interest...')
        svn=svn.sel(time=svn['time.year']>=byr[0])
        svn=svn.sel(time=svn['time.year']<byr[1])
        otime=svn['time'].sel(time=svn['time.year']==byr[0])
        print('\n Done.')

    # load percentile data
    if 'gwl' in byr:
        ds=xr.open_dataset('%s/ep%s%03d_%s.%s.native.nc' % (odir,varn,nt,byr,se))
    else:
        ds=xr.open_dataset('%s/ep%s%03d_%g-%g.%s.native.nc' % (odir,varn,nt,byr[0],byr[1],se))
    pct=ds['percentile']
    try:
        pvn=ds[varn]
    except:
        pvn=ds['__xarray_dataarray_variable__']

    lmp=[]
    csvn=np.empty([svn.shape[0],len(pct)-1,svn.shape[1],svn.shape[2]])
    print(csvn.shape)
    for i,p in enumerate(tqdm(pct)):
        imp=i
        if i==0:
            lmp=np.append(lmp,1/2*(0+p))
            ipvn=pvn[:,i,:,:]
            asvn=svn.groupby('time.dayofyear')-ipvn.groupby('time.dayofyear').mean('time')
            bsvn=np.ones_like(asvn.data)
            bsvn[asvn.data>=0]=np.nan
        elif i==len(pct)-1:
            imp=i-1
            lmp=np.append(lmp,1/2*(p+100))
            ipvn=pvn[:,i,:,:]
            asvn=svn.groupby('time.dayofyear')-ipvn.groupby('time.dayofyear').mean('time')
            bsvn=np.ones_like(asvn.data)
            bsvn[asvn.data<0]=np.nan
        elif i==len(pct)-2:
            continue
        else:
            lmp=np.append(lmp,1/2*(pct[i]+pct[i+1]))
            ipvn0=pvn[:,i,:,:]
            ipvn1=pvn[:,i+1,:,:]
            asvn0=svn.groupby('time.dayofyear')-ipvn0.groupby('time.dayofyear').mean('time')
            asvn1=svn.groupby('time.dayofyear')-ipvn1.groupby('time.dayofyear').mean('time')
            bsvn=np.ones_like(asvn.data)
            bsvn[asvn0.data<0]=np.nan
            bsvn[asvn1.data>=0]=np.nan

        csvn[:,imp,...]=bsvn

    csvn=xr.DataArray(csvn, coords={'time':svn['time'],'percentile':lmp,'lat':svn['lat'],'lon':svn['lon']}, dims=('time','percentile', 'lat', 'lon'))
    csvn=csvn.rename(ovar)

    if 'gwl' in byr:
        csvn.to_netcdf('%s/%s_%s.%s.nc' % (odir,ovar,byr,se))
    else:
        csvn.to_netcdf('%s/%s_%g-%g.%s.nc' % (odir,ovar,byr[0],byr[1],se))

# calc_tpd('IPSL-CM6A-LR')

if __name__ == '__main__':
    with ProgressBar():
        tasks=[dask.delayed(calc_tpd)(md) for md in lmd]
        # dask.compute(*tasks,scheduler='processes')
        dask.compute(*tasks,scheduler='single-threaded')
