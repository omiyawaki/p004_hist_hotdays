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
from scipy.signal import butter,sosfiltfilt
from winpct import get_window_indices,get_our_pct
from tqdm import tqdm
from util import mods,simu,emem,load_raw
from glade_utils import grid,listyr,ishr
np.set_printoptions(threshold=sys.maxsize)

# comdsect warmings across the ensembles

nt=30 # half-window size (days)
ntp=7
ft=np.ones(2*nt+1)
lday=np.arange(-nt,nt+1)
p=95
# lvn=['tas','mrsos','hfls','pr'] # input1
lvn=['mrsos'] # input1
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

# low pass butterworth filter for smoothing seasonal cycle
nf=10 # order
wn=1/10 # critical frequency [1/day]
lpf=butter(nf,wn,output='sos')

lmd=mods(fo) # create list of ensemble members

# load ocean indices
_,omi=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# for regridding
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
ogr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

def calc_llp(md,p):
    print(md)
    ens=emem(md)
    grd=grid(md)

    for varn in lvn:
        print(varn)
        if md=='IPSL-CM6A-LR' and varn=='huss':
            idir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
        else:
            idir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)

        tdir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,'tas')
        odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        if not os.path.exists(odir):
            os.makedirs(odir)

        if 'gwl' in byr:
            oname='%s/ll%g.p%g.%s_%s.%s.nc' % (odir,nt,p,varn,byr,se)
        else:
            oname='%s/ll%g.p%g.%s_%g-%g.%s.nc' % (odir,nt,p,varn,byr[0],byr[1],se)

        if checkexist:
            if os.path.isfile(oname):
                print('Output file already exists, skipping...')
                continue

        if 'gwl' in byr:
            idirg='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % ('ts','historical+%s'%fo,md,'tas')
            [ygwl,gwl]=pickle.load(open('%s/gwl%s.%s.pickle' % (idirg,'tas','ts'),'rb'))
            print(ygwl)
            idx=np.where(gwl==float(byr[-3:]))
            print(idx)

        # load raw data
        if ishr(md):
            if 'gwl' in byr:
                lyr=listyr(md,byr,ygwl=ygwl[idx])
            else:
                lyr=listyr(md,byr,ygwl=None)
            fn = ['%s/%s_%s_%s_%s_%s_%s_%s.nc' % (idir,varn,freq,md,fo,ens,grd,yr) for yr in lyr]
        else:
            fn = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir,varn,freq,md,fo,ens,grd)
        print('\n Loading data to composite...')
        ds = xr.open_mfdataset(fn)
        if md=='IITM-ESM' and varn=='mrsos':
            vn=1e3*ds[varn]
        else:
            vn=ds[varn]
        print('\n Done.')
        # save grid info
        gr = {}
        gr['lon'] = ds['lon']
        gr['lat'] = ds['lat']
        ds=None

        # load +nt days before and after
        if 'gwl' in byr:
            if ygwl[idx]==1850:
                print('\n %s does not warm to %s K. Skipping...'%(md,byr[-3:]))
                return

            print('\n Selecting data within range of interest...')
            otime=vn['time'].sel(time=vn['time.year']==2080)
            try:
                i0=vn.indexes['time'].get_loc('%g-01-01 00:00:00'%ygwl[idx].data-dyr,method='nearest').start
                i9=vn.indexes['time'].get_loc('%g-01-01 00:00:00'%ygwl[idx].data+dyr,method='nearest').stop
            except:
                i0=vn.indexes['time'].get_loc('%g-01-01 00:00:00'%ygwl[idx].data-dyr,method='nearest')
                i9=vn.indexes['time'].get_loc('%g-01-01 00:00:00'%ygwl[idx].data+dyr,method='nearest')
            vn=vn.isel(time=slice(i0-nt,i9+nt))
            print('\n Done.')

        else:
            print('\n Selecting data within range of interest...')
            if md=='IITM-ESM' and fo=='ssp370':
                yend=2098
            else:
                yend=byr[1]
            try:
                i0=vn.indexes['time'].get_loc('%g-01-01 00:00:00'%byr[0],method='nearest').start
                i9=vn.indexes['time'].get_loc('%g-01-01 00:00:00'%yend,method='nearest').stop
            except:
                i0=vn.indexes['time'].get_loc('%g-01-01 00:00:00'%byr[0],method='nearest')
                i9=vn.indexes['time'].get_loc('%g-01-01 00:00:00'%yend,method='nearest')
            if md=='CESM2':
                vn=vn.isel(time=slice(i0-nt,i9+nt-1))
            else:
                vn=vn.isel(time=slice(i0-nt,i9+nt))
            otime=vn['time'].sel(time=vn['time.year']==byr[0])
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

        print('\n Removing ocean points...')
        # remove ocean points
        time=vn['time']
        time1=time.sel(time=vn['time.year']==list(set(vn['time.year'].data))[1])
        vn=vn.data
        vn=np.reshape(vn,(vn.shape[0],vn.shape[1]*vn.shape[2]))
        vn=np.delete(vn,omi,axis=1)
        ngp=vn.shape[1]
        print('\n Done.')

        print('\n Smoothing the seasonal cycle and deseasonalizing...')
        # smooth seasonal cycle
        vn=xr.DataArray(vn,coords={'time':time,'gpi':np.arange(ngp)},dims=('time','gpi')).compute()
        svn=vn.groupby('time.dayofyear').mean('time')
        svn.data=sosfiltfilt(lpf,svn,axis=0)
        # deseasonalize
        vn.data=vn.groupby('time.dayofyear')-svn
        print('\n Done.')

        print('\n Computing lead-lag composite for all days...')
        llvn=[vn[i:-2*nt+i,:] for i in tqdm(range(len(ft)))]
        # last slice is an edge case
        llvn[-1]=vn[len(ft)-1:]
        print('\n Done.')

        # compute hot days
        # load temp data
        if 'gwl' in byr:
            fn='%s/lm.%s_%s.%s.nc' % (tdir,'tas',byr,se)
        else:
            fn='%s/lm.%s_%g-%g.%s.nc' % (tdir,'tas',byr[0],byr[1],se)
        print('\n Loading tas for conditioning by hot days...')
        ds = xr.open_mfdataset(fn)
        tvn = ds['tas'].compute()
        time2=tvn['time']
        print('\n Done.')

        print('\n Loading percentile data...')
        # load percentile data
        if 'gwl' in byr:
            ds=xr.open_dataset('%s/p.%s%03d_%s.%s.nc' % (tdir,'tas',ntp,byr,se))
        else:
            ds=xr.open_dataset('%s/p.%s%03d_%g-%g.%s.nc' % (tdir,'tas',ntp,byr[0],byr[1],se))
        ds=ds.sel(percentile=p)
        try:
            pvn=ds['tas']
        except:
            pvn=ds['__xarray_dataarray_variable__']
        print('\n Done.')

        print('\n Computing hot days...')
        # replace pvn time with actual dates
        pvn=xr.DataArray(pvn.data,coords={'time':time1,'gpi':np.arange(ngp)},dims=('time','gpi'))

        pvn=tvn.groupby('time.dayofyear')-pvn.groupby('time.dayofyear').mean(dim='time')
        dep=np.ones_like(pvn.data)
        dep[pvn.data<0]=np.nan # i.e. keep days exceeding percentile value
        print('\n Done.')

        dep=xr.DataArray(dep, coords={'time':time2,'gpi':pvn['gpi']}, dims=('time','gpi'))

        print('\n Composite conditioned on hot days...')
        svn=[]
        for lday in tqdm(range(len(llvn))):
            tmpvn=llvn[lday]*dep.data
            tmpvn=xr.DataArray(tmpvn, coords={'time':time2,'gpi':pvn['gpi']}, dims=('time','gpi'))
            svn.append(tmpvn.groupby('time.month').mean('time',skipna=True))
        print('\n Done.')

        print('\n Concatenating into single dataarray...')
        svn=xr.concat(svn,'lday').chunk({'month':12,'gpi':len(pvn['gpi'])})
        print('\n Saving output...')
        svn=svn.rename(varn)
        print(svn)
        svn.to_netcdf(oname,format='NETCDF4')
        print('\n Done.')

calc_llp('IITM-ESM',p)

# if __name__=='__main__':
#     with ProgressBar():
#         tasks=[dask.delayed(calc_llp)(md,p) for md in lmd]
#         # dask.compute(*tasks,scheduler='processes')
#         dask.compute(*tasks,scheduler='single-threaded')
