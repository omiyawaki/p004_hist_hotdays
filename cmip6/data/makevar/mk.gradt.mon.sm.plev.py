import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
from concurrent.futures import ProcessPoolExecutor as Pool
import pickle
import numpy as np
import xarray as xr
import constants as c
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid
from scipy.interpolate import interp1d
from xspharm import xspharm

# collect warmings across the ensembles

ntrunc,r=9,9        # hoskins spherical harmonics filter
slev=925 # in hPa
ivar='gradt_mon'
vn0='ta%g'%slev
varn='%s%g'%(ivar,slev)

fo = 'historical' # forcing (e.g., ssp245)
yr=[1980,2000]

# fo = 'ssp370' # forcing (e.g., ssp245)
# yr='gwl2.0'
# dyr=10

checkexist=False
freq='day'
se='sc'

lmd=mods(fo) # create list of ensemble members

def calc_gradt(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    chk=0
    idir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,vn0,md,ens,grd)

    # load raw data
    fn = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir,vn0,freq,md,fo,ens,grd)
    print('\n Loading data to composite...')
    vn = xr.open_mfdataset(fn)[vn0]
    print('\n Done.')

    # select data within time of interest
    if 'gwl' in yr:
        idirg='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % ('ts','historical+%s'%fo,md,'tas')
        [ygwl,gwl]=pickle.load(open('%s/gwl%s.%s.pickle' % (idirg,'tas','ts'),'rb'))
        print(ygwl)
        idx=np.where(gwl==float(yr[-3:]))
        print(idx)
        if ygwl[idx]==1850:
            print('\n %s does not warm to %s K. Skipping...'%(md,yr[-3:]))
            return

        print('\n Selecting data within range of interest...')
        vn=vn.sel(time=vn['time.year']>=ygwl[idx].data-dyr)
        vn=vn.sel(time=vn['time.year']<ygwl[idx].data+dyr)
        print('\n Done.')
    else:
        if md=='IITM-ESM' and fo=='ssp370':
            yend=2098
        else:
            yend=yr[1]
        print('\n Selecting data within range of interest...')
        vn=vn.sel(time=vn['time.year']>=yr[0])
        vn=vn.sel(time=vn['time.year']<yend)
        print('\n Done.')

    # mon mean
    vn=vn.groupby('time.month').mean('time')

    mon=vn['month']
    xlat=vn['lat']
    xlon=vn['lon']

    lat=np.deg2rad(vn['lat'].data)
    lon=np.deg2rad(vn['lon'].data)

    # smooth horizontally
    vn=vn.ffill(dim='lon')
    vn=vn.bfill(dim='lon')
    vn=vn.ffill(dim='lat')
    vn=vn.bfill(dim='lat')
    vn=vn.ffill(dim='month')
    vn=vn.bfill(dim='month')
    xsp=xspharm(vn,gridtype='regular')
    vn=xsp.exp_taper(vn,ntrunc=ntrunc,r=r)

    # compute grad(mon mean(T))
    lonm=1/2*(lon[1:]+lon[:-1])
    latm=1/2*(lat[1:]+lat[:-1])
    clat=np.transpose(np.tile(np.cos(lat),(1,1,1)),[0,2,1])
    clatm=np.transpose(np.tile(np.cos(latm),(1,1,1)),[0,2,1])
    vn=vn.data
    # zonal derivative
    vnym=1/2*(vn[:,1:,:]+vn[:,:-1,:]) # meridional midpoints
    dxm=1/(c.a*clatm)*(vnym[...,1:]-vnym[...,:-1])/(lon[1:]-lon[:-1])
    # meridional divergence
    vnxm=clat*1/2*(vn[...,1:]+vn[...,:-1]) # zonal midpoints
    dym=1/(c.a*clatm)*(vnxm[:,1:,:]-vnxm[:,:-1,:])/(lat[1:]-lat[:-1]).reshape([1,len(latm),1])

    # evaluate at original grid
    lonmd=np.rad2deg(lonm)
    latmd=np.rad2deg(latm)
    dxm=xr.DataArray(dxm,name='dx',coords={'month':mon,'lat':latmd,'lon':lonmd},dims=('month','lat','lon'))
    dym=xr.DataArray(dym,name='dy',coords={'month':mon,'lat':latmd,'lon':lonmd},dims=('month','lat','lon'))
    dx=dxm.interp(lat=xlat,lon=xlon)
    dy=dym.interp(lat=xlat,lon=xlon)

    # save as single dataset
    gradt=xr.merge([dx,dy])

    if 'gwl' in yr:
        gradt.to_netcdf('%s/mon.%s_%s.%s.nc' % (odir,varn,yr,se),format='NETCDF4')
    else:
        gradt.to_netcdf('%s/mon.%s_%g-%g.%s.nc' % (odir,varn,yr[0],yr[1],se),format='NETCDF4')


# calc_gradt('UKESM1-0-LL')
[calc_gradt(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_gradt,lmd)
