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

# collect warmings across the ensembles

nt=30 # rolling mean window [days]
slev=850 # in hPa
ivar='grad5t_doy'
vn0='ta850'
varn='%s%g'%(ivar,slev)

# fo = 'historical' # forcing (e.g., ssp245)
# yr=[1980,2000]

fo = 'ssp370' # forcing (e.g., ssp245)
yr='gwl2.0'
dyr=10

checkexist=False
freq='Eday'
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
    vn = xr.open_mfdataset(fn)['ta850']
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

    # doy mean
    vn=vn.groupby('time.dayofyear').mean('time')

    doy=vn['dayofyear']
    xlat=vn['lat']
    xlon=vn['lon']

    lat=np.deg2rad(vn['lat'].data)
    lon=np.deg2rad(vn['lon'].data)

    # compute grad(doy mean(T))
    clat=np.transpose(np.tile(np.cos(lat),(1,1,1)),[0,2,1])
    vn=vn.data
    vnc=clat*vn
    # zonal derivative
    dlon=lon[1]-lon[0]
    dx=1/(c.a*clat)*(vn[...,4:]-8*vn[...,3:-1]+8*vn[...,1:-3]-vn[...,:-4])/(12*dlon) # central
    dx=np.concatenate((1/(c.a*clat)*(vn[...,[-1]]-8*vn[...,[0]]+8*vn[...,[2]]-vn[...,[3]])/(12*dlon),dx),axis=-1) # second
    dx=np.concatenate((1/(c.a*clat)*(vn[...,[-2]]-8*vn[...,[-1]]+8*vn[...,[1]]-vn[...,[2]])/(12*dlon),dx),axis=-1) # first
    dx=np.concatenate((dx,1/(c.a*clat)*(vn[...,[-4]]-8*vn[...,[-3]]+8*vn[...,[-1]]-vn[...,[0]])/(12*dlon)),axis=-1) # second last
    dx=np.concatenate((dx,1/(c.a*clat)*(vn[...,[-3]]-8*vn[...,[-2]]+8*vn[...,[0]]-vn[...,[1]])/(12*dlon)),axis=-1) # last
    # meridional divergence
    dlat=lat[1]-lat[0]
    dy=1/(c.a*clat[:,2:-2,:])*(vnc[:,4:,:]-8*vnc[:,3:-1,:]+8*vnc[:,1:-3,:]-vnc[:,:-4,:])/(12*dlat) # central
    dy=np.concatenate((1/(c.a*clat[:,1,:])*(-3*vnc[:,[0],:]-10*vnc[:,[1],:]+18*vnc[:,[2],:]-6*vnc[:,[3],:]+vnc[:,[4],:])/(12*dlat),dy),axis=-2) # second
    dy=np.concatenate((1/(c.a*clat[:,0,:])*(-25*vnc[:,[0],:]+48*vnc[:,[1],:]-36*vnc[:,[2],:]+16*vnc[:,[3],:]-3*vnc[:,[4],:])/(12*dlat),dy),axis=-2) # first
    dy=np.concatenate((dy,1/(c.a*clat[:,-2,:])*(-vnc[:,[-5],:]+6*vnc[:,[-4],:]-18*vnc[:,[-3],:]+10*vnc[:,[-2],:]+3*vnc[:,[-1],:])/(12*dlat)),axis=-2) # second last
    dy=np.concatenate((dy,1/(c.a*clat[:,-1,:])*(3*vnc[:,[-5],:]-16*vnc[:,[-4],:]+36*vnc[:,[-3],:]-48*vnc[:,[-2],:]+25*vnc[:,[-1],:])/(12*dlat)),axis=-2) # last

    gradt=xr.Dataset(
            data_vars={'dx':(['dayofyear','lat','lon'],dx),'dy':(['dayofyear','lat','lon'],dy)},
            coords={'dayofyear':doy,'lat':xlat,'lon':xlon}
            )
    if 'gwl' in yr:
        gradt.to_netcdf('%s/doy.%s_%s.%s.nc' % (odir,varn,yr,se),format='NETCDF4')
    else:
        gradt.to_netcdf('%s/doy.%s_%g-%g.%s.nc' % (odir,varn,yr[0],yr[1],se),format='NETCDF4')


calc_gradt('UKESM1-0-LL')
# [calc_gradt(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_gradt,lmd)
