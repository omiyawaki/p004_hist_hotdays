import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
from concurrent.futures import ProcessPoolExecutor as Pool
import numpy as np
import xarray as xr
import constants as c
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid
from scipy.interpolate import interp1d
from xspharm import xspharm

# collect warmings across the ensembles

ntrunc,r=9,9
slev=850 # in hPa
ivar='gradt'

fo = 'historical' # forcing (e.g., ssp245)
# fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=False
freq='Eday'

lmd=mods(fo) # create list of ensemble members

def calc_grad(md):
    ens=emem(md)
    grd=grid(md)
    ovn='%s%g'%(ivar,slev)

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,ovn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    idir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'ta850',md,ens,grd)
    for _,_,files in os.walk(idir):
        for fn in files:
            ofn='%s/%s'%(odir,fn.replace('ta850',ovn))
            if checkexist and os.path.isfile(ofn):
                continue

            fn1='%s/%s'%(idir,fn)
            vn=xr.open_dataarray(fn1)
            time=vn['time']
            xlat=vn['lat']
            xlon=vn['lon']

            lat=np.deg2rad(vn['lat'].data)
            lon=np.deg2rad(vn['lon'].data)

            # smooth horizontally
            vn=vn.ffill(dim='lon')
            vn=vn.bfill(dim='lon')
            vn=vn.ffill(dim='lat')
            vn=vn.bfill(dim='lat')
            vn=vn.ffill(dim='time')
            vn=vn.bfill(dim='time')
            xsp=xspharm(vn,gridtype='regular')
            vn=xsp.exp_taper(vn,ntrunc=ntrunc,r=r)

            # compute grad(doy mean(T))
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
            dxm=xr.DataArray(dxm,name='dx',coords={'time':time,'lat':latmd,'lon':lonmd},dims=('time','lat','lon'))
            dym=xr.DataArray(dym,name='dy',coords={'time':time,'lat':latmd,'lon':lonmd},dims=('time','lat','lon'))
            dx=dxm.interp(lat=xlat,lon=xlon)
            dy=dym.interp(lat=xlat,lon=xlon)

            grad=xr.merge([dx,dy])
            grad.to_netcdf(ofn,format='NETCDF4')

calc_grad('UKESM1-0-LL')
# [calc_grad(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_grad,lmd)
