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

# collect warmings across the ensembles

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
    ovn1='%sx%g'%(ivar,slev)
    ovn2='%sy%g'%(ivar,slev)

    odir1='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,ovn1,md,ens,grd)
    odir2='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,ovn2,md,ens,grd)
    if not os.path.exists(odir1):
        os.makedirs(odir1)
    if not os.path.exists(odir2):
        os.makedirs(odir2)

    idir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'ta850',md,ens,grd)
    for _,_,files in os.walk(idir):
        for fn in files:
            ofn1='%s/%s'%(odir1,fn.replace('ta850',ovn1))
            ofn2='%s/%s'%(odir2,fn.replace('ta850',ovn2))
            if checkexist and (os.path.isfile(ofn1) and os.path.isfile(ofn2)):
                continue

            fn1='%s/%s'%(idir,fn)
            ta850=xr.open_dataarray(fn1)
            time=ta850['time']

            lat=np.deg2rad(ta850['lat'].data)
            lon=np.deg2rad(ta850['lon'].data)

            # compute t flux divergence
            gradx=ta850.copy()
            grady=ta850.copy()
            ta=ta850.data
            clat=np.transpose(np.tile(np.cos(lat),(1,1,1)),[0,2,1])
            cta=clat*ta
            # zonal derivative
            dlon=lon[1]-lon[0]
            dx=1/(c.a*clat)*(ta[...,2:]-ta[...,:-2])/(2*dlon)
            dx=np.concatenate((1/(c.a*clat)*(ta[...,[1]]-ta[...,[-1]])/(2*dlon),dx),axis=-1)
            dx=np.concatenate((dx,1/(c.a*clat)*(ta[...,[0]]-ta[...,[-2]])/(2*dlon)),axis=-1)
            # meridional divergence
            dy=1/(c.a*clat[:,1:-1,:])*(cta[:,2:,:]-cta[:,:-2,:])/(lat[2:]-lat[:-2]).reshape([1,len(lat)-2,1])
            dy=np.concatenate((1/(c.a*clat[:,0,:])*(cta[:,[1],:]-cta[:,[0],:])/(lat[1]-lat[0]),dy),axis=1)
            dy=np.concatenate((dy,1/(c.a*clat[:,-1,:])*(cta[:,[-1],:]-cta[:,[-2],:])/(lat[-1]-lat[-2])),axis=1)

            gradx.data=dx
            grady.data=dy
            gradx.to_netcdf(ofn1,format='NETCDF4')
            grady.to_netcdf(ofn2,format='NETCDF4')

calc_grad('UKESM1-0-LL')
# [calc_grad(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_grad,lmd)
