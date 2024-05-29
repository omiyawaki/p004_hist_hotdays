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

nt=30 # rolling mean window [days]
slev=850 # in hPa
ivar='gradt_rmean'
varn='%s%g'%(ivar,slev)

# fo = 'historical' # forcing (e.g., ssp245)
fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=False
freq='day'

lmd=mods(fo) # create list of ensemble members

def calc_gradt(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    chk=0
    idir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'ta850',md,ens,grd)
    for _,_,files in os.walk(idir):
        for fn in files:
            ofn='%s/%s'%(odir,fn.replace('ta850',varn))
            if checkexist and os.path.isfile(ofn):
                continue
            try:
                fn1='%s/%s'%(idir,fn)
                ds = xr.open_dataset(fn1)
                ta850=ds['ta850']
                time=ta850['time']
                xlat=ta850['lat']
                xlon=ta850['lon']

                lat=np.deg2rad(ds['lat'].data)
                lon=np.deg2rad(ds['lon'].data)

                # compute grad(rolling mean(T))
                ta850=ta850.rolling(time=nt,min_periods=1).mean()
                clat=np.transpose(np.tile(np.cos(lat),(1,1,1)),[0,2,1])
                ta850=ta850.data
                ta850c=clat*ta850
                # zonal derivative
                dx=1/(c.a*clat)*(ta850[...,2:]-ta850[...,:-2])/(lon[2:]-lon[:-2])
                dx=np.concatenate((1/(c.a*clat)*(ta850[...,[1]]-ta850[...,[0]])/(lon[1]-lon[0]),dx),axis=-1)
                dx=np.concatenate((dx,1/(c.a*clat)*(ta850[...,[-1]]-ta850[...,[-2]])/(lon[-1]-lon[-2])),axis=-1)
                # meridional divergence
                dy=1/(c.a*clat[:,1:-1,:])*(ta850c[:,2:,:]-ta850c[:,:-2,:])/(lat[2:]-lat[:-2]).reshape([1,len(lat)-2,1])
                dy=np.concatenate((1/(c.a*clat[:,0,:])*(ta850c[:,[1],:]-ta850c[:,[0],:])/(lat[1]-lat[0]),dy),axis=1)
                dy=np.concatenate((dy,1/(c.a*clat[:,-1,:])*(ta850c[:,[-1],:]-ta850c[:,[-2],:])/(lat[-1]-lat[-2])),axis=1)

                gradt=xr.Dataset(
                        data_vars={'dx':(['time','lat','lon'],dx),'dy':(['time','lat','lon'],dy)},
                        coords={'time':time,'lat':xlat,'lon':xlon}
                        )
                gradt.to_netcdf(ofn)
            except Exception as e:
                print(e)
                print('WARNING skipping %s'%ofn)

# calc_gradt('UKESM1-0-LL')

if __name__=='__main__':
    with Pool(max_workers=len(lmd)) as p:
        p.map(calc_gradt,lmd)
