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
ivar='fat'
varn='%s%g'%(ivar,slev)

# fo = 'historical' # forcing (e.g., ssp245)
fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=False
freq='Eday'

lmd=mods(fo) # create list of ensemble members

def calc_fa(md):
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
                try:
                    fn1=fn1.replace('ta850','ua850')
                    ds = xr.open_dataset(fn1)
                    ua850=ds['ua850']
                    fn1=fn1.replace('ua850','va850')
                    ds = xr.open_dataset(fn1)
                    va850=ds['va850']
                except:
                    fn1=fn1.replace('day','Eday')
                    fn1=fn1.replace('ta850','ua850')
                    ds = xr.open_dataset(fn1)
                    ua850=ds['ua850']
                    fn1=fn1.replace('ua850','va850')
                    ds = xr.open_dataset(fn1)
                    va850=ds['va850']

                lat=np.deg2rad(ds['lat'].data)
                lon=np.deg2rad(ds['lon'].data)

                # compute cpt flux divergence
                fa=ua850.copy()
                # compute cpt
                cpt=c.cpd*ta850.data
                # zonal and meridional transport
                clat=np.transpose(np.tile(np.cos(lat),(1,1,1)),[0,2,1])
                cptu=cpt*ua850.data
                cptv=cpt*va850.data*clat
                # zonal divergence
                fax=1/(c.a*clat)*(cptu[...,2:]-cptu[...,:-2])/(lon[2:]-lon[:-2])
                fax=np.concatenate((1/(c.a*clat)*(cptu[...,[1]]-cptu[...,[0]])/(lon[1]-lon[0]),fax),axis=-1)
                fax=np.concatenate((fax,1/(c.a*clat)*(cptu[...,[-1]]-cptu[...,[-2]])/(lon[-1]-lon[-2])),axis=-1)
                # meridional divergence
                fay=1/(c.a*clat[:,1:-1,:])*(cptv[:,2:,:]-cptv[:,:-2,:])/(lat[2:]-lat[:-2]).reshape([1,len(lat)-2,1])
                fay=np.concatenate((1/(c.a*clat[:,0,:])*(cptv[:,[1],:]-cptv[:,[0],:])/(lat[1]-lat[0]),fay),axis=1)
                fay=np.concatenate((fay,1/(c.a*clat[:,-1,:])*(cptv[:,[-1],:]-cptv[:,[-2],:])/(lat[-1]-lat[-2])),axis=1)

                # total horizontal divergence
                fa.data=fax+fay

                fa=fa.rename(varn)
                fa.to_netcdf(ofn)
            except Exception as e:
                print(e)
                print('WARNING skipping %s'%ofn)

calc_fa('UKESM1-0-LL')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_fa,lmd)
