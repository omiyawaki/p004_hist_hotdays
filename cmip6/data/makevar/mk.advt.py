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

varn='advt'

fo = 'historical' # forcing (e.g., ssp245)
# fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=False
freq='day'

lmd=mods(fo) # create list of ensemble members

def calc_adv(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    idir='/project/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'ta',md,ens,grd)
    for _,_,files in os.walk(idir):
        for fn in files:
            ofn='%s/%s'%(odir,fn.replace('ta',varn))
            if checkexist and os.path.isfile(ofn):
                continue
            # try:
            fn1='%s/%s'%(idir,fn)
            ds = xr.open_dataset(fn1)
            ta=ds['ta']
            try:
                fn1=fn1.replace('ta','ua')
                ds = xr.open_dataset(fn1)
                ua=ds['ua']
                fn1=fn1.replace('ua','va')
                ds = xr.open_dataset(fn1)
                va=ds['va']
            except:
                fn1=fn1.replace('day','Eday')
                fn1=fn1.replace('ta','ua')
                ds = xr.open_dataset(fn1)
                ua=ds['ua']
                fn1=fn1.replace('ua','va')
                ds = xr.open_dataset(fn1)
                va=ds['va']

            lat=np.deg2rad(ds['lat'].data)
            lon=np.deg2rad(ds['lon'].data)

            # compute cpt flux divergence
            adv=ua.copy()
            # compute cpt
            cpt=c.cpd*ta.data
            clat=np.transpose(np.tile(np.cos(lat),(1,1,1,1)),[0,1,3,2])
            cptc=clat*cpt
            # zonal derivative
            # dx=1/(c.a*clat)*(cpt[...,2:]-cpt[...,:-2])/(lon[2:]-lon[:-2])
            # dx=np.concatenate((1/(c.a*clat)*(cpt[...,[1]]-cpt[...,[0]])/(lon[1]-lon[0]),dx),axis=-1)
            # dx=np.concatenate((dx,1/(c.a*clat)*(cpt[...,[-1]]-cpt[...,[-2]])/(lon[-1]-lon[-2])),axis=-1)
            dlon=lon[1]-lon[0]
            dx=1/(c.a*clat)*(cpt[...,2:]-cpt[...,:-2])/(2*dlon)
            dx=np.concatenate((1/(c.a*clat)*(cpt[...,[1]]-cpt[...,[-1]])/(2*dlon),dx),axis=-1)
            dx=np.concatenate((dx,1/(c.a*clat)*(cpt[...,[0]]-cpt[...,[-2]])/(2*dlon)),axis=-1)
            # meridional divergence
            dy=1/(c.a*clat[:,:,1:-1,:])*(cptc[:,:,2:,:]-cptc[:,:,:-2,:])/(lat[2:]-lat[:-2]).reshape([1,1,len(lat)-2,1])
            dy=np.concatenate((1/(c.a*clat[:,:,[0],:])*(cptc[:,:,[1],:]-cptc[:,:,[0],:])/(lat[1]-lat[0]),dy),axis=-2)
            dy=np.concatenate((dy,1/(c.a*clat[:,:,[-1],:])*(cptc[:,:,[-1],:]-cptc[:,:,[-2],:])/(lat[-1]-lat[-2])),axis=-2)

            # total horizontal advection
            adv.data=ua.data*dx+va.data*dy

            adv=adv.rename(varn)
            adv.to_netcdf(ofn)
            # except Exception as e:
            #     print(e)
            #     print('WARNING skipping %s'%ofn)

calc_adv('CESM2')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_adv,lmd)
