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
from gridfill import fill

# collect warmings across the ensembles

ntrunc=18 # wavenumber for spherical harmonics filter
slev=925 # in hPa
ivar='advty'
varn='%s%g_t%g'%(ivar,slev,ntrunc)

# fo = 'historical' # forcing (e.g., ssp245)
fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=False
freq='Eday'

lmd=mods(fo) # create list of ensemble members
# lmd=['CESM2','IPSL-CM6A-LR','NorESM2-LM','NorESM2-MM','TaiESM1']

def calc_adv(md):
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
                    fn1=fn1.replace('ta850','va850')
                    ds = xr.open_dataset(fn1)
                    va850=ds['va850']
                except:
                    fn1=fn1.replace('day','Eday')
                    fn1=fn1.replace('ta850','va850')
                    ds = xr.open_dataset(fn1)
                    va850=ds['va850']

                deglat=ds['lat'].data
                lat=np.deg2rad(ds['lat'].data)
                lon=np.deg2rad(ds['lon'].data)

                # compute cpt flux divergence
                adv=va850.copy()
                # compute cpt
                cpt=c.cpd*ta850.data
                clat=np.transpose(np.tile(np.cos(lat),(1,1,1)),[0,2,1])
                cptc=clat*cpt
                # meridional divergence
                dy=1/(c.a*clat[:,1:-1,:])*(cptc[:,2:,:]-cptc[:,:-2,:])/(lat[2:]-lat[:-2]).reshape([1,len(lat)-2,1])
                dy=np.concatenate((1/(c.a*clat[:,0,:])*(cptc[:,[1],:]-cptc[:,[0],:])/(lat[1]-lat[0]),dy),axis=1)
                dy=np.concatenate((dy,1/(c.a*clat[:,-1,:])*(cptc[:,[-1],:]-cptc[:,[-2],:])/(lat[-1]-lat[-2])),axis=1)

                # total horizontal advection
                adv.data=va850.data*dy

                # throw out numerical artifact in pole
                if md in ['CESM2','IPSL-CM6A-LR','NorESM2-LM','NorESM2-MM','TaiESM1']:
                    adv.data[:,deglat>85,:]=np.nan

                # spherical harmonic smoothing
                adv=adv.ffill(dim='lon')
                adv=adv.bfill(dim='lon')
                adv=adv.ffill(dim='lat')
                adv=adv.bfill(dim='lat')
                adv=adv.ffill(dim='time')
                adv=adv.bfill(dim='time')
                xsp=xspharm(adv,gridtype='regular')
                adv=xsp.exp_taper(adv,ntrunc=ntrunc)

                adv=adv.rename(varn)
                adv.to_netcdf(ofn)
            except Exception as e:
                print(e)
                print('WARNING skipping %s'%ofn)

calc_adv('UKESM1-0-LL')
# [calc_adv(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_adv,lmd)
