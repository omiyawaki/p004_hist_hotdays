import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
from concurrent.futures import ProcessPoolExecutor as Pool
import warnings
import numpy as np
import xarray as xr
import constants as c
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid
from metpy.calc import saturation_mixing_ratio,specific_humidity_from_mixing_ratio
from metpy.units import units
from scipy.interpolate import interp1d

# collect warmings across the ensembles

slev=950 # in hPa
# livar=['zg']
livar=['ua','va','ta','zg']
lvarn=['%s%g'%(ivar,slev) for ivar in livar]

# fo = 'historical' # forcing (e.g., ssp245)
fo = 'ssp370' # forcing (e.g., ssp245)

freq='day'

lmd=mods(fo) # create list of ensemble members

def regrid(vn,targ):
    fint=interp1d(vn['lat'].data,vn.data,axis=1)
    targ.data=fint(targ['lat'].data)
    return targ

def maskss(vn,psd):
    return xr.where(psd<100*slev,np.nan,vn)

def calc_plev(md,ivar,varn):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('Loading ps...')
    psdir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,'Amon','ps',md,ens,grd)
    # load monthly ps
    ps=xr.open_mfdataset('%s/*.nc'%psdir)['ps'].load()
    print('\nDone...')

    idir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,ivar,md,ens,grd)
    for _,_,files in os.walk(idir):
        for fn in files:
            try:
                ds = xr.open_dataset('%s/%s'%(idir,fn))
                # select data at desired level
                vn=ds[ivar]
                vnlev=vn.sel(plev=100*slev,method='nearest')

                print('Interpolating ps...')
                # interpolate to daily ps
                psd=ps.interp(time=vnlev['time'],kwargs={"fill_value": "extrapolate"})
                print('\nDone...')

                if not psd['lat'].equals(vnlev['lat']):
                    vnlev=vnlev.interp(lat=psd['lat'],kwargs={"fill_value": "extrapolate"})
                if not psd['lon'].equals(vnlev['lon']):
                    vnlev=vnlev.interp(lon=psd['lon'],kwargs={"fill_value": "extrapolate"})

                # mask subsurface data
                vnlev.data=maskss(vnlev,psd).data

                vnlev=vnlev.rename(varn)
                vnlev.to_netcdf('%s/%s'%(odir,fn.replace(ivar,varn,1)))
            except Exception as e:
                print(e)
                print('\n WARNING: Skipping %s \n'%fn)

def loopvn(md):
    [calc_plev(md,ivar,varn) for ivar,varn in zip(livar,lvarn)]

# loopvn('MIROC-ES2L')
[loopvn(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(loopvn,lmd)
