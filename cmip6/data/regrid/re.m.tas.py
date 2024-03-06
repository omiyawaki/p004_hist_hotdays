import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor as Pool
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid
from regions import masklev0,masklev1,settype,retname,regionsets

# collect warmings across the ensembles

varn='tas'
ty='2d'
se='ts'
relb='sa'

fo1='historical' # forcing (e.g., ssp245)

fo2='ssp370' # forcing (e.g., ssp245)

fo='%s+%s'%(fo1,fo2)

freq='day'

lmd=mods(fo1) # create list of ensemble members

def calc_re(md):
    ens=emem(md)
    grd=grid(md)
    mtype=settype(relb)
    retn=retname(relb)
    re=regionsets(relb)

    idir1='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo1,freq,varn,md,ens,grd)
    idir2='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo2,freq,varn,md,ens,grd)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    # historical temp
    fn1 = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir1,varn,freq,md,fo1,ens,grd)
    tas1 = xr.open_mfdataset(fn1)[varn]

    # mask gridpoints outside region of interest
    if relb=='us':
        mask=masklev0(re,tas1,mtype).data
    else:
        mask=masklev1(None,tas1,re,mtype).data
    tas1.data=mask*tas1.data

    # create area weights
    cosw=np.cos(np.deg2rad(tas1['lat']))
    # take mean
    tas1=tas1.weighted(cosw).mean(('lon','lat'))

    # future temp
    fn2 = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir2,varn,freq,md,fo2,ens,grd)
    tas2 = xr.open_mfdataset(fn2)[varn]
    tas2.data=mask*tas2.data
    # create area weights
    cosw=np.cos(np.deg2rad(tas2['lat']))
    # take global mean
    tas2=tas2.weighted(cosw).mean(('lon','lat'))
    print(tas2.shape)

    # merge timeseries
    tas=xr.concat([tas1,tas2],dim='time')
    print(tas.shape)

    tas=tas.rename(varn)
    tas.to_netcdf('%s/re.%s.%s.%s.nc' % (odir,varn,se,relb))

calc_re('CESM2')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_re,lmd)

