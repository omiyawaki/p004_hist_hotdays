import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
from concurrent.futures import ProcessPoolExecutor as Pool
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
import constants as c
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid

# collect warmings across the ensembles

varn='tend'
pt=1100e2
p0=0

fo = 'historical' # forcing (e.g., ssp245)
# fo = 'ssp370' # forcing (e.g., ssp245)

freq='day'
checkexist=True

md='CESM2'

ens=emem(md)
grd=grid(md)

odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
if not os.path.exists(odir):
    os.makedirs(odir)

print('\nLoading ps...\n')
idir0='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,'Amon','ps',md,ens,grd)
ps=xr.open_mfdataset('%s/*.nc'%idir0)['ps'].mean('time').data
print('Done.\n')

print('\nLoading beta...\n')
idir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'beta',md,ens,grd)
beta=xr.open_mfdataset('%s/*.nc'%idir)['beta'].data
print('Done.\n')

# load mse
idir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'mse',md,ens,grd)
lfn=list(os.walk(idir))[0][2]

def calc_tend(fn):
    ofn='%s/%s'%(odir,fn.replace('mse',varn,1))
    if checkexist and os.path.isfile(ofn):
        return
    mse=xr.open_dataset('%s/%s'%(idir,fn))['mse']
    time=mse['time']
    plev=mse['plev'].data
    lat=mse['lat']
    lon=mse['lon']
    mse=mse.data

    plev_half=1/2*(plev[1:]+plev[:-1])
    plev_half=np.sort(np.append(plev_half,[pt,p0]))
    plev_half=plev_half[::-1] if plev[1]-plev[0]<0 else plev_half
    dplev=plev_half[1:]-plev_half[:-1]

    print('\nMasking subsurface data...\n')
    ps_tile=np.tile(ps,[mse.shape[0],mse.shape[1],1,1])
    pa=np.transpose(np.tile(plev,[mse.shape[0],mse.shape[2],mse.shape[3],1]),[0,3,1,2])
    mse[pa>ps_tile]=np.nan
    print('Done.\n')

    print('\nTaking time tendency...\n')
    dmsedt=np.empty_like(mse.data)
    dmsedt[1:-1,...] = (mse[2:,...]-mse[:-2,...])/(2*86400)
    dmsedt[0,...] = (mse[1,...]-mse[0,...])/86400
    dmsedt[-1,...] = (mse[-1,...]-mse[-2,...])/86400
    print('Done.\n')

    print('\nComputing vertical integral...\n')
    if plev[1]-plev[0]>0: # if pressure increases with index
        dvmsedt = 1/c.g * np.nansum( beta[None,...] * dmsedt * dplev[None,...,None,None], axis=1)
    else:
        dvmsedt = -1/c.g * np.nansum( beta[None,...] * dmsedt * dplev[None,...,None,None], axis=1)
    print('Done.\n')

    print('\nSaving output...\n')
    tend=xr.DataArray(dvmsedt,{'time':time,'lat':lat,'lon':lon},dims=('time','lat','lon'))
    tend=tend.rename(varn)
    tend.to_netcdf(ofn)
    print('Done.\n')

# calc_tend(lfn[0])
[calc_tend(fn) for fn in tqdm(lfn)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lfn)) as p:
#         p.map(calc_tend,lfn)
