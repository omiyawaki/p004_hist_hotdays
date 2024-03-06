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

varn='beta'
p0=1100e2   # bottom integral bound [hPa]
pt=0        # top integral bound [hPa]

# fo = 'historical' # forcing (e.g., ssp245)
fo = 'ssp370' # forcing (e.g., ssp245)

freq='day'
checkexist=False

lmd=mods(fo) # create list of ensemble members

def calc_beta(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    idir0='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,'Amon','ps',md,ens,grd)
    ds = xr.open_mfdataset('%s/*.nc'%idir0)
    ps = ds['ps'].load()
    yr0=str(ps['time.year'][0].data)
    yr1=str(ps['time.year'][-1].data)
    ps=ps.mean('time')

    # load arbitrary 3d data
    idir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'ta',md,ens,grd)
    files=list(os.walk(idir))[0][2][1]
    ofn='%s/%s'%(odir,files.replace('ta',varn,1).replace(files[-20:-16],yr0).replace(files[-11:-7],yr1))
    ds = xr.open_dataset('%s/%s'%(idir,files)).mean('time')
    ta=ds['ta']

    plev=ds['plev'].data
    plev_half=1/2*(plev[1:]+plev[:-1])
    plev_half=np.sort(np.append(plev_half,[pt,p0]))
    plev_full=np.sort(np.concatenate((plev,plev_half)))
    if plev[1]-plev[0]<0:
        plev_half=plev_half[::-1]
        plev_full=plev_full[::-1]
    dplev=plev_half[1:]-plev_half[:-1]

    # compute beta (following Trenberth 1991)
    beta=ta.copy()
    for ilon in tqdm(range(ps.shape[1])):
        for ilat in range(ps.shape[0]):
            ps_local=ps[ilat,ilon]

            for ilev in range(len(plev)):
                ilevf=2*ilev+1 # corresponding index in the plev_full_array
                if plev_full[ilevf-1]<ps_local:
                    beta.data[ilev,ilat,ilon]=1
                if plev_full[ilevf+1]>ps_local:
                    beta.data[ilev,ilat,ilon]=0
                if (plev_full[ilevf-1]>ps_local) and (plev_full[ilevf+1]<ps_local):
                    beta.data[ilev,ilat,ilon]=(ps_local-plev_full[ilevf+1])/(plev_full[ilevf-1]-plev_full[ilevf+1])

    beta=beta.rename(varn)
    beta.to_netcdf(ofn)

calc_beta('CESM2')
