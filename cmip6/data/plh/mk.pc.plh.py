import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
import constants as c
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid
from etregimes import bestfit

# collect warmings across the ensembles

varn='plh'
se='sc'
nt=7
p=95
doy=False

fo = 'historical' # forcing (e.g., ssp245)
byr=[1980,2000]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr=[2080,2100]

freq='day'

lmd=mods(fo) # create list of ensemble members

def get_ifn(varn,md):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if 'gwl' in byr:
        fn='%s/lm.%s_%s.%s.nc' % (idir,varn,byr,se)
    else:
        fn='%s/lm.%s_%g-%g.%s.nc' % (idir,varn,byr[0],byr[1],se)
    return fn

def get_ofn(varn,md):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)

    if 'gwl' in byr:
        fn='%s/pc%03d.%s_%s.%s' % (idir,p,varn,byr,se)
    else:
        fn='%s/pc%03d.%s_%g-%g.%s' % (idir,p,varn,byr[0],byr[1],se)

    if doy:
        fn='%s.doy.nc'%fn
    else:
        fn='%s.nc'%fn
    return fn

def get_fnhd(varn,md):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if 'gwl' in byr:
        fn='%s/p.%s%03d_%s.%s.nc' % (idir,varn,nt,byr,se)
    else:
        fn='%s/p.%s%03d_%g-%g.%s.nc' % (idir,varn,nt,byr[0],byr[1],se)
    return fn

def get_bc(md):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,'bc')
    iname='%s/pc%03d.%s.%g-%g.%s.pickle' % (idir,p,'bc',byr[0],byr[1],se)
    return pickle.load(open(iname,'rb'))

def eval_bc(sm,bc):
    return np.interp(sm,bc[0],bc[1])

def calc_plh(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Loading data...')
    ds=xr.open_dataset(get_ifn('mrsos',md))
    gpi=ds['gpi']
    time=ds['time']
    sm=ds['mrsos']
    tas=xr.open_dataset(get_ifn('tas',md))['tas']
    print('\n Done.')

    print('\n Loading hot day data...')
    ptas=xr.open_dataarray(get_fnhd('tas',md))
    ptas=ptas.sel(percentile=p)
    print('\n Done.')

    print('\n Computing lh using budyko curve...')
    bc=get_bc(md)
    plh=[]

    def plhmon(mon,sm,bc):
        ssm=sm.sel(time=sm['time.month']==mon)
        stas=tas.sel(time=tas['time.month']==mon)
        iday=list(set(tas['time.dayofyear'].data))
        sptas=ptas.sel(doy=iday).mean(dim='doy')
        ihd=stas.data-np.transpose(sptas.data[:,None],[1,0])
        ssm.data[ihd<0]=np.nan
        stas.data[ihd<0]=np.nan
        slh=ssm.copy()
        for igpi in tqdm(range(len(gpi))):
            try:
                slh.data[:,igpi]=eval_bc(ssm[:,igpi],bc[mon-1][igpi])
            except:
                slh.data[:,igpi]=np.nan
        return slh

    with ProgressBar():
        tasks=[dask.delayed(plhmon)(mon,sm,bc) for mon in np.arange(1,13,1)]
        plh=dask.compute(*tasks,scheduler='threads')

    plh=xr.concat(plh,'time').sortby('time')
    plh=plh.rename('plh')

    # climatology
    if doy:
        print('\n dayofyear mean...')
        plh = plh.groupby('time.dayofyear').mean('time',skipna=True)
        print('\n Done.')
    else:
        print('\n Monthly mean...')
        plh = plh.groupby('time.month').mean('time',skipna=True)
        print('\n Done.')

    # save plh
    oname=get_ofn(varn,md)
    plh.to_netcdf(oname,format='NETCDF4')

if __name__=='__main__':
    calc_plh('CESM2')

# if __name__ == '__main__':
#     with ProgressBar():
#         tasks=[dask.delayed(calc_plh)(md) for md in lmd]
#         dask.compute(*tasks,scheduler='processes')
#         # dask.compute(*tasks,scheduler='single-threaded')
