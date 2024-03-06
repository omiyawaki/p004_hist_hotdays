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
# from metpy.calc import saturation_mixing_ratio,specific_humidity_from_mixing_ratio
# from metpy.units import units

# collect warmings across the ensembles

varn='rastf'

fo = 'historical' # forcing (e.g., ssp245)

# fo = 'ssp370' # forcing (e.g., ssp245)

freq='day'

lmd=mods(fo) # create list of ensemble members

def calc_rastf(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    idir0='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,'Amon','rsdt',md,ens,grd)
    ds = xr.open_mfdataset('%s/*.nc'%idir0)
    rsdt = ds['rsdt'].load()
    rsdt=rsdt.resample(time='1D').interpolate('linear')
    # rsdt=rsdt.groupby('time.dayofyear').mean('time')

    idir0='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,'Amon','rsut',md,ens,grd)
    ds = xr.open_mfdataset('%s/*.nc'%idir0)
    rsut = ds['rsut'].load()
    rsut=rsut.resample(time='1D').interpolate('linear')
    # rsut=rsut.groupby('time.dayofyear').mean('time')

    chk=0
    idir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'rsds',md,ens,grd)
    for _,_,files in os.walk(idir):
        for fn in files:
            ofn='%s/%s'%(odir,fn.replace('rsds',varn))
            if os.path.isfile(ofn):
                continue
            fn1='%s/%s'%(idir,fn)
            ds = xr.open_dataset(fn1)
            rsds=ds['rsds']
            fn1=fn1.replace('rsds','rsus')
            ds = xr.open_dataset(fn1)
            rsus=ds['rsus']
            fn1=fn1.replace('rsus','rlds')
            ds = xr.open_dataset(fn1)
            rlds=ds['rlds']
            fn1=fn1.replace('rlds','rlus')
            ds = xr.open_dataset(fn1)
            rlus=ds['rlus']
            fn1=fn1.replace('rlus','rlut')
            ds = xr.open_dataset(fn1)
            rlut=ds['rlut']
            fn1=fn1.replace('rlut','hfss')
            ds = xr.open_dataset(fn1)
            hfss=ds['hfss']
            fn1=fn1.replace('hfss','hfls')
            ds = xr.open_dataset(fn1)
            hfls=ds['hfls']
            # compute ra+stf
            rastf=rlut.copy()
            rastf.data=rsdt.data-rsut.data-rlut.data-rsds.data+rsus.data-rlds.data+rlus.data+hfss.data+hfls.data
            rastf=rastf.rename(varn)
            rastf.to_netcdf(ofn)

calc_rastf('CanESM5')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_rastf,lmd)
