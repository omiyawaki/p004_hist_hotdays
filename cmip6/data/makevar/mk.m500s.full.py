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
from cmip6util import mods,simu,emem
from glade_utils import grid
# from metpy.calc import saturation_mixing_ratio,specific_humidity_from_mixing_ratio
# from metpy.units import units

# collect warmings across the ensembles

varn='m500s'

fo = 'historical' # forcing (e.g., ssp245)
byr=[1980,2000]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr=[2080,2100]

freq='day'

lmd=mods(fo) # create list of ensemble members

def saturation_mixing_ratio(p,t):
    # p in pa, T in K
    t=t-273.15
    es=1e2*6.112*np.exp(17.67*t/(t+243.5))
    rs=c.ep*es/(p-es)
    return rs/(1+rs)

def calc_m500s(md):
    ens=emem(md)
    grd=grid(fo,cl,md)

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    idir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'ta500',md,ens,grd)
    for _,_,files in os.walk(idir):
        for fn in files:
            fn1='%s/%s'%(idir,fn)
            fn2=fn1.replace('ta500','zg500')
            if md=='MIROC-ES2L':
                fn2=fn2.replace('day','Eday')
            ds = xr.open_dataset(fn1)
            ta500=ds['ta500']
            ds = xr.open_dataset(fn2)
            zg500=ds['zg500']
            # compute saturation sp humidity
            qs500=saturation_mixing_ratio(5e4,ta500)
            # compute mse
            m500s=c.cpd*ta500+c.g*zg500+c.Lv*qs500

            m500s=m500s.rename(varn)
            m500s.to_netcdf('%s/%s'%(odir,fn.replace('ta500',varn,1)))

calc_m500s('MIROC-ES2L')

# if __name__ == '__main__':
#     with ProgressBar():
#         tasks=[dask.delayed(calc_m500s)(md) for md in lmd]
#         dask.compute(*tasks,scheduler='processes')
