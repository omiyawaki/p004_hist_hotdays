import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
from concurrent.futures import ProcessPoolExecutor as Pool
import numpy as np
import xarray as xr
import constants as c
from tqdm import tqdm
from util import mods,simu,emem
from glade_utils import grid

checkexist=False

varn='mse'
ivars=['ta','hus','zg']

# fo='historical' # forcing (e.g., ssp245)
fo='ssp370' # forcing (e.g., ssp245)

freq='day'

# lmd=mods(fo) # create list of ensemble members
md='CESM2'

ens=emem(md)
grd=grid(md)

idir='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,ivars[0],md,ens,grd)
odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
if not os.path.exists(odir):
    os.makedirs(odir)

for _,_,files in os.walk(idir):
    lfn=[fn for fn in files]

def calc_mse(fn):
    ofn='%s/%s'%(odir,fn.replace(ivars[0],varn,1))
    if checkexist and os.path.isfile(ofn):
        return
    fn1='%s/%s'%(idir,fn)
    ds=xr.open_dataset(fn1)
    ta=ds[ivars[0]]
    # save grid info
    gr = {}
    gr['lon'] = ds['lon']
    gr['lat'] = ds['lat']
    fn1=fn1.replace(ivars[0],ivars[1])
    hus=xr.open_dataset(fn1)[ivars[1]]
    fn1=fn1.replace(ivars[1],ivars[2])
    zg=xr.open_dataset(fn1)[ivars[2]]

    # compute mse
    mse=ta.copy()
    mse.data=c.cpd*ta.data+c.Lv*hus.data+c.g*zg.data
    mse=mse.rename(varn)
    mse.to_netcdf(ofn)

if __name__=='__main__':
    with Pool(max_workers=len(lfn)) as p:
        p.map(calc_mse,lfn)
    

