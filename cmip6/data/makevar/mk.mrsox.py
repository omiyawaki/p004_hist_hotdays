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

# collect warmings across the ensembles

# ld=[10,100,200,300,400,500,600,700,800]
ld=np.arange(150,850+100,100)
# ld=np.arange(20,80+20,20)
# depth=200 # depth to integrate in cm

fo = 'historical' # forcing (e.g., ssp245)
# fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=False
freq='Eday'

md='CESM2'
# lmd=mods(fo) # create list of ensemble members

def calc_mrso(depth):
    ens=emem(md)
    grd=grid(md)
    varn='mrso%g'%depth

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    chk=0
    idir0='/project/mojave/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'mrsol',md,ens,grd)
    for _,_,files in os.walk(idir0):
        for fn in files:
            print(fn)
            ofn='%s/%s'%(odir,fn.replace('mrsol',varn))
            if checkexist and os.path.isfile(ofn):
                continue
            fn1='%s/%s'%(idir0,fn)
            ds = xr.open_dataset(fn1)
            mrsol=ds['mrsol']
            mrsol.data=np.cumsum(mrsol.data,axis=1)
            mrsol=mrsol.interp(depth=depth/100)
            mrsol=mrsol.rename(varn)
            mrsol.to_netcdf(ofn)


# calc_mrso(200)
[calc_mrso(depth) for depth in tqdm(ld)]

# if __name__=='__main__':
#     with Pool(max_workers=len(ld)) as p:
#         p.map(calc_mrso,ld)
