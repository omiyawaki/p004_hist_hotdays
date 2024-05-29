import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
from concurrent.futures import ProcessPoolExecutor as Pool
import numpy as np
import xarray as xr
import constants as c
from tqdm import tqdm
from util import casename
from cesmutils import realm,history

# collect warmings across the ensembles

varn='MSE'

# fo = 'historical' # forcing (e.g., ssp245)
fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=False
freq='day'

cname=casename(fo)

def calc_mse(md):
    rlm=realm(varn.lower())
    hst=history(varn.lower())

    idir='/project/amp02/miyawaki/data/share/cesm2/%s/%s/proc/tseries/%s_1' % (cname,rlm,freq)
    odir=idir
    for _,_,files in os.walk(idir):
        files=[fn for fn in files if '.T.' in fn]
        for fn in tqdm(files):
            ofn='%s/%s'%(odir,fn.replace('.T.','.%s.'%varn))
            if checkexist and os.path.isfile(ofn):
                continue
            fn1='%s/%s'%(idir,fn)
            ta=xr.open_dataset(fn1)['T']
            fn1=fn1.replace('.T.','.Q.')
            hus=xr.open_dataset(fn1)['Q']
            fn1=fn1.replace('.Q.','.Z3.')
            zg=xr.open_dataset(fn1)['Z3']

            # compute mse
            mse=c.cpd*ta+c.Lv*hus+c.g*zg

            # save
            mse=mse.rename(varn)
            mse.to_netcdf(ofn)

calc_mse('CESM2')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_mse,lmd)
