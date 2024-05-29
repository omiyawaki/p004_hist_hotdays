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

varn='HTEND'

# fo = 'historical' # forcing (e.g., ssp245)
fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=False
freq='day'

cname=casename(fo)

def calc_hten(md):
    rlm=realm(varn.lower())
    hst=history(varn.lower())

    idir='/project/amp02/miyawaki/data/share/cesm2/%s/%s/proc/tseries/%s_1' % (cname,rlm,freq)
    odir=idir
    for _,_,files in os.walk(idir):
        files=[fn for fn in files if '.MSE.' in fn]
        for fn in tqdm(files):
            print(fn)
            ofn='%s/%s'%(odir,fn.replace('.MSE.','.%s.'%varn))
            if checkexist and os.path.isfile(ofn):
                continue
            fn1='%s/%s'%(idir,fn)

            mse=xr.open_dataset(fn1)['MSE']
            hten=mse.copy()

            # compute mse tendency
            mse=mse.data
            dt=(mse[2:,...]-mse[:-2,...])/(2*86400)
            dt=np.concatenate(((mse[[1],...]-mse[[0],...])/86400,dt),axis=0)
            dt=np.concatenate((dt,(mse[[-1],...]-mse[[-2],...])/86400),axis=0)

            # total horizontal htenection
            hten.data=dt

            # save
            hten=hten.rename(varn)
            hten.to_netcdf(ofn)

calc_hten('CESM2')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_hten,lmd)
