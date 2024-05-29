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

varn='TTEND'

fo = 'historical' # forcing (e.g., ssp245)
# fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=False
freq='day'

cname=casename(fo)

def calc_tten(md):
    rlm=realm(varn.lower())
    hst=history(varn.lower())

    idir='/project/amp02/miyawaki/data/share/cesm2/%s/%s/proc/tseries/%s_1' % (cname,rlm,freq)
    odir=idir
    for _,_,files in os.walk(idir):
        files=[fn for fn in files if '.T.' in fn]
        for fn in tqdm(files):
            print(fn)
            ofn='%s/%s'%(odir,fn.replace('.T.','.%s.'%varn))
            if checkexist and os.path.isfile(ofn):
                continue
            fn1='%s/%s'%(idir,fn)

            ta=xr.open_dataset(fn1)['T']
            tten=ta.copy()

            # compute temperature tendency
            ta=ta.data
            dt=(ta[2:,...]-ta[:-2,...])/(2*86400)
            dt=np.concatenate(((ta[[1],...]-ta[[0],...])/86400,dt),axis=0)
            dt=np.concatenate((dt,(ta[[-1],...]-ta[[-2],...])/86400),axis=0)

            # total horizontal ttenection
            tten.data=c.cpd*dt

            # save
            tten=tten.rename(varn)
            tten.to_netcdf(ofn)

calc_tten('CESM2')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_tten,lmd)
