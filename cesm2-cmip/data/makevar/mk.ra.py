import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import numpy as np
import xarray as xr
import constants as c
from tqdm import tqdm
from util import casename
from cesmutils import realm,history

# collect warmings across the ensembles

varn='RA'
varn1='FSNTOA'
varn2='FLNT'
varn3='FSNS'
varn4='FLNS'
freq='day'

# fo = 'historical' # forcing (e.g., ssp245)
fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=True

cname=casename(fo)

def calc_ra(md):
    rlm=realm(varn.lower())
    hst=history(varn.lower())

    chk=0
    idir='/project/amp02/miyawaki/data/share/cesm2/%s/%s/proc/tseries/%s_1' % (cname,rlm,freq)
    odir=idir
    for _,_,files in os.walk(idir):
        files=[fn for fn in files if varn1 in fn]
        for fn in tqdm(files):
            ofn='%s/%s'%(odir,fn.replace(varn1,varn))
            if checkexist and os.path.isfile(ofn):
                continue
            fn1='%s/%s'%(idir,fn)
            fsnt=xr.open_dataset(fn1)[varn1]
            fn1=fn1.replace(varn1,varn2)
            flnt=xr.open_dataset(fn1)[varn2]
            fn1=fn1.replace(varn2,varn3)
            fsns=xr.open_dataset(fn1)[varn3]
            fn1=fn1.replace(varn3,varn4)
            flns=xr.open_dataset(fn1)[varn4]
            # compute total evapotranspiration
            ra=fsnt.copy()
            ra.data=fsnt.data-flnt.data-fsns.data+flns.data
            ra=ra.rename(varn)
            ra.to_netcdf(ofn)

calc_ra('CESM2')
