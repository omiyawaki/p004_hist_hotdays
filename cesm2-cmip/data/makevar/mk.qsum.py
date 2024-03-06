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

varn='QSUM'
varn1='QSOIL'
varn2='QVEGE'
varn3='QVEGT'
freq='day'

fo = 'historical' # forcing (e.g., ssp245)
# fo = 'ssp370' # forcing (e.g., ssp245)

cname=casename(fo)

def calc_qsum(md):
    rlm=realm(varn.lower())
    hst=history(varn.lower())

    chk=0
    idir='/project/amp02/miyawaki/data/share/cesm2/%s/%s/proc/tseries/%s_1' % (cname,rlm,freq)
    odir=idir
    for _,_,files in os.walk(idir):
        files=[fn for fn in files if varn1 in fn]
        for fn in tqdm(files):
            ofn='%s/%s'%(odir,fn.replace(varn1,varn))
            # if os.path.isfile(ofn):
            #     continue
            fn1='%s/%s'%(idir,fn)
            qsoil=xr.open_dataset(fn1)[varn1]
            fn1=fn1.replace(varn1,varn2)
            qvege=xr.open_dataset(fn1)[varn2]
            fn1=fn1.replace(varn2,varn3)
            qvegt=xr.open_dataset(fn1)[varn3]
            # compute total evapotranspiration
            qsum=qsoil.copy()
            qsum.data=qsoil.data+qvege.data+qvegt.data
            qsum=qsum.rename(varn)
            qsum.to_netcdf(ofn)

calc_qsum('CESM2')
