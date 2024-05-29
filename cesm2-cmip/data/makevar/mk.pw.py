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

varn='PW'

# fo = 'historical' # forcing (e.g., ssp245)
fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=False
freq='day'

cname=casename(fo)

def calc_pw(md):
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
            fn1=fn1.replace('.T.','.OMEGA.')
            wap=xr.open_dataset(fn1)['OMEGA']
            fn1=fn1.replace('.OMEGA.','.PS.')
            ps=xr.open_dataset(fn1)['PS']

            hyam=xr.open_dataset(fn1)['hyam']
            hybm=xr.open_dataset(fn1)['hybm']
            p0=xr.open_dataset(fn1)['P0']
            plev=hyam*p0+hybm*ps
            plev=np.transpose(plev.data,[1,0,2,3])

            # compute cpt flux divergence
            pw=wap.copy()
            # compute pressure work
            pw=-wap*c.Rd*ta/(plev*c.cpd)

            # save
            pw=pw.rename(varn)
            pw.to_netcdf(ofn)

calc_pw('CESM2')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_pw,lmd)
