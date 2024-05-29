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

varn='DWH'

# fo = 'historical' # forcing (e.g., ssp245)
fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=False
freq='day'

cname=casename(fo)

def calc_dwh(md):
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
            fn1=fn1.replace('.MSE.','.OMEGA.')
            wap=xr.open_dataset(fn1)['OMEGA']
            fn1=fn1.replace('.OMEGA.','.PS.')
            ps=xr.open_dataset(fn1)['PS']

            hyam=xr.open_dataset(fn1)['hyam']
            hybm=xr.open_dataset(fn1)['hybm']
            p0=xr.open_dataset(fn1)['P0']
            plev=hyam*p0+hybm*ps
            plev=np.transpose(plev.data,[1,0,2,3])

            # compute wh flux divergence
            dwh=wap.copy()
            # compute wh
            wh=wap.data*mse.data
            # vertical derivative
            dp=(wh[:,2:,...]-wh[:,:-2,...])/(plev[:,2:,...]-plev[:,:-2,...])
            dp=np.concatenate(((wh[:,[1],...]-wh[:,[0],...])/(plev[:,[1],...]-plev[:,[0],...]),dp),axis=1)
            dp=np.concatenate((dp,(wh[:,[-1],...]-wh[:,[-2],...])/(plev[:,[-1],...]-plev[:,[-2],...])),axis=1)

            # save
            dwh.data=dp
            dwh=dwh.rename(varn)
            dwh.to_netcdf(ofn)

calc_dwh('CESM2')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_dwh,lmd)
