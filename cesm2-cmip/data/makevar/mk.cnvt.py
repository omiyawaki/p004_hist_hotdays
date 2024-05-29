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

varn='CNVT'

# fo = 'historical' # forcing (e.g., ssp245)
fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=False
freq='day'

cname=casename(fo)

def calc_cnv(md):
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
            print(plev[0,:,-1,0])
            sys.exit()

            # compute cpt flux divergence
            cnv=wap.copy()
            # compute cpt
            cpt=c.cpd*ta.data
            # vertical derivative
            dp=(cpt[:,2:,...]-cpt[:,:-2,...])/(plev[:,2:,...]-plev[:,:-2,...])
            dp=np.concatenate(((cpt[:,[1],...]-cpt[:,[0],...])/(plev[:,[1],...]-plev[:,[0],...]),dp),axis=1)
            dp=np.concatenate((dp,(cpt[:,[-1],...]-cpt[:,[-2],...])/(plev[:,[-1],...]-plev[:,[-2],...])),axis=1)

            # total horizontal cnvection
            cnv.data=wap*dp

            # save
            cnv=cnv.rename(varn)
            cnv.to_netcdf(ofn)

calc_cnv('CESM2')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_cnv,lmd)
