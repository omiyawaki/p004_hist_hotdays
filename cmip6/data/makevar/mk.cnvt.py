import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
from concurrent.futures import ProcessPoolExecutor as Pool
import numpy as np
import xarray as xr
import constants as c
from tqdm import tqdm
from util import casename,mods,simu,emem
from glade_utils import grid

# collect warmings across the ensembles

varn='cnvt'

fo = 'historical' # forcing (e.g., ssp245)
# fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=False
freq='day'

cname=casename(fo)

def calc_cnv(md):
    ens=emem(md)
    grd=grid(md)

    odir='/project/amp02/miyawaki/data/share/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
    if not os.path.exists(odir):
        os.makedirs(odir)

    idir='/project/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,'ta',md,ens,grd)
    for _,_,files in os.walk(idir):
        for fn in tqdm(files):
            print(fn)
            ofn='%s/%s'%(odir,fn.replace('ta','.%s.'%varn))
            if checkexist and os.path.isfile(ofn):
                continue
            fn1='%s/%s'%(idir,fn)

            ta=xr.open_dataset(fn1)['ta']
            plev=ta['plev'].data
            fn1=fn1.replace('ta','wap')
            wap=xr.open_dataset(fn1)['wap']

            # compute cpt flux divergence
            cnv=wap.copy()
            # compute cpt
            cpt=c.cpd*ta.data
            # vertical derivative
            dp=(cpt[:,2:,...]-cpt[:,:-2,...])/(plev[2:]-plev[:-2]).reshape(1,len(plev)-2,1,1)
            dp=np.concatenate(((cpt[:,[1],...]-cpt[:,[0],...])/(plev[1]-plev[0]),dp),axis=1)
            dp=np.concatenate((dp,(cpt[:,[-1],...]-cpt[:,[-2],...])/(plev[-1]-plev[-2])),axis=1)

            # total horizontal cnvection
            cnv.data=wap*dp

            # save
            cnv=cnv.rename(varn)
            cnv.to_netcdf(ofn)

calc_cnv('CESM2')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_cnv,lmd)
