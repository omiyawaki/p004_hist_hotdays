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

varn='ADVT'

fo = 'historical' # forcing (e.g., ssp245)
# fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=False
freq='day'

cname=casename(fo)

def calc_adv(md):
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
            fn1=fn1.replace('.T.','.U.')
            ua=xr.open_dataset(fn1)['U']
            fn1=fn1.replace('.U.','.V.')
            va=xr.open_dataset(fn1)['V']

            lat=np.deg2rad(ta['lat'].data)
            lon=np.deg2rad(ta['lon'].data)

            # compute cpt flux divergence
            adv=ua.copy()
            # compute cpt
            cpt=c.cpd*ta.data
            clat=np.transpose(np.tile(np.cos(lat),(1,1,1,1)),[0,1,3,2])
            cptc=clat*cpt
            # zonal derivative
            dlon=lon[1]-lon[0]
            dx=1/(c.a*clat)*(cpt[...,2:]-cpt[...,:-2])/(2*dlon)
            dx=np.concatenate((1/(c.a*clat)*(cpt[...,[1]]-cpt[...,[-1]])/(2*dlon),dx),axis=-1)
            dx=np.concatenate((dx,1/(c.a*clat)*(cpt[...,[0]]-cpt[...,[-2]])/(2*dlon)),axis=-1)
            # meridional divergence
            dy=1/(c.a*clat[:,:,1:-1,:])*(cptc[:,:,2:,:]-cptc[:,:,:-2,:])/(lat[2:]-lat[:-2]).reshape([1,1,len(lat)-2,1])
            dy=np.concatenate((1/(c.a*clat[:,:,[0],:])*(cptc[:,:,[1],:]-cptc[:,:,[0],:])/(lat[1]-lat[0]),dy),axis=-2)
            dy=np.concatenate((dy,1/(c.a*clat[:,:,[-1],:])*(cptc[:,:,[-1],:]-cptc[:,:,[-2],:])/(lat[-1]-lat[-2])),axis=-2)

            # total horizontal advection
            adv.data=ua.data*dx+va.data*dy

            # save
            adv=adv.rename(varn)
            adv.to_netcdf(ofn)

calc_adv('CESM2')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_adv,lmd)
