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

varn='DIVT'

# fo = 'historical' # forcing (e.g., ssp245)
fo = 'ssp370' # forcing (e.g., ssp245)

checkexist=False
freq='day'

cname=casename(fo)

def calc_div(md):
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
            fn1=fn1.replace('.T.','.U.')
            ua=xr.open_dataset(fn1)['U']
            fn1=fn1.replace('.U.','.V.')
            va=xr.open_dataset(fn1)['V']

            lat=np.deg2rad(ta['lat'].data)
            lon=np.deg2rad(ta['lon'].data)

            # compute cpt flux divergence
            div=ua.copy()
            # compute cpt
            cpt=c.cpd*ta.data
            clat=np.transpose(np.tile(np.cos(lat),(1,1,1,1)),[0,1,3,2])
            cptu=ua.data*cpt.data
            cptv=clat*va.data*cpt.data
            # zonal derivative
            dx=1/(c.a*clat)*(cptu[...,2:]-cptu[...,:-2])/(lon[2:]-lon[:-2])
            dx=np.concatenate((1/(c.a*clat)*(cptu[...,[1]]-cptu[...,[0]])/(lon[1]-lon[0]),dx),axis=-1)
            dx=np.concatenate((dx,1/(c.a*clat)*(cptu[...,[-1]]-cptu[...,[-2]])/(lon[-1]-lon[-2])),axis=-1)
            # meridional divergence
            dy=1/(c.a*clat[:,:,1:-1,:])*(cptv[:,:,2:,:]-cptv[:,:,:-2,:])/(lat[2:]-lat[:-2]).reshape([1,len(lat)-2,1])
            dy=np.concatenate((1/(c.a*clat[:,:,[0],:])*(cptv[:,:,[1],:]-cptv[:,:,[0],:])/(lat[1]-lat[0]),dy),axis=-2)
            dy=np.concatenate((dy,1/(c.a*clat[:,:,[-1],:])*(cptv[:,:,[-1],:]-cptv[:,:,[-2],:])/(lat[-1]-lat[-2])),axis=-2)

            # total horizontal divection
            div.data=dx+dy

            # save
            div=div.rename(varn)
            div.to_netcdf(ofn)

calc_div('CESM2')

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(calc_div,lmd)
