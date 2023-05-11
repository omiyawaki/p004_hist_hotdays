import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
import pickle
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm
from cmip6util import mods
from utils import monname

lvn=['hfls']
# lvn=['pr','hfss','hfls','rsds','rsus','rlds','rlus']
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'

lmd=mods(fo1)

def calc_mmm(varn):
    for i,md in enumerate(tqdm(lmd)):
        print(md)
        idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn)
        idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn)

        c = 0
        dt={}

        # prc temp
        ds1=xr.open_dataset('%s/p%s_%s.%s.nc' % (idir1,varn,his,se))
        gr={}
        gr['lat']=ds1['lat']
        gr['lon']=ds1['lon']
        try:
            pvn1=ds1[varn]
        except:
            try:
                vn1=ds1['tpd']
            except:
                pvn1=ds1['__xarray_dataarray_variable__']
        pvn1.data=pvn1.data**2 # variance
        pvn1=pvn1.groupby('time.month').mean('time') # monthly means
        ds2=xr.open_dataset('%s/p%s_%s.%s.nc' % (idir2,varn,fut,se))
        try:
            pvn2=ds2[varn]
        except:
            try:
                vn1=ds1['tpd']
            except:
                pvn2=ds2['__xarray_dataarray_variable__']
        pvn2.data=pvn2.data**2 # variance
        pvn2=pvn2.groupby('time.month').mean('time') # monthly means

        # save individual model data
        if i==0:
            ipvn1=np.empty(np.insert(np.asarray(pvn1.shape),0,len(lmd)))
            ipvn2=np.empty(np.insert(np.asarray(pvn2.shape),0,len(lmd)))

        ipvn1[i,...]=pvn1
        ipvn2[i,...]=pvn2

    # compute mmm and std
    mpvn1=np.nanmean(ipvn1,axis=0)
    mpvn2=np.nanmean(ipvn2,axis=0)

    spvn1=np.nanstd(ipvn1,axis=0)
    spvn2=np.nanstd(ipvn2,axis=0)

    # convert back to std
    ipvn1=np.sqrt(ipvn1)
    ipvn2=np.sqrt(ipvn2)
    mpvn1=np.sqrt(mpvn1)
    mpvn2=np.sqrt(mpvn2)
    spvn1=np.sqrt(spvn1)
    spvn2=np.sqrt(spvn2)

    # save mmm and std
    odir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,'mmm',varn)
    odir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,'mmm',varn)
    if not os.path.exists(odir1):
        os.makedirs(odir1)
    if not os.path.exists(odir2):
        os.makedirs(odir2)

    pickle.dump([mpvn1,spvn1,gr], open('%s/stdp%s_%s.%s.pickle' % (odir1,varn,his,se), 'wb'), protocol=5)	
    pickle.dump([mpvn2,spvn2,gr], open('%s/stdp%s_%s.%s.pickle' % (odir2,varn,fut,se), 'wb'), protocol=5)	

    # save ensemble
    odir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,'mi',varn)
    odir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,'mi',varn)
    if not os.path.exists(odir1):
        os.makedirs(odir1)
    if not os.path.exists(odir2):
        os.makedirs(odir2)

    pickle.dump(ipvn1, open('%s/stdp%s_%s.%s.pickle' % (odir1,varn,his,se), 'wb'), protocol=5)	
    pickle.dump(ipvn2, open('%s/stdp%s_%s.%s.pickle' % (odir2,varn,fut,se), 'wb'), protocol=5)	

if __name__=='__main__':
    with ProgressBar():
        tasks=[dask.delayed(calc_mmm)(vn) for vn in lvn]
        dask.compute(*tasks,scheduler='processes')
        # dask.compute(*tasks,scheduler='single-threaded')
