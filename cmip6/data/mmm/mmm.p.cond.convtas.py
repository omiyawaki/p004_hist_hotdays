import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client
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
from util import mods
from utils import monname

lvn=['advty_doy850'] # input1
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'

lmd=mods(fo1)
# lmd=['CESM2']

def load_rgr(md,fo0,vn0):
    cvn='%s+%s'%('tas',vn0)
    idira='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s'%(se,fo0,md,cvn)
    return xr.open_dataarray('%s/rgr.%s.nc' % (idira,cvn))

def calc_mmm(varn):
    varn0=varn
    varn='%s_t'%varn0

    for i,md in enumerate(tqdm(lmd)):
        print(md)
        idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo1,md,varn0)
        idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo2,md,varn0)
        odir0 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        if not os.path.exists(odir0):
            os.makedirs(odir0)

        c = 0
        dt={}

        # load regression coef
        ae1=load_rgr(md,fo1,varn0)
        ae2=load_rgr(md,fo2,varn0)

        # prc conditioned on temp
        pvn1=xr.open_dataarray('%s/pc.%s_%s.%s.nc' % (idir1,varn0,his,se))
        pvn2=xr.open_dataarray('%s/pc.%s_%s.%s.nc' % (idir2,varn0,fut,se))

        # mean
        mvn1=xr.open_dataarray('%s/m.%s_%s.%s.nc' % (idir1,varn0,his,se))
        mvn2=xr.open_dataarray('%s/m.%s_%s.%s.nc' % (idir2,varn0,fut,se))

        # warming
        de1=ae1*(pvn1-np.transpose(mvn1.data[...,None],[0,2,1]))
        de2=ae2*(pvn2-np.transpose(mvn2.data[...,None],[0,2,1]))
        ddpvn=de2-de1

        # save individual model data
        ddpvn.to_netcdf('%s/ddpc.%s_%s_%s.%s.nc' % (odir0,varn,his,fut,se))

        if i==0:
            iddpvn=np.empty(np.insert(np.asarray(ddpvn.shape),0,len(lmd)))

        iddpvn[i,...]=ddpvn

    # compute mmm and std
    mddpvn=ddpvn.copy()
    mddpvn.data=np.nanmean(iddpvn,axis=0)

    sddpvn=ddpvn.copy()
    sddpvn.data=np.nanstd(iddpvn,axis=0)

    # save mmm and std
    odir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,'mmm',varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    mddpvn.to_netcdf('%s/ddpc.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))
    sddpvn.to_netcdf('%s/std.ddpc.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))

[calc_mmm(vn) for vn in lvn]
