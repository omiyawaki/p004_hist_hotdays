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

lvn=['tas'] # input1
se = 'ond+amj' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'

lmd=mods(fo1)
# lmd=['CESM2']

def calc_mmm(varn):
    varn0=varn

    for i,md in enumerate(tqdm(lmd)):
        print(md)
        idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo1,md,varn0)
        idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo2,md,varn)
        odir0 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        if not os.path.exists(odir0):
            os.makedirs(odir0)

        c = 0
        dt={}

        # mean
        mvn1=xr.open_dataarray('%s/m.%s_%s.%s.nc' % (idir1,varn0,his,se))
        mvn2=xr.open_dataarray('%s/m.%s_%s.%s.nc' % (idir2,varn0,fut,se))
        lat,lon=mvn1['lat'],mvn1['lon']

        # warming
        dvn=mvn2-mvn1 # mean

        # save individual model data
        dvn.to_netcdf('%s/d.%s_%s_%s.%s.nc' % (odir0,varn,his,fut,se))

        if i==0:
            imvn1=np.empty(np.insert(np.asarray(mvn1.shape),0,len(lmd)))
            imvn2=np.empty(np.insert(np.asarray(mvn2.shape),0,len(lmd)))
            idvn=np.empty(np.insert(np.asarray(dvn.shape),0,len(lmd)))

        imvn1[i,...]=mvn1
        imvn2[i,...]=mvn2
        idvn[i,...]=dvn

    # compute mmm and std
    mmvn1=mvn1.copy()
    mmvn2=mmvn1.copy()
    mmvn1.data=np.nanmean(imvn1,axis=0)
    mmvn2.data=np.nanmean(imvn2,axis=0)

    smvn1=mmvn1.copy()
    smvn2=mmvn1.copy()
    smvn1.data=np.nanstd(imvn1,axis=0)
    smvn2.data=np.nanstd(imvn2,axis=0)

    mdvn=np.nanmean(idvn,axis=0)
    mdvn=xr.DataArray(mdvn,coords={'lat':lat,'lon':lon},dims=('lat','lon'))

    sdvn=np.nanstd(idvn,axis=0)
    sdvn=xr.DataArray(sdvn,coords={'lat':lat,'lon':lon},dims=('lat','lon'))

    # save mmm and std
    odir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo1,'mmm',varn)
    odir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo2,'mmm',varn)
    odir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,'mmm',varn)
    if not os.path.exists(odir1):
        os.makedirs(odir1)
    if not os.path.exists(odir2):
        os.makedirs(odir2)
    if not os.path.exists(odir):
        os.makedirs(odir)

    mmvn1.to_netcdf('%s/md.%s_%s.%s.nc' % (odir1,varn,his,se))
    mmvn2.to_netcdf('%s/md.%s_%s.%s.nc' % (odir2,varn,fut,se))
    smvn1.to_netcdf('%s/std.md.%s_%s.%s.nc' % (odir1,varn,his,se))
    smvn2.to_netcdf('%s/std.md.%s_%s.%s.nc' % (odir2,varn,fut,se))

    mdvn.to_netcdf('%s/d.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))
    sdvn.to_netcdf('%s/std.d.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))

[calc_mmm(vn) for vn in lvn]
