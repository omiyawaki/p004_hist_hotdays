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

p=[95]
only95=False
lvn=['mrsos'] # input1
# lvn=['huss','pr','mrsos','hfls','hfss']
# lvn=['pr','hfss','hfls','rsds','rsus','rlds','rlus']
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'

lmd=mods(fo1)

def calc_mmm(varn):
    if 'ooplh_'==varn[:6] and varn!='ooplh_orig':
        varn0='ooplh'
    elif 'plh_'==varn[:4] and varn!='plh_orig':
        varn0='plh'
    else:
        varn0=varn

    for i,md in enumerate(tqdm(lmd)):
        print(md)
        idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo1,md,varn0)
        idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo2,md,varn)

        c = 0
        dt={}

        # load data
        cdir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo1,md,'csm')
        cdir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo2,md,'csm')
        ds1=xr.open_dataset('%s/%s.%s.%s.nc' % (cdir1,'csm',his,se))
        try:
            csm1=ds1['csm']
        except:
            csm1=ds1['__xarray_dataarray_variable__']
        ds2=xr.open_dataset('%s/%s.%s.%s.nc' % (cdir2,'csm',fut,se))
        try:
            csm2=ds2['csm']
        except:
            csm2=ds2['__xarray_dataarray_variable__']

        # replace inf with nans
        csm1.data[np.logical_or(csm1.data==np.inf,csm1.data==-np.inf)]=np.nan
        csm2.data[np.logical_or(csm2.data==np.inf,csm2.data==-np.inf)]=np.nan

        # prc conditioned on temp
        ds1=xr.open_dataset('%s/pc.%s_%s.%s.nc' % (idir1,varn0,his,se))
        try:
            pvn1=ds1[varn0]
        except:
            pvn1=ds1['__xarray_dataarray_variable__']
        ds2=xr.open_dataset('%s/pc.%s_%s.%s.nc' % (idir2,varn,fut,se))
        try:
            pvn2=ds2[varn]
        except:
            try:
                pvn2=ds2['plh']
            except:
                pvn2=ds2['__xarray_dataarray_variable__']

        if only95:
            pvn1=pvn1.sel(percentile=p)
            pvn2=pvn2.sel(percentile=p)

        # mean
        ds1=xr.open_dataset('%s/m.%s_%s.%s.nc' % (idir1,varn0,his,se))
        try:
            mvn1=ds1[varn0]
        except:
            mvn1=ds1['__xarray_dataarray_variable__']
        ds2=xr.open_dataset('%s/m.%s_%s.%s.nc' % (idir2,varn,fut,se))
        try:
            mvn2=ds2[varn0]
        except:
            try:
                mvn2=ds2['plh']
            except:
                mvn2=ds2['__xarray_dataarray_variable__']

        # anomaly from crit sm
        pvn2=pvn2-csm2
        pvn1=pvn1-csm1
        mvn2=mvn2-csm2
        mvn1=mvn1-csm1


        # warming
        dvn=mvn2-mvn1 # mean
        dpvn=pvn2-pvn1
        ddpvn=dpvn-np.transpose(dvn.data[...,None],[0,2,1])

        # save individual model data
        if i==0:
            imvn1=np.empty(np.insert(np.asarray(mvn1.shape),0,len(lmd)))
            imvn2=np.empty(np.insert(np.asarray(mvn2.shape),0,len(lmd)))
            ipvn1=np.empty(np.insert(np.asarray(pvn1.shape),0,len(lmd)))
            ipvn2=np.empty(np.insert(np.asarray(pvn2.shape),0,len(lmd)))
            idvn=np.empty(np.insert(np.asarray(dvn.shape),0,len(lmd)))
            idpvn=np.empty(np.insert(np.asarray(dpvn.shape),0,len(lmd)))
            iddpvn=np.empty(np.insert(np.asarray(ddpvn.shape),0,len(lmd)))

        imvn1[i,...]=mvn1
        imvn2[i,...]=mvn2
        ipvn1[i,...]=pvn1
        ipvn2[i,...]=pvn2
        idvn[i,...]=dvn
        idpvn[i,...]=dpvn
        iddpvn[i,...]=ddpvn

    # compute mmm and std
    mmvn1=mvn1.copy()
    mmvn2=mmvn1.copy()
    mmvn1.data=np.nanmean(imvn1,axis=0)
    mmvn2.data=np.nanmean(imvn2,axis=0)

    mpvn1=pvn1.copy()
    mpvn2=mpvn1.copy()
    mpvn1.data=np.nanmean(ipvn1,axis=0)
    mpvn2.data=np.nanmean(ipvn2,axis=0)

    mdvn=np.nanmean(idvn,axis=0)
    mdvn=xr.DataArray(mdvn,coords={'month':mpvn1['month'],'gpi':mpvn1['gpi']},dims=('month','gpi'))

    mdpvn=mpvn1.copy()
    mddpvn=mpvn1.copy()
    mdpvn.data=np.nanmean(idpvn,axis=0)
    mddpvn.data=np.nanmean(iddpvn,axis=0)

    smvn1=mmvn1.copy()
    smvn2=mmvn1.copy()
    smvn1.data=np.nanstd(imvn1,axis=0)
    smvn2.data=np.nanstd(imvn2,axis=0)

    spvn1=mpvn1.copy()
    spvn2=mpvn1.copy()
    spvn1.data=np.nanstd(ipvn1,axis=0)
    spvn2.data=np.nanstd(ipvn2,axis=0)

    sdvn=np.nanstd(idvn,axis=0)
    sdvn=xr.DataArray(sdvn,coords={'month':mpvn1['month'],'gpi':mpvn1['gpi']},dims=('month','gpi'))

    sdpvn=mpvn1.copy()
    sddpvn=mpvn2.copy()
    sdpvn.data=np.nanstd(idpvn,axis=0)
    sddpvn.data=np.nanstd(iddpvn,axis=0)

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

    mmvn1.to_netcdf('%s/anom.m.%s_%s.%s.nc' % (odir1,varn,his,se))
    mmvn2.to_netcdf('%s/anom.m.%s_%s.%s.nc' % (odir2,varn,fut,se))
    smvn1.to_netcdf('%s/std.anom.m.%s_%s.%s.nc' % (odir1,varn,his,se))
    smvn2.to_netcdf('%s/std.anom.m.%s_%s.%s.nc' % (odir2,varn,fut,se))

    mpvn1.to_netcdf('%s/anom.pc.%s_%s.%s.nc' % (odir1,varn,his,se))
    mpvn2.to_netcdf('%s/anom.pc.%s_%s.%s.nc' % (odir2,varn,fut,se))
    spvn1.to_netcdf('%s/std.anom.pc.%s_%s.%s.nc' % (odir1,varn,his,se))
    spvn2.to_netcdf('%s/std.anom.pc.%s_%s.%s.nc' % (odir2,varn,fut,se))

    mdvn.to_netcdf('%s/anom.d.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))
    sdvn.to_netcdf('%s/std.anom.d.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))

    mdpvn.to_netcdf('%s/anom.dpc.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))
    sdpvn.to_netcdf('%s/std.anom.dpc.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))

    mddpvn.to_netcdf('%s/anom.ddpc.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))
    sddpvn.to_netcdf('%s/std.anom.ddpc.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))

[calc_mmm(vn) for vn in lvn]

# if __name__=='__main__':
#     with Client(n_workers=len(lvn)):
#         tasks=[dask.delayed(calc_mmm)(vn) for vn in lvn]
#         # dask.compute(*tasks,scheduler='processes')
#         dask.compute(*tasks)
