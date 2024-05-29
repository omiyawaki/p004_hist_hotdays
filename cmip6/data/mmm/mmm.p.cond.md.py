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

lvn=['psl'] # input1
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'

lmd=mods(fo1)
# lmd=['CESM2']

def calc_mmm(varn):
    if varn=='ooplh_fixbc':
        varn0='ooplh'
        varnp='plh'
    elif varn=='ooplh_dbc':
        varn0='ooplh'
        varnp='plh'
    elif varn=='ooplh_rdbc':
        varn0='ooplh'
        varnp='plh'
    elif varn=='ooplh_rbcsm':
        varn0='ooplh'
        varnp='plh'
    elif varn=='ooplh_rddsm':
        varn0='ooplh'
        varnp='plh'
    elif varn=='ooplh_msm':
        varn0='ooplh'
        varnp='plh'
    elif varn=='ooplh_fixmsm':
        varn0='ooplh'
        varnp='plh'
    elif varn=='ooplh_fixasm':
        varn0='ooplh'
        varnp='plh'
    elif varn=='ooplh_mtr':
        varn0='ooplh'
        varnp='plh'
    elif varn=='ooplh_mtr':
        varn0='ooplh'
        varnp='plh'
    elif varn=='plh_fixbc':
        varn0='plh'
    elif varn=='oopef_fixbc':
        varn0='oopef'
        varnp='pef'
    elif varn=='oopef_fixmsm':
        varn0='oopef'
        varnp='pef'
    elif varn=='oopef_rddsm':
        varn0='oopef'
        varnp='pef'
    elif varn=='oopef3_fixbc':
        varn0='oopef3'
        varnp='pef3'
    elif varn=='pef_fixbc':
        varn0='pef'
    else:
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

        # prc conditioned on temp
        pvn1=xr.open_dataarray('%s/pc.%s_%s.%s.nc' % (idir1,varn0,his,se))
        pvn2=xr.open_dataarray('%s/pc.%s_%s.%s.nc' % (idir2,varn,fut,se))

        # warming
        mvn1=1/2*(pvn1.sel(percentile=[47.5]).data+pvn1.sel(percentile=[52.5]).data)
        mvn2=1/2*(pvn2.sel(percentile=[47.5]).data+pvn2.sel(percentile=[52.5]).data)
        de1=pvn1-mvn1
        de2=pvn2-mvn2
        dpvn=pvn2-pvn1
        ddpvn=de2-de1
        mvn1=xr.DataArray(mvn1.squeeze(),coords={'month':pvn1['month'],'gpi':pvn1['gpi']},dims=('month','gpi'))
        mvn2=xr.DataArray(mvn2.squeeze(),coords={'month':pvn2['month'],'gpi':pvn2['gpi']},dims=('month','gpi'))
        dmvn=mvn2-mvn1
        print(np.nanmax(ddpvn.data.flatten()))
        print(np.nanmin(ddpvn.data.flatten()))

        # save individual model data
        mvn1.to_netcdf('%s/md.%s_%s.%s.nc' % (idir1,varn,his,se))
        mvn2.to_netcdf('%s/md.%s_%s.%s.nc' % (idir2,varn,fut,se))
        dmvn.to_netcdf('%s/d.md.%s_%s_%s.%s.nc' % (odir0,varn,his,fut,se))
        dpvn.to_netcdf('%s/dpc.md.%s_%s_%s.%s.nc' % (odir0,varn,his,fut,se))
        ddpvn.to_netcdf('%s/ddpc.md.%s_%s_%s.%s.nc' % (odir0,varn,his,fut,se))

        if i==0:
            imvn1=np.empty(np.insert(np.asarray(mvn1.shape),0,len(lmd)))
            imvn2=np.empty(np.insert(np.asarray(mvn2.shape),0,len(lmd)))
            ipvn1=np.empty(np.insert(np.asarray(pvn1.shape),0,len(lmd)))
            ipvn2=np.empty(np.insert(np.asarray(pvn2.shape),0,len(lmd)))
            idmvn=np.empty(np.insert(np.asarray(dmvn.shape),0,len(lmd)))
            idpvn=np.empty(np.insert(np.asarray(dpvn.shape),0,len(lmd)))
            iddpvn=np.empty(np.insert(np.asarray(ddpvn.shape),0,len(lmd)))

        imvn1[i,...]=mvn1
        imvn2[i,...]=mvn2
        ipvn1[i,...]=pvn1
        ipvn2[i,...]=pvn2
        idmvn[i,...]=dmvn
        idpvn[i,...]=dpvn
        iddpvn[i,...]=ddpvn

    # compute mmm and std
    mmvn1=mvn1.copy()
    mmvn2=mvn2.copy()
    mmvn1.data=np.nanmean(imvn1,axis=0)
    mmvn2.data=np.nanmean(imvn2,axis=0)

    smvn1=mvn1.copy()
    smvn2=mvn2.copy()
    smvn1.data=np.nanstd(imvn1,axis=0)
    smvn2.data=np.nanstd(imvn2,axis=0)

    mpvn1=pvn1.copy()
    mpvn2=pvn2.copy()
    mpvn1.data=np.nanmean(ipvn1,axis=0)
    mpvn2.data=np.nanmean(ipvn2,axis=0)

    spvn1=pvn1.copy()
    spvn2=pvn2.copy()
    spvn1.data=np.nanstd(ipvn1,axis=0)
    spvn2.data=np.nanstd(ipvn2,axis=0)

    mdmvn=dmvn.copy()
    mdmvn.data=np.nanmean(idmvn,axis=0)
    sdmvn=dmvn.copy()
    sdmvn.data=np.nanstd(idmvn,axis=0)

    mdpvn=dpvn.copy()
    mdpvn.data=np.nanmean(idpvn,axis=0)
    sdpvn=dpvn.copy()
    sdpvn.data=np.nanstd(idpvn,axis=0)

    mddpvn=ddpvn.copy()
    mddpvn.data=np.nanmean(iddpvn,axis=0)
    sddpvn=ddpvn.copy()
    sddpvn.data=np.nanstd(iddpvn,axis=0)

    # save mmm and std
    odir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo1,'mmm',varn)
    if not os.path.exists(odir1):
        os.makedirs(odir1)
    odir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo2,'mmm',varn)
    if not os.path.exists(odir2):
        os.makedirs(odir2)
    odir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,'mmm',varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    mmvn1.to_netcdf('%s/md.%s_%s.%s.nc' % (odir1,varn,his,se))
    smvn1.to_netcdf('%s/std.md.%s_%s.%s.nc' % (odir1,varn,his,se))
    mmvn2.to_netcdf('%s/md.%s_%s.%s.nc' % (odir2,varn,fut,se))
    smvn2.to_netcdf('%s/std.md.%s_%s.%s.nc' % (odir2,varn,fut,se))

    mpvn1.to_netcdf('%s/pc.%s_%s.%s.nc' % (odir1,varn,his,se))
    spvn1.to_netcdf('%s/std.pc.%s_%s.%s.nc' % (odir1,varn,his,se))
    mpvn2.to_netcdf('%s/pc.%s_%s.%s.nc' % (odir2,varn,fut,se))
    spvn2.to_netcdf('%s/std.pc.%s_%s.%s.nc' % (odir2,varn,fut,se))

    mdmvn.to_netcdf('%s/d.md.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))
    sdmvn.to_netcdf('%s/std.d.md.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))

    mdpvn.to_netcdf('%s/dpc.md.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))
    sdpvn.to_netcdf('%s/std.dpc.md.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))

    mddpvn.to_netcdf('%s/ddpc.md.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))
    sddpvn.to_netcdf('%s/std.ddpc.md.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))

[calc_mmm(vn) for vn in lvn]
