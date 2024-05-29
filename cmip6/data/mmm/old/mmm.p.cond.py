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

lvn=['rsfc'] # input1
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'

lmd=mods(fo1)
# lmd=['UKESM1-0-LL']

def calc_mmm(varn):
    if varn=='ooplh_fixbc':
        varn0='ooplh'
        varnp='plh'
    elif varn=='ooplh_dbc':
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
                pvn2=ds2[varnp]
            except:
                pvn2=ds2['__xarray_dataarray_variable__']

        # mean
        ds1=xr.open_dataset('%s/m.%s_%s.%s.nc' % (idir1,varn0,his,se))
        try:
            mvn1=ds1[varn0]
        except:
            mvn1=ds1['__xarray_dataarray_variable__']

        if 'fixmsm' in varn:
            idirsp = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo2,md,'%s_fixbc'%varn0)
            ds2=xr.open_dataset('%s/m.%s_%s.%s.nc' % (idirsp,'%s_fixbc'%varn0,fut,se))
        elif 'msm' in varn:
            idirsp = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo2,md,varn0)
            ds2=xr.open_dataset('%s/m.%s_%s.%s.nc' % (idirsp,varn0,fut,se))
        else:
            ds2=xr.open_dataset('%s/m.%s_%s.%s.nc' % (idir2,varn,fut,se))
        try:
            mvn2=ds2[varn0]
        except:
            try:
                mvn2=ds2[varnp]
            except:
                mvn2=ds2['__xarray_dataarray_variable__']

        # warming
        dvn=mvn2-mvn1 # mean
        dpvn=pvn2-pvn1
        ddpvn=dpvn-np.transpose(dvn.data[...,None],[0,2,1])

        # save individual model data
        dvn.to_netcdf('%s/d.%s_%s_%s.%s.nc' % (odir0,varn,his,fut,se))
        dpvn.to_netcdf('%s/dpc.%s_%s_%s.%s.nc' % (odir0,varn,his,fut,se))
        ddpvn.to_netcdf('%s/ddpc.%s_%s_%s.%s.nc' % (odir0,varn,his,fut,se))

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

    mmvn1.to_netcdf('%s/md.%s_%s.%s.nc' % (odir1,varn,his,se))
    mmvn2.to_netcdf('%s/md.%s_%s.%s.nc' % (odir2,varn,fut,se))
    smvn1.to_netcdf('%s/std.md.%s_%s.%s.nc' % (odir1,varn,his,se))
    smvn2.to_netcdf('%s/std.md.%s_%s.%s.nc' % (odir2,varn,fut,se))

    mpvn1.to_netcdf('%s/pc.%s_%s.%s.nc' % (odir1,varn,his,se))
    mpvn2.to_netcdf('%s/pc.%s_%s.%s.nc' % (odir2,varn,fut,se))
    spvn1.to_netcdf('%s/std.pc.%s_%s.%s.nc' % (odir1,varn,his,se))
    spvn2.to_netcdf('%s/std.pc.%s_%s.%s.nc' % (odir2,varn,fut,se))

    mdvn.to_netcdf('%s/d.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))
    sdvn.to_netcdf('%s/std.d.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))

    mdpvn.to_netcdf('%s/dpc.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))
    sdpvn.to_netcdf('%s/std.dpc.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))

    mddpvn.to_netcdf('%s/ddpc.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))
    sddpvn.to_netcdf('%s/std.ddpc.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))

[calc_mmm(vn) for vn in lvn]
