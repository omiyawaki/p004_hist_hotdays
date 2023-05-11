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
from util import mods
from utils import monname

lvn=['mrsos'] # input1
# lvn=['huss','pr','mrsos','hfls','hfss']
# lvn=['pr','hfss','hfls','rsds','rsus','rlds','rlus']
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'
# fut='gwl2.0'

lmd=mods(fo1)

def calc_mmm(varn):
    for i,md in enumerate(tqdm(lmd)):
        print(md)
        idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo1,md,varn)
        idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo2,md,varn)

        c = 0
        dt={}

        # prc conditioned on temp
        ds1=xr.open_dataset('%s/pc.%s_%s.%s.nc' % (idir1,varn,his,se))
        try:
            pvn1=ds1[varn]
        except:
            pvn1=ds1['__xarray_dataarray_variable__']
        ds2=xr.open_dataset('%s/pc.%s_%s.%s.nc' % (idir2,varn,fut,se))
        try:
            pvn2=ds2[varn]
        except:
            pvn2=ds2['__xarray_dataarray_variable__']

        # warming
        dvn=pvn2[:,0,:]-pvn1[:,0,:] # mean
        dpvn=pvn2-pvn1
        ddpvn=dpvn-np.transpose(dvn.data[...,None],[0,2,1])

        # save individual model data
        if i==0:
            ipvn1=np.empty(np.insert(np.asarray(pvn1.shape),0,len(lmd)))
            ipvn2=np.empty(np.insert(np.asarray(pvn2.shape),0,len(lmd)))
            idvn=np.empty(np.insert(np.asarray(dvn.shape),0,len(lmd)))
            idpvn=np.empty(np.insert(np.asarray(dpvn.shape),0,len(lmd)))
            iddpvn=np.empty(np.insert(np.asarray(ddpvn.shape),0,len(lmd)))

        ipvn1[i,...]=pvn1
        ipvn2[i,...]=pvn2
        idvn[i,...]=dvn
        idpvn[i,...]=dpvn
        iddpvn[i,...]=ddpvn

    # compute mmm and std
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

    mpvn1.to_netcdf('%s/p.%s_%s.%s.nc' % (odir1,varn,his,se))
    mpvn2.to_netcdf('%s/p.%s_%s.%s.nc' % (odir2,varn,fut,se))
    spvn1.to_netcdf('%s/std.p.%s_%s.%s.nc' % (odir1,varn,his,se))
    spvn2.to_netcdf('%s/std.p.%s_%s.%s.nc' % (odir2,varn,fut,se))

    mdvn.to_netcdf('%s/d.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))
    sdvn.to_netcdf('%s/std.d.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))

    mdpvn.to_netcdf('%s/dp.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))
    sdpvn.to_netcdf('%s/std.dp.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))

    mddpvn.to_netcdf('%s/ddp.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))
    sddpvn.to_netcdf('%s/std.ddp.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))

    # # save ensemble
    # odir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo1,'mi',varn)
    # odir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo2,'mi',varn)
    # odir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,'mi',varn)
    # if not os.path.exists(odir1):
    #     os.makedirs(odir1)
    # if not os.path.exists(odir2):
    #     os.makedirs(odir2)
    # if not os.path.exists(odir):
    #     os.makedirs(odir)

    # pickle.dump(ipvn1, open('%s/p.%s_%s.%s.pickle' % (odir1,varn,his,se), 'wb'), protocol=5)	
    # pickle.dump(ipvn2, open('%s/p.%s_%s.%s.pickle' % (odir2,varn,fut,se), 'wb'), protocol=5)	
    # pickle.dump(idvn, open('%s/d.%s_%s_%s.%s.pickle' % (odir,varn,his,fut,se), 'wb'), protocol=5)	
    # pickle.dump(idpvn, open('%s/dp.%s_%s_%s.%s.pickle' % (odir,varn,his,fut,se), 'wb'), protocol=5)	
    # pickle.dump(iddpvn, open('%s/ddp.%s_%s_%s.%s.pickle' % (odir,varn,his,fut,se), 'wb'), protocol=5)	

if __name__=='__main__':
    with ProgressBar():
        tasks=[dask.delayed(calc_mmm)(vn) for vn in lvn]
        # dask.compute(*tasks,scheduler='processes')
        dask.compute(*tasks,scheduler='single-threaded')
