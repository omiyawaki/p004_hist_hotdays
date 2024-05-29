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

histslope=False
checkexist=False
lvn=['gflx'] # input1
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'

lmd=mods(fo1)

def load_rgr(md,fo0,vn0):
    cvn='%s+%s'%('tas',vn0)
    idira='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s'%(se,fo0,md,cvn)
    x1=xr.open_dataarray('%s/pw.rgr.%s.%s.nc' % (idira,cvn,'x1'))
    x2=xr.open_dataarray('%s/pw.rgr.%s.%s.nc' % (idira,cvn,'x2'))
    x3=xr.open_dataarray('%s/pw.rgr.%s.%s.nc' % (idira,cvn,'x3'))
    y1=xr.open_dataarray('%s/pw.rgr.%s.%s.nc' % (idira,cvn,'y1'))
    y2=xr.open_dataarray('%s/pw.rgr.%s.%s.nc' % (idira,cvn,'y2'))
    y3=xr.open_dataarray('%s/pw.rgr.%s.%s.nc' % (idira,cvn,'y3'))
    x=xr.concat([x1,x2,x3],'point')
    y=xr.concat([y1,y2,y3],'point')
    return x,y

def eval_rgr(xq,x,y):
    yq=np.empty_like(xq.data)
    for imn in tqdm(x['month']):
        for igp in x['gpi']:
            xqs=xq.isel(month=imn-1,gpi=igp)
            xs,ys=x.isel(month=imn-1,gpi=igp),y.isel(month=imn-1,gpi=igp)
            yq[imn-1,:,igp]=np.interp(xqs,xs,ys)
    return yq

def calc_mmm(varn):
    varn0=varn
    varn='%s_t'%varn0
    if histslope:
        varn='%s_hs'%varn

    for i,md in enumerate(tqdm(lmd)):
        print(md)
        idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo1,md,varn0)
        idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo2,md,varn0)
        odir0 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        if not os.path.exists(odir0):
            os.makedirs(odir0)

        ofn='%s/ddpc.md.%s_%s_%s.%s.nc' % (odir0,varn,his,fut,se)
        if checkexist and os.path.exists(ofn):
            ddpvn=xr.open_dataarray(ofn)
        else:
            # load regression coef
            x1,y1=load_rgr(md,fo1,varn0)
            x2,y2=load_rgr(md,fo2,varn0)

            # prc conditioned on temp
            pvn1=xr.open_dataarray('%s/pc.%s_%s.%s.nc' % (idir1,varn0,his,se))
            pvn2=xr.open_dataarray('%s/pc.%s_%s.%s.nc' % (idir2,varn0,fut,se))
            mon=pvn1['month']
            pct=pvn1['percentile']
            gpi=pvn1['gpi']

            # warming
            mvn1=1/2*(pvn1.sel(percentile=[47.5]).data+pvn1.sel(percentile=[52.5]).data)
            mvn2=1/2*(pvn2.sel(percentile=[47.5]).data+pvn2.sel(percentile=[52.5]).data)
            mvn1=xr.DataArray(mvn1,coords={'month':mon,'percentile':[50],'gpi':gpi},dims=('month','percentile','gpi'))
            mvn2=xr.DataArray(mvn2,coords={'month':mon,'percentile':[50],'gpi':gpi},dims=('month','percentile','gpi'))
            mvn1=eval_rgr(mvn1,x1,y1)
            pvn1=eval_rgr(pvn1,x1,y1)
            if histslope:
                mvn2=eval_rgr(mvn2,x1,y1)
                pvn2=eval_rgr(pvn2,x1,y1)
            else:
                mvn2=eval_rgr(mvn2,x2,y2)
                pvn2=eval_rgr(pvn2,x2,y2)
            ddpvn=pvn2-mvn2-(pvn1-mvn1)

            # save individual model data
            ddpvn=xr.DataArray(ddpvn,coords={'month':mon,'percentile':pct,'gpi':gpi},dims=('month','percentile','gpi'))
            ddpvn.to_netcdf(ofn)

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

    mddpvn.to_netcdf('%s/ddpc.md.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))
    sddpvn.to_netcdf('%s/std.ddpc.md.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))

[calc_mmm(vn) for vn in lvn]
