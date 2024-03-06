import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm
from util import mods
from utils import corr

nt=7 # window size in days
p=95
pref1='dp'
varn1='tas'
pref2='dp'
varn2='ef2'
varn='%s%s+%s%s'%(pref1,varn1,pref2,varn2)
se = 'sc'
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

def load_data(md,varn):
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

    # mean
    ds1=xr.open_dataset('%s/m.%s_%s.%s.nc' % (idir1,varn,his,se))
    try:
        mvn1=ds1[varn]
    except:
        mvn1=ds1['__xarray_dataarray_variable__']
    ds2=xr.open_dataset('%s/m.%s_%s.%s.nc' % (idir2,varn,fut,se))
    try:
        mvn2=ds2[varn]
    except:
        mvn2=ds2['__xarray_dataarray_variable__']

    # warming
    dvn=mvn2-mvn1 # mean
    pvn1=pvn1.sel(percentile=p)
    pvn2=pvn2.sel(percentile=p)
    dpvn=pvn2-pvn1
    ddpvn=dpvn-dvn
    # ddpvn=dpvn-np.transpose(dvn.data[...,None],[0,2,1])

    return ddpvn

lmd=mods(fo1)
for i,md in enumerate(tqdm(lmd)):
    print(md)
    ddpvn1=load_data(md,varn1)
    ddpvn2=load_data(md,varn2)

    # save individual model data
    if i==0:
        i1=np.empty(np.insert(np.asarray(ddpvn1.shape),0,len(lmd)))
        i2=np.empty(np.insert(np.asarray(ddpvn2.shape),0,len(lmd)))

    i1[i,...]=ddpvn1
    i2[i,...]=ddpvn2

md='mi'
odir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
if not os.path.exists(odir):
    os.makedirs(odir)

oname='%s/ms.rsq.%s_%s_%s.%s.summer' % (odir,varn,his,fut,se)

jja1=np.nanmean(i1[:,5:8,:],axis=1)
jja2=np.nanmean(i2[:,5:8,:],axis=1)
djf1=np.nanmean(np.roll(i1[:,0:3,:],1,axis=1),axis=1)
djf2=np.nanmean(np.roll(i2[:,0:3,:],1,axis=1),axis=1)

jjar=corr(jja1,jja2,0)
djfr=corr(djf1,djf2,0)

pickle.dump([jjar,djfr],open('%s.pickle'%oname, 'wb'),protocol=5)
