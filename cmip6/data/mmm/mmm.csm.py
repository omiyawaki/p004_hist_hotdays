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
from utils import monname

nt=7 # window size in days
varn='csm'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'

lmd=mods(fo1)

for i,md in enumerate(tqdm(lmd)):
    idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo1,md,varn)
    idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo2,md,varn)

    c = 0
    dt={}

    # load csm
    ds1=xr.open_dataset('%s/%s.%s.%s.nc' % (idir1,varn,his,se))
    try:
        vn1=ds1[varn]
    except:
        vn1=ds1['__xarray_dataarray_variable__']
    ds2=xr.open_dataset('%s/%s.%s.%s.nc' % (idir2,varn,fut,se))
    try:
        vn2=ds2[varn]
    except:
        vn2=ds2['__xarray_dataarray_variable__']

    # warming
    dvn=vn2-vn1

    # save individual model data
    if i==0:
        ivn1=np.empty(np.insert(np.asarray(vn1.shape),0,len(lmd)))
        ivn2=np.empty(np.insert(np.asarray(vn2.shape),0,len(lmd)))
        idvn=np.empty(np.insert(np.asarray(dvn.shape),0,len(lmd)))

    ivn1[i,...]=vn1
    ivn2[i,...]=vn2
    idvn[i,...]=dvn

# replace infs with nans
ivn1[np.logical_or(ivn1==np.inf,ivn1==-np.inf)]=np.nan
ivn2[np.logical_or(ivn2==np.inf,ivn2==-np.inf)]=np.nan
idvn[np.logical_or(idvn==np.inf,idvn==-np.inf)]=np.nan

# compute mmm and std
mvn1=vn1.copy()
mvn2=vn2.copy()
mdvn=mvn2.copy()
mvn1.data=np.nanmean(ivn1,axis=0)
mvn2.data=np.nanmean(ivn2,axis=0)
mdvn.data=np.nanmean(idvn,axis=0)

svn1=mvn1.copy()
svn2=mvn2.copy()
sdvn=mdvn.copy()
svn1.data=np.nanstd(ivn1,axis=0)
svn2.data=np.nanstd(ivn2,axis=0)
sdvn.data=np.nanstd(idvn,axis=0)

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

mvn1.to_netcdf('%s/%s.%s.%s.nc' % (odir1,varn,his,se))	
mvn2.to_netcdf('%s/%s.%s.%s.nc' % (odir2,varn,fut,se))	
mdvn.to_netcdf('%s/d.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))	

