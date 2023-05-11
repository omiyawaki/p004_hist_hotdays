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
from cmip6util import mods
from utils import monname

nt=7 # window size in days
varn='tas'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'

lmd=mods(fo1)

for i,md in enumerate(tqdm(lmd)):
    idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn)
    idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn)

    c = 0
    dt={}

    # mean temp
    ds1=xr.open_dataset('%s/m%s_%s.%s.nc' % (idir1,varn,his,se))
    gr={}
    gr['lat']=ds1['lat']
    gr['lon']=ds1['lon']
    try:
        vn1=ds1[varn]
    except:
        vn1=ds1['__xarray_dataarray_variable__']
    vn1=vn1.groupby('time.month').mean('time') # monthly means
    ds2=xr.open_dataset('%s/m%s_%s.%s.nc' % (idir2,varn,fut,se))
    try:
        vn2=ds2[varn]
    except:
        vn2=ds2['__xarray_dataarray_variable__']
    vn2=vn2.groupby('time.month').mean('time') # monthly means

    # prc temp
    ds1=xr.open_dataset('%s/wp%s%03d_%s.%s.nc' % (idir1,varn,nt,his,se))
    pct=ds1['percentile']
    gr['pct']=pct
    try:
        pvn1=ds1[varn]
    except:
        pvn1=ds1['__xarray_dataarray_variable__']
    pvn1=pvn1.groupby('time.month').mean('time') # monthly means
    ds2=xr.open_dataset('%s/wp%s%03d_%s.%s.nc' % (idir2,varn,nt,fut,se))
    try:
        pvn2=ds2[varn]
    except:
        pvn2=ds2['__xarray_dataarray_variable__']
    pvn2=pvn2.groupby('time.month').mean('time') # monthly means

    # warming
    dvn=vn2-vn1
    dpvn=pvn2-pvn1
    ddpvn=dpvn-np.transpose(dvn.data[...,None],[0,3,1,2])

    # save individual model data
    if i==0:
        ivn1=np.empty(np.insert(np.asarray(vn1.shape),0,len(lmd)))
        ivn2=np.empty(np.insert(np.asarray(vn2.shape),0,len(lmd)))
        ipvn1=np.empty(np.insert(np.asarray(pvn1.shape),0,len(lmd)))
        ipvn2=np.empty(np.insert(np.asarray(pvn2.shape),0,len(lmd)))
        idvn=np.empty(np.insert(np.asarray(dvn.shape),0,len(lmd)))
        idpvn=np.empty(np.insert(np.asarray(dpvn.shape),0,len(lmd)))
        iddpvn=np.empty(np.insert(np.asarray(ddpvn.shape),0,len(lmd)))

    ivn1[i,...]=vn1
    ivn2[i,...]=vn2
    ipvn1[i,...]=pvn1
    ipvn2[i,...]=pvn2
    idvn[i,...]=dvn
    idpvn[i,...]=dpvn
    iddpvn[i,...]=ddpvn

# compute mmm and std
mvn1=np.nanmean(ivn1,axis=0)
mvn2=np.nanmean(ivn2,axis=0)
mpvn1=np.nanmean(ipvn1,axis=0)
mpvn2=np.nanmean(ipvn2,axis=0)
mdvn=np.nanmean(idvn,axis=0)
mdpvn=np.nanmean(idpvn,axis=0)
mddpvn=np.nanmean(iddpvn,axis=0)

svn1=np.nanstd(ivn1,axis=0)
svn2=np.nanstd(ivn2,axis=0)
spvn1=np.nanstd(ipvn1,axis=0)
spvn2=np.nanstd(ipvn2,axis=0)
sdvn=np.nanstd(idvn,axis=0)
sdpvn=np.nanstd(idpvn,axis=0)
sddpvn=np.nanstd(iddpvn,axis=0)

# save mmm and std
odir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,'mmm',varn)
odir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,'mmm',varn)
odir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mmm',varn)
if not os.path.exists(odir1):
    os.makedirs(odir1)
if not os.path.exists(odir2):
    os.makedirs(odir2)
if not os.path.exists(odir):
    os.makedirs(odir)

pickle.dump([mvn1,svn1,gr], open('%s/m%s_%s.%s.nc' % (odir1,varn,his,se), 'wb'), protocol=5)	
pickle.dump([mvn2,svn2,gr], open('%s/m%s_%s.%s.nc' % (odir2,varn,fut,se), 'wb'), protocol=5)	
pickle.dump([mpvn1,spvn1,gr], open('%s/p%s_%s.%s.nc' % (odir1,varn,his,se), 'wb'), protocol=5)	
pickle.dump([mpvn2,spvn2,gr], open('%s/p%s_%s.%s.nc' % (odir2,varn,fut,se), 'wb'), protocol=5)	
pickle.dump([mdvn,sdvn,gr], open('%s/d%s_%s_%s.%s.nc' % (odir,varn,his,fut,se), 'wb'), protocol=5)	
pickle.dump([mdpvn,sdpvn,gr], open('%s/dp%s_%s_%s.%s.nc' % (odir,varn,his,fut,se), 'wb'), protocol=5)	
pickle.dump([mddpvn,sddpvn,gr], open('%s/ddp%s_%s_%s.%s.nc' % (odir,varn,his,fut,se), 'wb'), protocol=5)	

# save ensemble
odir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,'mi',varn)
odir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,'mi',varn)
odir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mi',varn)
if not os.path.exists(odir1):
    os.makedirs(odir1)
if not os.path.exists(odir2):
    os.makedirs(odir2)
if not os.path.exists(odir):
    os.makedirs(odir)

pickle.dump(ivn1, open('%s/m%s_%s.%s.nc' % (odir1,varn,his,se), 'wb'), protocol=5)	
pickle.dump(ivn2, open('%s/m%s_%s.%s.nc' % (odir2,varn,fut,se), 'wb'), protocol=5)	
pickle.dump(ipvn1, open('%s/p%s_%s.%s.nc' % (odir1,varn,his,se), 'wb'), protocol=5)	
pickle.dump(ipvn2, open('%s/p%s_%s.%s.nc' % (odir2,varn,fut,se), 'wb'), protocol=5)	
pickle.dump(idvn, open('%s/d%s_%s_%s.%s.nc' % (odir,varn,his,fut,se), 'wb'), protocol=5)	
pickle.dump(idpvn, open('%s/dp%s_%s_%s.%s.nc' % (odir,varn,his,fut,se), 'wb'), protocol=5)	
pickle.dump(iddpvn, open('%s/ddp%s_%s_%s.%s.nc' % (odir,varn,his,fut,se), 'wb'), protocol=5)	
