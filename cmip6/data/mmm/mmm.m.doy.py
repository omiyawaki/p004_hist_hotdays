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
from scipy.interpolate import interp1d
from tqdm import tqdm
from cmip6util import mods
from utils import monname

nt=7 # window size in days
varn='hfls'
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
    vn1=vn1.groupby('time.dayofyear').mean('time') # doy means
    ds2=xr.open_dataset('%s/m%s_%s.%s.nc' % (idir2,varn,fut,se))
    try:
        vn2=ds2[varn]
    except:
        vn2=ds2['__xarray_dataarray_variable__']
    vn2=vn2.groupby('time.dayofyear').mean('time') # doy means

    # warming
    dvn=vn2-vn1

    # save individual model data
    if i==0:
        ivn1=np.empty(np.insert(np.asarray(vn1.shape),0,len(lmd)))
        ivn2=np.empty(np.insert(np.asarray(vn2.shape),0,len(lmd)))
        idvn=np.empty(np.insert(np.asarray(dvn.shape),0,len(lmd)))

    try:
        ivn1[i,...]=vn1
        ivn2[i,...]=vn2
        idvn[i,...]=dvn
    except: # interpolate
        doy0=np.arange(ivn1.shape[1])
        doy=np.arange(vn1.shape[0])
        print('Interpolating %g doy to %g'%(len(doy),len(doy0)))
        fint=interp1d(doy,vn1,axis=0,bounds_error=False,fill_value='extrapolate')
        vn1=fint(doy0)
        fint=interp1d(doy,vn2,axis=0,bounds_error=False,fill_value='extrapolate')
        vn2=fint(doy0)
        fint=interp1d(doy,dvn,axis=0,bounds_error=False,fill_value='extrapolate')
        dvn=fint(doy0)
        ivn1[i,...]=vn1
        ivn2[i,...]=vn2
        idvn[i,...]=dvn

# compute mmm and std
mvn1=np.nanmean(ivn1,axis=0)
mvn2=np.nanmean(ivn2,axis=0)
mdvn=np.nanmean(idvn,axis=0)

svn1=np.nanstd(ivn1,axis=0)
svn2=np.nanstd(ivn2,axis=0)
sdvn=np.nanstd(idvn,axis=0)

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

pickle.dump([mvn1,svn1,gr], open('%s/m%s_%s.%s.doy.pickle' % (odir1,varn,his,se), 'wb'), protocol=5)	
pickle.dump([mvn2,svn2,gr], open('%s/m%s_%s.%s.doy.pickle' % (odir2,varn,fut,se), 'wb'), protocol=5)	
pickle.dump([mdvn,sdvn,gr], open('%s/d%s_%s_%s.%s.doy.pickle' % (odir,varn,his,fut,se), 'wb'), protocol=5)	

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

pickle.dump(ivn1, open('%s/m%s_%s.%s.doy.pickle' % (odir1,varn,his,se), 'wb'), protocol=5)	
pickle.dump(ivn2, open('%s/m%s_%s.%s.doy.pickle' % (odir2,varn,fut,se), 'wb'), protocol=5)	
pickle.dump(idvn, open('%s/d%s_%s_%s.%s.doy.pickle' % (odir,varn,his,fut,se), 'wb'), protocol=5)	
