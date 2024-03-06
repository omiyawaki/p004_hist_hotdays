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

pref1='d'
varn1='qsoil'
pref2='d'
ld=np.concatenate(([10],np.arange(20,100,20),np.arange(100,850,50)))
lvarn2=['ooplh%g'%d for d in ld]

se='sc'
sc='jja'
re='tr'
tlat=30
plat=50
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='gwl2.0'

def seasmean(vn,sc,lat):
    if sc=='jja':
        vnn=np.nanmean(vn[5:8,...],axis=0)
        vns=np.nanmean(np.roll(vn,1,axis=0)[:3,...],axis=0)
    vn=np.copy(vns)
    vn[:,lat>0]=vnn[:,lat>0]
    return vn

def regsl(vn,re,lat):
    if re=='tr':
        vn=np.delete(vn,np.abs(lat)>tlat,axis=1)
    elif re=='ml':
        vn=np.delete(vn,np.logical_or(np.abs(lat)<=tlat,np.abs(lat)>plat),axis=1)
    elif re=='hl':
        vn=np.delete(vn,np.abs(lat)<=plat,axis=1)
    elif re=='et':
        vn=np.delete(vn,np.abs(lat)<=tlat,axis=1)
    return vn

# load lmi lat lon 
lat,lon=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lmilatlon.pickle','rb'))

def calc_corr(varn2):
    varn='%s%s+%s%s'%(pref1,varn1,pref2,varn2)

    # load data to correlate
    md='CESM2'
    idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn1)
    idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn2)
    odir =  '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    i1=xr.open_dataarray('%s/%s.%s_%s_%s.%s.nc' % (idir1,pref1,varn1,his,fut,se))
    i2=xr.open_dataarray('%s/%s.%s_%s_%s.%s.nc' % (idir2,pref2,varn2,his,fut,se))
    ldim=len(i1.shape)
    pct=i1['percentile'].data if ldim==3 else 'mean'
    i1=i1.data
    i2=i2.data
    i1=i1[:,None,:] if ldim==2 else i1
    i2=i2[:,None,:] if ldim==2 else i2

    # take seasonal mean
    i1=seasmean(i1,sc,lat)
    i2=seasmean(i2,sc,lat)

    # regionally constrain if applicable
    i1=regsl(i1,re,lat)
    i2=regsl(i2,re,lat)

    r=corr(i1,i2,1)
    r=xr.DataArray(r,coords={'pct':pct})
    r.to_netcdf('%s/sp.rsq.%s_%s_%s.%s.%s.nc' % (odir,varn,his,fut,sc,re))

[calc_corr(varn2) for varn2 in tqdm(lvarn2)]
