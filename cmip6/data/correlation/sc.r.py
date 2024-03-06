import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
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
p=97.5
pref1='ddpc'
varn1='tas'
pref2='ddpc'
varn2='gflx'
varn='%s%s+%s%s'%(pref1,varn1,pref2,varn2)
se = 'sc'
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='gwl2.0'

lmd=mods(fo1)

for i,md in enumerate(tqdm(lmd)):
    idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn1)
    idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn2)
    odir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    i1=xr.open_dataarray('%s/%s.%s_%s_%s.%s.nc' % (idir1,pref1,varn1,his,fut,se)).sel(percentile=p).squeeze()
    i2=xr.open_dataarray('%s/%s.%s_%s_%s.%s.nc' % (idir2,pref2,varn2,his,fut,se)).sel(percentile=p).squeeze()

    r=corr(i1,i2,0)

    # store all models
    if i==0:
        re=np.empty([len(lmd),len(r)])
    re[i,:]=r

    # save correlation for individual model
    gpi=i1['gpi']
    r=xr.DataArray(r,coords={'gpi':gpi},dims=('gpi'))
    r.to_netcdf('%s/sc.r.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))

# save correlation for all models
re=xr.DataArray(re,coords={'models':range(len(lmd)),'gpi':gpi},dims=('models','gpi'))
odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,'mi',varn)
if not os.path.exists(odir):
    os.makedirs(odir)
re.to_netcdf('%s/sc.r.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se))
