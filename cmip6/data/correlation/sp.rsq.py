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
from utils import corr2d

nt=7 # window size in days
pref1='ddp'
varn1='tas'
pref2='d'
varn2='pr'
varn='%s%s+%s%s'%(pref1,varn1,pref2,varn2)
se = 'sc'
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

md='mi'
idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn2)
odir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
if not os.path.exists(odir):
    os.makedirs(odir)

i1=pickle.load(open('%s/%s%s_%s_%s.%s.nc' % (idir1,pref1,varn1,his,fut,se), 'rb'))
i2=pickle.load(open('%s/%s%s_%s_%s.%s.nc' % (idir2,pref2,varn2,his,fut,se), 'rb'))

print(i1.shape)
print(i2.shape)
sys.exit()

if i2.shape != i1.shape:
    i2=np.transpose(i2[...,None],[0,1,4,2,3])

r=corr2d(i1,i2,gr,(3,4))
print(r.shape)
sys.exit()
pickle.dump(r,open('%s/sq.rsq.%s_%s_%s.%s.nc' % (odir,varn,his,fut,se), 'wb'),protocol=5)
