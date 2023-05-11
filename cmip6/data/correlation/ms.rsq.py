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
from utils import corr

nt=7 # window size in days
pref1='ddp'
varn1='tas'
pref2='ddp'
varn2='hfss'
varn='%s%s+%s%s'%(pref1,varn1,pref2,varn2)
se = 'sc'
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'
ann=True

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

md='mi'
idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn2)
odir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
if not os.path.exists(odir):
    os.makedirs(odir)

# i1=pickle.load(open('%s/%s%s_%s_%s.%s.doy.pickle' % (idir1,pref1,varn1,his,fut,se), 'rb'))
# i2=pickle.load(open('%s/%s%s_%s.%02d.%s.doy.pickle' % (idir2,pref2,varn2,his,p,se), 'rb'))
i1=pickle.load(open('%s/%s%s_%s_%s.%s.pickle' % (idir1,pref1,varn1,his,fut,se), 'rb'))
i2=pickle.load(open('%s/%s%s_%s_%s.%s.pickle' % (idir2,pref2,varn2,his,fut,se), 'rb'))
# i2=pickle.load(open('%s/%s%s_%s.%02d.%s.pickle' % (idir2,pref2,varn2,his,p,se), 'rb'))

oname='%s/ms.rsq.%s_%s_%s.%s' % (odir,varn,his,fut,se)
if ann:
    oname='%s.ann'%oname
    i1=np.nanmean(i1,axis=1)
    i2=np.nanmean(i2,axis=1)

r=corr(i1,i2,0)

print(r.shape)

pickle.dump(r,open('%s.pickle'%oname, 'wb'),protocol=5)
