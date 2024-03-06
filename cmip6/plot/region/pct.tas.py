import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import pickle
import pwlf
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from tqdm import tqdm
from util import mods
from utils import corr,corr2d,monname
from regions import pointlocs
from CASutils import shapefile_utils as shp

relb='fourcorners'
re=['Utah','Colorado','Arizona','New Mexico']

pct=np.linspace(1,99,101)
varn='tas'
ise='sc'
ose='jja'
lmo=[6,7,8]

fo1='historical' # forcings 
yr1='1980-2000'

fo2='ssp370' # forcings 
yr2='2080-2100'

fo='%s+%s'%(fo1,fo2)
fod='%s-%s'%(fo2,fo1)

md1='CanESM5'
md2='MPI-ESM1-2-LR'
md='%s+%s'%(md1,md2)

# load ocean indices
_,omi=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

# mask
shpfile = "/project/cas/islas/shapefiles/usa/gadm36_USA_1.shp"
mask=shp.maskgen(shpfile,gr,re).data
mask=mask.flatten()
mask=np.delete(mask,omi)
print(mask.shape)

idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (ise,fo1,md1,varn)
idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (ise,fo2,md1,varn)
idir3 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (ise,fo1,md2,varn)
idir4 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (ise,fo2,md2,varn)

odir1= '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/selpoint/%s' % (ose,fo1,md,varn,relb)
if not os.path.exists(odir1):
    os.makedirs(odir1)
odir2= '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/selpoint/%s' % (ose,fo2,md,varn,relb)
if not os.path.exists(odir2):
    os.makedirs(odir2)
odir3= '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/selpoint/%s' % (ose,fo,md,varn,relb)
if not os.path.exists(odir3):
    os.makedirs(odir3)
odir4= '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/selpoint/%s' % (ose,fod,md,varn,relb)
if not os.path.exists(odir4):
    os.makedirs(odir4)

ds=xr.open_dataset('%s/lm.%s_%s.%s.nc' % (idir1,varn,yr1,ise))
vn1=ds[varn]
gpi=ds['gpi']
vn2=xr.open_dataset('%s/lm.%s_%s.%s.nc' % (idir2,varn,yr2,ise))[varn]
vn3=xr.open_dataset('%s/lm.%s_%s.%s.nc' % (idir3,varn,yr1,ise))[varn]
vn4=xr.open_dataset('%s/lm.%s_%s.%s.nc' % (idir4,varn,yr2,ise))[varn]

# extract data for season
vn1=vn1.sel(time=vn1['time.season']==ose.upper())
vn2=vn2.sel(time=vn2['time.season']==ose.upper())
vn3=vn3.sel(time=vn3['time.season']==ose.upper())
vn4=vn4.sel(time=vn4['time.season']==ose.upper())

# delete data outside masked region
vn1=np.delete(vn1.data,np.nonzero(np.isnan(mask)),axis=1)
vn1=vn1.flatten()
vn2=np.delete(vn2.data,np.nonzero(np.isnan(mask)),axis=1)
vn2=vn2.flatten()
vn3=np.delete(vn3.data,np.nonzero(np.isnan(mask)),axis=1)
vn3=vn3.flatten()
vn4=np.delete(vn4.data,np.nonzero(np.isnan(mask)),axis=1)
vn4=vn4.flatten()

# compute percentiles
pvn1=np.percentile(vn1,pct)
pvn2=np.percentile(vn2,pct)
pvn3=np.percentile(vn3,pct)
pvn4=np.percentile(vn4,pct)

tname=r'%s' % (relb)
oname1='%s/p.%s_%s_%s.%s' % (odir4,varn,yr1,yr2,ose)
fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
clf=ax.plot(pct,pvn2-pvn1,color='k',label=md1)
clf=ax.plot(pct,pvn4-pvn3,':',color='k',label=md2)
ax.set_title(r'%s' % (tname),fontsize=16)
ax.set_xlabel('$T$ Percentile')
ax.set_ylabel('$\Delta T$ (K)')
ax.legend(frameon=False)
fig.savefig('%s.png'%oname1, format='png', dpi=600)

tname=r'%s' % (relb)
oname1='%s/p.%s_%s_%s.%s' % (odir3,varn,yr1,yr2,ose)
fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
clf=ax.plot(pct,pvn1,color='tab:blue',label=md1)
clf=ax.plot(pct,pvn3,':',color='tab:blue',label=md2)
clf=ax.plot(pct,pvn2,color='tab:orange')
clf=ax.plot(pct,pvn4,':',color='tab:orange')
ax.set_title(r'%s' % (tname),fontsize=16)
ax.set_xlabel('Percentile')
ax.set_ylabel('$T$ (K)')
ax.legend(frameon=False)
fig.savefig('%s.png'%oname1, format='png', dpi=600)

tname=r'%s' % (relb)
oname1='%s/p.%s_%s.%s' % (odir1,varn,yr1,ose)
fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
clf=ax.plot(pct,pvn1,color='tab:blue',label=md1)
clf=ax.plot(pct,pvn3,':',color='tab:blue',label=md2)
ax.set_title(r'%s' % (tname),fontsize=16)
ax.set_xlabel('Percentile')
ax.set_ylabel('$T$ (K)')
ax.legend(frameon=False)
fig.savefig('%s.png'%oname1, format='png', dpi=600)

tname=r'%s' % (relb)
oname1='%s/p.%s_%s.%s' % (odir2,varn,yr2,ose)
fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
clf=ax.plot(pct,pvn2,color='tab:orange',label=md1)
clf=ax.plot(pct,pvn4,':',color='tab:orange',label=md2)
ax.set_title(r'%s' % (tname),fontsize=16)
ax.set_xlabel('Percentile')
ax.set_ylabel('$T$ (K)')
ax.legend(frameon=False)
fig.savefig('%s.png'%oname1, format='png', dpi=600)

