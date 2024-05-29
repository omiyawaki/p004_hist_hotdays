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
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LatitudeFormatter
from scipy.stats import linregress
from tqdm import tqdm
from util import mods
from utils import monname,varnlb,unitlb

nt=7 # window size in days
pref1='ddpc.md'
varn1='tas'
pref2='ddpc.md'
varn2='advty850'
varn='%s%s+%s%s'%(pref1,varn1,pref2,varn2)
plat=30
tlat=30
nhhl=True
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='gwl2.0'
skip5075=True

md='mmm'

idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,'mi',varn)
odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)

if not os.path.exists(odir):
    os.makedirs(odir)

# load land indices
lmi,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

def remap(v,gr):
    llv=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llv[lmi]=v.data
    return np.reshape(llv,(gr['lat'].size,gr['lon'].size))

# load data 
r=xr.open_dataarray('%s/sc.r.%s_%s_%s.%s.nc' % (idir,varn,his,fut,se))

r=np.nanmean(r,0) # MMM

# remap to lat-lon
r=remap(r,gr)

# repeat 0 deg lon info to 360 deg to prevent a blank line in contour
gr['lon'] = np.append(gr['lon'].data,360)
r=np.append(r,r[...,0][...,None],axis=1)

[mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

if nhhl:
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,3),constrained_layout=True)
    # ax.set_title(r'%s %s' % ('MMM',fo.upper()),fontsize=16)
    clf=ax.contourf(mlon, mlat, r, np.arange(-1,1+0.1,0.1), vmax=1, vmin=-1, transform=ccrs.PlateCarree(),cmap='RdBu_r')
    ax.coastlines()
    ax.set_extent((-180,180,plat,90),crs=ccrs.PlateCarree())
    gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.5,color='gray',y_inline=False)
    gl.ylocator=mticker.FixedLocator([])
    gl.yformatter=LatitudeFormatter()
    gl.xlines=False
    gl.left_labels=False
    gl.bottom_labels=False
    gl.right_labels=True
    gl.top_labels=False
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.set_ticks(np.arange(-1,1+0.2,0.2))
    cb.set_label(label=r'$R(\Delta\delta %s, \Delta\delta %s)$'%(varnlb(varn1),varnlb(varn2)))
    fig.savefig('%s/sc.r.%s.%s.nhhl.pdf' % (odir,varn,fo), format='pdf', dpi=600)
    fig.savefig('%s/sc.r.%s.%s.nhhl.png' % (odir,varn,fo), format='png', dpi=600)

# plot r (pct warming - mean warming)
fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,3),constrained_layout=True)
# ax.set_title(r'%s %s' % ('MMM',fo.upper()),fontsize=16)
clf=ax.contourf(mlon, mlat, r, np.arange(-1,1+0.1,0.1), vmax=1, vmin=-1, transform=ccrs.PlateCarree(),cmap='RdBu_r')
ax.coastlines()
cb=fig.colorbar(clf,location='bottom',aspect=50)
cb.set_ticks(np.arange(-1,1+0.2,0.2))
cb.set_label(label=r'$R(\Delta\delta %s, \Delta\delta %s)$'%(varnlb(varn1),varnlb(varn2)))
fig.savefig('%s/sc.r.%s.%s.pdf' % (odir,varn,fo), format='pdf', dpi=600)
fig.savefig('%s/sc.r.%s.%s.png' % (odir,varn,fo), format='png', dpi=600)
