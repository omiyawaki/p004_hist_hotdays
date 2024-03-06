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
from utils import monname,varnlb,unitlb
from regions import window,retname,refigsize

relb='ic'

retn=retname(relb)
rell=window(relb)
nt=7 # window size in days
mon=6 # month
p=95
varn='tas'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'
skip507599=True

lmd=mods(fo1)

# load land indices
lmi,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/regions' % (se,fo,'mi',varn)

if not os.path.exists(odir):
    os.makedirs(odir)

fig,ax=plt.subplots(nrows=4,ncols=4,subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=refigsize(relb),constrained_layout=True)
ax=ax.flatten()
fig.suptitle(r'%s %s %s' % (fo.upper(),retn,monname(mon-1)),fontsize=16)

for m,md in enumerate(tqdm(lmd)):
    idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo1,md,varn)
    idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo2,md,varn)

    # warming
    ds=xr.open_dataset('%s/pc.%s_%s.%s.nc' % (idir1,varn,his,se))
    vn1=ds[varn]
    pct=ds['percentile']
    gpi=ds['gpi']
    vn2=xr.open_dataarray('%s/pc.%s_%s.%s.nc' % (idir2,varn,fut,se))
    vn1=vn1.sel(month=mon)
    vn2=vn2.sel(month=mon)
    dvn=vn1.copy()
    dvn.data=vn2.data-vn1.data
    ddp=dvn.sel(percentile=pct==p).squeeze()-dvn.sel(percentile=pct==0).squeeze()


    # remap to lat x lon
    llddp=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llddp[lmi]=ddp.data
    llddp=np.reshape(llddp,(gr['lat'].size,gr['lon'].size))

    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    lon = np.append(gr['lon'].data,360)
    llddp = np.append(llddp, llddp[...,0][...,None],axis=1)

    [mlat,mlon] = np.meshgrid(gr['lat'], lon, indexing='ij')

    # plot pct warming - mean warming
    ax[m].set_extent(rell,crs=ccrs.PlateCarree())
    if md in ['CESM2']:
        clf=ax[m].contourf(mlon, mlat, llddp, np.arange(-1.5,1.5+0.101,0.101),extend='both', vmax=1.5, vmin=-1.5, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    else:
        clf=ax[m].contourf(mlon, mlat, llddp, np.arange(-1.5,1.5+0.1,0.1),extend='both', vmax=1.5, vmin=-1.5, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    ax[m].coastlines()
    gl=ax[m].gridlines(draw_labels=True,alpha=0.2)
    if not m in [0,1,2,3]:
        gl.xlabels_top=False
    if not m in [0,4,8,12]:
        gl.ylabels_left=False
    if not m in [3,7,11,15]:
        gl.ylabels_right=False
    if not m in [12,13,14,15]:
        gl.xlabels_bottom=False
    ax[m].set_title(r'%s' % (md.upper()),fontsize=16)
    if m==0:
        cb=fig.colorbar(clf,ax=ax.ravel().tolist(),location='bottom',aspect=50)
        cb.ax.tick_params(labelsize=16)
        cb.set_label(label=r'$\Delta \delta$ %s (%s)'%(varnlb(varn),unitlb(varn)),size=16)
    fig.savefig('%s/ddp%02d%s.%s.%s.%02d.png' % (odir,p,varn,fo,relb,mon), format='png', dpi=300)
