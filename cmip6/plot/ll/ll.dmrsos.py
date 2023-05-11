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

nt=30 # lead-lag window size in days
p=95
varn='mrsos'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'
lsl=[np.arange(0,30),np.arange(15,30),np.arange(23,30),[29],[30],[31],np.arange(31,38),np.arange(31,45),np.arange(31,61)]
llb=['-1 month','-2 weeks','-1 week','-1 day','0 day','+1 day','+1 week','+2 weeks','+1 month']

md='mmm'

# load land indices
lmi,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo1,md,varn)
idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo2,md,varn)
idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
odir1 = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo1,md,varn)
if not os.path.exists(odir1):
    os.makedirs(odir1)
odir2 = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo2,md,varn)
if not os.path.exists(odir2):
    os.makedirs(odir2)
odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
if not os.path.exists(odir):
    os.makedirs(odir)

# ll data
ds=xr.open_dataset('%s/ll%g.d%g.%s_%s_%s.%s.nc' % (idir,nt,p,varn,his,fut,se))
ll1=ds[varn]
lday=ds['lday']
gpi=ds['gpi']
nl,nm,ng=ll1.shape

# remap to lat x lon
llll1=np.nan*np.ones([nl,nm,gr['lat'].size*gr['lon'].size])
llll1[:,:,lmi]=ll1.data
llll1=np.reshape(llll1,(nl,nm,gr['lat'].size,gr['lon'].size))

# repeat 0 deg lon info to 360 deg to prevent a blank line in contour
gr['lon'] = np.append(gr['lon'].data,360)
llll1 = np.append(llll1, llll1[...,0][...,None],axis=3)

# take averages of ll
lmean1=[np.nanmean(llll1[sl,...],axis=0) for sl in lsl]

[mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

# plot lead-lag in historical climate
for m in tqdm(range(12)):
    fig,ax=plt.subplots(nrows=3,ncols=3,subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(9,7),constrained_layout=True)
    ax=ax.flatten()
    fig.suptitle(r'%s %s %s' % (md.upper(),fo.upper(),monname(m)),fontsize=16)
    for i,ll in enumerate(tqdm(lmean1)):
        clf=ax[i].contourf(mlon, mlat, ll[m,...], np.arange(-1.5,1.5+0.1,0.1),extend='both', vmax=1.5, vmin=-1.5, transform=ccrs.PlateCarree(), cmap='RdBu_r')
        ax[i].coastlines()
        ax[i].set_title(r'%s' % (llb[i]),fontsize=16)
        fig.savefig('%s/ds.%02d%s.%s.%02d.pdf' % (odir,p,varn,fo,m+1), format='pdf', dpi=300)
    cb=fig.colorbar(clf,ax=ax.ravel().tolist(),location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\Delta SM$ anomaly composited on $T^{%02d}$ day  (kg m$^{-2}$)'%(p),size=16)
    fig.savefig('%s/ds.%02d%s.%s.%02d.pdf' % (odir,p,varn,fo,m+1), format='pdf', dpi=300)
