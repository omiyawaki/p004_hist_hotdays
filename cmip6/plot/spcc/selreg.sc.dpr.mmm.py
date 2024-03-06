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
from utils import monname,filllongap,varnlb,unitlb
from regions import window,retname,refigsize

lrelb=['ic']

varn='pr'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fod='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'

md='mmm'

def set_cticks():
    cticks=np.arange(0,16,1)
    return cticks

# load land indices
lmi,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

def plot(varn,relb,foi=fo1,foo=fo1,yr=his):
    # grid
    rgdir='/project/amp/miyawaki/data/share/regrid'
    # open CESM data to get output grid
    cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
    cdat=xr.open_dataset(cfil)
    gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,foi,md,varn)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/regions' % (se,foo,md,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # warming
    ds=xr.open_dataset('%s/cm.%s_%s.%s.nc' % (idir,varn,his,se))
    cm=ds[varn]
    if varn=='pr':
        cm=86400*cm
    gpi=ds['gpi']

    # remap to lat x lon
    llcm=np.nan*np.ones([cm.shape[0],gr['lat'].size*gr['lon'].size])
    llcm[:,lmi]=cm.data
    llcm=np.reshape(llcm,(cm.shape[0],gr['lat'].size,gr['lon'].size))

    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    llcm,lon = filllongap(llcm,gr['lon'],axis=2)

    [mlat,mlon] = np.meshgrid(gr['lat'], lon, indexing='ij')

    cticks=set_cticks()
    # plot pct warming - mean warming
    fig,ax=plt.subplots(nrows=3,ncols=4,subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=refigsize(relb),constrained_layout=True)
    ax=ax.flatten()
    fig.suptitle(r'%s %s %s' % (md.upper(),foo.upper(),retn),fontsize=16)
    for m in tqdm(range(12)):
        ax[m].set_extent(rell,crs=ccrs.PlateCarree())
        clf=ax[m].contourf(mlon, mlat, llcm[m,...], cticks,extend='both', vmax=cticks[-1], vmin=cticks[0], transform=ccrs.PlateCarree(), cmap='BrBG')
        ax[m].coastlines()
        gl=ax[m].gridlines(draw_labels=True,alpha=0.2)
        if not m in [0,1,2,3]:
            gl.xlabels_top=False
        if not m in [0,4,8]:
            gl.ylabels_left=False
        if not m in [3,7,11]:
            gl.ylabels_right=False
        if not m in [8,9,10,11]:
            gl.xlabels_bottom=False
        ax[m].set_title(r'%s' % (monname(m)),fontsize=16)
        fig.savefig('%s/cm.%s.%s.%s.png' % (odir,varn,foo,relb), format='png', dpi=300)
    cb=fig.colorbar(clf,ax=ax.ravel().tolist(),location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'%s (%s)'%(varnlb(varn),unitlb(varn)),size=16)
    fig.savefig('%s/cm.%s.%s.%s.png' % (odir,varn,foo,relb), format='png', dpi=300)

for relb in lrelb:
    retn=retname(relb)
    rell=window(relb)
    plot(varn,relb,foi=fo1,foo=fo1,yr=his)
