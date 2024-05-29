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
import constants as c
from cartopy.mpl.ticker import LatitudeFormatter
from scipy.stats import linregress
from tqdm import tqdm
from util import mods
from utils import monname,varnlb,unitlb
from xspharm import xspharm

lvn=['tas']
vnp= 'tas'
tlat=30 # upper bound for low latitude
plat=30 # lower bound for high latitude
nhhl=True
reverse=True
# lvn=['ooplh','ooplh_fixbc','ooplh_fixmsm','ooplh_rddsm']
# vnp='ooplh'
se = 'ond+amj' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'
dpi=600
skip507599=True

md='mmm'

# load land indices
lmi,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

def cmap(vn):
    vbr=['td_mrsos','ti_pr','ti_ev']
    if vn in vbr:
        return 'BrBG'
    else:
        return 'RdBu_r'

def vmaxdd(vn):
    lvm={   
            'tas':  [12,1],
            }
    return lvm[vn]

def plot(vn):
    vmdd,dvmdd=vmaxdd(vn)
    vnlb=varnlb(vn)
    unlb=unitlb(vn)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    vno=vn
    if '_wm2' in vn:
        vn=vn.replace('_wm2','')

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # mean warming
    mvn=xr.open_dataarray('%s/d.%s_%s_%s.%s.nc' % (idir,vn,his,fut,se))
    lat,lon=mvn['lat'],mvn['lon']

    # spherical smoothing
    # xsp=xspharm(mvn, gridtype='regular')
    # mvn=xsp.exp_taper(mvn,ntrunc=18)
    mvn=mvn.data

    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    lon= np.append(lon.data,360)
    mvn = np.append(mvn, mvn[...,0][...,None],axis=1)

    [mlat,mlon] = np.meshgrid(lat, lon, indexing='ij')

    if nhhl==True:
        # plot pct warming - mean warming
        fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
        clf=ax.contourf(mlon, mlat, mvn, np.arange(-vmdd,vmdd+dvmdd,dvmdd),extend='both', vmax=vmdd, vmin=-vmdd, transform=ccrs.PlateCarree(), cmap=cmap(vn))
        ax.coastlines()
        ax.set_title(r'%s %s OND+AMJ' % (md.upper(),fo.upper()),fontsize=16)
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
        cb.ax.tick_params(labelsize=16)
        cb.set_label(label=r'$\Delta \overline{%s}$ (%s)'%(vnlb,unlb),size=16)
        fig.savefig('%s/ond+amj.m.%s.%s.%s.nhhl.png' % (odir,vn,fo,fut), format='png', dpi=dpi)

        # nhhl area mean warming
        clat=np.cos(np.deg2rad(lat.data)).reshape([len(lat),1])
        nanma=np.ones_like(mvn.data)
        nanma[np.isnan(mvn)]=0
        nanma[lat<plat]=0
        anhhl=np.sum(mvn*nanma*clat,axis=(0,1))/np.sum(nanma*clat,axis=(0,1))

        # zonal anomaly
        fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
        clf=ax.contourf(mlon, mlat, mvn-anhhl, np.arange(-8,8+0.5,0.5),extend='both', vmax=8, vmin=-8, transform=ccrs.PlateCarree(), cmap=cmap(vn))
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
        ax.set_title(r'%s %s OND+AMJ' % (md.upper(),fo.upper()),fontsize=16)
        cb=fig.colorbar(clf,location='bottom',aspect=50)
        cb.ax.tick_params(labelsize=16)
        cb.set_label(label=r'Spatial anomaly of $\Delta \overline{%s}$ (%s)'%(vnlb,unlb),size=16)
        fig.savefig('%s/ond+amj.m.zonanom.%s.%s.%s.nhhl.png' % (odir,vn,fo,fut), format='png', dpi=dpi)


    # plot pct warming - mean warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, mvn, np.arange(-vmdd,vmdd+dvmdd,dvmdd),extend='both', vmax=vmdd, vmin=-vmdd, transform=ccrs.PlateCarree(), cmap=cmap(vn))
    ax.coastlines()
    ax.set_title(r'%s %s OND+AMJ' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\Delta \overline{%s}$ (%s)'%(vnlb,unlb),size=16)
    fig.savefig('%s/ond+amj.m.%s.%s.%s.png' % (odir,vn,fo,fut), format='png', dpi=dpi)

    # zonal anomaly
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, mvn-np.nanmean(mvn,axis=1,keepdims=True), np.arange(-4,4+0.5,0.5),extend='both', vmax=4, vmin=-4, transform=ccrs.PlateCarree(), cmap=cmap(vn))
    ax.coastlines()
    ax.set_title(r'%s %s OND+AMJ' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'Zonal anomaly of $\Delta \overline{%s}$ (%s)'%(vnlb,unlb),size=16)
    fig.savefig('%s/ond+amj.m.zonanom.%s.%s.%s.png' % (odir,vn,fo,fut), format='png', dpi=dpi)


# run
[plot(vn) for vn in lvn]
