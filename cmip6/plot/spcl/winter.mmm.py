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

# lp=[2.5]
# p=lp[-1]

lp=[97.5]
p=lp[0]

# lp=[52.5,57.5,62.5,67.5,72.5,77.5,82.5,87.5,92.5,97.5]
# p=lp[0]

nhmon=[12,1,2]
shmon=[6,7,8]
lvn=['hfss']
vnp= 'hfss'
tlat=30
plat=30
nhhl=True
tropics=False
reverse=True
# lvn=['ooplh','ooplh_fixbc','ooplh_fixmsm','ooplh_rsm']
# vnp='ooplh'
se = 'sc' # season (ann, djf, mam, jja, son)

fo='historical' # forcings 
yr='1980-2000'

# fo='ssp370' # forcings 
# yr='gwl2.0'

dpi=600
skip507599=True

md='CESM2'

# load land indices
lmi,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

def loadmvn(fo,yr,vn):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    ds=xr.open_dataset('%s/m.%s_%s.%s.nc' % (idir,vn,yr,se))
    gpi=ds['gpi']
    try:
        pvn=ds[vn]
    except:
        pvn=ds['__xarray_dataarray_variable__']
        try:
            pvn=ds[varn[2:5]]
        except:
            sys.exit()
    return pvn,gpi

def loadpvn(fo,yr,vn):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    ds=xr.open_dataset('%s/pc.%s_%s.%s.nc' % (idir,vn,yr,se))
    pct=ds['percentile']
    gpi=ds['gpi']
    try:
        pvn=ds[vn]
    except:
        pvn=ds['__xarray_dataarray_variable__']
        try:
            pvn=ds[varn[2:5]]
        except:
            sys.exit()
    return pvn,pct,gpi

def cmap(vn):
    vbr=['td_mrsos','ti_pr','ti_ev']
    if vn in vbr:
        return 'BrBG'
    elif vn=='snc':
        return 'RdBu'
    else:
        return 'RdBu_r'

def vmax(vn):
    lvm={   
            'tas':  [30,5],
            'snc':  [1,0.1],
            'fsm':  [10,1],
            'hfss':  [20,2],
            'ta850':  [1,0.1],
            'wap850':  [50,5],
            'va850':  [20,2],
            }
    return lvm[vn]

def vmaxd(vn):
    lvm={   
            'tas':  [10,1],
            'snc':  [1,0.1],
            'fsm':  [10,1],
            'hfss':  [20,2],
            'ta850':  [4,0.25],
            'wap850':  [50,5],
            'va850':  [10,0.1],
            }
    return lvm[vn]

def plot(vn):
    vm,dvm=vmax(vn)
    vmd,dvmd=vmaxd(vn)
    vnlb,unlb=varnlb(vn),unitlb(vn)
    if 'tas' in vn:
        unlb='$^\circ$C'

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # climo 
    mvn=xr.open_dataarray('%s/md.%s_%s.%s.nc' % (idir,vn,yr,se))
    pvn=xr.open_dataarray('%s/pc.%s_%s.%s.nc' % (idir,vn,yr,se))
    pct=pvn['percentile']
    gpi=pvn['gpi']
    pvn=pvn.sel(percentile=pct.isin(lp)).mean('percentile')
    if reverse and (vn in ['fsm','gflx','hfss','hfls','fat850','fa850','advt850','advtx850','advty850','advm850','advmx850','advmy850','rfa'] or 'ooplh' in vn):
        mvn=-mvn
        pvn=-pvn
    if 'wap' in vn:
        mvn=mvn*86400/100
        pvn=pvn*86400/100
    if 'tas' in vn:
        mvn=mvn-273.15
        pvn=pvn-273.15

    # ndj and mjj means
    mvnnh=mvn.sel(month=mvn['month'].isin(nhmon)).mean('month')
    mvnsh=mvn.sel(month=mvn['month'].isin(shmon)).mean('month')
    pvnnh=pvn.sel(month=pvn['month'].isin(nhmon)).mean('month')
    pvnsh=pvn.sel(month=pvn['month'].isin(shmon)).mean('month')

    # remap to lat x lon
    llmvnnh=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llmvnnh[lmi]=mvnnh.data
    llmvnnh=np.reshape(llmvnnh,(gr['lat'].size,gr['lon'].size))
    llpvnnh=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llpvnnh[lmi]=pvnnh.data
    llpvnnh=np.reshape(llpvnnh,(gr['lat'].size,gr['lon'].size))

    llmvnsh=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llmvnsh[lmi]=mvnsh.data
    llmvnsh=np.reshape(llmvnsh,(gr['lat'].size,gr['lon'].size))
    llpvnsh=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llpvnsh[lmi]=pvnsh.data
    llpvnsh=np.reshape(llpvnsh,(gr['lat'].size,gr['lon'].size))

    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    llmvnnh = np.append(llmvnnh, llmvnnh[...,0][...,None],axis=1)
    llmvnsh = np.append(llmvnsh, llmvnsh[...,0][...,None],axis=1)
    llpvnnh = np.append(llpvnnh, llpvnnh[...,0][...,None],axis=1)
    llpvnsh = np.append(llpvnsh, llpvnsh[...,0][...,None],axis=1)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # use djf for nh, jja for sh
    llmvn=np.copy(llmvnsh)
    llmvn[gr['lat']>0]=llmvnnh[gr['lat']>0]
    llpvn=np.copy(llpvnsh)
    llpvn[gr['lat']>0]=llpvnnh[gr['lat']>0]

    if tropics:
        # plot TROPICS ONLY
        fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
        # ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
        ax.set_title(r'%s %s DJF+JJA' % (md.upper(),fo.upper()),fontsize=16)
        clf=ax.contourf(mlon, mlat, llmvn, np.arange(-vm,vm+dvm,dvm),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(),cmap=cmap(vn))
        ax.coastlines()
        ax.set_extent((-180,180,-tlat,tlat),crs=ccrs.PlateCarree())
        gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.5,color='gray',y_inline=False)
        gl.ylocator=mticker.FixedLocator([])
        gl.yformatter=LatitudeFormatter()
        gl.xlines=False
        gl.left_labels=False
        gl.bottom_labels=False
        gl.right_labels=True
        gl.top_labels=False
        cb=fig.colorbar(clf,location='bottom',aspect=50)
        cb.ax.tick_params(labelsize=12)
        cb.set_label(label=r'$%s^{%g}$ (%s)'%(vnlb,50,unlb),size=16)
        fig.savefig('%s/djf+jja.m%s.%s.%s.tr.png' % (odir,vn,fo,yr), format='png', dpi=dpi)

        # plot TROPICS ONLY
        fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
        # ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
        ax.set_title(r'%s %s DJF+JJA' % (md.upper(),fo.upper()),fontsize=16)
        clf=ax.contourf(mlon, mlat, llpvn, np.arange(-vm,vm+dvm,dvm),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(),cmap=cmap(vn))
        ax.coastlines()
        ax.set_extent((-180,180,-tlat,tlat),crs=ccrs.PlateCarree())
        gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.5,color='gray',y_inline=False)
        gl.ylocator=mticker.FixedLocator([])
        gl.yformatter=LatitudeFormatter()
        gl.xlines=False
        gl.left_labels=False
        gl.bottom_labels=False
        gl.right_labels=True
        gl.top_labels=False
        cb=fig.colorbar(clf,location='bottom',aspect=50)
        cb.ax.tick_params(labelsize=12)
        cb.set_label(label=r'$%s^{%g}$ (%s)'%(vnlb,p,unlb),size=16)
        fig.savefig('%s/djf+jja.p%02d%s.%s.%s.tr.png' % (odir,p,vn,fo,yr), format='png', dpi=dpi)

    if nhhl:
        fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
        # ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
        ax.set_title(r'%s %s DJF+JJA' % (md.upper(),fo.upper()),fontsize=16)
        clf=ax.contourf(mlon, mlat, llmvn, np.arange(-vm,vm+dvm,dvm),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(),cmap=cmap(vn))
        if 'tas' in vn:
            ax.contour(mlon,mlat,llmvn,[0],transform=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_extent((-180,180,plat,90),crs=ccrs.PlateCarree())
        gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.5,color='gray',y_inline=False)
        gl.yformatter=LatitudeFormatter()
        gl.ylocator=mticker.FixedLocator([])
        gl.xlines=False
        gl.left_labels=False
        gl.bottom_labels=False
        gl.right_labels=True
        gl.top_labels=False
        cb=fig.colorbar(clf,location='bottom',aspect=50)
        cb.ax.tick_params(labelsize=12)
        cb.set_label(label=r'$%s^{%g}$ (%s)'%(vnlb,50,unlb),size=16)
        fig.savefig('%s/djf+jja.m%s.%s.%s.nhhl.png' % (odir,vn,fo,yr), format='png', dpi=dpi)

        fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
        # ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
        ax.set_title(r'%s %s DJF+JJA' % (md.upper(),fo.upper()),fontsize=16)
        clf=ax.contourf(mlon, mlat, llpvn, np.arange(-vm,vm+dvm,dvm),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(),cmap=cmap(vn))
        if 'tas' in vn:
            ax.contour(mlon,mlat,llpvn,[0],transform=ccrs.PlateCarree())
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
        cb.ax.tick_params(labelsize=12)
        cb.set_label(label=r'$%s^{%g}$ (%s)'%(vnlb,p,unlb),size=16)
        fig.savefig('%s/djf+jja.p%02d%s.%s.%s.nhhl.png' % (odir,p,vn,fo,yr), format='png', dpi=dpi)


    # plot pct
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, llmvn, np.arange(-vm,vm+dvm,dvm),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(), cmap=cmap(vn))
    ax.coastlines()
    ax.set_title(r'%s %s DJF+JJA' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$%s^{%g}$ (%s)'%(vnlb,50,unlb),size=16)
    fig.savefig('%s/djf+jja.m%s.%s.%s.png' % (odir,vn,fo,yr), format='png', dpi=dpi)

    # plot pct
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, llpvn, np.arange(-vm,vm+dvm,dvm),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(), cmap=cmap(vn))
    ax.coastlines()
    ax.set_title(r'%s %s DJF+JJA' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$%s^{%g}$ (%s)'%(vnlb,p,unlb),size=16)
    fig.savefig('%s/djf+jja.p%02d%s.%s.%s.png' % (odir,p,vn,fo,yr), format='png', dpi=dpi)

    # plot pct
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, llpvn-llmvn, np.arange(-vmd,vmd+dvmd,dvmd),extend='both', vmax=vmd, vmin=-vmd, transform=ccrs.PlateCarree(), cmap=cmap(vn))
    ax.coastlines()
    ax.set_title(r'%s %s DJF+JJA' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\delta %s^{%g}$ (%s)'%(vnlb,p,unlb),size=16)
    fig.savefig('%s/djf+jja.dp%02d%s.%s.%s.png' % (odir,p,vn,fo,yr), format='png', dpi=dpi)

# run
[plot(vn) for vn in lvn]
