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
p=97.5
lvn=['wap850']
vnp= 'wap850'
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

md='mmm'

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
    else:
        return 'RdBu_r'

def vmax(vn):
    lvm={   
            'tas':  [1,0.1],
            'ta850':  [1,0.1],
            'wap850':  [50,5],
            'twas': [1,0.1],
            'hurs': [5,0.5],
            'ef':  [0.05,0.005],
            'ef2':  [0.05,0.005],
            'ef3':  [0.05,0.005],
            'mrsos': [2,0.2],
            'sfcWind': [0.5,0.005],
            'rsfc': [10,1],
            'swsfc': [10,1],
            'lwsfc': [10,1],
            'hfls': [10,1],
            'hfss': [10,1],
            'gflx': [5,0.5],
            'plh': [10,1],
            'plh_fixbc': [10,1],
            'ooef': [0.05,0.005],
            'ooef2': [0.05,0.005],
            'ooef3': [0.05,0.005],
            'oopef': [0.05,0.005],
            'oopef2': [0.05,0.005],
            'oopef3': [0.05,0.005],
            'oopef_fixbc': [0.05,0.005],
            'oopef3_fixbc': [0.05,0.005],
            'ooplh': [10,1],
            'ooplh_msm': [10,1],
            'ooplh_fixmsm': [10,1],
            'ooplh_orig': [10,1],
            'ooplh_fixbc': [10,1],
            'ooplh_rsm': [10,1],
            'td_mrsos': [2,0.1],
            'ti_pr': [5,0.5],
            'fa850': [0.3,0.03],
            'fat850': [0.3,0.03],
            'advt850': [0.1,0.01],
            'advtx850': [0.1,0.01],
            'advty850': [0.1,0.01],
            'advm850': [0.1,0.01],
            'advmx850': [0.1,0.01],
            'advmy850': [0.1,0.01],
            }
    return lvm[vn]

def vmaxd(vn):
    lvm={   
            'tas':  [4,0.25],
            'ta850':  [4,0.25],
            'wap850':  [50,5],
            'twas': [4,0.25],
            'hurs': [10,0.5],
            'ef':  [0.1,0.01],
            'ef2':  [0.1,0.01],
            'ef3':  [0.1,0.01],
            'mrsos': [4,0.25],
            'sfcWind': [1,0.01+1e-5],
            'rsfc': [20,2],
            'swsfc': [20,2+1e-5],
            'lwsfc': [20,2],
            'hfls': [20,2],
            'hfss': [20,2],
            'gflx': [10,1],
            'plh': [20,2],
            'plh_fixbc': [20,2],
            'ooef': [0.1,0.01],
            'ooef2': [0.1,0.01],
            'ooef3': [0.1,0.01],
            'oopef': [0.1,0.01],
            'oopef2': [0.1,0.01],
            'oopef3': [0.1,0.01],
            'oopef_fixbc': [0.1,0.01],
            'oopef3_fixbc': [0.1,0.01],
            'ooplh': [20,2],
            'ooplh_msm': [20,2],
            'ooplh_fixmsm': [20,2],
            'ooplh_orig': [20,2],
            'ooplh_fixbc': [20,2],
            'ooplh_rsm': [20,2],
            'td_mrsos': [2,0.1],
            'ti_pr': [10,1],
            'fa850': [0.3,0.03],
            'fat850': [0.3,0.03],
            'advt850': [0.1,0.01],
            'advtx850': [0.1,0.01],
            'advty850': [0.1,0.01],
            'advm850': [0.1,0.01],
            'advmx850': [0.1,0.01],
            'advmy850': [0.1,0.01],
            }
    return lvm[vn]

def plot(vn):
    vm,dvm=vmax(vn)
    vmd,dvmd=vmaxd(vn)

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # climo 
    try:
        mvn=xr.open_dataarray('%s/m.%s_%s.%s.nc' % (idir,vn,yr,se))
    except:
        mvn=xr.open_dataarray('%s/md.%s_%s.%s.nc' % (idir,vn,yr,se))
    pvn=xr.open_dataarray('%s/pc.%s_%s.%s.nc' % (idir,vn,yr,se))
    pct=pvn['percentile']
    gpi=pvn['gpi']
    pvn=pvn.sel(percentile=pct==p).squeeze()
    if reverse and (vn in ['gflx','hfss','hfls','fat850','fa850','advt850','advtx850','advty850','advm850','advmx850','advmy850','rfa'] or 'ooplh' in vn):
        mvn=-mvn
        pvn=-pvn
    if 'wap' in vn:
        mvn=mvn*86400/100
        pvn=pvn*86400/100

    # ndj and mjj means
    mvnj=np.nanmean(mvn.data[4:7,:]  ,axis=0)
    mvnd=np.nanmean(np.roll(mvn.data,2,axis=0)[:3,:],axis=0)
    pvnj=np.nanmean(pvn.data[4:7,:]  ,axis=0)
    pvnd=np.nanmean(np.roll(pvn.data,2,axis=0)[:3,:],axis=0)

    # remap to lat x lon
    llmvnj=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llmvnj[lmi]=mvnj.data
    llmvnj=np.reshape(llmvnj,(gr['lat'].size,gr['lon'].size))
    llpvnj=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llpvnj[lmi]=pvnj.data
    llpvnj=np.reshape(llpvnj,(gr['lat'].size,gr['lon'].size))

    llmvnd=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llmvnd[lmi]=mvnd.data
    llmvnd=np.reshape(llmvnd,(gr['lat'].size,gr['lon'].size))
    llpvnd=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llpvnd[lmi]=pvnd.data
    llpvnd=np.reshape(llpvnd,(gr['lat'].size,gr['lon'].size))

    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    llmvnj = np.append(llmvnj, llmvnj[...,0][...,None],axis=1)
    llmvnd = np.append(llmvnd, llmvnd[...,0][...,None],axis=1)
    llpvnj = np.append(llpvnj, llpvnj[...,0][...,None],axis=1)
    llpvnd = np.append(llpvnd, llpvnd[...,0][...,None],axis=1)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # use djf for nh, jja for sh
    llmvn=np.copy(llmvnj)
    llmvn[gr['lat']>0]=llmvnd[gr['lat']>0]
    llpvn=np.copy(llpvnj)
    llpvn[gr['lat']>0]=llpvnd[gr['lat']>0]

    if tropics:
        # plot TROPICS ONLY
        fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
        # ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
        ax.set_title(r'%s %s NDJ+MJJ' % (md.upper(),fo.upper()),fontsize=16)
        clf=ax.contourf(mlon, mlat, llmvn, np.arange(-vm,vm+dvm,dvm),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(),cmap='RdBu_r')
        ax.coastlines()
        ax.set_extent((-180,180,-30,30),crs=ccrs.PlateCarree())
        gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.5,color='gray',y_inline=False)
        gl.ylocator=mticker.FixedLocator([-50,-30,0,30,50])
        gl.yformatter=LatitudeFormatter()
        gl.xlines=False
        gl.left_labels=False
        gl.bottom_labels=False
        gl.right_labels=True
        gl.top_labels=False
        cb=fig.colorbar(clf,location='bottom',aspect=50)
        cb.ax.tick_params(labelsize=12)
        cb.set_label(label=r'$%s$ (%s)'%(varnlb(vn),unitlb(vn)),size=16)
        fig.savefig('%s/ndj+mjj.m%02d%s.%s.%s.tr.pdf' % (odir,p,vn,fo,yr), format='pdf', dpi=dpi)
        fig.savefig('%s/ndj+mjj.m%02d%s.%s.%s.tr.png' % (odir,p,vn,fo,yr), format='png', dpi=dpi)

        # plot TROPICS ONLY
        fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
        # ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
        ax.set_title(r'%s %s NDJ+MJJ' % (md.upper(),fo.upper()),fontsize=16)
        clf=ax.contourf(mlon, mlat, llpvn, np.arange(-vm,vm+dvm,dvm),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(),cmap='RdBu_r')
        ax.coastlines()
        ax.set_extent((-180,180,-30,30),crs=ccrs.PlateCarree())
        gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.5,color='gray',y_inline=False)
        gl.ylocator=mticker.FixedLocator([-50,-30,0,30,50])
        gl.yformatter=LatitudeFormatter()
        gl.xlines=False
        gl.left_labels=False
        gl.bottom_labels=False
        gl.right_labels=True
        gl.top_labels=False
        cb=fig.colorbar(clf,location='bottom',aspect=50)
        cb.ax.tick_params(labelsize=12)
        cb.set_label(label=r'$%s$ (%s)'%(varnlb(vn),unitlb(vn)),size=16)
        fig.savefig('%s/ndj+mjj.p%02d%s.%s.%s.tr.pdf' % (odir,p,vn,fo,yr), format='pdf', dpi=dpi)
        fig.savefig('%s/ndj+mjj.p%02d%s.%s.%s.tr.png' % (odir,p,vn,fo,yr), format='png', dpi=dpi)

    # plot pct
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, llmvn, np.arange(-vm,vm+dvm,dvm),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(), cmap=cmap(vn))
    ax.coastlines()
    ax.set_title(r'%s %s NDJ+MJJ' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$%s$ (%s)'%(varnlb(vn),unitlb(vn)),size=16)
    fig.savefig('%s/ndj+mjj.m%02d%s.%s.%s.png' % (odir,p,vn,fo,yr), format='png', dpi=dpi)

    # plot pct
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, llpvn, np.arange(-vm,vm+dvm,dvm),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(), cmap=cmap(vn))
    ax.coastlines()
    ax.set_title(r'%s %s NDJ+MJJ' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$%s$ (%s)'%(varnlb(vn),unitlb(vn)),size=16)
    fig.savefig('%s/ndj+mjj.p%02d%s.%s.%s.png' % (odir,p,vn,fo,yr), format='png', dpi=dpi)

# run
[plot(vn) for vn in lvn]
