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

mn=7
nt=7 # window size in days
p=95
lvn=['pr']
vnp='pr'
se = 'sc' # season (ann, djf, mam, jja, son)
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

def vmaxdd(vn):
    lvm={   'tas':  [1,0.1],
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
            'td_mrsos': [2,0.1],
            'ti_pr': [5,0.5],
            }
    return lvm[vn]

def vmaxd(vn):
    lvm={   'tas':  [4,0.25],
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
            'td_mrsos': [2,0.1],
            'ti_pr': [10,1],
            }
    return lvm[vn]

def plot(vn):
    vmdd,dvmdd=vmaxdd(vn)
    vmd,dvmd=vmaxd(vn)

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # warming
    dvn=xr.open_dataset('%s/d.%s_%s_%s.%s.nc' % (idir,vn,his,fut,se))['__xarray_dataarray_variable__']
    try:
        dpvn=xr.open_dataset('%s/dpc.%s_%s_%s.%s.nc' % (idir,vn,his,fut,se))[vn]
    except:
        dpvn=xr.open_dataset('%s/dpc.%s_%s_%s.%s.nc' % (idir,vn,his,fut,se))[vnp]
    ds=xr.open_dataset('%s/ddpc.%s_%s_%s.%s.nc' % (idir,vn,his,fut,se))
    try:
        ddpvn=ds[vn]
    except:
        ddpvn=ds[vnp]
    pct=ds['percentile']
    gpi=ds['gpi']
    dpvn=dpvn.sel(percentile=pct==p).squeeze()
    ddpvn=ddpvn.sel(percentile=pct==p).squeeze()

    # jja and djf means
    dvnj  =dvn.data[mn-1,:]
    dpvnj =dpvn.data[mn-1,:]
    ddpvnj=ddpvn.data[mn-1,:]
    dvnd  =dvn.data[(mn+5)%12,:]
    dpvnd =dpvn.data[(mn+5)%12,:]
    ddpvnd=ddpvn.data[(mn+5)%12,:]

    # remap to lat x lon
    lldvnj=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    lldpvnj=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llddpvnj=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    lldvnj[lmi]=dvnj.data
    lldpvnj[lmi]=dpvnj.data
    llddpvnj[lmi]=ddpvnj.data
    lldvnj=np.reshape(lldvnj,(gr['lat'].size,gr['lon'].size))
    lldpvnj=np.reshape(lldpvnj,(gr['lat'].size,gr['lon'].size))
    llddpvnj=np.reshape(llddpvnj,(gr['lat'].size,gr['lon'].size))

    lldvnd=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    lldpvnd=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llddpvnd=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    lldvnd[lmi]=dvnd.data
    lldpvnd[lmi]=dpvnd.data
    llddpvnd[lmi]=ddpvnd.data
    lldvnd=np.reshape(lldvnd,(gr['lat'].size,gr['lon'].size))
    lldpvnd=np.reshape(lldpvnd,(gr['lat'].size,gr['lon'].size))
    llddpvnd=np.reshape(llddpvnd,(gr['lat'].size,gr['lon'].size))

    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    lldvnj = np.append(lldvnj, lldvnj[...,0][...,None],axis=1)
    lldpvnj = np.append(lldpvnj, lldpvnj[...,0][...,None],axis=1)
    llddpvnj = np.append(llddpvnj, llddpvnj[...,0][...,None],axis=1)
    lldvnd = np.append(lldvnd, lldvnd[...,0][...,None],axis=1)
    lldpvnd = np.append(lldpvnd, lldpvnd[...,0][...,None],axis=1)
    llddpvnd = np.append(llddpvnd, llddpvnd[...,0][...,None],axis=1)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # use jja for nh, djf for sh
    lldvn=np.copy(lldvnd)
    lldpvn=np.copy(lldpvnd)
    llddpvn=np.copy(llddpvnd)
    lldvn[gr['lat']>0]=lldvnj[gr['lat']>0]
    lldpvn[gr['lat']>0]=lldpvnj[gr['lat']>0]
    llddpvn[gr['lat']>0]=llddpvnj[gr['lat']>0]

    # plot pct warming - mean warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, llddpvn, np.arange(-vmdd,vmdd+dvmdd,dvmdd),extend='both', vmax=vmdd, vmin=-vmdd, transform=ccrs.PlateCarree(), cmap=cmap(vn))
    ax.coastlines()
    ax.set_title(r'%s %s %s+%s' % (md.upper(),fo.upper(),monname(mn-1),monname((mn+5)%12)),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\Delta \delta %s^{%02d}$ (%s)'%(varnlb(vn),p,unitlb(vn)),size=16)
    fig.savefig('%s/mn.ddp%02d%s.%s.%s.%s.png' % (odir,p,vn,fo,fut,mn), format='png', dpi=dpi)

    # plot pct warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, lldpvn, np.arange(-vmd,vmd+dvmd,dvmd),extend='both', vmax=vmd, vmin=-vmd, transform=ccrs.PlateCarree(), cmap=cmap(vn))
    ax.coastlines()
    ax.set_title(r'%s %s %s+%s' % (md.upper(),fo.upper(),monname(mn-1),monname((mn+5)%12)),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=12)
    cb.set_label(label=r'$\Delta %s^{%02d}$ (%s)'%(varnlb(vn),p,unitlb(vn)),size=16)
    fig.savefig('%s/mn.dp%02d%s.%s.%s.%s.png' % (odir,p,vn,fo,fut,mn), format='png', dpi=dpi)

    # plot mean warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, lldvn, np.arange(-vmd,vmd+dvmd,dvmd),extend='both', vmax=vmd, vmin=-vmd, transform=ccrs.PlateCarree(), cmap=cmap(vn))
    ax.coastlines()
    ax.set_title(r'%s %s %s+%s' % (md.upper(),fo.upper(),monname(mn-1),monname((mn+5)%12)),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=12)
    cb.set_label(label=r'$\Delta \overline{%s}$ (%s)'%(varnlb(vn),unitlb(vn)),size=16)
    fig.savefig('%s/mn.d%s.%s.%s.%s.png' % (odir,vn,fo,fut,mn), format='png', dpi=dpi)

# run
[plot(vn) for vn in lvn]
