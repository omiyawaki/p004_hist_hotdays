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
p=2.5
tlat=30
plat=30
lvn=['advt850_t18']
vnp= 'advt850_t18'
nhhl=True
tr=False
reverse=True
# lvn=['ooplh','ooplh_fixbc','ooplh_fixmsm','ooplh_rddsm']
# vnp='ooplh'
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

def cmap(vn):
    vbr=['td_mrsos','ti_pr','ti_ev']
    if vn in vbr:
        return 'BrBG'
    else:
        return 'RdBu_r'

def vmaxdd(vn):
    lvm={   
            'tas':  [3,0.3],
            'advt850_t18_t':  [3,0.3],
            'advty850_t18_t':  [3,0.3],
            'ta850':  [1,0.1],
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
            'ooplh_rddsm': [10,1],
            'td_mrsos': [2,0.1],
            'ti_pr': [5,0.5],
            'fa850': [0.3,0.03],
            'fat850': [0.3,0.03],
            'advt850': [0.1,0.01],
            'advt_doy850': [0.1,0.01],
            'advty_doy850': [0.1,0.01],
            'advt850_t18': [0.1,0.01],
            'advtx850': [0.1,0.01],
            'advty850': [0.1,0.01],
            'advty850_t18': [0.1,0.01],
            'advm850': [0.1,0.01],
            'advmx850': [0.1,0.01],
            'advmy850': [0.1,0.01],
            }
    return lvm[vn]

def plot(vn):
    vmdd,dvmdd=vmaxdd(vn)

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # warming
    ddpvn=xr.open_dataarray('%s/ddpc.md.%s_%s_%s.%s.nc' % (idir,vn,his,fut,se))
    pct=ddpvn['percentile']
    gpi=ddpvn['gpi']
    ddpvn=ddpvn.sel(percentile=pct==p).squeeze()
    if reverse and (vn in ['gflx','hfss','hfls','fat850','fa850','rfa'] or 'ooplh' in vn or 'adv' in vn):
        ddpvn=-ddpvn

    # ond and amj means
    ddpvno=np.nanmean(ddpvn.data[9:12,:],axis=0)
    ddpvna=np.nanmean(ddpvn.data[3:6,:],axis=0)

    # remap to lat x lon
    llddpvno=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llddpvno[lmi]=ddpvno.data
    llddpvno=np.reshape(llddpvno,(gr['lat'].size,gr['lon'].size))

    llddpvna=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    llddpvna[lmi]=ddpvna.data
    llddpvna=np.reshape(llddpvna,(gr['lat'].size,gr['lon'].size))

    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    llddpvno = np.append(llddpvno, llddpvno[...,0][...,None],axis=1)
    llddpvna = np.append(llddpvna, llddpvna[...,0][...,None],axis=1)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # use ond for nh, amj for sh
    llddpvn=np.copy(llddpvna)
    llddpvn[gr['lat']>0]=llddpvno[gr['lat']>0]

    # plot TROPICS ONLY
    if tr:
        fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
        # ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
        ax.set_title(r'%s %s OND+AMJ' % (md.upper(),fo.upper()),fontsize=16)
        clf=ax.contourf(mlon, mlat, llddpvn, np.arange(-vmdd,vmdd+dvmdd,dvmdd),extend='both', vmax=vmdd, vmin=-vmdd, transform=ccrs.PlateCarree(),cmap='RdBu_r')
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
        cb.set_label(label=r'$\Delta \delta %s$ (%s)'%(varnlb(vn),unitlb(vn)),size=16)
        fig.savefig('%s/ond+amj.ddp%02d%s.%s.%s.tr.pdf' % (odir,p,vn,fo,fut), format='pdf', dpi=dpi)
        fig.savefig('%s/ond+amj.ddp%02d%s.%s.%s.tr.png' % (odir,p,vn,fo,fut), format='png', dpi=dpi)

    if nhhl:
        fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
        clf=ax.contourf(mlon, mlat, llddpvn, np.arange(-vmdd,vmdd+dvmdd,dvmdd),extend='both', vmax=vmdd, vmin=-vmdd, transform=ccrs.PlateCarree(), cmap=cmap(vn))
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
        cb.set_label(label=r'$\Delta \delta %s$ (%s)'%(varnlb(vn),unitlb(vn)),size=16)
        fig.savefig('%s/ond+amj.ddp%02d%s.%s.%s.nhhl.png' % (odir,p,vn,fo,fut), format='png', dpi=dpi)

    # plot pct warming - mean warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, llddpvn, np.arange(-vmdd,vmdd+dvmdd,dvmdd),extend='both', vmax=vmdd, vmin=-vmdd, transform=ccrs.PlateCarree(), cmap=cmap(vn))
    ax.coastlines()
    ax.set_title(r'%s %s OND+AMJ' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\Delta \delta %s$ (%s)'%(varnlb(vn),unitlb(vn)),size=16)
    fig.savefig('%s/ond+amj.ddp%02d%s.%s.%s.png' % (odir,p,vn,fo,fut), format='png', dpi=dpi)

# run
[plot(vn) for vn in lvn]
