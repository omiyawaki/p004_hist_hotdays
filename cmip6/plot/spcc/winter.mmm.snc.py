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
gvn='snc'
lvn=['fsm']
vnp= 'fsm'
domp=True
nhmon=[12,1,2]
shmon=[6,7,8]
tlat=30 # upper bound for low latitude
plat=30 # lower bound for high latitude
nhhl=True
tropics=False
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

md='CESM2'

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
            'tas':  [2,0.2],
            'snc':  [0.5,0.05],
            'advt850_t':  [2,0.2],
            'advty_mon850_t':  [2,0.2],
            'advty_mon850_t_hs':  [2,0.2],
            'advt850_t18_t':  [2,0.2],
            'advty850_t18_t':  [2,0.2],
            'advt850_t_hs':  [2,0.2],
            'ta850':  [2,0.2],
            'twas': [2,0.2],
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
            'gflx': [10,1],
            'fsm': [10,1],
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
            'advt850_wm2': [50,5],
            'advt850': [0.1,0.01],
            'advt_doy850': [0.1,0.01],
            'advt_mon850': [0.1,0.01],
            'advty_doy850': [0.1,0.01],
            'advty_mon850': [0.1,0.01],
            'advty_mon925': [0.1,0.01],
            'advt850_t18': [0.1,0.01],
            'advtx850': [0.03,0.003],
            'advtx850_t18': [0.03,0.003],
            'advty850': [0.1,0.01],
            'advm850': [0.1,0.01],
            'advmx850': [0.03,0.003],
            'advmy850': [0.1,0.01],
            }
    return lvm[vn]

def vmaxd(vn):
    lvm={   
            'tas':  [2,0.2],
            'snc':  [0.5,0.05],
            'advt850_t':  [2,0.2],
            'advty_mon850_t':  [2,0.2],
            'advty_mon850_t_hs':  [2,0.2],
            'advt850_t18_t':  [2,0.2],
            'advty850_t18_t':  [2,0.2],
            'advt850_t_hs':  [2,0.2],
            'ta850':  [2,0.2],
            'twas': [2,0.2],
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
            'gflx': [10,1],
            'fsm': [10,1],
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
            'advt850_wm2': [50,5],
            'advt850': [0.1,0.01],
            'advt_doy850': [0.1,0.01],
            'advt_mon850': [0.1,0.01],
            'advty_doy850': [0.1,0.01],
            'advty_mon850': [0.1,0.01],
            'advty_mon925': [0.1,0.01],
            'advt850_t18': [0.1,0.01],
            'advtx850': [0.03,0.003],
            'advtx850_t18': [0.03,0.003],
            'advty850': [0.1,0.01],
            'advm850': [0.1,0.01],
            'advmx850': [0.03,0.003],
            'advmy850': [0.1,0.01],
            }
    return lvm[vn]

def plot(vn):
    vmdd,dvmdd=vmaxdd(vn)
    vmd,dvmd=vmaxd(vn)
    vnlb=varnlb(vn)
    unlb=unitlb(vn)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    vno=vn
    if '_wm2' in vn:
        vn=vn.replace('_wm2','')

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s' % (se,fo,md)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # warming
    def loadvar(px,varn):
        xvn=xr.open_dataarray('%s/%s/%s.md.%s_%s_%s.%s.nc' % (idir,varn,px,varn,his,fut,se))
        if 'pc' in px:
            pct=xvn['percentile']
            xvn=xvn.sel(percentile=pct==p).squeeze()
        if reverse and (varn in ['fsm','gflx','hfss','hfls','fat850','fa850','advt850_wm2','advt850','advtx850','advty850','advm850','advmx850','advmy850','rfa'] or 'ooplh' in varn or 'adv' in varn) and '_t' not in varn:
            xvn=-xvn
        if '_wm2' in vno:
            xvn=1.16*1500*xvn
        return xvn

    if domp:
        dmvn=loadvar('d',vn)
        dpvn=loadvar('dpc',vn)
        gmvn=loadvar('d',gvn)
        gpvn=loadvar('dpc',gvn)
    ddpvn=loadvar('ddpc',vn)
    gdpvn=loadvar('ddpc',gvn)
    pct=ddpvn['percentile']
    gpi=ddpvn['gpi']

    # jja and djf means
    def nhsh(xvn):
        xnh=xvn.sel(month=xvn['month'].isin(nhmon)).mean('month')
        xsh=xvn.sel(month=xvn['month'].isin(shmon)).mean('month')
        return xnh,xsh

    if domp:
        dmvnnh,dmvnsh=nhsh(dmvn)
        dpvnnh,dpvnsh=nhsh(dpvn)
        gmvnnh,gmvnsh=nhsh(gmvn)
        gpvnnh,gpvnsh=nhsh(gpvn)
    ddpvnnh,ddpvnsh=nhsh(ddpvn)
    gdpvnnh,gdpvnsh=nhsh(gdpvn)

    # remap to lat x lon
    def remap(xvn):
        llxvn=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
        llxvn[lmi]=xvn.data
        llxvn=np.reshape(llxvn,(gr['lat'].size,gr['lon'].size))
        return llxvn

    if domp:
        lldmvnnh,lldmvnsh=remap(dmvnnh),remap(dmvnsh)
        lldpvnnh,lldpvnsh=remap(dpvnnh),remap(dpvnsh)
        llgmvnnh,llgmvnsh=remap(gmvnnh),remap(gmvnsh)
        llgpvnnh,llgpvnsh=remap(gpvnnh),remap(gpvnsh)
    llddpvnnh,llddpvnsh=remap(ddpvnnh),remap(ddpvnsh)
    llgdpvnnh,llgdpvnsh=remap(gdpvnnh),remap(gdpvnsh)

    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    def replon(xvn):
        xvn = np.append(xvn, xvn[...,0][...,None],axis=1)
        return xvn

    if domp:
        lldmvnnh, lldmvnsh =replon(lldmvnnh), replon(lldmvnsh)
        lldpvnnh, lldpvnsh =replon(lldpvnnh), replon(lldpvnsh)
        llgmvnnh, llgmvnsh =replon(llgmvnnh), replon(llgmvnsh)
        llgpvnnh, llgpvnsh =replon(llgpvnnh), replon(llgpvnsh)
    llddpvnnh,llddpvnsh=replon(llddpvnnh),replon(llddpvnsh)
    llgdpvnnh,llgdpvnsh=replon(llgdpvnnh),replon(llgdpvnsh)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # use djf for nh, jja for sh
    def mrghemi(nh,sh):
        xvn=np.copy(sh)
        xvn[gr['lat']>0]=nh[gr['lat']>0]
        return xvn

    if domp:
        lldmvn=mrghemi(lldmvnnh,lldmvnsh)
        lldpvn=mrghemi(lldpvnnh,lldpvnsh)
        llgmvn=mrghemi(llgmvnnh,llgmvnsh)
        llgpvn=mrghemi(llgpvnnh,llgpvnsh)
    llddpvn=mrghemi(llddpvnnh,llddpvnsh)
    llgdpvn=mrghemi(llgdpvnnh,llgdpvnsh)

    if nhhl:
        if domp:
            # plot NH HL ONLY
            fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
            # ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
            ax.set_title(r'%s %s DJF+JJA' % (md.upper(),fo.upper()),fontsize=16)
            clf=ax.contourf(mlon, mlat, lldmvn, np.arange(-vmd,vmd+dvmd,dvmd),extend='both', vmax=vmd, vmin=-vmd, transform=ccrs.PlateCarree(),cmap='RdBu_r')
            ax.contour(mlon,mlat,llgmvn,np.arange(-0.5,0,0.1),linewidths=0.5,transform=ccrs.PlateCarree())
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
            cb.set_label(label=r'$\Delta %s^{%g}$ (%s)'%(vnlb,50,unlb),size=16)
            fig.savefig('%s/djf+jja.d%02d%s.%s.%s.hl.snc.png' % (odir,50,vn,fo,fut), format='png', dpi=dpi)

            # plot NH HL ONLY
            fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
            # ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
            ax.set_title(r'%s %s DJF+JJA' % (md.upper(),fo.upper()),fontsize=16)
            clf=ax.contourf(mlon, mlat, lldpvn, np.arange(-vmd,vmd+dvmd,dvmd),extend='both', vmax=vmd, vmin=-vmd, transform=ccrs.PlateCarree(),cmap='RdBu_r')
            ax.contour(mlon,mlat,llgpvn,np.arange(-0.5,0,0.1),linewidths=0.5,transform=ccrs.PlateCarree())
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
            cb.set_label(label=r'$\Delta %s^{%g}$ (%s)'%(vnlb,p,unlb),size=16)
            fig.savefig('%s/djf+jja.dp%02d%s.%s.%s.hl.snc.png' % (odir,p,vn,fo,fut), format='png', dpi=dpi)

        # plot NH HL ONLY
        fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
        # ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
        ax.set_title(r'%s %s DJF+JJA' % (md.upper(),fo.upper()),fontsize=16)
        clf=ax.contourf(mlon, mlat, llddpvn, np.arange(-vmdd,vmdd+dvmdd,dvmdd),extend='both', vmax=vmdd, vmin=-vmdd, transform=ccrs.PlateCarree(),cmap='RdBu_r')
        ax.contour(mlon,mlat,llgdpvn,np.arange(-0.5,0,0.1),linewidths=0.5,transform=ccrs.PlateCarree())
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
        cb.set_label(label=r'$\Delta \delta %s$ (%s)'%(vnlb,unlb),size=16)
        fig.savefig('%s/djf+jja.ddp%02d%s.%s.%s.hl.snc.png' % (odir,p,vn,fo,fut), format='png', dpi=dpi)


    # plot pct warming - mean warming
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, llddpvn, np.arange(-vmdd,vmdd+dvmdd,dvmdd),extend='both', vmax=vmdd, vmin=-vmdd, transform=ccrs.PlateCarree(), cmap=cmap(vn))
    ax.contour(mlon,mlat,llgdpvn,np.arange(-0.5,0,0.1),linewidths=1,transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(r'%s %s DJF+JJA' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\Delta \delta %s$ (%s)'%(vnlb,unlb),size=16)
    fig.savefig('%s/winter.ddp%02d%s.%s.%s.snc.png' % (odir,p,vn,fo,fut), format='png', dpi=dpi)

# run
[plot(vn) for vn in lvn]
