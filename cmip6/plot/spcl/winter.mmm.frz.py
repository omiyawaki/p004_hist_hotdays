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
tvn='tas'
lvn=['fsm']
vnp= 'fsm'
tlat=30
plat=30
nhhl=True
tropics=False
reverse=True
# lvn=['ooplh','ooplh_fixbc','ooplh_fixmsm','ooplh_rsm']
# vnp='ooplh'
se = 'sc' # season (ann, djf, mam, jja, son)

# fo='historical' # forcings 
# yr='1980-2000'

fo='ssp370' # forcings 
yr='gwl2.0'

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

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s' % (se,fo,md)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # climo 
    def loadvn(px,varn):
        xvn=xr.open_dataarray('%s/%s/%s.%s_%s.%s.nc' % (idir,varn,px,varn,yr,se))
        if 'pc' in px:
            xvn=xvn.sel(percentile=xvn['percentile'].isin(lp)).mean('percentile')
        if reverse and (varn in ['fsm','gflx','hfss','hfls','fat850','fa850','advt850','advtx850','advty850','advm850','advmx850','advmy850','rfa'] or 'ooplh' in varn):
            xvn=-xvn
        if 'wap' in varn:
            xvn=xvn*86400/100
        if 'tas' in varn:
            xvn=xvn-273.15
        return xvn

    mvn=loadvn('md',vn)
    pvn=loadvn('pc',vn)
    mtas=loadvn('md',tvn)
    ptas=loadvn('pc',tvn)
    gpi=pvn['gpi']

    # seas means
    def selmon(xvn,smon):
        return xvn.sel(month=xvn['month'].isin(smon)).mean('month')

    mvnnh,mvnsh=selmon(mvn,nhmon),selmon(mvn,shmon)
    pvnnh,pvnsh=selmon(pvn,nhmon),selmon(pvn,shmon)
    mtasnh,mtassh=selmon(mtas,nhmon),selmon(mtas,shmon)
    ptasnh,ptassh=selmon(ptas,nhmon),selmon(ptas,shmon)

    # remap to lat x lon
    def remap(xvn):
        llxvn=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
        llxvn[lmi]=xvn.data
        return np.reshape(llxvn,(gr['lat'].size,gr['lon'].size))

    llmvnnh,llmvnsh=remap(mvnnh),remap(mvnsh)
    llpvnnh,llpvnsh=remap(pvnnh),remap(pvnsh)
    llmtasnh,llmtassh=remap(mtasnh),remap(mtassh)
    llptasnh,llptassh=remap(ptasnh),remap(ptassh)

    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    def replon(xvn):
        return np.append(xvn, xvn[...,0][...,None],axis=1)

    llmvnnh,llmvnsh=replon(llmvnnh),replon(llmvnsh)
    llpvnnh,llpvnsh=replon(llpvnnh),replon(llpvnsh)
    llmtasnh,llmtassh=replon(llmtasnh),replon(llmtassh)
    llptasnh,llptassh=replon(llptasnh),replon(llptassh)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # use djf for nh, jja for sh
    def mergehemi(nh,sh):
        xvn=np.copy(sh)
        xvn[gr['lat']>0]=nh[gr['lat']>0]
        return xvn

    llmvn=mergehemi(llmvnnh,llmvnsh)
    llpvn=mergehemi(llpvnnh,llpvnsh)
    llmtas=mergehemi(llmtasnh,llmtassh)
    llptas=mergehemi(llptasnh,llptassh)

    # scatter
    fig,ax=plt.subplots(figsize=(5,4),constrained_layout=True)
    ax.plot(llmtas.flatten(),llmvn.flatten(),'.k')
    ax.set_xlabel('$%s^{%g}$ (%s)'%(varnlb(tvn),50,unitlb(tvn)))
    ax.set_ylabel('$%s^{%g}$ (%s)'%(vnlb,50,unlb))
    fig.savefig('%s/scat.m%g.%s.%s.png'%(odir,50,tvn,vn),format='png',dpi=dpi)

    fig,ax=plt.subplots(figsize=(5,4),constrained_layout=True)
    ax.plot(llptas.flatten(),llpvn.flatten(),'.k')
    ax.set_xlabel('$%s^{%g}$ (%s)'%(varnlb(tvn),p,unitlb(tvn)))
    ax.set_ylabel('$%s^{%g}$ (%s)'%(vnlb,p,unlb))
    fig.savefig('%s/scat.p%g.%s.%s.png'%(odir,p,tvn,vn),format='png',dpi=dpi)

    if nhhl:
        fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
        # ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
        ax.set_title(r'%s %s DJF+JJA' % (md.upper(),fo.upper()),fontsize=16)
        clf=ax.contourf(mlon, mlat, llmvn, np.arange(-vm,vm+dvm,dvm),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(),cmap=cmap(vn))
        ax.contour(mlon,mlat,llmtas,[-2,5],linewidths=1,transform=ccrs.PlateCarree())
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
        fig.savefig('%s/djf+jja.m%s.%s.%s.nhhl.frz.png' % (odir,vn,fo,yr), format='png', dpi=dpi)

        fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
        # ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
        ax.set_title(r'%s %s DJF+JJA' % (md.upper(),fo.upper()),fontsize=16)
        clf=ax.contourf(mlon, mlat, llpvn, np.arange(-vm,vm+dvm,dvm),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(),cmap=cmap(vn))
        ax.contour(mlon,mlat,llptas,[-2,5],linewidths=1,transform=ccrs.PlateCarree())
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
        fig.savefig('%s/djf+jja.p%02d%s.%s.%s.nhhl.frz.png' % (odir,p,vn,fo,yr), format='png', dpi=dpi)


    # plot pct
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, llmvn, np.arange(-vm,vm+dvm,dvm),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(), cmap=cmap(vn))
    ax.contour(mlon,mlat,llmtas,[0],linewidths=1,transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(r'%s %s DJF+JJA' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$%s^{%g}$ (%s)'%(vnlb,50,unlb),size=16)
    fig.savefig('%s/djf+jja.m%s.%s.%s.frz.png' % (odir,vn,fo,yr), format='png', dpi=dpi)

    # plot pct
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, llpvn, np.arange(-vm,vm+dvm,dvm),extend='both', vmax=vm, vmin=-vm, transform=ccrs.PlateCarree(), cmap=cmap(vn))
    ax.contour(mlon,mlat,llptas,[0],linewidths=1,transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(r'%s %s DJF+JJA' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$%s^{%g}$ (%s)'%(vnlb,p,unlb),size=16)
    fig.savefig('%s/djf+jja.p%02d%s.%s.%s.frz.png' % (odir,p,vn,fo,yr), format='png', dpi=dpi)

    # plot pct
    fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
    clf=ax.contourf(mlon, mlat, llpvn-llmvn, np.arange(-vmd,vmd+dvmd,dvmd),extend='both', vmax=vmd, vmin=-vmd, transform=ccrs.PlateCarree(), cmap=cmap(vn))
    ax.contour(mlon,mlat,llptas,[0],linewidths=1,transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(r'%s %s DJF+JJA' % (md.upper(),fo.upper()),fontsize=16)
    cb=fig.colorbar(clf,location='bottom',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\delta %s^{%g}$ (%s)'%(vnlb,p,unlb),size=16)
    fig.savefig('%s/djf+jja.dp%02d%s.%s.%s.frz.png' % (odir,p,vn,fo,yr), format='png', dpi=dpi)

# run
[plot(vn) for vn in lvn]
