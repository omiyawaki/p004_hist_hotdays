import os
import sys
import pickle
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm

se='ts'
fo1='historical'
fo2='ssp370'
fo='%s+%s'%(fo1,fo2)
yr='1950-2020'
varn='tas'
pc=[1e-3,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,82,85,87,90,92,95,97,99] # percentiles to compute
p=95
ipc=pc.index(p)


tavg='summer'

# load land indices
lmi,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

def rgll(vn):
    # regrid to lat lon
    rvn=np.nan*np.ones([gr['lat'].size*gr['lon'].size])
    rvn[lmi]=vn.data
    rvn=np.reshape(rvn,(gr['lat'].size,gr['lon'].size))
    return np.append(rvn,rvn[:,0][:,None],axis=1)

def selhemi(jja,djf,gr):
    hemi=np.copy(jja)
    if tavg=='summer':
        hemi[gr['lat']>0]=jja[gr['lat']>0]
        hemi[gr['lat']<=0]=djf[gr['lat']<=0]
    elif tavg=='winter':
        hemi[gr['lat']>=0]=djf[gr['lat']>0]
        hemi[gr['lat']<0]=jja[gr['lat']<0]
    return hemi

def getslope(seas,md):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,varn)
    mstats=pickle.load(open('%s/trend.m.%s.%s.%s.pickle' %      (idir,varn,yr,seas), 'rb'))	
    pstats=pickle.load(open('%s/trend.pc.%s.%s.%s.pickle' %      (idir,varn,yr,seas), 'rb'))[ipc]	
    if md=='mmm':
        ms=10*mstats # convert to warming per dec
        ps=10*pstats # convert to warming per dec
    else:
        ms=10*mstats['slope'] # convert to warming per dec
        ps=10*pstats['slope'] # convert to warming per dec
    ms=rgll(ms)
    ps=rgll(ps)
    gra=gr.copy()
    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gra['lon'] = np.append(gr['lon'].data,360)
    return ms,ps,gra

def plot(md):
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    msj,psj,gra=getslope('jja',md)
    msd,psd,_ =getslope('djf',md)
    ms=selhemi(msj,msd,gra)
    ps=selhemi(psj,psd,gra)

    [mlat,mlon] = np.meshgrid(gra['lat'], gra['lon'], indexing='ij')

    # plot trends
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
    # vmax=np.max(slope[str(pc)])
    vmax=0.3
    clf=ax.contourf(mlon, mlat, ps-ms, np.arange(-vmax,vmax,0.05),extend='both', vmax=vmax, vmin=-vmax, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    # ax.contourf(mlon, mlat, sig[str(pc)], 3, hatches=['','....'], alpha=0, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(r'%s %s (%s)' % (tavg.upper(),md,yr))
    cb=plt.colorbar(clf,location='bottom')
    cb.set_label(r'$T^{%s}-\overline{T}$ Trend (K dec$^{-1}$)' % p)
    plt.savefig('%s/trend.%s.p%02d-m.%s.png' % (odir,varn,p,se), format='png', dpi=600)
    # plt.savefig('%s/trend.%s.p%02d-m.%s.pdf' % (odir,varn,p,se), format='pdf', dpi=600)
    plt.close()

plot('mmm')
