import os,sys
from datetime import datetime
from rae_arb import rae_arb
# from rcae import rcae
from tqdm import tqdm
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
sys.path.append('/home/miyawaki/scripts/common')
import constants as c
from pathlib import Path

checkexist=False # check for existing data

# Load GCM data

tstr1='19500101-19591231'
tstr2='19500101-19591231'
tstrm='195001-199912'
tstrl='19500101-19991231'

# tstr1='20000101-20091231'
# tstr2='20000101-20091231'
# tstrm='200001-201412'
# tstrl='20000101-20141231'

itime=1
lat=70
lon=205
idir0='./data'
if not checkexist: [os.remove(fn) for fn in Path(idir0).glob('*.nc')]
if not os.path.exists(idir0): os.makedirs(idir0)
idir1='/project/amp02/miyawaki/data/share/cesm2/b.e21.BHIST.f09_g17.CMIP6-historical.011/atm/proc/tseries/day_1'
idirm='/project/amp02/miyawaki/data/share/cesm2/b.e21.BHIST.f09_g17.CMIP6-historical.011/atm/proc/tseries/month_1'
idirl='/project/amp02/miyawaki/data/share/cesm2/b.e21.BHIST.f09_g17.CMIP6-historical.011/lnd/proc/tseries/day_1'

# load hybrid level coefficients
fn='b.e21.BHIST.f09_g17.CMIP6-historical.011.cam.h1.%s.%s.nc'%('T',tstr1)
hyam=xr.open_dataset('%s/%s'%(idir1,fn))['hyam'].data
hybm=xr.open_dataset('%s/%s'%(idir1,fn))['hybm'].data
p0=xr.open_dataset('%s/%s'%(idir1,fn))['P0'].data

def load_mycesm(vn):
    nfn='%s/%s.nc'%(idir0,vn.lower())
    try:
        print('Loading proc data')
        return xr.open_dataarray(nfn)
    except:
        print('Loading raw data')
        if vn=='TTGWORO':
            fn='b.e21.BHIST.f09_g17.CMIP6-historical.011.cam.h0.%s.%s.nc'%(vn,tstrm)
            xvn=xr.open_dataset('%s/%s'%(idirm,fn))[vn].sel(lat=lat,lon=lon,method='nearest').isel(time=itime)
        elif vn in ['FGR12','ESOIT']:
            fn='b.e21.BHIST.f09_g17.CMIP6-historical.011.clm2.h5.%s.%s.nc'%(vn,tstrl)
            xvn=xr.open_dataset('%s/%s'%(idirl,fn))[vn].sel(lat=lat,lon=lon,method='nearest').isel(time=itime)
        else:
            fn='b.e21.BHIST.f09_g17.CMIP6-historical.011.cam.h1.%s.%s.nc'%(vn,tstr1)
            xvn=xr.open_dataset('%s/%s'%(idir1,fn))[vn].sel(lat=lat,lon=lon,method='nearest').isel(time=itime)
        xvn.to_netcdf(nfn)
        return xvn

gps=load_mycesm('PS')
gplev=hyam*p0+hybm*gps.data
gta=load_mycesm('T')
gts=load_mycesm('TS')
gqrs=load_mycesm('QRS')
gqrl=load_mycesm('QRL')
gdtcond=load_mycesm('DTCOND')
gdtv=load_mycesm('DTV')
gadvt=load_mycesm('ADVT')
gcnvt=load_mycesm('CNVT')
# gdwh=load_mycesm('DWH')
gpw=load_mycesm('PW')
gpblp=load_mycesm('PBLP')
# gttgw=load_mycesm('TTGWORO')
# gfgr12=load_mycesm('FGR12')
gttend=load_mycesm('TTEND')
ghtend=load_mycesm('HTEND')
# gesoit=load_mycesm('ESOIT')

# # infer gadv
# gadvt=c.cpd*(gqrs+gqrl+gdtv)-ghtend-gdwh
# gadvt=c.cpd*(gqrs+gqrl+gdtcond+gdtv-gpw)-gttend-gcnvt

def load_cmip(vn):
    nfn='%s/%s.nc'%(idir0,vn.lower())
    try:
        print('Loading proc data')
        return xr.open_dataarray(nfn)
    except:
        print('Loading raw data')
        idir2='/project/mojave/cmip6/historical/day/%s/CESM2/r11i1p1f1/gn'%vn
        fn='%s_day_CESM2_historical_r11i1p1f1_gn_%s.nc'%(vn,tstr2)
        xvn=xr.open_dataset('%s/%s'%(idir2,fn))[vn].sel(lat=lat,lon=lon,method='nearest').isel(time=itime)
        xvn.to_netcdf(nfn)
        return xvn

grsds=load_cmip('rsds')
grsus=load_cmip('rsus')
ghfls=load_cmip('hfls')
ghfss=load_cmip('hfss')

def load_mycmip(vn):
    nfn='%s/%s.nc'%(idir0,vn.lower())
    try:
        print('Loading proc data')
        return xr.open_dataarray(nfn)
    except:
        print('Loading raw data')
        idir2='/project/amp02/miyawaki/data/share/cmip6/historical/day/%s/CESM2/r11i1p1f1/gn'%vn
        fn='%s_day_CESM2_historical_r11i1p1f1_gn_%s.nc'%(vn,tstr2)
        xvn=xr.open_dataset('%s/%s'%(idir2,fn))[vn].sel(lat=lat,lon=lon,method='nearest').isel(time=itime)
        xvn.to_netcdf(nfn)
        return xvn

# gadvt=load_mycmip('advt')
# gadvt=gadvt.interp(plev=gplev,method='nearest')
# gadvt.data[np.isnan(gadvt.data)]=0
# print(gadvt.data)

# col model params
plev = np.linspace(gplev[-1],gplev[0],50)
ps=gps.data
LH=ghfls.data
SH=ghfss.data
Fs = grsds.data-grsus.data#-gfgr12.data+gesoit.data
print(Fs)
Fa = 100
b = 1
beta = 0.2
n = 2

# Qa=(-gadvt)/c.g
# Qa=(c.cpd*(gqrs+gdtcond+gdtv))/c.g
Qa=(-gadvt+c.cpd*(gqrs+gdtcond+gdtv))/c.g
# Qa=(-gadvt-gcnvt-gttend+c.cpd*(-gpw+gqrs+gdtcond+gdtv))/c.g
# Qa=-gadvt/c.cpd+gqrs+gdtcond+gdtv
Qa['lev']=gplev
Qa=Qa.interp(lev=plev)

# fbl=0.8 # fractional boundary layer depth
fbl=1-n*gpblp.data/ps

tau0s =1
taubl = fbl*tau0s

odir='/project/amp/miyawaki/plots/p004/rae/comp'
if not os.path.exists(odir): os.makedirs(odir)

# RAE + turbulent flux + arbitrary heating
arbT, arbTs, _,_,_ = rae_arb(tau0s, taubl, b, beta, Fs, Qa.data, LH, SH, plev, ps, n)
print(arbT)
print(arbTs)

fig, ax = plt.subplots()
ax.plot(arbTs, 1e-2*ps, '.',color='tab:red')
ax.plot(gts, 1e-2*gps, '.',color='k')
ax.plot(arbT, 1e-2*plev, '-',color='tab:red',label='SCM')
ax.plot(gta, 1e-2*gplev, '-',color='k',label='CESM2')
# ax.plot(np.append(arbTs, arbT), 1e-2*np.append(ps, plev), '-',color='tab:red',label='SCM')
# ax.plot(np.append(gta,gts), 1e-2*np.append(gplev,gps), '-',color='k',label='CESM2')
ax.set_ylim(ax.get_ylim()[::-1]) # invert r1 axis
ax.set_xlabel('T (K)')
ax.set_ylabel('p (hPa)')
fig.set_size_inches(4,3)
plt.tight_layout()
plt.savefig('%s/testcol.png'%odir, format='png', dpi=600)

