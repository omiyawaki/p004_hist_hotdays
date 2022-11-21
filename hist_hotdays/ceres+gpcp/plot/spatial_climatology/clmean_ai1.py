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

varn='ai1'
# lse = ['jja'] # season (ann, djf, mam, jja, son)
lse = ['ann','jja','djf','mam','son'] # season (ann, djf, mam, jja, son)
byr='2000-2022'

# load cesm land fraction data
ds=xr.open_dataset('/project/amp/miyawaki/data/share/lfrac/sftlf_fx_CESM2_historical_r1i1p1f1_gn.nc')
lf=ds['sftlf']/100
lm=np.heaviside(lf-0.5,1).data # land mask
lm[lm==0]=np.nan

for se in lse:
    idir = '/project/amp/miyawaki/data/p004/hist_hotdays/ceres+gpcp/%s/%s' % (se,varn)
    odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/ceres+gpcp/%s/%s' % (se,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    c = 0

    [mai1, gr] = pickle.load(open('%s/clmean.mai1.%s.%s.pickle' % (idir,byr,se), 'rb'))
    mai1=lm*mai1 # mask out ocean
    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    mai1 = np.append(mai1, mai1[:,0][:,None],axis=1)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # plot climatological ai1
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
    vmin=-0.5
    vmax=6
    clf=ax.contourf(mlon, mlat, mai1, [0.5,1,1.5,2,3,4,5,6],extend='both', vmax=vmax, vmin=vmin, transform=ccrs.PlateCarree(), cmap='terrain')
    # ax.contour(mlon, mlat, mai1, [0.7,1.2,2.0,4.0,6.0],colors='gray', transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(r'%s CERES+GPCP ($2000-2022$)' % (se.upper()))
    cb=plt.colorbar(clf,location='bottom')
    cb.set_label(r'Climatological $\frac{R_{\mathrm{net}}}{LP}$ (unitless)')
    plt.savefig('%s/clima.%s.%s.pdf' % (odir,byr,se), format='pdf', dpi=300)
    plt.close()
