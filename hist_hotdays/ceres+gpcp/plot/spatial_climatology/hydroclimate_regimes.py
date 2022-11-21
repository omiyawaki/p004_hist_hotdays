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

varn='hr'
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

    [hr, gr] = pickle.load(open('%s/hr.%s.%s.pickle' % (idir,byr,se), 'rb'))
    hr=lm*hr # mask out ocean
    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    hr = np.append(hr, hr[:,0][:,None],axis=1)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # plot climatological ai1
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
    vmin=-0.5
    vmax=6.5
    # custom colormap
    cm=mcolors.ListedColormap(['indigo','darkgreen','royalblue','limegreen','yellow','orangered','maroon'])
    clf=ax.contourf(mlon, mlat, hr, np.arange(vmin,vmax+1),vmax=vmax, vmin=vmin, transform=ccrs.PlateCarree(), cmap=cm)
    ax.coastlines()
    ax.set_title(r'%s CERES+GPCP ($2000-2022$)' % (se.upper()))
    cb=plt.colorbar(clf,location='bottom')
    cb.set_ticks(np.arange(0,7))
    cb.set_ticklabels(['CH','TH','H','SH','SA','A','HA'])
    cb.ax.tick_params(length=0)
    cb.set_label(r'Hydroclimate Regime')
    plt.savefig('%s/clima.%s.%s.pdf' % (odir,byr,se), format='pdf', dpi=300)
    plt.close()
