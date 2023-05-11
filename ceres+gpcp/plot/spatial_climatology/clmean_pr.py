import os
import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm

varn='pr'
# lse = ['jja'] # season (ann, djf, mam, jja, son)
lse = ['ann','jja','djf','mam','son'] # season (ann, djf, mam, jja, son)
byr='2000-2022'

for se in lse:
    idir = '/project/amp/miyawaki/data/p004/hist_hotdays/ceres+gpcp/%s/%s' % (se,varn)
    odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/ceres+gpcp/%s/%s' % (se,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    c = 0

    [mpr, gr] = pickle.load(open('%s/clmean.pr.%s.%s.pickle' % (idir,byr,se), 'rb'))
    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    mpr = np.append(mpr, mpr[:,0][:,None],axis=1)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # plot climatological pr
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
    vmin=0
    vmax=10
    vint=1
    clf=ax.contourf(mlon, mlat, mpr*86400, np.arange(vmin,vmax+vint,vint),extend='both', vmax=vmax, vmin=vmin, transform=ccrs.PlateCarree(), cmap='Greens')
    # clf=ax.contourf(mlon, mlat, mpr, transform=ccrs.PlateCarree())
    # ax.contour(mlon, mlat, mpr, [0.7,1.2,2.0,4.0,6.0],colors='gray', transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(r'%s GPCP ($2000-2022$)' % (se.upper()))
    cb=plt.colorbar(clf,location='bottom')
    cb.set_label(r'Climatological $LP$ (mm d$^{-1}$)')
    plt.savefig('%s/clima.%s.%s.pdf' % (odir,byr,se), format='pdf', dpi=300)
    plt.close()
