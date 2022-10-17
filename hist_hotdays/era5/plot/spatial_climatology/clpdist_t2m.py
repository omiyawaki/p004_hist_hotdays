import os
import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm

varn='t2m'
# lse = ['jja','ann','djf','mam','son'] # season (ann, djf, mam, jja, son)
lse = ['jja'] # season (ann, djf, mam, jja, son)
lpc = [1,5,95,99] # percentile (choose from lpc below)

for se in lse:
    print(se.upper())
    idir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)
    odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/era5/%s/%s' % (se,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    for ipc in range(len(lpc)):
        pc = lpc[ipc]
        [dt2m, gr] = pickle.load(open('%s/cldist.%02d.%s.pickle' % (idir,pc,se), 'rb'))
        # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
        gr['lon'] = np.append(gr['lon'].data,360)
        dt2m = np.append(dt2m, dt2m[:,0][:,None],axis=1)

        [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

        # plot climatological distance from 95th to 50th prc
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
        # vmax=np.max(slope[str(pc)])
        vlim=20
        if pc < 50:
            vmin=-vlim
            vmax=0+1
        else:
            vmin=0
            vmax=vlim+1
        clf=ax.contourf(mlon, mlat, dt2m, np.arange(vmin,vmax,1),extend='both', vmax=vlim, vmin=-vlim, transform=ccrs.PlateCarree(), cmap='RdBu_r')
        ax.coastlines()
        ax.set_title(r'%s ERA5 ($1950-2020$)' % (se.upper()))
        cb=plt.colorbar(clf,location='bottom')
        cb.set_label(r'Climatological $T^{%s}_\mathrm{2\,m} - T^{50}_\mathrm{2\,m}$  (K)' % pc)
        plt.savefig('%s/cldist_t%02d.%s.pdf' % (odir,pc,se), format='pdf', dpi=300)
        plt.close()

