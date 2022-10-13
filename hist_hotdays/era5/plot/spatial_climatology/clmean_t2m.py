import os
import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm

lse = ['jja','ann','djf','mam','son'] # season (ann, djf, mam, jja, son)
lpc = [1,5,50,95,99] # percentile (choose from lpc below)

for se in lse:
    print(se.upper())
    idir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s' % (se)
    odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/era5/%s' % (se)

    if not os.path.exists(odir):
        os.makedirs(odir)

    c = 0
    clima={}
    for ipc in range(len(lpc)):
        pc = lpc[ipc]
        [mt2m, gr] = pickle.load(open('%s/clmean_%02d.%s.pickle' % (idir,pc,se), 'rb'))
        # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
        gr['lon'] = np.append(gr['lon'].data,360)
        mt2m = np.append(mt2m, mt2m[:,0][:,None],axis=1)

        clima[str(pc)] = mt2m 

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # plot climatological t2m
    for pc in tqdm(lpc):
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
        # vmax=np.max(slope[str(pc)])
        vmin=220
        vmax=320
        clf=ax.contourf(mlon, mlat, clima[str(pc)], np.arange(vmin,vmax,5),extend='both', vmax=vmax, vmin=vmin, transform=ccrs.PlateCarree(), cmap='RdBu_r')
        ax.contour(mlon, mlat, clima[str(pc)], 273.15,colors='gray', transform=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_title(r'%s ERA5 ($1950-2020$)' % (se.upper()))
        cb=plt.colorbar(clf,location='bottom')
        cb.set_label(r'Climatological $T^{%s}_\mathrm{2\,m}$ (K)' % pc)
        plt.savefig('%s/clima_t%02d.%s.pdf' % (odir,pc,se), format='pdf', dpi=300)
        plt.close()

    # plot ratios of hot to average day in climatology
    for pc in tqdm(lpc):
        if pc == 50:
            continue
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
        # transparent colormap
        if se=='ann':
            vmin=0.8
            vmax=1.2
            if pc>50:
                lvs=np.arange(1,vmax,0.01)
            else:
                lvs=np.arange(vmin,1,0.01)
        else:
            vmin=0.95
            vmax=1.05
            if pc>50:
                lvs=np.arange(1,vmax,1e-3)
            else:
                lvs=np.arange(vmin,1,1e-3)
        colors = [(0.5,0.5,0.5,c) for c in np.linspace(0,1,100)]
        clf=ax.contourf(mlon, mlat, clima[str(pc)]/clima['50'], lvs,extend='both', vmax=vmax, vmin=vmin, transform=ccrs.PlateCarree(), cmap='RdBu_r')
        ax.coastlines()
        ax.set_title(r'%s ERA5 ($1950-2020$)' % (se.upper()))
        cb=plt.colorbar(clf,location='bottom')
        cb.set_label(r'$\frac{T^{%s}_\mathrm{2\,m}}{T^{50}_\mathrm{2\,m}}$ (unitless)' % pc)
        plt.savefig('%s/ratioT50_clima_t%02d.%s.pdf' % (odir,pc,se), format='pdf', dpi=300)
        plt.close()

