import os
import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm

lse = ['ann','djf','mam','son','jja'] # season (ann, djf, mam, jja, son)
lpc = [1,5,50,95,99] # percentile (choose from lpc below)

for se in lse:
    idir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s' % (se)
    odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/era5/%s' % (se)

    if not os.path.exists(odir):
        os.makedirs(odir)

    c = 0
    slope={}
    sig={}
    for ipc in range(len(lpc)):
        pc = lpc[ipc]
        [stats, gr] = pickle.load(open('%s/regress_%02d.%s.pickle' % (idir,pc,se), 'rb'))
        # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
        gr['lon'] = np.append(gr['lon'].data,360)
        for stat in stats:
            stats[stat] = np.append(stats[stat], stats[stat][:,0][:,None],axis=1)

        slope[str(pc)] = 10*stats['slope'] # convert to warming per decade
        sig[str(pc)] = np.zeros_like(slope[str(pc)])
        sig[str(pc)][np.where(stats['pvalue'] > 0.05)] = 99

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # plot trends
    for pc in lpc:
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
        # vmax=np.max(slope[str(pc)])
        vmax=0.75
        clf=ax.contourf(mlon, mlat, slope[str(pc)], np.arange(-vmax,vmax,0.05),extend='both', vmax=vmax, vmin=-vmax, transform=ccrs.PlateCarree(), cmap='RdBu_r')
        ax.contourf(mlon, mlat, sig[str(pc)], 3, hatches=['','....'], alpha=0, transform=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_title(r'%s ERA5 ($1950-2020$)' % (se.upper()))
        cb=plt.colorbar(clf,location='bottom')
        cb.set_label(r'$T^{%s}_\mathrm{2\,m}$ Trend (K dec$^{-1}$)' % pc)
        plt.savefig('%s/trend_t%02d.%s.pdf' % (odir,pc,se), format='pdf', dpi=300)
        plt.close()

    # plot trend ratios
    for pc in lpc:
        if pc == 50:
            continue
        cooling = np.zeros_like(slope['50'])
        cooling[np.where(slope['50']<=0)] = 99
        sigoverlap = np.zeros_like(sig[str(pc)])
        sigoverlap[np.where(np.logical_or(sig[str(pc)]>0, sig['50']>0))] = 99
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
        # transparent colormap
        colors = [(0.5,0.5,0.5,c) for c in np.linspace(0,1,100)]
        Greys_alpha = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)
        clf=ax.contourf(mlon, mlat, slope[str(pc)]/slope['50'], np.arange(0,2,0.1),extend='both', vmax=2, vmin=-0, transform=ccrs.PlateCarree(), cmap='RdBu_r')
        plt.rcParams['hatch.color']=[0.5,0.5,0.5]
        ax.contourf(mlon, mlat, sigoverlap, 3, hatches=['','....'], alpha=0, transform=ccrs.PlateCarree())
        ax.contour(mlon, mlat, slope['50'],0, linewidths=1, colors='w', transform=ccrs.PlateCarree())
        # ax.contourf(mlon, mlat, cooling, 3, transform=ccrs.PlateCarree(),cmap=Greys_alpha)
        ax.coastlines()
        ax.set_title(r'%s ERA5 ($1950-2020$)' % (se.upper()))
        cb=plt.colorbar(clf,location='bottom')
        cb.set_label(r'$\frac{T^{%s}_\mathrm{2\,m}\mathrm{\,Trend}}{T^{50}_\mathrm{2\,m}\mathrm{\,Trend}}$ (unitless)' % pc)
        plt.savefig('%s/ratioT50_t%02d.%s.pdf' % (odir,pc,se), format='pdf', dpi=300)
        plt.close()

