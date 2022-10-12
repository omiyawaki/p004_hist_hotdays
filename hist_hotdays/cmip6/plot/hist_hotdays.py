import os
import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm

lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
lfo=['ghg','aaer','bmb','ee','xaaer'] # forcings 
# lfo=['ghg'] # forcings 
cl='fut-his'
his='1920-1940'
fut='2030-2050'
lpc = [1,5,50,95,99] # percentile (choose from lpc below)

for se in lse:
    for fo in lfo:
        idir = '/project/amp/miyawaki/data/p004/hist_hotdays/cesm2-sf/%s/%s/%s' % (se,cl,fo)
        odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/cesm2-sf/%s/%s/%s' % (se,cl,fo)

        if not os.path.exists(odir):
            os.makedirs(odir)

        c = 0
        dt={}
        for ipc in range(len(lpc)):
            pc = lpc[ipc]
            [stats, gr] = pickle.load(open('%s/diff_%02d.%s.%s.%s.pickle' % (idir,pc,his,fut,se), 'rb'))
            # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
            gr['lon'] = np.append(gr['lon'].data,360)
            for stat in stats:
                stats[stat] = np.append(stats[stat], stats[stat][:,0][:,None],axis=1)

            dt[str(pc)] = stats['mean'] 

        [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

        # plot spatial map of warming 
        for pc in lpc:
            ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
            # vmax=np.max(dt[str(pc)])
            vmax=5
            clf=ax.contourf(mlon, mlat, dt[str(pc)], np.arange(-vmax,vmax,0.25),extend='both', vmax=vmax, vmin=-vmax, transform=ccrs.PlateCarree(), cmap='RdBu_r')
            ax.coastlines()
            ax.set_title(r'%s CESM2-SF %s' % (se.upper(),fo.upper()))
            cb=plt.colorbar(clf,location='bottom')
            cb.set_label(r'$\Delta T^{%s}_\mathrm{2\,m}$ (K)' % pc)
            plt.savefig('%s/warming_t%02d.%s.%s.pdf' % (odir,pc,fo,se), format='pdf', dpi=300)
            plt.close()

        # plot warming ratios
        for pc in lpc:
            if pc == 50:
                continue
            cooling = np.zeros_like(dt['50'])
            cooling[np.where(dt['50']<=0)] = 99
            ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
            # transparent colormap
            colors = [(0.5,0.5,0.5,c) for c in np.linspace(0,1,100)]
            Greys_alpha = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)
            clf=ax.contourf(mlon, mlat, dt[str(pc)]/dt['50'], np.arange(0,2,0.1),extend='both', vmax=2, vmin=-0, transform=ccrs.PlateCarree(), cmap='RdBu_r')
            plt.rcParams['hatch.color']=[0.5,0.5,0.5]
            ax.contour(mlon, mlat, dt[str(pc)], 0, linewidth=1, colors='w', transform=ccrs.PlateCarree())
            # ax.contourf(mlon, mlat, cooling, 3, transform=ccrs.PlateCarree(),cmap=Greys_alpha)
            ax.coastlines()
            ax.set_title(r'%s CESM2-SF %s' % (se.upper(),fo.upper()))
            cb=plt.colorbar(clf,location='bottom')
            cb.set_label(r'$\frac{\Delta T^{%s}_\mathrm{2\,m}}{\Delta T^{50}_\mathrm{2\,m}}$ (unitless)' % pc)
            plt.savefig('%s/ratioT50_t%02d.%s.%s.pdf' % (odir,pc,fo,se), format='pdf', dpi=300)
            plt.close()

