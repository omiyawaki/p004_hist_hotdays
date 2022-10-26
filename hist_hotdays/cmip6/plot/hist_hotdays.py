import os
import sys
sys.path.append('../data/')
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm
from cmip6util import mods

# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
# lse = ['ann','jja'] # season (ann, djf, mam, jja, son)
lse = ['jja'] # season (ann, djf, mam, jja, son)
lfo=['ssp245'] # forcings 
cl='fut-his'
his='1980-2000'
fut='2080-2100'
lpc = [1,5,50,95,99] # percentile (choose from lpc below)
mmm=True # multimodel mean?
# mmm=False # multimodel mean?

for se in lse:
    for fo in lfo:
        if mmm:
            lmd=['mmm']
        else:
            lmd=mods(fo)
        
        for md in lmd:
            idir = '/project2/tas1/miyawaki/projects/000_hotdays/data/cmip6/%s/%s/%s/%s' % (se,cl,fo,md)
            odir = '/project2/tas1/miyawaki/projects/000_hotdays/plots/cmip6/%s/%s/%s/%s' % (se,cl,fo,md)

            if not os.path.exists(odir):
                os.makedirs(odir)

            c = 0
            dt={}
            for ipc in range(len(lpc)):
                pc = lpc[ipc]
                if md=='mmm':
                    [stats, gr] = pickle.load(open('%s/diff_%02d.%s.%s.%s.pickle' % (idir,pc,his,fut,se), 'rb'))
                    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                    gr['lon'] = np.append(gr['lon'].data,360)
                    for stat in stats:
                        stats[stat] = np.append(stats[stat], stats[stat][:,0][:,None],axis=1)

                    dt[str(pc)] = stats['mean'] 
                else:
                    [dt2m, gr] = pickle.load(open('%s/diff_%02d.%s.%s.%s.pickle' % (idir,pc,his,fut,se), 'rb'))
                    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                    gr['lon'] = np.append(gr['lon'].data,360)
                    dt2m = np.append(dt2m, dt2m[:,0][:,None],axis=1)
                    dt[str(pc)]=dt2m

            [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

            # plot spatial map of warming 
            for pc in lpc:
                ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
                # vmax=np.max(dt[str(pc)])
                vmax=5
                clf=ax.contourf(mlon, mlat, dt[str(pc)], np.arange(-vmax,vmax,0.25),extend='both', vmax=vmax, vmin=-vmax, transform=ccrs.PlateCarree(), cmap='RdBu_r')
                ax.coastlines()
                ax.set_title(r'%s %s %s' % (se.upper(),md.upper(), fo.upper()))
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
                clf=ax.contourf(mlon, mlat, dt[str(pc)]/dt['50'], np.arange(0.5,1.5,0.05),extend='both', vmax=1.5, vmin=0.5, transform=ccrs.PlateCarree(), cmap='RdBu_r')
                # plt.rcParams['hatch.color']=[0.5,0.5,0.5]
                # ax.contour(mlon, mlat, dt['50'], 0, linewidth=1, colors='w', transform=ccrs.PlateCarree())
                # ax.contourf(mlon, mlat, cooling, 3, transform=ccrs.PlateCarree(),cmap=Greys_alpha)
                # ax.set_extent([xm.min(), xm.max(), ym.min(), ym.max()])
                ax.coastlines()
                ax.set_title(r'%s %s %s' % (se.upper(),md.upper(),fo.upper()))
                cb=plt.colorbar(clf,location='bottom')
                cb.set_label(r'$\frac{\Delta T^{%s}_\mathrm{2\,m}}{\Delta T^{50}_\mathrm{2\,m}}$ (unitless)' % pc)
                plt.savefig('%s/ratioT50_t%02d.%s.%s.pdf' % (odir,pc,fo,se), format='pdf', dpi=300)
                plt.close()

            # plot warming difference 
            for pc in lpc:
                if pc == 50:
                    continue
                cooling = np.zeros_like(dt['50'])
                cooling[np.where(dt['50']<=0)] = 99
                ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
                # transparent colormap
                colors = [(0.5,0.5,0.5,c) for c in np.linspace(0,1,100)]
                Greys_alpha = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)
                clf=ax.contourf(mlon, mlat, dt[str(pc)]-dt['50'], np.arange(-2.5,2.5,0.25),extend='both', vmax=2.5, vmin=-2.5, transform=ccrs.PlateCarree(), cmap='RdBu_r')
                # plt.rcParams['hatch.color']=[0.5,0.5,0.5]
                # ax.contour(mlon, mlat, dt['50'], 0, linewidth=1, colors='w', transform=ccrs.PlateCarree())
                # ax.contourf(mlon, mlat, cooling, 3, transform=ccrs.PlateCarree(),cmap=Greys_alpha)
                # ax.set_extent([xm.min(), xm.max(), ym.min(), ym.max()])
                ax.coastlines()
                ax.set_title(r'%s %s %s' % (se.upper(),md.upper(),fo.upper()))
                cb=plt.colorbar(clf,location='bottom')
                cb.set_label(r'$\Delta T^{%s}_\mathrm{2\,m}-\Delta T^{50}_\mathrm{2\,m}$ (K)' % pc)
                plt.savefig('%s/diffT50_t%02d.%s.%s.pdf' % (odir,pc,fo,se), format='pdf', dpi=300)
                plt.close()
