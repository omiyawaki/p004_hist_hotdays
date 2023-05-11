import os
import sys
sys.path.append('../../data/')
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm

varn='t2m'
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
# lse = ['ann','jja'] # season (ann, djf, mam, jja, son)
lse = ['jja'] # season (ann, djf, mam, jja, son)
lfo=['lens'] # forcings 
cl='fut-his'
his='1980-2000'
fut='2080-2100'
# lpc = [1,5,50,95,99] # percentile (choose from lpc below)
lpc=[95]
mmm=True # multimodel mean?
# mmm=False # multimodel mean?

for pc in lpc:
    for se in lse:
        for fo in lfo:

            idir = '/project/amp/miyawaki/data/p004/hist_hotdays/cesm2-sf/%s/%s/%s' % (se,cl,fo)
            odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/cesm2-sf/%s/%s/%s/%s/%s_%s' % (se,cl,fo,varn,his,fut)
            
            if not os.path.exists(odir):
                os.makedirs(odir)

            c = 0
            dt={}
            [stats, gr] = pickle.load(open('%s/rdist.%s.%02d.%s.%s.%s.pickle' % (idir,varn,pc,his,fut,se), 'rb'))
            # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
            gr['lon'] = np.append(gr['lon'].data,360)
            for stat in stats:
                stats[stat] = np.append(stats[stat], stats[stat][:,0][:,None],axis=1)

            rdist = stats['mean'] 

            [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

            # plot warming ratios
            for pc in lpc:
                ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
                # transparent colormap
                clf=ax.contourf(mlon, mlat, rdist, np.arange(0.5,1.5,0.05),extend='both', vmax=1.5, vmin=0.5, transform=ccrs.PlateCarree(), cmap='RdBu_r')
                ax.coastlines()
                ax.set_title(r'%s CESM2 %s' % (se.upper(),fo.upper()))
                cb=plt.colorbar(clf,location='bottom')
                cb.set_label(r'$\frac{(T^{%s}_\mathrm{2\,m}-T^{50}_\mathrm{2\,m})_\mathrm{%s}}{(T^{%s}_\mathrm{2\,m}-T^{50}_\mathrm{2\,m})_\mathrm{%s}}$ (unitless)' % (pc,fut,pc,his))
                plt.savefig('%s/rdist.%s.%02d.%s.%s.pdf' % (odir,varn,pc,fo,se), format='pdf', dpi=300)
                plt.close()
