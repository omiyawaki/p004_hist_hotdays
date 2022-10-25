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
from cmip6util import mods

varn='t2m'
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
# lse = ['ann','jja'] # season (ann, djf, mam, jja, son)
lse = ['jja'] # season (ann, djf, mam, jja, son)
lfo=['ssp245'] # forcings 
cl='fut-his'
his='1980-2000'
fut='2080-2100'
# lpc = [1,5,50,95,99] # percentile (choose from lpc below)
lpc=[1,5,95,99]

for pc in lpc:
    for se in lse:
        for fo in lfo:
            odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/cmip6/%s/%s/%s/%s' % (se,cl,fo,'mi')
            if not os.path.exists(odir):
                os.makedirs(odir)

            lmd=mods(fo)
            
            for pc in lpc:
                fig,ax=plt.subplots(nrows=4,ncols=5,subplot_kw={'projection':ccrs.Robinson(central_longitude=240)},figsize=(11,8.5))
                ax=ax.flatten()
                fig.suptitle(r'%s %s' % (se.upper(),fo.upper()))
                for imd in range(len(lmd)):
                    md=lmd[imd]
                    idir = '/project/amp/miyawaki/data/p004/hist_hotdays/cmip6/%s/%s/%s/%s' % (se,cl,fo,md)

                    c = 0
                    dt={}

                    [rdist, gr] = pickle.load(open('%s/rdist.%02d.%s.%s.%s.pickle' % (idir,pc,his,fut,se), 'rb'))
                    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                    gr['lon'] = np.append(gr['lon'].data,360)
                    rdist = np.append(rdist, rdist[:,0][:,None],axis=1)

                    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

                    # plot warming ratios
                    clf=ax[imd].contourf(mlon, mlat, rdist, np.arange(0.5,1.5,0.05),extend='both', vmax=1.5, vmin=0.5, transform=ccrs.PlateCarree(), cmap='RdBu_r')
                    ax[imd].coastlines()
                    ax[imd].set_title(r'%s' % (md.upper()))
                    plt.savefig('%s/rdist.%s.%02d.%s.%s.pdf' % (odir,varn,pc,fo,se), format='pdf', dpi=300)
                cb=fig.colorbar(clf,ax=ax,location='bottom')
                cb.set_label(r'$\frac{(T^{%s}_\mathrm{2\,m}-T^{50}_\mathrm{2\,m})_\mathrm{%s}}{(T^{%s}_\mathrm{2\,m}-T^{50}_\mathrm{2\,m})_\mathrm{%s}}$ (unitless)' % (pc,fut,pc,his))
                plt.savefig('%s/rdist.%s.%02d.%s.%s.pdf' % (odir,varn,pc,fo,se), format='pdf', dpi=128)
                plt.close()
