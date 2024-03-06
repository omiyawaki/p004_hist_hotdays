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

varn='tas'
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
# lse = ['ann','jja'] # season (ann, djf, mam, jja, son)
lse = ['jja'] # season (ann, djf, mam, jja, son)
lfo=['ssp370'] # forcings 
cl='fut-his'
his='1980-2000'
fut='2080-2100'
# lpc = [1,5,50,95,99] # percentile (choose from lpc below)
lpc=[0,95,99]
mmm=True # multimodel mean?
# mmm=False # multimodel mean?

for pc in lpc:
    for se in lse:
        for fo in lfo:
            if mmm:
                lmd=['mmm']
            else:
                lmd=mods(fo)
            
            for md in lmd:
                idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
                odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

                if not os.path.exists(odir):
                    os.makedirs(odir)

                c = 0
                dt={}

                if md=='mmm':
                    [stats, gr] = pickle.load(open('%s/diff_%02d.%s.%s.%s.pickle' % (idir,pc,his,fut,se), 'rb'))
                    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                    gr['lon'] = np.append(gr['lon'].data,360)
                    for stat in stats:
                        stats[stat] = np.append(stats[stat], stats[stat][:,0][:,None],axis=1)

                    diff = stats['mean'] 
                    stdev = stats['stdev'] 
                else:
                    [diff, gr] = pickle.load(open('%s/diff_%02d.%s.%s.%s.pickle' % (idir,pc,his,fut,se), 'rb'))
                    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                    gr['lon'] = np.append(gr['lon'].data,360)
                    diff = np.append(diff, diff[:,0][:,None],axis=1)

                [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

                if md=='mmm':
                    # plot warming differences
                    ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
                    # transparent colormap
                    vlim=2.5
                    clf=ax.contourf(mlon, mlat, stdev, np.arange(0,vlim,0.25),extend='both', vmax=vlim, vmin=0, transform=ccrs.PlateCarree())
                    ax.coastlines()
                    ax.set_title(r'%s %s %s' % (se.upper(),md.upper(),fo.upper()))
                    cb=plt.colorbar(clf,location='bottom')
                    if pc==0:
                        cb.set_label(r'$\sigma(\Delta \overline{T}_\mathrm{2\,m})$ (K)')
                    else:
                        cb.set_label(r'$\sigma(\Delta T^{>%s}_\mathrm{2\,m})$ (K)' % (pc))
                    plt.savefig('%s/stdev.%s.%02d.%s.%s.pdf' % (odir,varn,pc,fo,se), format='pdf', dpi=300)
                    plt.close()

                # plot warming differences
                ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
                # transparent colormap
                vlim=10
                clf=ax.contourf(mlon, mlat, diff, np.arange(-vlim,vlim,1),extend='both', vmax=vlim, vmin=-vlim, transform=ccrs.PlateCarree(), cmap='RdBu_r')
                ax.coastlines()
                ax.set_title(r'%s %s %s' % (se.upper(),md.upper(),fo.upper()))
                cb=plt.colorbar(clf,location='bottom')
                if pc==0:
                    cb.set_label(r'$\Delta \overline{T}_\mathrm{2\,m}$ (K)')
                else:
                    cb.set_label(r'$\Delta T^{>%s}_\mathrm{2\,m}$ (K)' % (pc))
                plt.savefig('%s/diff.%s.%02d.%s.%s.pdf' % (odir,varn,pc,fo,se), format='pdf', dpi=300)
                plt.close()

