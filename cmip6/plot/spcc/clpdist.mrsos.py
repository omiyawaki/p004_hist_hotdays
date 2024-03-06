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

varn='mrsos'
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
# lse = ['ann','jja'] # season (ann, djf, mam, jja, son)
lse = ['jja'] # season (ann, djf, mam, jja, son)
lfo=['ssp370'] # forcings 
cl='fut-his'
his='1980-2000'
fut='2080-2100'
# lpc = [1,5,50,95,99] # percentile (choose from lpc below)
lpc=[95,99]
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
                    [stats, gr] = pickle.load(open('%s/rdist.%s.%02d.%s.%s.%s.pickle' % (idir,varn,pc,his,fut,se), 'rb'))
                    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                    gr['lon'] = np.append(gr['lon'].data,360)
                    for stat in stats:
                        stats[stat] = np.append(stats[stat], stats[stat][:,0][:,None],axis=1)

                    rdist = stats['mean'] 

                    [stats, gr] = pickle.load(open('%s/ddist.%s.%02d.%s.%s.%s.pickle' % (idir,varn,pc,his,fut,se), 'rb'))
                    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                    gr['lon'] = np.append(gr['lon'].data,360)
                    for stat in stats:
                        stats[stat] = np.append(stats[stat], stats[stat][:,0][:,None],axis=1)

                    ddist = stats['mean'] 
                else:
                    [rdist, gr] = pickle.load(open('%s/rdist.%s.%02d.%s.%s.%s.pickle' % (idir,varn,pc,his,fut,se), 'rb'))
                    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                    gr['lon'] = np.append(gr['lon'].data,360)
                    rdist = np.append(rdist, rdist[:,0][:,None],axis=1)

                    [ddist, gr] = pickle.load(open('%s/ddist.%s.%02d.%s.%s.%s.pickle' % (idir,varn,pc,his,fut,se), 'rb'))
                    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                    gr['lon'] = np.append(gr['lon'].data,360)
                    ddist = np.append(ddist, ddist[:,0][:,None],axis=1)

                [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

                # plot warming differences
                ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
                # transparent colormap
                vlim=2
                clf=ax.contourf(mlon, mlat, ddist, np.arange(-vlim,vlim,vlim/10),extend='both', vmax=vlim, vmin=-vlim, transform=ccrs.PlateCarree(), cmap='RdBu_r')
                ax.coastlines()
                ax.set_title(r'%s %s %s' % (se.upper(),md.upper(),fo.upper()))
                cb=plt.colorbar(clf,location='bottom')
                cb.set_label(r'$(SM^{>%s}_\mathrm{2\,m}-\overline{SM}_\mathrm{2\,m})_\mathrm{%s}-(SM^{>%s}_\mathrm{2\,m}-\overline{SM}_\mathrm{2\,m})_\mathrm{%s}$ (kg m$^{-2}$)' % (pc,fut,pc,his))
                plt.savefig('%s/ddist.%s.%02d.%s.%s.pdf' % (odir,varn,pc,fo,se), format='pdf', dpi=300)
                plt.close()
