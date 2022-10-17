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
lse = ['ann'] # season (ann, djf, mam, jja, son)
lfo=['ssp245'] # forcings 
cl='fut' # his, fut
if cl=='his':
    yr='1980-2000'
elif cl=='fut':
    yr='2080-2100'
# lpc = [1,5,50,95,99] # percentile (choose from lpc below)
lpc=[95]
# mmm=True # multimodel mean?
mmm=False # multimodel mean?

for pc in lpc:
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
                if md=='mmm':
                    [stats, gr] = pickle.load(open('%s/cldist.%s.%02d.%s.%s.pickle' % (idir,varn,pc,yr,se), 'rb'))
                    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                    gr['lon'] = np.append(gr['lon'].data,360)
                    for stat in stats:
                        stats[stat] = np.append(stats[stat], stats[stat][:,0][:,None],axis=1)

                    cldist = stats['mean'] 
                else:
                    [cldist, gr] = pickle.load(open('%s/cldist.%02d.%s.%s.pickle' % (idir,pc,yr,se), 'rb'))
                    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                    gr['lon'] = np.append(gr['lon'].data,360)
                    cldist = np.append(cldist, cldist[:,0][:,None],axis=1)

                [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

                # plot warming ratios
                for pc in lpc:
                    ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))

                    vlim=20
                    if pc<50:
                        vmin=-vlim
                        vmax=0+1
                    else:
                        vmin=0
                        vmax=vlim+1
                    clf=ax.contourf(mlon, mlat, cldist, np.arange(vmin,vmax,1),extend='both', vmax=vlim, vmin=-vlim, transform=ccrs.PlateCarree(), cmap='RdBu_r')
                    ax.coastlines()
                    ax.set_title(r'%s %s %s (%s)' % (se.upper(),md.upper(),fo.upper(),yr))
                    cb=plt.colorbar(clf,location='bottom')
                    cb.set_label(r'$T^{%s}_\mathrm{2\,m}-T^{50}_\mathrm{2\,m}$ (K)' % (pc))
                    plt.savefig('%s/cldist.%s.%02d.%s.%s.pdf' % (odir,varn,pc,fo,se), format='pdf', dpi=300)
                    plt.close()
