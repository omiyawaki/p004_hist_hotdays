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

mk=1.25
varn='t2m'
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
# lse = ['ann','jja'] # season (ann, djf, mam, jja, son)
lse = ['jja'] # season (ann, djf, mam, jja, son)
lfo=['ssp245'] # forcings 
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

                if md=='mmm':
                    [stats, gr] = pickle.load(open('%s/rdist.%s.%02d.%s.%s.%s.pickle' % (idir,varn,pc,his,fut,se), 'rb'))
                    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                    gr['lon'] = np.append(gr['lon'].data,360)
                    for stat in stats:
                        stats[stat] = np.append(stats[stat], stats[stat][:,0][:,None],axis=1)

                    rdist = stats['mean'] 
                else:
                    [rdist, gr] = pickle.load(open('%s/rdist.%02d.%s.%s.%s.pickle' % (idir,pc,his,fut,se), 'rb'))
                    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                    gr['lon'] = np.append(gr['lon'].data,360)
                    rdist = np.append(rdist, rdist[:,0][:,None],axis=1)

                [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

                # plot warming ratios
                for pc in lpc:
                    # save sea (southeast asia) definition (note I'm ignoring the redundant data I added earlier)
                    sea=np.where(np.logical_and.reduce((rdist[:,:-1]>mk, mlat[:,:-1]<40, mlon[:,:-1]<270))) 
                    mgr=[mlat[:,:-1],mlon[:,:-1]]
                    pickle.dump([sea, mgr], open('%s/defsea.%s.%02d.%s.%s.pickle' % (odir,varn,pc,fo,se), 'wb'), protocol=5)	
                    # redo with the redundant data retained
                    sea=np.where(np.logical_and.reduce((rdist>mk, mlat<40, mlon<270)))
                    hlsea=np.zeros_like(rdist)
                    hlsea[sea]=99
                    ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
                    # transparent colormap
                    clf=ax.contourf(mlon, mlat, rdist, np.arange(0.5,1.5,0.05),extend='both', vmax=1.5, vmin=0.5, transform=ccrs.PlateCarree(), cmap='RdBu_r')
                    # ax.contour(mlon, mlat, rdist, [mk], linewidth=1, colors='yellow', transform=ccrs.PlateCarree())
                    # colors = [(0,0.5,0,c) for c in np.linspace(0,1,100)]
                    # Greys_alpha = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)
                    # ax.contourf(mlon, mlat, hlsea, 3, transform=ccrs.PlateCarree(),cmap=Greys_alpha)
                    plt.rcParams['hatch.color']=[1,1,1]
                    # plt.rcParams['hatch.color']=[0,0.5,0]
                    ax.contourf(mlon, mlat, hlsea, 3, hatches=['','/////'], transform=ccrs.PlateCarree(), alpha=0)
                    ax.coastlines()
                    ax.set_title(r'%s %s %s' % (se.upper(),md.upper(),fo.upper()))
                    cb=plt.colorbar(clf,location='bottom')
                    cb.set_label(r'$\frac{(T^{%s}_\mathrm{2\,m}-T^{50}_\mathrm{2\,m})_\mathrm{%s}}{(T^{%s}_\mathrm{2\,m}-T^{50}_\mathrm{2\,m})_\mathrm{%s}}$ (unitless)' % (pc,fut,pc,his))
                    plt.savefig('%s/rdist.mark%g.%s.%02d.%s.%s.pdf' % (odir,mk,varn,pc,fo,se), format='pdf', dpi=300)
                    plt.close()
