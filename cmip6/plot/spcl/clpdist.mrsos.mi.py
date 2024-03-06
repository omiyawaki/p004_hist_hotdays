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
# lfo=['ssp370'] # forcings 
lfo=['historical'] # forcings 
if cl=='his':
    yr='1980-2000'
elif cl=='fut':
    yr='2080-2100'
# lpc = [1,5,50,95,99] # percentile (choose from lpc below)
lpc=[95,99]

for pc in lpc:
    for se in lse:
        for fo in lfo:
            odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mi',varn)
            if not os.path.exists(odir):
                os.makedirs(odir)

            lmd=mods(fo)
            
            for pc in lpc:
                fig,ax=plt.subplots(nrows=4,ncols=5,subplot_kw={'projection':ccrs.Robinson(central_longitude=240)},figsize=(11,8.5))
                ax=ax.flatten()
                fig.suptitle(r'%s %s' % (se.upper(),fo.upper()))
                for imd in range(len(lmd)):
                    md=lmd[imd]
                    idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

                    c = 0
                    dt={}

                    [cldist, gr] = pickle.load(open('%s/cldist.%s.%02d.%s.%s.pickle' % (idir,varn,pc,yr,se), 'rb'))
                    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                    gr['lon'] = np.append(gr['lon'].data,360)
                    cldist = np.append(cldist, cldist[:,0][:,None],axis=1)

                    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

                    # plot warming ratios
                    vlim=30
                    clf=ax[imd].contourf(mlon, mlat, cldist, np.arange(-vlim,vlim,5),extend='both', vmax=vlim, vmin=-vlim, transform=ccrs.PlateCarree(), cmap='RdBu_r')
                    ax[imd].coastlines()
                    ax[imd].set_title(r'%s' % (md.upper()))
                    plt.savefig('%s/cldist.%s.%02d.%s.%s.%s.pdf' % (odir,varn,pc,fo,yr,se), format='pdf', dpi=128)
                cb=fig.colorbar(clf,ax=ax,location='bottom')
                cb.set_label(r'$SM^{>%s}_\mathrm{2\,m}-\overline{SM}_\mathrm{2\,m}$ (kg m$^{-2}$)' % (pc))
                plt.savefig('%s/cldist.%s.%02d.%s.%s.%s.pdf' % (odir,varn,pc,fo,yr,se), format='pdf', dpi=128)
                plt.close()
