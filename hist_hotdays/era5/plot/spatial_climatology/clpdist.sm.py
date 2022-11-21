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

varn='sm'
lse = ['djf'] # season (ann, djf, mam, jja, son)
# yr='1959-2020'
yr='1980-2000'
lpc=[95,99]

for pc in lpc:
    for se in lse:
        
        idir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)
        odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/era5/%s/%s' % (se,varn)

        if not os.path.exists(odir):
            os.makedirs(odir)

        [cldist, gr] = pickle.load(open('%s/cldist.%s.%02d.%s.%s.pickle' % (idir,varn,pc,yr,se), 'rb'))
        # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
        gr['lon'] = np.append(gr['lon'].data,360)
        cldist = np.append(cldist, cldist[:,0][:,None],axis=1)

        [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

        # plot warming ratios
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))

        vlim=30
        clf=ax.contourf(mlon, mlat, cldist, np.arange(-vlim,vlim,5),extend='both', vmax=vlim, vmin=-vlim, transform=ccrs.PlateCarree(), cmap='RdBu_r')
        ax.coastlines()
        ax.set_title(r'%s ERA5 (%s)' % (se.upper(),yr))
        cb=plt.colorbar(clf,location='bottom')
        cb.set_label(r'$SM^{>%s}_\mathrm{2\,m}-\overline{SM}_\mathrm{2\,m}$ (kg m$^{-2}$)' % (pc))
        plt.savefig('%s/cldist.%s.%02d.%s.%s.pdf' % (odir,varn,pc,yr,se), format='pdf', dpi=300)
        plt.close()
