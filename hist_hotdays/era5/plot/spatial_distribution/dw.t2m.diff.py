import os
import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm

varn='t2m'
lse = ['jja'] # season (ann, djf, mam, jja, son)
lpc=[95]
by0=[1950,1970]
by1=[2000,2020]

for se in lse:
    idir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)
    odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/era5/%s/%s' % (se,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    c = 0
    for ipc in range(len(lpc)):
        pc = lpc[ipc]
        [dw0, gr] = pickle.load(open('%s/clmean.dw%s.%02d.%g.%g.%s.pickle' % (idir,varn,pc,by0[0],by0[1],se), 'rb'))
        [dw1, gr] = pickle.load(open('%s/clmean.dw%s.%02d.%g.%g.%s.pickle' % (idir,varn,pc,by1[0],by1[1],se), 'rb'))
        # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
        gr['lon'] = np.append(gr['lon'].data,360)
        dw0 = np.append(dw0, dw0[:,0][:,None],axis=1)
        dw1 = np.append(dw1, dw1[:,0][:,None],axis=1)

        [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

        ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
        clf=ax.contourf(mlon, mlat, dw1/dw0, np.arange(0.5,1.5,0.05),extend='both', vmax=1.5, vmin=0.5, transform=ccrs.PlateCarree(), cmap='RdBu_r')
        ax.coastlines()
        ax.set_title(r'%s ERA5' % (se.upper()))
        cb=plt.colorbar(clf,location='bottom')
        cb.set_label(r'$\frac{(T^{%s}_\mathrm{2\,m}-T^{50}_\mathrm{2\,m})_{%g-%g}}{(T^{%s}_\mathrm{2\,m}-T^{50}_\mathrm{2\,m})_{%g-%g}}$ (unitless)' % (pc,by1[1],by1[0],pc,by0[1],by0[0]))
        plt.savefig('%s/rdist_t%02d.%s.pdf' % (odir,pc,se), format='pdf', dpi=300)
        plt.close()

