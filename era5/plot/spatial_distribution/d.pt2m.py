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
lse = ['djf'] # season (ann, djf, mam, jja, son)
pc=95


def loadslope(pc,y0,y1):
    [t2m, gr] = pickle.load(open('%s/clmean.dw%s.%02d.%g.%g.%s.pickle' % (idir,varn,pc,y0,y1,se), 'rb'))
    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    t2m = np.append(t2m, t2m[:,0][:,None],axis=1)
    return gr,t2m

for se in lse:
    idir = '/project/amp/miyawaki/data/p004/era5/%s/%s' % (se,varn)
    odir = '/project/amp/miyawaki/plots/p004/era5/%s/%s' % (se,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)


    gr,dt1=loadslope(pc,1950,1970)
    _,dt2=loadslope(pc,2000,2020)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # plot trends
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
    # vmax=np.max(slope[str(pc)])
    vmax=1
    clf=ax.contourf(mlon, mlat, dt2-dt1, np.arange(-vmax,vmax,0.1),extend='both', vmax=vmax, vmin=-vmax, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    # ax.contourf(mlon, mlat, sig[str(pc)], 3, hatches=['','....'], alpha=0, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(r'%s ERA5 ($1950-2020$)' % (se.upper()))
    cb=plt.colorbar(clf,location='bottom')
    cb.set_label(r'$\Delta T^{%s}_\mathrm{2\,m} -\Delta T^{50}_\mathrm{2\,m}$ (K)' % pc)
    plt.savefig('%s/diff.t%02d.t%02d.%s.pdf' % (odir,pc,50,se), format='pdf', dpi=300)
    plt.close()

