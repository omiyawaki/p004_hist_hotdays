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
pc=95

se='winter'
odir = '/project/amp/miyawaki/plots/p004/era5/%s/%s' % (se,varn)

def getslope(se):
    idir = '/project/amp/miyawaki/data/p004/era5/%s/%s' % (se,varn)
    [stats, gr] = pickle.load(open('%s/regress.dw.%s.%02d.%s.pickle' % (idir,varn,pc,se), 'rb'))
    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
    gr['lon'] = np.append(gr['lon'].data,360)
    for stat in stats:
        stats[stat] = np.append(stats[stat], stats[stat][:,0][:,None],axis=1)
    slope = 10*stats['slope'] # convert to warming per decade
    return slope,gr

if not os.path.exists(odir):
    os.makedirs(odir)

sjja,gr=getslope('jja')
sdjf,_=getslope('djf')
slope=np.copy(sjja)
if se=='summer':
    slope[gr['lat']>0]=sjja[gr['lat']>0]
    slope[gr['lat']<=0]=sdjf[gr['lat']<=0]
elif se=='winter':
    slope[gr['lat']>=0]=sdjf[gr['lat']>0]
    slope[gr['lat']<0]=sjja[gr['lat']<0]

[mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

# plot trends
ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
# vmax=np.max(slope[str(pc)])
vmax=0.3
clf=ax.contourf(mlon, mlat, slope, np.arange(-vmax,vmax,0.05),extend='both', vmax=vmax, vmin=-vmax, transform=ccrs.PlateCarree(), cmap='RdBu_r')
# ax.contourf(mlon, mlat, sig[str(pc)], 3, hatches=['','....'], alpha=0, transform=ccrs.PlateCarree())
ax.coastlines()
ax.set_title(r'%s ERA5 ($1950-2020$)' % (se.upper()))
cb=plt.colorbar(clf,location='bottom')
cb.set_label(r'$T^{%s}_\mathrm{2\,m}-T^{50}_\mathrm{2\,m}$ Trend (K dec$^{-1}$)' % pc)
plt.savefig('%s/diffT50_t%02d.%s.trend.pdf' % (odir,pc,se), format='pdf', dpi=300)
plt.close()

