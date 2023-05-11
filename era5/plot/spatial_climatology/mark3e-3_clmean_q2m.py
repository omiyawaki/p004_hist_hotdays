import os
import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm

mk=3.3e-3
varn='q2m'
lse = ['jja'] # season (ann, djf, mam, jja, son)
lpc = [5] # percentile (choose from lpc below)

for se in lse:
    print(se.upper())
    idir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)
    odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/era5/%s/%s' % (se,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    c = 0
    clima={}
    for ipc in range(len(lpc)):
        pc = lpc[ipc]
        [mq2m, gr] = pickle.load(open('%s/clmean_%02d.%s.pickle' % (idir,pc,se), 'rb'))
        # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
        gr['lon'] = np.append(gr['lon'].data,360)
        mq2m = np.append(mq2m, mq2m[:,0][:,None],axis=1)

        clima[str(pc)] = mq2m 

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # plot climatological q2m
    for pc in tqdm(lpc):
        # save swus (southwest usa) definition  (ignore redundant 360 deg data)
        swus=np.where(np.logical_and.reduce((clima[str(pc)][:,:-1]<mk,mlat[:,:-1]>0,mlat[:,:-1]<50,mlon[:,:-1]>240,mlon[:,:-1]<300)))
        mgr=[mlat[:,:-1],mlon[:,:-1]]
        pickle.dump([swus,mgr],open('%s/defswus.%s.%02d.%s.pickle'%(odir,varn,pc,se),'wb'),protocol=5)
        # redo with redundant data
        swus=np.where(np.logical_and.reduce((clima[str(pc)]<mk,mlat>0,mlat<50,mlon>240,mlon<300)))
        hlswus=np.zeros_like(clima[str(pc)])
        hlswus[swus]=99

        ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
        # vmax=np.max(slope[str(pc)])
        vmin=1e-3
        vmax=3e-2
        clf=ax.contourf(mlon, mlat, clima[str(pc)], np.arange(vmin,vmax,1e-3),extend='both', vmax=vmax, vmin=vmin, transform=ccrs.PlateCarree(), cmap='gist_earth_r')
        plt.rcParams['hatch.color']=[0.5,0,0]
        ax.contourf(mlon, mlat, hlswus, 3,hatches=['','/////'],alpha=0,transform=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_title(r'%s ERA5 ($1950-2020$)' % (se.upper()))
        cb=plt.colorbar(clf,location='bottom')
        cb.set_label(r'Climatological $q^{%s}_\mathrm{2\,m}$ (kg kg$^{-1}$)' % pc)
        plt.savefig('%s/clima.%s.mark%s.%02d.%s.pdf' % (odir,varn,mk,pc,se), format='pdf', dpi=300)
        plt.close()

