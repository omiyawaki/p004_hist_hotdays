import os
import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm

varn='pr'
lse = ['jja'] # season (ann, djf, mam, jja, son)
lpc = [50,95,99] # percentile (choose from lpc below)

for se in lse:
    print(se.upper())
    idir = '/project/amp/miyawaki/data/p004/hist_hotdays/gpcp/%s/%s' % (se,varn)
    odir = '/project/amp/miyawaki/plots/p004/hist_hotdays/gpcp/%s/%s' % (se,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    c = 0
    clima={}
    for ipc in range(len(lpc)):
        pc = lpc[ipc]
        [mpr, gr] = pickle.load(open('%s/clmean_%02d.%s.pickle' % (idir,pc,se), 'rb'))
        # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
        gr['lon'] = np.append(gr['lon'].data,360)
        mpr = np.append(mpr, mpr[:,0][:,None],axis=1)

        clima[str(pc)] = mpr 

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

    # plot climatological pr
    for pc in tqdm(lpc):
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
        # vmax=np.max(slope[str(pc)])
        vmin=0
        if pc==50:
            vmax=10
            vint=1
        elif pc==95:
            vmax=50
            vint=5
        elif pc==99:
            vmax=75
            vint=5
        elif pc in [1,5]:
            vmax=1
            vint=0.1
        clf=ax.contourf(mlon, mlat, clima[str(pc)], np.arange(vmin,vmax+vint,vint),extend='both', vmax=vmax, vmin=vmin, transform=ccrs.PlateCarree(), cmap='Greens')
        # ax.contour(mlon, mlat, clima[str(pc)], 8,colors='gray', transform=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_title(r'%s ERA5 ($1997-2020$)' % (se.upper()))
        cb=plt.colorbar(clf,location='bottom')
        cb.set_label(r'Climatological $P^{%s}_\mathrm{2\,m}$ (mm d$^{-1}$)' % pc)
        plt.savefig('%s/clima_p%02d.%s.pdf' % (odir,pc,se), format='pdf', dpi=300)
        plt.close()

    # # plot diff of wet to average day in climatology
    # for pc in tqdm(lpc):
    #     if pc == 50:
    #         continue
    #     ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
    #     # transparent colormap
    #     if se=='ann':
    #         vmin=-20
    #         vmax=20
    #         if pc>50:
    #             lvs=np.arange(0,vmax+1,1)
    #         else:
    #             lvs=np.arange(vmin,1,1)
    #     else:
    #         vmin=-20
    #         vmax=20
    #         if pc>50:
    #             lvs=np.arange(0,vmax+1,1)
    #         else:
    #             lvs=np.arange(vmin,1,1)
    #     colors = [(0.5,0.5,0.5,c) for c in np.linspace(0,1,100)]
    #     clf=ax.contourf(mlon, mlat, clima[str(pc)]-clima['50'], lvs,extend='both', vmax=vmax, vmin=vmin, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    #     ax.coastlines()
    #     ax.set_title(r'%s ERA5 ($1997-2020$)' % (se.upper()))
    #     cb=plt.colorbar(clf,location='bottom')
    #     cb.set_label(r'$P^{%s}_\mathrm{2\,m}-P^{50}_\mathrm{2\,m}$ (mm d$^{-1}$)' % pc)
    #     plt.savefig('%s/diffP50_clima_p%02d.%s.pdf' % (odir,pc,se), format='pdf', dpi=300)
    #     plt.close()

