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

varn='hfls'
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
                    rstdev = stats['stdev'] 

                    [stats, gr] = pickle.load(open('%s/ddist.%s.%02d.%s.%s.%s.pickle' % (idir,varn,pc,his,fut,se), 'rb'))
                    # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                    gr['lon'] = np.append(gr['lon'].data,360)
                    for stat in stats:
                        stats[stat] = np.append(stats[stat], stats[stat][:,0][:,None],axis=1)

                    ddist = stats['mean'] 
                    dstdev = stats['stdev'] 
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

                if md=='mmm':
                    # # plot warming ratios
                    # ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
                    # # transparent colormap
                    # vmax=0.5
                    # clf=ax.contourf(mlon, mlat, rstdev, np.arange(0,vmax,0.05),extend='both', vmax=vmax, vmin=0, transform=ccrs.PlateCarree(), cmap='viridis')
                    # ax.coastlines()
                    # ax.set_title(r'%s %s %s' % (se.upper(),md.upper(),fo.upper()))
                    # cb=plt.colorbar(clf,location='bottom')
                    # cb.set_label(r'$\sigma\left(\frac{(LH^{>%s}_\mathrm{2\,m}-\overline{LH}_\mathrm{2\,m})_\mathrm{%s}}{(LH^{>%s}_\mathrm{2\,m}-\overline{LH}_\mathrm{2\,m})_\mathrm{%s}}\right)$ (unitless)' % (pc,fut,pc,his))
                    # plt.savefig('%s/rstdev.%s.%02d.%s.%s.pdf' % (odir,varn,pc,fo,se), format='pdf', dpi=300)
                    # plt.close()

                    # plot warming differences
                    ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
                    # transparent colormap
                    vmax=20
                    clf=ax.contourf(mlon, mlat, dstdev, np.arange(0,vmax+vmax/10,vmax/10),extend='both', vmax=vmax, vmin=0, transform=ccrs.PlateCarree(), cmap='viridis')
                    ax.coastlines()
                    ax.set_title(r'%s %s %s' % (se.upper(),md.upper(),fo.upper()))
                    cb=plt.colorbar(clf,location='bottom')
                    cb.set_label(r'$\sigma((LH^{>%s}_\mathrm{2\,m}-\overline{LH}_\mathrm{2\,m})_\mathrm{%s}-(LH^{>%s}_\mathrm{2\,m}-\overline{LH}_\mathrm{2\,m})_\mathrm{%s})$ (W m$^{-2}$)' % (pc,fut,pc,his))
                    plt.savefig('%s/dstdev.%s.%02d.%s.%s.pdf' % (odir,varn,pc,fo,se), format='pdf', dpi=300)
                    plt.close()

                # # plot warming ratios
                # ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
                # # transparent colormap
                # clf=ax.contourf(mlon, mlat, rdist, np.arange(0.5,1.5,0.05),extend='both', vmax=1.5, vmin=0.5, transform=ccrs.PlateCarree(), cmap='RdBu_r')
                # ax.coastlines()
                # ax.set_title(r'%s %s %s' % (se.upper(),md.upper(),fo.upper()))
                # cb=plt.colorbar(clf,location='bottom')
                # cb.set_label(r'$\frac{(LH^{>%s}_\mathrm{2\,m}-\overline{LH}_\mathrm{2\,m})_\mathrm{%s}}{(LH^{>%s}_\mathrm{2\,m}-\overline{LH}_\mathrm{2\,m})_\mathrm{%s}}$ (unitless)' % (pc,fut,pc,his))
                # plt.savefig('%s/rdist.%s.%02d.%s.%s.pdf' % (odir,varn,pc,fo,se), format='pdf', dpi=300)
                # plt.close()

                # plot warming differences
                ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
                # transparent colormap
                vmax=20
                clf=ax.contourf(mlon, mlat, ddist, np.arange(-vmax,vmax+vmax/10,vmax/10),extend='both', vmax=vmax, vmin=-vmax, transform=ccrs.PlateCarree(), cmap='RdBu_r')
                ax.coastlines()
                ax.set_title(r'%s %s %s' % (se.upper(),md.upper(),fo.upper()))
                cb=plt.colorbar(clf,location='bottom')
                cb.set_label(r'$(LH^{>%s}_\mathrm{2\,m}-\overline{LH}_\mathrm{2\,m})_\mathrm{%s}-(LH^{>%s}_\mathrm{2\,m}-\overline{LH}_\mathrm{2\,m})_\mathrm{%s}$ (W m$^{-2}$)' % (pc,fut,pc,his))
                plt.savefig('%s/ddist.%s.%02d.%s.%s.pdf' % (odir,varn,pc,fo,se), format='pdf', dpi=300)
                plt.close()
