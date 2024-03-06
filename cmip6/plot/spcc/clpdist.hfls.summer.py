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
se1='jja' # NH summer
se2='djf' # SH summer
se='%s+%s'%(se1,se2)
lfo=['ssp370'] # forcings 
cl='fut-his'
his='1980-2000'
fut='2080-2100'
# lpc = [1,5,50,95,99] # percentile (choose from lpc below)
lpc=[95,99]
mmm=True # multimodel mean?
# mmm=False # multimodel mean?

for pc in lpc:
    for fo in lfo:
        if mmm:
            lmd=['mmm']
        else:
            lmd=mods(fo)
        
        for md in lmd:
            idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se1,cl,fo,md,varn)
            idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se2,cl,fo,md,varn)
            odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

            if not os.path.exists(odir):
                os.makedirs(odir)

            c = 0
            dt={}

            if md=='mmm':
                # [stats1, gr] = pickle.load(open('%s/rdist.%s.%02d.%s.%s.%s.pickle' % (idir1,varn,pc,his,fut,se1), 'rb'))
                # [stats2, _] = pickle.load(open('%s/rdist.%s.%02d.%s.%s.%s.pickle' % (idir2,varn,pc,his,fut,se2), 'rb'))
                # # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                # gr['lon'] = np.append(gr['lon'].data,360)
                # for stat in stats1:
                #     stats1[stat] = np.append(stats1[stat], stats1[stat][:,0][:,None],axis=1)
                #     stats2[stat] = np.append(stats2[stat], stats2[stat][:,0][:,None],axis=1)

                # rdist1 = stats1['mean'] 
                # rstdev1 = stats1['stdev'] 
                # rdist2 = stats2['mean'] 
                # rstdev2 = stats2['stdev'] 

                [stats1, gr] = pickle.load(open('%s/ddist.%s.%02d.%s.%s.%s.pickle' % (idir1,varn,pc,his,fut,se1), 'rb'))
                [stats2, _] = pickle.load(open('%s/ddist.%s.%02d.%s.%s.%s.pickle' % (idir2,varn,pc,his,fut,se2), 'rb'))
                # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                gr['lon'] = np.append(gr['lon'].data,360)
                for stat in stats1:
                    stats1[stat] = np.append(stats1[stat], stats1[stat][:,0][:,None],axis=1)
                    stats2[stat] = np.append(stats2[stat], stats2[stat][:,0][:,None],axis=1)

                ddist1 = stats1['mean'] 
                dstdev1 = stats1['stdev'] 
                ddist2 = stats2['mean'] 
                dstdev2 = stats2['stdev'] 

            else:
                [rdist1, gr] = pickle.load(open('%s/rdist.%s.%02d.%s.%s.%s.pickle' % (idir1,varn,pc,his,fut,se1), 'rb'))
                [rdist2, _] = pickle.load(open('%s/rdist.%s.%02d.%s.%s.%s.pickle' % (idir2,varn,pc,his,fut,se2), 'rb'))
                # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                gr['lon'] = np.append(gr['lon'].data,360)
                rdist1 = np.append(rdist1, rdist1[:,0][:,None],axis=1)
                rdist2 = np.append(rdist2, rdist2[:,0][:,None],axis=1)

                [ddist1, gr] = pickle.load(open('%s/ddist.%s.%02d.%s.%s.%s.pickle' % (idir1,varn,pc,his,fut,se1), 'rb'))
                [ddist2, gr] = pickle.load(open('%s/ddist.%s.%02d.%s.%s.%s.pickle' % (idir2,varn,pc,his,fut,se2), 'rb'))
                # repeat 0 deg lon info to 360 deg to prevent a blank line in contour
                gr['lon'] = np.append(gr['lon'].data,360)
                ddist1 = np.append(ddist1, ddist1[:,0][:,None],axis=1)
                ddist2 = np.append(ddist2, ddist2[:,0][:,None],axis=1)

            [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

            # merge hemisphere data
            rdist=np.copy(rdist2)
            np.copyto(rdist,rdist1,where=mlat>=0)
            ddist=np.copy(ddist2)
            np.copyto(ddist,ddist1,where=mlat>=0)

            if md=='mmm':
                rstdev=np.copy(rstdev2)
                np.copyto(rstdev,rstdev1,where=mlat>=0)
                dstdev=np.copy(dstdev2)
                np.copyto(dstdev,dstdev1,where=mlat>=0)

                # plot warming ratios
                ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
                # transparent colormap
                vmax=0.5
                clf=ax.contourf(mlon, mlat, rstdev, np.arange(0,vmax,0.05),extend='both', vmax=vmax, vmin=0, transform=ccrs.PlateCarree(), cmap='viridis')
                ax.coastlines()
                ax.set_title(r'%s %s %s' % (se.upper(),md.upper(),fo.upper()))
                cb=plt.colorbar(clf,location='bottom')
                cb.set_label(r'$\sigma\left(\frac{(T^{>%s}_\mathrm{2\,m}-\overline{T}_\mathrm{2\,m})_\mathrm{%s}}{(T^{>%s}_\mathrm{2\,m}-\overline{T}_\mathrm{2\,m})_\mathrm{%s}}\right)$ (unitless)' % (pc,fut,pc,his))
                plt.savefig('%s/rstdev.%s.%02d.%s.%s.pdf' % (odir,varn,pc,fo,se), format='pdf', dpi=300)
                plt.close()

                # plot warming differences
                ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
                # transparent colormap
                vmax=1.5
                clf=ax.contourf(mlon, mlat, dstdev, np.arange(0,vmax,0.1),extend='both', vmax=vmax, vmin=0, transform=ccrs.PlateCarree(), cmap='viridis')
                ax.coastlines()
                ax.set_title(r'%s %s %s' % (se.upper(),md.upper(),fo.upper()))
                cb=plt.colorbar(clf,location='bottom')
                cb.set_label(r'$\sigma((T^{>%s}_\mathrm{2\,m}-\overline{T}_\mathrm{2\,m})_\mathrm{%s}-(T^{>%s}_\mathrm{2\,m}-\overline{T}_\mathrm{2\,m})_\mathrm{%s})$ (K)' % (pc,fut,pc,his))
                plt.savefig('%s/dstdev.%s.%02d.%s.%s.pdf' % (odir,varn,pc,fo,se), format='pdf', dpi=300)
                plt.close()

            # plot warming ratios
            ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
            # transparent colormap
            clf=ax.contourf(mlon, mlat, rdist, np.arange(0.5,1.5,0.05),extend='both', vmax=1.5, vmin=0.5, transform=ccrs.PlateCarree(), cmap='RdBu_r')
            ax.coastlines()
            ax.set_title(r'%s %s %s' % (se.upper(),md.upper(),fo.upper()))
            cb=plt.colorbar(clf,location='bottom')
            cb.set_label(r'$\frac{(T^{>%s}_\mathrm{2\,m}-\overline{T}_\mathrm{2\,m})_\mathrm{%s}}{(T^{>%s}_\mathrm{2\,m}-\overline{T}_\mathrm{2\,m})_\mathrm{%s}}$ (unitless)' % (pc,fut,pc,his))
            plt.savefig('%s/rdist.%s.%02d.%s.%s.pdf' % (odir,varn,pc,fo,se), format='pdf', dpi=300)
            plt.close()

            # plot warming differences
            ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
            # transparent colormap
            clf=ax.contourf(mlon, mlat, ddist, np.arange(-2.5,2.5,0.25),extend='both', vmax=2.5, vmin=-2.5, transform=ccrs.PlateCarree(), cmap='RdBu_r')
            ax.coastlines()
            ax.set_title(r'%s %s %s' % (se.upper(),md.upper(),fo.upper()))
            cb=plt.colorbar(clf,location='bottom')
            cb.set_label(r'$(T^{>%s}_\mathrm{2\,m}-\overline{T}_\mathrm{2\,m})_\mathrm{%s}-(T^{>%s}_\mathrm{2\,m}-\overline{T}_\mathrm{2\,m})_\mathrm{%s}$ (K)' % (pc,fut,pc,his))
            plt.savefig('%s/ddist.%s.%02d.%s.%s.pdf' % (odir,varn,pc,fo,se), format='pdf', dpi=300)
            plt.close()
