import os
import sys
sys.path.append('/home/miyawaki/scripts/common')
sys.path.append('/home/miyawaki/scripts/p004/scripts/cmip6/data')
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tqdm import tqdm
from cmip6util import mods

md='mi'
varn1='ef'
varn2='ef'
varn='%s+%s'%(varn1,varn2)
lse = ['jja'] # season (ann, djf, mam, jja, son)

fo='ssp370'
cl='fut-his'
his='1980-2000'
fut='2080-2100'
lpc=[0,95,99]

for pc in lpc:
    for se in lse:

        idir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,'his','historical','CESM2','hd')
        [_,gr]=pickle.load(open('%s/cd%s_%s.%g.%s.pickle' % (idir,'hd',his,95,se),'rb'))
        la=gr['lat']
        lo=gr['lon']

        idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)

        odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/global' % (se,cl,fo,md,varn)

        if not os.path.exists(odir):
            os.makedirs(odir)

        ev=pickle.load(open('%s/ev.dpred.%s.act.%s.%s.%g.%s.pickle' % (idir,varn,his,fut,pc,se),'rb'))

        [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

        # exvar map (ALL)
        mtd='ALL'
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
        clf=ax.contourf(mlon, mlat, ev[0,...], np.arange(0,1,0.1),extend='both', vmax=1, vmin=0, transform=ccrs.PlateCarree())
        ax.coastlines()
        if pc==0:
            ax.set_title(r'$\Delta \overline{EF}$ Intermodel Variance Explained by' '\n' r'%s' '\n' '%s %s' % (mtd,se.upper(),fo.upper()))
        else:
            ax.set_title(r'$\Delta EF^{%s}$ Intermodel Variance Explained by' '\n' r'%s' '\n' '%s %s' % (pc,mtd,se.upper(),fo.upper()))
        cb=plt.colorbar(clf,location='bottom')
        cb.set_label(r'$R^2$')
        plt.savefig('%s/ev.act.dpred.%s.%g.%s.all.pdf' % (odir,varn,pc,se), format='pdf', dpi=300)
        plt.close()

        # exvar map (DBC)
        mtd='$\Delta BC$'
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
        clf=ax.contourf(mlon, mlat, ev[1,...], np.arange(0,1,0.1),extend='both', vmax=1, vmin=0, transform=ccrs.PlateCarree())
        ax.coastlines()
        if pc==0:
            ax.set_title(r'$\Delta \overline{EF}$ Intermodel Variance Explained by' '\n' r'%s' '\n' '%s %s' % (mtd,se.upper(),fo.upper()))
        else:
            ax.set_title(r'$\Delta EF^{%s}$ Intermodel Variance Explained by' '\n' r'%s' '\n' '%s %s' % (pc,mtd,se.upper(),fo.upper()))
        cb=plt.colorbar(clf,location='bottom')
        cb.set_label(r'$R^2$')
        plt.savefig('%s/ev.act.dpred.%s.%g.%s.dbc.pdf' % (odir,varn,pc,se), format='pdf', dpi=300)
        plt.close()

        # exvar map (CBC)
        mtd='$BC_H$'
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
        clf=ax.contourf(mlon, mlat, ev[2,...], np.arange(0,1,0.1),extend='both', vmax=1, vmin=0, transform=ccrs.PlateCarree())
        ax.coastlines()
        if pc==0:
            ax.set_title(r'$\Delta \overline{EF}$ Intermodel Variance Explained by' '\n' r'%s' '\n' '%s %s' % (mtd,se.upper(),fo.upper()))
        else:
            ax.set_title(r'$\Delta EF^{%s}$ Intermodel Variance Explained by' '\n' r'%s' '\n' '%s %s' % (pc,mtd,se.upper(),fo.upper()))
        cb=plt.colorbar(clf,location='bottom')
        cb.set_label(r'$R^2$')
        plt.savefig('%s/ev.act.dpred.%s.%g.%s.cbc.pdf' % (odir,varn,pc,se), format='pdf', dpi=300)
        plt.close()

        # exvar map (DSM)
        mtd='$\Delta SM$'
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
        clf=ax.contourf(mlon, mlat, ev[3,...], np.arange(0,1,0.1),extend='both', vmax=1, vmin=0, transform=ccrs.PlateCarree())
        ax.coastlines()
        if pc==0:
            ax.set_title(r'$\Delta \overline{EF}$ Intermodel Variance Explained by' '\n' r'%s' '\n' '%s %s' % (mtd,se.upper(),fo.upper()))
        else:
            ax.set_title(r'$\Delta EF^{%s}$ Intermodel Variance Explained by' '\n' r'%s' '\n' '%s %s' % (pc,mtd,se.upper(),fo.upper()))
        cb=plt.colorbar(clf,location='bottom')
        cb.set_label(r'$R^2$')
        plt.savefig('%s/ev.act.dpred.%s.%g.%s.dsm.pdf' % (odir,varn,pc,se), format='pdf', dpi=300)
        plt.close()

        # exvar map (CSM)
        mtd='$SM_H$'
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=240))
        clf=ax.contourf(mlon, mlat, ev[4,...], np.arange(0,1,0.1),extend='both', vmax=1, vmin=0, transform=ccrs.PlateCarree())
        ax.coastlines()
        if pc==0:
            ax.set_title(r'$\Delta \overline{EF}$ Intermodel Variance Explained by' '\n' r'%s' '\n' '%s %s' % (mtd,se.upper(),fo.upper()))
        else:
            ax.set_title(r'$\Delta EF^{%s}$ Intermodel Variance Explained by' '\n' r'%s' '\n' '%s %s' % (pc,mtd,se.upper(),fo.upper()))
        cb=plt.colorbar(clf,location='bottom')
        cb.set_label(r'$R^2$')
        plt.savefig('%s/ev.act.dpred.%s.%g.%s.csm.pdf' % (odir,varn,pc,se), format='pdf', dpi=300)
        plt.close()

