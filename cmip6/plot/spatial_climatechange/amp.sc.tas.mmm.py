import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm
from cmip6util import mods
from utils import monname

nt=7 # window size in days
varn='tas'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'
skip507599=True

md='mmm'

idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

if not os.path.exists(odir):
    os.makedirs(odir)

# warming
dvn,sdvn,gr=pickle.load(open('%s/d%s_%s_%s.%s.nc' % (idir,varn,his,fut,se), 'rb'))	
dpvn,sdpvn,_=pickle.load(open('%s/dp%s_%s_%s.%s.nc' % (idir,varn,his,fut,se), 'rb'))	
ddpvn,sddpvn,_=pickle.load(open('%s/ddp%s_%s_%s.%s.nc' % (idir,varn,his,fut,se), 'rb'))	
# repeat 0 deg lon info to 360 deg to prevent a blank line in contour
gr['lon'] = np.append(gr['lon'].data,360)
dvn = np.append(dvn, dvn[...,0][...,None],axis=2)
dpvn = np.append(dpvn, dpvn[...,0][...,None],axis=3)
ddpvn = np.append(ddpvn, ddpvn[...,0][...,None],axis=3)
sdvn = np.append(sdvn, sdvn[...,0][...,None],axis=2)
sdpvn = np.append(sdpvn, sdpvn[...,0][...,None],axis=3)
sddpvn = np.append(sddpvn, sddpvn[...,0][...,None],axis=3)

addpvn = np.max(ddpvn,axis=0)-np.min(ddpvn,axis=0)

[mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

# plot pct warming - mean warming
for i,p in enumerate(gr['pct']):
    if p==95:
        fig,ax=plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(5,4),constrained_layout=True)
        ax.set_title(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
        clf=ax.contourf(mlon, mlat, addpvn[i,...], np.arange(0,1.5+0.1,0.1),extend='both', vmax=1.5, vmin=0, transform=ccrs.PlateCarree())
        ax.coastlines()
        cb=fig.colorbar(clf,location='bottom',aspect=50)
        cb.ax.tick_params(labelsize=12)
        cb.set_label(label=r'Seasonal amplitude of $\Delta \delta T^{%02d}_\mathrm{2\,m}$ (K)'%(p),size=16)
        fig.savefig('%s/amp.ddp%02d%s.%s.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)

