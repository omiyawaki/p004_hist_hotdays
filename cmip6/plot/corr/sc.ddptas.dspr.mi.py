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
pref1='ddp'
varn1='tas'
pref2='dsp'
varn2='pr'
varn='%s%s+%s%s'%(pref1,varn1,pref2,varn2)
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'
skip5075=True

lmd=mods(fo1)

idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mi',varn)
idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mmm',varn1)
odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mi',varn)

if not os.path.exists(odir):
    os.makedirs(odir)

# correlation
_,_,gr=pickle.load(open('%s/d%s_%s_%s.%s.pickle' % (idir1,'tas',his,fut,se), 'rb'))	
r=pickle.load(open('%s/sc.rsq.%s_%s_%s.%s.pickle' % (idir,varn,his,fut,se), 'rb'))	

# repeat 0 deg lon info to 360 deg to prevent a blank line in contour
gr['lon'] = np.append(gr['lon'].data,360)
r=np.append(r,r[...,0][...,None],axis=3)
rsq=r**2
print(rsq.shape)

[mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

# plot rsq (pct warming - mean warming)
for i,p in enumerate(gr['pct']):
    if p in [50,75,99]:
        continue
    fig,ax=plt.subplots(nrows=4,ncols=5,subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(12,8),constrained_layout=True)
    ax=ax.flatten()
    fig.suptitle('%s'%fo.upper())
    for imd in range(len(lmd)):
        md=lmd[imd]
        ax[imd].set_title(r'%s' % (md.upper()))
        clf=ax[imd].contourf(mlon, mlat, rsq[imd,i,...], np.arange(0,1+0.1,0.1), vmax=1, vmin=0, transform=ccrs.PlateCarree())
        ax[imd].coastlines()
        fig.savefig('%s/sc.rsq%02d%s.%s.mi.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)
    cb=fig.colorbar(clf,ax=ax,location='bottom',aspect=50)
    cb.set_label(label=r'$R^2(\Delta \delta T^{%02d}_\mathrm{2\,m},\delta P^{%02d})$'%(p,p))
    fig.savefig('%s/sc.rsq%02d%s.%s.mi.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)
