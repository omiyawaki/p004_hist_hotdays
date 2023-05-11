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
p=95
pref1='ddp'
varn1='tas'
pref2='ddp'
varn2='hfls'
varn='%s%s+%s%s'%(pref1,varn1,pref2,varn2)
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'
ann=False

md='mi'

idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mmm',varn1)
odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

if not os.path.exists(odir):
    os.makedirs(odir)

# correlation
_,_,gr=pickle.load(open('%s/d%s_%s_%s.%s.pickle' % (idir1,'tas',his,fut,se), 'rb'))	
r=pickle.load(open('%s/ms.rsq.%s_%s_%s.%s.pickle' % (idir,varn,his,fut,se), 'rb'))	
# repeat 0 deg lon info to 360 deg to prevent a blank line in contour
gr['lon'] = np.append(gr['lon'].data,360)
if ann:
    r=r[-1,...]
    r=np.append(r,r[...,0][...,None],axis=1)
else:
    r=r[:,-1,...]
    r=np.append(r,r[...,0][...,None],axis=2)
rsq=r**2

[mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

# plot rsq (pct warming - mean warming)
fig,ax=plt.subplots(nrows=3,ncols=4,subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(12,7),constrained_layout=True)
ax=ax.flatten()
fig.suptitle(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
for m in tqdm(range(12)):
    clf=ax[m].contourf(mlon, mlat, rsq[m,...], np.arange(0,1+0.1,0.1), vmax=1, vmin=0, transform=ccrs.PlateCarree())
    ax[m].coastlines()
    ax[m].set_title(r'%s' % (monname(m)),fontsize=16)
    fig.savefig('%s/ms.rsq%02d%s.%s.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)
cb=fig.colorbar(clf,ax=ax.ravel().tolist(),location='bottom',aspect=50)
cb.ax.tick_params(labelsize=16)
cb.set_label(label=r'$R^2(\Delta\delta T^{%02d}, \Delta\delta LH^{%02d})$'%(p,p),size=16)
fig.savefig('%s/ms.rsq%02d%s.%s.pdf' % (odir,p,varn,fo), format='pdf', dpi=300)
