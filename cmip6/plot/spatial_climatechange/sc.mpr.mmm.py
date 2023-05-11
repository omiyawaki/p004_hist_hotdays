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
from utils import rainmap

nt=7 # window size in days
varn='pr'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'

md='mmm'

idirt = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,'tas')
idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn)
idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn)
odir1 = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn)
odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

if not os.path.exists(odir1):
    os.makedirs(odir1)

if not os.path.exists(odir):
    os.makedirs(odir)

c = 0
dt={}

# mean temp
vn1,_,gr=pickle.load(open('%s/m%s_%s.%s.nc' % (idir1,varn,his,se),'rb'))
vn2,_,_=pickle.load(open('%s/m%s_%s.%s.nc' % (idir2,varn,fut,se),'rb'))
# repeat 0 deg lon info to 360 deg to prevent a blank line in contour
gr['lon'] = np.append(gr['lon'].data,360)
evn1 = np.append(vn1, vn1[...,0][...,None],axis=2)
evn2 = np.append(vn2, vn2[...,0][...,None],axis=2)

# warming
dvn=evn2-evn1

[mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')

# plot climatology
fig,ax=plt.subplots(nrows=3,ncols=4,subplot_kw={'projection': ccrs.Robinson(central_longitude=240)},figsize=(12,7),constrained_layout=True)
ax=ax.flatten()
fig.suptitle(r'%s %s' % (md.upper(),fo.upper()),fontsize=16)
for m in tqdm(range(12)):
    clf=ax[m].contourf(mlon, mlat, 86400*evn1[m,...], np.arange(0,16+1,1),extend='both', vmax=16, vmin=0, transform=ccrs.PlateCarree(), cmap=rainmap(100))
    ax[m].coastlines()
    ax[m].set_title(r'%s' % (monname(m)),fontsize=16)
    fig.savefig('%s/m%s.%s.pdf' % (odir1,varn,fo), format='pdf', dpi=300)
cb=fig.colorbar(clf,ax=ax.ravel().tolist(),location='bottom',aspect=50)
cb.ax.tick_params(labelsize=16)
cb.set_label(label=r'$\overline{P}$ (mm d$^{-1}$)',size=16)
fig.savefig('%s/m%s.%s.pdf' % (odir1,varn,fo), format='pdf', dpi=300)
