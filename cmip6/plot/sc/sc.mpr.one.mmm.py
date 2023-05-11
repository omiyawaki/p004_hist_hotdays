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

iloc=[110,85] # SEA

nt=7 # window size in days
varn='pr'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'

md='mmm'

idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn)
idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn)

c = 0
dt={}

# mean
[vn1,svn1,gr]=pickle.load(open('%s/m%s_%s.%s.doy.pickle' % (idir1,varn,his,se),'rb'))
[vn2,svn2,_]=pickle.load(open('%s/m%s_%s.%s.doy.pickle' % (idir2,varn,fut,se),'rb'))
vn1=vn1[...,iloc[0],iloc[1]]
vn2=vn2[...,iloc[0],iloc[1]]
svn1=svn1[...,iloc[0],iloc[1]]
svn2=svn2[...,iloc[0],iloc[1]]
# convert to mm/d
vn1=86400*vn1
vn2=86400*vn2
svn1=86400*svn1
svn2=86400*svn2

# warming
dvn=vn2-vn1
dsvn=np.sqrt(1/2*(svn1**2+svn2**2))

odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,md,varn,gr['lat'][iloc[0]],gr['lon'][iloc[1]])

if not os.path.exists(odir):
    os.makedirs(odir)

doy=np.arange(vn1.shape[0])+1
dmn=np.arange(0,len(doy),np.ceil(len(doy)/12))
dmnmp=np.arange(0,len(doy),np.ceil(len(doy)/12))+np.ceil(len(doy)/24)

# plot historical temp seasonal cycle
fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
[ax.axvline(_dmn,color='k',linewidth=0.1) for _dmn in dmn]
ax.fill_between(doy,vn1-svn1,vn1+svn1,color='k',alpha=0.2,edgecolor=None)
ax.plot(doy,vn1,'k',label='Mean')
ax.set_xlim([doy[0],doy[-1]])
ax.set_xticks(dmnmp)
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
# ax.set_xlabel('Day of Year')
ax.set_ylabel('$P$ (mm d$^{-1}$)')
ax.set_title(r'%s %s' '\n' r'[%+05.1f,%+05.1f]' % (md.upper(),fo1.upper(),gr['lat'][iloc[0]],gr['lon'][iloc[1]]))
plt.legend(frameon=False)
fig.savefig('%s/p%s.%s.pdf' % (odir,varn,fo1), format='pdf', dpi=300)
plt.close()

# plot warming seasonal cycle
fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
[ax.axvline(_dmn,color='k',linewidth=0.1) for _dmn in dmn]
ax.fill_between(doy,dvn-dsvn,dvn+dsvn,color='k',alpha=0.2,edgecolor=None)
ax.plot(doy,dvn,'k',label='Mean')
ax.set_xlim([doy[0],doy[-1]])
ax.set_xticks(dmnmp)
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
# ax.set_xlabel('Day of Year')
ax.set_ylabel('$\Delta P$ (mm d$^{-1}$)')
ax.set_title(r'%s %s' '\n' r'[%+05.1f,%+05.1f]' % (md.upper(),fo1.upper(),gr['lat'][iloc[0]],gr['lon'][iloc[1]]))
plt.legend(frameon=False)
fig.savefig('%s/dp%s.%s.pdf' % (odir,varn,fo1), format='pdf', dpi=300)
plt.close()
