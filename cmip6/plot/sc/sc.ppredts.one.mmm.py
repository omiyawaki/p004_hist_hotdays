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
varn='predts'
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

# mean temp
[vn1,svn1,gr]=pickle.load(open('%s/m%s_%s.%s.doy.pickle' % (idir1,varn,his,se),'rb'))
[vn2,svn2,_]=pickle.load(open('%s/m%s_%s.%s.doy.pickle' % (idir2,varn,fut,se),'rb'))
vn1=vn1[...,iloc[0],iloc[1]]
vn2=vn2[...,iloc[0],iloc[1]]
svn1=svn1[...,iloc[0],iloc[1]]
svn2=svn2[...,iloc[0],iloc[1]]

# prc temp
[pvn1,spvn1,_]=pickle.load(open('%s/p%s_%s.%s.doy.pickle' % (idir1,varn,his,se),'rb'))
[pvn2,spvn2,_]=pickle.load(open('%s/p%s_%s.%s.doy.pickle' % (idir2,varn,fut,se),'rb'))
pvn1=pvn1[...,iloc[0],iloc[1]]
pvn2=pvn2[...,iloc[0],iloc[1]]
spvn1=spvn1[...,iloc[0],iloc[1]]
spvn2=spvn2[...,iloc[0],iloc[1]]

# warming
dvn=vn2-vn1
dpvn=pvn2-pvn1
ddpvn=dpvn-dvn[:,None]
sdvn=np.sqrt(1/2*(svn1**2+svn2**2))
sdpvn=np.sqrt(1/2*(spvn1**2+spvn2**2))
sddpvn=np.sqrt(1/2*(sdvn[:,None]**2+sdpvn**2))

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
ax.fill_between(doy,pvn1[:,-2]-spvn1[:,-2],pvn1[:,-2]+spvn1[:,-2],color='tab:orange',alpha=0.2,edgecolor=None)
ax.fill_between(doy,pvn1[:,-1]-spvn1[:,-1],pvn1[:,-1]+spvn1[:,-1],color='tab:red',alpha=0.2,edgecolor=None)
ax.plot(doy,vn1,'k',label='Mean')
ax.plot(doy,pvn1[:,-2],'tab:orange',label='95th')
ax.plot(doy,pvn1[:,-1],'tab:red',label='99th')
ax.set_xlim([doy[0],doy[-1]])
ax.set_xticks(dmnmp)
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
# ax.set_xlabel('Day of Year')
ax.set_ylabel('$T_{q=q_s}$ (K)')
ax.set_title(r'%s %s' '\n' r'[%+05.1f,%+05.1f]' % (md.upper(),fo1.upper(),gr['lat'][iloc[0]],gr['lon'][iloc[1]]))
plt.legend()
fig.savefig('%s/p%s.%s.pdf' % (odir,varn,fo1), format='pdf', dpi=300)
plt.close()

# plot warming seasonal cycle
fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
[ax.axvline(_dmn,color='k',linewidth=0.1) for _dmn in dmn]
ax.fill_between(doy,dvn-sdvn,dvn+sdvn,color='k',alpha=0.2,edgecolor=None)
ax.fill_between(doy,dpvn[:,-2]-sdpvn[:,-2],dpvn[:,-2]+sdpvn[:,-2],color='tab:orange',alpha=0.2,edgecolor=None)
ax.fill_between(doy,dpvn[:,-1]-sdpvn[:,-1],dpvn[:,-1]+sdpvn[:,-1],color='tab:red',alpha=0.2,edgecolor=None)
ax.plot(doy,dvn,'k',label='Mean')
ax.plot(doy,dpvn[:,-2],'tab:orange',label='95th')
ax.plot(doy,dpvn[:,-1],'tab:red',label='99th')
ax.set_xlim([doy[0],doy[-1]])
ax.set_xticks(dmnmp)
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
# ax.set_xlabel('Day of Year')
ax.set_ylabel('$\Delta T_{q=q_s}$ (K)')
ax.set_title(r'%s %s' '\n' r'[%+05.1f,%+05.1f]' % (md.upper(),fo1.upper(),gr['lat'][iloc[0]],gr['lon'][iloc[1]]))
plt.legend(frameon=False)
fig.savefig('%s/dp%s.%s.pdf' % (odir,varn,fo1), format='pdf', dpi=300)
plt.close()
