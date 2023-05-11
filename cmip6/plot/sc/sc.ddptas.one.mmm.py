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
from regions import pointlocs

lre=['zambia','amazon','sahara','sea']

nt=7 # window size in days
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

md='mmm'

idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn2)

c = 0
dt={}

# temp
[rvn1,rsvn1,gr]=pickle.load(open('%s/%s%s_%s_%s.%s.pickle' % (idir1,pref1,varn1,his,fut,se),'rb'))

# hfls
[rvn2,rsvn2,_]=pickle.load(open('%s/%s%s_%s_%s.%s.pickle' % (idir2,pref2,varn2,his,fut,se),'rb'))

# # prc temp
# [pvn1,spvn1,_]=pickle.load(open('%s/p%s_%s.%s.doy.pickle' % (idir1,varn,his,se),'rb'))
# [pvn2,spvn2,_]=pickle.load(open('%s/p%s_%s.%s.doy.pickle' % (idir2,varn,fut,se),'rb'))
# pvn1=pvn1[...,iloc[0],iloc[1]]
# pvn2=pvn2[...,iloc[0],iloc[1]]
# spvn1=spvn1[...,iloc[0],iloc[1]]
# spvn2=spvn2[...,iloc[0],iloc[1]]

for re in lre:
    print(re)
    iloc=pointlocs(re)
    la=gr['lat'][iloc[0]]
    lo=gr['lon'][iloc[1]]

    vn1=rvn1[...,iloc[0],iloc[1]]
    svn1=rsvn1[...,iloc[0],iloc[1]]
    vn1=vn1[:,-2]
    svn1=svn1[:,-2]

    vn2=rvn2[...,iloc[0],iloc[1]]
    svn2=rsvn2[...,iloc[0],iloc[1]]
    vn2=vn2[:,-2]
    svn2=svn2[:,-2]

    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn,re)

    if not os.path.exists(odir):
        os.makedirs(odir)

    doy=np.arange(vn1.shape[0])+1

    # doy=np.arange(vn1.shape[0])+1
    # dmn=np.arange(0,len(doy),np.ceil(len(doy)/12))
    # dmnmp=np.arange(0,len(doy),np.ceil(len(doy)/12))+np.ceil(len(doy)/24)

    # plot seasonal cycle of ddptas and ddphfls
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    # [ax.axvline(_dmn,color='k',linewidth=0.1) for _dmn in dmn]
    ax.axhline(0,linewidth=0.5,color='k')
    ax.fill_between(doy,vn1-svn1,vn1+svn1,color='k',alpha=0.2,edgecolor=None)
    ax.plot(doy,vn1,'k')
    ax.set_xlim([doy[0],doy[-1]])
    # ax.set_xticks(dmnmp)
    ax.set_xticks(doy)
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    # ax.set_xlabel('Day of Year')
    ax.set_ylim([-0.25,1.25])
    ax.set_ylabel('$\Delta\delta T^{95}$ (K)')
    ax.set_title(r'%s %s' '\n' r'[%+05.1f,%+05.1f]' % (md.upper(),fo1.upper(),gr['lat'][iloc[0]],gr['lon'][iloc[1]]))
    sax=ax.twinx()
    sax.fill_between(doy,vn2-svn2,vn2+svn2,color='tab:blue',alpha=0.2,edgecolor=None)
    sax.plot(doy,vn2,'tab:blue')
    sax.set_ylabel('$\Delta\delta LH^{95}$ (W m$^{-2}$)')
    sax.set_ylim([-20,4])
    sax.set_ylim(sax.get_ylim()[::-1]) # invert axis
    # plt.legend()
    fig.savefig('%s/%s.%s.pdf' % (odir,varn,fo1), format='pdf', dpi=300)
    plt.close()
