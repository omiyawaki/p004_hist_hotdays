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
varn='predtm'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'

lmd=mods(fo1)

for md in lmd:
    idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn)
    idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn)

    c = 0
    dt={}

    # mean temp
    ds1=xr.open_dataset('%s/m%s_%s.%s.nc' % (idir1,varn,his,se))
    gr={}
    gr['lat']=ds1['lat']
    gr['lon']=ds1['lon']
    try:
        vn1=ds1[varn]
    except:
        vn1=ds1['__xarray_dataarray_variable__']
    ds2=xr.open_dataset('%s/m%s_%s.%s.nc' % (idir2,varn,fut,se))
    try:
        vn2=ds2[varn]
    except:
        vn2=ds2['__xarray_dataarray_variable__']
    vn1=vn1.data[...,iloc[0],iloc[1]]
    vn2=vn2.data[...,iloc[0],iloc[1]]

    # prc temp
    ds1=xr.open_dataset('%s/wp%s%03d_%s.%s.nc' % (idir1,varn,nt,his,se))
    pct=ds1['percentile']
    try:
        pvn1=ds1[varn]
    except:
        pvn1=ds1['__xarray_dataarray_variable__']
    ds2=xr.open_dataset('%s/wp%s%03d_%s.%s.nc' % (idir2,varn,nt,fut,se))
    try:
        pvn2=ds2[varn]
    except:
        pvn2=ds2['__xarray_dataarray_variable__']
    pvn1=pvn1.data[...,iloc[0],iloc[1]]
    pvn2=pvn2.data[...,iloc[0],iloc[1]]

    # warming
    dvn=vn2-vn1
    dpvn=pvn2-pvn1
    ddpvn=dpvn-dvn[:,None]

    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,md,varn,gr['lat'][iloc[0]],gr['lon'][iloc[1]])

    if not os.path.exists(odir):
        os.makedirs(odir)

    doy=np.arange(vn1.shape[0])+1
    dmn=np.arange(0,len(doy),np.ceil(len(doy)/12))
    dmnmp=np.arange(0,len(doy),np.ceil(len(doy)/12))+np.ceil(len(doy)/24)

    # plot historical temp seasonal cycle
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    [ax.axvline(_dmn,color='k',linewidth=0.1) for _dmn in dmn]
    ax.plot(doy,vn1,'k',label='Mean')
    ax.plot(doy,pvn1[:,-2],'tab:orange',label='95th')
    ax.plot(doy,pvn1[:,-1],'tab:red',label='99th')
    ax.set_xlim([doy[0],doy[-1]])
    ax.set_xticks(dmnmp)
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    # ax.set_xlabel('Day of Year')
    ax.set_ylabel('$T_{2\,m}$ (K)')
    ax.set_title(r'%s %s' '\n' r'[%+05.1f,%+05.1f]' % (md.upper(),fo1.upper(),gr['lat'][iloc[0]],gr['lon'][iloc[1]]))
    plt.legend(frameon=False)
    fig.savefig('%s/p%s.%s.pdf' % (odir,varn,fo1), format='pdf', dpi=300)
    plt.close()

    # plot warming seasonal cycle
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    [ax.axvline(_dmn,color='k',linewidth=0.1) for _dmn in dmn]
    ax.plot(doy,dvn,'k',label='Mean')
    ax.plot(doy,dpvn[:,-2],'tab:orange',label='95th')
    ax.plot(doy,dpvn[:,-1],'tab:red',label='99th')
    ax.set_xlim([doy[0],doy[-1]])
    ax.set_xticks(dmnmp)
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    # ax.set_xlabel('Day of Year')
    ax.set_ylabel('Predicted $\Delta T_{2\,m}$ (K)')
    ax.set_title(r'%s %s' '\n' r'[%+05.1f,%+05.1f]' % (md.upper(),fo1.upper(),gr['lat'][iloc[0]],gr['lon'][iloc[1]]))
    plt.legend(frameon=False)
    fig.savefig('%s/dp%s.%s.pdf' % (odir,varn,fo1), format='pdf', dpi=300)
    plt.close()
