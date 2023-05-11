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

md='mi'

largs=[
    {
    'landonly':False, # only use land grid points for rsq
    'troponly':False, # only look at tropics
    },
    {
    'landonly':True, # only use land grid points for rsq
    'troponly':False, # only look at tropics
    },
    {
    'landonly':True, # only use land grid points for rsq
    'troponly':True, # only look at tropics
    },
]

for flags in largs:
    idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
    idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mmm',varn1)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    iname='%s/sp.rsq.%s_%s_%s.%s' % (idir,varn,his,fut,se)
    if flags['landonly']:
        iname='%s.land'%iname
    if flags['troponly']:
        iname='%s.trop'%iname

    # correlation
    _,_,gr=pickle.load(open('%s/d%s_%s_%s.%s.pickle' % (idir1,'tas',his,fut,se), 'rb'))	
    r=pickle.load(open('%s.pickle' % (iname), 'rb'))	
    rsq=r**2

    # plot rsq (pct warming - mean warming)
    for i,p in enumerate(gr['pct']):
        if skip5075 and (p==50 or p==75):
            continue

        oname='%s/sp.rsq%02d%s.%s'%(odir,p,varn,fo)
        if flags['landonly']:
            oname='%s.land'%oname
        if flags['troponly']:
            oname='%s.trop'%oname

        fig,ax=plt.subplots(figsize=(5,4),constrained_layout=True)
        ax.set_title(r'%s %s' % ('MMM',fo.upper()),fontsize=16)
        ax.plot(np.arange(12)+1,np.nanmean(rsq[...,i],0),'k')
        ax.set_xlabel('Month')
        ax.set_xticks(np.arange(12)+1)
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
        ax.set_ylabel('Global mean $R^2(\Delta\delta T^{%02d},\delta P^{%02d})$'%(p,p))
        if flags['landonly']:
            ax.set_ylabel('Land $R^2(\Delta\delta T^{%02d},\delta P^{%02d})$'%(p,p))
        if flags['landonly'] and flags['troponly']:
            ax.set_ylabel('Tropical land $R^2(\Delta\delta T^{%02d},\delta P^{%02d})$'%(p,p))
        fig.savefig('%s.pdf' % (oname), format='pdf', dpi=300)
