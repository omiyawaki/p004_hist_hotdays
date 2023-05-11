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
vnpre='d' # prefix of variable to compute
bnpre='ddp' # prefix of variable to bin
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'
skip507599=True

md='mmm'

idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mi',varn)
odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

if not os.path.exists(odir):
    os.makedirs(odir)

lflag=[
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

for flag in lflag:
    # warming
    iname='%s/b.%s.%s%s_%s_%s.%s' % (idir,bnpre,vnpre,varn,his,fut,se)
    if flag['landonly']:
        iname='%s.land'%iname
    if flag['troponly']:
        iname='%s.trop'%iname

    dat,pct,lbn=pickle.load(open('%s.pickle' % (iname), 'rb'))	
    dat=np.nanmean(dat,0)

    [mlbn,mpct] = np.meshgrid(lbn, pct, indexing='ij')

    # plot 
    fig,ax=plt.subplots(figsize=(4,4),constrained_layout=True)
    clf=ax.contourf(mpct, mlbn, np.transpose(dat), np.arange(-2,10+0.1,0.1),extend='both', vmax=10, vmin=-10, cmap='RdBu_r')
    tname=r'%s' % (md.upper())
    if flag['landonly']:
        tname='%s, Land'%tname
    if flag['troponly']:
        tname='%s, Tropics'%tname
    ax.set_title(tname)
    ax.set_ylim([-1.5,1.5])
    ax.set_xlabel('$T_\mathrm{2\,m}$ percentile')
    ax.set_ylabel('$\Delta \delta T^x_\mathrm{2\,m}$ (K)')
    cb=fig.colorbar(clf,location='right',aspect=50)
    # cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\Delta \overline{T}_\mathrm{2\,m}$ (K)')
    pname='%s/b.%s.%s%s.%s' % (odir,bnpre,vnpre,varn,fo)
    if flag['landonly']:
        pname='%s.land'%pname
    if flag['troponly']:
        pname='%s.trop'%pname
    fig.savefig('%s.pdf' %pname, format='pdf', dpi=300)
