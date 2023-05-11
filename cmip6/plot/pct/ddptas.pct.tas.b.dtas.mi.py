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
vnpre='ddp' # prefix of variable to compute
bnpre='d' # prefix of variable to bin
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='2080-2100'
skip507599=True
troplat=20

md='mi'

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

# load mean warming
idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mi',varn)
dtas=pickle.load(open('%s/d%s_%s_%s.%s.pickle' % (idir,varn,his,fut,se), 'rb'))	

idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mi',varn)
odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

if not os.path.exists(odir):
    os.makedirs(odir)

lflag=[
    {
    'landonly':True, # only use land grid points for rsq
    'troponly':True, # only look at tropics
    },
    {
    'landonly':True, # only use land grid points for rsq
    'troponly':False, # only look at tropics
    },
    {
    'landonly':False, # only use land grid points for rsq
    'troponly':False, # only look at tropics
    },
]

for flag in lflag:
    # warming
    iname='%s/b.%s.%s%s_%s_%s.%s' % (idir,bnpre,vnpre,varn,his,fut,se)

    # load land mask
    lm,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomask.pickle','rb'))

    # cos weighting
    w=np.cos(np.deg2rad(gr['lat']))
    w=np.tile(w,(dtas.shape[1],dtas.shape[3],1))
    w=np.transpose(w,[0,2,1])

    if flag['landonly']:
        iname='%s.land'%iname
    if flag['troponly']:
        iname='%s.trop'%iname
        lats=np.transpose(np.tile(gr['lat'].data,(len(gr['lon']),1)),[1,0])
        lm[np.abs(lats)>troplat]=np.nan
    lm=np.tile(lm,(dtas.shape[1],1,1))

    dat,pct,lbn=pickle.load(open('%s.pickle' % (iname), 'rb'))	

    [mlbn,mpct] = np.meshgrid(lbn, pct, indexing='ij')

    lmd=mods(fo1)

    fig,ax=plt.subplots(nrows=4,ncols=5,figsize=(12,8),constrained_layout=True)
    ax=ax.flatten()
    if flag['landonly']:
        tname='Land'
    if flag['troponly']:
        tname='%s, Tropics'%tname
    if not flag['landonly'] and flag['troponly']:
        tname='Global'
    fig.suptitle(tname,fontsize=16)
    pname='%s/b.%s.%s%s.%s' % (odir,bnpre,vnpre,varn,fo)
    if flag['landonly']:
        pname='%s.land'%pname
    if flag['troponly']:
        pname='%s.trop'%pname

    for imd,md in enumerate(tqdm(lmd)):
        idat=dat[imd,...]

        if flag['landonly']:
            fdtas=lm*dtas[imd,...]
            w=lm*w
        else:
            fdtas=dtas

        # mean warming
        mdtas=np.nansum(w*fdtas)/np.nansum(w)

        # plot 
        ax[imd].axhline(mdtas,color='k',linewidth=0.5)
        clf=ax[imd].contourf(mpct, mlbn, np.transpose(idat), np.arange(-1.5,1.5+0.1,0.1),extend='both', vmax=1.5, vmin=-1.5, cmap='RdBu_r')
        ax[imd].set_title('%s'%(md.upper()),fontsize=16)
        ax[imd].set_ylim([0,8])
        ax[imd].set_xlabel('$T_\mathrm{2\,m}$ percentile')
        ax[imd].set_ylabel('$\Delta \overline{T}_\mathrm{2\,m}$ (K)')
        fig.savefig('%s.pdf' %pname, format='pdf', dpi=300)
    #MMM
    idat=np.nanmean(dat,0)
    if flag['landonly']:
        fdtas=lm*np.nanmean(dtas,0)
        w=lm*w
    else:
        fdtas=dtas
    # mean warming
    mdtas=np.nansum(w*fdtas)/np.nansum(w)
    # plot 
    ax[-1].axhline(mdtas,color='k',linewidth=0.5)
    clf=ax[-1].contourf(mpct, mlbn, np.transpose(idat), np.arange(-1.5,1.5+0.1,0.1),extend='both', vmax=1.5, vmin=-1.5, cmap='RdBu_r')
    ax[-1].set_title('MMM',fontsize=16)
    ax[-1].set_ylim([0,8])
    ax[-1].set_xlabel('$T_\mathrm{2\,m}$ percentile')
    ax[-1].set_ylabel('$\Delta \overline{T}_\mathrm{2\,m}$ (K)')

    cb=fig.colorbar(clf,ax=ax.ravel().tolist(),location='right',aspect=50)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=r'$\Delta \delta T^x_\mathrm{2\,m}$ (K)',size=16)
    fig.savefig('%s.pdf' %pname, format='pdf', dpi=300)
