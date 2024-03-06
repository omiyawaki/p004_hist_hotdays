import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor as Pool
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
from scipy import ndimage
from scipy.stats import linregress
from tqdm import tqdm
from util import mods
from utils import monname,varnlb,unitlb
from mpl_toolkits.axes_grid1 import Divider, Size

lre=['tr','ml','hl'] # tr=tropics, ml=midlatitudes, hl=high lat, et=extratropics
tlat=30 # latitude bound for tropics
plat=50 # midlatitude bound
cval=0.4 # threshold DdT value for drawing contour line
filt=False # only look at gridpoints with max exceeding value below
fmax=0.5
title=False
xlb=True
ylboverride=True
fs=(3.5,3)
pds=(1,0.5)
axs=(1.5,2)
h=[Size.Fixed(pds[0]), Size.Fixed(axs[0])]
v=[Size.Fixed(pds[1]), Size.Fixed(axs[1])]

p=95 # percentile
varn1='hfls'
varnp='hfls'
varn2='ooplh'
varnp2='ooplh'
if varn1==varn2:
    varnc=varn1
else:
    varnc='%s+%s'%(varn1,varn2)

if ylboverride or varn2=='ooplh':
    ylb=True
else:
    ylb=False

se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'
skip507599=True

md='mmm'

def cline(vn):
    d={ 'hfls':         'tab:blue',
        'ooplh':        'k',
        'ooplh_msm':    'blue',
        'ooplh_fixmsm': 'orange',
        'ooplh_fixasm': 'blue',
        'ooplh_fixbc':  'blue',
        'ooplh_dbc':    'magenta',
        'ooplh_rbcsm':  'cyan',
        'ooplh_rddsm':  'green',
        'ooplh_mtr':    'blue',
            }
    return d[vn]

def ymax(vn):
    d={ 'hfls':         [-3,1],
        'hfss':         [-3,1],
        'rsfc':         [-3,1],
        'ooplh':        [-3,1],
        'ooplh_msm':    [-3,1],
        'ooplh_fixmsm': [-3,1],
        'ooplh_fixasm': [-3,1],
        'ooplh_fixbc':  [-3,1],
        'ooplh_dbc':    [-3,1],
        'ooplh_rbcsm':  [-3,1],
        'ooplh_rddsm':  [-3,1],
        'ooplh_mtr':    [-3,1],
        'tas':          [-1,1],
        'pr':           [-1,1],
        'mrsos':        [-1,1],
        'ef':           [-0.05,0.05],
        'ef2':          [-0.05,0.05],
        'ef3':          [-0.05,0.05],
        'ooef':         [-0.05,0.05],
        'oopef':        [-0.05,0.05],
        'oopef_fixbc':  [-0.05,0.05],
            }
    return d[vn]

def vstr(vn):
    d={ 'ooplh':        r'$BC_{all}$',
        'ooplh_fixbc':  r'$BC_{hist}$',
        'ooplh_dbc':    r'$SM_{hist}$',
        'ooplh_rbcsm':  r'(a)$-$(b)$-$(c)',
        'ooplh_rddsm':  r'(b)$-$(c)',
        'ooplh_fixmsm': r'$BC_{hist}$, $\Delta\delta SM=0$',
        'ooplh_fixasm': r'$BC_{hist}$, $\Delta\delta SM=0$',
        'ooplh_mtr':    r'$\frac{\mathrm{d}LH}{\mathrm{d}SM}_{hist}\Delta SM$',
        'oopef':        r'$BC_{all}$',
        'oopef_fixbc':  r'$BC_{hist}$',
        'mrsos':        r'$SM$',
        'pr':           r'$P$',
            }
    return d[vn]

def plot(re):
    # plot strings
    if 'ooplh' in varn1 or 'oopef' in varn1:
        tstr=vstr(varn1)
    elif 'mrsos' in varn1 or 'pr' in varn1:
        tstr=vstr(varn1)
    else:
        if re=='tr':
            tstr='Tropics'
        elif re=='ml':
            tstr='Midlatitudes'
        elif re=='hl':
            tstr='High latitudes'
        elif re=='et':
            tstr='Extratropics'
    fstr='.filt' if filt else ''

    # load land indices
    lmi,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

    # grid
    rgdir='/project/amp/miyawaki/data/share/regrid'
    # open CESM data to get output grid
    cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
    cdat=xr.open_dataset(cfil)
    gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

    def remap(v,gr):
        llv=np.nan*np.ones([12,gr['lat'].size*gr['lon'].size])
        llv[:,lmi]=v.data
        llv=np.reshape(llv,(12,gr['lat'].size,gr['lon'].size))
        return llv

    def regsl(v,ma):
        v=v*ma
        v=np.reshape(v,[v.shape[0],v.shape[1]*v.shape[2]])
        return v[:,~np.isnan(v).any(axis=0)]

    def regsla(v,gr,ma):
        return regsl(v,ma)

    def sortmax(v):
        im=np.argmax(v,axis=0)
        idx=np.argsort(im)
        return v[:,idx],idx

    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varnc)
    if not os.path.exists(odir):
        os.makedirs(odir)

    def load_vn(varn,fo,byr,px='m'):
        idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        return xr.open_dataarray('%s/%s.%s_%s.%s.nc' % (idir,px,varn,byr,se))

    aimon=load_vn('ooai',fo1,his,px='m') # historical AI (monthly)

    # variable of interest

    def load_vn(varn0,varnp0):
        idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn0)
        ds=xr.open_dataset('%s/ddpc.%s_%s_%s.%s.nc' % (idir,varn0,his,fut,se))
        if varn1=='pr':
            ddpvn1=86400*ds[varn1]
        else:
            try:
                ddpvn1=ds[varn1]
            except:
                ddpvn1=ds[varnp0]
        pct=ds['percentile']
        return ddpvn1.sel(percentile=pct==p).squeeze()

    ddpvn1=load_vn(varn1,varnp)
    ddpvn2=load_vn(varn2,varnp2)

    # reorder months by AI
    idxai=np.argsort(aimon.data,axis=0)
    ddpvn1.data=np.take_along_axis(ddpvn1.data,idxai,axis=0)
    ddpvn2.data=np.take_along_axis(ddpvn2.data,idxai,axis=0)

    # remap to lat x lon
    ddpvn1=remap(ddpvn1,gr)
    ddpvn2=remap(ddpvn2,gr)

    # mask greenland and antarctica
    aagl=pickle.load(open('/project/amp/miyawaki/data/share/aa_gl/cesm2/aa_gl.pickle','rb'))
    ddpvn1=ddpvn1*aagl
    ddpvn2=ddpvn2*aagl

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')
    awgt=np.cos(np.deg2rad(mlat)) # area weight
    awgt=np.tile(awgt,(12,1,1))

    # make sure nans are consistent accross T and SM
    nidx=np.logical_or(np.isnan(ddpvn1),np.isnan(ddpvn2))
    ddpvn1[nidx]=np.nan
    ddpvn2[nidx]=np.nan
    awgt[nidx]=np.nan

    # subselect by latitude and/or high seasonal cycle
    if filt:
        ah=np.nan*np.ones_like(mlat)
        ah[mddpvn>fmax]=1
    else:
        ah=np.ones_like(mlat)
    if re=='tr':
        ah[np.abs(gr['lat'])>tlat]=np.nan
    elif re=='ml':
        ah[np.logical_or(np.abs(gr['lat'])<=tlat,np.abs(gr['lat'])>plat)]=np.nan
    elif re=='hl':
        ah[np.abs(gr['lat'])<=plat]=np.nan
    elif re=='et':
        ah[np.abs(gr['lat'])<=tlat]=np.nan

    # select region of interest
    ahddp1=regsla(ddpvn1,gr,ah)
    ahddp2=regsla(ddpvn2,gr,ah)
    awgt=regsla(awgt,gr,ah)

    # area weighted spatial mean
    ahddp1=np.nansum(awgt*ahddp1,axis=1)/np.nansum(awgt,axis=1)
    ahddp2=np.nansum(awgt*ahddp2,axis=1)/np.nansum(awgt,axis=1)

    mon=np.arange(12)
    # plot
    fig=plt.figure(figsize=fs)
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax = fig.add_axes(divider.get_position(),axes_locator=divider.new_locator(nx=1, ny=1))
    ax.axhline(0,linewidth=0.5,color='tab:gray')
    clf=ax.plot(mon,ahddp1,color=cline(varn1),label='Actual')
    clf=ax.plot(mon,ahddp2,color=cline(varn2))
    # ax.set_title(tstr)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xticks(np.arange(1,11+2,2))
    ax.set_xticklabels(np.arange(2,12+2,2))
    # ax.set_yticks(np.arange(0,1+0.2,0.2))
    ax.set_xlim([-0.5,11.5])
    ax.set_ylim(ymax(varn1))
    if varn1=='hfls':
        ax.legend(frameon=False)
    if ylb:
        ax.set_ylabel('$\Delta\delta %s^{%02d}$ (%s)'%(varnlb(varn1),p,unitlb(varn1)))
    else:
        ax.set_yticklabels([])
    if not xlb:
        ax.set_xticklabels([])
    fig.savefig('%s/line.ai.ddp%02d%s.%s%s.ah.avg.%s.png' % (odir,p,varnc,fo,fstr,re), format='png', dpi=600)
    fig.savefig('%s/line.ai.ddp%02d%s.%s%s.ah.avg.%s.pdf' % (odir,p,varnc,fo,fstr,re), format='pdf', dpi=600)

[plot(re) for re in tqdm(lre)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lre)) as p:
#         p.map(plot,lre)
