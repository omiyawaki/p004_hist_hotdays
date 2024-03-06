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
axs=(2,2)
h=[Size.Fixed(pds[0]), Size.Fixed(axs[0])]
v=[Size.Fixed(pds[1]), Size.Fixed(axs[1])]

p=95 # percentile
varn='tas'
varn1='ooplh_fixbc'
varnp='ooplh'
varn2='ooplh_fixmsm'
varnp2='ooplh'
varn3='ooplh_rddsm'
varnp3='ooplh'
varnc='%s+%s+%s'%(varn1,varn2,varn3)

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
        'ooplh_fixmsm': 'salmon',
        'ooplh_fixasm': 'blue',
        'ooplh_fixbc':  'slateblue',
        'ooplh_dbc':    'chocolate',
        'ooplh_rbcsm':  'teal',
        'ooplh_rddsm':  'yellowgreen',
        'ooplh_mtr':    'blue',
            }
    return d[vn]

def ymax(vn):
    d={ 'hfls':         [-10,1],
        'hfss':         [-10,1],
        'rsfc':         [-10,1],
        'ooplh':        [-10,1],
        'ooplh_msm':    [-10,1],
        'ooplh_fixmsm': [-10,1],
        'ooplh_fixasm': [-10,1],
        'ooplh_fixbc':  [-10,1],
        'ooplh_dbc':    [-10,1],
        'ooplh_rbcsm':  [-10,1],
        'ooplh_rddsm':  [-10,1],
        'ooplh_mtr':    [-10,1],
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
        'ooplh_rbcsm':  r'Residual',
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
        sv=np.roll(v,6,axis=0) # seasonality shifted by 6 months
        v[:,gr['lat']<0,:]=sv[:,gr['lat']<0,:]
        return regsl(v,ma)

    def sortmax(v):
        im=np.argmax(v,axis=0)
        idx=np.argsort(im)
        return v[:,idx],idx

    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varnc)
    if not os.path.exists(odir):
        os.makedirs(odir)

    # warming
    ds=xr.open_dataset('%s/ddpc.%s_%s_%s.%s.nc' % (idir,varn,his,fut,se))
    ddpvn=ds[varn]
    pct=ds['percentile']
    gpi=ds['gpi']
    ddpvn=ddpvn.sel(percentile=pct==p).squeeze()

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
        gpi=ds['gpi']
        return ddpvn1.sel(percentile=pct==p).squeeze()

    ddpvn1=load_vn(varn1,varnp)
    ddpvn2=load_vn(varn2,varnp2)
    ddpvn3=load_vn(varn3,varnp3)

    # remap to lat x lon
    ddpvn=remap(ddpvn,gr)
    ddpvn1=remap(ddpvn1,gr)
    ddpvn2=remap(ddpvn2,gr)
    ddpvn3=remap(ddpvn3,gr)

    addpvn = np.max(ddpvn,axis=0)-np.min(ddpvn,axis=0)
    mddpvn = np.max(ddpvn,axis=0)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')
    awgt=np.cos(np.deg2rad(mlat)) # area weight
    awgt=np.tile(awgt,(12,1,1))

    # make sure nans are consistent accross T and SM
    nidx=np.logical_or(np.isnan(ddpvn),np.isnan(ddpvn1))
    nidx=np.logical_or(np.isnan(nidx),np.isnan(ddpvn2))
    nidx=np.logical_or(np.isnan(nidx),np.isnan(ddpvn3))
    ddpvn[nidx]=np.nan
    ddpvn1[nidx]=np.nan
    ddpvn2[nidx]=np.nan
    ddpvn3[nidx]=np.nan

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
    ahddp3=regsla(ddpvn3,gr,ah)

    # repeat above for tas if varn1 is not tas
    ahddp=regsla(ddpvn,gr,ah)
    # get index when sorted by month of maximum ddptas response
    ahddp,ahidx=sortmax(ahddp)
    # sort
    ahddp1=ahddp1[:,ahidx]
    ahddp2=ahddp2[:,ahidx]
    ahddp3=ahddp3[:,ahidx]
    # resorted idx
    ahidxrs=np.argmax(ahddp,axis=0)

    # make weights and reorder
    ahw=regsla(awgt,gr,ah)
    ahw=ahw[:,ahidx]

    # cumulatively add weights
    ahcw=np.flip(np.cumsum(ahw,axis=1),axis=1)/np.sum(ahw,axis=1,keepdims=True)

    # bin into 12
    def binned(vn,w,idx):
        bvn=np.empty([12,12])
        bw=np.zeros(13)
        for im in range(12):
            bvn[im,:]=[np.sum(w[im,idx==i]*vn[im,idx==i])/np.sum(w[im,idx==i]) for i in range(12)]
            bw[im+1]=np.sum(w[im,idx==im])+bw[im]
        return bvn,bw/bw[-1]
    ahddp1,ahcw=binned(ahddp1,ahw,ahidxrs)
    ahddp2,_=binned(ahddp2,ahw,ahidxrs)
    ahddp3,_=binned(ahddp3,ahw,ahidxrs)
    # repeat for DdT
    ahddp,ahcw=binned(ahddp,ahw,ahidxrs)

    # convert from edge to mp values
    ahcwmp=1/2*(ahcw[1:]+ahcw[:-1])

    # list of maximum value
    mon=range(12)
    ahddp1=[ahddp1[i,i] for i in mon]
    ahddp2=[ahddp2[i,i] for i in mon]
    ahddp3=[ahddp3[i,i] for i in mon]

    # plot
    fig=plt.figure(figsize=fs)
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax = fig.add_axes(divider.get_position(),axes_locator=divider.new_locator(nx=1, ny=1))
    ax.plot(mon,ahddp1,color=cline(varn1))
    ax.plot(mon,ahddp2,color=cline(varn2))
    ax.plot(mon,ahddp3,color=cline(varn3))
    # ax.set_title(tstr)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xticks(np.arange(1,11+2,2))
    ax.set_xticklabels(np.arange(2,12+2,2))
    # ax.set_yticks(np.arange(0,1+0.2,0.2))
    ax.set_xlim([-0.5,11.5])
    ax.set_ylim(ymax(varn1))
    if ylb:
        ax.set_ylabel('%s'%(unitlb(varn1)))
    else:
        ax.set_yticklabels([])
    if not xlb:
        ax.set_xticklabels([])
    fig.savefig('%s/line.sc.ddp%02d%s.%s%s.ah.max.%s.png' % (odir,p,varnc,fo,fstr,re), format='png', dpi=600)
    fig.savefig('%s/line.sc.ddp%02d%s.%s%s.ah.max.%s.pdf' % (odir,p,varnc,fo,fstr,re), format='pdf', dpi=600)

[plot(re) for re in tqdm(lre)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lre)) as p:
#         p.map(plot,lre)
