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
varn='tas'
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
        'ooplh_dbc':    'blue',
        'ooplh_rbcsm':  'blue',
        'ooplh_rddsm':  'green',
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

    # remap to lat x lon
    ddpvn=remap(ddpvn,gr)
    ddpvn1=remap(ddpvn1,gr)
    ddpvn2=remap(ddpvn2,gr)

    addpvn = np.max(ddpvn,axis=0)-np.min(ddpvn,axis=0)
    mddpvn = np.max(ddpvn,axis=0)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')
    awgt=np.cos(np.deg2rad(mlat)) # area weight
    awgt=np.tile(awgt,(12,1,1))

    # make sure nans are consistent accross T and SM
    nidx=np.logical_or(np.isnan(ddpvn),np.isnan(ddpvn1))
    nidx=np.logical_or(np.isnan(nidx),np.isnan(ddpvn2))
    ddpvn[nidx]=np.nan
    ddpvn1[nidx]=np.nan
    ddpvn2[nidx]=np.nan

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
    nh=ah.copy()
    nh[gr['lat']<=0]=np.nan
    sh=ah.copy()
    sh[gr['lat']>0]=np.nan

    # select region of interest
    nhddp1=regsl(ddpvn1,nh)
    shddp1=regsl(ddpvn1,sh)
    ahddp1=regsla(ddpvn1,gr,ah)
    nhddp2=regsl(ddpvn2,nh)
    shddp2=regsl(ddpvn2,sh)
    ahddp2=regsla(ddpvn2,gr,ah)

    # convert sh to months since winter solstice
    shddp1=np.roll(shddp1,6,axis=0)
    shddp2=np.roll(shddp2,6,axis=0)

    # repeat above for tas if varn1 is not tas
    if varn1 != 'tas':
        nhddp=regsl(ddpvn,nh)
        shddp=regsl(ddpvn,sh)
        ahddp=regsla(ddpvn,gr,ah)
        shddp=np.roll(shddp,6,axis=0)
        # get index when sorted by month of maximum ddptas response
        nhddp,nhidx=sortmax(nhddp)
        shddp,shidx=sortmax(shddp)
        ahddp,ahidx=sortmax(ahddp)
        # sort
        nhddp1=nhddp1[:,nhidx]
        shddp1=shddp1[:,shidx]
        ahddp1=ahddp1[:,ahidx]
        nhddp2=nhddp2[:,nhidx]
        shddp2=shddp2[:,shidx]
        ahddp2=ahddp2[:,ahidx]
        # resorted idx
        nhidxrs=np.argmax(nhddp,axis=0)
        shidxrs=np.argmax(shddp,axis=0)
        ahidxrs=np.argmax(ahddp,axis=0)
    else:
        # sort by month of maximum response
        nhddp1,nhidx=sortmax(nhddp1)
        shddp1,shidx=sortmax(shddp1)
        ahddp1,ahidx=sortmax(ahddp1)
        # resorted idx
        nhidxrs=np.argmax(nhddp1,axis=0)
        shidxrs=np.argmax(shddp1,axis=0)
        ahidxrs=np.argmax(ahddp1,axis=0)
        nhddp=nhddp1.copy()
        shddp=shddp1.copy()
        ahddp=ahddp1.copy()

    # make weights and reorder
    nhw=regsl(awgt,nh)
    shw=regsl(awgt,sh)
    ahw=regsla(awgt,gr,ah)
    nhw=nhw[:,nhidx]
    shw=shw[:,shidx]
    ahw=ahw[:,ahidx]

    # cumulatively add weights
    nhcw=np.flip(np.cumsum(nhw,axis=1),axis=1)/np.sum(nhw,axis=1,keepdims=True)
    shcw=np.flip(np.cumsum(shw,axis=1),axis=1)/np.sum(shw,axis=1,keepdims=True)
    ahcw=np.flip(np.cumsum(ahw,axis=1),axis=1)/np.sum(ahw,axis=1,keepdims=True)

    # bin into 12
    def binned(vn,w,idx):
        bvn=np.empty([12,12])
        bw=np.zeros(13)
        for im in range(12):
            bvn[im,:]=[np.sum(w[im,idx==i]*vn[im,idx==i])/np.sum(w[im,idx==i]) for i in range(12)]
            bw[im+1]=np.sum(w[im,idx==im])+bw[im]
        return bvn,bw/bw[-1]
    nhddp1,nhcw=binned(nhddp1,nhw,nhidxrs)
    shddp1,shcw=binned(shddp1,shw,shidxrs)
    ahddp1,ahcw=binned(ahddp1,ahw,ahidxrs)
    nhddp2,_=binned(nhddp2,nhw,nhidxrs)
    shddp2,_=binned(shddp2,shw,shidxrs)
    ahddp2,_=binned(ahddp2,ahw,ahidxrs)
    # repeat for DdT
    nhddp,nhcw=binned(nhddp,nhw,nhidxrs)
    shddp,shcw=binned(shddp,shw,shidxrs)
    ahddp,ahcw=binned(ahddp,ahw,ahidxrs)

    # convert from edge to mp values
    nhcwmp=1/2*(nhcw[1:]+nhcw[:-1])
    shcwmp=1/2*(shcw[1:]+shcw[:-1])
    ahcwmp=1/2*(ahcw[1:]+ahcw[:-1])

    # list of maximum value
    mon=range(12)
    nhddp1=[nhddp1[i,i] for i in mon]
    shddp1=[shddp1[i,i] for i in mon]
    ahddp1=[ahddp1[i,i] for i in mon]
    nhddp2=[nhddp2[i,i] for i in mon]
    shddp2=[shddp2[i,i] for i in mon]
    ahddp2=[ahddp2[i,i] for i in mon]

    # plot
    fig=plt.figure(figsize=fs)
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax = fig.add_axes(divider.get_position(),axes_locator=divider.new_locator(nx=1, ny=1))
    clf=ax.plot(mon,ahddp1,color=cline(varn1))
    clf=ax.plot(mon,ahddp2,color=cline(varn2))
    # ax.set_title(tstr)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xticks(np.arange(1,11+2,2))
    ax.set_xticklabels(np.arange(2,12+2,2))
    # ax.set_yticks(np.arange(0,1+0.2,0.2))
    ax.set_xlim([-0.5,11.5])
    ax.set_ylim(ymax(varn1))
    if ylb:
        ax.set_ylabel('$\Delta\delta %s^{%02d}$ (%s)'%(varnlb(varn1),p,unitlb(varn1)))
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
