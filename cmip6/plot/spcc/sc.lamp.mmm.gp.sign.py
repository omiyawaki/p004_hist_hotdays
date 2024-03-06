import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor as Pool
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
from scipy import ndimage
from scipy.stats import ttest_1samp as tt1
from sklearn.utils import resample
from tqdm import tqdm
from util import mods
from utils import monname,varnlb,unitlb
from mpl_toolkits.axes_grid1 import Divider,Size

nb=int(1e2) # number of bootstrap resamples
lre=['tr','ml','hl'] # tr=tropics, ml=midlatitudes, hl=high lat, et=extratropics
tlat=30 # latitude bound for tropics
plat=50 # midlatitude bound
cval=0.4 # threshold DdT value for drawing contour line
filt=False # only look at gridpoints with max exceeding value below
fmax=0.5
title=True
xlb=True
ylboverride=True
cboverride=True
titleoverride=True
fs=(3.5,3)
pds=(1,0.5)
axs=(1.5,2)
h=[Size.Fixed(pds[0]), Size.Fixed(axs[0])]
v=[Size.Fixed(pds[1]), Size.Fixed(axs[1])]

p=97.5 # percentile
varn='tas'
varn1='fat850'
varnp='fat850'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'
skip507599=True

md='mmm'

def ct(vn):
    d={ 
        'ooplh_msm':    'blue',
        'ooplh_fixmsm': 'orange',
        'ooplh_fixasm': 'blue',
        'ooplh_fixbc':  'blue',
        'ooplh_dbc':    'blue',
        'ooplh_rbcsm':  'blue',
        'ooplh_rddsm':  'green',
        'ooplh_mtr':    'blue',
            }
    try:
        color=d[vn]
    except:
        color='black'
    return color

def cmap(vn):
    d={ 'hfls':         'RdBu_r',
        'hfss':         'RdBu_r',
        'rsfc':         'RdBu_r',
        'ooplh':        'RdBu_r',
        'ooplh_msm':    'RdBu_r',
        'ooplh_fixmsm': 'RdBu_r',
        'ooplh_fixasm': 'RdBu_r',
        'ooplh_fixbc':  'RdBu_r',
        'ooplh_dbc':    'RdBu_r',
        'ooplh_rbcsm':  'RdBu_r',
        'ooplh_rddsm':  'RdBu_r',
        'ooplh_mtr':    'RdBu_r',
        'tas':          'RdBu_r',
        'pr':           'BrBG',
        'mrsos':        'BrBG',
        'td_mrsos':     'BrBG',
        'ti_pr':        'BrBG',
        'ti_ev':        'BrBG',
        'ef':           'RdBu_r',
        'ef2':          'RdBu_r',
        'ef3':          'RdBu_r',
        'ooef':         'RdBu_r',
        'oopef':        'RdBu_r',
        'oopef_fixbc':  'RdBu_r',
            }
    return d[vn]

def vmax(vn):
    d={ 'hfls':         10,
        'hfss':         10,
        'rsfc':         10,
        'ooplh':        10,
        'ooplh_msm':    10,
        'ooplh_fixmsm': 10,
        'ooplh_fixasm': 10,
        'ooplh_fixbc':  10,
        'ooplh_dbc':    10,
        'ooplh_rbcsm':  10,
        'ooplh_rddsm':  10,
        'ooplh_mtr':    10,
        'tas':          1,
        'pr':           1,
        'mrsos':        1,
        'td_mrsos':     1,
        'ti_pr':        1,
        'ti_ev':        1,
        'ef':           0.05,
        'ef2':          0.05,
        'ef3':          0.05,
        'ooef':         0.05,
        'oopef':        0.05,
        'oopef_fixbc':  0.05,
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
        'td_mrsos':     r'$SM_{\mathrm{30\,d}}$',
        'ti_pr':        r'$P_{\mathrm{30\,d}}$',
        'ti_ev':        r'$-E_{\mathrm{30\,d}}$',
        'pr':           r'$P$',
            }
    return d[vn]

def plot(re):
    if ylboverride:
        ylb=True
    else:
        ylb=True if re=='tr' else False
    if cboverride:
        showcb=True
    else:
        showcb=True if re=='hl' else False
    # plot strings
    if titleoverride:
        tstr='$%s$'%varnlb(varn1)
    else:
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

    def eremap(v,gr):
        llv=np.nan*np.ones([v.shape[0],12,gr['lat'].size*gr['lon'].size])
        llv[:,:,lmi]=v.data
        llv=np.reshape(llv,(v.shape[0],12,gr['lat'].size,gr['lon'].size))
        return llv

    # def regsl(v,ma):
    #     v=v*ma
    #     v=np.reshape(v,[v.shape[0],v.shape[1]*v.shape[2]])
    #     v[:,np.isnan(v).any(axis=0)]=np.nan
    #     return v

    # def regsla(v,gr,ma):
    #     sv=np.roll(v,6,axis=0) # seasonality shifted by 6 months
    #     v[:,gr['lat']<0,:]=sv[:,gr['lat']<0,:]
    #     return regsl(v,ma)

    def regsl(v,ma):
        v=v*ma
        v=np.reshape(v,[v.shape[0],v.shape[1]*v.shape[2]])
        kidx=~np.isnan(v).any(axis=0)
        return v[:,kidx],kidx

    def regsla(v,gr,ma):
        sv=np.roll(v,6,axis=0) # seasonality shifted by 6 months
        v[:,gr['lat']<0,:]=sv[:,gr['lat']<0,:]
        return regsl(v,ma)

    def eregsl(v,ma,kidx):
        v=v*np.moveaxis(ma[...,None],-1,0)
        v=np.reshape(v,[v.shape[0],v.shape[1],v.shape[2]*v.shape[3]])
        return v[:,:,kidx]

    def eregsla(v,gr,ma,kidx):
        sv=np.roll(v,6,axis=1) # seasonality shifted by 6 months
        v[:,:,gr['lat']<0,:]=sv[:,:,gr['lat']<0,:]
        return eregsl(v,ma,kidx)

    def sortmax(v):
        im=np.argmax(v,axis=0)
        idx=np.argsort(im)
        return v[:,idx],idx

    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    def load_mmm(varn,varnp):
        idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        ds=xr.open_dataset('%s/ddpc.%s_%s_%s.%s.nc' % (idir,varn,his,fut,se))
        ddpvn=ds[varnp]
        pct=ds['percentile']
        gpi=ds['gpi']
        return ddpvn.sel(percentile=pct==p).squeeze()

    ddpvn=load_mmm(varn,varn)
    ddpvn1=load_mmm(varn1,varnp)
    if varn1=='ti_ev':
        ddpvn1=-ddpvn1

    # variable of interest
    odir1 = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn1)
    if not os.path.exists(odir1):
        os.makedirs(odir1)

    # remap to lat x lon
    ddpvn=remap(ddpvn,gr)
    ddpvn1=remap(ddpvn1,gr)

    addpvn = np.max(ddpvn,axis=0)-np.min(ddpvn,axis=0)
    mddpvn = np.max(ddpvn,axis=0)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')
    awgt=np.cos(np.deg2rad(mlat)) # area weight
    awgt=np.tile(awgt,(12,1,1))

    # make sure nans are consistent accross T and SM
    nidx=np.isnan(ddpvn)
    ddpvn[nidx]=np.nan
    ddpvn1[nidx]=np.nan

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

    # select region
    ahddp,kidx=regsla(ddpvn,gr,ah)
    ahddp1,_=regsla(ddpvn1,gr,ah)

    # get index when sorted by month of maximum ddptas response
    ahddp,ahidx=sortmax(ahddp)
    # sort
    ahddp1=ahddp1[...,ahidx]
    # resorted idx
    ahidxrs=np.argmax(ahddp,axis=0)

    # make weights and reorder
    ahw,_=regsla(awgt,gr,ah)
    ahw=ahw[:,ahidx]

    # cumulatively add weights
    ahcw=np.flip(np.nancumsum(ahw,axis=1),axis=1)/np.nansum(ahw,axis=1,keepdims=True)

    # bin DdT
    def binned(vn,w,idx):
        bvn=np.empty([12,12])
        bw=np.zeros(13)
        for im in range(12):
            bvn[im,:]=[np.nansum(w[im,idx==i]*vn[im,idx==i])/np.nansum(w[im,idx==i]) for i in range(12)]
            bw[im+1]=np.nansum(w[im,idx==im])+bw[im]
        return bvn,bw/bw[-1]
    ahddp,ahcw=binned(ahddp,ahw,ahidxrs)
    ahddp1,_=binned(ahddp1,ahw,ahidxrs)

    # convert from edge to mp values
    ahcwmp=1/2*(ahcw[1:]+ahcw[:-1])

    mon=range(12)
    [ahmbin,ahmmon] = np.meshgrid(ahcwmp,mon, indexing='ij')

    # plot gp vs seasonal cycle of varn1 PCOLORMESH
    fig=plt.figure(figsize=fs)
    divider=Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax=fig.add_axes(divider.get_position(),axes_locator=divider.new_locator(nx=1, ny=1))
    clf=ax.pcolormesh(ahmmon,ahmbin,np.transpose(ahddp1),vmin=-vmax(varn1),vmax=vmax(varn1),cmap=cmap(varn1))
    ax.text(0.5,1.05,tstr,c=ct(varn1),ha='center',va='center',transform=ax.transAxes)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xticks(np.arange(1,11+2,2))
    ax.set_xticklabels(np.arange(2,12+2,2))
    ax.set_yticks(np.arange(0,1+0.2,0.2))
    # ax.set_xticklabels(np.arange(2,12+2,2))
    if ylb:
        ax.set_ylabel('Cumulative area fraction')
    else:
        ax.set_yticklabels([])
    if not xlb:
        ax.set_xticklabels([])
    ax.set_xlim([-0.5,11.5])
    ax.set_ylim([0,1])
    ax.set_ylim(ax.get_ylim()[::-1])
    x=np.arange(12)
    y=ahcwmp
    z=np.diag(np.diag(np.ones_like(ahddp)))
    # make dense grid
    scale=100
    yy=ndimage.zoom(y,scale,order=0)
    zz=ndimage.zoom(z,scale,order=0)
    xx=np.linspace(x.min(),x.max(),zz.shape[0])
    # extend out to edge
    xx=np.insert(xx,0,-0.5)
    yy=np.insert(yy,0,0)
    zz=np.insert(zz,0,zz[0,:],axis=0)
    zz=np.insert(zz,0,zz[:,0],axis=1)
    xx=np.append(xx,12.5)
    yy=np.append(yy,1)
    zz=np.insert(zz,-1,zz[-1,:],axis=0)
    zz=np.insert(zz,-1,zz[:,-1],axis=1)
    ax.contour(xx,yy,zz,levels=[0.5],colors='k',linewidths=0.5,corner_mask=False)
    if showcb:
        cb=plt.colorbar(clf,cax=fig.add_axes([(pds[0]+axs[0]+0.15)/fs[0],pds[1]/fs[1],0.1/fs[0],axs[1]/fs[1]]))
        if titleoverride:
            cb.set_label('%s'%(unitlb(varn1)))
        else:
            cb.set_label('$\Delta\delta %s^{%02d}$ (%s)'%(varnlb(varn1),p,unitlb(varn1)))
    fig.savefig('%s/sc.ddp%02d%s.%s%s.ah.gp.%s.png' % (odir1,p,varn1,fo,fstr,re), format='png', dpi=600,backend='pgf')
    fig.savefig('%s/sc.ddp%02d%s.%s%s.ah.gp.%s.pdf' % (odir1,p,varn1,fo,fstr,re), format='pdf', dpi=600,backend='pgf')

[plot(re) for re in tqdm(lre)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lre)) as p:
#         p.map(plot,lre)
