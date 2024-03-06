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

nbs=int(1e2) # number of bootstrap resamples
lre=['tr','ml','hl'] # tr=tropics, ml=midlatitudes, hl=high lat, et=extratropics
tlat=30 # latitude bound for tropics
plat=50 # midlatitude bound
alc=0.05 # significance level (that mmm is different from 0)
cval=0.4 # threshold DdT value for drawing contour line
npmx=20 # number of bins for AI percentiles
dpmx=100/npmx
lpmx=np.arange(0,100+dpmx,dpmx)
mppmx=1/2*(lpmx[1:]+lpmx[:-1])
filt=False # only look at gridpoints with max exceeding value below
fmax=0.5
title=True
xlb=True
ylboverride=True
cboverride=True
titleoverride=False
fs=(3.5,3)
pds=(1,0.5)
axs=(1.5,2)
h=[Size.Fixed(pds[0]), Size.Fixed(axs[0])]
v=[Size.Fixed(pds[1]), Size.Fixed(axs[1])]

p=95 # percentile
varn='tas'
varn1='tas'
varnp='tas'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'
skip507599=True

lmd0=mods(fo1)
lmd=['mmm']

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
        'ti_ro':        'BrBG',
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
        'ti_ro':        1,
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

def plot(re,md):
    if ylboverride:
        ylb=True
    else:
        ylb=True if re=='tr' else False
    if cboverride:
        showcb=True
    else:
        showcb=True if re=='hl' else False
    # plot strings
    if md=='mmm':
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
    else:
        tstr=md
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

    def regsl2d(v,ma,kidx):
        v=v*ma
        v=np.reshape(v,[v.shape[0]*v.shape[1]])
        return v[kidx]

    def sortai(v):
        idx=np.argsort(v)
        return v[idx],idx

    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    def load_vn(varn,fo,byr,px='m'):
        idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        return xr.open_dataarray('%s/%s.%s_%s.%s.nc' % (idir,px,varn,byr,se))

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

    def load_vn(idir0):
        ddpvne=xr.open_dataarray('%s/ddpc.%s_%s_%s.%s.nc' % (idir0,varn1,his,fut,se))
        if varn1=='pr': ddpvne=86400*ddpvne
        return ddpvne.sel(percentile=p).squeeze()

    # load data for each model
    def load_idir(md):
        return '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn1)
    idirs=[load_idir(md0) for md0 in lmd0]

    # remap to lat x lon
    ddpvn=remap(ddpvn,gr)
    ddpvn1=remap(ddpvn1,gr)

    # repeat for ensemble if mmm
    if md=='mmm':
        ddpvne=[load_vn(idir0) for idir0 in tqdm(idirs)]
        ddpvne=xr.concat(ddpvne,'model')
        ddpvne=eremap(ddpvne,gr)

    mx=np.nanmax(ddpvn,axis=0)

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')
    awgt=np.cos(np.deg2rad(mlat)) # area weight

    # make sure nans are consistent
    nidx=np.isnan(ddpvn1)
    if md=='mmm':
        for imd in range(ddpvne.shape[0]):
            nidx=np.logical_or(np.isnan(nidx),np.isnan(ddpvne[imd,...]))
    ddpvn1[nidx]=np.nan
    if md=='mmm': ddpvne[:,nidx]=np.nan

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
    ahddp1,kidx=regsla(ddpvn1,gr,ah)
    if md=='mmm': ahddpe=eregsla(ddpvne,gr,ah,kidx)
    mx=regsl2d(mx,ah,kidx)
    # weights
    ahw=regsl2d(awgt,ah,kidx)

    # bin DdT
    def binned(vn,w,idx):
        nb=idx.max()-1
        bvn=np.empty([12,nb])
        for im in range(12):
            bvn[im,:]=[np.nansum(w[idx==i]*vn[im,idx==i])/np.nansum(w[idx==i]) for i in 1+np.arange(nb)]
        return bvn
    pmx=np.nanpercentile(mx,lpmx)
    imx=np.digitize(mx,pmx)
    ahddp1=binned(ahddp1,ahw,imx)

    if md=='mmm':
        # bin into 12
        def ebinned(vn,w,idx):
            nb=idx.max()-1 # number of bins
            bvn=np.empty([vn.shape[0],12,nb])
            for imd in range(vn.shape[0]):
                for im in range(12):
                    bvn[imd,im,:]=[np.nansum(w[idx==i]*vn[imd,im,idx==i])/np.nansum(w[idx==i]) for i in range(nb)]
            return bvn
        ahddpe=ebinned(ahddpe,ahw,imx)

        # bootstrap
        s=ahddpe.shape
        ahddpe=np.reshape(ahddpe,(s[0],s[1]*s[2]))
        al=np.ones(ahddpe.shape[1])
        for ig in tqdm(range(ahddpe.shape[1])):
            sa=ahddpe[:,ig]
            bs=[np.nanmean(resample(sa)) for _ in range(nbs)]
            bs=np.array(bs)
            pl,pu=np.percentile(bs,100*np.array([alc/2,1-alc/2]))
            if pl<0 and pu>0: al[ig]=0  # 0 is inside confidence interval
        al=np.reshape(al,(s[1],s[2]))

    mon=range(12)
    [ahmmon,ahmbin] = np.meshgrid(mon,mppmx, indexing='ij')

    # plot gp vs seasonal cycle of varn1 PCOLORMESH
    fig=plt.figure(figsize=fs)
    divider=Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax=fig.add_axes(divider.get_position(),axes_locator=divider.new_locator(nx=1, ny=1))
    clf=ax.pcolormesh(ahmmon,ahmbin,ahddp1,vmin=-vmax(varn1),vmax=vmax(varn1),cmap=cmap(varn1))
    if md=='mmm':
        hatch=plt.fill_between([-0.5,11.5],0,100,hatch='///////',color='none',edgecolor='gray',linewidths=0.3)
        ax.pcolormesh(ahmmon,ahmbin,np.ma.masked_where(al==0,ahddp1),vmin=-vmax(varn1),vmax=vmax(varn1),cmap=cmap(varn1))
    # ax.set_title()
    ax.text(0.5,1.05,tstr,c=ct(varn1),ha='center',va='center',transform=ax.transAxes)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xticks(np.arange(1,11+2,2))
    ax.set_xticklabels(np.arange(2,12+2,2))
    ax.set_yticks(100*np.arange(0,1+0.2,0.2))
    # ax.set_xticklabels(np.arange(2,12+2,2))
    if ylb:
        ax.set_ylabel('Annual Maximum Percentile')
    else:
        ax.set_yticklabels([])
    if not xlb:
        ax.set_xticklabels([])
    ax.set_xlim([-0.5,11.5])
    ax.set_ylim([0,100])
    if showcb:
        cb=plt.colorbar(clf,cax=fig.add_axes([(pds[0]+axs[0]+0.15)/fs[0],pds[1]/fs[1],0.1/fs[0],axs[1]/fs[1]]))
        if titleoverride:
            cb.set_label('%s'%(unitlb(varn1)))
        else:
            cb.set_label('$\Delta\delta %s^{%02d}$ (%s)'%(varnlb(varn1),p,unitlb(varn1)))
    fig.savefig('%s/sc.ddp%02d%s.%s%s.ah.max.sign.%s.png' % (odir1,p,varn1,fo,fstr,re), format='png', dpi=600,backend='pgf')
    fig.savefig('%s/sc.ddp%02d%s.%s%s.ah.max.sign.%s.pdf' % (odir1,p,varn1,fo,fstr,re), format='pdf', dpi=600,backend='pgf')

def wrapper(md):
    print(md)
    [plot(re,md) for re in lre]

# wrapper('CESM2')

[wrapper(md) for md in tqdm(lmd)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lmd)) as p:
#         p.map(wrapper,lmd)
