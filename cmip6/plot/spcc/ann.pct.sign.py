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
bstype='bins' # bootstrap by mmm bin samples or multimodel samples of bin means
tlat=30 # latitude bound for tropics
plat=50 # midlatitude bound
alc=0.05 # significance level (that mmm is different from 0)
cval=0.4 # threshold DdT value for drawing contour line
npai=20 # number of bins for AI percentiles
dpai=100/npai
lpai=np.arange(0,100+dpai,dpai)
mppai=1/2*(lpai[1:]+lpai[:-1])
filt=False # only look at gridpoints with max exceeding value below
fmax=0.5
title=True
xlb=True
ylboverride=True
cboverride=True
titleoverride=False

p=2.5 # percentile
varn='tas'
varn1='hfls'
varnp='hfls'
se = 'sc' # season (ann, djf, mam, jja, son)
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
# fut='2080-2100'
fut='gwl2.0'
skip507599=True

lmd=mods(fo1)
md='mmm'

def ct(vn):
    d={ 
        'ooplh_msm':    'blue',
        'ooplh_fixmsm': 'orange',
        'ooplh_fixasm': 'blue',
        'ooplh_fixbc':  'blue',
        'ooplh_dbc':    'magenta',
        'ooplh_rbcsm':  'cyan',
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
        'oopef':        'RdBu_r',
        'oopef_msm':    'RdBu_r',
        'oopef_fixmsm': 'RdBu_r',
        'oopef_fixasm': 'RdBu_r',
        'oopef_fixbc':  'RdBu_r',
        'oopef_dbc':    'RdBu_r',
        'oopef_rbcsm':  'RdBu_r',
        'oopef_rddsm':  'RdBu_r',
        'oopef_mtr':    'RdBu_r',
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
        'oosf':         'RdBu_r',
        'ooef':         'RdBu_r',
            }
    return d[vn]

def vmax(vn):
    d={ 'hfls':         5,
        'hfss':         5,
        'rsfc':         5,
        'ooplh':        5,
        'ooplh_msm':    5,
        'ooplh_fixmsm': 5,
        'ooplh_fixasm': 5,
        'ooplh_fixbc':  5,
        'ooplh_dbc':    5,
        'ooplh_rbcsm':  5,
        'ooplh_rddsm':  5,
        'ooplh_mtr':    5,
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
        'oosf':         0.05,
        'ooef':         0.05,
        'oopef':        0.05,
        'oopef_fixbc':  0.05,
        'oopef_fixmsm': 0.05,
        'oopef_rddsm':  0.05,
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
        'oopef_fixmsm': r'$BC_{hist}$, $\Delta\delta SM=0$',
        'oopef_rbcsm':  r'(a)$-$(b)$-$(c)',
        'oopef_rddsm':  r'(b)$-$(c)',
        'oopef_dbc':    r'$SM_{hist}$',
        'mrsos':        r'$SM$',
        'td_mrsos':     r'$SM_{\mathrm{30\,d}}$',
        'ti_pr':        r'$P_{\mathrm{30\,d}}$',
        'ti_ev':        r'$-E_{\mathrm{30\,d}}$',
        'ti_ro':        r'$-R_{\mathrm{30\,d}}$',
        'pr':           r'$P$',
            }
    return d[vn]

def plot():
    if ylboverride:
        ylb=True
    else:
        ylb=False
    if cboverride:
        showcb=True
    else:
        showcb=False
    # plot strings
    if titleoverride:
        tstr='$%s$'%varnlb(varn1)
    else:
        if 'ooplh' in varn1 or 'oopef' in varn1:
            tstr=vstr(varn1)
        elif 'mrsos' in varn1 or 'pr' in varn1:
            tstr=vstr(varn1)
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
        llv=np.nan*np.ones([v.shape[0],v.shape[1],gr['lat'].size*gr['lon'].size])
        llv[...,lmi]=v.data
        llv=np.reshape(llv,(v.shape[0],v.shape[1],gr['lat'].size,gr['lon'].size))
        return llv

    def eremap(v,gr):
        llv=np.nan*np.ones([v.shape[0],v.shape[1],v.shape[2],gr['lat'].size*gr['lon'].size])
        llv[...,lmi]=v.data
        llv=np.reshape(llv,(v.shape[0],v.shape[1],v.shape[2],gr['lat'].size,gr['lon'].size))
        return llv

    def regsl(v,ma):
        v=v*ma
        v=np.reshape(v,[v.shape[0],v.shape[1],v.shape[2]*v.shape[3]])
        kidx=~np.isnan(v).any(axis=(0,1))
        return v[...,kidx],kidx

    def regsla(v,gr,ma):
        return regsl(v,ma)

    def eregsl(v,ma,kidx):
        v=v*np.moveaxis(ma[...,None],-1,0)
        v=np.reshape(v,[v.shape[0],v.shape[1],v.shape[2],v.shape[3]*v.shape[4]])
        return v[...,kidx]

    def eregsla(v,gr,ma,kidx):
        return eregsl(v,ma,kidx)

    def regsl2d(v,ma,kidx):
        v=v*ma
        v=np.reshape(v,[v.shape[0]*v.shape[1]])
        return v[kidx]

    def sortai(v):
        idx=np.argsort(v)
        return v[idx],idx

    def load_vn(varn,fo,byr,px='m'):
        idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        return xr.open_dataarray('%s/%s.%s_%s.%s.nc' % (idir,px,varn,byr,se))

    def load_mmm(varn,varnp):
        idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        ds=xr.open_dataset('%s/ddpc.%s_%s_%s.%s.nc' % (idir,varn,his,fut,se))
        ddpvn=ds[varnp]
        pct=ds['percentile']
        gpi=ds['gpi']
        return ddpvn

    tvn1=load_vn(varn,fo1,his,px='pc')
    ddpvn1=load_mmm(varn1,varnp)
    pct=ddpvn1['percentile']
    if varn1 in ['ti_ev','gflx','hfss','hfls'] or 'ooplh' in varn1:
        ddpvn1=-ddpvn1

    # variable of interest
    odir1 = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn1)
    if not os.path.exists(odir1):
        os.makedirs(odir1)

    def load_vn(idir0):
        ddpvne=xr.open_dataarray('%s/ddpc.%s_%s_%s.%s.nc' % (idir0,varn1,his,fut,se))
        if varn1=='pr': ddpvne=86400*ddpvne
        return ddpvne

    # load data for each model
    def load_idir(md):
        return '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn1)
    idirs=[load_idir(md0) for md0 in lmd]
    ddpvne=[load_vn(idir0) for idir0 in tqdm(idirs)]
    ddpvne=xr.concat(ddpvne,'model')

    # remap to lat x lon
    tvn1=remap(tvn1,gr)
    ddpvn1=remap(ddpvn1,gr)
    ddpvne=eremap(ddpvne,gr)

    # mask greenland and antarctica
    aagl=pickle.load(open('/project/amp/miyawaki/data/share/aa_gl/cesm2/aa_gl.pickle','rb'))
    aagl[gr['lat']<-60,:]=np.nan # remove fringe grid points not captured by AA mask
    tvn1=tvn1*aagl
    ddpvn1=ddpvn1*aagl
    ddpvne=ddpvne*aagl

    [mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')
    awgt=np.cos(np.deg2rad(mlat)) # area weight

    # make sure nans are consistent
    nidx=np.isnan(ddpvn1)
    for imd in range(ddpvne.shape[0]):
        nidx=np.logical_or(np.isnan(nidx),np.isnan(ddpvne[imd,...]))
    tvn1[nidx]=np.nan
    ddpvn1[nidx]=np.nan
    ddpvne[:,nidx]=np.nan

    # annual mean zonal mean
    annz=np.nanmean(ddpvn1,axis=(0)) # keep zonal var
    ann1=np.nanmean(ddpvn1,axis=(0,-1))
    anne=np.nanmean(ddpvne,axis=(1,-1))

    def make_ah(re):
        ah=np.ones_like(mlat)
        if re=='tr':
            ah[np.abs(gr['lat'])>tlat]=np.nan
        elif re=='ml':
            ah[np.logical_or(np.abs(gr['lat'])<=tlat,np.abs(gr['lat'])>plat)]=np.nan
        elif re=='hl':
            ah[np.abs(gr['lat'])<=plat]=np.nan
        elif re=='et':
            ah[np.abs(gr['lat'])<=tlat]=np.nan
        return ah

    def selreg(re):
        ah=make_ah(re)
        # select region
        ahddp1,kidx=regsla(ddpvn1,gr,ah)
        tvn,_=regsla(tvn1,gr,ah)
        ahddpe=eregsla(ddpvne,gr,ah,kidx)
        # weights
        ahw=regsl2d(awgt,ah,kidx)
        # area weighted mean
        ahddpg=ahddp1.copy()
        ahddp1=np.sum(ahw*ahddp1,axis=-1)/np.sum(ahw)
        ahddpe=np.sum(ahw*ahddpe,axis=-1)/np.sum(ahw)
        tvn=np.sum(ahw*tvn,axis=-1)/np.sum(ahw)
        return ahddp1,ahddpe,tvn

    tr_ahddp1,tr_ahddpe,tr_tvn=selreg('tr')
    et_ahddp1,et_ahddpe,et_tvn=selreg('et')
    ml_ahddp1,ml_ahddpe,ml_tvn=selreg('ml')
    hl_ahddp1,hl_ahddpe,hl_tvn=selreg('hl')

    # if bstype=='bins':
    #     s=ann1.shape
    #     # bootstrap
    #     al=np.ones([s[0],s[1]])
    #     for im in tqdm(range(s[0])):
    #         for ib in range(s[1]):
    #             sa=annz[im,ib,:]
    #             bs=[np.nanmean(resample(sa)) for _ in range(nbs)]
    #             bs=np.array(bs)
    #             pl,pu=np.percentile(bs,100*np.array([alc/2,1-alc/2]))
    #             if pl<0 and pu>0: al[im,ib]=0  # 0 is inside confidence interval
    # elif bstype=='models':
    #     if md=='mmm':
    #         # bootstrap
    #         s=anne.shape
    #         anne=np.reshape(anne,(s[0],s[1]*s[2]))
    #         al=np.ones(ahddpe.shape[1])
    #         for ig in tqdm(range(ahddpe.shape[1])):
    #             sa=ahddpe[:,ig]
    #             bs=[np.nanmean(resample(sa)) for _ in range(nbs)]
    #             bs=np.array(bs)
    #             pl,pu=np.percentile(bs,100*np.array([alc/2,1-alc/2]))
    #             if pl<0 and pu>0: al[ig]=0  # 0 is inside confidence interval
    #         al=np.reshape(al,(s[1],s[2]))

    [ahmlat,ahmpct] = np.meshgrid(gr['lat'],pct, indexing='ij')

    # regions
    tr_ahddp1=np.nanmean(tr_ahddp1,axis=0)
    et_ahddp1=np.nanmean(et_ahddp1,axis=0)
    ml_ahddp1=np.nanmean(ml_ahddp1,axis=0)
    hl_ahddp1=np.nanmean(hl_ahddp1,axis=0)
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    ax.plot(pct,tr_ahddp1,color='tab:orange',label='Tropics')
    ax.plot(pct,et_ahddp1,color='tab:blue',label='Extratropics')
    ax.set_xlabel('Percentile')
    ax.set_ylabel('$\Delta\delta$ %s (%s)'%(varnlb(varn1),unitlb(varn1)))
    ax.set_title('ANN')
    ax.legend(frameon=False)
    fig.savefig('%s/ann.%s.%s%s.ah.pct.reg.png'%(odir1,varn1,fo,fstr),format='png',dpi=600)

    # plot gp vs seasonal cycle of varn1 PCOLORMESH
    fig,ax=plt.subplots(figsize=(5,2.5),constrained_layout=True)
    clf=ax.pcolormesh(ahmlat,ahmpct,np.transpose(ann1),vmin=-vmax(varn1),vmax=vmax(varn1),cmap=cmap(varn1))
    # if bstype=='bins' or md=='mmm':
    #     hatch=plt.fill_between([-0.5,11.5],0,100,hatch='///////',color='none',edgecolor='gray',linewidths=0.3)
    #     ax.pcolormesh(ahmlat,ahmpct,np.ma.masked_where(np.transpose(al==0),np.transpose(ann1)),vmin=-vmax(varn1),vmax=vmax(varn1),cmap=cmap(varn1))
    # ax.contour(ahmlat,ahmpct,np.transpose(ann1),[0],linewidths=0.5,colors='gray')
    # ax.set_title()
    ax.text(0.5,1.05,'ANN',c=ct(varn1),ha='center',va='center',transform=ax.transAxes)
    ax.set_xlabel('Latitude')
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xticks(np.arange(-90,90+30,30))
    ax.set_yticks(100*np.arange(0,1+0.2,0.2))
    if ylb:
        ax.set_ylabel('Percentile')
    else:
        ax.set_yticklabels([])
    if not xlb:
        ax.set_xticklabels([])
    ax.set_xlim([-90,90])
    ax.set_ylim([0,100])
    if 'ooplh'==varnp or varn1 in ['td_mrsos','ti_pr','ti_ev','ti_ro']:
        # clf=ax.pcolormesh(ahmlat,ahmpct,ahc,vmin=0,vmax=vmax(varnc),cmap='Pastel1')
        x=gr['lat']
        y=mppai
        z=cat
        # make dense grid
        scale=100
        yy=ndimage.zoom(y,scale,order=0)
        zz=ndimage.zoom(z,scale,order=0)
        xx=np.linspace(x.min(),x.max(),zz.shape[0])
        # extend out to edge
        xx=np.insert(xx,0,-90.5)
        yy=np.insert(yy,0,0)
        zz=np.insert(zz,0,zz[0,:],axis=0)
        zz=np.insert(zz,0,zz[:,0],axis=1)
        xx=np.append(xx,90.5)
        yy=np.append(yy,1)
        zz=np.insert(zz,-1,zz[-1,:],axis=0)
        zz=np.insert(zz,-1,zz[:,-1],axis=1)
        ax.contour(xx,yy,np.transpose(zz),levels=[0,1,2],vmin=0,vmax=9,cmap='Pastel1',linewidths=1,corner_mask=False)
    if showcb:
        cb=plt.colorbar(clf)
        if titleoverride:
            cb.set_label('%s'%(unitlb(varn1)))
        else:
            cb.set_label('$\Delta\delta$ %s (%s)'%(varnlb(varn1),unitlb(varn1)))
    fig.savefig('%s/ann.%s.%s%s.ah.pct.sign.png' % (odir1,varn1,fo,fstr), format='png', dpi=600)
    fig.savefig('%s/ann.%s.%s%s.ah.pct.sign.pdf' % (odir1,varn1,fo,fstr), format='pdf', dpi=600)

plot()

# if __name__=='__main__':
#     with Pool(max_workers=len(lre)) as p:
#         p.map(plot,lre)
