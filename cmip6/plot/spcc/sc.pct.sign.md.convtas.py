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
# lre=['et','tr'] # tr=tropics, ml=midlatitudes, hl=high lat, et=extratropics
lre=['hl'] # tr=tropics, ml=midlatitudes, hl=high lat, et=extratropics
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
titleoverride=True
fs=(3.5,3)
pds=(1,0.5)
axs=(1.5,2)
h=[Size.Fixed(pds[0]), Size.Fixed(axs[0])]
v=[Size.Fixed(pds[1]), Size.Fixed(axs[1])]

p=97.5 # percentile
varn='tas'
varn1='advt_rmean850_t'
varnp='advt_rmean850_t'
cvn='%s+%s'%(varn,varn1)
reverse=True
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

# lmd=['CESM2']
# md='CESM2'

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
    return 'RdBu_r'

def vmax(vn):
    return 1 # original 1

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
        'huss':         r'$q$',
        'hurs':         r'$RH$',
        'mrsos':        r'$SM$',
        'td_mrsos':     r'$SM_{\mathrm{30\,d}}$',
        'ti_pr':        r'$P_{\mathrm{30\,d}}$',
        'ti_ev':        r'$-E_{\mathrm{30\,d}}$',
        'ti_ro':        r'$-R_{\mathrm{30\,d}}$',
        'pr':           r'$P$',
        'rfa':          r'$-\langle\nabla\cdotF_a\rangle$',
        'ta850':        r'$T_{850}$',
        'pblh':         r'$\mathrm{PBLH}$',
        'wap850':       r'$\omega_{850}$',
        'wapt850':      r'$(\omega T)_{850}$',
        'fa850':        r'$-\nabla\cdotF_{a,\,850}$',
        'fat850':       r'$-\nabla\cdot(vc_pT)_{850}$',
        'advt850_wm2':  r'$-\rho z_{850}(uc_p\partial_xT+vc_p\partial_yT)_{850}$',
        'advt850':      r'$-a*(uc_p\partial_xT+vc_p\partial_yT)_{850}$',
        'advtx850':     r'$-(uc_p\partial_xT)_{850}$',
        'advty850':     r'$-(vc_p\partial_yT)_{850}$',
        'advm850':      r'$-(u\partial_xm+v\partial_ym)_{850}$',
        'advmx850':     r'$-(u\partial_xm)_{850}$',
        'advmy850':     r'$-(v\partial_ym)_{850}$',
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
        tstr='$a %s$'%varnlb(varn1)
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
        sv=np.roll(v,6,axis=0) # seasonality shifted by 6 months
        v[:,:,gr['lat']<0,:]=sv[:,:,gr['lat']<0,:]
        return regsl(v,ma)

    def eregsl(v,ma,kidx):
        v=v*np.moveaxis(ma[...,None],-1,0)
        v=np.reshape(v,[v.shape[0],v.shape[1],v.shape[2],v.shape[3]*v.shape[4]])
        return v[...,kidx]

    def eregsla(v,gr,ma,kidx):
        sv=np.roll(v,6,axis=1) # seasonality shifted by 6 months
        v[:,:,:,gr['lat']<0,:]=sv[:,:,:,gr['lat']<0,:]
        return eregsl(v,ma,kidx)

    def regsl2d(v,ma,kidx):
        v=v*ma
        v=np.reshape(v,[v.shape[0]*v.shape[1]])
        return v[kidx]

    def sortai(v):
        idx=np.argsort(v)
        return v[idx],idx

    def load_vn(varn,fo,byr,px='m'):
        if '_wm2' in varn: varn=varn.replace('_wm2','')
        idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
        return xr.open_dataarray('%s/%s.%s_%s.%s.nc' % (idir,px,varn,byr,se))

    tvn1=load_vn(varn,fo1,his,px='pc')

    # variable of interest
    odir1 = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn1)
    if not os.path.exists(odir1):
        os.makedirs(odir1)

    def load_vn(varn,idir0):
        if '_wm2' in varn: varn=varn.replace('_wm2','')
        ddpvne=xr.open_dataarray('%s/ddpc.md.%s_%s_%s.%s.nc' % (idir0,varn,his,fut,se))
        if varn=='pr': ddpvne=86400*ddpvne
        if '_wm2' in varn: ddpvne=1.16*1500*ddpvne # rho*z850
        return ddpvne

    # load data for each model
    def load_idir(varn,md):
        if '_wm2' in varn: varn=varn.replace('_wm2','')
        return '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    idirs=[load_idir(varn1,md0) for md0 in lmd]
    ddpvne=[load_vn(varn1,idir0) for idir0 in tqdm(idirs)]
    ddpvne=xr.concat(ddpvne,'model')
    pct=ddpvne['percentile']

    def load_rgr(md):
        idira='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s'%(se,fo1,md,cvn)
        return xr.open_dataarray('%s/rgr.%s.nc' % (idira,cvn))

    # load regression coef
    ae=[load_rgr(md) for md in tqdm(lmd)]
    ae=xr.concat(ae,'model')
    ae=ae.expand_dims(dim={'percentile':len(pct)},axis=2)

    # convert to tas
    ddpvne=ae*ddpvne
    ddpvn1=ddpvne.mean('model')

    # remap to lat x lon
    tvn1=remap(tvn1,gr)
    ddpvn1=remap(ddpvn1,gr)
    ddpvne=eremap(ddpvne,gr)

    # mask greenland and antarctica
    aagl=pickle.load(open('/project/amp/miyawaki/data/share/aa_gl/cesm2/aa_gl.pickle','rb'))
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
    tvn1,_=regsla(tvn1,gr,ah)
    ahddpe=eregsla(ddpvne,gr,ah,kidx)
    # weights
    ahw=regsl2d(awgt,ah,kidx)

    # area weighted mean
    ahddpg=ahddp1.copy()
    ahddp1=np.sum(ahw*ahddp1,axis=-1)/np.sum(ahw)
    ahddpe=np.sum(ahw*ahddpe,axis=-1)/np.sum(ahw)
    tvn1=np.sum(ahw*tvn1,axis=-1)/np.sum(ahw)

    if bstype=='bins':
        s=ahddp1.shape
        # bootstrap
        al=np.ones([s[0],s[1]])
        for im in tqdm(range(s[0])):
            for ib in range(s[1]):
                sa=ahddpg[im,ib,:]
                bs=[np.nanmean(resample(sa)) for _ in range(nbs)]
                bs=np.array(bs)
                pl,pu=np.percentile(bs,100*np.array([alc/2,1-alc/2]))
                if pl<0 and pu>0: al[im,ib]=0  # 0 is inside confidence interval
    elif bstype=='models':
        if md=='mmm':
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
    [ahmmon,ahmpct] = np.meshgrid(mon,pct, indexing='ij')

    # # load et regimes data
    # varnc='cat'
    # ddir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,'mrsos')
    # cat=xr.open_dataarray('%s/sc.%s.%s_%s.%s.nc'%(ddir,varnc,his,fut,re))

    # ann mean
    aahddp1=np.nanmean(ahddp1,axis=0)
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    ax.plot(pct,aahddp1,'-k')
    ax.set_xlabel('Percentile')
    ax.set_ylabel('$\Delta\delta$ %s (%s)'%(varnlb(varn1),unitlb(varn1)))
    ax.set_title('%s ANN'%tstr)
    fig.savefig('%s/ann.%s.%s%s.ah.pct.sign.%s.convtas.png'%(odir1,varn1,fo,fstr,re),format='png',dpi=600)

    # plot gp vs seasonal cycle of varn1 PCOLORMESH
    fig=plt.figure(figsize=fs)
    divider=Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax=fig.add_axes(divider.get_position(),axes_locator=divider.new_locator(nx=1, ny=1))
    clf=ax.pcolormesh(ahmmon,ahmpct,ahddp1,vmin=-vmax(varn1),vmax=vmax(varn1),cmap=cmap(varn1))
    if bstype=='bins' or md=='mmm':
        hatch=plt.fill_between([-0.5,11.5],0,100,hatch='///////',color='none',edgecolor='gray',linewidths=0.3)
        ax.pcolormesh(ahmmon,ahmpct,np.ma.masked_where(al==0,ahddp1),vmin=-vmax(varn1),vmax=vmax(varn1),cmap=cmap(varn1))
    # ax.contour(ahmmon,ahmpct,tvn1,[273.15])
    # ax.set_title()
    ax.text(0.5,1.05,tstr,c=ct(varn1),ha='center',va='center',transform=ax.transAxes)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xticks(np.arange(1,11+2,2))
    ax.set_xticklabels(np.arange(2,12+2,2))
    ax.set_yticks(100*np.arange(0,1+0.2,0.2))
    # ax.set_xticklabels(np.arange(2,12+2,2))
    if ylb:
        ax.set_ylabel('Percentile')
    else:
        ax.set_yticklabels([])
    if not xlb:
        ax.set_xticklabels([])
    ax.set_xlim([-0.5,11.5])
    ax.set_ylim([0,100])
    # if 'ooplh'==varnp or varn1 in ['td_mrsos','ti_pr','ti_ev','ti_ro']:
    #     # clf=ax.pcolormesh(ahmmon,ahmpct,ahc,vmin=0,vmax=vmax(varnc),cmap='Pastel1')
    #     x=np.arange(12)
    #     y=mppai
    #     z=cat
    #     # make dense grid
    #     scale=100
    #     yy=ndimage.zoom(y,scale,order=0)
    #     zz=ndimage.zoom(z,scale,order=0)
    #     xx=np.linspace(x.min(),x.max(),zz.shape[0])
    #     # extend out to edge
    #     xx=np.insert(xx,0,-0.5)
    #     yy=np.insert(yy,0,0)
    #     zz=np.insert(zz,0,zz[0,:],axis=0)
    #     zz=np.insert(zz,0,zz[:,0],axis=1)
    #     xx=np.append(xx,12.5)
    #     yy=np.append(yy,1)
    #     zz=np.insert(zz,-1,zz[-1,:],axis=0)
    #     zz=np.insert(zz,-1,zz[:,-1],axis=1)
    #     ax.contour(xx,yy,np.transpose(zz),levels=[0,1,2],vmin=0,vmax=9,cmap='Pastel1',linewidths=1,corner_mask=False)
    if showcb:
        cb=plt.colorbar(clf,cax=fig.add_axes([(pds[0]+axs[0]+0.15)/fs[0],pds[1]/fs[1],0.1/fs[0],axs[1]/fs[1]]))
        if titleoverride:
            cb.set_label('K')
        else:
            cb.set_label('$\Delta\delta A(%s)$ (K)'%(varnlb(varn1)))
    fig.savefig('%s/sc.%s.%s%s.ah.pct.sign.%s.md.convtas.png' % (odir1,varn1,fo,fstr,re), format='png', dpi=600,backend='pgf')

[plot(re) for re in tqdm(lre)]

# if __name__=='__main__':
#     with Pool(max_workers=len(lre)) as p:
#         p.map(plot,lre)
