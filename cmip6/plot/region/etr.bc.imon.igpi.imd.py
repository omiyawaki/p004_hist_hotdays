import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import pickle
import numpy as np
import xesmf as xe
import xarray as xr
import constants as c
from tqdm import tqdm
from util import mods,simu,emem
from utils import monname,varnlb,unitlb
from scipy.stats import gaussian_kde
from glade_utils import grid
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms

# collect warmings across the ensembles

varn='bc'
varny='hfls'

if varny=='ef':
    rmg1=True
else:
    rmg1=False

se='sc'
nt=7
p=95
doy=False
only95=True

####################################
# for test plot
imon=0
# igpi=9925
igpi=7015

fo0 = 'historical' # forcing (e.g., ssp245)
byr0=[1980,2000]

fo = 'historical' # forcing (e.g., ssp245)
byr=[1980,2000]

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr='gwl2.0'

freq='day'

# lmd=mods(fo) # create list of ensemble members
md='CESM2'

####################################

# load land indices
lmi,omi=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

[mlat,mlon] = np.meshgrid(gr['lat'], gr['lon'], indexing='ij')
vlat=mlat.flatten()
vlon=mlon.flatten()
vlat=np.delete(vlat,omi)
vlon=np.delete(vlon,omi)
print('Test month is %g, location is lat=%g, lon=%g'%(imon+1,vlat[igpi],vlon[igpi]))

def get_vn(varn,md,fo,byr):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if 'gwl' in byr:
        fn='%s/lm.%s_%s.%s.nc' % (idir,varn,byr,se)
    else:
        fn='%s/lm.%s_%g-%g.%s.nc' % (idir,varn,byr[0],byr[1],se)
    return fn

def get_fn(md):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if 'gwl' in byr:
        iname='%s/%s.%s.%s.pickle' % (idir,varn,byr,se)
    else:
        iname='%s/%s.%g-%g.%s.pickle' % (idir,varn,byr[0],byr[1],se)
    return iname

def get_fn_bs(md,imon,igpi):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if 'gwl' in byr:
        iname='%s/%s_bs.%s.%s.%s.pickle' % (idir,varn,byr,se,igpi)
    else:
        iname='%s/%s_bs.%g-%g.%s.%s.pickle' % (idir,varn,byr[0],byr[1],se,igpi)
    return iname

def get_bc(md):
    iname=get_fn(md)
    return pickle.load(open(iname,'rb'))[imon][igpi]

def get_bc_bs(md,imon,igpi):
    iname=get_fn_bs(md,imon,igpi)
    return pickle.load(open(iname,'rb'))

def int_bc(bc,x):
    return np.interp(x,bc[0],bc[1])

def calc_mmm(lmd):
    odir='/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Loading data...')
    ds=xr.open_dataset(get_vn(varny,md,fo,byr))
    gpi=ds['gpi']
    time=ds['time']
    vn1=ds[varny]
    vn2=xr.open_dataset(get_vn('mrsos',md,fo,byr))['mrsos']
    if fo=='historical':
        vn0=vn2.copy()
    else:
        vn0=xr.open_dataset(get_vn('mrsos',md,fo0,byr0))['mrsos']
    print('\n Done.')

    print('\n Computing soil moisture anomaly...')
    vn2=vn2-vn0.mean('time')
    print('\n Done.')

    print('\n Selecting month and gpi...')
    svn1=vn1.sel(time=vn1['time.month']==imon+1)
    svn2=vn2.sel(time=vn2['time.month']==imon+1)
    nvn1=svn1.data[...,igpi].flatten()
    nvn2=svn2.data[...,igpi].flatten()
    # remove nans
    nans=np.logical_or(np.isnan(nvn1),np.isnan(nvn2))
    nvn1=nvn1[~nans]
    nvn2=nvn2[~nans]
    if rmg1:
        # remove ef over 1
        efg1=nvn1>1
        nvn1=nvn1[~efg1]
        nvn2=nvn2[~efg1]
    print('\n Done.')

    print('\n Computing kde...')
    kde=gaussian_kde(np.vstack([nvn2,nvn1]))
    x1v=np.linspace(np.min(nvn1),np.max(nvn1),51)
    x2v=np.linspace(np.min(nvn2),np.max(nvn2),51)
    [x1,x2]=np.meshgrid(x1v,x2v,indexing='ij')
    pdf=kde(np.vstack([x2.ravel(),x1.ravel()]))
    pdf=np.reshape(pdf,x1.shape)
    print('\n Done.')

    print('\n Computing BCs...')
    mbc=get_bc(md)
    tbc=get_bc_bs(md,imon,igpi)[imon][0]
    # lbc=[get_bc(md) for md in tqdm(lmd)] # [model][month][gpi][sm][lh]

    print('\n Interpolate to uniform grid...')
    ibc=[int_bc(bc,x2v) for bc in tqdm(tbc)] # [model][month][gpi][sm][lh]
    print('\n Done.')

    print('\n Compute spread...')
    sbc=np.stack(ibc,axis=0)
    abc=np.nanmean(sbc,axis=0)
    dbc=np.nanstd(sbc,axis=0)
    cbc=sms.DescrStatsW(sbc).tconfint_mean() # CI
    pbc=np.percentile(sbc,[5,95],axis=0) # percentiles
    mnbc,mxbc=[np.nanmin(sbc,axis=0),np.nanmax(sbc,axis=0)] # range
    print('\n Average stdev=%g'%np.nanmean(dbc))
    print('\n Average spread=%g'%np.nanmean(cbc[1]-cbc[0]))
    print('\n Done.')

    # range
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    # ax.contour(x2,x1,pdf,np.logspace(-2,0,11))
    ax.fill_between(x2v,mnbc,mxbc,color='k',alpha=0.3,edgecolor=None)
    # ax.plot(x2v,abc,'-',color='tab:blue')
    ax.scatter(nvn2,nvn1,s=0.5,c='k')
    ax.plot(mbc[0],mbc[1],'.-k')
    ax.set_xlabel('$\delta %s$ (%s)'%(varnlb('mrsos'),unitlb('mrsos')))
    ax.set_ylabel('$%s$ (%s)'%(varnlb(varny),unitlb(varny)))
    # plt.legend(frameon=False,bbox_to_anchor=(1,1),loc='upper left',prop={'size':8})
    fig.savefig('%s/%s.%s.png'%(odir,varn,len(tbc)), format='png', dpi=600)

    # plot spaghetti
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    [ax.plot(bc[0],bc[1],'.-',color='tab:blue',alpha=0.2) for i,bc in enumerate(tbc)]
    ax.plot(mbc[0],mbc[1],'.-k')
    ax.scatter(nvn2,nvn1,s=0.5,c='k')
    ax.set_xlabel('$\delta %s$ (%s)'%(varnlb('mrsos'),unitlb('mrsos')))
    ax.set_ylabel('$%s$ (%s)'%(varnlb(varny),unitlb(varny)))
    # plt.legend(frameon=False,bbox_to_anchor=(1,1),loc='upper left',prop={'size':8})
    fig.savefig('%s/%s.%s.spagh.png'%(odir,varn,len(tbc)), format='png', dpi=600)

if __name__=='__main__':
    calc_mmm(md)
