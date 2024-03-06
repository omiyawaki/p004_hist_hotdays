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
from etregimes import bestfit
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms

# collect warmings across the ensembles

varn='bc'
se='sc'
nt=7
p=95
doy=False
only95=True

####################################
# for test plot
imon=0
igpi=7033
# igpi=7015

fo0 = 'historical' # forcing (e.g., ssp245)
byr0=[1980,2000]

fo1 = 'historical' # forcing (e.g., ssp245)
byr1=[1980,2000]

fo2 = 'ssp370' # forcing (e.g., ssp245)
byr2='gwl2.0'

fo='%s+%s'%(fo1,fo2)
byr='%s+%s'%(byr1,byr2)

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

def get_fn(md,fo,byr):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,'bc')
    if 'gwl' in byr:
        iname='%s/%s.%s.%s.pickle' % (idir,varn,byr,se)
    else:
        iname='%s/%s.%g-%g.%s.pickle' % (idir,varn,byr[0],byr[1],se)
    return iname

def get_fn_bs(md,fo,byr,imon,igpi):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,'bc')
    if 'gwl' in byr:
        iname='%s/%s_bs.%s.%s.%s.pickle' % (idir,varn,byr,se,igpi)
    else:
        iname='%s/%s_bs.%g-%g.%s.%s.pickle' % (idir,varn,byr[0],byr[1],se,igpi)
    return iname

def get_bc(md,fo,byr):
    iname=get_fn(md,fo,byr)
    return pickle.load(open(iname,'rb'))[imon][igpi]

def get_bc_bs(md,fo,byr,imon,igpi):
    iname=get_fn_bs(md,fo,byr,imon,igpi)
    return pickle.load(open(iname,'rb'))

def int_bc(bc,x):
    return np.interp(x,bc[0],bc[1])

def calc_mmm(lmd):
    odir1='/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo1,md,varn)
    if not os.path.exists(odir1):
        os.makedirs(odir1)
    odir='/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Loading data...')
    ds=xr.open_dataset(get_vn('hfls',md,fo1,byr1))
    gpi=ds['gpi']
    time=ds['time']
    vn11=ds['hfls']
    vn12=xr.open_dataset(get_vn('mrsos',md,fo1,byr1))['mrsos']
    vn21=xr.open_dataset(get_vn('hfls',md,fo2,byr2))['hfls']
    vn22=xr.open_dataset(get_vn('mrsos',md,fo2,byr2))['mrsos']
    print('\n Done.')

    print('\n Computing soil moisture anomaly...')
    vn12=vn12-vn12.mean('time')
    vn22=vn22-vn12.mean('time')
    print('\n Done.')

    print('\n Selecting month and gpi...')
    def selmg(vn1):
        svn1=vn1.sel(time=vn1['time.month']==imon+1)
        return svn1.data[...,igpi].flatten()
        return
    nvn11=selmg(vn11)
    nvn12=selmg(vn12)
    nvn21=selmg(vn21)
    nvn22=selmg(vn22)
    print('\n Done.')

    nans=np.logical_or(np.isnan(nvn11),np.isnan(nvn12))
    nvn11=nvn11[~nans]
    nvn12=nvn12[~nans]
    nans=np.logical_or(np.isnan(nvn21),np.isnan(nvn22))
    nvn21=nvn21[~nans]
    nvn22=nvn22[~nans]

    print('\n Computing kde...')
    def ckde(nvn1,nvn2):
        kde=gaussian_kde(np.vstack([nvn2,nvn1]))
        x1v=np.linspace(np.min(nvn1),np.max(nvn1),51)
        x2v=np.linspace(np.min(nvn2),np.max(nvn2),51)
        [x1,x2]=np.meshgrid(x1v,x2v,indexing='ij')
        pdf=kde(np.vstack([x2.ravel(),x1.ravel()]))
        return np.reshape(pdf,x1.shape),x2v,x2,x1
    pdf1,x2v,x2,x1=ckde(nvn11,nvn12)
    pdf2,_,_,_=ckde(nvn21,nvn22)
    print('\n Done.')

    print('\n Computing BCs...')
    mbc1=get_bc(md,fo1,byr1)
    tbc1=get_bc_bs(md,fo1,byr1,imon,igpi)[imon][0]
    mbc2=get_bc(md,fo2,byr2)
    tbc2=get_bc_bs(md,fo2,byr2,imon,igpi)[imon][0]
    # lbc=[get_bc(md) for md in tqdm(lmd)] # [model][month][gpi][sm][lh]

    print('\n Interpolate to uniform grid...')
    ibc1=[int_bc(bc,x2v) for bc in tqdm(tbc1)] # [model][month][gpi][sm][lh]
    ibc2=[int_bc(bc,x2v) for bc in tqdm(tbc2)] # [model][month][gpi][sm][lh]
    print('\n Done.')

    print('\n Compute spread...')
    def cspr(ibc):
        sbc=np.stack(ibc,axis=0)
        abc=np.nanmean(sbc,axis=0)
        dbc=np.nanstd(sbc,axis=0)
        cbc=sms.DescrStatsW(sbc).tconfint_mean() # CI
        pbc=np.percentile(sbc,[5,95],axis=0) # percentiles
        print('\n Average stdev=%g'%np.nanmean(dbc))
        print('\n Average spread=%g'%np.nanmean(cbc[1]-cbc[0]))
        print('\n Done.')
        return abc,dbc,cbc,pbc
    abc1,dbc1,cbc1,pbc1=cspr(ibc1)
    abc2,dbc2,cbc2,pbc2=cspr(ibc2)

    # plot 5-95 percentile
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    ax.contour(x2,x1,pdf1)
    ax.fill_between(x2v,pbc1[0],pbc1[1],alpha=0.3)
    ax.plot(x2v,abc1,'-',color='tab:blue')
    ax.plot(mbc1[0],mbc1[1],'.-k')
    ax.set_xlabel('$\delta %s$ (%s)'%(varnlb('mrsos'),unitlb('mrsos')))
    ax.set_ylabel('$%s$ (%s)'%(varnlb('hfls'),unitlb('hfls')))
    fig.savefig('%s/%s.%s.png'%(odir1,varn,len(tbc1)), format='png', dpi=600)

    # plot spaghetti
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    ax.scatter(nvn12,nvn11,s=0.5,c='lightblue')
    [ax.plot(bc[0],bc[1],'.-',color='lightblue',alpha=0.2) for i,bc in enumerate(tbc1)]
    ax.plot(mbc1[0],mbc1[1],'.-',color='royalblue')
    ax.set_xlabel('$\delta %s$ (%s)'%(varnlb('mrsos'),unitlb('mrsos')))
    ax.set_ylabel('$%s$ (%s)'%(varnlb('hfls'),unitlb('hfls')))
    fig.savefig('%s/%s.%s.spagh.png'%(odir1,varn,len(tbc1)), format='png', dpi=600)

    # plot spaghetti
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    [ax.plot(bc[0],bc[1],'.-',color='lightblue',alpha=0.2) for i,bc in enumerate(tbc1)]
    [ax.plot(bc[0],bc[1],'.-',color='bisque',alpha=0.2) for i,bc in enumerate(tbc2)]
    ax.plot(mbc1[0],mbc1[1],'.-',color='royalblue')
    ax.plot(mbc2[0],mbc2[1],'.-',color='darkorange')
    ax.set_xlabel('$\delta %s$ (%s)'%(varnlb('mrsos'),unitlb('mrsos')))
    ax.set_ylabel('$%s$ (%s)'%(varnlb('hfls'),unitlb('hfls')))
    fig.savefig('%s/%s.%s.spagh.png'%(odir,varn,len(tbc1)), format='png', dpi=600)

if __name__=='__main__':
    calc_mmm(md)
