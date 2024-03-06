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

varn='bcef3'
varny='ef3'

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

fo1 = 'historical' # forcing (e.g., ssp245)
fo2 = 'ssp370' # forcing (e.g., ssp245)
byr1=[1980,2000]
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
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if 'gwl' in byr:
        iname='%s/%s.%s.%s.pickle' % (idir,varn,byr,se)
    else:
        iname='%s/%s.%g-%g.%s.pickle' % (idir,varn,byr[0],byr[1],se)
    return iname

def get_fn_bs(md,imon,igpi,fo,byr):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if 'gwl' in byr:
        iname='%s/%s_bs.%s.%s.%s.pickle' % (idir,varn,byr,se,igpi)
    else:
        iname='%s/%s_bs.%g-%g.%s.%s.pickle' % (idir,varn,byr[0],byr[1],se,igpi)
    return iname

def get_bc(md,fo,byr):
    iname=get_fn(md,fo,byr)
    return pickle.load(open(iname,'rb'))[imon][igpi]

def get_bc_bs(md,imon,igpi,fo,byr):
    iname=get_fn_bs(md,imon,igpi,fo,byr)
    return pickle.load(open(iname,'rb'))

def int_bc(bc,x):
    return np.interp(x,bc[0],bc[1])

def calc_mmm(lmd):
    odir='/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    def loaddata(fo,byr):
        ds=xr.open_dataset(get_vn(varny,md,fo,byr))
        gpi=ds['gpi']
        time=ds['time']
        vn1=ds[varny]
        vn2=xr.open_dataset(get_vn('mrsos',md,fo,byr))['mrsos']
        return vn1,vn2,time,gpi

    print('\n Loading data...')
    vn1h,vn2h,timeh,gpi=loaddata(fo1,byr1)
    vn1f,vn2f,timef,gpi=loaddata(fo2,byr2)
    print('\n Done.')

    print('\n Computing soil moisture anomaly...')
    vn2f=vn2f-vn2h.mean('time')
    vn2h=vn2h-vn2h.mean('time')
    print('\n Done.')

    def selreg(vn):
        vn=vn.sel(time=vn['time.month']==imon+1)
        return vn.data[...,igpi].flatten()

    print('\n Selecting month and gpi...')
    nvn1h=selreg(vn1h)
    nvn2h=selreg(vn2h)
    nvn1f=selreg(vn1f)
    nvn2f=selreg(vn2f)
    print('\n Done.')

    # remove nans
    def rmnan(vn1,vn2,rmg1):
        nans=np.logical_or(np.isnan(vn1),np.isnan(vn2))
        vn1=vn1[~nans]
        vn2=vn2[~nans]
        if rmg1:
            # remove ef over 1
            efg1=vn1>1
            vn1=vn1[~efg1]
            vn2=vn2[~efg1]
        return vn1,vn2

    print('\n Removing nans...')
    nvn1h,nvn2h=rmnan(nvn1h,nvn2h,rmg1)
    nvn1f,nvn2f=rmnan(nvn1f,nvn2f,rmg1)
    print('\n Done.')

    def ckde(vn1,vn2):
        kde=gaussian_kde(np.vstack([vn2,vn1]))
        x1v=np.linspace(np.min(vn1),np.max(vn1),51)
        x2v=np.linspace(np.min(vn2),np.max(vn2),51)
        [x1,x2]=np.meshgrid(x1v,x2v,indexing='ij')
        pdf=kde(np.vstack([x2.ravel(),x1.ravel()]))
        pdf=np.reshape(pdf,x1.shape)
        return pdf,x1,x2,x1v,x2v

    print('\n Computing kde...')
    pdfh,x1h,x2h,x1vh,x2vh=ckde(nvn1h,nvn2h)
    pdff,x1f,x2f,x1vf,x2vf=ckde(nvn1f,nvn2f)
    print('\n Done.')

    print('\n Computing BCs...')
    mbch=get_bc(md,fo1,byr1)
    tbch=get_bc_bs(md,imon,igpi,fo1,byr1)[imon][0]
    mbcf=get_bc(md,fo2,byr2)
    tbcf=get_bc_bs(md,imon,igpi,fo2,byr2)[imon][0]

    print('\n Interpolate to uniform grid...')
    ibch=[int_bc(bc,x2vh) for bc in tqdm(tbch)]
    ibcf=[int_bc(bc,x2vf) for bc in tqdm(tbcf)]
    print('\n Done.')

    def spr(ibc):
        sbc=np.stack(ibc,axis=0)
        abc=np.nanmean(sbc,axis=0)
        dbc=np.nanstd(sbc,axis=0)
        cbc=sms.DescrStatsW(sbc).tconfint_mean() # CI
        pbc=np.percentile(sbc,[5,95],axis=0) # percentiles
        mnbc,mxbc=[np.nanmin(sbc,axis=0),np.nanmax(sbc,axis=0)]
        print('\n Average stdev=%g'%np.nanmean(dbc))
        print('\n Average spread=%g'%np.nanmean(cbc[1]-cbc[0]))
        return abc,dbc,cbc,pbc,mnbc,mxbc

    print('\n Compute spread...')
    abch,dbch,cbch,pbch,mnbch,mxbch=spr(ibch)
    abcf,dbcf,cbcf,pbcf,mnbcf,mxbcf=spr(ibcf)
    print('\n Done.')

    # range
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    ax.fill_between(x2vh,mnbch,mxbch,color='tab:blue',alpha=0.3,edgecolor=None)
    ax.fill_between(x2vf,mnbch,mxbcf,color='tab:orange',alpha=0.3,edgecolor=None)
    # ax.plot(x2vh,abch,'-',color='tab:blue')
    # ax.plot(x2vf,abcf,'-',color='tab:orange')
    ax.plot(mbch[0],mbch[1],'.-',color='tab:blue')
    ax.plot(mbcf[0],mbcf[1],'.-',color='tab:orange')
    ax.set_xlabel('$\delta %s$ (%s)'%(varnlb('mrsos'),unitlb('mrsos')))
    ax.set_ylabel('$%s$ (%s)'%(varnlb(varny),unitlb(varny)))
    # plt.legend(frameon=False,bbox_to_anchor=(1,1),loc='upper left',prop={'size':8})
    fig.savefig('%s/%s.%s.png'%(odir,varn,len(tbch)), format='png', dpi=600)

if __name__=='__main__':
    calc_mmm(md)
