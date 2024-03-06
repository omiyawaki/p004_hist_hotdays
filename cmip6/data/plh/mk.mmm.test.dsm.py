import os
import sys
sys.path.append('../')
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
from glade_utils import grid
from etregimes import bestfit
import matplotlib.pyplot as plt

# collect warmings across the ensembles

varn='bc'
varn1='hfls'
varn2='mrsos'
se='sc'
nt=7
p=95
doy=False
only95=True

####################################
# for test plot
imon=1
# igpi=9925
igpi=7015

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
print('Test month is %g, location is lat=%g, lon=%g'%(imon-1,vlat[igpi],vlon[igpi]))

####################################

fo = 'historical' # forcing (e.g., ssp245)
yr='1980-2000'

# fo = 'ssp370' # forcing (e.g., ssp245)
# byr='2080-2100'

freq='day'

lmd=mods(fo) # create list of ensemble members

def load_hotdata(md,fo,varn,yr):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    ds=xr.open_dataset('%s/pc.%s_%s.%s.nc' % (idir,varn,yr,se))
    ds=ds.sel(percentile=p)
    vn=ds[varn].squeeze()[imon,igpi]
    return vn

def load_meandata(md,fo,varn,yr):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if md=='mmm':
        ds=xr.open_dataset('%s/cm.%s_%s.%s.nc' % (idir,varn,yr,se))
    else:
        ds=xr.open_dataset('%s/lm.%s_%s.%s.nc' % (idir,varn,yr,se))
        ds=ds.groupby('time.month').mean(dim='time')
    vn=ds[varn][imon,igpi]
    return vn

def selreg(md,varn,fo=fo,yr=yr,dtype='mean'):
    if dtype=='mean':
        vn=load_meandata(md,fo,varn,yr)
    elif dtype=='hot':
        vn=load_hotdata(md,fo,varn,yr)
    return vn

def get_fn(md):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,'bc')
    iname='%s/%s.%s.%s.pickle' % (idir,varn,yr,se)
    return iname

def get_bc(md):
    iname=get_fn(md)
    return pickle.load(open(iname,'rb'))[imon][igpi]

def calc_mmm(lmd):
    odir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,'mmm',varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Computing BCs...')
    mbc=get_bc('mmm')
    lpsm1=[selreg(md,varn2,fo=fo,yr=yr,dtype='hot') for md in lmd]
    lplh1=[selreg(md,varn1,fo=fo,yr=yr,dtype='hot') for md in lmd]
    lmsm1=[selreg(md,varn2,fo=fo,yr=yr,dtype='mean') for md in lmd]
    # lmlh1=[selreg(md,varn1,fo=fo,yr=yr,dtype='mean') for md in lmd]
    ldsm1=[psm1-msm1 for (psm1,msm1) in zip(lpsm1,lmsm1)]
    ldlh1=[np.interp(dsm1,mbc[0],mbc[1]) for dsm1 in ldsm1]

    # plot
    fig,ax=plt.subplots(figsize=(5,3),constrained_layout=True)
    ax.plot(mbc[0],mbc[1],'.-k',label='%s'%'MMM')
    [ax.plot(dsm1,dlh1,'+',label='%s'%lmd[i].upper()) for i,(dsm1,dlh1) in enumerate(zip(ldsm1,ldlh1))]
    ax.set_xlabel('$\delta %s$ (%s)'%(varnlb('mrsos'),unitlb('mrsos')))
    ax.set_ylabel('$%s$ (%s)'%(varnlb('hfls'),unitlb('hfls')))
    plt.legend(frameon=False,bbox_to_anchor=(1,1),loc='upper left',prop={'size':8})
    fig.savefig('test.sm.dsm.png', format='png', dpi=600)

if __name__=='__main__':
    calc_mmm(lmd)
