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
from scipy.stats import linregress
from glade_utils import grid
from etregimes import bestfit
import matplotlib.pyplot as plt
import statsmodels.api as sm

# collect warmings across the ensembles

varn='blh'
se='sc'
doy=False
only95=True

####################################
# for test plot
imon=0
# igpi=9925
igpi=7015

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

def calc_mmm(lmd):
    odir='/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('\n Loading data...')
    ds=xr.open_dataset(get_vn('hfls',md,fo,byr))
    gpi=ds['gpi']
    time=ds['time']
    vn1=ds['hfls']
    vn2=xr.open_dataset(get_vn('lh2ce',md,fo,byr))['lh2ce']
    print('\n Done.')

    print('\n Selecting month and gpi...')
    svn1=vn1.sel(time=vn1['time.month']==imon+1)
    svn2=vn2.sel(time=vn2['time.month']==imon+1)
    nvn1=svn1.data[...,igpi].flatten()
    nvn2=svn2.data[...,igpi].flatten()
    nans=np.logical_or(np.isnan(nvn1),np.isnan(nvn2))
    nvn1=nvn1[~nans]
    nvn2=nvn2[~nans]
    print('\n Done.')

    print('\n Computing kde...')
    kde=gaussian_kde(np.vstack([nvn2,nvn1]))
    x1v=np.linspace(np.min(nvn1),np.max(nvn1),51)
    x2v=np.linspace(np.min(nvn2),np.max(nvn2),51)
    [x1,x2]=np.meshgrid(x1v,x2v,indexing='ij')
    pdf=kde(np.vstack([x2.ravel(),x1.ravel()]))
    pdf=np.reshape(pdf,x1.shape)
    print('\n Done.')

    print('\n Computing regression...')
    lrm=sm.OLS(nvn1,nvn2)
    lrr=lrm.fit()
    print('\n Done.')

    # plot
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    ax.scatter(nvn2,nvn1,s=0.5,c='gray')
    ax.contour(x2,x1,pdf)
    ax.plot(x2v,lrr.params[0]*x2v)
    ax.annotate(r'$R^2=%.2f$'%lrr.rsquared,xy=(0.05,0.9),xycoords='axes fraction')
    ax.set_xlabel(r'$%s$ (%s)'%(varnlb('lh2ce'),unitlb('lh2ce')))
    ax.set_ylabel(r'$%s$ (%s)'%(varnlb('hfls'),unitlb('hfls')))
    # plt.legend(frameon=False,bbox_to_anchor=(1,1),loc='upper left',prop={'size':8})
    fig.savefig('%s/%s.png'%(odir,varn), format='png', dpi=600)

if __name__=='__main__':
    calc_mmm(md)
