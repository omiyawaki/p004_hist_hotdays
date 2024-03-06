import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import pickle
import pwlf
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from tqdm import tqdm
from util import mods
from utils import corr,corr2d,monname,varnlb,unitlb
from regions import pointlocs,masklev0,settype,retname,regionsets
from CASutils import shapefile_utils as shp

# relb='fourcorners'
# re=['Utah','Colorado','Arizona','New Mexico']

lrelb=['ic']
varn='mrsos'

pct=np.linspace(1,99,101)
ise='sc'
ose='jja'
lmo=[6,7,8]

fo1='historical' # forcings 
yr1='1980-2000'

fo2='ssp370' # forcings 
yr2='2080-2100'

fo='%s+%s'%(fo1,fo2)
fod='%s-%s'%(fo2,fo1)

md1='CESM2'
md2=''
if md2=='':
    md=md1
else:
    md='%s+%s'%(md1,md2)

# load ocean indices
_,omi=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

def proc(idir,yr,mask):
    vn=xr.open_dataset('%s/lm.%s_%s.%s.nc' % (idir,varn,yr,ise))[varn]
    vn=vn.sel(time=vn['time.season']==ose.upper())
    vn=np.delete(vn.data,np.nonzero(np.isnan(mask)),axis=1)
    vn=vn.flatten()
    return np.percentile(vn,pct)

def plot(relb,md1,md2,fo=fo1):
    # mask
    mask=masklev0(re,gr,mtype).data
    mask=mask.flatten()
    mask=np.delete(mask,omi)
    print(mask.shape)

    idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (ise,fo1,md1,varn)
    idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (ise,fo2,md1,varn)
    idir3 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (ise,fo1,md2,varn)
    idir4 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (ise,fo2,md2,varn)

    odir1= '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/regions' % (ose,fo1,md,varn)
    if not os.path.exists(odir1):
        os.makedirs(odir1)
    odir2= '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/regions' % (ose,fo2,md,varn)
    if not os.path.exists(odir2):
        os.makedirs(odir2)
    odir3= '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/regions' % (ose,fo,md,varn)
    if not os.path.exists(odir3):
        os.makedirs(odir3)
    odir4= '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/regions' % (ose,fod,md,varn)
    if not os.path.exists(odir4):
        os.makedirs(odir4)

    pvn1=proc(idir1,yr1,mask)
    pvn2=proc(idir2,yr2,mask)
    if md2 is not '':
        pvn3=proc(idir3,yr1,mask)
        pvn4=proc(idir4,yr2,mask)

    if md2=='':
        tname=r'%s %s' % (md1,retn)
    else:
        tname=r'%s' % (retn)

    oname1='%s/p.%s_%s_%s.%s.%s' % (odir4,varn,yr1,yr2,ose,relb)
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    clf=ax.plot(pct,pvn2-pvn1,color='k',label=md1)
    if md2 is not '':
        clf=ax.plot(pct,pvn4-pvn3,':',color='k',label=md2)
        ax.legend(frameon=False)
    ax.set_title(r'%s' % (tname),fontsize=16)
    ax.set_xlabel('$SM$ Percentile')
    ax.set_ylabel('$\Delta$ %s (%s)'%(varnlb(varn),unitlb(varn)))
    fig.savefig('%s.png'%oname1, format='png', dpi=600)

    oname1='%s/p.%s_%s_%s.%s.%s' % (odir3,varn,yr1,yr2,ose,relb)
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    clf=ax.plot(pct,pvn1,color='tab:blue',label=md1)
    clf=ax.plot(pct,pvn2,color='tab:orange')
    if md2 is not '':
        clf=ax.plot(pct,pvn3,':',color='tab:blue',label=md2)
        clf=ax.plot(pct,pvn4,':',color='tab:orange')
        ax.legend(frameon=False)
    ax.set_title(r'%s' % (tname),fontsize=16)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('%s (%s)'%(varnlb(varn),unitlb(varn)))
    fig.savefig('%s.png'%oname1, format='png', dpi=600)

    oname1='%s/p.%s_%s.%s.%s' % (odir1,varn,yr1,ose,relb)
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    clf=ax.plot(pct,pvn1,color='tab:blue',label=md1)
    if md2 is not '':
        clf=ax.plot(pct,pvn3,':',color='tab:blue',label=md2)
        ax.legend(frameon=False)
    ax.set_title(r'%s' % (tname),fontsize=16)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('%s (%s)'%(varnlb(varn),unitlb(varn)))
    fig.savefig('%s.png'%oname1, format='png', dpi=600)

    oname1='%s/p.%s_%s.%s.%s' % (odir2,varn,yr2,ose,relb)
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    clf=ax.plot(pct,pvn2,color='tab:orange',label=md1)
    if md2 is not '':
        clf=ax.plot(pct,pvn4,':',color='tab:orange',label=md2)
        ax.legend(frameon=False)
    ax.set_title(r'%s' % (tname),fontsize=16)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('%s (%s)'%(varnlb(varn),unitlb(varn)))
    fig.savefig('%s.png'%oname1, format='png', dpi=600)

for relb in lrelb:
    mtype=settype(relb)
    retn=retname(relb)
    re=regionsets(relb)
    plot(relb,md1,md2,fo=fo1)
