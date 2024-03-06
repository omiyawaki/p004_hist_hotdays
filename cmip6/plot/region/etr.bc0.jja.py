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
from regions import pointlocs,masklev0,settype,retname,regionsets,masklev1
from CASutils import shapefile_utils as shp
from etregimes import bestfit

# relb='cp'
# re=['Kansas','Nebraska','Iowa','Missouri']

relb='ic'

pct=np.linspace(1,99,101)
p=95
varn1='hfls'
varn2='mrsos'
varn3='pr'
vr=[0,15]
bvn2=np.arange(0,50+2,2)
varn='%s+%s'%(varn1,varn2)
se='sc'

fo1='historical' # forcings 
yr1='1980-2000'

fo2='ssp370' # forcings 
yr2='2080-2100'

fo='%s+%s'%(fo1,fo2)
fod='%s-%s'%(fo2,fo1)
yr='%s+%s'%(yr1,yr2)

# lmd=mods(fo1)
lmd=['CESM2']
mmm=False
# fo=fo1
# yr=yr1

def set_clim(relb):
    clim={  
            'ic':   [0,15],
            'fc':   [0,8],
            'cp':   [0,8],
            }
    return clim[relb]

# load ocean indices
_,omi=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# grid
rgdir='/project/amp/miyawaki/data/share/regrid'
# open CESM data to get output grid
cfil='%s/f.e11.F1850C5CNTVSST.f09_f09.002.cam.h0.PHIS.040101-050012.nc'%rgdir
cdat=xr.open_dataset(cfil)
gr=xr.Dataset({'lat': (['lat'], cdat['lat'].data)}, {'lon': (['lon'], cdat['lon'].data)})

def load_data(md,fo,varn,mask,yr):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    ds=xr.open_dataset('%s/lm.%s_%s.%s.nc' % (idir,varn,yr,se))
    vn=ds[varn]
    if varn=='pr': vn=86400*vn # convert pr to mm/d
    time=ds['time']
    # delete data outside masked region
    vn=np.delete(vn.data,np.nonzero(np.isnan(mask)),axis=1)
    vn=xr.DataArray(vn,coords={'time':time,'gpi':range(vn.shape[1])},dims=('time','gpi'))
    return vn

def load_hotdata(md,fo,varn,mask,yr):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    ds=xr.open_dataset('%s/pc.%s_%s.%s.nc' % (idir,varn,yr,se))
    ds=ds.sel(percentile=p)
    vn=ds[varn].squeeze()
    gpi=ds['gpi']
    nd,ng=vn.shape
    if varn=='pr': vn=86400*vn # convert pr to mm/d
    month=ds['month']
    # delete data outside masked region
    vn=np.delete(vn.data,np.nonzero(np.isnan(mask)),axis=1)
    vn=np.nanmean(vn,axis=1)
    return vn

def load_meandata(md,fo,varn,mask,yr):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    if md=='mmm':
        ds=xr.open_dataset('%s/cm.%s_%s.%s.nc' % (idir,varn,yr,se))
    else:
        ds=xr.open_dataset('%s/lm.%s_%s.%s.nc' % (idir,varn,yr,se))
        ds=ds.groupby('time.month').mean(dim='time')
    vn=ds[varn]
    if varn=='pr': vn=86400*vn # convert pr to mm/d
    # delete data outside masked region
    vn=np.delete(vn.data,np.nonzero(np.isnan(mask)),axis=1)
    vn=np.nanmean(vn,axis=1)
    return vn

def selreg(md,varn,relb,fo=fo1,yr=yr1):
    # mask
    if settype(relb)=='state':
        mask=masklev1(settype(relb),gr,re,mtype).data
    else:
        mask=masklev0(re,gr,mtype).data
    mask=mask.flatten()
    mask=np.delete(mask,omi)
    print(mask.shape)

    print('Loading %s...'%varn1)
    vn=load_data(md,fo,varn,mask,yr)
    return vn

def mp(md,varn,relb,fo=fo1,yr=yr1,dtype='mean'):
    # mask
    if settype(relb)=='state':
        mask=masklev1(settype(relb),gr,re,mtype).data
    else:
        mask=masklev0(re,gr,mtype).data
    mask=mask.flatten()
    mask=np.delete(mask,omi)
    print(mask.shape)

    print('Loading %s...'%varn)
    if dtype=='mean':
        vn=load_meandata(md,fo,varn,mask,yr)
    elif dtype=='hot':
        vn=load_hotdata(md,fo,varn,mask,yr)
    return vn

for md in lmd:
    mtype=settype(relb)
    retn=retname(relb)
    re=regionsets(relb)
    if mmm:
        md='mmm'
        vn1=[selreg(lmd[i],varn1,relb,fo=fo1,yr=yr1) for i in tqdm(range(len(lmd)))]
        vn2=[selreg(lmd[i],varn2,relb,fo=fo1,yr=yr1) for i in tqdm(range(len(lmd)))]
        vn3=[selreg(lmd[i],varn3,relb,fo=fo1,yr=yr1) for i in tqdm(range(len(lmd)))]
        vn1f=[selreg(lmd[i],varn1,relb,fo=fo2,yr=yr2) for i in tqdm(range(len(lmd)))]
        vn2f=[selreg(lmd[i],varn2,relb,fo=fo2,yr=yr2) for i in tqdm(range(len(lmd)))]
        vn3f=[selreg(lmd[i],varn3,relb,fo=fo2,yr=yr2) for i in tqdm(range(len(lmd)))]
    else:
        vn1=selreg(md,varn1,relb,fo=fo1,yr=yr1)
        vn2=selreg(md,varn2,relb,fo=fo1,yr=yr1)
        vn3=selreg(md,varn3,relb,fo=fo1,yr=yr1)
        vn1f=selreg(md,varn1,relb,fo=fo2,yr=yr2)
        vn2f=selreg(md,varn2,relb,fo=fo2,yr=yr2)
        vn3f=selreg(md,varn3,relb,fo=fo2,yr=yr2)

    psm1=mp(md,varn2,relb,fo=fo1,yr=yr1,dtype='hot')
    psm2=mp(md,varn2,relb,fo=fo2,yr=yr2,dtype='hot')
    plh1=mp(md,varn1,relb,fo=fo1,yr=yr1,dtype='hot')
    plh2=mp(md,varn1,relb,fo=fo2,yr=yr2,dtype='hot')
    msm1=mp(md,varn2,relb,fo=fo1,yr=yr1,dtype='mean')
    msm2=mp(md,varn2,relb,fo=fo2,yr=yr2,dtype='mean')
    mlh1=mp(md,varn1,relb,fo=fo1,yr=yr1,dtype='mean')
    mlh2=mp(md,varn1,relb,fo=fo2,yr=yr2,dtype='mean')

    ddir= '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,fo,md,varn,'regions')
    odir= '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (se,fo,md,varn,'regions')
    if not os.path.exists(ddir):
        os.makedirs(ddir)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('Plotting...')
    clim=set_clim(relb)
    dname1='%s/bc.%s_%s.%s.%s' % (ddir,varn,yr,'jja',relb)
    oname1='%s/bc.%s_%s.%s.%s' % (odir,varn,yr,'jja',relb)
    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    tname=r'%s %s' % (md.upper(),retn)
    ax.set_title(r'%s' % (tname))
    ax.set_xlabel('%s (%s)'%(varnlb(varn2),unitlb(varn2)))
    ax.set_ylabel('%s (%s)'%(varnlb(varn1),unitlb(varn1)))
    xc=[]
    bc=[]

    # select months
    mon=[6,7,8]
    imon=[5,6,7]
    if mmm:
        mvn1=[vn1[i].sel(time=vn1[i]['time.month']==mon) for i in tqdm(range(len(lmd)))]
        mvn2=[vn2[i].sel(time=vn2[i]['time.month']==mon) for i in tqdm(range(len(lmd)))]
        mvn3=[vn3[i].sel(time=vn3[i]['time.month']==mon) for i in tqdm(range(len(lmd)))]
        mvn1=xr.concat(mvn1,dim='time')
        mvn2=xr.concat(mvn2,dim='time')
        mvn3=xr.concat(mvn3,dim='time')
        mvn1f=[vn1f[i].sel(time=vn1f[i]['time.month']==mon) for i in tqdm(range(len(lmd)))]
        mvn2f=[vn2f[i].sel(time=vn2f[i]['time.month']==mon) for i in tqdm(range(len(lmd)))]
        mvn3f=[vn3f[i].sel(time=vn3f[i]['time.month']==mon) for i in tqdm(range(len(lmd)))]
        mvn1f=xr.concat(mvn1f,dim='time')
        mvn2f=xr.concat(mvn2f,dim='time')
        mvn3f=xr.concat(mvn3f,dim='time')
    else:
        mvn1=vn1.sel(time=np.in1d(vn1['time.month'],mon))
        mvn2=vn2.sel(time=np.in1d(vn2['time.month'],mon))
        mvn3=vn3.sel(time=np.in1d(vn3['time.month'],mon))
        mvn1f=vn1f.sel(time=np.in1d(vn1f['time.month'],mon))
        mvn2f=vn2f.sel(time=np.in1d(vn2f['time.month'],mon))
        mvn3f=vn3f.sel(time=np.in1d(vn3f['time.month'],mon))
    vn1=vn1.data.flatten()
    vn2=vn2.data.flatten()
    vn3=vn3.data.flatten()
    vn1f=vn1f.data.flatten()
    vn2f=vn2f.data.flatten()
    vn3f=vn3f.data.flatten()
    # bin
    nans=np.logical_or(np.isnan(vn1),np.isnan(vn2))
    vn1=vn1[~nans]
    vn2=vn2[~nans]
    vn3=vn3[~nans]
    nans=np.logical_or(np.isnan(vn1f),np.isnan(vn2f))
    vn1f=vn1f[~nans]
    vn2f=vn2f[~nans]
    vn3f=vn3f[~nans]
    # dg=np.digitize(vn2,bvn2)
    # bvn1=np.array([vn1[dg==i].mean() for i in range(1,len(bvn2)+1)])
    # svn1=np.array([vn1[dg==i].std() for i in range(1,len(bvn2)+1)])
    # bvn3=np.array([vn3[dg==i].mean() for i in range(1,len(bvn2)+1)])

    # # create kde
    # kde=gaussian_kde(np.vstack([vn2,vn1]))
    # # x1=np.linspace(np.min(vn1),np.max(vn1),51)
    # # x2=np.linspace(np.min(vn2),np.max(vn2),51)
    # # [x1,x2]=np.meshgrid(x1,x2,indexing='ij')
    # pdf=kde(np.vstack([vn2,vn1]))
    # # pdf=np.reshape(pdf,x1.shape)
    # print(np.max(pdf))

    # find best fit line
    f1,f2=bestfit(vn2,vn1)
    xc.append(f2['xc'])
    bc.append(f2['line'])
    f1f,f2f=bestfit(vn2f,vn1f)

    # ax.errorbar(bvn2,bvn1,yerr=svn1,fmt='.',color='k',alpha=0.2)
    clf=ax.scatter(vn2,vn1,s=0.5,c=vn3,vmin=vr[0],vmax=vr[1],cmap='BrBG',alpha=0.2)
    ax.plot(f2['line'][0],f2['line'][1],'-',color='tab:blue')
    ax.plot(f2f['line'][0],f2f['line'][1],'-',color='tab:orange')
    ax.plot(np.mean(msm1[imon]),np.mean(mlh1[imon]),'o',color='tab:blue')
    ax.plot(np.mean(psm1[imon]),np.mean(plh1[imon]),'+',color='tab:blue')
    ax.plot(np.mean(msm2[imon]),np.mean(mlh2[imon]),'o',color='tab:orange')
    ax.plot(np.mean(psm2[imon]),np.mean(plh2[imon]),'+',color='tab:orange')
    # clf=ax.contourf(x2,x1,pdf,np.logspace(-5,-3,101),extend='both',norm=mcolors.LogNorm())
    # ax.axvline(f2['xc'],linestyle=':',color='tab:purple')
    cb=fig.colorbar(clf,location='right')
    cb.set_label(label='%s (%s)'%(varnlb(varn3),unitlb(varn3)),size=12)
    cb.ax.tick_params(labelsize=12)
    fig.savefig('%s.png'%oname1, format='png', dpi=600)
    print(xc)
    pickle.dump([xc,bc],open('%s.pickle'%dname1,'wb'), protocol=5)

    if mmm:
        sys.exit()
