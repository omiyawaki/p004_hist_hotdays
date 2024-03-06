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
from etregimes import bestfit

# relb='fourcorners'
# re=['Utah','Colorado','Arizona','New Mexico']

relb='ic'

pct=np.linspace(1,99,101)
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

lmd=mods(fo1)
mmm=True
fo=fo2
yr=yr2

def set_clim(relb):
    clim={  
            'ic':   [0,15],
            'fc':   [0,8],
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

def load_data(md,fo,varn,mask):
    idir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,varn)
    ds=xr.open_dataset('%s/lm.%s_%s.%s.nc' % (idir,varn,yr,se))
    vn=ds[varn]
    if varn=='pr': vn=86400*vn # convert pr to mm/d
    time=ds['time']
    # delete data outside masked region
    vn=np.delete(vn.data,np.nonzero(np.isnan(mask)),axis=1)
    vn=xr.DataArray(vn,coords={'time':time,'gpi':range(vn.shape[1])},dims=('time','gpi'))
    return vn

def selreg(md,varn,relb,fo=fo1):
    # mask
    mask=masklev0(re,gr,mtype).data
    mask=mask.flatten()
    mask=np.delete(mask,omi)
    print(mask.shape)

    print('Loading %s...'%varn1)
    vn=load_data(md,fo,varn,mask)
    return vn

for md in lmd:
    mtype=settype(relb)
    retn=retname(relb)
    re=regionsets(relb)
    if mmm:
        md='mmm'
        vn1=[selreg(lmd[i],varn1,relb,fo=fo) for i in tqdm(range(len(lmd)))]
        vn2=[selreg(lmd[i],varn2,relb,fo=fo) for i in tqdm(range(len(lmd)))]
        vn3=[selreg(lmd[i],varn3,relb,fo=fo) for i in tqdm(range(len(lmd)))]
    else:
        vn1=selreg(md,varn1,relb,fo=fo)
        vn2=selreg(md,varn2,relb,fo=fo)
        vn3=selreg(md,varn3,relb,fo=fo)

    ddir= '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,fo,md,varn,'regions')
    odir= '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (se,fo,md,varn,'regions')
    if not os.path.exists(ddir):
        os.makedirs(ddir)
    if not os.path.exists(odir):
        os.makedirs(odir)

    print('Plotting...')
    clim=set_clim(relb)
    dname1='%s/bc.%s_%s.%s.%s' % (ddir,varn,yr,se,relb)
    oname1='%s/bc.%s_%s.%s.%s' % (odir,varn,yr,se,relb)
    fig,ax=plt.subplots(nrows=3,ncols=4,figsize=(9,7),constrained_layout=True)
    tname=r'%s %s' % (md.upper(),retn)
    fig.suptitle(r'%s' % (tname))
    fig.supxlabel('%s (%s)'%(varnlb(varn2),unitlb(varn2)))
    fig.supylabel('%s (%s)'%(varnlb(varn1),unitlb(varn1)))
    ax=ax.flatten()
    xc=[]
    bc=[]
    for i,mon in enumerate(tqdm(np.arange(1,13,1))):
        ax[i].set_title(monname(i))
        # select month
        if mmm:
            mvn1=[vn1[i].sel(time=vn1[i]['time.month']==mon) for i in tqdm(range(len(lmd)))]
            mvn2=[vn2[i].sel(time=vn2[i]['time.month']==mon) for i in tqdm(range(len(lmd)))]
            mvn3=[vn3[i].sel(time=vn3[i]['time.month']==mon) for i in tqdm(range(len(lmd)))]
            mvn1=xr.concat(mvn1,dim='time')
            mvn2=xr.concat(mvn2,dim='time')
            mvn3=xr.concat(mvn3,dim='time')
        else:
            mvn1=vn1.sel(time=vn1['time.month']==mon)
            mvn2=vn2.sel(time=vn2['time.month']==mon)
            mvn3=vn3.sel(time=vn3['time.month']==mon)
        mvn1=mvn1.data.flatten()
        mvn2=mvn2.data.flatten()
        mvn3=mvn3.data.flatten()
        # bin
        nans=np.logical_or(np.isnan(mvn1),np.isnan(mvn2))
        mvn1=mvn1[~nans]
        mvn2=mvn2[~nans]
        mvn3=mvn3[~nans]
        dg=np.digitize(mvn2,bvn2)
        bvn1=np.array([mvn1[dg==i].mean() for i in range(1,len(bvn2)+1)])
        svn1=np.array([mvn1[dg==i].std() for i in range(1,len(bvn2)+1)])
        bvn3=np.array([mvn3[dg==i].mean() for i in range(1,len(bvn2)+1)])
        # find best fit line
        f1,f2=bestfit(mvn2,mvn1)
        xc.append(f2['xc'])
        bc.append(f2['line'])

        ax[i].errorbar(bvn2,bvn1,yerr=svn1,fmt='.',color='k',alpha=0.2)
        clf=ax[i].scatter(bvn2,bvn1,s=3,c=bvn3,vmin=vr[0],vmax=vr[1],cmap='BrBG')
        ax[i].axvline(f2['xc'],linestyle=':',color='tab:purple')
        ax[i].plot(f2['line'][0],f2['line'][1],'-k')
        if i==0:
            cb=fig.colorbar(clf,ax=ax.ravel().tolist(),location='right')
            cb.set_label(label='%s (%s)'%(varnlb(varn3),unitlb(varn3)),size=16)
            cb.ax.tick_params(labelsize=16)
        fig.savefig('%s.png'%oname1, format='png', dpi=600)
    print(xc)
    pickle.dump([xc,bc],open('%s.pickle'%dname1,'wb'), protocol=5)

    if mmm:
        sys.exit()
