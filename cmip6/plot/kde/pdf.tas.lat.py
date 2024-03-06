import os,sys
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from KDEpy import FFTKDE
from matplotlib import pyplot as plt
sys.path.append('/home/miyawaki/scripts/p004/scripts/cmip6/data')
from util import mods

lvn=['tas']
lre=['hl','ml','tr']
mn=7 # set to 0 for all months

fo1='historical'
by1='1980-2000'

fo2='ssp370'
by2='gwl2.0'

fo='%s+%s'%(fo1,fo2)

tlat=30
plat=50
npdf=101 # number of points to use for pdf
dpi=600
se='sc'

md='mmm'
# md='CESM2'
lmd=mods(fo1)

# load land indices
lmi,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomi.pickle','rb'))

# load land lat lon
llat,llon=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lmilatlon.pickle','rb'))

# greenland and antarctica mask
aagl=pickle.load(open('/project/amp/miyawaki/data/share/aa_gl/cesm2/aa_gl.pickle','rb'))

def tstr(re,mn):
    if re=='tr': tstr='Tropics\n %s'%mstr(mn)
    elif re=='ml': tstr='Midlatitudes\n %s'%mstr(mn)
    elif re=='hl': tstr='High Latitudes\n %s'%mstr(mn)
    return tstr

def mstr(mn):
    if mn==1: mstr='month'
    else:     mstr='months'
    mstr='%g %s after winter solstice'%(mn,mstr)
    return mstr

def sellat(xvn,reg):
    if reg=='tr':
        idx=np.abs(llat)<=tlat
    elif reg=='ml':
        idx=np.logical_and(np.abs(llat)>tlat,np.abs(llat)<=plat)
    elif reg=='hl':
        idx=np.abs(llat)>plat
    return xvn[:,idx],llat[idx]

def mn_sel_shift(xvn,re):
    xvn=xvn*aagl.flatten()[lmi]
    xvn,slat=sellat(xvn,re)
    if not mn==0:
        nxvn=xvn.copy()
        sxvn=xvn.copy()
        nxvn=nxvn[:,slat>0].sel(time=nxvn['time.month']==mn)
        sxvn=sxvn[:,slat<0].sel(time=sxvn['time.month']==(mn+6)%12)
        xvn=np.concatenate([nxvn.data.flatten(),sxvn.data.flatten()])
    else:
        xvn=xvn.data.flatten()
    return xvn

def load_lm(varn,byr,se,fo,md,re):
    if md=='mmm':
        for i,md in enumerate(tqdm(lmd)):
            idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,varn)
            if i==0:
                xvn=xr.open_dataarray('%s/lm.%s_%s.%s.nc'%(idir,varn,byr,se))
                xvn=mn_sel_shift(xvn,re)
            else:
                xvn0=xr.open_dataarray('%s/lm.%s_%s.%s.nc'%(idir,varn,byr,se))
                xvn0=mn_sel_shift(xvn0,re)
                xvn=np.concatenate([xvn,xvn0])
    else:
        idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,varn)
        xvn=xr.open_dataarray('%s/lm.%s_%s.%s.nc'%(idir,varn,byr,se))
        xvn=mn_sel_shift(xvn,re)
    return xvn

def make_kde(xvn):
    xvn=xvn.flatten()
    xvn=xvn[~np.isnan(xvn)]
    avg=np.mean(xvn)
    p05=np.percentile(xvn,5)
    p95=np.percentile(xvn,95)
    x,y=FFTKDE(kernel='gaussian',bw=1.0).fit(xvn).evaluate()
    return x,y,avg,p05,p95

def make_pdf(md,vn,fo,by,re):
    xvn=load_lm(vn,by,se,fo,md,re)
    x,pdf,avg,p05,p95=make_kde(xvn)
    return x,pdf,avg,p05,p95

def plot_pdf(md,vn,re):
    odir='/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,vn)
    if not os.path.exists(odir): os.makedirs(odir)

    x1,pdf1,avg1,p051,p951=make_pdf(md,vn,fo1,by1,re)
    x2,pdf2,avg2,p052,p952=make_pdf(md,vn,fo2,by2,re)

    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    if re=='hl':
        ax.axvline(273.15,color='k',linestyle='--',linewidth=0.5)
    ax.annotate('%.01f K'%(avg2-avg1),xy=(avg2,0),xytext=(0,3),ha='right',textcoords='offset points',color='gray',fontsize=6)
    ax.annotate('%.01f K'%(p052-p051),xy=(p052,0),xytext=(0,3),ha='right',textcoords='offset points',color='navy',fontsize=6)
    ax.annotate('%.01f K'%(p952-p951),xy=(p952,0),xytext=(0,3),ha='right',textcoords='offset points',color='maroon',fontsize=6)
    ax.plot([avg1,avg2],[0,0],color='gray')
    ax.plot([p051,p052],[0,0],color='navy')
    ax.plot([p951,p952],[0,0],color='maroon')
    ax.plot(x1,pdf1,color='lightgray')
    ax.plot(x2,pdf2,color='k')
    ax.set_xlabel('T (K)')
    ax.set_ylabel('Density (unitless)')
    ax.set_title(tstr(re,mn))
    fig.savefig('%s/pdf.%s.%s.%s.%02d.pdf'%(odir,vn,fo,re,mn),format='pdf',dpi=dpi)
    fig.savefig('%s/pdf.%s.%s.%s.%02d.png'%(odir,vn,fo,re,mn),format='png',dpi=dpi)

    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    cdf1=np.cumsum(pdf1)
    cdf2=np.cumsum(pdf2)
    ax.plot(x1,cdf1/cdf1[-1])
    ax.plot(x2,cdf2/cdf2[-1])
    ax.set_xlabel('T (K)')
    ax.set_ylabel('Cumulative Density (unitless)')
    ax.set_title(tstr(re,mn))
    fig.savefig('%s/cdf.%s.%s.%s.%02d.pdf'%(odir,vn,fo,re,mn),format='pdf',dpi=dpi)
    fig.savefig('%s/cdf.%s.%s.%s.%02d.png'%(odir,vn,fo,re,mn),format='png',dpi=dpi)

def loop_vn(md,re):
    [plot_pdf(md,vn,re) for vn in lvn]

def loop_re(md):
    [loop_vn(md,re) for re in tqdm(lre)]

loop_re(md)
