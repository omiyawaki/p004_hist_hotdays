import os,sys
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from matplotlib import pyplot as plt
sys.path.append('/home/miyawaki/scripts/p004/scripts/cmip6/data')
from util import mods

lvn=['tas']
lre=['hl','ml','tr']
mn=10 # set to 0 for all months

fo1='historical'
by1='1980-2000'

fo2='ssp370'
by2='gwl2.0'

fo='%s+%s'%(fo1,fo2)

tlat=30
plat=50
npct=101 # number of points to use for pct
mu=0.5 # sigmoid parameter
nu=2 # sigmoid parameter
x=np.linspace(0,1,npct)
x=1e2*(1+((x*(1-mu))/(mu*(1-x)))**-nu)**-1
x=x[1::2]
dpi=600
se='sc'

# md='mmm'
md='CESM2'
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

def calc_pct(xvn):
    xvn=xvn.flatten()
    xvn=xvn[~np.isnan(xvn)]
    pct=np.percentile(xvn,x)
    return pct

def make_pct(md,vn,fo,by,re):
    xvn=load_lm(vn,by,se,fo,md,re)
    pct=calc_pct(xvn)
    return pct

def plot_pct(md,vn,re):
    odir='/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,vn)
    if not os.path.exists(odir): os.makedirs(odir)

    pct1=make_pct(md,vn,fo1,by1,re)
    pct2=make_pct(md,vn,fo2,by2,re)

    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    ax.plot(x,pct1,color='lightgray')
    ax.plot(x,pct2,color='k')
    ax.set_xlabel('Percentile (unitless)')
    ax.set_ylabel('$T$ (K)')
    ax.set_title(tstr(re,mn))
    fig.savefig('%s/pct.%s.%s.%s.%02d.pdf'%(odir,vn,fo,re,mn),format='pdf',dpi=dpi)
    fig.savefig('%s/pct.%s.%s.%s.%02d.png'%(odir,vn,fo,re,mn),format='png',dpi=dpi)

    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    ax.axvline(np.interp(273.15,pct1,x),color='k',linestyle='--',linewidth=0.5)
    ax.plot(x,pct2-pct1,color='k')
    ax.set_xlabel('Percentile (unitless)')
    ax.set_ylabel('$\Delta T$ (K)')
    ax.set_title(tstr(re,mn))
    ax2=ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels(['%0.f'%np.interp(i,x,pct1) for i in ax.get_xticks()])
    ax2.set_xlabel('T$_{hist}$ (K)')
    fig.savefig('%s/d.pct.%s.%s.%s.%02d.pdf'%(odir,vn,fo,re,mn),format='pdf',dpi=dpi)
    fig.savefig('%s/d.pct.%s.%s.%s.%02d.png'%(odir,vn,fo,re,mn),format='png',dpi=dpi)

def loop_vn(md,re):
    [plot_pct(md,vn,re) for vn in lvn]

def loop_re(md):
    [loop_vn(md,re) for re in tqdm(lre)]

loop_re(md)
