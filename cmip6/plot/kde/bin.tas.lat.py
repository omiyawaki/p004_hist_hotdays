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
mn=7 # set to 0 for all months

fo1='historical'
by1='1980-2000'

fo2='ssp370'
by2='gwl2.0'

fo='%s+%s'%(fo1,fo2)

tlat=30
plat=50
x=np.arange(180,315+2.5,2.5) # Ts bins
xe=x[::2] # edges
x=x[1::2] # midpoints
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

def calc_tbins(xvn,bins):
    xvn=xvn.flatten()
    if bins is None: bins=np.digitize(xvn,xe)
    tbins=np.array([np.mean(xvn[bins==b]) for b in 1+np.arange(len(x))])
    return tbins,bins

def make_tbins(md,vn,fo,by,re,bins=None):
    xvn=load_lm(vn,by,se,fo,md,re)
    tbins,ibins=calc_tbins(xvn,bins)
    return tbins,ibins

def plot_tbins(md,vn,re):
    odir='/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,vn)
    if not os.path.exists(odir): os.makedirs(odir)

    tbins1,ibins=make_tbins(md,vn,fo1,by1,re)
    tbins2,_=make_tbins(md,vn,fo2,by2,re,bins=ibins)

    # fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    # ax.plot(x,tbins1,color='lightgray')
    # ax.plot(x,tbins2,color='k')
    # ax.set_xlabel('Percentile (unitless)')
    # ax.set_ylabel('$T$ (K)')
    # ax.set_title(tstr(re,mn))
    # fig.savefig('%s/tbins.%s.%s.%s.%02d.pdf'%(odir,vn,fo,re,mn),format='pdf',dpi=dpi)
    # fig.savefig('%s/tbins.%s.%s.%s.%02d.png'%(odir,vn,fo,re,mn),format='png',dpi=dpi)

    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    ax.axvline(np.interp(273.15,tbins1,x),color='k',linestyle='--',linewidth=0.5)
    ax.plot(x,tbins2-tbins1,color='k')
    ax.set_xlabel('Historical T (K)')
    ax.set_ylabel('$\Delta T$ (K)')
    ax.set_title(tstr(re,mn))
    # ax2=ax.twiny()
    # ax2.set_xticks(ax.get_xticks())
    # ax2.set_xbound(ax.get_xbound())
    # ax2.set_xticklabels(['%0.f'%np.interp(i,x,tbins1) for i in ax.get_xticks()])
    # ax2.set_xlabel('T$_{hist}$ (K)')
    fig.savefig('%s/d.tbins.%s.%s.%s.%02d.pdf'%(odir,vn,fo,re,mn),format='pdf',dpi=dpi)
    fig.savefig('%s/d.tbins.%s.%s.%s.%02d.png'%(odir,vn,fo,re,mn),format='png',dpi=dpi)

def loop_vn(md,re):
    [plot_tbins(md,vn,re) for vn in lvn]

def loop_re(md):
    [loop_vn(md,re) for re in tqdm(lre)]

loop_re(md)
