import os,sys
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from matplotlib import pyplot as plt
sys.path.append('/home/miyawaki/scripts/p004/scripts/cmip6/data')
from util import mods

lvn=['tas']
lre=['hl']
mn=9 # set to 0 for all months

fo1='historical'
by1='1980-2000'

fo2='ssp370'
by2='gwl2.0'

fo='%s+%s'%(fo1,fo2)

tlat=30
plat=50
dpi=600
se='sc'

md='mmm'
# md='CESM2'

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
    return xvn[:,:,idx],llat[idx]

def mn_sel_shift(xvn,re):
    xvn=xvn*aagl.flatten()[lmi]
    xvn,slat=sellat(xvn,re)
    nxvn=xvn.copy()
    sxvn=xvn.copy()
    nxvn=nxvn.sel(month=mn)[:,slat>0]
    sxvn=sxvn.sel(month=(mn+5)%12+1)[:,slat<0]
    nlat=np.cos(np.deg2rad(slat[slat>0]))
    slat=np.cos(np.deg2rad(slat[slat<0]))
    nlat[np.isnan(nxvn[0,:].data)]=np.nan
    slat[np.isnan(sxvn[0,:].data)]=np.nan
    xvn=1/2*(np.nansum(nlat*nxvn,axis=1)/np.nansum(nlat)+np.nansum(slat*sxvn,axis=1)/np.nansum(slat))
    return xvn

def load_dp(varn,byr1,byr2,se,fo1,fo2,md,re):
    idir1='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s'%(se,fo1,md,varn)
    pvn1=xr.open_dataarray('%s/pc.%s_%s.%s.nc'%(idir1,varn,byr1,se))
    x=pvn1['percentile']
    idir2='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s'%(se,fo2,md,varn)
    pvn2=xr.open_dataarray('%s/pc.%s_%s.%s.nc'%(idir2,varn,byr2,se))
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s-%s/%s/%s'%(se,fo2,fo1,md,varn)
    dpvn=xr.open_dataarray('%s/dpc.%s_%s_%s.%s.nc'%(idir,varn,byr1,byr2,se))
    pvn1=mn_sel_shift(pvn1,re)
    pvn2=mn_sel_shift(pvn2,re)
    dpvn=mn_sel_shift(dpvn,re)
    return dpvn,pvn1,pvn2,x

def plot_pct(md,vn,re):
    odir='/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,vn)
    if not os.path.exists(odir): os.makedirs(odir)

    dpvn,pct1,pct2,x=load_dp(vn,by1,by2,se,fo1,fo2,md,re)

    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    ax.plot(x,pct1,color='lightgray')
    ax.plot(x,pct2,color='k')
    ax.set_xlabel('Percentile (unitless)')
    ax.set_ylabel('$T$ (K)')
    ax.set_title(tstr(re,mn))
    fig.savefig('%s/pct.%s.%s.%s.%02d.alt.pdf'%(odir,vn,fo,re,mn),format='pdf',dpi=dpi)
    fig.savefig('%s/pct.%s.%s.%s.%02d.alt.png'%(odir,vn,fo,re,mn),format='png',dpi=dpi)

    fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
    # ax.axvline(np.interp(273.15,pct1,x),color='k',linestyle='--',linewidth=0.5)
    ax.plot(x,dpvn,color='k')
    ax.set_xlabel('Percentile (unitless)')
    ax.set_ylabel('$\Delta T$ (K)')
    ax.set_title(tstr(re,mn))
    ax2=ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels(['%0.f'%np.interp(i,x,pct1) for i in ax.get_xticks()])
    ax2.set_xlabel('T$_{hist}$ (K)')
    fig.savefig('%s/d.pct.%s.%s.%s.%02d.alt.pdf'%(odir,vn,fo,re,mn),format='pdf',dpi=dpi)
    fig.savefig('%s/d.pct.%s.%s.%s.%02d.alt.png'%(odir,vn,fo,re,mn),format='png',dpi=dpi)

def loop_vn(md,re):
    [plot_pct(md,vn,re) for vn in lvn]

def loop_re(md):
    [loop_vn(md,re) for re in tqdm(lre)]

loop_re(md)
