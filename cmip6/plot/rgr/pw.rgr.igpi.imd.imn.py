import os,sys
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from scipy.stats import gaussian_kde,shapiro,skew,kurtosis
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import statsmodels.api as sm
sys.path.append('/home/miyawaki/scripts/common')
from utils import monname,varnlb,unitlb
from mrsos import bestfit

se='sc'

fo='historical'
yr='1980-2000'

# fo='ssp370'
# yr='gwl2.0'

vn1='tas'
vn2='mrsos'
vn='%s+%s'%(vn1,vn2)

# (lat,lon) of interest
lat,lon=(70,205)

# identify igpi for select (lat,lon)
llat,llon=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lmilatlon.pickle','rb'))
dlat,dlon=(np.abs(llat-lat), np.abs(llon-lon))
ilat,ilon=(np.where(dlat==dlat.min()), np.where(dlon==dlon.min()))
igpi=np.intersect1d(ilat,ilon)[0]

# colors
lc=pl.cm.twilight(np.linspace(0,1,12))

def rgr(md):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s'%(se,fo,md)
    # raw percentile data
    pv1=xr.open_dataarray('%s/%s/p.%s_%s.%s.nc'%(idir,vn1,vn1,yr,se)).isel(gpi=igpi)
    pv2=xr.open_dataarray('%s/%s/pc.%s_%s.%s.nc'%(idir,vn2,vn2,yr,se)).isel(gpi=igpi)

    # load raw data
    v1=xr.open_dataarray('%s/%s/lm.%s_%s.%s.nc'%(idir,vn1,vn1,yr,se)).isel(gpi=igpi)
    v2=xr.open_dataarray('%s/%s/lm.%s_%s.%s.nc'%(idir,vn2,vn2,yr,se)).isel(gpi=igpi)

    # plot regression slope
    odir='/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,vn)
    if not os.path.exists(odir): os.makedirs(odir)

    pa=np.empty(12)
    a=np.empty(12)
    m=np.arange(1,13,1)
    for i,mn in enumerate(tqdm(m)):
        fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
        spv1,spv2=(pv1.sel(month=mn).data, pv2.sel(month=mn).data)
        spv2=1/2*(spv2[1:]+spv2[:-1])
        pfit=bestfit(spv2,spv1)
        [pX,pY]=pfit['line']
        sv1,sv2=(v1.sel(time=v1['time.month']==mn).data, v2.sel(time=v2['time.month']==mn).data)
        fit=bestfit(sv2,sv1)
        [X,Y]=fit['line']
        pres=sv1-np.interp(sv1,X,Y)
        res=sv1-fit['yp']
        if i==0:
            lres=res
        else:
            lres=np.concatenate((lres,res))
        px=np.linspace(spv2.min(),spv2.max(),101)
        x=np.linspace(sv2.min(),sv2.max(),101)
        ax.plot(sv2,sv1,'.',color=lc[i],markersize=2)
        ax.plot(X,Y,'-',color=lc[i])
        ax.plot(spv2,spv1,'+k',markersize=6)
        ax.plot(pX,pY,'-k')
        ax.set_title('%s'%monname(i))
        ax.set_xlabel(r'$%s$ (%s)'%(varnlb(vn2),unitlb(vn2)))
        ax.set_ylabel(r'$%s$ (%s)'%(varnlb(vn1),unitlb(vn1)))
        plt.savefig('%s/pw.rgr.%s.%g.%g.%g.png'%(odir,vn,lat,lon,mn),format='png',dpi=600)

        # qq plot of residuals
        fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
        sm.qqplot(pres.flatten(),ax=ax,line='s',markerfacecolor=lc[i],markeredgecolor=lc[i],markersize=2)
        psha=shapiro(pres.flatten())
        ax.text(0.05,0.85,'W=%.02f \n p=%.02f'%(psha.statistic,psha.pvalue),transform=ax.transAxes)
        plt.savefig('%s/pw.qq.res.binned.%s.%g.%g.%g.png'%(odir,vn,lat,lon,mn),format='png',dpi=600)

        # binned regression residual distribution
        k=gaussian_kde(pres.flatten())
        r=np.linspace(pres.min(),pres.max(),101)
        pdf=k(r)
        fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
        ax.plot(r,pdf,color=lc[i])
        psk,pku=(skew(pres.flatten()),kurtosis(pres.flatten()))
        ax.text(0.05,0.85,'skew$=%.02f$ \nkurtosis$=%.02f$'%(psk,pku),transform=ax.transAxes)
        ax.set_title('%s'%monname(i))
        ax.set_xlabel('Residual (K)')
        ax.set_ylabel(r'Probability density')
        plt.savefig('%s/pw.pdf.res.binned.%s.%g.%g.%g.png'%(odir,vn,lat,lon,mn),format='png',dpi=600)

        # qq plot of residuals
        fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
        sm.qqplot(res.flatten(),ax=ax,line='s',markerfacecolor=lc[i],markeredgecolor=lc[i],markersize=2)
        sha=shapiro(res.flatten())
        ax.text(0.05,0.85,'W=%.02f \n p=%.02f'%(sha.statistic,sha.pvalue),transform=ax.transAxes)
        plt.savefig('%s/pw.qq.res.%s.%g.%g.%g.png'%(odir,vn,lat,lon,mn),format='png',dpi=600)

        # all regression residual distribution
        k=gaussian_kde(res.flatten())
        r=np.linspace(res.min(),res.max(),101)
        pdf=k(r)
        fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
        ax.plot(r,pdf,color=lc[i])
        sk,ku=(skew(res.flatten()),kurtosis(res.flatten()))
        ax.text(0.05,0.85,'skew$=%.02f$\nkurtosis$=%.02f$'%(sk,ku),transform=ax.transAxes)
        ax.set_title('%s'%monname(i))
        ax.set_xlabel('Residual (K)')
        ax.set_ylabel(r'Probability density')
        plt.savefig('%s/pw.pdf.res.%s.%g.%g.%g.png'%(odir,vn,lat,lon,mn),format='png',dpi=600)

rgr('CESM2')
